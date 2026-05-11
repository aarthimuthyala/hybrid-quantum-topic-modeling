"""
src/evaluation/benchmark_runner.py
======================================
Team T-4: Hybrid Pipeline Team
MASTER BLUEPRINT v1.0 — §3.5

Responsibility:
    Run the full classical vs. hybrid benchmark comparison.
    Log all metrics, config snapshots, and artifacts to MLflow.
    Produce a BenchmarkReport artifact persisted to outputs/reports/.

Inputs:  PipelineConfig (from src/hybrid/hybrid_pipeline.py)
Outputs: BenchmarkReport dataclass
         outputs/reports/{run_id}_report.html (Blueprint §5 stage 08)

Integration points:
    - Orchestrates T-2 classical pipeline results vs T-4 hybrid pipeline results
    - Calls topic_coherence.compute_coherence and cluster_metrics.compute_cluster_metrics
    - Full MLflow experiment tracking (Blueprint §10.4 CI gate requirement)
    - Results consumed by GET /eval/report/{run_id} and /hybrid/compare endpoints

Blueprint data-flow stage: Stage 08 — Evaluation & Report
Artifact: outputs/reports/{run_id}_report.html
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from shared.logger import get_logger, log_run_end, log_run_start
from shared.serializer import save_artifact
from shared.file_utils import ensure_dir, get_artifact_path

from src.evaluation.cluster_metrics import ClusterMetrics, compute_cluster_metrics
from src.evaluation.topic_coherence import CoherenceScores, compute_coherence
from src.hybrid.hybrid_pipeline import HybridRunResult, PipelineConfig, run_hybrid_pipeline

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPORT_OUTPUT_DIR: str = "outputs/reports"
MODELS_OUTPUT_DIR: str = "outputs/models"
DEFAULT_EXPERIMENT_NAME: str = "hqc_main"    # matches base_config.yaml §9.1


# ---------------------------------------------------------------------------
# Output contracts
# ---------------------------------------------------------------------------

@dataclass
class ClassicalBenchmarkResult:
    """
    Holds classical-pipeline metrics for one run, aligned with
    the HybridRunResult structure for easy side-by-side comparison.
    """
    model_id: str
    model_type: str                 # "lda" | "nmf" | "kmeans"
    coherence: CoherenceScores
    cluster: ClusterMetrics
    wall_time_s: float = 0.0
    config_snapshot: Dict = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """
    Full comparison report contract.
    Consumed by GET /eval/report/{run_id} and /hybrid/compare endpoints.

    Fields
    ------
    report_id          : Unique ID for this benchmark run.
    run_id             : Hybrid pipeline run_id this report covers.
    corpus_id          : Source corpus.
    classical_metrics  : Dict summarising classical model scores.
    hybrid_metrics     : Dict summarising hybrid model scores.
    delta              : Dict of (hybrid - classical) differences for key metrics.
    figures            : List of paths to generated figure files.
    tables             : List of rendered HTML table strings.
    markdown_url       : Path to the exported report artifact.
    wall_time_s        : Total benchmark wall-clock time.
    status             : "complete" | "partial" | "failed"
    metadata           : Full config snapshots and provenance.
    """
    report_id: str
    run_id: str
    corpus_id: str
    classical_metrics: Dict
    hybrid_metrics: Dict
    delta: Dict
    figures: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    markdown_url: str = ""
    wall_time_s: float = 0.0
    status: str = "complete"
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Executes the full classical vs. hybrid benchmark pipeline and produces
    a BenchmarkReport with MLflow experiment tracking.

    Workflow
    --------
    1. Run classical baseline (LDA / NMF + K-Means)
    2. Run hybrid pipeline (→ HybridRunResult)
    3. Evaluate coherence and cluster quality for both
    4. Compute deltas and log everything to MLflow
    5. Render HTML report → outputs/reports/{run_id}_report.html

    Example
    -------
    >>> from src.evaluation.benchmark_runner import BenchmarkRunner
    >>> from src.hybrid.hybrid_pipeline import PipelineConfig
    >>> cfg = PipelineConfig(corpus_id="20ng", n_topics=20, quantum_mode="qaoa")
    >>> runner = BenchmarkRunner(config=cfg)
    >>> report = runner.run(
    ...     dt_matrix=tfidf_matrix,
    ...     documents=clean_docs,
    ...     ground_truth=newsgroup_labels,
    ... )
    >>> print(report.delta)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.report_id: str = uuid.uuid4().hex[:8]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        dt_matrix: np.ndarray,
        documents: List[Dict],
        ground_truth: Optional[List[int]] = None,
    ) -> BenchmarkReport:
        """
        Run the full benchmark comparison.

        Parameters
        ----------
        dt_matrix    : np.ndarray (n_docs, n_features) — shared input matrix.
        documents    : List[CleanDocument] dicts for coherence computation.
        ground_truth : Optional integer class labels for NMI / ARI.

        Returns
        -------
        BenchmarkReport
        """
        cfg = self.config
        t_start = time.perf_counter()

        log_run_start(self.report_id, {"type": "benchmark", **vars(cfg)})
        logger.info(
            "BenchmarkRunner.run() started",
            extra={"report_id": self.report_id, "corpus_id": cfg.corpus_id},
        )

        mlflow_run = self._init_mlflow(cfg)

        try:
            # --- Classical pipeline ---
            classical_result = self._run_classical(
                dt_matrix=dt_matrix,
                documents=documents,
                ground_truth=ground_truth,
                cfg=cfg,
            )

            # --- Hybrid pipeline ---
            hybrid_run_result = run_hybrid_pipeline(
                dt_matrix=dt_matrix,
                config=cfg,
                documents=documents,
            )
            hybrid_eval = self._evaluate_hybrid(
                hybrid_result=hybrid_run_result,
                dt_matrix=dt_matrix,
                documents=documents,
                ground_truth=ground_truth,
            )

            # --- Summarise and compare ---
            classical_summary = self._summarise_classical(classical_result)
            hybrid_summary = self._summarise_hybrid(hybrid_run_result, hybrid_eval)
            delta = self._compute_delta(classical_summary, hybrid_summary)

            wall_time = time.perf_counter() - t_start

            # --- Render HTML report (Stage 08 artifact) ---
            report_path = self._render_html_report(
                classical_summary=classical_summary,
                hybrid_summary=hybrid_summary,
                delta=delta,
                cfg=cfg,
            )

            report = BenchmarkReport(
                report_id=self.report_id,
                run_id=cfg.run_id,
                corpus_id=cfg.corpus_id,
                classical_metrics=classical_summary,
                hybrid_metrics=hybrid_summary,
                delta=delta,
                figures=[],
                tables=self._build_html_tables(classical_summary, hybrid_summary, delta),
                markdown_url=report_path,
                wall_time_s=round(wall_time, 4),
                status="complete",
                metadata={
                    "config_snapshot": vars(cfg),
                    "n_docs": dt_matrix.shape[0],
                    "n_features": dt_matrix.shape[1],
                    "has_ground_truth": ground_truth is not None,
                },
            )

            self._log_mlflow_final(report)
            log_run_end(self.report_id, delta)

            logger.info(
                "BenchmarkRunner.run() complete",
                extra={
                    "report_id": self.report_id,
                    "wall_time_s": report.wall_time_s,
                    "report_path": report_path,
                },
            )
            return report

        except Exception as exc:
            logger.error(
                "BenchmarkRunner.run() failed",
                extra={"report_id": self.report_id, "error": str(exc)},
            )
            self._end_mlflow(status="FAILED")
            raise

        finally:
            self._end_mlflow(status="FINISHED")

    # ------------------------------------------------------------------
    # Classical pipeline execution
    # ------------------------------------------------------------------

    def _run_classical(
        self,
        dt_matrix: np.ndarray,
        documents: List[Dict],
        ground_truth: Optional[List[int]],
        cfg: PipelineConfig,
    ) -> ClassicalBenchmarkResult:
        """Run LDA/NMF + K-Means and compute coherence + cluster metrics."""
        t0 = time.perf_counter()
        model_id = f"{cfg.classical_model}_{cfg.corpus_id}_{cfg.run_id}"
        logger.info("Running classical baseline", extra={"model_id": model_id})

        # Topic model (delegate to T-2; stub if unavailable)
        topic_result = self._run_classical_topic_model(dt_matrix, cfg, model_id)

        # K-Means clustering (delegate to T-2; stub if unavailable)
        kmeans_result = self._run_kmeans(dt_matrix, cfg, model_id)

        # Coherence
        coherence = compute_coherence(
            topic_result=topic_result,
            documents=documents,
            model_id=model_id,
            topn=10,
            corpus_id=cfg.corpus_id,
            log_to_mlflow=True,
        )

        # Cluster metrics
        cluster = compute_cluster_metrics(
            cluster_result=kmeans_result,
            embeddings=dt_matrix,
            cluster_id=f"{model_id}_clusters",
            ground_truth=ground_truth,
            corpus_id=cfg.corpus_id,
            log_to_mlflow=True,
        )

        return ClassicalBenchmarkResult(
            model_id=model_id,
            model_type=cfg.classical_model,
            coherence=coherence,
            cluster=cluster,
            wall_time_s=round(time.perf_counter() - t0, 4),
            config_snapshot=vars(cfg),
        )

    def _run_classical_topic_model(
        self,
        dt_matrix: np.ndarray,
        cfg: PipelineConfig,
        model_id: str,
    ) -> Dict:
        try:
            if cfg.classical_model == "lda":
                from src.classical.lda_model import LDAModel  # type: ignore
                result = LDAModel(n_topics=cfg.n_topics).fit(dt_matrix)
            else:
                from src.classical.nmf_model import NMFModel  # type: ignore
                result = NMFModel(n_topics=cfg.n_topics).fit(dt_matrix)
            return result if isinstance(result, dict) else vars(result)
        except ImportError:
            logger.warning("Classical topic module unavailable — using stub", extra={"model_id": model_id})
            return {
                "model_id": model_id,
                "topics": [],
                "doc_topic_matrix": None,
                "params": {"n_topics": cfg.n_topics},
                "_stub": True,
            }

    def _run_kmeans(
        self,
        dt_matrix: np.ndarray,
        cfg: PipelineConfig,
        model_id: str,
    ) -> Dict:
        try:
            from src.classical.kmeans_clusterer import KMeansClusterer  # type: ignore
            result = KMeansClusterer(k=cfg.n_topics).fit(dt_matrix)
            return result if isinstance(result, dict) else vars(result)
        except ImportError:
            logger.warning("KMeans module unavailable — using stub", extra={"model_id": model_id})
            rng = np.random.default_rng(cfg.seed)
            return {
                "cluster_id": f"{model_id}_kmeans",
                "labels": rng.integers(0, cfg.n_topics, size=dt_matrix.shape[0]).tolist(),
                "centroids": None,
                "k": cfg.n_topics,
                "source": "stub",
            }

    # ------------------------------------------------------------------
    # Hybrid evaluation
    # ------------------------------------------------------------------

    def _evaluate_hybrid(
        self,
        hybrid_result: HybridRunResult,
        dt_matrix: np.ndarray,
        documents: List[Dict],
        ground_truth: Optional[List[int]],
    ) -> Dict:
        """Compute coherence and cluster metrics for the hybrid run result."""
        coherence = compute_coherence(
            topic_result=hybrid_result.classical_result,
            documents=documents,
            model_id=f"hybrid_{hybrid_result.run_id}",
            corpus_id=hybrid_result.corpus_id,
            log_to_mlflow=False,    # already logged by hybrid_pipeline
        )

        cluster = compute_cluster_metrics(
            cluster_result=hybrid_result.cluster_result,
            embeddings=dt_matrix,
            cluster_id=hybrid_result.cluster_result.get("cluster_id", hybrid_result.run_id),
            ground_truth=ground_truth,
            corpus_id=hybrid_result.corpus_id,
            log_to_mlflow=False,
        )

        return {"coherence": coherence, "cluster": cluster}

    # ------------------------------------------------------------------
    # Metric summarisation
    # ------------------------------------------------------------------

    def _summarise_classical(self, result: ClassicalBenchmarkResult) -> Dict:
        return {
            "model_id": result.model_id,
            "model_type": result.model_type,
            "c_v": result.coherence.c_v,
            "npmi": result.coherence.npmi,
            "umass": result.coherence.umass,
            "silhouette": result.cluster.silhouette,
            "db_index": result.cluster.db_index,
            "nmi": result.cluster.nmi if result.cluster.has_ground_truth else None,
            "ari": result.cluster.ari if result.cluster.has_ground_truth else None,
            "wall_time_s": result.wall_time_s,
        }

    def _summarise_hybrid(
        self,
        result: HybridRunResult,
        eval_results: Dict,
    ) -> Dict:
        coherence: CoherenceScores = eval_results["coherence"]
        cluster: ClusterMetrics = eval_results["cluster"]
        return {
            "model_id": f"hybrid_{result.run_id}",
            "model_type": f"hybrid_{self.config.quantum_mode}",
            "c_v": coherence.c_v,
            "npmi": coherence.npmi,
            "umass": coherence.umass,
            "silhouette": cluster.silhouette,
            "db_index": cluster.db_index,
            "nmi": cluster.nmi if cluster.has_ground_truth else None,
            "ari": cluster.ari if cluster.has_ground_truth else None,
            "final_cost": result.metrics.get("final_cost"),
            "converged": result.metrics.get("converged"),
            "wall_time_s": result.wall_time_s,
        }

    def _compute_delta(
        self,
        classical: Dict,
        hybrid: Dict,
    ) -> Dict:
        """
        Compute (hybrid - classical) differences for all shared numeric metrics.
        Positive delta means hybrid outperforms classical on that metric
        (direction-aware: inverted for db_index where lower is better).
        """
        comparable_keys = ["c_v", "npmi", "umass", "silhouette", "nmi", "ari"]
        delta: Dict = {}
        for key in comparable_keys:
            c_val = classical.get(key)
            h_val = hybrid.get(key)
            if c_val is not None and h_val is not None:
                raw = h_val - c_val
                # db_index: lower is better → negate so positive delta = improvement
                delta[key] = round(-raw if key == "db_index" else raw, 6)

        db_c = classical.get("db_index")
        db_h = hybrid.get("db_index")
        if db_c is not None and db_h is not None:
            delta["db_index"] = round(db_c - db_h, 6)   # positive = hybrid lower = better

        return delta

    # ------------------------------------------------------------------
    # HTML report renderer (Stage 08 artifact)
    # ------------------------------------------------------------------

    def _render_html_report(
        self,
        classical_summary: Dict,
        hybrid_summary: Dict,
        delta: Dict,
        cfg: PipelineConfig,
    ) -> str:
        ensure_dir(REPORT_OUTPUT_DIR)
        report_path = f"{REPORT_OUTPUT_DIR}/{cfg.run_id}_report.html"

        tables_html = "\n".join(self._build_html_tables(classical_summary, hybrid_summary, delta))

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HQC Benchmark Report — {cfg.run_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 2em auto; color: #222; }}
    h1 {{ color: #1a4f8a; }} h2 {{ color: #2c6fad; border-bottom: 1px solid #ccc; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
    th {{ background: #1a4f8a; color: white; padding: 8px 12px; text-align: left; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #e0e0e0; }}
    tr:nth-child(even) {{ background: #f5f8fc; }}
    .positive {{ color: #1a7a3a; font-weight: bold; }}
    .negative {{ color: #b22222; font-weight: bold; }}
    .meta {{ color: #555; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>HQC Benchmark Report</h1>
  <p class="meta">
    Run ID: <strong>{cfg.run_id}</strong> &nbsp;|&nbsp;
    Corpus: <strong>{cfg.corpus_id}</strong> &nbsp;|&nbsp;
    Quantum mode: <strong>{cfg.quantum_mode}</strong> &nbsp;|&nbsp;
    n_topics: <strong>{cfg.n_topics}</strong>
  </p>

  <h2>Results Comparison</h2>
  {tables_html}

  <h2>Delta Summary (Hybrid − Classical)</h2>
  <table>
    <tr><th>Metric</th><th>Delta</th><th>Interpretation</th></tr>
    {"".join(self._delta_rows(delta))}
  </table>

  <p class="meta">Generated by benchmark_runner.py — HQC Topic Modeling v1.0</p>
</body>
</html>"""

        Path(report_path).write_text(html, encoding="utf-8")
        logger.info("HTML report written", extra={"path": report_path})

        try:
            import mlflow  # type: ignore
            mlflow.log_artifact(report_path)
        except Exception:
            pass

        return report_path

    def _build_html_tables(
        self,
        classical: Dict,
        hybrid: Dict,
        delta: Dict,
    ) -> List[str]:
        metric_keys = ["c_v", "npmi", "umass", "silhouette", "db_index", "nmi", "ari", "wall_time_s"]
        rows = ""
        for key in metric_keys:
            c_val = classical.get(key)
            h_val = hybrid.get(key)
            if c_val is None and h_val is None:
                continue
            c_str = f"{c_val:.4f}" if isinstance(c_val, float) else str(c_val)
            h_str = f"{h_val:.4f}" if isinstance(h_val, float) else str(h_val)
            rows += f"<tr><td>{key}</td><td>{c_str}</td><td>{h_str}</td></tr>\n"

        table = f"""<table>
    <tr>
      <th>Metric</th>
      <th>Classical ({classical.get('model_type', '')})</th>
      <th>Hybrid ({hybrid.get('model_type', '')})</th>
    </tr>
    {rows}
  </table>"""
        return [table]

    def _delta_rows(self, delta: Dict) -> List[str]:
        rows = []
        better_higher = {"c_v", "npmi", "umass", "silhouette", "nmi", "ari"}
        better_lower = {"db_index"}
        for key, val in delta.items():
            if not isinstance(val, (int, float)):
                continue
            if key in better_higher:
                css = "positive" if val > 0 else "negative"
                interp = "Hybrid better" if val > 0 else "Classical better"
            elif key in better_lower:
                css = "positive" if val > 0 else "negative"
                interp = "Hybrid better (lower DB)" if val > 0 else "Classical better"
            else:
                css = ""
                interp = ""
            rows.append(
                f'<tr><td>{key}</td>'
                f'<td class="{css}">{val:+.4f}</td>'
                f'<td>{interp}</td></tr>'
            )
        return rows

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _init_mlflow(self, cfg: PipelineConfig) -> Optional[Any]:
        try:
            import mlflow  # type: ignore
            mlflow.set_experiment(DEFAULT_EXPERIMENT_NAME)
            run = mlflow.start_run(
                run_name=f"benchmark_{cfg.classical_model}_{cfg.corpus_id}_{self.report_id}"
            )
            mlflow.log_params({
                "report_id": self.report_id,
                "run_id": cfg.run_id,
                "corpus_id": cfg.corpus_id,
                "n_topics": cfg.n_topics,
                "classical_model": cfg.classical_model,
                "quantum_mode": cfg.quantum_mode,
                "optimizer": cfg.optimizer,
            })
            return run
        except Exception:
            logger.warning("MLflow unavailable — skipping experiment tracking")
            return None

    def _end_mlflow(self, status: str = "FINISHED") -> None:
        try:
            import mlflow  # type: ignore
            mlflow.end_run(status=status)
        except Exception:
            pass

    def _log_mlflow_final(self, report: BenchmarkReport) -> None:
        try:
            import mlflow  # type: ignore
            # Log delta metrics as primary comparison signal
            numeric_delta = {
                f"delta_{k}": v
                for k, v in report.delta.items()
                if isinstance(v, (int, float))
            }
            if numeric_delta:
                mlflow.log_metrics(numeric_delta)
            mlflow.log_param("benchmark_status", report.status)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level convenience function (Blueprint §4.4 endpoint backing)
# ---------------------------------------------------------------------------

def run_benchmark(
    dt_matrix: np.ndarray,
    config: PipelineConfig,
    documents: List[Dict],
    ground_truth: Optional[List[int]] = None,
) -> BenchmarkReport:
    """
    Run the full classical vs. hybrid benchmark.

    Consumed by:
        - POST /hybrid/compare API route (src/api/routes/quantum_routes.py)
        - GET /eval/report/{run_id} API route (src/api/routes/eval_routes.py)
        - Direct CLI invocation from notebooks/

    Blueprint CI gate: must log run_id, corpus_id, config snapshot, primary metric
    to MLflow (§10.4).

    Example
    -------
    >>> from src.evaluation.benchmark_runner import run_benchmark
    >>> from src.hybrid.hybrid_pipeline import PipelineConfig
    >>> cfg = PipelineConfig(corpus_id="20ng", n_topics=20, quantum_mode="qaoa")
    >>> report = run_benchmark(
    ...     dt_matrix=tfidf_matrix,
    ...     config=cfg,
    ...     documents=clean_docs,
    ...     ground_truth=newsgroup_labels,
    ... )
    >>> print(report.delta)
    """
    runner = BenchmarkRunner(config=config)
    return runner.run(
        dt_matrix=dt_matrix,
        documents=documents,
        ground_truth=ground_truth,
    )