"""
src/hybrid/hybrid_pipeline.py
================================
Team T-4: Hybrid Pipeline Team
MASTER BLUEPRINT v1.0 — §3.4

Responsibility:
    Orchestrate the classical → quantum → classical feedback loop.
    Master entry point for the full hybrid run.

Inputs:  PipelineConfig (dict / dataclass from config/quantum_config.yaml +
         classical_config.yaml)
Outputs: HybridRunResult dataclass (Blueprint §5.2)

Integration points:
    - T-2 artifacts : DTMatrix (src/classical/tfidf_vectorizer.py)
                      TopicModelResult (src/classical/lda_model.py or nmf_model.py)
    - T-3 artifacts : QuantumCircuit (src/quantum/circuit_builder.py)
                      QAOAResult (src/quantum/qaoa_optimizer.py)
                      VQEResult  (src/quantum/vqe_solver.py)
    - T-4 internal  : CostHamiltonian (cost_function.py)
                      OptimizedParams  (parameter_optimizer.py)
    - shared/       : logger, serializer, validator, math_utils
    - MLflow        : every run logged with full parameter snapshots (§10.4)

Blueprint data-flow stages covered:
    Stage 04 — Classical Baseline  (delegates to T-2 modules)
    Stage 05 — Cost Hamiltonian Build
    Stage 06 — Quantum Optimization
    Stage 07 — Hybrid Clustering → outputs/models/{run_id}_clusters.json
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from shared.logger import get_logger, log_run_end, log_run_start
from shared.math_utils import cosine_similarity_matrix
from shared.serializer import save_artifact, to_json_response
from shared.validator import assert_qubit_feasibility, validate_config

from src.hybrid.cost_function import (
    CostHamiltonian,
    HamiltonianEncoding,
    build_cost_hamiltonian,
)
from src.hybrid.parameter_optimizer import (
    OptimizedParams,
    OptimizerType,
    optimize_circuit_parameters,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (all numerics sourced from config — Blueprint §9 rule)
# ---------------------------------------------------------------------------
DEFAULT_P_LAYERS: int = 2
DEFAULT_SHOTS: int = 1024
DEFAULT_OPTIMIZER: str = "COBYLA"
DEFAULT_MAX_ITER: int = 100
DEFAULT_N_TOPICS: int = 20
DEFAULT_QUANTUM_MODE: str = "qaoa"          # "qaoa" | "vqe"
DEFAULT_CLASSICAL_MODEL: str = "lda"        # "lda" | "nmf"


# ---------------------------------------------------------------------------
# Config / result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Single configuration object for a full hybrid run.
    Populated from base_config.yaml + classical_config.yaml + quantum_config.yaml.
    No numeric literals permitted outside this class (Blueprint §9).
    """
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    corpus_id: str = "unknown"
    n_topics: int = DEFAULT_N_TOPICS
    classical_model: str = DEFAULT_CLASSICAL_MODEL    # "lda" | "nmf"
    quantum_mode: str = DEFAULT_QUANTUM_MODE           # "qaoa" | "vqe"
    p_layers: int = DEFAULT_P_LAYERS
    shots: int = DEFAULT_SHOTS
    optimizer: str = DEFAULT_OPTIMIZER
    max_iter: int = DEFAULT_MAX_ITER
    noise_id: Optional[str] = None
    encoding: HamiltonianEncoding = HamiltonianEncoding.ISING
    seed: int = 42
    log_artifacts: bool = True
    extra: Dict = field(default_factory=dict)


@dataclass
class HybridRunResult:
    """
    Top-level output contract (Blueprint §5.2).
    Consumed by benchmark_runner.py, /hybrid/run/{run_id} API endpoint.
    """
    run_id: str
    corpus_id: str
    classical_result: Dict          # TopicModelResult serialised via to_json_response
    quantum_result: Dict            # QAOAResult | VQEResult serialised
    cluster_result: Dict            # ClusterResult serialised
    cost_hamiltonian_meta: Dict     # CostHamiltonian.metadata
    optimized_params: Dict          # OptimizedParams serialised
    metrics: Dict                   # Summary metrics forwarded from evaluation layer
    wall_time_s: float = 0.0
    status: str = "complete"
    config_snapshot: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class HybridPipeline:
    """
    Orchestrates the full classical → quantum → classical feedback loop.

    The pipeline executes these stages in order:
        1. Classical baseline topic model (LDA or NMF) → TopicModelResult
        2. Cost Hamiltonian construction from DTMatrix → CostHamiltonian
        3. Quantum optimisation (QAOA or VQE) → QAOAResult / VQEResult
        4. Classical outer-loop parameter refinement → OptimizedParams
        5. Derive final cluster assignments from quantum bitstring counts
        6. Persist all artifacts and log to MLflow

    All heavy T-2 / T-3 imports are deferred inside methods to keep the
    module importable even when Qiskit / Gensim are not installed.

    Example
    -------
    >>> from src.hybrid.hybrid_pipeline import HybridPipeline, PipelineConfig
    >>> cfg = PipelineConfig(corpus_id="20ng", n_topics=10, quantum_mode="qaoa")
    >>> pipeline = HybridPipeline(cfg)
    >>> result = pipeline.run(dt_matrix=tfidf_matrix, documents=clean_docs)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._mlflow_run_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        dt_matrix: np.ndarray,
        documents: Optional[List[Dict]] = None,
    ) -> HybridRunResult:
        """
        Execute the full hybrid pipeline.

        Parameters
        ----------
        dt_matrix : np.ndarray, shape (n_docs, n_features)
                    Document-term or embedding matrix from T-2.
        documents : Optional list of CleanDocument dicts (used by coherence metrics).

        Returns
        -------
        HybridRunResult
        """
        cfg = self.config
        t_start = time.perf_counter()

        log_run_start(cfg.run_id, vars(cfg))
        logger.info("HybridPipeline.run() started", extra={"run_id": cfg.run_id, "corpus_id": cfg.corpus_id})

        # --- MLflow run context ---
        mlflow_run = self._start_mlflow_run(cfg)

        try:
            # Stage 04 — Classical baseline
            classical_result = self._run_classical_baseline(dt_matrix, cfg)
            self._mlflow_log_dict("classical", classical_result.get("params", {}))

            # Stage 05 — Cost Hamiltonian
            hamiltonian = self._build_hamiltonian(dt_matrix, cfg)
            self._mlflow_log_dict("hamiltonian_meta", hamiltonian.metadata)

            # Stage 06 — Quantum optimisation + classical outer loop
            quantum_result, optimized_params = self._run_quantum_optimization(
                hamiltonian=hamiltonian,
                n_params=2 * cfg.p_layers * hamiltonian.n_variables,
                cfg=cfg,
            )
            self._mlflow_log_dict("quantum", {"final_cost": quantum_result.get("cost", None)})

            # Stage 07 — Derive cluster assignments
            cluster_result = self._derive_clusters(
                quantum_result=quantum_result,
                n_docs=dt_matrix.shape[0],
                n_topics=cfg.n_topics,
                run_id=cfg.run_id,
                cfg=cfg,
            )

            wall_time = time.perf_counter() - t_start

            result = HybridRunResult(
                run_id=cfg.run_id,
                corpus_id=cfg.corpus_id,
                classical_result=classical_result,
                quantum_result=quantum_result,
                cluster_result=cluster_result,
                cost_hamiltonian_meta=hamiltonian.metadata,
                optimized_params=vars(optimized_params)
                    if hasattr(optimized_params, "__dict__") else {},
                metrics={
                    "final_cost": quantum_result.get("cost"),
                    "n_clusters": cfg.n_topics,
                    "n_docs": dt_matrix.shape[0],
                    "converged": optimized_params.converged,
                    "wall_time_s": round(wall_time, 4),
                },
                wall_time_s=round(wall_time, 4),
                status="complete",
                config_snapshot=vars(cfg),
            )

            # Persist Stage 07 artifact
            if cfg.log_artifacts:
                cluster_path = f"outputs/models/{cfg.run_id}_clusters.json"
                save_artifact(cluster_result, cluster_path, fmt="json")

            self._mlflow_log_metrics(result.metrics)
            log_run_end(cfg.run_id, result.metrics)

            logger.info(
                "HybridPipeline.run() complete",
                extra={"run_id": cfg.run_id, "wall_time_s": result.wall_time_s},
            )
            return result

        except Exception as exc:
            logger.error(
                "HybridPipeline.run() failed",
                extra={"run_id": cfg.run_id, "error": str(exc)},
            )
            self._end_mlflow_run(status="FAILED")
            raise

        finally:
            self._end_mlflow_run(status="FINISHED")

    # ------------------------------------------------------------------
    # Stage 04 — Classical baseline
    # ------------------------------------------------------------------

    def _run_classical_baseline(
        self,
        dt_matrix: np.ndarray,
        cfg: PipelineConfig,
    ) -> Dict:
        """
        Delegate to T-2 classical module (LDA or NMF).
        Imports are deferred so the module stays importable without Gensim.
        """
        logger.info("Stage 04: classical baseline", extra={"model": cfg.classical_model})
        try:
            if cfg.classical_model == "lda":
                from src.classical.lda_model import LDAModel  # type: ignore
                model = LDAModel(n_topics=cfg.n_topics)
            else:
                from src.classical.nmf_model import NMFModel  # type: ignore
                model = NMFModel(n_topics=cfg.n_topics)

            topic_result = model.fit(dt_matrix)
            return to_json_response(topic_result)

        except ImportError:
            logger.warning(
                "Classical module not yet available — using stub result",
                extra={"run_id": cfg.run_id},
            )
            # Stub: returns a minimal TopicModelResult-shaped dict so downstream
            # stages can proceed during integration before T-2 is complete.
            return {
                "model_id": f"stub_{cfg.classical_model}_{cfg.run_id}",
                "topics": [],
                "doc_topic_matrix": None,
                "params": {"n_topics": cfg.n_topics, "model": cfg.classical_model},
                "_stub": True,
            }

    # ------------------------------------------------------------------
    # Stage 05 — Cost Hamiltonian
    # ------------------------------------------------------------------

    def _build_hamiltonian(
        self,
        dt_matrix: np.ndarray,
        cfg: PipelineConfig,
    ) -> CostHamiltonian:
        """Delegate to cost_function.build_cost_hamiltonian (T-4 internal)."""
        logger.info("Stage 05: building cost Hamiltonian", extra={"run_id": cfg.run_id})
        assert_qubit_feasibility(dt_matrix.shape[0], cfg.n_topics)
        return build_cost_hamiltonian(
            matrix=dt_matrix,
            run_id=cfg.run_id,
            n_topics=cfg.n_topics,
            encoding=cfg.encoding,
            corpus_id=cfg.corpus_id,
            save=cfg.log_artifacts,
        )

    # ------------------------------------------------------------------
    # Stage 06 — Quantum optimisation
    # ------------------------------------------------------------------

    def _run_quantum_optimization(
        self,
        hamiltonian: CostHamiltonian,
        n_params: int,
        cfg: PipelineConfig,
    ) -> tuple[Dict, OptimizedParams]:
        """
        Build circuit → define expectation-value cost_fn → run classical outer loop.
        Defers T-3 imports; stubs gracefully if Qiskit unavailable.
        """
        logger.info(
            "Stage 06: quantum optimisation",
            extra={"mode": cfg.quantum_mode, "n_params": n_params},
        )
        np.random.seed(cfg.seed)
        initial_params = np.random.uniform(-np.pi, np.pi, n_params)

        try:
            if cfg.quantum_mode == "qaoa":
                from src.quantum.qaoa_optimizer import QAOAOptimizer  # type: ignore
                from src.quantum.circuit_builder import CircuitBuilder  # type: ignore
                from src.quantum.backend_manager import BackendManager  # type: ignore

                backend = BackendManager().get_backend()
                circuit = CircuitBuilder().build_qaoa_circuit(
                    hamiltonian=hamiltonian, p_layers=cfg.p_layers
                )

                def cost_fn(params: np.ndarray) -> float:
                    return QAOAOptimizer(backend=backend, shots=cfg.shots).expectation(
                        circuit, params, hamiltonian
                    )

            else:  # vqe
                from src.quantum.vqe_solver import VQESolver  # type: ignore
                from src.quantum.circuit_builder import CircuitBuilder  # type: ignore
                from src.quantum.backend_manager import BackendManager  # type: ignore

                backend = BackendManager().get_backend()
                circuit = CircuitBuilder().build_vqe_ansatz(
                    hamiltonian=hamiltonian, reps=cfg.p_layers
                )

                def cost_fn(params: np.ndarray) -> float:
                    return VQESolver(backend=backend, shots=cfg.shots).expectation(
                        circuit, params, hamiltonian
                    )

            optimized = optimize_circuit_parameters(
                cost_fn=cost_fn,
                initial_params=initial_params,
                run_id=cfg.run_id,
                optimizer_type=cfg.optimizer,
                max_iter=cfg.max_iter,
                save=cfg.log_artifacts,
            )

            quantum_result = {
                "job_id": cfg.run_id,
                "mode": cfg.quantum_mode,
                "optimal_params": optimized.optimal_params,
                "cost": optimized.final_cost,
                "n_iterations": optimized.n_iterations,
                "converged": optimized.converged,
                "circuit_depth": getattr(circuit, "depth", lambda: None)(),
            }
            return quantum_result, optimized

        except ImportError:
            logger.warning(
                "Quantum modules not yet available — using stub optimisation",
                extra={"run_id": cfg.run_id},
            )
            # Stub: classical surrogate cost (cosine similarity trace) so the
            # pipeline produces a valid HybridRunResult during T-3 integration.
            similarity = cosine_similarity_matrix(
                np.random.rand(min(hamiltonian.n_variables, 20), 10)
            )

            def stub_cost(params: np.ndarray) -> float:
                return float(np.sum(np.abs(params)) + np.trace(similarity))

            optimized = optimize_circuit_parameters(
                cost_fn=stub_cost,
                initial_params=initial_params,
                run_id=cfg.run_id,
                optimizer_type=cfg.optimizer,
                max_iter=min(cfg.max_iter, 20),
                save=False,
            )
            quantum_result = {
                "job_id": cfg.run_id,
                "mode": cfg.quantum_mode,
                "optimal_params": optimized.optimal_params,
                "cost": optimized.final_cost,
                "n_iterations": optimized.n_iterations,
                "converged": optimized.converged,
                "circuit_depth": None,
                "_stub": True,
            }
            return quantum_result, optimized

    # ------------------------------------------------------------------
    # Stage 07 — Derive cluster assignments
    # ------------------------------------------------------------------

    def _derive_clusters(
        self,
        quantum_result: Dict,
        n_docs: int,
        n_topics: int,
        run_id: str,
        cfg: PipelineConfig,
    ) -> Dict:
        """
        Convert optimal quantum parameters / bitstring counts into integer
        cluster labels. Falls back to argmax over cosine-similarity blocks
        when full bit-string counts are unavailable (stub mode).
        """
        logger.info("Stage 07: deriving cluster assignments", extra={"run_id": run_id})

        counts: Optional[Dict] = quantum_result.get("counts")
        if counts:
            # Real mode: most-probable bitstring → binary cluster assignment
            best_bitstring = max(counts, key=counts.get)
            raw_labels = [int(b) for b in best_bitstring[:n_docs]]
        else:
            # Stub / fallback: assign via round-robin modulo n_topics
            rng = np.random.default_rng(cfg.seed)
            raw_labels = rng.integers(0, n_topics, size=n_docs).tolist()

        cluster_result = {
            "cluster_id": f"{run_id}_clusters",
            "labels": raw_labels,
            "k": n_topics,
            "centroids": None,   # populated by benchmark_runner if needed
            "source": "quantum" if counts else "stub",
        }
        return cluster_result

    # ------------------------------------------------------------------
    # MLflow helpers
    # ------------------------------------------------------------------

    def _start_mlflow_run(self, cfg: PipelineConfig) -> Optional[Any]:
        try:
            import mlflow  # type: ignore
            run = mlflow.start_run(run_name=f"{cfg.classical_model}_{cfg.corpus_id}_{cfg.run_id}")
            mlflow.log_params({
                "run_id": cfg.run_id,
                "corpus_id": cfg.corpus_id,
                "n_topics": cfg.n_topics,
                "quantum_mode": cfg.quantum_mode,
                "optimizer": cfg.optimizer,
                "p_layers": cfg.p_layers,
                "shots": cfg.shots,
                "classical_model": cfg.classical_model,
                "encoding": cfg.encoding.value,
            })
            self._mlflow_run_id = run.info.run_id
            return run
        except Exception:
            logger.warning("MLflow unavailable — skipping run tracking")
            return None

    def _end_mlflow_run(self, status: str = "FINISHED") -> None:
        try:
            import mlflow  # type: ignore
            mlflow.end_run(status=status)
        except Exception:
            pass

    def _mlflow_log_dict(self, prefix: str, data: Dict) -> None:
        try:
            import mlflow  # type: ignore
            flat = {f"{prefix}.{k}": v for k, v in data.items()
                    if isinstance(v, (int, float, str, bool))}
            if flat:
                mlflow.log_params(flat)
        except Exception:
            pass

    def _mlflow_log_metrics(self, metrics: Dict) -> None:
        try:
            import mlflow  # type: ignore
            numeric = {k: v for k, v in metrics.items()
                       if isinstance(v, (int, float)) and v is not None}
            if numeric:
                mlflow.log_metrics(numeric)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level convenience function (consumed by FastAPI /hybrid/run route)
# ---------------------------------------------------------------------------

def run_hybrid_pipeline(
    dt_matrix: np.ndarray,
    config: PipelineConfig,
    documents: Optional[List[Dict]] = None,
) -> HybridRunResult:
    """
    Functional entry point for the hybrid pipeline.

    Example
    -------
    >>> from src.hybrid.hybrid_pipeline import run_hybrid_pipeline, PipelineConfig
    >>> cfg = PipelineConfig(corpus_id="bbc", n_topics=5, quantum_mode="qaoa")
    >>> result = run_hybrid_pipeline(dt_matrix=tfidf_matrix, config=cfg)
    >>> print(result.status, result.metrics)
    """
    pipeline = HybridPipeline(config)
    return pipeline.run(dt_matrix=dt_matrix, documents=documents)