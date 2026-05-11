"""
src/evaluation/cluster_metrics.py
=====================================
Team T-4: Hybrid Pipeline Team
MASTER BLUEPRINT v1.0 — §3.5

Responsibility:
    Compute Silhouette, Davies-Bouldin, NMI, and ARI cluster quality scores.

Inputs:  ClusterResult (from hybrid_pipeline.py or src/classical/kmeans_clusterer.py),
         optional ground_truth labels
Outputs: ClusterMetrics dataclass

Integration points:
    - Consumes T-4 hybrid ClusterResult and T-2 KMeans ClusterResult
    - Results consumed by benchmark_runner.py and /eval/cluster/{cluster_id} endpoint
    - Uses shared/logger.get_logger
    - MLflow logging (Blueprint §10.4)

Blueprint API endpoint: GET /eval/cluster/{cluster_id}
    → { cluster_id, silhouette, db_index, nmi }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from shared.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SILHOUETTE_SAMPLE_CAP: int = 5000     # avoid O(n²) on large corpora
MIN_CLUSTERS_FOR_METRICS: int = 2
NMI_AVERAGE_METHOD: str = "arithmetic"


# ---------------------------------------------------------------------------
# Output contract (Blueprint §5.2 / §4.5)
# ---------------------------------------------------------------------------

@dataclass
class ClusterMetrics:
    """
    Cluster quality result contract.

    Fields
    ------
    cluster_id    : Identifier of the evaluated cluster result.
    silhouette    : float ∈ [-1, 1] — higher is better.
    db_index      : float ≥ 0 — lower is better (Davies-Bouldin).
    nmi           : float ∈ [0, 1] — higher is better (only when ground_truth given).
    ari           : float ∈ [-1, 1] — higher is better (only when ground_truth given).
    n_clusters    : int — actual number of distinct clusters found.
    n_docs        : int — total documents evaluated.
    has_ground_truth : bool — whether NMI / ARI were computable.
    metadata      : Dict — provenance and config info.
    """
    cluster_id: str
    silhouette: float
    db_index: float
    nmi: float
    ari: float
    n_clusters: int
    n_docs: int
    has_ground_truth: bool = False
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class ClusterMetricsEvaluator:
    """
    Computes intrinsic (Silhouette, Davies-Bouldin) and extrinsic
    (NMI, ARI) cluster quality metrics.

    Silhouette and Davies-Bouldin require the embedding matrix to compute
    inter/intra-cluster distances. NMI and ARI require ground-truth labels.

    Example
    -------
    >>> evaluator = ClusterMetricsEvaluator()
    >>> metrics = evaluator.evaluate(
    ...     cluster_result=cluster_result_dict,
    ...     embeddings=tfidf_matrix,         # np.ndarray (n_docs, n_features)
    ...     ground_truth=[0,1,0,2,1,...],    # optional
    ...     cluster_id="abc123_clusters",
    ... )
    >>> print(metrics.silhouette, metrics.db_index, metrics.nmi)
    """

    def __init__(self, silhouette_sample_cap: int = SILHOUETTE_SAMPLE_CAP) -> None:
        self.silhouette_sample_cap = silhouette_sample_cap

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        cluster_result: Dict,
        embeddings: np.ndarray,
        cluster_id: str,
        ground_truth: Optional[List[int]] = None,
        corpus_id: str = "unknown",
        log_to_mlflow: bool = True,
    ) -> ClusterMetrics:
        """
        Compute cluster quality metrics.

        Parameters
        ----------
        cluster_result : ClusterResult dict (labels, k, cluster_id, …).
        embeddings     : np.ndarray (n_docs, n_features) — the representation
                         space used to compute distance-based metrics.
        cluster_id     : Identifier string for this result.
        ground_truth   : Optional list of integer class labels for NMI / ARI.
        corpus_id      : Source corpus identifier for provenance.
        log_to_mlflow  : Emit metrics to active MLflow run.

        Returns
        -------
        ClusterMetrics
        """
        labels = np.asarray(cluster_result.get("labels", []), dtype=int)
        n_docs = len(labels)
        n_clusters = len(np.unique(labels))

        logger.info(
            "Computing cluster metrics",
            extra={"cluster_id": cluster_id, "n_docs": n_docs, "n_clusters": n_clusters},
        )

        silhouette = self._silhouette(embeddings, labels, n_clusters)
        db_index = self._davies_bouldin(embeddings, labels, n_clusters)

        has_gt = ground_truth is not None and len(ground_truth) == n_docs
        nmi, ari = 0.0, 0.0
        if has_gt:
            nmi, ari = self._extrinsic(labels, ground_truth)

        metrics = ClusterMetrics(
            cluster_id=cluster_id,
            silhouette=round(silhouette, 6),
            db_index=round(db_index, 6),
            nmi=round(nmi, 6),
            ari=round(ari, 6),
            n_clusters=n_clusters,
            n_docs=n_docs,
            has_ground_truth=has_gt,
            metadata={
                "corpus_id": corpus_id,
                "k": cluster_result.get("k"),
                "source": cluster_result.get("source", "unknown"),
                "embeddings_shape": list(embeddings.shape),
            },
        )

        if log_to_mlflow:
            self._log_mlflow(metrics)

        logger.info(
            "Cluster metrics computed",
            extra={
                "cluster_id": cluster_id,
                "silhouette": metrics.silhouette,
                "db_index": metrics.db_index,
                "nmi": metrics.nmi if has_gt else "N/A",
            },
        )
        return metrics

    # ------------------------------------------------------------------
    # Intrinsic metrics
    # ------------------------------------------------------------------

    def _silhouette(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> float:
        if n_clusters < MIN_CLUSTERS_FOR_METRICS:
            logger.warning("Silhouette requires ≥ 2 clusters; returning 0.0")
            return 0.0
        n = len(labels)
        # Subsample to avoid O(n²) memory on large corpora
        if n > self.silhouette_sample_cap:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=self.silhouette_sample_cap, replace=False)
            return float(silhouette_score(embeddings[idx], labels[idx]))
        return float(silhouette_score(embeddings, labels))

    def _davies_bouldin(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_clusters: int,
    ) -> float:
        if n_clusters < MIN_CLUSTERS_FOR_METRICS:
            logger.warning("Davies-Bouldin requires ≥ 2 clusters; returning 0.0")
            return 0.0
        try:
            return float(davies_bouldin_score(embeddings, labels))
        except Exception as exc:
            logger.warning("Davies-Bouldin failed", extra={"reason": str(exc)})
            return 0.0

    # ------------------------------------------------------------------
    # Extrinsic metrics (require ground truth)
    # ------------------------------------------------------------------

    def _extrinsic(
        self,
        labels: np.ndarray,
        ground_truth: List[int],
    ) -> tuple[float, float]:
        gt = np.asarray(ground_truth, dtype=int)
        nmi = float(
            normalized_mutual_info_score(gt, labels, average_method=NMI_AVERAGE_METHOD)
        )
        ari = float(adjusted_rand_score(gt, labels))
        return nmi, ari

    # ------------------------------------------------------------------
    # MLflow logging
    # ------------------------------------------------------------------

    def _log_mlflow(self, metrics: ClusterMetrics) -> None:
        try:
            import mlflow  # type: ignore
            log_dict: Dict[str, float] = {
                "cluster_silhouette": metrics.silhouette,
                "cluster_db_index": metrics.db_index,
            }
            if metrics.has_ground_truth:
                log_dict["cluster_nmi"] = metrics.nmi
                log_dict["cluster_ari"] = metrics.ari
            mlflow.log_metrics(log_dict)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level convenience function (Blueprint §4.5 endpoint backing)
# ---------------------------------------------------------------------------

def compute_cluster_metrics(
    cluster_result: Dict,
    embeddings: np.ndarray,
    cluster_id: str,
    ground_truth: Optional[List[int]] = None,
    corpus_id: str = "unknown",
    log_to_mlflow: bool = True,
) -> ClusterMetrics:
    """
    Compute cluster quality metrics.

    Consumed by:
        - benchmark_runner.py (T-4)
        - GET /eval/cluster/{cluster_id} API route (src/api/routes/eval_routes.py)

    Example
    -------
    >>> from src.evaluation.cluster_metrics import compute_cluster_metrics
    >>> metrics = compute_cluster_metrics(
    ...     cluster_result=hybrid_result.cluster_result,
    ...     embeddings=tfidf_matrix,
    ...     cluster_id="abc123_clusters",
    ...     ground_truth=newsgroup_labels,
    ...     corpus_id="20ng",
    ... )
    >>> print(metrics.silhouette, metrics.db_index, metrics.nmi)
    """
    evaluator = ClusterMetricsEvaluator()
    return evaluator.evaluate(
        cluster_result=cluster_result,
        embeddings=embeddings,
        cluster_id=cluster_id,
        ground_truth=ground_truth,
        corpus_id=corpus_id,
        log_to_mlflow=log_to_mlflow,
    )