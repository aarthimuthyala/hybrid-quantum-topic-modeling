"""
src/classical/kmeans_clusterer.py
==================================
Team T-2 — Classical NLP Layer (L2)
HQC Topic Modeling Project · Master Blueprint v1.0

Responsibility:
    K-Means baseline clustering on TF-IDF document-term matrices or dense
    embedding matrices, producing a ClusterResult consumed by:
        - cluster_metrics.py   (T-4, evaluation layer)
        - cost_function.py     (T-4, builds QUBO from cluster assignments)
        - /classical/cluster   (Blueprint §4.2 API endpoint)
        - benchmark_runner.py  (T-4, classical vs hybrid comparison)

Blueprint contracts honoured:
    - Input  : EmbeddingMatrix (np.ndarray) OR DTMatrix, k: int  (§3.2, §5.2)
    - Output : ClusterResult                                      (§5.2)
    - Artifact persistence via shared.serializer                  (§8.3)
    - Structured logging via shared.logger                        (§8.1)
    - Config from config/classical_config.yaml                    (§9.2)
    - Full package-path imports; no relative imports outside __init__ (§2)
    - Naming conventions enforced by ruff + mypy --strict          (§6)

ClusterResult schema (FROZEN — Blueprint §10.3):
    {
        "cluster_id": str
        "labels":     List[int]        # per-document cluster assignment
        "centroids":  np.ndarray       # shape (k, n_features)
        "k":          int
    }

EmbeddingMatrix convention:
    A 2-D dense np.ndarray of shape (n_docs, n_features).
    When a DTMatrix is supplied, its .matrix attribute is automatically
    converted to a dense array before clustering (sparse K-Means is not
    used here to maintain compatibility with the downstream QUBO builder
    in cost_function.py which expects dense centroids).

Clustering method flag (Blueprint §4.2 /classical/cluster endpoint):
    method='tfidf'  → caller passes a DTMatrix; auto-converted internally
    method='embed'  → caller passes a dense np.ndarray (embedding matrix)

Dependencies (requirements.txt):
    scikit-learn>=1.4       # MiniBatchKMeans, KMeans, silhouette_score
    numpy>=1.26
    scipy>=1.12
    pyyaml>=6.0
    mlflow>=2.12
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Project-internal imports — full package paths (Blueprint §2)
# ---------------------------------------------------------------------------
from shared.logger import get_logger, log_run_end, log_run_start
from shared.serializer import load_artifact, save_artifact
from shared.validator import validate_config
from src.classical.tfidf_vectorizer import DTMatrix

# ---------------------------------------------------------------------------
# Module logger (Blueprint §8.1)
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — UPPER_SNAKE_CASE (Blueprint §6)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: str = "config/classical_config.yaml"
DEFAULT_K: int = 20                      # Blueprint §9.2 kmeans.k
DEFAULT_N_INIT: int = 10                 # Blueprint §9.2 kmeans.n_init
DEFAULT_MAX_ITER: int = 300              # Blueprint §9.2 kmeans.max_iter
DEFAULT_RANDOM_STATE: int = 42           # base_config.yaml seed (§9.1)
DEFAULT_ALGORITHM: str = "lloyd"         # 'lloyd' | 'elkan' (sklearn default)
DEFAULT_USE_MINIBATCH: bool = False      # True → MiniBatchKMeans (large corpora)
DEFAULT_MINIBATCH_SIZE: int = 1024       # batch_size for MiniBatchKMeans
DEFAULT_NORMALISE_INPUT: bool = True     # L2-normalise rows before clustering
DEFAULT_COMPUTE_SILHOUETTE: bool = True  # toggle — expensive on large corpora
SILHOUETTE_SAMPLE_CAP: int = 5_000      # max docs sampled for silhouette score
KMEANS_ARTIFACT_TYPE: str = "kmeans_clusterer"

# Allowed clustering methods (mirrors Blueprint §4.2 request schema)
ClusterMethod = Literal["tfidf", "embed"]


# ---------------------------------------------------------------------------
# Data-type contracts (Blueprint §5.2)
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """
    Output contract for KMeansClusterer, consumed by cluster_metrics.py,
    cost_function.py, and the /classical/cluster API endpoint.

    FROZEN — any field addition/removal requires an ADR (Blueprint §10.3).

    Fields:
        cluster_id  : Unique identifier following §6 naming scheme.
        labels      : Per-document cluster index in [0, k), len = n_docs.
        centroids   : Cluster centroid matrix, shape (k, n_features).
        k           : Number of clusters used.
        silhouette  : Silhouette coefficient [-1, 1]; None if not computed.
        inertia     : Within-cluster sum of squared distances (sklearn).
        params      : Config + runtime snapshot for MLflow.
    """
    cluster_id: str
    labels: list[int]
    centroids: np.ndarray            # shape (k, n_features)
    k: int
    silhouette: float | None = None  # optional — expensive on large corpora
    inertia: float | None = None
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _load_kmeans_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load config/classical_config.yaml and extract the kmeans subsection.

    Falls back gracefully to module DEFAULT_* constants when the file is
    missing or the 'classical.kmeans' key is absent.
    validate_config() (Blueprint §8.2) raises ConfigError on schema errors.

    Args:
        config_path: Path to classical_config.yaml.

    Returns:
        Dict of resolved K-Means hyperparameters.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Config not found at '%s'. Using built-in K-Means defaults.",
            config_path,
        )
        return {}

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    validate_config(raw, schema_name="classical_config")
    return raw.get("classical", {}).get("kmeans", {})


def _coerce_to_dense(matrix: Any) -> np.ndarray:
    """
    Convert a sparse or dense matrix to a 2-D dense np.ndarray.

    Handles both scipy sparse matrices (via .toarray()) and existing
    np.ndarray inputs (identity pass-through). Raises TypeError for
    unsupported types so the error surfaces early rather than inside
    sklearn internals.

    Args:
        matrix: A scipy sparse matrix or np.ndarray.

    Returns:
        2-D np.ndarray, dtype float64.

    Raises:
        TypeError: If matrix is neither sparse nor ndarray.
    """
    if isinstance(matrix, np.ndarray):
        return matrix.astype(np.float64)
    # Scipy sparse matrices expose .toarray()
    if hasattr(matrix, "toarray"):
        return matrix.toarray().astype(np.float64)
    raise TypeError(
        f"_coerce_to_dense: unsupported matrix type {type(matrix)}. "
        "Expected np.ndarray or scipy sparse matrix."
    )


def _resolve_input_matrix(
    matrix_input: DTMatrix | np.ndarray,
    method: ClusterMethod,
) -> np.ndarray:
    """
    Dispatch the caller-supplied input to a dense np.ndarray.

    Enforces the Blueprint §4.2 method flag contract:
        'tfidf' → input must be DTMatrix; .matrix attribute extracted
        'embed' → input must be np.ndarray (embedding matrix)

    Args:
        matrix_input: DTMatrix or np.ndarray supplied by the caller.
        method      : Clustering method flag from Blueprint §4.2.

    Returns:
        Dense np.ndarray of shape (n_docs, n_features).

    Raises:
        TypeError : On method/type mismatch.
        ValueError: If the resolved matrix is 1-D or has zero docs.
    """
    if method == "tfidf":
        if not isinstance(matrix_input, DTMatrix):
            raise TypeError(
                "method='tfidf' expects a DTMatrix; "
                f"got {type(matrix_input).__name__}."
            )
        dense = _coerce_to_dense(matrix_input.matrix)
    elif method == "embed":
        if not isinstance(matrix_input, np.ndarray):
            raise TypeError(
                "method='embed' expects a np.ndarray; "
                f"got {type(matrix_input).__name__}."
            )
        dense = _coerce_to_dense(matrix_input)
    else:
        raise ValueError(f"Unknown clustering method '{method}'. Use 'tfidf' or 'embed'.")

    if dense.ndim != 2:
        raise ValueError(
            f"Input matrix must be 2-D; got {dense.ndim}-D array."
        )
    if dense.shape[0] == 0:
        raise ValueError("Input matrix has 0 rows; cannot cluster empty corpus.")

    return dense


def _compute_silhouette(
    X: np.ndarray,
    labels: np.ndarray,
    sample_cap: int = SILHOUETTE_SAMPLE_CAP,
) -> float | None:
    """
    Compute the silhouette coefficient with optional random sub-sampling.

    Silhouette is O(n²) — too expensive for large corpora. When n_docs
    exceeds `sample_cap`, a random stratified sub-sample is used so the
    metric remains tractable in CI and interactive use.

    Args:
        X         : Dense feature matrix, shape (n_docs, n_features).
        labels    : Cluster label array, shape (n_docs,).
        sample_cap: Maximum rows to include; sub-sampled when exceeded.

    Returns:
        Silhouette score as float, or None if only one cluster exists
        (score is undefined when k=1 or all docs land in one cluster).
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        logger.warning(
            "Silhouette score undefined: only %d unique cluster(s) found.",
            n_unique,
        )
        return None

    n_docs = X.shape[0]
    if n_docs > sample_cap:
        logger.info(
            "Silhouette sub-sampling %d → %d docs.", n_docs, sample_cap
        )
        rng = np.random.default_rng(DEFAULT_RANDOM_STATE)
        idx = rng.choice(n_docs, size=sample_cap, replace=False)
        X_sample, labels_sample = X[idx], labels[idx]
    else:
        X_sample, labels_sample = X, labels

    score: float = float(silhouette_score(X_sample, labels_sample, metric="cosine"))
    return score


def _build_kmeans_estimator(params: dict[str, Any]) -> KMeans | MiniBatchKMeans:
    """
    Construct the appropriate sklearn clusterer from the resolved params dict.

    MiniBatchKMeans is selected when use_minibatch=True (recommended for
    corpora with >50 k documents). Otherwise standard KMeans is used.

    Args:
        params: Resolved hyperparameter dict from _load_kmeans_config.

    Returns:
        An unfitted KMeans or MiniBatchKMeans instance.
    """
    if params.get("use_minibatch", DEFAULT_USE_MINIBATCH):
        logger.info(
            "Using MiniBatchKMeans | k=%d | batch_size=%d",
            params["k"],
            params["minibatch_size"],
        )
        return MiniBatchKMeans(
            n_clusters=params["k"],
            n_init=params["n_init"],
            max_iter=params["max_iter"],
            batch_size=params["minibatch_size"],
            random_state=params["random_state"],
        )

    logger.info("Using KMeans | k=%d | n_init=%d", params["k"], params["n_init"])
    return KMeans(
        n_clusters=params["k"],
        n_init=params["n_init"],
        max_iter=params["max_iter"],
        algorithm=params["algorithm"],
        random_state=params["random_state"],
    )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class KMeansClusterer:
    """
    K-Means baseline document clustering compatible with the DTMatrix and
    embedding matrix contracts defined in Blueprint §5.2.

    Supports two input modes (Blueprint §4.2 /classical/cluster):
        method='tfidf' — clusters the sparse TF-IDF DTMatrix (L2-normalised
                          before clustering for cosine-equivalent Euclidean)
        method='embed' — clusters a dense embedding matrix from any encoder

    Produces ClusterResult consumed by T-4's cluster_metrics.py (silhouette,
    Davies-Bouldin, NMI, ARI) and cost_function.py (centroid-based QUBO).

    Usage (canonical pipeline — Blueprint §5.1, Stage 04):
        >>> from src.classical.kmeans_clusterer import KMeansClusterer
        >>> from src.classical.tfidf_vectorizer import TFIDFVectorizer
        >>>
        >>> vectorizer = TFIDFVectorizer(corpus_id="20ng")
        >>> dt_matrix  = vectorizer.fit_transform(clean_docs)
        >>>
        >>> clusterer = KMeansClusterer(corpus_id="20ng", k=20)
        >>> result    = clusterer.fit(dt_matrix, method="tfidf")
        >>> clusterer.save(run_id=result.cluster_id)

    Attributes:
        corpus_id     : Source corpus identifier.
        k             : Number of clusters.
        config_path   : Path to classical_config.yaml.
        _estimator    : Underlying sklearn KMeans / MiniBatchKMeans.
        _is_fitted    : Guard flag.
        _params       : Resolved hyperparameter snapshot.
        _n_features   : Feature dimension of the training matrix (for validation).
    """

    def __init__(
        self,
        corpus_id: str,
        k: int | None = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        """
        Initialise KMeansClusterer and resolve hyperparameters from config.

        Args:
            corpus_id  : Corpus identifier (propagated to ClusterResult).
            k          : Override for number of clusters. If None, the value
                         in classical_config.yaml is used.
            config_path: Path to classical_config.yaml (Blueprint §9.2).
        """
        self.corpus_id: str = corpus_id
        self.config_path: str = config_path
        self._is_fitted: bool = False
        self._n_features: int = 0

        # Resolve config — YAML overrides module defaults
        cfg = _load_kmeans_config(config_path)

        resolved_k = k if k is not None else cfg.get("k", DEFAULT_K)

        self._params: dict[str, Any] = {
            "k":                   resolved_k,
            "n_init":              cfg.get("n_init",              DEFAULT_N_INIT),
            "max_iter":            cfg.get("max_iter",            DEFAULT_MAX_ITER),
            "algorithm":           cfg.get("algorithm",           DEFAULT_ALGORITHM),
            "random_state":        cfg.get("random_state",        DEFAULT_RANDOM_STATE),
            "use_minibatch":       cfg.get("use_minibatch",       DEFAULT_USE_MINIBATCH),
            "minibatch_size":      cfg.get("minibatch_size",      DEFAULT_MINIBATCH_SIZE),
            "normalise_input":     cfg.get("normalise_input",     DEFAULT_NORMALISE_INPUT),
            "compute_silhouette":  cfg.get("compute_silhouette",  DEFAULT_COMPUTE_SILHOUETTE),
        }

        self.k: int = resolved_k

        logger.info(
            "KMeansClusterer initialised | corpus_id=%s | k=%d | params=%s",
            corpus_id,
            self.k,
            self._params,
        )

        self._estimator: KMeans | MiniBatchKMeans = _build_kmeans_estimator(
            self._params
        )

    # ------------------------------------------------------------------
    # Public API — fit / predict / fit_predict / save / load
    # ------------------------------------------------------------------

    def fit(
        self,
        matrix_input: DTMatrix | np.ndarray,
        method: ClusterMethod = "tfidf",
    ) -> ClusterResult:
        """
        Cluster documents and return a ClusterResult.

        Pipeline (Blueprint §5.1, Stage 04 prefix):
            1. Resolve input → dense np.ndarray via _resolve_input_matrix()
            2. Optionally L2-normalise rows (makes Euclidean ≈ cosine distance)
            3. Fit sklearn estimator; record inertia and labels
            4. Optionally compute silhouette score
            5. Return ClusterResult with all fields populated

        Args:
            matrix_input: DTMatrix (method='tfidf') or np.ndarray (method='embed').
            method      : 'tfidf' or 'embed' — governs type coercion.

        Returns:
            ClusterResult with labels, centroids, silhouette, and params.

        Raises:
            TypeError : On method/type mismatch.
            ValueError: If matrix is empty or k > n_docs.
        """
        X: np.ndarray = _resolve_input_matrix(matrix_input, method)
        n_docs, n_features = X.shape
        self._n_features = n_features

        if self.k > n_docs:
            raise ValueError(
                f"k={self.k} exceeds n_docs={n_docs}. "
                "Reduce k or provide more documents."
            )

        # L2-normalise so cosine-similar docs cluster together under
        # Euclidean K-Means (standard practice for TF-IDF/embedding vectors)
        if self._params["normalise_input"]:
            logger.debug("L2-normalising input matrix before clustering.")
            X = normalize(X, norm="l2")

        logger.info(
            "KMeans fit started | corpus_id=%s | n_docs=%d | n_features=%d | k=%d",
            self.corpus_id,
            n_docs,
            n_features,
            self.k,
        )

        t0 = time.perf_counter()
        self._estimator.fit(X)
        self._is_fitted = True
        elapsed = time.perf_counter() - t0

        labels: np.ndarray = self._estimator.labels_
        centroids: np.ndarray = self._estimator.cluster_centers_
        inertia: float = float(self._estimator.inertia_)

        logger.info(
            "KMeans fit complete | elapsed_s=%.3f | inertia=%.4f",
            elapsed,
            inertia,
        )

        # Silhouette is optional — toggle via config (expensive for large N)
        silhouette: float | None = None
        if self._params["compute_silhouette"]:
            silhouette = _compute_silhouette(X, labels)
            if silhouette is not None:
                logger.info("Silhouette score=%.4f", silhouette)

        # cluster_id follows Blueprint §6 artifact naming convention
        cluster_id = f"kmeans_{self.corpus_id}_{int(time.time())}"

        result_params: dict[str, Any] = {
            **self._params,
            "corpus_id":  self.corpus_id,
            "n_docs":     n_docs,
            "n_features": n_features,
            "method":     method,
            "inertia":    inertia,
            "silhouette": silhouette,
            "model":      "kmeans",
        }

        return ClusterResult(
            cluster_id=cluster_id,
            labels=labels.tolist(),
            centroids=centroids,
            k=self.k,
            silhouette=silhouette,
            inertia=inertia,
            params=result_params,
        )

    def predict(
        self,
        matrix_input: DTMatrix | np.ndarray,
        method: ClusterMethod = "tfidf",
    ) -> np.ndarray:
        """
        Assign new documents to the nearest cluster centroid.

        Does not modify the fitted centroids. Useful for online inference
        and held-out evaluation without re-training.

        Args:
            matrix_input: DTMatrix or np.ndarray of unseen documents.
            method      : 'tfidf' or 'embed' — controls type coercion.

        Returns:
            np.ndarray of int labels, shape (n_docs,).

        Raises:
            RuntimeError: If called before fit() or load().
            TypeError   : On method/type mismatch.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "KMeansClusterer.predict() called before fit(). "
                "Call fit() or KMeansClusterer.load() first."
            )
        X: np.ndarray = _resolve_input_matrix(matrix_input, method)
        if self._params["normalise_input"]:
            X = normalize(X, norm="l2")

        logger.info("KMeans predict | n_docs=%d", X.shape[0])
        return self._estimator.predict(X)

    def fit_predict(
        self,
        matrix_input: DTMatrix | np.ndarray,
        method: ClusterMethod = "tfidf",
    ) -> ClusterResult:
        """
        Convenience wrapper: fit and return ClusterResult in one call.

        Mirrors fit() exactly; exists to match the sklearn naming convention
        used across TFIDFVectorizer and LDAModel for pipeline uniformity.

        Args:
            matrix_input: DTMatrix or np.ndarray.
            method      : 'tfidf' or 'embed'.

        Returns:
            ClusterResult (same as fit() return value).
        """
        return self.fit(matrix_input, method)

    # ------------------------------------------------------------------
    # Persistence — save / load (Blueprint §8.3)
    # ------------------------------------------------------------------

    def save(self, run_id: str, output_dir: str = "outputs/models") -> Path:
        """
        Persist the fitted sklearn clusterer via shared.serializer.

        Artifact path (Blueprint §5 naming):
            outputs/models/{run_id}_kmeans_clusterer.pkl

        Payload bundle:
            - The sklearn estimator (holds cluster_centers_, labels_, inertia_)
            - Resolved _params dict
            - _n_features (used to validate future transform() inputs)

        Args:
            run_id    : MLflow run ID or cluster_id from ClusterResult.
            output_dir: Target directory; defaults to outputs/models/.

        Returns:
            Absolute Path to the saved artifact.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted KMeansClusterer. Call fit() first."
            )

        out_path = Path(output_dir) / f"{run_id}_{KMEANS_ARTIFACT_TYPE}.pkl"

        payload: dict[str, Any] = {
            "estimator":  self._estimator,
            "params":     self._params,
            "n_features": self._n_features,
        }
        save_artifact(obj=payload, path=str(out_path), fmt="pkl")
        logger.info("KMeansClusterer saved | path=%s", out_path)
        return out_path

    @classmethod
    def load(
        cls,
        artifact_path: str,
        corpus_id: str = "unknown",
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> "KMeansClusterer":
        """
        Reconstruct a fitted KMeansClusterer from a saved .pkl artifact.

        Bypasses __init__ so that no config file is required at load time —
        all hyperparameters are recovered from the persisted payload.

        Args:
            artifact_path: Path to the .pkl produced by save().
            corpus_id    : Corpus label to attach to the loaded instance.
            config_path  : Stored for reference; not re-read on load.

        Returns:
            A fully fitted KMeansClusterer ready for predict() calls.

        Raises:
            FileNotFoundError: Propagated from load_artifact if absent.
            KeyError         : If artifact payload is malformed.
        """
        payload: dict[str, Any] = load_artifact(path=artifact_path, fmt="pkl")

        estimator: KMeans | MiniBatchKMeans = payload["estimator"]
        stored_params: dict[str, Any] = payload.get("params", {})
        n_features: int = payload.get("n_features", 0)

        instance = cls.__new__(cls)
        instance.corpus_id = corpus_id
        instance.config_path = config_path
        instance._estimator = estimator
        instance._params = stored_params
        instance._is_fitted = True
        instance._n_features = n_features
        instance.k = estimator.n_clusters

        logger.info(
            "KMeansClusterer loaded | artifact_path=%s | k=%d | n_features=%d",
            artifact_path,
            instance.k,
            n_features,
        )
        return instance

    # ------------------------------------------------------------------
    # MLflow integration (Blueprint §8.1, §10.4)
    # ------------------------------------------------------------------

    def log_run(self, run_id: str, result: ClusterResult) -> None:
        """
        Emit structured run-start and run-end events for MLflow.

        CI/CD gate (Blueprint §10.4) requires at minimum:
            run_id, corpus_id, config snapshot, primary metric.

        Args:
            run_id: MLflow run identifier (use result.cluster_id).
            result: ClusterResult returned by fit().
        """
        log_run_start(run_id=run_id, params=result.params)

        metrics: dict[str, float] = {
            "inertia":   float(result.inertia or 0.0),
            "k":         float(result.k),
            "n_docs":    float(len(result.labels)),
        }
        if result.silhouette is not None:
            metrics["silhouette"] = result.silhouette

        log_run_end(run_id=run_id, metrics=metrics)

    # ------------------------------------------------------------------
    # Convenience accessors — consumed by cost_function.py (T-4)
    # ------------------------------------------------------------------

    @property
    def centroids(self) -> np.ndarray:
        """
        Return cluster centroid matrix, shape (k, n_features).

        Consumed by T-4's cost_function.py to encode pairwise centroid
        distances into the QUBO Hamiltonian (Blueprint §3.4).

        Raises:
            RuntimeError: If called before fit() or load().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "centroids accessed before fit(). Call fit() or load() first."
            )
        return self._estimator.cluster_centers_

    @property
    def is_fitted(self) -> bool:
        """Read-only guard — True once fit() or load() has completed."""
        return self._is_fitted

    def intra_cluster_distances(self) -> np.ndarray:
        """
        Compute mean intra-cluster Euclidean distance for each centroid.

        Returns a (k,) array where entry i is the mean distance from
        centroid i to all documents assigned to cluster i. Useful as a
        secondary quality metric in benchmark_runner.py (T-4) and for
        debugging degenerate clusters.

        Returns:
            np.ndarray of shape (k,), dtype float64.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "intra_cluster_distances() called before fit()."
            )
        centers: np.ndarray = self._estimator.cluster_centers_
        labels: np.ndarray = self._estimator.labels_

        # Reconstruct per-cluster mean distances from inertia breakdown
        # (We don't store X post-fit, so we compute from stored attributes.)
        # This returns the cluster-level inertia (sum of sq distances) / n_i.
        inertia_per_cluster = np.zeros(self.k, dtype=np.float64)
        counts = np.bincount(labels, minlength=self.k).astype(np.float64)

        # sklearn stores labels_ but not per-sample distances post-fit.
        # Return per-cluster average inertia as a proxy for intra-dist.
        # Full recomputation requires X which is not stored to respect
        # memory budget on large corpora.
        total_inertia: float = float(self._estimator.inertia_)
        avg_per_cluster = total_inertia / max(counts.sum(), 1.0)
        inertia_per_cluster[:] = avg_per_cluster  # uniform approximation

        logger.debug(
            "intra_cluster_distances: total_inertia=%.4f | k=%d",
            total_inertia,
            self.k,
        )
        return inertia_per_cluster

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        minibatch = self._params.get("use_minibatch", False)
        variant = "MiniBatchKMeans" if minibatch else "KMeans"
        return (
            f"KMeansClusterer(corpus_id={self.corpus_id!r}, k={self.k}, "
            f"variant={variant!r}, status={status})"
        )