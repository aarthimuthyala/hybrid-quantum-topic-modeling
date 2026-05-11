"""
src/classical/lda_model.py
===========================
Team T-2 — Classical NLP Layer (L2)
HQC Topic Modeling Project · Master Blueprint v1.0

Responsibility:
    Train a Latent Dirichlet Allocation (LDA) topic model on a DTMatrix
    produced by tfidf_vectorizer.py, and expose a standard interface of
    fit / transform / save / load that is identical in signature to
    nmf_model.py so that hybrid_pipeline.py (T-4) can swap models
    transparently.

Blueprint contracts honoured:
    - Input  : DTMatrix, n_topics: int              (§3.2, §5.2)
    - Output : TopicModelResult                     (§5.2)
    - Artifact persistence via shared.serializer    (§8.3)
    - Structured logging via shared.logger          (§8.1)
    - Config from config/classical_config.yaml      (§9.2)
    - Full package-path imports                     (§2)
    - Naming conventions                            (§6)
    - MLflow artifact key: {model_id}_{type}.pkl    (§5.1)

TopicModelResult schema (frozen — Blueprint §10.3):
    {
        "model_id":         str
        "topics":           List[TopicVector]        # top-N terms per topic
        "doc_topic_matrix": np.ndarray               # shape (n_docs, n_topics)
        "params":           dict                     # config + MLflow snapshot
    }

TopicVector schema:
    {
        "topic_id": int
        "terms":    List[str]   # ordered highest-weight first
        "weights":  List[float]
    }

Dependencies (requirements.txt):
    scikit-learn>=1.4       # LatentDirichletAllocation
    numpy>=1.26
    scipy>=1.12
    pyyaml>=6.0
    mlflow>=2.12
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation

# ---------------------------------------------------------------------------
# Project-internal imports — full package paths only (Blueprint §2)
# ---------------------------------------------------------------------------
from shared.logger import get_logger, log_run_end, log_run_start
from shared.serializer import load_artifact, save_artifact
from shared.validator import validate_config

# Re-use DTMatrix defined in tfidf_vectorizer to keep the contract DRY.
# T-4's hybrid_pipeline.py imports DTMatrix from this canonical location.
from src.classical.tfidf_vectorizer import DTMatrix  # noqa: E402

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: str = "config/classical_config.yaml"
DEFAULT_N_TOPICS: int = 20
DEFAULT_MAX_ITER: int = 100
DEFAULT_LEARNING_METHOD: str = "online"   # 'online' | 'batch'
DEFAULT_LEARNING_OFFSET: float = 10.0
DEFAULT_RANDOM_STATE: int = 42            # sourced from base_config.yaml seed
DEFAULT_N_JOBS: int = -1                  # use all available cores
DEFAULT_TOP_N_TERMS: int = 15             # terms surfaced per topic in result
LDA_ARTIFACT_TYPE: str = "lda_model"
DOC_TOPIC_ARTIFACT_TYPE: str = "doc_topic_matrix"


# ---------------------------------------------------------------------------
# Data-type contracts (Blueprint §5.2)
# ---------------------------------------------------------------------------

@dataclass
class TopicVector:
    """
    Represents a single discovered topic as an ordered list of terms.

    Fields:
        topic_id : Zero-based index within the model's topic set.
        terms    : Top-N vocabulary terms, sorted descending by weight.
        weights  : Corresponding per-term weights (normalised φ row).
    """
    topic_id: int
    terms: list[str]
    weights: list[float]


@dataclass
class TopicModelResult:
    """
    Unified output contract for lda_model.py and nmf_model.py.
    Consumed by topic_coherence.py, benchmark_runner.py, and the
    /classical/model/{model_id} API endpoint (Blueprint §4.2, §5.2).

    FROZEN — changes require an ADR (Blueprint §10.3).

    Fields:
        model_id        : Globally unique model identifier.
        topics          : List of TopicVector objects (one per topic).
        doc_topic_matrix: Dense array mapping each doc to topic weights.
                          Shape: (n_docs, n_topics).
        params          : Full config snapshot for MLflow reproducibility.
    """
    model_id: str
    topics: list[TopicVector]
    doc_topic_matrix: np.ndarray        # shape (n_docs, n_topics)
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def _load_lda_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load config/classical_config.yaml and extract the LDA subsection.

    Merges YAML values with module-level defaults so partial configs are safe.
    validate_config() raises ConfigError on schema violation (Blueprint §8.2).

    Args:
        config_path: Filesystem path to classical_config.yaml.

    Returns:
        Resolved LDA hyperparameter dict.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Config not found at '%s'. Falling back to built-in defaults.",
            config_path,
        )
        return {}

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    validate_config(raw, schema_name="classical_config")

    lda_cfg: dict[str, Any] = raw.get("classical", {}).get("lda", {})
    return lda_cfg


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class LDAModel:
    """
    Latent Dirichlet Allocation topic model with a Blueprint-compliant
    fit / transform / save / load interface.

    Interface is intentionally identical to NMFModel so that
    hybrid_pipeline.py (T-4) can substitute one for the other without
    code changes.

    Usage (canonical pipeline — Blueprint §5.1, Stage 04):
        >>> from src.classical.lda_model import LDAModel
        >>> from src.classical.tfidf_vectorizer import TFIDFVectorizer
        >>>
        >>> vectorizer = TFIDFVectorizer(corpus_id="20ng")
        >>> dt_matrix = vectorizer.fit_transform(clean_docs)
        >>>
        >>> model = LDAModel(corpus_id="20ng", n_topics=20)
        >>> result = model.fit(dt_matrix)
        >>> model.save(run_id=result.model_id)

    MLflow run naming convention (Blueprint §6):
        lda_{corpus_id}_{timestamp}   e.g. lda_20ng_20250510

    Attributes:
        corpus_id    : Source corpus identifier.
        n_topics     : Number of latent topics to discover.
        config_path  : Path to classical_config.yaml.
        _lda         : The underlying sklearn LatentDirichletAllocation.
        _is_fitted   : Guard flag for premature method calls.
        _params      : Resolved hyperparameter snapshot.
        _feature_names: Vocabulary list from the DTMatrix used in fit().
    """

    def __init__(
        self,
        corpus_id: str,
        n_topics: int | None = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        """
        Initialise LDAModel and resolve hyperparameters from config.

        Args:
            corpus_id  : Corpus identifier (propagated to TopicModelResult).
            n_topics   : Override for the number of topics. If None, the
                         value in classical_config.yaml is used.
            config_path: Path to classical_config.yaml (Blueprint §9.2).
        """
        self.corpus_id: str = corpus_id
        self.config_path: str = config_path
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

        # Resolve config
        cfg = _load_lda_config(config_path)

        # n_topics: constructor arg > YAML > default constant
        resolved_n_topics = (
            n_topics
            if n_topics is not None
            else cfg.get("n_topics", DEFAULT_N_TOPICS)
        )

        self._params: dict[str, Any] = {
            "n_topics":        resolved_n_topics,
            "max_iter":        cfg.get("max_iter",         DEFAULT_MAX_ITER),
            "learning_method": cfg.get("learning_method",  DEFAULT_LEARNING_METHOD),
            "learning_offset": cfg.get("learning_offset",  DEFAULT_LEARNING_OFFSET),
            "doc_topic_prior": cfg.get("alpha",            None),  # 'auto' → None
            "topic_word_prior":cfg.get("eta",              None),  # 'auto' → None
            "random_state":    cfg.get("random_state",     DEFAULT_RANDOM_STATE),
            "n_jobs":          cfg.get("n_jobs",           DEFAULT_N_JOBS),
            "top_n_terms":     cfg.get("top_n_terms",      DEFAULT_TOP_N_TERMS),
        }

        self.n_topics: int = resolved_n_topics

        logger.info(
            "LDAModel initialised | corpus_id=%s | n_topics=%d | params=%s",
            corpus_id,
            self.n_topics,
            self._params,
        )

        # Instantiate sklearn estimator
        self._lda: LatentDirichletAllocation = LatentDirichletAllocation(
            n_components=self._params["n_topics"],
            max_iter=self._params["max_iter"],
            learning_method=self._params["learning_method"],
            learning_offset=self._params["learning_offset"],
            doc_topic_prior=self._params["doc_topic_prior"],
            topic_word_prior=self._params["topic_word_prior"],
            random_state=self._params["random_state"],
            n_jobs=self._params["n_jobs"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_topics(self) -> list[TopicVector]:
        """
        Convert the fitted LDA components_ matrix to TopicVector objects.

        Each row of self._lda.components_ is a topic's word-weight
        distribution over the vocabulary (un-normalised counts).
        We normalise per-row and extract the top-N terms.

        Returns:
            List of TopicVector, one per topic, weights in descending order.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("_extract_topics() called before fit().")

        top_n: int = self._params["top_n_terms"]
        vocab: list[str] = self._feature_names
        topics: list[TopicVector] = []

        # components_ shape: (n_topics, n_vocab)
        # Normalise each row to a probability distribution
        components = self._lda.components_
        row_sums = components.sum(axis=1, keepdims=True)
        # Avoid division by zero on a degenerate row
        normalised = np.divide(
            components,
            row_sums,
            out=np.zeros_like(components),
            where=row_sums != 0,
        )

        for topic_idx in range(self.n_topics):
            row = normalised[topic_idx]
            # argsort descending — take top_n indices
            top_indices: np.ndarray = np.argsort(row)[::-1][:top_n]
            terms: list[str] = [vocab[i] for i in top_indices]
            weights: list[float] = [float(row[i]) for i in top_indices]

            topics.append(
                TopicVector(topic_id=topic_idx, terms=terms, weights=weights)
            )

        return topics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dt_matrix: DTMatrix) -> TopicModelResult:
        """
        Train LDA on the provided document-term matrix.

        Persists nothing to disk — call save() explicitly to write the
        model artifact (Blueprint §5.1 Stage 04).

        Args:
            dt_matrix: DTMatrix from tfidf_vectorizer.fit_transform().

        Returns:
            TopicModelResult with topics and doc–topic assignments.

        Raises:
            ValueError : If dt_matrix.matrix is empty.
            RuntimeError: On sklearn convergence errors (propagated).
        """
        if dt_matrix.matrix.shape[0] == 0:
            raise ValueError("DTMatrix contains zero documents; cannot fit LDA.")

        n_docs, vocab_size = dt_matrix.matrix.shape
        self._feature_names = dt_matrix.feature_names

        logger.info(
            "LDA fit started | corpus_id=%s | n_docs=%d | vocab_size=%d | n_topics=%d",
            self.corpus_id,
            n_docs,
            vocab_size,
            self.n_topics,
        )

        t0 = time.perf_counter()
        self._lda.fit(dt_matrix.matrix)
        self._is_fitted = True
        elapsed = time.perf_counter() - t0

        # sklearn 1.x stores per-iteration bound in bound_
        perplexity: float = float(
            self._lda.perplexity(dt_matrix.matrix)
        )
        logger.info(
            "LDA fit complete | elapsed_s=%.3f | perplexity=%.4f | "
            "n_iter=%d",
            elapsed,
            perplexity,
            self._lda.n_iter_,
        )

        # Build doc–topic assignment matrix
        doc_topic_matrix: np.ndarray = self._lda.transform(dt_matrix.matrix)

        # Extract human-readable topic representations
        topics: list[TopicVector] = self._extract_topics()

        # Generate a unique model_id following §6 artifact naming
        model_id = f"lda_{self.corpus_id}_{int(time.time())}"

        # Assemble full params snapshot for MLflow
        result_params: dict[str, Any] = {
            **self._params,
            "corpus_id":  self.corpus_id,
            "n_docs":     n_docs,
            "vocab_size": vocab_size,
            "perplexity": perplexity,
            "n_iter_":    self._lda.n_iter_,
            "model":      "lda",
        }

        return TopicModelResult(
            model_id=model_id,
            topics=topics,
            doc_topic_matrix=doc_topic_matrix,
            params=result_params,
        )

    def transform(self, dt_matrix: DTMatrix) -> np.ndarray:
        """
        Infer topic distributions for new documents (without re-fitting).

        Useful for held-out evaluation sets or API inference calls
        routed through /classical/model/{model_id} (Blueprint §4.2).

        Args:
            dt_matrix: DTMatrix of unseen documents; must use the same
                       vocabulary as the training DTMatrix.

        Returns:
            np.ndarray of shape (n_docs, n_topics) — row-normalised
            document–topic probability distributions.

        Raises:
            RuntimeError: If called before fit().
            ValueError  : If dt_matrix is empty.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "LDAModel.transform() called before fit(). "
                "Train the model first or load a saved artifact."
            )
        if dt_matrix.matrix.shape[0] == 0:
            raise ValueError("DTMatrix is empty; cannot transform.")

        logger.info(
            "LDA transform | n_docs=%d", dt_matrix.matrix.shape[0]
        )
        return self._lda.transform(dt_matrix.matrix)

    def fit_transform(self, dt_matrix: DTMatrix) -> TopicModelResult:
        """
        Convenience method: fit the model and return a TopicModelResult
        in a single call.

        Mirrors the sklearn API pattern used by tfidf_vectorizer.py so
        that pipeline code stays uniform across L2 modules.

        Args:
            dt_matrix: DTMatrix from the ingestion/vectorisation step.

        Returns:
            TopicModelResult (same as fit() return value).
        """
        return self.fit(dt_matrix)

    # ------------------------------------------------------------------
    # Persistence (Blueprint §8.3)
    # ------------------------------------------------------------------

    def save(self, run_id: str, output_dir: str = "outputs/models") -> Path:
        """
        Persist the fitted sklearn LDA estimator via shared.serializer.

        Artifact path (Blueprint §5 naming):
            outputs/models/{run_id}_lda_model.pkl

        Args:
            run_id    : MLflow run ID or model_id from TopicModelResult.
            output_dir: Destination directory.

        Returns:
            Path to the saved .pkl file.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted LDAModel.")

        out_path = Path(output_dir) / f"{run_id}_{LDA_ARTIFACT_TYPE}.pkl"

        # save_artifact logs the path to MLflow as a side effect (Blueprint §8.3)
        save_artifact(
            obj={"lda": self._lda, "feature_names": self._feature_names},
            path=str(out_path),
            fmt="pkl",
        )
        logger.info("LDAModel saved | path=%s", out_path)
        return out_path

    @classmethod
    def load(
        cls,
        artifact_path: str,
        corpus_id: str = "unknown",
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> "LDAModel":
        """
        Reconstruct an LDAModel from a previously saved artifact.

        Args:
            artifact_path: Path to the .pkl file produced by save().
            corpus_id    : Corpus label to attach to the loaded instance.
            config_path  : Classical config path (for metadata only).

        Returns:
            A fitted LDAModel ready for transform() calls.
        """
        payload: dict = load_artifact(path=artifact_path, fmt="pkl")
        sklearn_lda: LatentDirichletAllocation = payload["lda"]
        feature_names: list[str] = payload["feature_names"]

        # Bypass __init__ to avoid re-reading config
        instance = cls.__new__(cls)
        instance.corpus_id = corpus_id
        instance.config_path = config_path
        instance._lda = sklearn_lda
        instance._feature_names = feature_names
        instance._is_fitted = True
        instance.n_topics = sklearn_lda.n_components
        instance._params = {
            "n_topics":        sklearn_lda.n_components,
            "max_iter":        sklearn_lda.max_iter,
            "learning_method": sklearn_lda.learning_method,
            "random_state":    sklearn_lda.random_state,
            "top_n_terms":     DEFAULT_TOP_N_TERMS,
        }

        logger.info(
            "LDAModel loaded | artifact_path=%s | n_topics=%d | vocab_size=%d",
            artifact_path,
            instance.n_topics,
            len(feature_names),
        )
        return instance

    # ------------------------------------------------------------------
    # MLflow integration helpers (Blueprint §8.1)
    # ------------------------------------------------------------------

    def log_run(self, run_id: str, result: TopicModelResult) -> None:
        """
        Emit structured run-start and run-end events to MLflow.

        Designed to be called immediately after fit() when the
        TopicModelResult is available.

        Args:
            run_id : MLflow run identifier.
            result : TopicModelResult returned by fit().
        """
        log_run_start(run_id=run_id, params=result.params)

        # Extract the primary metric(s) for MLflow (Blueprint §10.4)
        metrics: dict[str, float] = {
            "perplexity": float(result.params.get("perplexity", 0.0)),
            "n_topics":   float(self.n_topics),
            "n_docs":     float(result.params.get("n_docs", 0)),
        }
        log_run_end(run_id=run_id, metrics=metrics)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def topic_word_matrix(self) -> np.ndarray:
        """
        Return the raw (un-normalised) topic × word weight matrix.

        Shape: (n_topics, vocab_size). Useful for downstream cost_function.py
        when building the QUBO Hamiltonian from topic-word distributions.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("topic_word_matrix accessed before fit().")
        return self._lda.components_

    @property
    def is_fitted(self) -> bool:
        """Read-only flag indicating whether the model has been trained."""
        return self._is_fitted

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"LDAModel(corpus_id={self.corpus_id!r}, n_topics={self.n_topics}, "
            f"status={status})"
        )