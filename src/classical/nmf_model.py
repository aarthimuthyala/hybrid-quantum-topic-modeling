"""
src/classical/nmf_model.py
===========================
Team T-2 — Classical NLP Layer (L2)
HQC Topic Modeling Project · Master Blueprint v1.0

Responsibility:
    Train a Non-negative Matrix Factorization (NMF) topic model on a
    DTMatrix produced by tfidf_vectorizer.py, and expose a standard
    interface of fit / transform / save / load that is *interface-identical*
    to lda_model.py so that hybrid_pipeline.py (T-4) can substitute one
    model for the other without any call-site changes.

Blueprint contracts honoured:
    - Input  : DTMatrix, n_topics: int              (§3.2, §5.2)
    - Output : TopicModelResult                     (§5.2)
    - Interface parity with lda_model.py            (§3.2 note)
    - Artifact persistence via shared.serializer    (§8.3)
    - Structured logging via shared.logger          (§8.1)
    - Config from config/classical_config.yaml      (§9.2)
    - Full package-path imports; no relative imports (§2)
    - Naming conventions enforced by ruff            (§6)

Data contracts re-used from lda_model (FROZEN — Blueprint §10.3):
    - TopicVector        : { topic_id, terms, weights }
    - TopicModelResult   : { model_id, topics, doc_topic_matrix, params }

NMF-specific notes:
    - Input DTMatrix must be *non-negative* — TF-IDF values satisfy this
      by construction (sklearn TfidfVectorizer always outputs ≥ 0).
    - NMF decomposes X ≈ W · H where:
          W  shape (n_docs,   n_topics)   → doc-topic matrix
          H  shape (n_topics, n_vocab)    → topic-word matrix (components_)
    - The 'cd' solver (coordinate descent) is the Blueprint §9.2 default
      and is more stable than 'mu' (multiplicative update) for sparse
      TF-IDF matrices.

Dependencies (requirements.txt):
    scikit-learn>=1.4       # NMF
    numpy>=1.26
    scipy>=1.12
    pyyaml>=6.0
    mlflow>=2.12
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.decomposition import NMF

# ---------------------------------------------------------------------------
# Project-internal imports — full package paths only (Blueprint §2)
# ---------------------------------------------------------------------------
from shared.logger import get_logger, log_run_end, log_run_start
from shared.serializer import load_artifact, save_artifact
from shared.validator import validate_config

# Import shared data-contract types from lda_model — single source of truth.
# NMFModel must produce identical TopicModelResult / TopicVector objects so
# that topic_coherence.py and benchmark_runner.py (T-4) treat both models
# interchangeably (Blueprint §3.2, §10.3).
from src.classical.lda_model import TopicModelResult, TopicVector

# DTMatrix lives in tfidf_vectorizer — the L2 canonical definition.
from src.classical.tfidf_vectorizer import DTMatrix

# ---------------------------------------------------------------------------
# Module logger (Blueprint §8.1)
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — UPPER_SNAKE_CASE (Blueprint §6)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: str = "config/classical_config.yaml"
DEFAULT_N_TOPICS: int = 20
DEFAULT_MAX_ITER: int = 200           # Blueprint §9.2 nmf.max_iter default
DEFAULT_SOLVER: str = "cd"            # Blueprint §9.2 nmf.solver default
DEFAULT_BETA_LOSS: str = "frobenius"  # Frobenius norm — standard for TF-IDF
DEFAULT_ALPHA_W: float = 0.0          # L1/L2 regularisation weight on W
DEFAULT_ALPHA_H: float = 0.0          # L1/L2 regularisation weight on H
DEFAULT_L1_RATIO: float = 0.0         # 0 → L2-only, 1 → L1-only
DEFAULT_RANDOM_STATE: int = 42        # matches base_config.yaml seed (§9.1)
DEFAULT_TOP_N_TERMS: int = 15         # per-topic terms surfaced in result
DEFAULT_INIT: str = "nndsvda"         # best deterministic initialisation
NMF_ARTIFACT_TYPE: str = "nmf_model"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_nmf_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load config/classical_config.yaml and extract the NMF subsection.

    Falls back gracefully to module-level DEFAULT_* constants when the
    config file is absent or the 'classical.nmf' key is missing.
    validate_config() (Blueprint §8.2) raises ConfigError on schema errors.

    Args:
        config_path: Path to classical_config.yaml.

    Returns:
        Dict with resolved NMF hyperparameters.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Config file not found at '%s'. Using built-in NMF defaults.",
            config_path,
        )
        return {}

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    validate_config(raw, schema_name="classical_config")

    nmf_cfg: dict[str, Any] = raw.get("classical", {}).get("nmf", {})
    return nmf_cfg


def _extract_nmf_topics(
    components: np.ndarray,
    feature_names: list[str],
    top_n: int,
) -> list[TopicVector]:
    """
    Convert the NMF H-matrix (components_) to a list of TopicVector objects.

    Each row of `components` is a topic's un-normalised word-weight vector.
    We L1-normalise per row so that weights sum to 1 (a pseudo-probability
    distribution) and then return the top-N highest-weight terms.

    This helper is intentionally module-private (leading underscore) and
    reused by both fit() and load() so the extraction logic is not duplicated.

    Args:
        components   : NMF H-matrix, shape (n_topics, n_vocab).
        feature_names: Ordered vocabulary matching matrix columns.
        top_n        : Number of top terms to include per topic.

    Returns:
        List of TopicVector, one per topic, in ascending topic_id order.
    """
    topics: list[TopicVector] = []
    n_topics, n_vocab = components.shape

    for topic_idx in range(n_topics):
        row = components[topic_idx]

        # L1 normalise — convert raw NMF weights to a probability-like dist
        row_sum = row.sum()
        normalised = row / row_sum if row_sum > 0 else row

        # Select top-N indices sorted by descending weight
        top_indices: np.ndarray = np.argsort(normalised)[::-1][:top_n]
        terms: list[str] = [feature_names[i] for i in top_indices]
        weights: list[float] = [float(normalised[i]) for i in top_indices]

        topics.append(TopicVector(topic_id=topic_idx, terms=terms, weights=weights))

    return topics


def _validate_dt_matrix(dt_matrix: DTMatrix, caller: str) -> None:
    """
    Guard against empty or structurally invalid DTMatrix inputs.

    Centralises validation logic shared by fit() and transform() so error
    messages are consistent across both call sites.

    Args:
        dt_matrix: DTMatrix to validate.
        caller   : Name of the calling method (for error messages).

    Raises:
        ValueError: If the matrix has zero documents or zero features.
    """
    n_docs, n_vocab = dt_matrix.matrix.shape
    if n_docs == 0:
        raise ValueError(
            f"NMFModel.{caller}() received an empty DTMatrix (0 documents)."
        )
    if n_vocab == 0:
        raise ValueError(
            f"NMFModel.{caller}() received a DTMatrix with 0 vocabulary features."
        )


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class NMFModel:
    """
    Non-negative Matrix Factorization topic model with an interface that is
    *byte-for-byte compatible* with LDAModel so that T-4's hybrid_pipeline.py
    can swap models via a single config flag.

    Produces:
        - TopicModelResult  (identical schema to LDAModel output)
        - Artifact .pkl     (same naming convention: {run_id}_nmf_model.pkl)

    Usage (canonical pipeline — Blueprint §5.1, Stage 04):
        >>> from src.classical.nmf_model import NMFModel
        >>> from src.classical.tfidf_vectorizer import TFIDFVectorizer
        >>>
        >>> vectorizer = TFIDFVectorizer(corpus_id="20ng")
        >>> dt_matrix  = vectorizer.fit_transform(clean_docs)
        >>>
        >>> model  = NMFModel(corpus_id="20ng", n_topics=20)
        >>> result = model.fit(dt_matrix)
        >>> model.save(run_id=result.model_id)
        >>> model.log_run(run_id=result.model_id, result=result)

    MLflow run naming convention (Blueprint §6):
        nmf_{corpus_id}_{unix_timestamp}   e.g. nmf_20ng_20250510123456

    Attributes:
        corpus_id      : Source corpus identifier, propagated to artifacts.
        n_topics       : Number of latent topics (NMF rank k).
        config_path    : Path to classical_config.yaml.
        _nmf           : Underlying sklearn NMF estimator.
        _is_fitted     : Guard flag — prevents premature transform/save calls.
        _params        : Resolved hyperparameter snapshot (logged to MLflow).
        _feature_names : Vocabulary list from the DTMatrix used during fit().
    """

    def __init__(
        self,
        corpus_id: str,
        n_topics: int | None = None,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        """
        Initialise NMFModel and resolve hyperparameters from config.

        Args:
            corpus_id  : Corpus identifier (propagated to TopicModelResult).
            n_topics   : Override for the number of topics (rank k). If None,
                         the value in classical_config.yaml is used.
            config_path: Path to classical_config.yaml (Blueprint §9.2).
        """
        self.corpus_id: str = corpus_id
        self.config_path: str = config_path
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

        # Resolve config — YAML values take priority over module constants
        cfg = _load_nmf_config(config_path)

        resolved_n_topics = (
            n_topics
            if n_topics is not None
            else cfg.get("n_topics", DEFAULT_N_TOPICS)
        )

        self._params: dict[str, Any] = {
            "n_topics":     resolved_n_topics,
            "max_iter":     cfg.get("max_iter",     DEFAULT_MAX_ITER),
            "solver":       cfg.get("solver",       DEFAULT_SOLVER),
            "beta_loss":    cfg.get("beta_loss",    DEFAULT_BETA_LOSS),
            "alpha_W":      cfg.get("alpha_W",      DEFAULT_ALPHA_W),
            "alpha_H":      cfg.get("alpha_H",      DEFAULT_ALPHA_H),
            "l1_ratio":     cfg.get("l1_ratio",     DEFAULT_L1_RATIO),
            "init":         cfg.get("init",         DEFAULT_INIT),
            "random_state": cfg.get("random_state", DEFAULT_RANDOM_STATE),
            "top_n_terms":  cfg.get("top_n_terms",  DEFAULT_TOP_N_TERMS),
        }

        self.n_topics: int = resolved_n_topics

        logger.info(
            "NMFModel initialised | corpus_id=%s | n_topics=%d | params=%s",
            corpus_id,
            self.n_topics,
            self._params,
        )

        # Instantiate sklearn estimator from resolved params
        self._nmf: NMF = NMF(
            n_components=self._params["n_topics"],
            max_iter=self._params["max_iter"],
            solver=self._params["solver"],
            beta_loss=self._params["beta_loss"],
            alpha_W=self._params["alpha_W"],
            alpha_H=self._params["alpha_H"],
            l1_ratio=self._params["l1_ratio"],
            init=self._params["init"],
            random_state=self._params["random_state"],
        )

    # ------------------------------------------------------------------
    # Public API — fit / transform / fit_transform
    # (interface-identical to LDAModel — Blueprint §3.2)
    # ------------------------------------------------------------------

    def fit(self, dt_matrix: DTMatrix) -> TopicModelResult:
        """
        Factorise the document-term matrix and extract topic representations.

        Learns W (doc-topic) and H (topic-word) matrices from dt_matrix.matrix
        using sklearn NMF. Stores H internally as self._nmf.components_ and
        the vocabulary as self._feature_names so that transform() can later
        project new documents into the same topic space.

        Does NOT persist to disk — call save() explicitly (Blueprint §5.1).

        Args:
            dt_matrix: DTMatrix from tfidf_vectorizer.fit_transform().
                       Must have non-negative values (guaranteed by TF-IDF).

        Returns:
            TopicModelResult containing:
                - model_id        : "nmf_{corpus_id}_{timestamp}"
                - topics          : List[TopicVector] (top-N terms per topic)
                - doc_topic_matrix: W matrix, shape (n_docs, n_topics)
                - params          : Full config + runtime snapshot

        Raises:
            ValueError  : If dt_matrix is empty.
            RuntimeError: On sklearn convergence failure (propagated).
        """
        _validate_dt_matrix(dt_matrix, caller="fit")
        n_docs, vocab_size = dt_matrix.matrix.shape
        self._feature_names = dt_matrix.feature_names

        logger.info(
            "NMF fit started | corpus_id=%s | n_docs=%d | vocab_size=%d | "
            "n_topics=%d | solver=%s",
            self.corpus_id,
            n_docs,
            vocab_size,
            self.n_topics,
            self._params["solver"],
        )

        t0 = time.perf_counter()

        # fit_transform is more efficient than fit + transform for NMF
        # because W is a by-product of factorisation; we capture it here
        # and re-expose it via transform() for new documents later.
        doc_topic_matrix: np.ndarray = self._nmf.fit_transform(dt_matrix.matrix)
        self._is_fitted = True

        elapsed = time.perf_counter() - t0

        # Reconstruction error — NMF analogue of LDA perplexity
        reconstruction_err: float = float(self._nmf.reconstruction_err_)
        n_iter: int = self._nmf.n_iter_

        logger.info(
            "NMF fit complete | elapsed_s=%.3f | reconstruction_err=%.6f | "
            "n_iter=%d",
            elapsed,
            reconstruction_err,
            n_iter,
        )

        # Build human-readable topic descriptors from H matrix
        topics: list[TopicVector] = _extract_nmf_topics(
            components=self._nmf.components_,
            feature_names=self._feature_names,
            top_n=self._params["top_n_terms"],
        )

        model_id = f"nmf_{self.corpus_id}_{int(time.time())}"

        result_params: dict[str, Any] = {
            **self._params,
            "corpus_id":          self.corpus_id,
            "n_docs":             n_docs,
            "vocab_size":         vocab_size,
            "reconstruction_err": reconstruction_err,
            "n_iter_":            n_iter,
            "model":              "nmf",
        }

        return TopicModelResult(
            model_id=model_id,
            topics=topics,
            doc_topic_matrix=doc_topic_matrix,
            params=result_params,
        )

    def transform(self, dt_matrix: DTMatrix) -> np.ndarray:
        """
        Project new documents into the fitted topic space.

        Uses the frozen H matrix (self._nmf.components_) learned during fit()
        to infer per-document topic weights for unseen documents. Suitable for
        held-out evaluation sets and live API inference (Blueprint §4.2).

        Args:
            dt_matrix: DTMatrix of unseen documents. Must use the same
                       vocabulary as the training DTMatrix (same TFIDFVectorizer).

        Returns:
            np.ndarray of shape (n_docs, n_topics) — W matrix for new docs.

        Raises:
            RuntimeError: If called before fit() or load().
            ValueError  : If dt_matrix is empty.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "NMFModel.transform() called before fit(). "
                "Train the model or call NMFModel.load() first."
            )
        _validate_dt_matrix(dt_matrix, caller="transform")

        n_docs = dt_matrix.matrix.shape[0]
        logger.info("NMF transform | n_docs=%d", n_docs)

        return self._nmf.transform(dt_matrix.matrix)

    def fit_transform(self, dt_matrix: DTMatrix) -> TopicModelResult:
        """
        Convenience wrapper: fit on dt_matrix and return TopicModelResult.

        Mirrors the API of TFIDFVectorizer.fit_transform() and LDAModel
        so that pipeline code is uniform across all L2 modules.

        Args:
            dt_matrix: DTMatrix from the vectorisation step.

        Returns:
            TopicModelResult (identical to fit() return value).
        """
        return self.fit(dt_matrix)

    # ------------------------------------------------------------------
    # Persistence — save / load (Blueprint §8.3)
    # ------------------------------------------------------------------

    def save(self, run_id: str, output_dir: str = "outputs/models") -> Path:
        """
        Persist the fitted sklearn NMF estimator via shared.serializer.

        Artifact path follows the Blueprint §5 naming convention:
            outputs/models/{run_id}_nmf_model.pkl

        The payload bundle includes:
            - The sklearn NMF object (carries components_, n_iter_, etc.)
            - feature_names list (needed to reconstruct TopicVector on load)

        save_artifact() (Blueprint §8.3) logs the path to MLflow as a
        side effect and records its SHA-256 hash in data/manifest.json.

        Args:
            run_id    : MLflow run ID or model_id from TopicModelResult.
            output_dir: Target directory; defaults to outputs/models/.

        Returns:
            Absolute Path to the saved .pkl artifact.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted NMFModel. Call fit() first."
            )

        out_path = Path(output_dir) / f"{run_id}_{NMF_ARTIFACT_TYPE}.pkl"

        payload: dict[str, Any] = {
            "nmf":           self._nmf,
            "feature_names": self._feature_names,
            "params":        self._params,
        }

        save_artifact(obj=payload, path=str(out_path), fmt="pkl")
        logger.info("NMFModel saved | path=%s", out_path)
        return out_path

    @classmethod
    def load(
        cls,
        artifact_path: str,
        corpus_id: str = "unknown",
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> "NMFModel":
        """
        Reconstruct a fitted NMFModel from a previously saved .pkl artifact.

        Bypasses __init__ so that no config file is required at load time —
        all hyperparameters are recovered from the persisted payload.

        Args:
            artifact_path: Filesystem path to the .pkl file from save().
            corpus_id    : Corpus label to attach to the instance.
            config_path  : Stored for reference; not re-read on load.

        Returns:
            A fully fitted NMFModel ready for transform() and
            topic_word_matrix calls.

        Raises:
            FileNotFoundError: Propagated from load_artifact if path absent.
            KeyError         : If the artifact payload is malformed.
        """
        payload: dict[str, Any] = load_artifact(path=artifact_path, fmt="pkl")

        sklearn_nmf: NMF = payload["nmf"]
        feature_names: list[str] = payload["feature_names"]
        stored_params: dict[str, Any] = payload.get("params", {})

        # Reconstruct instance without touching __init__
        instance = cls.__new__(cls)
        instance.corpus_id = corpus_id
        instance.config_path = config_path
        instance._nmf = sklearn_nmf
        instance._feature_names = feature_names
        instance._is_fitted = True
        instance.n_topics = sklearn_nmf.n_components
        instance._params = stored_params or {
            "n_topics":    sklearn_nmf.n_components,
            "max_iter":    sklearn_nmf.max_iter,
            "solver":      sklearn_nmf.solver,
            "top_n_terms": DEFAULT_TOP_N_TERMS,
        }

        logger.info(
            "NMFModel loaded | artifact_path=%s | n_topics=%d | vocab_size=%d",
            artifact_path,
            instance.n_topics,
            len(feature_names),
        )
        return instance

    # ------------------------------------------------------------------
    # MLflow integration (Blueprint §8.1, §10.4)
    # ------------------------------------------------------------------

    def log_run(self, run_id: str, result: TopicModelResult) -> None:
        """
        Emit structured run-start and run-end events compatible with the
        shared.logger contract (Blueprint §8.1).

        Must be called after fit() when TopicModelResult is available.
        CI/CD gate (Blueprint §10.4) requires at minimum:
            run_id, corpus_id, config snapshot, primary metric.

        Args:
            run_id: MLflow run identifier (use result.model_id).
            result: TopicModelResult returned by fit().
        """
        log_run_start(run_id=run_id, params=result.params)

        metrics: dict[str, float] = {
            "reconstruction_err": float(
                result.params.get("reconstruction_err", 0.0)
            ),
            "n_topics": float(self.n_topics),
            "n_docs":   float(result.params.get("n_docs", 0)),
            "n_iter":   float(result.params.get("n_iter_", 0)),
        }
        log_run_end(run_id=run_id, metrics=metrics)

    # ------------------------------------------------------------------
    # Convenience accessors — consumed by cost_function.py (T-4)
    # ------------------------------------------------------------------

    @property
    def topic_word_matrix(self) -> np.ndarray:
        """
        Return the NMF H-matrix: topic × word weight distribution.

        Shape: (n_topics, vocab_size). Consumed by T-4's cost_function.py
        to build the QUBO / Ising Hamiltonian from topic-word co-occurrence
        structure (Blueprint §3.4).

        Raises:
            RuntimeError: If called before fit() or load().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "topic_word_matrix accessed before fit(). "
                "Call fit() or NMFModel.load() first."
            )
        return self._nmf.components_

    @property
    def is_fitted(self) -> bool:
        """Read-only guard flag — True once fit() or load() has completed."""
        return self._is_fitted

    def get_topic_terms(
        self, topic_id: int, top_n: int | None = None
    ) -> list[tuple[str, float]]:
        """
        Return (term, weight) pairs for a single topic.

        Convenience accessor for inspection and API serialisation
        (/classical/model/{model_id} route — Blueprint §4.2).

        Args:
            topic_id: Zero-based topic index in [0, n_topics).
            top_n   : Number of terms to return; defaults to top_n_terms
                      stored in self._params.

        Returns:
            List of (term, weight) tuples, sorted by descending weight.

        Raises:
            RuntimeError : If called before fit().
            IndexError   : If topic_id is out of range.
        """
        if not self._is_fitted:
            raise RuntimeError("get_topic_terms() called before fit().")
        if not (0 <= topic_id < self.n_topics):
            raise IndexError(
                f"topic_id {topic_id} out of range [0, {self.n_topics})."
            )

        n = top_n or self._params.get("top_n_terms", DEFAULT_TOP_N_TERMS)
        row = self._nmf.components_[topic_id]
        row_sum = row.sum()
        normalised = row / row_sum if row_sum > 0 else row

        top_indices = np.argsort(normalised)[::-1][:n]
        return [
            (self._feature_names[i], float(normalised[i])) for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"NMFModel(corpus_id={self.corpus_id!r}, n_topics={self.n_topics}, "
            f"solver={self._params.get('solver')!r}, status={status})"
        )