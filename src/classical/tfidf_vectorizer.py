"""
src/classical/tfidf_vectorizer.py
==================================
Team T-2 — Classical NLP Layer (L2)
HQC Topic Modeling Project · Master Blueprint v1.0

Responsibility:
    Build a TF-IDF document-term matrix (DTMatrix) from a list of
    CleanDocument objects produced by the ingestion layer (T-1).
    Persist the fitted vectorizer to disk so downstream modules
    (lda_model.py, nmf_model.py, kmeans_clusterer.py, cost_function.py)
    can reuse the same vocabulary without retraining.

Blueprint contracts honoured:
    - Input  : List[CleanDocument]  (§5.2)
    - Output : DTMatrix             (§5.2, §3.2)
    - Artifact persistence via shared.serializer.save_artifact (§8.3)
    - Structured logging via shared.logger.get_logger           (§8.1)
    - All hyperparameters sourced from config/classical_config.yaml (§9.2)
    - Full package-path imports; no relative imports outside __init__.py (§2)
    - Naming: PascalCase classes, snake_case functions, UPPER_SNAKE constants (§6)

DTMatrix schema (frozen contract — §10.3):
    {
        "matrix":       scipy.sparse.csr_matrix   # shape (n_docs, vocab_size)
        "feature_names": List[str]                # ordered vocabulary terms
        "corpus_id":    str
        "doc_ids":      List[str]
        "params":       dict                      # config snapshot for MLflow
    }

Dependencies (requirements.txt):
    scikit-learn>=1.4
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
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Project-internal imports — always use full package paths (Blueprint §2)
# ---------------------------------------------------------------------------
from shared.logger import get_logger, log_run_end, log_run_start
from shared.serializer import load_artifact, save_artifact
from shared.validator import validate_config

# ---------------------------------------------------------------------------
# Module logger (Blueprint §8.1)
# ---------------------------------------------------------------------------
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — all uppercase (Blueprint §6)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: str = "config/classical_config.yaml"
DEFAULT_MAX_FEATURES: int = 50_000
DEFAULT_MIN_DF: int = 2
DEFAULT_MAX_DF: float = 0.95
DEFAULT_NGRAM_RANGE: tuple[int, int] = (1, 2)
DEFAULT_SUBLINEAR_TF: bool = True
VECTORIZER_ARTIFACT_TYPE: str = "tfidf_vectorizer"
DT_MATRIX_ARTIFACT_TYPE: str = "dt_matrix"


# ---------------------------------------------------------------------------
# Data-type contracts (Blueprint §5.2)
# ---------------------------------------------------------------------------

@dataclass
class CleanDocument:
    """
    Ingestion-layer output contract (T-1 artifact, Blueprint §5.2).
    T-2 treats this as read-only; never mutate fields.

    Fields:
        doc_id  : Globally unique document identifier.
        tokens  : List of whitespace-normalised word tokens.
        lemmas  : Lemmatised form of each token (preferred for TF-IDF).
    """
    doc_id: str
    tokens: list[str]
    lemmas: list[str]


@dataclass
class DTMatrix:
    """
    Document-term matrix contract shared with lda_model.py, nmf_model.py,
    kmeans_clusterer.py, and cost_function.py (Blueprint §5.2, §10.3).

    FROZEN — do not add or remove fields without an ADR (Blueprint §10.3).

    Fields:
        matrix        : Sparse TF-IDF matrix, shape (n_docs, vocab_size).
        feature_names : Vocabulary in column order.
        corpus_id     : Identifier of the source corpus.
        doc_ids       : Ordered list of doc_id strings matching matrix rows.
        params        : Config snapshot logged to MLflow for reproducibility.
    """
    matrix: csr_matrix
    feature_names: list[str]
    corpus_id: str
    doc_ids: list[str]
    params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def _load_classical_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load and validate config/classical_config.yaml.

    The 'tfidf' subsection is merged with defaults so that partial configs
    remain valid. validate_config() (Blueprint §8.2) raises ConfigError if
    schema constraints are violated.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Fully resolved tfidf configuration dict.
    """
    path = Path(config_path)
    if not path.exists():
        logger.warning(
            "Config file not found at '%s'. Using built-in defaults.",
            config_path,
        )
        return {}

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # validate_config raises shared.validator.ConfigError on schema violation
    validate_config(raw, schema_name="classical_config")

    # Drill down to the tfidf subsection; fall back to empty dict if absent
    tfidf_cfg: dict[str, Any] = raw.get("classical", {}).get("tfidf", {})
    return tfidf_cfg


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class TFIDFVectorizer:
    """
    Wraps sklearn TfidfVectorizer to produce a DTMatrix compatible with
    every downstream L2 and L3 consumer defined in the Master Blueprint.

    Usage (canonical pipeline — Blueprint §5.1, Stage 04 preamble):
        >>> from src.classical.tfidf_vectorizer import TFIDFVectorizer
        >>> vectorizer = TFIDFVectorizer(corpus_id="20ng")
        >>> dt_matrix = vectorizer.fit_transform(clean_docs)
        >>> vectorizer.save()          # persists to outputs/models/
        >>> dt_matrix_reload = TFIDFVectorizer.load(model_path).transform(new_docs)

    All numeric hyperparameters are sourced from config/classical_config.yaml
    under the `classical.tfidf` key. Any key absent from the YAML falls back
    to the module-level DEFAULT_* constants above.

    Attributes:
        corpus_id      : Source corpus identifier (propagated into DTMatrix).
        config_path    : Path to classical_config.yaml.
        _vectorizer    : The underlying sklearn TfidfVectorizer instance.
        _is_fitted     : Guard flag; raised on premature transform/save calls.
        _params        : Resolved hyperparameter snapshot for MLflow logging.
    """

    def __init__(
        self,
        corpus_id: str,
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> None:
        """
        Initialise the vectorizer and resolve hyperparameters from config.

        Args:
            corpus_id  : Identifier for the corpus being vectorised. Stored
                         in the DTMatrix and artifact file names.
            config_path: Path to classical_config.yaml. Defaults to the
                         project-root relative path specified in §9.2.
        """
        self.corpus_id: str = corpus_id
        self.config_path: str = config_path
        self._is_fitted: bool = False

        # Resolve config — override defaults with YAML values where present
        cfg = _load_classical_config(config_path)
        self._params: dict[str, Any] = {
            "max_features":  cfg.get("max_features",  DEFAULT_MAX_FEATURES),
            "min_df":        cfg.get("min_df",         DEFAULT_MIN_DF),
            "max_df":        cfg.get("max_df",         DEFAULT_MAX_DF),
            "ngram_range":   tuple(cfg.get("ngram_range", list(DEFAULT_NGRAM_RANGE))),
            "sublinear_tf":  cfg.get("sublinear_tf",   DEFAULT_SUBLINEAR_TF),
            "analyzer":      cfg.get("analyzer",       "word"),
            "strip_accents": cfg.get("strip_accents",  "unicode"),
            "use_lemmas":    cfg.get("use_lemmas",     True),  # custom flag
        }

        logger.info(
            "TFIDFVectorizer initialised | corpus_id=%s | params=%s",
            corpus_id,
            self._params,
        )

        # Build the sklearn estimator from resolved params
        self._vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=self._params["max_features"],
            min_df=self._params["min_df"],
            max_df=self._params["max_df"],
            ngram_range=self._params["ngram_range"],
            sublinear_tf=self._params["sublinear_tf"],
            analyzer=self._params["analyzer"],
            strip_accents=self._params["strip_accents"],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _docs_to_strings(self, docs: list[CleanDocument]) -> list[str]:
        """
        Convert CleanDocument objects to raw strings for sklearn.

        If `use_lemmas` is True (default, recommended for topic modelling),
        the lemma list is joined; otherwise the token list is used.

        Args:
            docs: List of CleanDocument from the ingestion layer.

        Returns:
            List of plain strings, one per document, in the same order.
        """
        use_lemmas: bool = self._params["use_lemmas"]
        texts: list[str] = []
        for doc in docs:
            word_list = doc.lemmas if use_lemmas else doc.tokens
            texts.append(" ".join(word_list))
        return texts

    # ------------------------------------------------------------------
    # Public API — fit / transform / fit_transform
    # ------------------------------------------------------------------

    def fit(self, docs: list[CleanDocument]) -> "TFIDFVectorizer":
        """
        Learn the vocabulary and IDF weights from the provided documents.

        Does NOT return a DTMatrix — call transform() or fit_transform()
        when the matrix is needed immediately.

        Args:
            docs: List of CleanDocument objects from T-1 ingestion layer.

        Returns:
            self (fluent interface).

        Raises:
            ValueError: If docs is empty.
        """
        if not docs:
            raise ValueError("Cannot fit TFIDFVectorizer on an empty document list.")

        logger.info("Fitting TF-IDF vectorizer | n_docs=%d", len(docs))
        t0 = time.perf_counter()

        texts = self._docs_to_strings(docs)
        self._vectorizer.fit(texts)
        self._is_fitted = True

        elapsed = time.perf_counter() - t0
        vocab_size = len(self._vectorizer.vocabulary_)
        logger.info(
            "TF-IDF fit complete | vocab_size=%d | elapsed_s=%.3f",
            vocab_size,
            elapsed,
        )
        return self

    def transform(self, docs: list[CleanDocument]) -> DTMatrix:
        """
        Transform documents into a DTMatrix using the fitted vocabulary.

        Args:
            docs: List of CleanDocument objects to transform.

        Returns:
            DTMatrix with sparse TF-IDF matrix and metadata.

        Raises:
            RuntimeError : If called before fit().
            ValueError   : If docs is empty.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "TFIDFVectorizer.transform() called before fit(). "
                "Call fit() or fit_transform() first."
            )
        if not docs:
            raise ValueError("Cannot transform an empty document list.")

        logger.info("Transforming %d documents to DTMatrix.", len(docs))
        t0 = time.perf_counter()

        texts = self._docs_to_strings(docs)
        sparse_matrix: csr_matrix = self._vectorizer.transform(texts)
        feature_names: list[str] = list(self._vectorizer.get_feature_names_out())
        doc_ids: list[str] = [d.doc_id for d in docs]

        elapsed = time.perf_counter() - t0
        logger.info(
            "DTMatrix produced | shape=%s | nnz=%d | elapsed_s=%.3f",
            sparse_matrix.shape,
            sparse_matrix.nnz,
            elapsed,
        )

        return DTMatrix(
            matrix=sparse_matrix,
            feature_names=feature_names,
            corpus_id=self.corpus_id,
            doc_ids=doc_ids,
            params=dict(self._params),  # copy — immutable snapshot
        )

    def fit_transform(self, docs: list[CleanDocument]) -> DTMatrix:
        """
        Fit vocabulary on docs and immediately return the DTMatrix.

        Equivalent to calling fit(docs).transform(docs) but avoids
        converting docs to strings twice.

        Args:
            docs: List of CleanDocument objects (training split).

        Returns:
            DTMatrix for the provided documents.

        Raises:
            ValueError: If docs is empty.
        """
        if not docs:
            raise ValueError("Cannot fit_transform on an empty document list.")

        logger.info(
            "fit_transform | corpus_id=%s | n_docs=%d", self.corpus_id, len(docs)
        )

        # Use the more efficient sklearn fit_transform under the hood
        t0 = time.perf_counter()
        texts = self._docs_to_strings(docs)
        sparse_matrix: csr_matrix = self._vectorizer.fit_transform(texts)
        self._is_fitted = True

        feature_names: list[str] = list(self._vectorizer.get_feature_names_out())
        doc_ids: list[str] = [d.doc_id for d in docs]
        elapsed = time.perf_counter() - t0

        logger.info(
            "fit_transform complete | shape=%s | vocab_size=%d | elapsed_s=%.3f",
            sparse_matrix.shape,
            len(feature_names),
            elapsed,
        )

        return DTMatrix(
            matrix=sparse_matrix,
            feature_names=feature_names,
            corpus_id=self.corpus_id,
            doc_ids=doc_ids,
            params=dict(self._params),
        )

    # ------------------------------------------------------------------
    # Persistence — save / load (Blueprint §8.3)
    # ------------------------------------------------------------------

    def save(self, run_id: str, output_dir: str = "outputs/models") -> Path:
        """
        Persist the fitted sklearn vectorizer to disk via shared.serializer.

        Artifact path follows the Blueprint §5 naming scheme:
            outputs/models/{run_id}_tfidf_vectorizer.pkl

        Args:
            run_id    : MLflow run identifier or unique label.
            output_dir: Output directory; defaults to outputs/models/.

        Returns:
            Path to the saved artifact.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted TFIDFVectorizer.")

        artifact_name = f"{run_id}_{VECTORIZER_ARTIFACT_TYPE}"
        out_path = Path(output_dir) / f"{artifact_name}.pkl"

        save_artifact(
            obj=self._vectorizer,
            path=str(out_path),
            fmt="pkl",
        )
        logger.info("TFIDFVectorizer saved | path=%s", out_path)
        return out_path

    @classmethod
    def load(
        cls,
        artifact_path: str,
        corpus_id: str = "unknown",
        config_path: str = DEFAULT_CONFIG_PATH,
    ) -> "TFIDFVectorizer":
        """
        Reconstruct a TFIDFVectorizer from a previously saved artifact.

        Args:
            artifact_path: Path to the .pkl file written by save().
            corpus_id    : Corpus identifier to attach to the loaded instance.
            config_path  : Classical config path (used for metadata only).

        Returns:
            A fitted TFIDFVectorizer ready for transform() calls.
        """
        sklearn_vec: TfidfVectorizer = load_artifact(path=artifact_path, fmt="pkl")

        instance = cls.__new__(cls)
        instance.corpus_id = corpus_id
        instance.config_path = config_path
        instance._vectorizer = sklearn_vec
        instance._is_fitted = True
        # Reconstruct params from the loaded sklearn object
        instance._params = {
            "max_features": sklearn_vec.max_features,
            "min_df":       sklearn_vec.min_df,
            "max_df":       sklearn_vec.max_df,
            "ngram_range":  sklearn_vec.ngram_range,
            "sublinear_tf": sklearn_vec.sublinear_tf,
        }

        logger.info(
            "TFIDFVectorizer loaded | artifact_path=%s | vocab_size=%d",
            artifact_path,
            len(sklearn_vec.vocabulary_),
        )
        return instance

    # ------------------------------------------------------------------
    # MLflow integration helper
    # ------------------------------------------------------------------

    def log_run(self, run_id: str, n_docs: int) -> None:
        """
        Emit structured run-start / run-end events compatible with the
        shared.logger contract (Blueprint §8.1).

        Args:
            run_id: MLflow run identifier.
            n_docs: Number of documents processed in this run.
        """
        params = {
            **self._params,
            "corpus_id": self.corpus_id,
            "n_docs":    n_docs,
            "model":     "tfidf",
        }
        log_run_start(run_id=run_id, params=params)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"TFIDFVectorizer(corpus_id={self.corpus_id!r}, status={status}, "
            f"max_features={self._params.get('max_features')})"
        )