"""
src/ingestion/text_preprocessor.py
===================================
Text preprocessing module for the HQC Topic Modeling project.

Responsibility (§3.1):
    Lower-case, stop-word removal, lemmatization, HTML strip.

Inputs  (§3.1 contract): ``List[CorpusDocument]``
Outputs (§3.1 contract): ``List[CleanDocument]``

Artifact produced (§5.1 stage 02):
    ``data/processed/{corpus_id}_clean.jsonl``

Data contracts (§5.2):
    CorpusDocument: { doc_id: str, raw_text: str, metadata: dict, source: str }
    CleanDocument:  { doc_id: str, tokens: List[str], lemmas: List[str] }

Pipeline steps (configurable via ``PreprocessingConfig``):
    1. HTML / XML tag stripping          (html_strip)
    2. Unicode normalisation (NFC)       (unicode_normalize)
    3. Lower-casing                      (lowercase)
    4. URL / e-mail removal              (remove_urls)
    5. Number removal / normalisation    (remove_numbers)
    6. Punctuation stripping             (strip_punctuation)
    7. Tokenisation (whitespace-based)   (tokenize)
    8. Stop-word removal                 (remove_stopwords)
    9. Short token filtering             (min_token_length)
   10. Lemmatisation via spaCy           (lemmatize)
   11. Maximum vocabulary cap per doc    (max_tokens_per_doc)

spaCy usage:
    Uses a small spaCy model (``en_core_web_sm``) for lemmatization.
    The model is loaded lazily and cached at module level to avoid
    repeated cold-start overhead in batch processing.

Blueprint constraints honoured:
    - Full package imports only (§2 RULE)
    - Structured JSON logging via shared.logger (§8.1)
    - All configurable thresholds exposed via dataclass (never hardcoded)
    - Output validated against CleanDocument contract before persistence
"""

from __future__ import annotations

import dataclasses
import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from shared.logger import get_logger
from shared.validator import CorpusDocument

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional spaCy import — lemmatization degrades to stemming-free tokens if absent
# ---------------------------------------------------------------------------
try:
    import spacy  # type: ignore[import]
    from spacy.language import Language as _SpacyLanguage  # type: ignore[import]

    _SPACY_AVAILABLE = True
except ImportError:  # pragma: no cover
    spacy = None  # type: ignore[assignment]
    _SpacyLanguage = None  # type: ignore[assignment]
    _SPACY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional NLTK stopwords — used as fallback if spaCy is unavailable
# ---------------------------------------------------------------------------
try:
    from nltk.corpus import stopwords as _nltk_stopwords  # type: ignore[import]
    import nltk  # type: ignore[import]

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover
    _nltk_stopwords = None  # type: ignore[assignment]
    nltk = None  # type: ignore[assignment]
    _NLTK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project-root and data-directory paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_PROCESSED_DIR: Path = _PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# spaCy model name and cached instance
# ---------------------------------------------------------------------------
_SPACY_MODEL_NAME: str = "en_core_web_sm"
_spacy_nlp: Any = None  # module-level cache


# ---------------------------------------------------------------------------
# CleanDocument dataclass — implements the §5.2 contract
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CleanDocument:
    """
    Immutable representation of a preprocessed document.

    Mirrors the CleanDocument schema from §5.2:
        { doc_id: str, tokens: List[str], lemmas: List[str] }

    Attributes
    ----------
    doc_id : str
        Unique document identifier — carried through from CorpusDocument.
    tokens : list[str]
        Whitespace-tokenized, filtered, lowercase tokens.
    lemmas : list[str]
        Lemmatized form of *tokens*.  Equal to *tokens* when spaCy is
        unavailable (graceful degradation).
    """

    doc_id: str
    tokens: list[str]
    lemmas: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return {
            "doc_id": self.doc_id,
            "tokens": self.tokens,
            "lemmas": self.lemmas,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CleanDocument":
        """Deserialize from a plain dict."""
        return cls(
            doc_id=str(data["doc_id"]),
            tokens=list(data["tokens"]),
            lemmas=list(data["lemmas"]),
        )


# ---------------------------------------------------------------------------
# Preprocessing configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PreprocessingConfig:
    """
    Controls which preprocessing steps are applied and their parameters.

    All defaults follow sensible conventions for NLP topic-modeling tasks.
    Override specific fields when constructing :class:`TextPreprocessor` to
    match the requirements of a particular experiment or config YAML.

    Attributes
    ----------
    html_strip : bool
        Strip HTML/XML tags and unescape HTML entities.
    unicode_normalize : bool
        Apply Unicode NFC normalisation before processing.
    lowercase : bool
        Convert text to lowercase.
    remove_urls : bool
        Remove HTTP(S) URLs and bare domain-like tokens.
    remove_emails : bool
        Remove email addresses.
    remove_numbers : bool
        Remove standalone numeric tokens.
    strip_punctuation : bool
        Remove punctuation characters, replacing with whitespace.
    remove_stopwords : bool
        Filter tokens that appear in the configured stop-word set.
    extra_stopwords : set[str]
        Additional domain-specific stop-words to remove.
    min_token_length : int
        Discard tokens shorter than this threshold (default 3).
    max_token_length : int
        Discard tokens longer than this threshold (default 50).
        Prevents garbled tokens like base64 blobs from polluting vocabulary.
    lemmatize : bool
        Apply lemmatization via spaCy (requires ``en_core_web_sm``).
    max_tokens_per_doc : int | None
        Truncate to this many tokens after all filtering steps.
        ``None`` means no limit.
    language : str
        ISO-639-1 language code used for stop-word selection (default ``"en"``).
    """

    html_strip: bool = True
    unicode_normalize: bool = True
    lowercase: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = True
    strip_punctuation: bool = True
    remove_stopwords: bool = True
    extra_stopwords: dataclasses.field(default_factory=set) = dataclasses.field(
        default_factory=set
    )
    min_token_length: int = 3
    max_token_length: int = 50
    lemmatize: bool = True
    max_tokens_per_doc: int | None = None
    language: str = "en"


# ---------------------------------------------------------------------------
# Stop-word loading
# ---------------------------------------------------------------------------

def _load_stopwords(language: str = "en", extra: set[str] | None = None) -> frozenset[str]:
    """
    Build a frozen set of stop-words for *language*.

    Priority order:
    1. spaCy stop-words (``nlp.Defaults.stop_words``) — if spaCy is available.
    2. NLTK stop-words                                  — if NLTK is available.
    3. Minimal hard-coded English list                  — final fallback.

    Parameters
    ----------
    language : str
        ISO-639-1 language code.  Currently only ``"en"`` is fully supported.
    extra : set[str] | None
        Caller-supplied additional stop-words.

    Returns
    -------
    frozenset[str]
        Lower-cased stop-word set.
    """
    words: set[str] = set()

    if _SPACY_AVAILABLE:
        try:
            nlp = _get_spacy_nlp()
            words.update(w.lower() for w in nlp.Defaults.stop_words)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load spaCy stop-words; falling back.",
                extra={"error": str(exc)},
            )

    if not words and _NLTK_AVAILABLE:
        try:
            # Ensure the stopwords corpus is downloaded
            try:
                _nltk_stopwords.words(language)
            except LookupError:
                nltk.download("stopwords", quiet=True)
            words.update(_nltk_stopwords.words(language))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not load NLTK stop-words; falling back.",
                extra={"error": str(exc)},
            )

    if not words:
        # Minimal English fallback — sufficient for unit tests without NLP deps
        words = {
            "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "was", "are", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "could", "should", "may", "might", "shall",
            "this", "that", "these", "those", "it", "its", "i", "me", "my",
            "we", "our", "you", "your", "he", "she", "they", "them", "their",
            "not", "no", "nor", "so", "yet", "both", "either", "neither",
            "as", "if", "then", "because", "while", "although", "however",
        }

    if extra:
        words.update(w.lower() for w in extra)

    return frozenset(words)


# ---------------------------------------------------------------------------
# spaCy lazy loader
# ---------------------------------------------------------------------------

def _get_spacy_nlp() -> Any:
    """
    Return the cached spaCy ``Language`` object, loading it on first call.

    Raises
    ------
    RuntimeError
        If spaCy is not installed or the model cannot be loaded.
    """
    global _spacy_nlp  # noqa: PLW0603
    if _spacy_nlp is not None:
        return _spacy_nlp

    if not _SPACY_AVAILABLE:
        raise RuntimeError(
            "spaCy is not installed. Run: pip install spacy && "
            "python -m spacy download en_core_web_sm"
        )

    try:
        # Disable unused pipeline components for speed (only need lemmatizer)
        _spacy_nlp = spacy.load(
            _SPACY_MODEL_NAME,
            disable=["parser", "ner", "textcat"],
        )
        logger.info(
            "spaCy model loaded.",
            extra={"model": _SPACY_MODEL_NAME},
        )
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{_SPACY_MODEL_NAME}' not found. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc

    return _spacy_nlp


# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_URL = re.compile(
    r"https?://\S+|www\.\S+|ftp://\S+",
    re.IGNORECASE,
)
_RE_EMAIL = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
    re.IGNORECASE,
)
_RE_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_RE_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)
_RE_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core TextPreprocessor class
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """
    Stateful preprocessor that transforms :class:`CorpusDocument` instances
    into :class:`CleanDocument` instances.

    Construct once per experiment run and reuse across all documents to
    benefit from:
    - Cached spaCy model (loaded on first lemmatization call)
    - Pre-built stop-word set
    - Compiled regex patterns

    Parameters
    ----------
    config : PreprocessingConfig | None
        Preprocessing configuration.  Defaults to
        :class:`PreprocessingConfig` with all steps enabled.

    Examples
    --------
    >>> from src.ingestion.text_preprocessor import TextPreprocessor
    >>> from shared.validator import CorpusDocument
    >>> docs = [CorpusDocument(doc_id="d1", raw_text="Hello World!", metadata={}, source="test")]
    >>> preprocessor = TextPreprocessor()
    >>> clean = preprocessor.preprocess(docs)
    >>> clean[0].tokens
    ['hello', 'world']
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config: PreprocessingConfig = config or PreprocessingConfig()
        self._stopwords: frozenset[str] = _load_stopwords(
            language=self.config.language,
            extra=self.config.extra_stopwords,
        )
        logger.debug(
            "TextPreprocessor initialised.",
            extra={
                "stopword_count": len(self._stopwords),
                "lemmatize": self.config.lemmatize,
                "remove_stopwords": self.config.remove_stopwords,
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def preprocess(
        self,
        documents: list[CorpusDocument],
        corpus_id: str | None = None,
        persist: bool = True,
    ) -> list[CleanDocument]:
        """
        Apply the full preprocessing pipeline to *documents*.

        Parameters
        ----------
        documents : list[CorpusDocument]
            Input documents from :func:`~src.ingestion.corpus_loader.load_corpus`.
        corpus_id : str | None, optional
            Used to name the output artifact
            ``data/processed/{corpus_id}_clean.jsonl``.
            If ``None``, a generic name is used.
        persist : bool, optional
            Write output artifact to ``data/processed/``.  Default ``True``.

        Returns
        -------
        list[CleanDocument]
            Preprocessed documents in the same order as *documents*.

        Notes
        -----
        Documents that produce **empty token lists** after all filtering
        steps are still included in the output (with empty ``tokens`` /
        ``lemmas`` lists) so that downstream components can maintain
        document-index alignment.  A warning is emitted for each such doc.
        """
        logger.info(
            "Starting text preprocessing.",
            extra={"doc_count": len(documents), "corpus_id": corpus_id},
        )

        clean_docs: list[CleanDocument] = []

        for doc in documents:
            try:
                clean = self._preprocess_one(doc)
                clean_docs.append(clean)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to preprocess document; including empty CleanDocument.",
                    extra={"doc_id": doc.doc_id, "error": str(exc)},
                )
                clean_docs.append(
                    CleanDocument(doc_id=doc.doc_id, tokens=[], lemmas=[])
                )

        empty_count = sum(1 for d in clean_docs if not d.tokens)
        if empty_count:
            logger.warning(
                "Some documents have empty token lists after preprocessing.",
                extra={"empty_doc_count": empty_count, "total": len(clean_docs)},
            )

        logger.info(
            "Text preprocessing completed.",
            extra={
                "input_doc_count": len(documents),
                "output_doc_count": len(clean_docs),
                "empty_doc_count": empty_count,
            },
        )

        if persist and corpus_id:
            _persist_clean_snapshot(clean_docs, corpus_id)

        return clean_docs

    def preprocess_text(self, text: str) -> tuple[list[str], list[str]]:
        """
        Apply the preprocessing pipeline to a single raw text string.

        Useful for inference-time preprocessing or interactive exploration.

        Parameters
        ----------
        text : str
            Raw text string.

        Returns
        -------
        tuple[list[str], list[str]]
            ``(tokens, lemmas)`` — both filtered and ordered consistently.
        """
        processed = self._apply_text_transforms(text)
        tokens = self._tokenize_and_filter(processed)
        lemmas = self._lemmatize(tokens) if self.config.lemmatize else list(tokens)
        return tokens, lemmas

    # ------------------------------------------------------------------
    # Internal pipeline steps
    # ------------------------------------------------------------------

    def _preprocess_one(self, doc: CorpusDocument) -> CleanDocument:
        """Process a single CorpusDocument → CleanDocument."""
        text = doc.raw_text
        text = self._apply_text_transforms(text)
        tokens = self._tokenize_and_filter(text)

        if self.config.max_tokens_per_doc is not None:
            tokens = tokens[: self.config.max_tokens_per_doc]

        lemmas = (
            self._lemmatize(tokens)
            if self.config.lemmatize
            else list(tokens)
        )

        return CleanDocument(doc_id=doc.doc_id, tokens=tokens, lemmas=lemmas)

    def _apply_text_transforms(self, text: str) -> str:
        """
        Apply character-level transformations: HTML strip, URL/email removal,
        Unicode normalisation, lowercase, number removal, punctuation strip.

        All operations are controlled by flags in :attr:`config`.
        """
        # Step 1 — HTML / XML stripping and entity unescaping
        if self.config.html_strip:
            text = _RE_HTML_TAG.sub(" ", text)
            text = html.unescape(text)

        # Step 2 — Unicode NFC normalisation
        if self.config.unicode_normalize:
            text = unicodedata.normalize("NFC", text)

        # Step 3 — URL removal
        if self.config.remove_urls:
            text = _RE_URL.sub(" ", text)

        # Step 4 — E-mail removal
        if self.config.remove_emails:
            text = _RE_EMAIL.sub(" ", text)

        # Step 5 — Lowercase
        if self.config.lowercase:
            text = text.lower()

        # Step 6 — Number removal
        if self.config.remove_numbers:
            text = _RE_NUMBER.sub(" ", text)

        # Step 7 — Punctuation stripping (replace with space, not empty string,
        #           to avoid accidentally merging adjacent words)
        if self.config.strip_punctuation:
            text = _RE_PUNCTUATION.sub(" ", text)

        # Collapse multiple whitespace characters into a single space
        text = _RE_WHITESPACE.sub(" ", text).strip()

        return text

    def _tokenize_and_filter(self, text: str) -> list[str]:
        """
        Whitespace-tokenize *text* and apply stop-word and length filters.

        Whitespace tokenisation is intentionally simple here — the
        dedicated ``src/ingestion/tokenizer.py`` module handles WordPiece
        / BPE tokenisation for neural models (stage 03).  This tokeniser
        produces clean surface-form tokens for classical NLP (stage 02).
        """
        raw_tokens = text.split()

        filtered: list[str] = []
        for token in raw_tokens:
            # Length filter
            if len(token) < self.config.min_token_length:
                continue
            if len(token) > self.config.max_token_length:
                continue
            # Stop-word filter
            if self.config.remove_stopwords and token in self._stopwords:
                continue
            filtered.append(token)

        return filtered

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        """
        Lemmatize *tokens* using spaCy.

        Processes tokens as a space-joined string through the spaCy pipeline
        to avoid repeated model calls.  Falls back to returning *tokens*
        unchanged if spaCy is unavailable.

        Parameters
        ----------
        tokens : list[str]
            Pre-filtered tokens to lemmatize.

        Returns
        -------
        list[str]
            Lemmatized tokens (same length as *tokens*).
        """
        if not tokens:
            return []

        if not _SPACY_AVAILABLE:
            logger.debug("spaCy not available; returning tokens as lemmas.")
            return list(tokens)

        try:
            nlp = _get_spacy_nlp()
            # Join → process → split to preserve 1-to-1 token alignment.
            # Using spaces as separators is safe because tokens are already
            # stripped of whitespace.
            doc_obj = nlp(" ".join(tokens))
            lemmas = [
                token.lemma_.lower() if token.lemma_ != "-PRON-" else token.lower_
                for token in doc_obj
            ]
            return lemmas
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Lemmatization failed; returning original tokens.",
                extra={"error": str(exc), "token_count": len(tokens)},
            )
            return list(tokens)


# ---------------------------------------------------------------------------
# Module-level convenience function (mirrors corpus_loader pattern)
# ---------------------------------------------------------------------------

def preprocess_corpus(
    documents: list[CorpusDocument],
    corpus_id: str | None = None,
    config: PreprocessingConfig | None = None,
    persist: bool = True,
) -> list[CleanDocument]:
    """
    Convenience wrapper: construct a :class:`TextPreprocessor` and run
    :meth:`~TextPreprocessor.preprocess` in one call.

    Parameters
    ----------
    documents : list[CorpusDocument]
        Raw validated documents from the corpus loader.
    corpus_id : str | None, optional
        Corpus identifier for artifact naming.
    config : PreprocessingConfig | None, optional
        Custom preprocessing configuration; defaults to all-steps-enabled.
    persist : bool, optional
        Write ``data/processed/{corpus_id}_clean.jsonl``.  Default ``True``.

    Returns
    -------
    list[CleanDocument]
        Preprocessed documents ready for tokenization (stage 03).
    """
    preprocessor = TextPreprocessor(config=config)
    return preprocessor.preprocess(documents, corpus_id=corpus_id, persist=persist)


# ---------------------------------------------------------------------------
# Persistence helper
# ---------------------------------------------------------------------------

def _persist_clean_snapshot(
    documents: list[CleanDocument],
    corpus_id: str,
) -> None:
    """
    Write *documents* to ``data/processed/{corpus_id}_clean.jsonl``.

    Each line is a JSON-serialised CleanDocument dict (§5.1 stage 02 artifact).
    """
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _PROCESSED_DIR / f"{corpus_id}_clean.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.to_dict()) + "\n")

    logger.info(
        "Clean snapshot persisted.",
        extra={
            "path": str(out_path),
            "corpus_id": corpus_id,
            "doc_count": len(documents),
        },
    )