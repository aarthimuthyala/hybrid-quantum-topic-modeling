"""
src/ingestion/vocab_builder.py
===============================
Vocabulary construction and persistence module for the HQC Topic Modeling project.

Responsibility (§3.1):
    Build and persist vocabulary mapping (token → int).

Inputs  (§3.1 contract): ``List[TokenizedDocument]``
Outputs (§3.1 contract): ``VocabIndex`` (saved to ``data/splits/``)

Artifact produced (§5.1 stage 03):
    ``data/splits/{corpus_id}_vocab.pkl``

VocabIndex contract (inferred from §3.1 and §5.1):
    A ``VocabIndex`` encapsulates the bidirectional token↔id mapping and
    associated corpus statistics.  Its public interface is:

        vocab_index.token2id  : dict[str, int]
        vocab_index.id2token  : dict[int, str]
        vocab_index.vocab_size: int
        vocab_index.doc_freq  : dict[str, int]   # document frequency per token
        vocab_index.corpus_freq: dict[str, int]  # total occurrence count
        vocab_index.save(path)
        VocabIndex.load(path) → VocabIndex

    The vocabulary is derived from ``TokenizedDocument.token_ids`` + the
    tokeniser's ``get_vocab()`` method.  Alternatively, when constructed
    directly from text (e.g. for classical NLP pipelines), it can be
    built from ``List[CleanDocument]`` token lists.

Blueprint constraints honoured:
    - Full package imports only (§2 RULE)
    - All numeric thresholds configurable via ``VocabConfig``
    - Structured JSON logging via ``shared.logger`` (§8.1)
    - SHA-256 tracking via ``data/manifest.json`` (§7.2)
    - Artifact saved as ``.pkl`` to ``data/splits/`` (§5.1)
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from shared.logger import get_logger
from src.ingestion.text_preprocessor import CleanDocument
from src.ingestion.tokenizer import (
    CorpusTokenizer,
    TokenizedDocument,
    PAD_TOKEN,
    UNK_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN,
    MASK_TOKEN,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_SPLITS_DIR: Path = _PROJECT_ROOT / "data" / "splits"
_MANIFEST_PATH: Path = _PROJECT_ROOT / "data" / "manifest.json"

# ---------------------------------------------------------------------------
# Reserved special token IDs (must be consistent with tokenizer.py)
# ---------------------------------------------------------------------------
_SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN)


# ---------------------------------------------------------------------------
# VocabConfig dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class VocabConfig:
    """
    Configuration for vocabulary construction.

    Attributes
    ----------
    max_vocab_size : int
        Hard cap on number of entries (excluding special tokens).
        Tokens are ranked by document frequency descending; low-frequency
        tokens are pruned when the cap is exceeded.  Default 50_000.
    min_doc_freq : int
        Minimum number of documents a token must appear in to be retained.
        Filters rare tokens that add noise without benefit.  Default 2.
    max_doc_freq_ratio : float
        Maximum proportion of documents a token may appear in.  Tokens
        above this threshold are treated as corpus-level stop-words and
        pruned.  Range (0.0, 1.0].  Default 0.95.
    include_special_tokens : bool
        Whether to include HF special tokens ([PAD], [UNK], …) in the
        vocabulary index.  Should match the tokeniser's behaviour.
        Default ``True``.
    """

    max_vocab_size: int = 50_000
    min_doc_freq: int = 2
    max_doc_freq_ratio: float = 0.95
    include_special_tokens: bool = True


# ---------------------------------------------------------------------------
# VocabIndex — the main output contract
# ---------------------------------------------------------------------------

class VocabIndex:
    """
    Bidirectional token ↔ integer ID mapping with corpus-frequency statistics.

    This is the ``VocabIndex`` output type specified in §3.1 for
    ``vocab_builder.py``.

    Attributes
    ----------
    token2id : dict[str, int]
        Maps token string → integer ID.
    id2token : dict[int, str]
        Reverse mapping: integer ID → token string.
    vocab_size : int
        Number of entries (including special tokens when present).
    doc_freq : dict[str, int]
        Number of documents each token appears in.
    corpus_freq : dict[str, int]
        Total number of occurrences of each token across the full corpus.
    corpus_id : str
        Identifier of the corpus this vocabulary was built from.
    config : VocabConfig
        The configuration used to build this vocabulary.
    """

    def __init__(
        self,
        token2id: dict[str, int],
        doc_freq: dict[str, int],
        corpus_freq: dict[str, int],
        corpus_id: str,
        config: VocabConfig,
    ) -> None:
        self.token2id: dict[str, int] = token2id
        self.id2token: dict[int, str] = {v: k for k, v in token2id.items()}
        self.doc_freq: dict[str, int] = doc_freq
        self.corpus_freq: dict[str, int] = corpus_freq
        self.corpus_id: str = corpus_id
        self.config: VocabConfig = config

    @property
    def vocab_size(self) -> int:
        """Number of entries in the vocabulary."""
        return len(self.token2id)

    def lookup(self, token: str) -> int:
        """
        Return the integer ID for *token*, or the UNK ID if unknown.

        Parameters
        ----------
        token : str
            Token string to look up.

        Returns
        -------
        int
            Integer ID, or ``token2id[UNK_TOKEN]`` (1) if not found.
        """
        return self.token2id.get(token, self.token2id.get(UNK_TOKEN, 1))

    def reverse_lookup(self, token_id: int) -> str:
        """
        Return the token string for *token_id*, or ``UNK_TOKEN`` if unknown.

        Parameters
        ----------
        token_id : int

        Returns
        -------
        str
        """
        return self.id2token.get(token_id, UNK_TOKEN)

    def ids_to_tokens(self, ids: list[int]) -> list[str]:
        """Decode a list of integer IDs to token strings."""
        return [self.reverse_lookup(i) for i in ids]

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Encode a list of token strings to integer IDs."""
        return [self.lookup(t) for t in tokens]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a plain JSON-safe dict (for inspection / audit only).

        Note: The full vocab is included; for large vocabularies this may
        produce large JSON objects.  Use :meth:`save` for persistence.
        """
        return {
            "corpus_id": self.corpus_id,
            "vocab_size": self.vocab_size,
            "token2id": self.token2id,
            "doc_freq": self.doc_freq,
            "corpus_freq": self.corpus_freq,
        }

    def save(self, path: str | Path | None = None) -> Path:
        """
        Persist the ``VocabIndex`` to disk as a ``.pkl`` file.

        Default location: ``data/splits/{corpus_id}_vocab.pkl`` (§5.1 artifact).

        Parameters
        ----------
        path : str | Path | None, optional
            Explicit output path.  If ``None``, uses the canonical path
            ``data/splits/{corpus_id}_vocab.pkl``.

        Returns
        -------
        Path
            The absolute path where the file was written.
        """
        if path is None:
            _SPLITS_DIR.mkdir(parents=True, exist_ok=True)
            out_path = _SPLITS_DIR / f"{self.corpus_id}_vocab.pkl"
        else:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # Record SHA-256 in manifest (§7.2)
        raw_bytes = out_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        _update_vocab_manifest(self.corpus_id, sha256, self.vocab_size)

        logger.info(
            "VocabIndex saved.",
            extra={
                "path": str(out_path),
                "corpus_id": self.corpus_id,
                "vocab_size": self.vocab_size,
                "sha256": sha256[:12] + "…",
            },
        )
        return out_path

    @classmethod
    def load(cls, path: str | Path) -> "VocabIndex":
        """
        Load a ``VocabIndex`` from a ``.pkl`` file.

        Parameters
        ----------
        path : str | Path
            Path to the ``.pkl`` file produced by :meth:`save`.

        Returns
        -------
        VocabIndex

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        TypeError
            If the file does not contain a ``VocabIndex`` instance.
        """
        out_path = Path(path)
        if not out_path.exists():
            raise FileNotFoundError(f"VocabIndex file not found: {out_path}")

        with out_path.open("rb") as fh:
            obj = pickle.load(fh)  # noqa: S301

        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected VocabIndex, got {type(obj).__name__} in {out_path}"
            )

        logger.info(
            "VocabIndex loaded.",
            extra={"path": str(out_path), "vocab_size": obj.vocab_size},
        )
        return obj

    def __repr__(self) -> str:
        return (
            f"VocabIndex(corpus_id={self.corpus_id!r}, "
            f"vocab_size={self.vocab_size}, "
            f"min_doc_freq={self.config.min_doc_freq})"
        )


# ---------------------------------------------------------------------------
# Internal manifest helper
# ---------------------------------------------------------------------------

def _update_vocab_manifest(corpus_id: str, sha256: str, vocab_size: int) -> None:
    """Append vocab SHA-256 entry to ``data/manifest.json`` (§7.2)."""
    manifest: dict[str, Any] = {}
    if _MANIFEST_PATH.exists():
        try:
            with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    key = f"{corpus_id}_vocab"
    manifest[key] = {"sha256": sha256, "vocab_size": vocab_size}

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


# ---------------------------------------------------------------------------
# VocabBuilder class
# ---------------------------------------------------------------------------

class VocabBuilder:
    """
    Constructs a :class:`VocabIndex` from tokenized or cleaned documents.

    Two construction paths are provided:

    Path A — from ``List[TokenizedDocument]`` + a fitted ``CorpusTokenizer``:
        The tokeniser already holds the definitive token→id mapping.
        ``VocabBuilder`` enriches it with document-frequency statistics
        derived from the token ID sequences.
        Use: :meth:`build_from_tokenized`

    Path B — from ``List[CleanDocument]`` (surface tokens only):
        Builds the vocabulary directly from token strings without sub-word
        splitting.  Useful for classical NLP pipelines (LDA/NMF) that
        operate on bag-of-words representations.
        Use: :meth:`build_from_clean`

    Parameters
    ----------
    config : VocabConfig | None
        Vocabulary construction configuration.
    """

    def __init__(self, config: VocabConfig | None = None) -> None:
        self.config: VocabConfig = config or VocabConfig()

    # ------------------------------------------------------------------
    # Path A — from TokenizedDocuments
    # ------------------------------------------------------------------

    def build_from_tokenized(
        self,
        documents: list[TokenizedDocument],
        tokenizer: CorpusTokenizer,
        corpus_id: str,
        persist: bool = True,
    ) -> VocabIndex:
        """
        Build a :class:`VocabIndex` from ``List[TokenizedDocument]`` and a
        fitted :class:`~src.ingestion.tokenizer.CorpusTokenizer`.

        Satisfies the §3.1 contract:
            Input:  ``List[TokenizedDocument]``
            Output: ``VocabIndex`` (saved to ``data/splits/``)

        Parameters
        ----------
        documents : list[TokenizedDocument]
            Tokenized documents from :class:`~src.ingestion.tokenizer.CorpusTokenizer`.
        tokenizer : CorpusTokenizer
            Fitted tokeniser whose ``get_vocab()`` provides the base
            token → id mapping.
        corpus_id : str
            Corpus identifier used for artifact naming.
        persist : bool, optional
            Save to ``data/splits/{corpus_id}_vocab.pkl``.  Default ``True``.

        Returns
        -------
        VocabIndex
            Enriched vocabulary index with document-frequency statistics.
        """
        logger.info(
            "Building VocabIndex from tokenized documents.",
            extra={"doc_count": len(documents), "corpus_id": corpus_id},
        )

        # Base token→id mapping from the tokeniser
        base_vocab: dict[str, int] = tokenizer.get_vocab()

        # Build id→token reverse map for frequency counting
        id2token: dict[int, str] = {v: k for k, v in base_vocab.items()}

        # Compute per-token document frequency and corpus frequency
        doc_freq: Counter[str] = Counter()
        corpus_freq: Counter[str] = Counter()

        for doc in documents:
            # Use a set for doc_freq (count each token once per document)
            seen_in_doc: set[str] = set()
            for token_id in doc.token_ids:
                token = id2token.get(token_id, UNK_TOKEN)
                # Skip padding and unknown
                if token in (PAD_TOKEN,):
                    continue
                corpus_freq[token] += 1
                seen_in_doc.add(token)
            doc_freq.update(seen_in_doc)

        # Apply pruning rules from VocabConfig
        filtered_vocab = self._prune_vocab(
            base_vocab, doc_freq, corpus_freq, total_docs=len(documents)
        )

        vocab_index = VocabIndex(
            token2id=filtered_vocab,
            doc_freq=dict(doc_freq),
            corpus_freq=dict(corpus_freq),
            corpus_id=corpus_id,
            config=self.config,
        )

        logger.info(
            "VocabIndex built from tokenized documents.",
            extra={
                "corpus_id": corpus_id,
                "raw_vocab_size": len(base_vocab),
                "pruned_vocab_size": vocab_index.vocab_size,
            },
        )

        if persist:
            vocab_index.save()

        return vocab_index

    # ------------------------------------------------------------------
    # Path B — from CleanDocuments
    # ------------------------------------------------------------------

    def build_from_clean(
        self,
        documents: list[CleanDocument],
        corpus_id: str,
        use_lemmas: bool = False,
        persist: bool = True,
    ) -> VocabIndex:
        """
        Build a :class:`VocabIndex` from surface tokens in ``CleanDocument``
        objects, without sub-word splitting.

        Used by classical NLP models (LDA, NMF, TF-IDF) that operate on
        bag-of-words representations and do not require sub-word tokenisation.

        Parameters
        ----------
        documents : list[CleanDocument]
            Pre-processed documents from :class:`~src.ingestion.text_preprocessor.TextPreprocessor`.
        corpus_id : str
            Corpus identifier for artifact naming.
        use_lemmas : bool, optional
            If ``True``, build vocabulary from the ``lemmas`` field;
            otherwise use ``tokens``.  Default ``False``.
        persist : bool, optional
            Save to ``data/splits/{corpus_id}_vocab.pkl``.  Default ``True``.

        Returns
        -------
        VocabIndex
            Vocabulary index with integer IDs assigned by descending
            document frequency (most common token = lowest non-special ID).
        """
        logger.info(
            "Building VocabIndex from CleanDocuments.",
            extra={
                "doc_count": len(documents),
                "corpus_id": corpus_id,
                "use_lemmas": use_lemmas,
            },
        )

        doc_freq: Counter[str] = Counter()
        corpus_freq: Counter[str] = Counter()

        for doc in documents:
            token_list = doc.lemmas if use_lemmas else doc.tokens
            seen_in_doc: set[str] = set(token_list)
            doc_freq.update(seen_in_doc)
            corpus_freq.update(token_list)

        # Build token → id mapping
        # Special tokens first (IDs 0–4), then corpus tokens ranked by doc_freq
        token2id: dict[str, int] = {}

        if self.config.include_special_tokens:
            for idx, special in enumerate(_SPECIAL_TOKENS):
                token2id[special] = idx
            next_id = len(_SPECIAL_TOKENS)
        else:
            next_id = 0

        total_docs = len(documents)

        # Rank candidates by document frequency descending, then alphabetically
        candidates = sorted(
            doc_freq.items(),
            key=lambda x: (-x[1], x[0]),
        )

        for token, df in candidates:
            # Minimum document frequency filter
            if df < self.config.min_doc_freq:
                continue
            # Maximum document frequency ratio filter (stop-word guard)
            if total_docs > 0 and (df / total_docs) > self.config.max_doc_freq_ratio:
                continue
            # Skip tokens that are already reserved (special tokens)
            if token in token2id:
                continue
            # Vocabulary size cap (special tokens don't count against cap)
            non_special = next_id - (len(_SPECIAL_TOKENS) if self.config.include_special_tokens else 0)
            if non_special >= self.config.max_vocab_size:
                break
            token2id[token] = next_id
            next_id += 1

        vocab_index = VocabIndex(
            token2id=token2id,
            doc_freq=dict(doc_freq),
            corpus_freq=dict(corpus_freq),
            corpus_id=corpus_id,
            config=self.config,
        )

        logger.info(
            "VocabIndex built from CleanDocuments.",
            extra={
                "corpus_id": corpus_id,
                "candidate_tokens": len(doc_freq),
                "final_vocab_size": vocab_index.vocab_size,
                "use_lemmas": use_lemmas,
            },
        )

        if persist:
            vocab_index.save()

        return vocab_index

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _prune_vocab(
        self,
        base_vocab: dict[str, int],
        doc_freq: Counter[str],
        corpus_freq: Counter[str],
        total_docs: int,
    ) -> dict[str, int]:
        """
        Apply ``min_doc_freq``, ``max_doc_freq_ratio``, and
        ``max_vocab_size`` filters to *base_vocab*.

        Special tokens are always retained regardless of frequency.

        Returns a new token→id dict with IDs re-assigned contiguously
        starting from 0 to avoid sparse ID spaces after pruning.
        """
        special_set = set(_SPECIAL_TOKENS)

        # Determine which tokens survive frequency filtering
        survivors: list[tuple[str, int]] = []  # (token, original_id)

        for token, original_id in base_vocab.items():
            if token in special_set:
                # Always keep special tokens
                survivors.append((token, original_id))
                continue

            df = doc_freq.get(token, 0)

            if df < self.config.min_doc_freq:
                continue
            if total_docs > 0 and (df / total_docs) > self.config.max_doc_freq_ratio:
                continue

            survivors.append((token, original_id))

        # Sort: special tokens first (by original ID), then by doc_freq descending
        def _sort_key(item: tuple[str, int]) -> tuple[int, int, str]:
            token, original_id = item
            is_special = 0 if token in special_set else 1
            return (is_special, -doc_freq.get(token, 0), token)

        survivors.sort(key=_sort_key)

        # Apply max_vocab_size cap (special tokens exempt)
        filtered: list[tuple[str, int]] = []
        non_special_count = 0
        for token, original_id in survivors:
            if token in special_set:
                filtered.append((token, original_id))
            else:
                if non_special_count < self.config.max_vocab_size:
                    filtered.append((token, original_id))
                    non_special_count += 1

        # Re-assign contiguous IDs
        new_vocab: dict[str, int] = {}
        for new_id, (token, _) in enumerate(filtered):
            new_vocab[token] = new_id

        pruned_count = len(base_vocab) - len(new_vocab)
        if pruned_count > 0:
            logger.debug(
                "Vocabulary pruned.",
                extra={
                    "original": len(base_vocab),
                    "retained": len(new_vocab),
                    "pruned": pruned_count,
                },
            )

        return new_vocab


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def build_vocab(
    tokenized_docs: list[TokenizedDocument] | None = None,
    clean_docs: list[CleanDocument] | None = None,
    tokenizer: CorpusTokenizer | None = None,
    corpus_id: str = "corpus",
    config: VocabConfig | None = None,
    use_lemmas: bool = False,
    persist: bool = True,
) -> VocabIndex:
    """
    Convenience wrapper that dispatches to the appropriate ``VocabBuilder``
    method depending on which inputs are supplied.

    Exactly one of (*tokenized_docs* + *tokenizer*) or *clean_docs* must
    be provided.

    Parameters
    ----------
    tokenized_docs : list[TokenizedDocument] | None
        Output of :func:`~src.ingestion.tokenizer.tokenize_corpus`.
        Must be paired with *tokenizer*.
    clean_docs : list[CleanDocument] | None
        Output of :func:`~src.ingestion.text_preprocessor.preprocess_corpus`.
        Used for classical NLP vocabularies (no sub-word splitting).
    tokenizer : CorpusTokenizer | None
        Fitted tokeniser; required when *tokenized_docs* is provided.
    corpus_id : str, optional
        Corpus identifier for artifact naming.  Default ``"corpus"``.
    config : VocabConfig | None, optional
        Vocabulary config.  Defaults to ``VocabConfig()``.
    use_lemmas : bool, optional
        Only relevant for the ``clean_docs`` path.  Default ``False``.
    persist : bool, optional
        Save artifact to ``data/splits/``.  Default ``True``.

    Returns
    -------
    VocabIndex

    Raises
    ------
    ValueError
        If neither or both input paths are provided.
    """
    if tokenized_docs is not None and clean_docs is not None:
        raise ValueError(
            "Provide either 'tokenized_docs' (+ 'tokenizer') "
            "or 'clean_docs', not both."
        )
    if tokenized_docs is None and clean_docs is None:
        raise ValueError(
            "One of 'tokenized_docs' or 'clean_docs' must be provided."
        )

    builder = VocabBuilder(config=config)

    if tokenized_docs is not None:
        if tokenizer is None:
            raise ValueError(
                "'tokenizer' must be provided when using 'tokenized_docs'."
            )
        return builder.build_from_tokenized(
            tokenized_docs,
            tokenizer=tokenizer,
            corpus_id=corpus_id,
            persist=persist,
        )

    return builder.build_from_clean(
        clean_docs,  # type: ignore[arg-type]  # guarded above
        corpus_id=corpus_id,
        use_lemmas=use_lemmas,
        persist=persist,
    )