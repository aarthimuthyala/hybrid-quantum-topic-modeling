"""
src/ingestion/tokenizer.py
==========================
Sub-word tokenisation module for the HQC Topic Modeling project.

Responsibility (§3.1):
    Word-piece / BPE tokenisation; produce token IDs.

Inputs  (§3.1 contract): ``List[CleanDocument]``
Outputs (§3.1 contract): ``List[TokenizedDocument]``

Artifact produced (§5.1 stage 03 — jointly with vocab_builder.py):
    ``data/splits/{corpus_id}_vocab.pkl``

Data contracts (§5.2):
    CleanDocument:     { doc_id: str, tokens: List[str], lemmas: List[str] }
    TokenizedDocument: { doc_id: str, token_ids: List[int], attention_mask: List[int] }

Sub-word strategy:
    HuggingFace ``tokenizers`` library is the primary backend (as mandated by
    §1.1 L1 key technologies: "HuggingFace Tokenizers").  Two tokeniser
    sub-types are supported, selectable via ``TokenizerConfig.algorithm``:

    - ``"wordpiece"``  — WordPiece (BERT-style); trained from scratch on the
                         corpus vocabulary, or loaded from a pre-trained HF
                         model checkpoint.
    - ``"bpe"``        — Byte-Pair Encoding (GPT-style); same fit/load options.

    Fallback mode (``algorithm="whitespace"``):
        When the ``tokenizers`` library is absent (e.g. minimal CI
        environment), the module falls back to a simple integer-encoded
        whitespace tokeniser backed by an in-memory vocabulary dict.

Padding / truncation:
    Both are applied during encoding to produce fixed-length sequences.
    ``max_length`` defaults to 512 to match standard transformer limits;
    override via ``TokenizerConfig.max_length``.
    ``attention_mask`` is 1 for real tokens, 0 for padding positions.

Blueprint constraints honoured:
    - Full package imports only (§2 RULE — Flat Imports Only)
    - All numeric hyperparameters sourced from ``TokenizerConfig`` (§9)
    - Structured JSON logging via ``shared.logger`` (§8.1)
    - Output artifact written to ``data/splits/`` (§5.1 stage 03)
"""

from __future__ import annotations

import dataclasses
import json
import pickle
from pathlib import Path
from typing import Any

from shared.logger import get_logger
from src.ingestion.text_preprocessor import CleanDocument

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional HuggingFace tokenizers import
# ---------------------------------------------------------------------------
try:
    from tokenizers import Tokenizer as _HFTokenizer  # type: ignore[import]
    from tokenizers.models import WordPiece as _WordPiece, BPE as _BPE  # type: ignore[import]
    from tokenizers.trainers import (  # type: ignore[import]
        WordPieceTrainer as _WordPieceTrainer,
        BpeTrainer as _BpeTrainer,
    )
    from tokenizers.pre_tokenizers import Whitespace as _Whitespace  # type: ignore[import]
    from tokenizers.processors import TemplateProcessing as _TemplateProcessing  # type: ignore[import]

    _HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _HF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_SPLITS_DIR: Path = _PROJECT_ROOT / "data" / "splits"

# ---------------------------------------------------------------------------
# Special token constants (BERT-compatible defaults)
# ---------------------------------------------------------------------------
PAD_TOKEN: str = "[PAD]"
UNK_TOKEN: str = "[UNK]"
CLS_TOKEN: str = "[CLS]"
SEP_TOKEN: str = "[SEP]"
MASK_TOKEN: str = "[MASK]"

PAD_ID: int = 0  # Canonical padding token ID — must match VocabIndex convention


# ---------------------------------------------------------------------------
# TokenizedDocument dataclass — §5.2 contract
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class TokenizedDocument:
    """
    Immutable representation of a tokenized document.

    Implements the TokenizedDocument schema from §5.2:
        { doc_id: str, token_ids: List[int], attention_mask: List[int] }

    Attributes
    ----------
    doc_id : str
        Unique document identifier carried through from CleanDocument.
    token_ids : list[int]
        Integer IDs corresponding to sub-word or surface-form tokens.
        Padded / truncated to ``TokenizerConfig.max_length``.
    attention_mask : list[int]
        Binary mask: 1 for real tokens, 0 for padding positions.
        Always the same length as ``token_ids``.
    """

    doc_id: str
    token_ids: list[int]
    attention_mask: list[int]

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.attention_mask):
            raise ValueError(
                f"token_ids (len={len(self.token_ids)}) and "
                f"attention_mask (len={len(self.attention_mask)}) "
                "must have equal length."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-safe dict."""
        return {
            "doc_id": self.doc_id,
            "token_ids": self.token_ids,
            "attention_mask": self.attention_mask,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenizedDocument":
        """Deserialize from a plain dict."""
        return cls(
            doc_id=str(data["doc_id"]),
            token_ids=list(data["token_ids"]),
            attention_mask=list(data["attention_mask"]),
        )


# ---------------------------------------------------------------------------
# Tokenizer configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TokenizerConfig:
    """
    Configuration for the corpus tokeniser.

    Attributes
    ----------
    algorithm : str
        Tokenisation algorithm.  One of ``"wordpiece"``, ``"bpe"``,
        ``"whitespace"``.  ``"whitespace"`` is the fallback mode used
        when the ``tokenizers`` library is absent.
    max_length : int
        Maximum sequence length (inclusive of special tokens).
        Sequences are truncated to this length; shorter sequences are
        padded.  Default 512.
    vocab_size : int
        Target vocabulary size when training a new tokeniser from scratch.
        Ignored when loading a pre-trained checkpoint.  Default 30_000.
    min_frequency : int
        Minimum token frequency to include in the vocabulary during
        training.  Default 2.
    pretrained_name_or_path : str | None
        HuggingFace model name (e.g. ``"bert-base-uncased"``) or local
        path to a saved tokeniser directory.  When set, the tokeniser is
        loaded from this checkpoint instead of being trained.
    add_special_tokens : bool
        Prepend ``[CLS]`` and append ``[SEP]`` to each sequence.
        Default ``True``.
    use_lemmas : bool
        When ``True``, tokenise the ``lemmas`` field of each
        ``CleanDocument``; otherwise tokenise the ``tokens`` field.
        Default ``False`` (use surface tokens — lemmas lose morphology
        information that sub-word models can exploit).
    """

    algorithm: str = "wordpiece"
    max_length: int = 512
    vocab_size: int = 30_000
    min_frequency: int = 2
    pretrained_name_or_path: str | None = None
    add_special_tokens: bool = True
    use_lemmas: bool = False


# ---------------------------------------------------------------------------
# Internal fallback tokeniser (whitespace + integer encoding)
# ---------------------------------------------------------------------------

class _WhitespaceIntTokenizer:
    """
    Minimal integer-encoding tokeniser used when the ``tokenizers``
    library is unavailable.

    Vocabulary is built from the corpus during :meth:`fit` and stored
    as a ``{token: id}`` dict.  Unknown tokens map to ``UNK_ID``.
    This tokeniser does NOT perform sub-word splitting; it maps each
    surface token to its integer ID directly.
    """

    def __init__(self, vocab_size: int = 30_000, min_frequency: int = 2) -> None:
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        # Special tokens always occupy the first four IDs
        self._vocab: dict[str, int] = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            CLS_TOKEN: 2,
            SEP_TOKEN: 3,
        }
        self._fitted = False

    @property
    def unk_id(self) -> int:
        return self._vocab[UNK_TOKEN]

    @property
    def pad_id(self) -> int:
        return self._vocab[PAD_TOKEN]

    def fit(self, token_sequences: list[list[str]]) -> None:
        """Build vocabulary from *token_sequences*."""
        from collections import Counter

        freq: Counter[str] = Counter()
        for seq in token_sequences:
            freq.update(seq)

        # Sort by frequency descending, then alphabetically for determinism
        candidates = sorted(
            ((tok, cnt) for tok, cnt in freq.items() if cnt >= self.min_frequency),
            key=lambda x: (-x[1], x[0]),
        )

        next_id = len(self._vocab)
        for token, _ in candidates:
            if token in self._vocab:
                continue
            self._vocab[token] = next_id
            next_id += 1
            if next_id >= self.vocab_size:
                break

        self._fitted = True
        logger.info(
            "Fallback whitespace tokeniser fitted.",
            extra={"vocab_size": len(self._vocab)},
        )

    def encode(
        self,
        tokens: list[str],
        max_length: int = 512,
        add_special_tokens: bool = True,
    ) -> tuple[list[int], list[int]]:
        """
        Encode *tokens* to (token_ids, attention_mask).

        Truncates to ``max_length`` (accounting for special tokens),
        then pads with ``PAD_ID`` to reach ``max_length``.
        """
        if not self._fitted:
            raise RuntimeError("Tokeniser must be fitted before encoding.")

        ids = [self._vocab.get(t, self.unk_id) for t in tokens]

        if add_special_tokens:
            cls_id = self._vocab[CLS_TOKEN]
            sep_id = self._vocab[SEP_TOKEN]
            # Reserve 2 positions for CLS / SEP
            ids = [cls_id] + ids[: max_length - 2] + [sep_id]
        else:
            ids = ids[:max_length]

        real_len = len(ids)
        pad_len = max_length - real_len
        attention_mask = [1] * real_len + [0] * pad_len
        token_ids = ids + [self.pad_id] * pad_len

        return token_ids, attention_mask

    def get_vocab(self) -> dict[str, int]:
        """Return the token→id mapping."""
        return dict(self._vocab)

    def vocab_size_actual(self) -> int:
        """Return the actual number of entries in the vocabulary."""
        return len(self._vocab)


# ---------------------------------------------------------------------------
# Public CorpusTokenizer class
# ---------------------------------------------------------------------------

class CorpusTokenizer:
    """
    High-level tokeniser that wraps either the HuggingFace ``tokenizers``
    library (WordPiece / BPE) or the built-in whitespace fallback.

    Lifecycle
    ---------
    1. Construct with a :class:`TokenizerConfig`.
    2. Call :meth:`fit` **or** :meth:`load` to initialise the vocabulary.
    3. Call :meth:`tokenize` to encode a list of ``CleanDocument`` objects.
    4. Call :meth:`save` to persist the trained tokeniser for reuse.

    Parameters
    ----------
    config : TokenizerConfig | None
        Tokeniser configuration.  Defaults to ``TokenizerConfig()`` with
        WordPiece algorithm and max_length=512.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config: TokenizerConfig = config or TokenizerConfig()
        self._hf_tokenizer: Any = None       # HuggingFace Tokenizer object
        self._fallback: _WhitespaceIntTokenizer | None = None
        self._fitted: bool = False

        logger.debug(
            "CorpusTokenizer created.",
            extra={
                "algorithm": self.config.algorithm,
                "max_length": self.config.max_length,
                "vocab_size": self.config.vocab_size,
                "hf_available": _HF_AVAILABLE,
            },
        )

    # ------------------------------------------------------------------
    # Fit / Load
    # ------------------------------------------------------------------

    def fit(self, documents: list[CleanDocument]) -> "CorpusTokenizer":
        """
        Train the tokeniser vocabulary from *documents*.

        When ``config.pretrained_name_or_path`` is set, this method
        delegates to :meth:`load` instead of training.

        Parameters
        ----------
        documents : list[CleanDocument]
            Pre-processed documents whose tokens / lemmas provide the
            training corpus.

        Returns
        -------
        CorpusTokenizer
            Self, for method chaining.
        """
        if self.config.pretrained_name_or_path:
            return self.load(self.config.pretrained_name_or_path)

        token_sequences = self._extract_token_sequences(documents)

        if _HF_AVAILABLE and self.config.algorithm != "whitespace":
            self._fit_hf(token_sequences)
        else:
            if not _HF_AVAILABLE and self.config.algorithm != "whitespace":
                logger.warning(
                    "HuggingFace 'tokenizers' library not available; "
                    "falling back to whitespace tokeniser.",
                    extra={"requested_algorithm": self.config.algorithm},
                )
            self._fit_fallback(token_sequences)

        self._fitted = True
        logger.info(
            "CorpusTokenizer fitted.",
            extra={
                "algorithm": self.config.algorithm,
                "doc_count": len(documents),
                "actual_vocab_size": self.vocab_size,
            },
        )
        return self

    def load(self, path_or_name: str) -> "CorpusTokenizer":
        """
        Load a pre-trained tokeniser from a local path or HuggingFace Hub.

        Parameters
        ----------
        path_or_name : str
            Local directory containing a ``tokenizer.json`` file, or a
            HuggingFace model identifier (e.g. ``"bert-base-uncased"``).

        Returns
        -------
        CorpusTokenizer
            Self, for method chaining.

        Raises
        ------
        RuntimeError
            If the ``tokenizers`` library is unavailable.
        FileNotFoundError
            If *path_or_name* is a local path that does not exist.
        """
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "Loading pre-trained tokenisers requires the 'tokenizers' "
                "library: pip install tokenizers"
            )

        local_path = Path(path_or_name)
        if local_path.exists():
            tokenizer_file = local_path / "tokenizer.json"
            if not tokenizer_file.exists():
                raise FileNotFoundError(
                    f"No tokenizer.json found at {local_path}. "
                    "Ensure the directory contains a saved HF tokeniser."
                )
            self._hf_tokenizer = _HFTokenizer.from_file(str(tokenizer_file))
        else:
            # Attempt to load from HuggingFace Hub via AutoTokenizer
            try:
                from transformers import AutoTokenizer  # type: ignore[import]

                hf_tok = AutoTokenizer.from_pretrained(path_or_name)
                # Wrap in a fast tokenizer if available
                self._hf_tokenizer = hf_tok
            except ImportError as exc:
                raise RuntimeError(
                    "Loading from HuggingFace Hub requires 'transformers': "
                    "pip install transformers"
                ) from exc

        self._fitted = True
        logger.info(
            "Pre-trained tokeniser loaded.",
            extra={"source": path_or_name},
        )
        return self

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def tokenize(
        self,
        documents: list[CleanDocument],
        corpus_id: str | None = None,
        persist: bool = True,
    ) -> list[TokenizedDocument]:
        """
        Encode *documents* and return ``List[TokenizedDocument]``.

        Satisfies the §3.1 contract:
            Input:  ``List[CleanDocument]``
            Output: ``List[TokenizedDocument]``

        Parameters
        ----------
        documents : list[CleanDocument]
            Pre-processed documents from :class:`TextPreprocessor`.
        corpus_id : str | None, optional
            Used for logging; does not affect encoding.
        persist : bool, optional
            When ``True`` (default), writes a ``.jsonl`` snapshot to
            ``data/splits/{corpus_id}_tokens.jsonl``.

        Returns
        -------
        list[TokenizedDocument]
            One ``TokenizedDocument`` per input document, in the same order.

        Raises
        ------
        RuntimeError
            If :meth:`fit` or :meth:`load` has not been called first.
        """
        if not self._fitted:
            raise RuntimeError(
                "Tokeniser must be fitted or loaded before calling tokenize(). "
                "Call fit(documents) or load(path) first."
            )

        logger.info(
            "Tokenizing documents.",
            extra={"doc_count": len(documents), "corpus_id": corpus_id},
        )

        tokenized: list[TokenizedDocument] = []
        for doc in documents:
            try:
                token_ids, attention_mask = self._encode_doc(doc)
                tokenized.append(
                    TokenizedDocument(
                        doc_id=doc.doc_id,
                        token_ids=token_ids,
                        attention_mask=attention_mask,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Failed to tokenize document; using empty sequence.",
                    extra={"doc_id": doc.doc_id, "error": str(exc)},
                )
                empty = [PAD_ID] * self.config.max_length
                tokenized.append(
                    TokenizedDocument(
                        doc_id=doc.doc_id,
                        token_ids=empty,
                        attention_mask=[0] * self.config.max_length,
                    )
                )

        logger.info(
            "Tokenization complete.",
            extra={"doc_count": len(tokenized), "corpus_id": corpus_id},
        )

        if persist and corpus_id:
            self._persist_token_snapshot(tokenized, corpus_id)

        return tokenized

    def encode_text(self, text: str) -> tuple[list[int], list[int]]:
        """
        Encode a raw string directly to (token_ids, attention_mask).

        Useful for inference-time encoding outside the batch pipeline.

        Parameters
        ----------
        text : str
            Raw or pre-cleaned text.

        Returns
        -------
        tuple[list[int], list[int]]
            ``(token_ids, attention_mask)`` of length ``config.max_length``.
        """
        if not self._fitted:
            raise RuntimeError("Tokeniser must be fitted before encoding.")

        tokens = text.lower().split()
        return self._encode_tokens(tokens)

    # ------------------------------------------------------------------
    # Save / load artefacts
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """
        Persist the tokeniser to *directory*.

        - HuggingFace backend → saves ``tokenizer.json`` via HF API.
        - Fallback backend    → saves vocabulary as ``vocab.json`` and
          serialises the full object as ``tokenizer_fallback.pkl``.

        Parameters
        ----------
        directory : str | Path
            Target directory.  Created if it does not exist.

        Returns
        -------
        Path
            The directory where the tokeniser was saved.
        """
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._hf_tokenizer is not None and _HF_AVAILABLE:
            save_path = out_dir / "tokenizer.json"
            if hasattr(self._hf_tokenizer, "save"):
                self._hf_tokenizer.save(str(save_path))
            else:
                # transformers AutoTokenizer
                self._hf_tokenizer.save_pretrained(str(out_dir))
        elif self._fallback is not None:
            vocab_path = out_dir / "vocab.json"
            with vocab_path.open("w", encoding="utf-8") as fh:
                json.dump(self._fallback.get_vocab(), fh, indent=2)
            pkl_path = out_dir / "tokenizer_fallback.pkl"
            with pkl_path.open("wb") as fh:
                pickle.dump(self._fallback, fh)
        else:
            raise RuntimeError("No fitted tokeniser to save.")

        logger.info("Tokeniser saved.", extra={"directory": str(out_dir)})
        return out_dir

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Return the actual vocabulary size of the fitted tokeniser."""
        if self._hf_tokenizer is not None:
            if hasattr(self._hf_tokenizer, "get_vocab_size"):
                return self._hf_tokenizer.get_vocab_size()
            if hasattr(self._hf_tokenizer, "vocab_size"):
                return self._hf_tokenizer.vocab_size
        if self._fallback is not None:
            return self._fallback.vocab_size_actual()
        return 0

    def get_vocab(self) -> dict[str, int]:
        """Return the token → integer ID mapping."""
        if self._hf_tokenizer is not None:
            if hasattr(self._hf_tokenizer, "get_vocab"):
                return self._hf_tokenizer.get_vocab()
        if self._fallback is not None:
            return self._fallback.get_vocab()
        raise RuntimeError("Tokeniser has not been fitted or loaded.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_token_sequences(
        self, documents: list[CleanDocument]
    ) -> list[list[str]]:
        """Return token or lemma lists from *documents* per config."""
        if self.config.use_lemmas:
            return [doc.lemmas for doc in documents]
        return [doc.tokens for doc in documents]

    def _fit_hf(self, token_sequences: list[list[str]]) -> None:
        """Train a HuggingFace WordPiece or BPE tokeniser."""
        algo = self.config.algorithm.lower()

        if algo == "wordpiece":
            model = _WordPiece(unk_token=UNK_TOKEN)
            trainer = _WordPieceTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=[PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN],
            )
        elif algo == "bpe":
            model = _BPE(unk_token=UNK_TOKEN)
            trainer = _BpeTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=[PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN],
            )
        else:
            raise ValueError(
                f"Unknown HF tokeniser algorithm: {algo!r}. "
                "Choose 'wordpiece', 'bpe', or 'whitespace'."
            )

        tokenizer = _HFTokenizer(model)
        tokenizer.pre_tokenizer = _Whitespace()

        # HF trainers expect an iterator of strings
        def _sentence_iter() -> Any:
            for seq in token_sequences:
                yield " ".join(seq)

        tokenizer.train_from_iterator(_sentence_iter(), trainer=trainer)

        # Configure post-processor for CLS/SEP if enabled
        if self.config.add_special_tokens:
            cls_id = tokenizer.token_to_id(CLS_TOKEN)
            sep_id = tokenizer.token_to_id(SEP_TOKEN)
            if cls_id is not None and sep_id is not None:
                tokenizer.post_processor = _TemplateProcessing(
                    single=f"{CLS_TOKEN} $A {SEP_TOKEN}",
                    special_tokens=[
                        (CLS_TOKEN, cls_id),
                        (SEP_TOKEN, sep_id),
                    ],
                )

        # Enable padding and truncation at the model level
        pad_id = tokenizer.token_to_id(PAD_TOKEN) or 0
        tokenizer.enable_padding(
            pad_id=pad_id,
            pad_token=PAD_TOKEN,
            length=self.config.max_length,
        )
        tokenizer.enable_truncation(max_length=self.config.max_length)

        self._hf_tokenizer = tokenizer

    def _fit_fallback(self, token_sequences: list[list[str]]) -> None:
        """Train the internal whitespace fallback tokeniser."""
        self._fallback = _WhitespaceIntTokenizer(
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
        )
        self._fallback.fit(token_sequences)

    def _encode_doc(self, doc: CleanDocument) -> tuple[list[int], list[int]]:
        """Encode a single CleanDocument to (token_ids, attention_mask)."""
        tokens = doc.lemmas if self.config.use_lemmas else doc.tokens
        return self._encode_tokens(tokens)

    def _encode_tokens(
        self, tokens: list[str]
    ) -> tuple[list[int], list[int]]:
        """Encode a list of string tokens to (token_ids, attention_mask)."""
        if self._hf_tokenizer is not None:
            text = " ".join(tokens)
            if hasattr(self._hf_tokenizer, "encode"):
                # Native HF tokenizers.Tokenizer
                encoding = self._hf_tokenizer.encode(text)
                token_ids: list[int] = encoding.ids
                attention_mask: list[int] = encoding.attention_mask
            else:
                # transformers AutoTokenizer
                enc = self._hf_tokenizer(
                    text,
                    max_length=self.config.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors=None,
                )
                token_ids = list(enc["input_ids"])
                attention_mask = list(enc["attention_mask"])
            return token_ids, attention_mask

        if self._fallback is not None:
            return self._fallback.encode(
                tokens,
                max_length=self.config.max_length,
                add_special_tokens=self.config.add_special_tokens,
            )

        raise RuntimeError("No backend available; call fit() first.")

    def _persist_token_snapshot(
        self, documents: list[TokenizedDocument], corpus_id: str
    ) -> None:
        """Write tokenized docs to ``data/splits/{corpus_id}_tokens.jsonl``."""
        _SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _SPLITS_DIR / f"{corpus_id}_tokens.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for doc in documents:
                fh.write(json.dumps(doc.to_dict()) + "\n")
        logger.info(
            "Token snapshot persisted.",
            extra={"path": str(out_path), "doc_count": len(documents)},
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def tokenize_corpus(
    documents: list[CleanDocument],
    config: TokenizerConfig | None = None,
    corpus_id: str | None = None,
    persist: bool = True,
) -> tuple[list[TokenizedDocument], CorpusTokenizer]:
    """
    Convenience wrapper: fit a new :class:`CorpusTokenizer` and tokenize
    *documents* in one call.

    Parameters
    ----------
    documents : list[CleanDocument]
        Pre-processed documents from the text preprocessor.
    config : TokenizerConfig | None, optional
        Tokeniser configuration; defaults to WordPiece with max_length=512.
    corpus_id : str | None, optional
        Corpus identifier for artifact naming and logging.
    persist : bool, optional
        Write snapshot to ``data/splits/``.  Default ``True``.

    Returns
    -------
    tuple[list[TokenizedDocument], CorpusTokenizer]
        The tokenized documents and the fitted tokeniser instance (which
        should be passed to :mod:`src.ingestion.vocab_builder`).
    """
    tokenizer = CorpusTokenizer(config=config)
    tokenizer.fit(documents)
    tokenized = tokenizer.tokenize(documents, corpus_id=corpus_id, persist=persist)
    return tokenized, tokenizer