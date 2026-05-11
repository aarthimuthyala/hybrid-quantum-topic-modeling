"""
src/ingestion/corpus_loader.py
==============================
Corpus loading module for the HQC Topic Modeling project.

Responsibility (§3.1):
    Load raw corpora from disk or URL; return a standardised
    ``List[CorpusDocument]``.

Inputs  (§3.1 contract): ``file_path: str`` or ``url: str``
Outputs (§3.1 contract): ``List[CorpusDocument]``

Artifact produced (§5.1 stage 01):
    ``data/raw/{corpus_id}.jsonl``

Data versioning (§7.2):
    SHA-256 hashes are computed for every loaded corpus and stored in
    ``data/manifest.json``.  On subsequent loads the hash is re-validated
    to guarantee reproducibility.

Supported source formats:
    - ``.jsonl``  — one CorpusDocument JSON object per line
    - ``.json``   — JSON array of CorpusDocument objects
    - ``.txt``    — plain text, one document per line (doc_id auto-generated)
    - ``.csv``    — CSV with at minimum a ``text`` column; optional ``id``
                    and ``source`` columns
    - URL         — HTTP/HTTPS endpoint returning JSON (array or newline-delimited)

All loaded documents are validated against the CorpusDocument schema via
``shared.validator.validate_corpus_doc`` before being returned.

Blueprint constraints honoured:
    - Full package imports only (§2 RULE — Flat Imports Only)
    - All hyperparameters sourced from config/ (§9 — never hardcoded)
    - Structured JSON logging via shared.logger (§8.1)
    - SHA-256 hash tracking per §7.2 Data Versioning Policy
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from shared.logger import get_logger
from shared.validator import CorpusDocument, CorpusValidationError, validate_corpus_doc

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project-root and data-directory paths
# ---------------------------------------------------------------------------
# src/ingestion/corpus_loader.py → src/ingestion/ → src/ → project_root/
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
_RAW_DIR: Path = _PROJECT_ROOT / "data" / "raw"
_MANIFEST_PATH: Path = _PROJECT_ROOT / "data" / "manifest.json"

# ---------------------------------------------------------------------------
# Constants (all non-configurable values — numeric limits are in config/)
# ---------------------------------------------------------------------------
_URL_TIMEOUT_SECONDS: int = 30  # network timeout for HTTP fetches
_JSONL_ENCODING: str = "utf-8"
_CSV_TEXT_COLUMNS: tuple[str, ...] = ("text", "raw_text", "content", "body")
_CSV_ID_COLUMNS: tuple[str, ...] = ("id", "doc_id", "docid")
_CSV_SOURCE_COLUMNS: tuple[str, ...] = ("source", "origin", "dataset")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_sha256(data: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def _update_manifest(corpus_id: str, sha256: str, doc_count: int) -> None:
    """
    Persist *sha256* and *doc_count* for *corpus_id* to ``data/manifest.json``.

    The manifest is a flat JSON object mapping corpus_id → record.
    Creates the file if absent; merges without overwriting other entries.
    """
    manifest: dict[str, Any] = {}
    if _MANIFEST_PATH.exists():
        try:
            with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Could not read manifest; will overwrite.",
                extra={"path": str(_MANIFEST_PATH), "error": str(exc)},
            )

    manifest[corpus_id] = {"sha256": sha256, "doc_count": doc_count}

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    logger.debug(
        "Manifest updated.",
        extra={"corpus_id": corpus_id, "sha256": sha256[:12] + "…"},
    )


def _verify_manifest(corpus_id: str, sha256: str) -> None:
    """
    If a prior hash exists in the manifest for *corpus_id*, compare it
    to *sha256* and raise ``ValueError`` on mismatch.

    A mismatch indicates the source file has changed since it was first
    ingested, which violates the Data Versioning Policy (§7.2).
    """
    if not _MANIFEST_PATH.exists():
        return  # First run — no prior hash to compare

    try:
        with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
            manifest: dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return  # Unreadable manifest — skip verification

    prior = manifest.get(corpus_id, {}).get("sha256")
    if prior and prior != sha256:
        raise ValueError(
            f"SHA-256 mismatch for corpus '{corpus_id}': "
            f"stored={prior}, computed={sha256}. "
            "The source file has changed since the last ingestion. "
            "Update data/manifest.json or re-run the corpus download script."
        )


def _infer_corpus_id(path: Path) -> str:
    """
    Derive a corpus_id from a file path stem.

    Strategy:
      - Strip numeric suffixes and UUID-like segments.
      - Map known dataset names to their standardised corpus_id (§7.1).
      - Fallback to the lowercased stem.
    """
    stem = path.stem.lower()
    # Map canonical dataset names (§7.1)
    _KNOWN = {
        "20newsgroups": "20ng",
        "20news": "20ng",
        "reuters": "reuters",
        "bbc": "bbc",
        "bbc_news": "bbc",
        "ag_news": "agnews_mini",
        "agnews": "agnews_mini",
    }
    for key, cid in _KNOWN.items():
        if key in stem:
            return cid
    # Strip trailing numeric/UUID segments
    clean = re.sub(r"[_-]?[0-9a-f\-]{8,}$", "", stem)
    return clean or stem


def _parse_jsonl(raw_bytes: bytes, source_label: str) -> list[dict[str, Any]]:
    """
    Parse newline-delimited JSON bytes into a list of dicts.

    Skips blank lines and lines starting with ``//`` (JSON-comment convention).
    """
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        raw_bytes.decode(_JSONL_ENCODING).splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        try:
            obj = json.loads(stripped)
            if not isinstance(obj, dict):
                logger.warning(
                    "Skipping non-object JSON line.",
                    extra={"source": source_label, "line": lineno},
                )
                continue
            records.append(obj)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Skipping malformed JSON line.",
                extra={"source": source_label, "line": lineno, "error": str(exc)},
            )
    return records


def _parse_json_array(raw_bytes: bytes, source_label: str) -> list[dict[str, Any]]:
    """Parse a JSON array of objects from *raw_bytes*."""
    try:
        data = json.loads(raw_bytes.decode(_JSONL_ENCODING))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in '{source_label}': {exc}"
        ) from exc
    if not isinstance(data, list):
        raise ValueError(
            f"Expected a JSON array in '{source_label}', got {type(data).__name__}."
        )
    return [d for d in data if isinstance(d, dict)]


def _parse_txt(raw_bytes: bytes, source_label: str) -> list[dict[str, Any]]:
    """
    Parse a plain-text file where each non-blank line is one document.
    Auto-generates ``doc_id`` values as ``{source_label}_{lineno:06d}``.
    """
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        raw_bytes.decode(_JSONL_ENCODING).splitlines(), start=1
    ):
        text = line.strip()
        if not text:
            continue
        records.append(
            {
                "doc_id": f"{source_label}_{lineno:06d}",
                "raw_text": text,
                "metadata": {},
                "source": source_label,
            }
        )
    return records


def _parse_csv(raw_bytes: bytes, source_label: str) -> list[dict[str, Any]]:
    """
    Parse a CSV file into CorpusDocument-shaped dicts.

    Column discovery:
      - text   → first column whose lowercase name is in ``_CSV_TEXT_COLUMNS``
      - doc_id → first column in ``_CSV_ID_COLUMNS`` (auto-generated if absent)
      - source → first column in ``_CSV_SOURCE_COLUMNS`` (defaults to source_label)
    """
    text_stream = io.StringIO(raw_bytes.decode(_JSONL_ENCODING))
    reader = csv.DictReader(text_stream)

    if reader.fieldnames is None:
        raise ValueError(f"CSV file '{source_label}' appears to be empty or has no header.")

    lowered: dict[str, str] = {f.lower(): f for f in reader.fieldnames}

    # Resolve column names
    text_col = next(
        (lowered[c] for c in _CSV_TEXT_COLUMNS if c in lowered), None
    )
    id_col = next((lowered[c] for c in _CSV_ID_COLUMNS if c in lowered), None)
    src_col = next(
        (lowered[c] for c in _CSV_SOURCE_COLUMNS if c in lowered), None
    )

    if text_col is None:
        raise ValueError(
            f"CSV '{source_label}' has no recognised text column. "
            f"Expected one of {_CSV_TEXT_COLUMNS}. "
            f"Found: {list(reader.fieldnames)}"
        )

    records: list[dict[str, Any]] = []
    for rownum, row in enumerate(reader, start=1):
        raw_text = row.get(text_col, "").strip()
        if not raw_text:
            continue  # skip blank rows silently
        doc_id = (
            row.get(id_col, "").strip()
            if id_col
            else f"{source_label}_{rownum:06d}"
        ) or f"{source_label}_{rownum:06d}"
        source = (
            row.get(src_col, "").strip()
            if src_col
            else source_label
        ) or source_label
        # Remaining columns become metadata
        metadata = {
            k: v
            for k, v in row.items()
            if k not in {text_col, id_col, src_col}
        }
        records.append(
            {
                "doc_id": doc_id,
                "raw_text": raw_text,
                "metadata": metadata,
                "source": source,
            }
        )
    return records


def _dicts_to_corpus_documents(
    records: list[dict[str, Any]],
    source_label: str,
    skip_invalid: bool = False,
) -> list[CorpusDocument]:
    """
    Convert raw dicts to validated :class:`CorpusDocument` instances.

    Parameters
    ----------
    records : list[dict]
        Raw document dicts.
    source_label : str
        Used for logging and as fallback ``source`` field.
    skip_invalid : bool
        If ``True``, log a warning and skip documents that fail validation
        instead of raising.  Defaults to ``False`` (strict mode).

    Returns
    -------
    list[CorpusDocument]
        Validated corpus documents.
    """
    documents: list[CorpusDocument] = []
    skipped = 0

    for idx, record in enumerate(records):
        # Inject source if missing
        if "source" not in record or not record["source"]:
            record = {**record, "source": source_label}
        # Auto-generate doc_id if missing
        if "doc_id" not in record or not record["doc_id"]:
            record = {**record, "doc_id": f"{source_label}_{idx:06d}"}
        # Default metadata to empty dict
        if "metadata" not in record:
            record = {**record, "metadata": {}}

        try:
            documents.append(validate_corpus_doc(record))
        except CorpusValidationError as exc:
            skipped += 1
            if skip_invalid:
                logger.warning(
                    "Skipping invalid document.",
                    extra={
                        "doc_id": exc.doc_id,
                        "source": source_label,
                        "errors": exc.errors,
                    },
                )
            else:
                raise

    if skipped:
        logger.info(
            "Skipped invalid documents during loading.",
            extra={"skipped": skipped, "source": source_label},
        )
    return documents


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_corpus_from_file(
    file_path: str,
    corpus_id: str | None = None,
    skip_invalid: bool = False,
    persist_raw: bool = True,
) -> list[CorpusDocument]:
    """
    Load a corpus from a local file and return validated
    :class:`CorpusDocument` instances.

    Satisfies the §3.1 contract:
        Input:  ``file_path: str``
        Output: ``List[CorpusDocument]``

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the corpus file.  Supported
        extensions: ``.jsonl``, ``.json``, ``.txt``, ``.csv``.
    corpus_id : str | None, optional
        Explicit corpus identifier.  If ``None``, inferred from the
        file name stem (see :func:`_infer_corpus_id`).
    skip_invalid : bool, optional
        When ``True``, malformed documents are logged and skipped rather
        than raising :class:`~shared.validator.CorpusValidationError`.
        Defaults to ``False`` (strict mode).
    persist_raw : bool, optional
        When ``True`` (default), writes a validated ``.jsonl`` snapshot
        to ``data/raw/{corpus_id}.jsonl`` and updates
        ``data/manifest.json`` with the SHA-256 hash.

    Returns
    -------
    list[CorpusDocument]
        Non-empty list of validated corpus documents.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    ValueError
        If the file format is unsupported or if a SHA-256 hash mismatch
        is detected against ``data/manifest.json``.
    CorpusValidationError
        If a document fails schema validation and *skip_invalid* is
        ``False``.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file, got a directory: {path}")

    corpus_id = corpus_id or _infer_corpus_id(path)
    suffix = path.suffix.lower()

    logger.info(
        "Loading corpus from file.",
        extra={"file_path": str(path), "corpus_id": corpus_id, "format": suffix},
    )

    raw_bytes = path.read_bytes()
    sha256 = _compute_sha256(raw_bytes)

    # Verify against prior manifest entry (§7.2 Data Versioning Policy)
    _verify_manifest(corpus_id, sha256)

    # Dispatch to format-specific parser
    if suffix == ".jsonl":
        records = _parse_jsonl(raw_bytes, corpus_id)
    elif suffix == ".json":
        records = _parse_json_array(raw_bytes, corpus_id)
    elif suffix == ".txt":
        records = _parse_txt(raw_bytes, corpus_id)
    elif suffix == ".csv":
        records = _parse_csv(raw_bytes, corpus_id)
    else:
        raise ValueError(
            f"Unsupported corpus file format: '{suffix}'. "
            "Supported: .jsonl, .json, .txt, .csv"
        )

    documents = _dicts_to_corpus_documents(records, corpus_id, skip_invalid)

    logger.info(
        "Corpus loaded successfully.",
        extra={"corpus_id": corpus_id, "doc_count": len(documents)},
    )

    if persist_raw:
        _persist_raw_snapshot(documents, corpus_id, sha256)

    return documents


def load_corpus_from_url(
    url: str,
    corpus_id: str | None = None,
    skip_invalid: bool = False,
    persist_raw: bool = True,
) -> list[CorpusDocument]:
    """
    Load a corpus from an HTTP/HTTPS URL and return validated
    :class:`CorpusDocument` instances.

    Satisfies the §3.1 contract:
        Input:  ``url: str``
        Output: ``List[CorpusDocument]``

    The remote endpoint must return either:
    - A JSON array of CorpusDocument objects (Content-Type: application/json), or
    - Newline-delimited JSON (Content-Type: application/x-ndjson or text/plain).

    Parameters
    ----------
    url : str
        HTTP or HTTPS URL pointing to the corpus data.
    corpus_id : str | None, optional
        Explicit corpus identifier.  If ``None``, derived from the URL
        path's last segment.
    skip_invalid : bool, optional
        Skip malformed documents instead of raising.  Defaults to ``False``.
    persist_raw : bool, optional
        Write a ``.jsonl`` snapshot to ``data/raw/``.  Defaults to ``True``.

    Returns
    -------
    list[CorpusDocument]
        Validated corpus documents.

    Raises
    ------
    ValueError
        If the URL scheme is not HTTP/HTTPS.
    URLError
        On network failures (propagated from :mod:`urllib`).
    """
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid URL scheme: {url!r}. Only http:// and https:// are supported."
        )

    # Derive corpus_id from URL if not explicit
    if corpus_id is None:
        url_path = url.split("?")[0].rstrip("/")
        stem = url_path.split("/")[-1]
        corpus_id = re.sub(r"\.[a-zA-Z0-9]+$", "", stem).lower() or "url_corpus"

    logger.info(
        "Fetching corpus from URL.",
        extra={"url": url, "corpus_id": corpus_id},
    )

    try:
        with urlopen(url, timeout=_URL_TIMEOUT_SECONDS) as response:  # noqa: S310
            raw_bytes: bytes = response.read()
            content_type: str = response.headers.get("Content-Type", "")
    except URLError as exc:
        logger.error(
            "Failed to fetch corpus URL.",
            extra={"url": url, "error": str(exc)},
        )
        raise

    sha256 = _compute_sha256(raw_bytes)

    # Parse based on content type
    if "json" in content_type.lower():
        # Attempt JSON array first; fall back to JSONL
        try:
            records = _parse_json_array(raw_bytes, corpus_id)
        except ValueError:
            records = _parse_jsonl(raw_bytes, corpus_id)
    else:
        # Treat as JSONL by default (most NLP datasets use this format)
        records = _parse_jsonl(raw_bytes, corpus_id)

    documents = _dicts_to_corpus_documents(records, corpus_id, skip_invalid)

    logger.info(
        "Corpus fetched from URL successfully.",
        extra={"url": url, "corpus_id": corpus_id, "doc_count": len(documents)},
    )

    if persist_raw:
        _persist_raw_snapshot(documents, corpus_id, sha256)

    return documents


def load_corpus(
    file_path: str | None = None,
    url: str | None = None,
    corpus_id: str | None = None,
    skip_invalid: bool = False,
    persist_raw: bool = True,
) -> list[CorpusDocument]:
    """
    Unified entry-point that dispatches to :func:`load_corpus_from_file`
    or :func:`load_corpus_from_url` depending on which argument is supplied.

    Exactly one of *file_path* or *url* must be provided.

    Parameters
    ----------
    file_path : str | None
        Local file path.  Mutually exclusive with *url*.
    url : str | None
        Remote URL.  Mutually exclusive with *file_path*.
    corpus_id : str | None, optional
        Explicit corpus identifier; inferred if ``None``.
    skip_invalid : bool, optional
        Skip malformed documents.  Defaults to ``False``.
    persist_raw : bool, optional
        Write raw snapshot to ``data/raw/``.  Defaults to ``True``.

    Returns
    -------
    list[CorpusDocument]
        Validated corpus documents.

    Raises
    ------
    ValueError
        If both or neither of *file_path* / *url* are provided.
    """
    if file_path is not None and url is not None:
        raise ValueError(
            "Provide exactly one of 'file_path' or 'url', not both."
        )
    if file_path is None and url is None:
        raise ValueError(
            "One of 'file_path' or 'url' must be provided."
        )

    if file_path is not None:
        return load_corpus_from_file(
            file_path,
            corpus_id=corpus_id,
            skip_invalid=skip_invalid,
            persist_raw=persist_raw,
        )
    return load_corpus_from_url(
        url,  # type: ignore[arg-type]  # already guarded above
        corpus_id=corpus_id,
        skip_invalid=skip_invalid,
        persist_raw=persist_raw,
    )


# ---------------------------------------------------------------------------
# Internal persistence helper
# ---------------------------------------------------------------------------

def _persist_raw_snapshot(
    documents: list[CorpusDocument],
    corpus_id: str,
    sha256: str,
) -> None:
    """
    Write *documents* to ``data/raw/{corpus_id}.jsonl`` and update
    ``data/manifest.json`` (§5.1 stage 01 artifact, §7.2 hash tracking).

    Each line is a JSON-serialised CorpusDocument dict.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RAW_DIR / f"{corpus_id}.jsonl"

    with out_path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.model_dump()) + "\n")

    _update_manifest(corpus_id, sha256, len(documents))

    logger.info(
        "Raw snapshot persisted.",
        extra={
            "path": str(out_path),
            "corpus_id": corpus_id,
            "doc_count": len(documents),
            "sha256": sha256[:12] + "…",
        },
    )


def generate_doc_id(prefix: str = "doc") -> str:
    """
    Generate a unique document ID using UUID-4.

    Used internally when a source file does not provide identifiers and
    auto-numbering is insufficient (e.g. concurrent ingestion workers).

    Parameters
    ----------
    prefix : str, optional
        Short label prepended to the UUID fragment.  Defaults to ``"doc"``.

    Returns
    -------
    str
        A string of the form ``"{prefix}_{uuid4_hex[:12]}"``.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"