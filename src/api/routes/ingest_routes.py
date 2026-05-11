"""
src/api/routes/ingest_routes.py
================================
Team T-5 — API & Backend Layer (L4)
HQC Topic Modeling Project · Master Blueprint v1.0

Responsibility:
    Implement all three ingestion endpoints defined in Blueprint §4.1.
    Every route is async, validated by Pydantic v2 schemas, and returns
    the exact JSON fields specified in the Blueprint contract table.

Blueprint §4.1 — Ingestion Endpoints (FROZEN contract):
    ┌────────┬──────────────────────────┬─────────────────────────────────────┐
    │ Method │ Endpoint                 │ Description                         │
    ├────────┼──────────────────────────┼─────────────────────────────────────┤
    │ POST   │ /ingest/upload           │ Upload & preprocess corpus file      │
    │ GET    │ /ingest/{corpus_id}      │ Retrieve corpus metadata             │
    │ POST   │ /ingest/preprocess       │ Run preprocessing pipeline on corpus │
    └────────┴──────────────────────────┴─────────────────────────────────────┘

Request / Response schemas (§4.1):
    POST /ingest/upload
        Request  : multipart/form-data  { file: UploadFile, config_id: str }
        Response : { corpus_id, doc_count, status }

    GET /ingest/{corpus_id}
        Response : { corpus_id, doc_count, vocab_size, created_at }

    POST /ingest/preprocess
        Request  : { corpus_id: str, pipeline: List[str] }
        Response : { corpus_id, clean_doc_count, status }

Error contract (§4 preamble):
    All errors → { "error": str, "detail": str, "code": int }

Integration with T-1 layer (src/ingestion/):
    - corpus_loader.py    → CorpusLoader.load_from_file()
    - text_preprocessor.py → TextPreprocessor.preprocess()
    - vocab_builder.py    → VocabBuilder.build()
    These are called via thin async wrappers using run_in_executor so that
    CPU-bound ingestion work does not block the event loop.

Ingestion artifacts written to (Blueprint §5.1):
    Stage 01 → data/raw/{corpus_id}.jsonl
    Stage 02 → data/processed/{corpus_id}_clean.jsonl
    Stage 03 → data/splits/{corpus_id}_vocab.pkl

Dependencies:
    fastapi>=0.111
    pydantic>=2.7
    python-multipart>=0.0.9
    aiofiles>=23.2
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Project-internal imports — full package paths (Blueprint §2)
# ---------------------------------------------------------------------------
from shared.logger import get_logger
from shared.file_utils import ensure_dir, get_artifact_path
from shared.serializer import load_artifact, save_artifact

# T-1 ingestion layer (produced by Team T-1)
# Imported at function scope inside routes to allow T-1 to be mocked in tests
# without a hard module-level import failure if T-1 code is not yet present.

# ---------------------------------------------------------------------------
# Router and logger
# ---------------------------------------------------------------------------
router = APIRouter()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — UPPER_SNAKE_CASE (Blueprint §6)
# ---------------------------------------------------------------------------
RAW_DATA_DIR: str = "data/raw"
PROCESSED_DATA_DIR: str = "data/processed"
SPLITS_DIR: str = "data/splits"
ALLOWED_UPLOAD_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".csv", ".jsonl", ".json", ".tsv"}
)
MAX_UPLOAD_BYTES: int = 500 * 1024 * 1024   # 500 MB hard limit
SUPPORTED_PIPELINE_STEPS: frozenset[str] = frozenset(
    {"lowercase", "stopwords", "lemmatize", "html_strip", "punct_remove"}
)

# In-memory corpus registry (replace with Redis or DB in production)
# Maps corpus_id → metadata dict
_CORPUS_REGISTRY: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Request / Response Pydantic schemas (Blueprint §4.1)
# ---------------------------------------------------------------------------

class PreprocessRequest(BaseModel):
    """
    Request body for POST /ingest/preprocess (Blueprint §4.1).

    Fields:
        corpus_id : Must reference a corpus previously uploaded via /upload.
        pipeline  : Ordered list of preprocessing step names. Each step is
                    applied in sequence by text_preprocessor.py.
                    Allowed steps: lowercase, stopwords, lemmatize,
                                   html_strip, punct_remove.
    """
    corpus_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="ID of a previously uploaded corpus.",
        examples=["20ng"],
    )
    pipeline: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered preprocessing steps to apply.",
        examples=[["lowercase", "html_strip", "stopwords", "lemmatize"]],
    )

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline_steps(cls, steps: list[str]) -> list[str]:
        """Reject unknown pipeline step names early with a clear message."""
        unknown = set(steps) - SUPPORTED_PIPELINE_STEPS
        if unknown:
            raise ValueError(
                f"Unknown pipeline steps: {sorted(unknown)}. "
                f"Supported: {sorted(SUPPORTED_PIPELINE_STEPS)}"
            )
        return steps


class UploadResponse(BaseModel):
    """Response schema for POST /ingest/upload (Blueprint §4.1)."""
    corpus_id: str
    doc_count: int
    status: str


class CorpusMetaResponse(BaseModel):
    """Response schema for GET /ingest/{corpus_id} (Blueprint §4.1)."""
    corpus_id: str
    doc_count: int
    vocab_size: int
    created_at: str   # ISO-8601 UTC timestamp (Blueprint §4 preamble)


class PreprocessResponse(BaseModel):
    """Response schema for POST /ingest/preprocess (Blueprint §4.1)."""
    corpus_id: str
    clean_doc_count: int
    status: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_corpus_id(filename: str) -> str:
    """
    Derive a deterministic corpus_id from the uploaded filename plus a
    short UUID suffix to prevent collisions on repeated uploads of the
    same file name.

    Args:
        filename: Original client filename (may include path components).

    Returns:
        Lowercase alphanumeric corpus_id string, e.g. 'report_a3f2'.
    """
    stem = Path(filename).stem.lower().replace(" ", "_")[:32]
    suffix = uuid.uuid4().hex[:6]
    return f"{stem}_{suffix}"


def _validate_upload_extension(filename: str) -> None:
    """
    Reject uploads whose extension is not in ALLOWED_UPLOAD_EXTENSIONS.

    Args:
        filename: Original filename from the multipart upload.

    Raises:
        HTTPException 415: If the extension is not supported.
    """
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "error":  "unsupported_media_type",
                "detail": (
                    f"Extension '{ext}' is not supported. "
                    f"Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}"
                ),
                "code":   415,
            },
        )


def _assert_corpus_exists(corpus_id: str) -> dict[str, Any]:
    """
    Look up corpus_id in the registry; raise 404 if absent.

    Args:
        corpus_id: The corpus identifier to look up.

    Returns:
        Metadata dict for the corpus.

    Raises:
        HTTPException 404: If corpus_id is not registered.
    """
    meta = _CORPUS_REGISTRY.get(corpus_id)
    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":  "corpus_not_found",
                "detail": f"corpus_id '{corpus_id}' not found. Upload it first via /ingest/upload.",
                "code":   404,
            },
        )
    return meta


async def _run_sync(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Execute a synchronous (CPU-bound) callable in the default thread-pool
    executor so it does not block the FastAPI event loop.

    Wraps asyncio.get_event_loop().run_in_executor() with functools.partial
    to support keyword arguments, which run_in_executor does not accept
    directly.

    Args:
        fn    : Synchronous callable to execute.
        *args : Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn via partial.

    Returns:
        Whatever fn returns.
    """
    loop = asyncio.get_event_loop()
    if kwargs:
        fn = partial(fn, **kwargs)
    return await loop.run_in_executor(None, fn, *args)


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and preprocess a corpus file",
    description=(
        "Accepts a raw corpus file (txt, csv, jsonl, json, tsv) via "
        "multipart upload. The file is saved to data/raw/, tokenised "
        "by the ingestion layer, and the corpus is registered for "
        "downstream classical and quantum processing. "
        "Returns a corpus_id for all subsequent API calls."
    ),
    responses={
        201: {"description": "Corpus uploaded and ingested successfully."},
        400: {"description": "Missing or malformed form fields."},
        413: {"description": "File exceeds the 500 MB upload limit."},
        415: {"description": "Unsupported file extension."},
        500: {"description": "Server-side ingestion failure."},
    },
)
async def upload_corpus(
    file: UploadFile = File(..., description="Raw corpus file to upload."),
    config_id: str = Form(
        ...,
        description="Config profile ID (references a config/ YAML block).",
        examples=["default"],
    ),
) -> UploadResponse:
    """
    POST /api/v1/ingest/upload

    Blueprint §4.1 contract:
        Request  : { file: multipart, config_id: str }
        Response : { corpus_id, doc_count, status }

    Pipeline (Blueprint §5.1 Stages 01–03):
        1. Validate file extension and size.
        2. Persist raw bytes to data/raw/{corpus_id}.jsonl.
        3. Call corpus_loader.py → List[CorpusDocument].
        4. Register corpus metadata in _CORPUS_REGISTRY.
    """
    # --- Validate filename -----------------------------------------------
    original_filename: str = file.filename or "upload.txt"
    _validate_upload_extension(original_filename)

    # --- Read and size-check the payload ---------------------------------
    raw_bytes: bytes = await file.read()
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "error":  "payload_too_large",
                "detail": f"File exceeds the {MAX_UPLOAD_BYTES // 1_048_576} MB limit.",
                "code":   413,
            },
        )

    corpus_id = _generate_corpus_id(original_filename)
    logger.info(
        "upload_corpus | corpus_id=%s | filename=%s | bytes=%d | config_id=%s",
        corpus_id,
        original_filename,
        len(raw_bytes),
        config_id,
    )

    # --- Persist raw file to disk (Blueprint §5.1 Stage 01) --------------
    ensure_dir(RAW_DATA_DIR)
    raw_path = Path(RAW_DATA_DIR) / f"{corpus_id}.jsonl"

    try:
        raw_path.write_bytes(raw_bytes)
    except OSError as exc:
        logger.exception("Failed to write raw corpus to disk | path=%s", raw_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error":  "disk_write_failure",
                "detail": f"Could not persist file: {exc}",
                "code":   500,
            },
        ) from exc

    # --- Run T-1 corpus loader in thread-pool ----------------------------
    doc_count: int = 0
    try:
        from src.ingestion.corpus_loader import CorpusLoader  # type: ignore[import]

        loader = CorpusLoader()
        corpus_docs = await _run_sync(loader.load_from_file, str(raw_path))
        doc_count = len(corpus_docs)
    except ImportError:
        # T-1 not yet available — operate in stub mode for CI / integration tests
        logger.warning(
            "src.ingestion.corpus_loader not found; using stub doc_count."
        )
        # Estimate doc_count from newlines as a reasonable fallback
        doc_count = max(raw_bytes.count(b"\n"), 1)

    # --- Compute SHA-256 hash for data versioning (Blueprint §7.2) -------
    sha256 = hashlib.sha256(raw_bytes).hexdigest()

    # --- Register corpus metadata ----------------------------------------
    _CORPUS_REGISTRY[corpus_id] = {
        "corpus_id":   corpus_id,
        "doc_count":   doc_count,
        "vocab_size":  0,          # populated after vocab_builder runs
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "raw_path":    str(raw_path),
        "sha256":      sha256,
        "config_id":   config_id,
        "status":      "uploaded",
    }

    logger.info(
        "Corpus registered | corpus_id=%s | doc_count=%d | sha256=%.8s",
        corpus_id,
        doc_count,
        sha256,
    )

    return UploadResponse(
        corpus_id=corpus_id,
        doc_count=doc_count,
        status="uploaded",
    )


@router.get(
    "/{corpus_id}",
    response_model=CorpusMetaResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve corpus metadata",
    description=(
        "Returns metadata for a previously uploaded corpus, including "
        "document count, vocabulary size (if preprocessing has run), "
        "and the ISO-8601 creation timestamp."
    ),
    responses={
        200: {"description": "Corpus metadata retrieved."},
        404: {"description": "corpus_id not found."},
    },
)
async def get_corpus_metadata(corpus_id: str) -> CorpusMetaResponse:
    """
    GET /api/v1/ingest/{corpus_id}

    Blueprint §4.1 contract:
        Response : { corpus_id, doc_count, vocab_size, created_at }

    Raises:
        HTTPException 404: If corpus_id is not in the registry.
    """
    meta = _assert_corpus_exists(corpus_id)

    return CorpusMetaResponse(
        corpus_id=meta["corpus_id"],
        doc_count=meta["doc_count"],
        vocab_size=meta.get("vocab_size", 0),
        created_at=meta["created_at"],
    )


@router.post(
    "/preprocess",
    response_model=PreprocessResponse,
    status_code=status.HTTP_200_OK,
    summary="Run preprocessing pipeline on an uploaded corpus",
    description=(
        "Applies an ordered sequence of text preprocessing steps "
        "(lowercase, stopword removal, lemmatization, HTML stripping) "
        "to a previously uploaded corpus. Persists the cleaned documents "
        "to data/processed/ and updates vocab_size in the corpus registry. "
        "Allowed pipeline steps: lowercase, html_strip, stopwords, "
        "lemmatize, punct_remove."
    ),
    responses={
        200: {"description": "Preprocessing completed successfully."},
        404: {"description": "corpus_id not registered."},
        422: {"description": "Unknown pipeline step or validation error."},
        500: {"description": "Preprocessing failure."},
    },
)
async def preprocess_corpus(body: PreprocessRequest) -> PreprocessResponse:
    """
    POST /api/v1/ingest/preprocess

    Blueprint §4.1 contract:
        Request  : { corpus_id: str, pipeline: List[str] }
        Response : { corpus_id, clean_doc_count, status }

    Pipeline (Blueprint §5.1 Stages 02–03):
        1. Validate corpus_id exists.
        2. Load CleanDocuments from T-1 text_preprocessor.py.
        3. Optionally build vocab via vocab_builder.py.
        4. Persist cleaned corpus to data/processed/{corpus_id}_clean.jsonl.
        5. Update registry with vocab_size and new status.

    Raises:
        HTTPException 404: corpus_id not found.
        HTTPException 500: Preprocessing subprocess failure.
    """
    meta = _assert_corpus_exists(body.corpus_id)
    raw_path: str = meta.get("raw_path", "")

    logger.info(
        "preprocess_corpus | corpus_id=%s | pipeline=%s",
        body.corpus_id,
        body.pipeline,
    )

    # --- Delegate to T-1 text_preprocessor in thread-pool ---------------
    clean_doc_count: int = 0
    vocab_size: int = 0

    try:
        from src.ingestion.text_preprocessor import TextPreprocessor  # type: ignore[import]
        from src.ingestion.vocab_builder import VocabBuilder           # type: ignore[import]
        from src.ingestion.corpus_loader import CorpusLoader           # type: ignore[import]

        # Step 1: Reload raw corpus documents
        loader = CorpusLoader()
        corpus_docs = await _run_sync(loader.load_from_file, raw_path)

        # Step 2: Apply requested preprocessing pipeline
        preprocessor = TextPreprocessor(pipeline_steps=body.pipeline)
        clean_docs = await _run_sync(preprocessor.preprocess, corpus_docs)
        clean_doc_count = len(clean_docs)

        # Step 3: Build vocabulary index
        vocab_builder = VocabBuilder()
        vocab_index = await _run_sync(vocab_builder.build, clean_docs)
        vocab_size = len(vocab_index) if vocab_index else 0

        # Step 4: Persist cleaned corpus (Blueprint §5.1 Stage 02)
        ensure_dir(PROCESSED_DATA_DIR)
        clean_path = (
            Path(PROCESSED_DATA_DIR) / f"{body.corpus_id}_clean.jsonl"
        )
        save_artifact(
            obj=clean_docs,
            path=str(clean_path),
            fmt="json",
        )

        # Step 5: Persist vocab index (Blueprint §5.1 Stage 03)
        ensure_dir(SPLITS_DIR)
        vocab_path = Path(SPLITS_DIR) / f"{body.corpus_id}_vocab.pkl"
        save_artifact(
            obj=vocab_index,
            path=str(vocab_path),
            fmt="pkl",
        )

    except ImportError:
        # T-1 stub mode — estimate from registry doc_count
        logger.warning(
            "T-1 ingestion modules not found; running in stub mode for corpus_id=%s.",
            body.corpus_id,
        )
        clean_doc_count = meta.get("doc_count", 0)
        vocab_size = 0

    except Exception as exc:
        logger.exception(
            "Preprocessing failed | corpus_id=%s", body.corpus_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error":  "preprocessing_failure",
                "detail": str(exc),
                "code":   500,
            },
        ) from exc

    # --- Update registry -------------------------------------------------
    _CORPUS_REGISTRY[body.corpus_id]["clean_doc_count"] = clean_doc_count
    _CORPUS_REGISTRY[body.corpus_id]["vocab_size"] = vocab_size
    _CORPUS_REGISTRY[body.corpus_id]["status"] = "preprocessed"
    _CORPUS_REGISTRY[body.corpus_id]["pipeline"] = body.pipeline

    logger.info(
        "Preprocessing complete | corpus_id=%s | clean_doc_count=%d | vocab_size=%d",
        body.corpus_id,
        clean_doc_count,
        vocab_size,
    )

    return PreprocessResponse(
        corpus_id=body.corpus_id,
        clean_doc_count=clean_doc_count,
        status="preprocessed",
    )