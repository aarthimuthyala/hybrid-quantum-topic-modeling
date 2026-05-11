"""
shared/serializer.py
====================
Artifact serialization and deserialization utilities for the HQC project.

Implements the three public functions mandated by §8.3 of the Master Blueprint:
    - save_artifact(obj, path, fmt)  → serialize to pkl/json/npy; log to MLflow
    - load_artifact(path, fmt)       → deserialize; validate SHA-256 against manifest
    - to_json_response(obj)          → convert Result dataclass to API-safe dict

Supported formats (``fmt`` parameter):
    - ``"pkl"``   — Python pickle (any Python object; not cross-language safe)
    - ``"json"``  — JSON (dicts, lists, primitives; np.ndarray → list on save)
    - ``"npy"``   — NumPy binary format (np.ndarray only)

SHA-256 tracking (§7.2 Data Versioning Policy):
    Every ``save_artifact`` call records the hex SHA-256 digest of the
    written bytes in ``data/manifest.json`` under the key derived from
    the artifact path.  Every ``load_artifact`` call re-computes the digest
    and compares it to the stored value, raising ``ArtifactIntegrityError``
    on mismatch.

Blueprint constraints honoured:
    - Full package imports only (§2 RULE)
    - Structured JSON logging via ``shared.logger`` (§8.1)
    - MLflow artifact logging (graceful degradation when MLflow absent)
    - No hardcoded paths — all resolved relative to project root
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

from shared.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore[import]

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

try:
    import mlflow  # type: ignore[import]

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
_MANIFEST_PATH: Path = _PROJECT_ROOT / "data" / "manifest.json"

# ---------------------------------------------------------------------------
# Supported format constants
# ---------------------------------------------------------------------------
_FMT_PKL = "pkl"
_FMT_JSON = "json"
_FMT_NPY = "npy"
_SUPPORTED_FMTS = (_FMT_PKL, _FMT_JSON, _FMT_NPY)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class ArtifactIntegrityError(Exception):
    """
    Raised when a loaded artifact's SHA-256 digest does not match the
    value recorded in ``data/manifest.json`` (§7.2 Data Versioning Policy).

    Attributes
    ----------
    path : str
        Path to the artifact file.
    stored_hash : str
        SHA-256 digest recorded in the manifest at save-time.
    computed_hash : str
        SHA-256 digest re-computed from the file bytes at load-time.
    """

    def __init__(self, path: str, stored_hash: str, computed_hash: str) -> None:
        super().__init__(
            f"SHA-256 mismatch for artifact '{path}': "
            f"stored={stored_hash[:12]}…, computed={computed_hash[:12]}…. "
            "The file may have been modified after saving.  "
            "Re-generate the artifact or update data/manifest.json."
        )
        self.path = path
        self.stored_hash = stored_hash
        self.computed_hash = computed_hash


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _read_manifest() -> dict[str, Any]:
    """Load ``data/manifest.json``, returning an empty dict if absent."""
    if not _MANIFEST_PATH.exists():
        return {}
    try:
        with _MANIFEST_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not read manifest; treating as empty.",
            extra={"path": str(_MANIFEST_PATH), "error": str(exc)},
        )
        return {}


def _write_manifest(manifest: dict[str, Any]) -> None:
    """Persist *manifest* to ``data/manifest.json``."""
    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _manifest_key(path: Path) -> str:
    """
    Derive a manifest key from an artifact path.

    Uses the path relative to the project root when possible; falls back
    to the absolute path string.
    """
    try:
        return str(path.resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _compute_sha256(data: bytes) -> str:
    """Return hex-encoded SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# JSON encoder for numpy and dataclasses
# ---------------------------------------------------------------------------

class _ArtifactJSONEncoder(json.JSONEncoder):
    """
    Extended JSON encoder that handles types commonly found in ML artifacts:
    - ``numpy.ndarray``  → list (via ``.tolist()``)
    - ``numpy.integer``  → int
    - ``numpy.floating`` → float
    - dataclasses        → dict (via ``dataclasses.asdict``)
    - ``Path``           → str
    - ``set``            → sorted list
    """

    def default(self, obj: Any) -> Any:
        if _NUMPY_AVAILABLE:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


# ---------------------------------------------------------------------------
# Public API — §8.3
# ---------------------------------------------------------------------------

def save_artifact(
    obj: Any,
    path: str | Path,
    fmt: str = _FMT_PKL,
    log_to_mlflow: bool = True,
) -> Path:
    """
    Serialize *obj* to *path* in the specified *fmt* and log to MLflow.

    Satisfies the §8.3 contract:
        ``save_artifact(obj, path, fmt)`` — Serialize to pkl/json/npy;
        log path to MLflow as artifact.

    Parameters
    ----------
    obj : Any
        The Python object to serialize.
        - ``"pkl"``  → any picklable object
        - ``"json"`` → dict/list/primitive; numpy arrays coerced to lists
        - ``"npy"``  → ``numpy.ndarray`` only
    path : str | Path
        Destination file path.  The parent directory is created if it
        does not exist.  The file extension is appended if absent:
        ``.pkl``, ``.json``, or ``.npy``.
    fmt : str, optional
        Serialization format: ``"pkl"``, ``"json"``, or ``"npy"``.
        Default ``"pkl"``.
    log_to_mlflow : bool, optional
        When ``True`` (default) and an MLflow run is active, log the
        artifact path via ``mlflow.log_artifact``.

    Returns
    -------
    Path
        Absolute path where the artifact was written.

    Raises
    ------
    ValueError
        If *fmt* is not one of the supported formats.
    TypeError
        If *obj* is not a ``numpy.ndarray`` and ``fmt="npy"``.
    ImportError
        If NumPy is not installed and ``fmt="npy"`` is requested.
    """
    fmt = fmt.lower()
    if fmt not in _SUPPORTED_FMTS:
        raise ValueError(
            f"Unsupported artifact format: {fmt!r}. "
            f"Choose one of {_SUPPORTED_FMTS}."
        )

    out_path = Path(path)
    # Append correct extension if missing
    if out_path.suffix.lstrip(".").lower() != fmt:
        out_path = out_path.with_suffix(f".{fmt}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize
    if fmt == _FMT_PKL:
        raw_bytes = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        out_path.write_bytes(raw_bytes)

    elif fmt == _FMT_JSON:
        json_str = json.dumps(obj, cls=_ArtifactJSONEncoder, indent=2)
        raw_bytes = json_str.encode("utf-8")
        out_path.write_bytes(raw_bytes)

    elif fmt == _FMT_NPY:
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required for fmt='npy': pip install numpy"
            )
        if not isinstance(obj, np.ndarray):
            raise TypeError(
                f"fmt='npy' requires a numpy.ndarray, got {type(obj).__name__}."
            )
        np.save(str(out_path), obj, allow_pickle=False)
        raw_bytes = out_path.read_bytes()  # re-read for consistent SHA-256

    sha256 = _compute_sha256(raw_bytes if fmt != _FMT_NPY else out_path.read_bytes())

    # Update manifest (§7.2)
    manifest = _read_manifest()
    manifest[_manifest_key(out_path)] = {
        "sha256": sha256,
        "fmt": fmt,
        "size_bytes": out_path.stat().st_size,
    }
    _write_manifest(manifest)

    logger.info(
        "Artifact saved.",
        extra={
            "path": str(out_path),
            "fmt": fmt,
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256[:12] + "…",
        },
    )

    # MLflow artifact logging
    if log_to_mlflow and _MLFLOW_AVAILABLE and mlflow.active_run() is not None:
        try:
            mlflow.log_artifact(str(out_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MLflow log_artifact failed; continuing.",
                extra={"path": str(out_path), "error": str(exc)},
            )

    return out_path


def load_artifact(
    path: str | Path,
    fmt: str | None = None,
    verify_hash: bool = True,
) -> Any:
    """
    Deserialize an artifact from *path*, optionally verifying its SHA-256.

    Satisfies the §8.3 contract:
        ``load_artifact(path, fmt)`` — Deserialize; validate SHA-256
        against ``data/manifest.json``.

    Parameters
    ----------
    path : str | Path
        Path to the artifact file.
    fmt : str | None, optional
        Serialization format.  If ``None``, inferred from the file
        extension (``.pkl``, ``.json``, ``.npy``).
    verify_hash : bool, optional
        When ``True`` (default), compare the file's SHA-256 against the
        manifest entry and raise :class:`ArtifactIntegrityError` on
        mismatch.  Set to ``False`` only for testing or emergency recovery.

    Returns
    -------
    Any
        Deserialized Python object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the format cannot be inferred and *fmt* is ``None``.
    ArtifactIntegrityError
        If ``verify_hash=True`` and the stored/computed SHA-256 differ.
    """
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Artifact file not found: {in_path}")

    # Infer format from extension if not specified
    if fmt is None:
        ext = in_path.suffix.lstrip(".").lower()
        if ext not in _SUPPORTED_FMTS:
            raise ValueError(
                f"Cannot infer format from extension '{in_path.suffix}'. "
                f"Provide fmt explicitly from {_SUPPORTED_FMTS}."
            )
        fmt = ext
    fmt = fmt.lower()

    raw_bytes = in_path.read_bytes()

    # SHA-256 verification (§7.2)
    if verify_hash:
        computed = _compute_sha256(raw_bytes)
        manifest = _read_manifest()
        key = _manifest_key(in_path)
        stored_entry = manifest.get(key, {})
        stored = stored_entry.get("sha256")

        if stored is None:
            logger.warning(
                "No manifest entry found for artifact; skipping hash check.",
                extra={"path": str(in_path), "key": key},
            )
        elif stored != computed:
            raise ArtifactIntegrityError(
                path=str(in_path),
                stored_hash=stored,
                computed_hash=computed,
            )

    # Deserialize
    if fmt == _FMT_PKL:
        obj = pickle.loads(raw_bytes)  # noqa: S301
    elif fmt == _FMT_JSON:
        obj = json.loads(raw_bytes.decode("utf-8"))
    elif fmt == _FMT_NPY:
        if not _NUMPY_AVAILABLE:
            raise ImportError("NumPy is required to load .npy files: pip install numpy")
        import io
        obj = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    else:
        raise ValueError(f"Unsupported fmt: {fmt!r}")

    logger.info(
        "Artifact loaded.",
        extra={"path": str(in_path), "fmt": fmt},
    )
    return obj


def to_json_response(obj: Any) -> dict[str, Any]:
    """
    Convert a Result dataclass or any serializable object to an
    API-safe dict, stripping ``numpy.ndarray`` fields.

    Satisfies the §8.3 contract:
        ``to_json_response(obj)`` — Convert any Result dataclass to
        API-safe dict (removes np.ndarrays).

    ``numpy.ndarray`` values are replaced with their shape and dtype
    (as a summary dict) rather than being serialised inline, since
    large matrices would bloat API responses.  Callers that need the
    actual array should use :func:`save_artifact` / :func:`load_artifact`.

    Parameters
    ----------
    obj : Any
        A dataclass instance, dict, list, or any JSON-serializable object.
        Supported result types: any class with a ``to_dict()`` method,
        any ``dataclasses.dataclass``, or a plain dict/list.

    Returns
    -------
    dict[str, Any]
        A flat, JSON-serializable dict suitable for FastAPI response models
        (all ``numpy.ndarray`` values replaced with summary dicts).

    Examples
    --------
    >>> result = TopicModelResult(model_id="m1", topics=[...], doc_topic_matrix=np.eye(3), params={})
    >>> to_json_response(result)
    {"model_id": "m1", "topics": [...], "doc_topic_matrix": {"__ndarray__": true, "shape": [3, 3], "dtype": "float64"}, "params": {}}
    """
    # Step 1 — coerce to dict
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        raw: Any = obj.to_dict()
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        raw = dataclasses.asdict(obj)
    elif isinstance(obj, dict):
        raw = obj
    else:
        # Fall back: attempt JSON round-trip via the custom encoder
        try:
            json_str = json.dumps(obj, cls=_ArtifactJSONEncoder)
            return json.loads(json_str)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Cannot convert {type(obj).__name__!r} to JSON-safe dict: {exc}"
            ) from exc

    # Step 2 — recursively strip / summarize ndarray values
    return _sanitize_for_api(raw)


# ---------------------------------------------------------------------------
# Internal recursive sanitizer
# ---------------------------------------------------------------------------

def _sanitize_for_api(obj: Any) -> Any:
    """
    Recursively walk *obj* and replace ``numpy.ndarray`` with a summary dict.

    Summary format:
        { "__ndarray__": true, "shape": [...], "dtype": "<dtype_str>" }

    All other numpy scalars are coerced to native Python types.
    """
    if _NUMPY_AVAILABLE:
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)

    if isinstance(obj, dict):
        return {k: _sanitize_for_api(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        sanitized = [_sanitize_for_api(item) for item in obj]
        return sanitized if isinstance(obj, list) else tuple(sanitized)

    if isinstance(obj, set):
        return sorted(_sanitize_for_api(item) for item in obj)

    if isinstance(obj, Path):
        return str(obj)

    # Primitive types (str, int, float, bool, None) pass through unchanged
    return obj