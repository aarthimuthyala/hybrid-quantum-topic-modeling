"""
shared/file_utils.py
====================
File system utility functions for the HQC Topic Modeling project.

Implements the four public functions mandated by §8.4 of the Master Blueprint:
    - ensure_dir(path)                    → mkdir -p wrapper; returns Path
    - safe_write(path, data, mode)        → atomic write with temp-file swap
    - read_text(path, encoding)           → read text file with logging
    - list_files(directory, pattern, recursive) → glob helper; returns List[Path]

Design principles:
    - All writes are atomic: data is first written to a sibling ``.tmp``
      file, then renamed over the destination.  This prevents downstream
      readers from observing a partially-written file, which is critical
      when pipeline stages run concurrently (§6.2).
    - Every function emits structured JSON log lines via ``shared.logger``
      so that file-I/O events are fully traceable in production logs.
    - No function hard-codes paths; all callers supply explicit paths and
      the helpers operate on whatever they receive.
    - Thread safety: ``safe_write`` uses ``os.replace`` (atomic on POSIX
      and Windows ≥ Vista) and a per-path lock drawn from a module-level
      registry to prevent concurrent writes to the same destination.

Blueprint constraints honoured:
    - Full package imports only (§2 RULE)
    - Structured JSON logging via ``shared.logger`` (§8.1)
    - No hardcoded paths — all resolved by callers
"""

from __future__ import annotations

import fnmatch
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterator

from shared.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-path write lock registry — prevents concurrent overwrites to the same file
# ---------------------------------------------------------------------------
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_LOCK_REGISTRY_LOCK = threading.Lock()


def _get_write_lock(path: Path) -> threading.Lock:
    """
    Return (creating if necessary) the threading.Lock associated with the
    canonical absolute path string of *path*.

    Using one lock per destination path means concurrent writes to
    *different* files are never unnecessarily serialized.
    """
    key = str(path.resolve())
    with _LOCK_REGISTRY_LOCK:
        if key not in _WRITE_LOCKS:
            _WRITE_LOCKS[key] = threading.Lock()
        return _WRITE_LOCKS[key]


# ---------------------------------------------------------------------------
# Public API — §8.4
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """
    Ensure *path* exists as a directory, creating it (and all parents)
    if necessary.

    Satisfies the §8.4 contract:
        ``ensure_dir(path)`` — ``mkdir -p`` wrapper.

    This is a thin wrapper around ``Path.mkdir(parents=True,
    exist_ok=True)`` that also emits a structured log event so that
    directory-creation events are traceable in production logs.

    Parameters
    ----------
    path : str | Path
        Target directory path.  May be relative or absolute; it is
        resolved to an absolute path before creation.

    Returns
    -------
    Path
        The resolved absolute ``Path`` object.

    Raises
    ------
    NotADirectoryError
        If *path* already exists as a *file* (not a directory).
    PermissionError
        If the process lacks write permission on the parent directory.

    Examples
    --------
    >>> from shared.file_utils import ensure_dir
    >>> p = ensure_dir("data/processed/subdir")
    >>> p.is_dir()
    True
    """
    dir_path = Path(path).resolve()

    if dir_path.exists() and not dir_path.is_dir():
        raise NotADirectoryError(
            f"Path exists but is not a directory: {dir_path}"
        )

    already_existed = dir_path.exists()
    dir_path.mkdir(parents=True, exist_ok=True)

    if not already_existed:
        logger.debug(
            "Directory created.",
            extra={"path": str(dir_path)},
        )

    return dir_path


def safe_write(
    path: str | Path,
    data: str | bytes,
    mode: str = "w",
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Path:
    """
    Atomically write *data* to *path* using a temp-file-then-rename strategy.

    Satisfies the §8.4 contract:
        ``safe_write(path, data, mode)`` — atomic write with temp-file swap.

    The write sequence is:
        1. Open a sibling ``.tmp`` file in the same directory as *path*.
        2. Write all *data* to the temp file.
        3. Flush and fsync the temp file (ensures durability on power loss).
        4. ``os.replace(tmp, path)`` — atomic rename on POSIX / Windows.

    A per-path ``threading.Lock`` serialises concurrent writes to the
    same destination, preventing torn-write races in multi-threaded
    pipelines.

    Parameters
    ----------
    path : str | Path
        Destination file path.  Parent directory is created if absent.
    data : str | bytes
        Content to write.  When ``mode="w"`` the data must be a ``str``;
        when ``mode="wb"`` it must be ``bytes``.
    mode : str, optional
        Write mode: ``"w"`` (text, default) or ``"wb"`` (binary).
        Any other value raises ``ValueError``.
    encoding : str, optional
        Character encoding used for text mode.  Ignored in binary mode.
        Default ``"utf-8"``.
    newline : str | None, optional
        Newline translation passed to ``open()`` in text mode.
        ``None`` uses the platform default.  Pass ``""`` to disable
        translation (useful for JSONL files that already contain ``\\n``).

    Returns
    -------
    Path
        The resolved absolute path of the successfully written file.

    Raises
    ------
    ValueError
        If *mode* is not ``"w"`` or ``"wb"``.
    TypeError
        If *data* type is inconsistent with *mode*.
    OSError
        On low-level I/O failure (propagated from the OS).

    Examples
    --------
    >>> from shared.file_utils import safe_write
    >>> p = safe_write("data/test.txt", "hello\\n")
    >>> p.read_text()
    'hello\\n'
    """
    if mode not in ("w", "wb"):
        raise ValueError(
            f"mode must be 'w' or 'wb', got {mode!r}."
        )

    out_path = Path(path).resolve()
    ensure_dir(out_path.parent)

    write_lock = _get_write_lock(out_path)

    with write_lock:
        # Create the temp file in the same directory as the destination so
        # that os.replace() is guaranteed to be atomic (same filesystem).
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            suffix=".tmp",
            prefix=out_path.stem + "_",
            dir=str(out_path.parent),
        )
        tmp_path = Path(tmp_path_str)

        try:
            if mode == "wb":
                if not isinstance(data, (bytes, bytearray)):
                    raise TypeError(
                        f"mode='wb' requires bytes, got {type(data).__name__}."
                    )
                with os.fdopen(tmp_fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
            else:
                if not isinstance(data, str):
                    raise TypeError(
                        f"mode='w' requires str, got {type(data).__name__}."
                    )
                # Close the raw fd first; reopen in text mode via open()
                os.close(tmp_fd)
                tmp_fd = -1  # mark as closed
                with open(tmp_path, "w", encoding=encoding, newline=newline) as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())

            # Atomic rename — overwrites destination if it already exists
            os.replace(tmp_path, out_path)

        except Exception:
            # Best-effort cleanup of the temp file on any failure
            if tmp_fd >= 0:
                try:
                    os.close(tmp_fd)
                except OSError:
                    pass
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    logger.debug(
        "File written atomically.",
        extra={
            "path": str(out_path),
            "mode": mode,
            "size_bytes": out_path.stat().st_size,
        },
    )
    return out_path


def read_text(
    path: str | Path,
    encoding: str = "utf-8",
    errors: str = "strict",
) -> str:
    """
    Read and return the full text content of a file.

    Satisfies the §8.4 contract:
        ``read_text(path, encoding)`` — read text file with logging.

    Parameters
    ----------
    path : str | Path
        Path to the file to read.
    encoding : str, optional
        Character encoding.  Default ``"utf-8"``.
    errors : str, optional
        Error handling strategy passed to ``open()``.  Use ``"replace"``
        for lenient reading of files with mixed encodings.  Default
        ``"strict"`` (raises ``UnicodeDecodeError`` on bad bytes).

    Returns
    -------
    str
        Full file content as a string.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    IsADirectoryError
        If *path* is a directory.
    UnicodeDecodeError
        If *encoding* cannot decode the file and *errors*``="strict"``.

    Examples
    --------
    >>> content = read_text("data/raw/20ng.jsonl")
    """
    in_path = Path(path).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {in_path}")
    if in_path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {in_path}")

    content = in_path.read_text(encoding=encoding, errors=errors)

    logger.debug(
        "File read.",
        extra={
            "path": str(in_path),
            "size_bytes": in_path.stat().st_size,
            "encoding": encoding,
        },
    )
    return content


def read_bytes(path: str | Path) -> bytes:
    """
    Read and return the raw bytes of a file.

    Companion to :func:`read_text` for binary artifacts (e.g. ``.pkl``,
    ``.npy``).

    Parameters
    ----------
    path : str | Path
        Path to the file to read.

    Returns
    -------
    bytes
        Full file content as raw bytes.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    IsADirectoryError
        If *path* is a directory.
    """
    in_path = Path(path).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"File not found: {in_path}")
    if in_path.is_dir():
        raise IsADirectoryError(f"Expected a file, got a directory: {in_path}")

    data = in_path.read_bytes()

    logger.debug(
        "File read (binary).",
        extra={"path": str(in_path), "size_bytes": len(data)},
    )
    return data


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    """
    Return a sorted list of files in *directory* matching *pattern*.

    Satisfies the §8.4 contract:
        ``list_files(directory, pattern, recursive)`` — glob helper;
        returns ``List[Path]``.

    Parameters
    ----------
    directory : str | Path
        Root directory to search.
    pattern : str, optional
        Glob pattern applied to file names only (not full paths).
        Examples: ``"*.jsonl"``, ``"*_clean.*"``, ``"vocab_*.pkl"``.
        Default ``"*"`` matches all files.
    recursive : bool, optional
        When ``True``, descends into sub-directories.
        When ``False`` (default), only the immediate directory is searched.

    Returns
    -------
    list[Path]
        Sorted list of absolute ``Path`` objects for matching files.
        Directories are excluded; only regular files are returned.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist.
    NotADirectoryError
        If *directory* exists but is not a directory.

    Examples
    --------
    >>> jsonl_files = list_files("data/raw", "*.jsonl")
    >>> vocab_files = list_files("data/splits", "*.pkl", recursive=True)
    """
    dir_path = Path(directory).resolve()

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, got a file: {dir_path}"
        )

    if recursive:
        # rglob returns all descendants; filter to files matching pattern
        matches: list[Path] = sorted(
            p for p in dir_path.rglob(pattern) if p.is_file()
        )
    else:
        matches = sorted(
            p for p in dir_path.glob(pattern) if p.is_file()
        )

    logger.debug(
        "Files listed.",
        extra={
            "directory": str(dir_path),
            "pattern": pattern,
            "recursive": recursive,
            "match_count": len(matches),
        },
    )
    return matches


# ---------------------------------------------------------------------------
# Additional helpers used internally across the pipeline
# ---------------------------------------------------------------------------

def iter_jsonl(path: str | Path, encoding: str = "utf-8") -> Iterator[dict]:
    """
    Yield parsed JSON objects from a newline-delimited JSON file,
    skipping blank lines and ``//``-prefixed comment lines.

    This is the standard way to stream large JSONL artifacts (corpus
    snapshots, token files) without loading them fully into memory.

    Parameters
    ----------
    path : str | Path
        Path to the ``.jsonl`` file.
    encoding : str, optional
        File encoding.  Default ``"utf-8"``.

    Yields
    ------
    dict
        One parsed JSON object per non-blank, non-comment line.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    import json

    in_path = Path(path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {in_path}")

    with in_path.open("r", encoding=encoding) as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                yield json.loads(line)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Skipping malformed JSONL line.",
                    extra={"path": str(in_path), "line": lineno, "error": str(exc)},
                )


def copy_file(src: str | Path, dst: str | Path, overwrite: bool = True) -> Path:
    """
    Copy *src* to *dst*, creating parent directories as needed.

    Uses :func:`safe_write` under the hood so the copy is atomic.

    Parameters
    ----------
    src : str | Path
        Source file path.
    dst : str | Path
        Destination file path.
    overwrite : bool, optional
        If ``False`` and *dst* exists, raise ``FileExistsError``.
        Default ``True``.

    Returns
    -------
    Path
        Resolved absolute destination path.

    Raises
    ------
    FileNotFoundError
        If *src* does not exist.
    FileExistsError
        If *dst* exists and *overwrite* is ``False``.
    """
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if dst_path.exists() and not overwrite:
        raise FileExistsError(
            f"Destination already exists and overwrite=False: {dst_path}"
        )

    data = src_path.read_bytes()
    result = safe_write(dst_path, data, mode="wb")

    logger.debug(
        "File copied.",
        extra={"src": str(src_path), "dst": str(result)},
    )
    return result


def file_size_bytes(path: str | Path) -> int:
    """
    Return the size in bytes of the file at *path*.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    int
        File size in bytes.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p.stat().st_size


def human_readable_size(size_bytes: int) -> str:
    """
    Convert a byte count to a human-readable string (e.g. ``"12.3 MB"``).

    Parameters
    ----------
    size_bytes : int
        Number of bytes.

    Returns
    -------
    str
        Human-readable size string using binary prefixes (KiB, MiB, GiB).
    """
    if size_bytes < 0:
        raise ValueError(f"size_bytes must be non-negative, got {size_bytes}.")
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes //= 1024
    return f"{size_bytes:.1f} PiB"
from pathlib import Path
from typing import Union


def get_artifact_path(
    artifact_type: str,
    filename: str,
    base_dir: Union[str, Path] = "outputs"
) -> Path:
    """
    Generate a standardized artifact path.

    Args:
        artifact_type: Type of artifact (models, figures, reports, etc.)
        filename: Name of the file
        base_dir: Base output directory

    Returns:
        Path object for artifact storage
    """
    base_path = Path(base_dir)
    artifact_dir = base_path / artifact_type

    artifact_dir.mkdir(parents=True, exist_ok=True)

    return artifact_dir / filename