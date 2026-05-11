"""
shared/logger.py
================
Structured JSON logging utility for the HQC Topic Modeling project.

Implements the three public functions mandated by §8.1 of the Master Blueprint:
    - get_logger(name, level)   → returns a structured JSON logger
    - log_run_start(run_id, params) → emits a standardized run-start event
    - log_run_end(run_id, metrics)  → emits a standardized run-end event

Design constraints (from §1.2 and §8):
    - All inter-module communication uses JSON payloads.
    - Every MLflow-emitting helper gracefully degrades if MLflow is absent
      (e.g. during unit tests with MLFLOW_TRACKING_URI unset).
    - Log level is sourced from config/base_config.yaml (log_level: INFO)
      but can be overridden per-module at construction time.
    - Thread-safe: a module-level registry (_LOGGERS) prevents duplicate
      handlers when get_logger() is called multiple times with the same name.

Usage example:
    from shared.logger import get_logger, log_run_start, log_run_end

    logger = get_logger(__name__)
    log_run_start("run_abc123", {"n_topics": 20, "corpus_id": "20ng"})
    ...
    log_run_end("run_abc123", {"coherence_c_v": 0.54})
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Optional MLflow import — degrades gracefully when not installed / configured
# ---------------------------------------------------------------------------
try:
    import mlflow  # type: ignore[import]

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger registry — prevents duplicate handler registration
# ---------------------------------------------------------------------------
_LOGGERS: dict[str, logging.Logger] = {}
_REGISTRY_LOCK = threading.Lock()

# Default log level mirrors config/base_config.yaml → project.log_level
_DEFAULT_LEVEL: str = "INFO"

# ISO-8601 timestamp format (UTC, no microseconds for readability)
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """
    Renders each LogRecord as a single-line JSON object so that log
    aggregators (ELK, CloudWatch, Splunk, etc.) can parse structured fields
    without regex scraping.

    Output shape:
        {
            "timestamp": "2025-05-10T12:00:00Z",
            "level":     "INFO",
            "logger":    "src.ingestion.corpus_loader",
            "message":   "Loaded 18846 documents from 20ng corpus.",
            "module":    "corpus_loader",
            "line":      42,
            "extra":     { ...any extra fields passed to logger.info(...) }
        }
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Pull timestamp from the record (already set by the logging framework)
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(_TS_FORMAT)

        payload: dict[str, Any] = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }

        # Collect any caller-supplied extra fields (e.g. run_id, corpus_id)
        # Standard LogRecord attributes that we do NOT want to re-emit as extras
        _STANDARD_ATTRS = frozenset(
            logging.LogRecord(
                "", 0, "", 0, "", (), None
            ).__dict__.keys()
            | {"message", "asctime"}
        )
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STANDARD_ATTRS
        }
        if extras:
            payload["extra"] = extras

        # Attach exception info when present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


# ---------------------------------------------------------------------------
# Public API — §8.1
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    level: str | int = _DEFAULT_LEVEL,
) -> logging.Logger:
    """
    Return a structured JSON logger for the given *name*.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module, e.g.
        ``"src.ingestion.corpus_loader"``.  Using ``__name__`` ensures
        log messages are traceable to the exact source file.
    level : str | int, optional
        Log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, ``"CRITICAL"``) or the corresponding integer constant
        from the :mod:`logging` module.  Defaults to ``"INFO"`` which
        mirrors ``config/base_config.yaml → project.log_level``.

    Returns
    -------
    logging.Logger
        Configured logger with a single :class:`logging.StreamHandler`
        writing JSON to ``sys.stdout``.  Duplicate handlers are never
        added; calling this function multiple times with the same *name*
        returns the cached instance.

    Notes
    -----
    - ``propagate`` is set to ``False`` to prevent double-printing when
      a root logger is also configured.
    - The handler streams to ``stdout`` (not ``stderr``) so that container
      log drivers capture structured output on fd-1.
    """
    with _REGISTRY_LOCK:
        if name in _LOGGERS:
            return _LOGGERS[name]

        logger = logging.getLogger(name)

        # Resolve level — accept both string and int forms
        numeric_level: int = (
            level
            if isinstance(level, int)
            else getattr(logging, str(level).upper(), logging.INFO)
        )
        logger.setLevel(numeric_level)

        # Avoid propagating to root to prevent duplicate log lines
        logger.propagate = False

        # Only add our handler once (guards against re-import edge cases)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(numeric_level)
            handler.setFormatter(_JsonFormatter())
            logger.addHandler(handler)

        _LOGGERS[name] = logger
        return logger


def log_run_start(run_id: str, params: dict[str, Any]) -> None:
    """
    Emit a standardized run-start event to stdout (via the shared logger)
    and, when MLflow is active, log all *params* to the current MLflow run.

    This function satisfies the contract defined in §8.1:
        ``log_run_start(run_id, params)`` — Emit standardized run-start
        event to MLflow and stdout.

    Parameters
    ----------
    run_id : str
        Unique run identifier.  By convention (§6) this follows the
        pattern ``{model}_{corpus}_{timestamp}``, e.g.
        ``"lda_20ng_20250510"``.
    params : dict[str, Any]
        Arbitrary key-value pairs describing the run configuration
        (e.g. ``{"n_topics": 20, "corpus_id": "20ng"}``).  Values are
        coerced to strings before being passed to MLflow, because the
        MLflow Tracking API only accepts string param values.

    Side-effects
    ------------
    - Writes a JSON log line at INFO level to stdout.
    - If MLflow is importable and a run is active, logs each param via
      ``mlflow.log_param``.  Silently skips MLflow logging if no active
      run exists (common during unit tests).
    """
    _run_logger = get_logger("shared.logger")
    _run_logger.info(
        "Run started.",
        extra={"event": "run_start", "run_id": run_id, "params": params},
    )

    if _MLFLOW_AVAILABLE and mlflow.active_run() is not None:
        try:
            for key, value in params.items():
                mlflow.log_param(key, value)
            mlflow.set_tag("run_id", run_id)
            mlflow.set_tag(
                "run_start_utc",
                datetime.now(tz=timezone.utc).strftime(_TS_FORMAT),
            )
        except Exception as exc:  # noqa: BLE001
            # MLflow failures must never crash the pipeline
            _run_logger.warning(
                "MLflow log_param failed during run_start; continuing.",
                extra={"run_id": run_id, "error": str(exc)},
            )


def log_run_end(run_id: str, metrics: dict[str, float | int | str]) -> None:
    """
    Emit a standardized run-end event to stdout and flush all log handlers.

    This function satisfies the contract defined in §8.1:
        ``log_run_end(run_id, metrics)`` — Emit standardized run-end
        event; flush handlers.

    Parameters
    ----------
    run_id : str
        The same identifier passed to :func:`log_run_start`.
    metrics : dict[str, float | int | str]
        Evaluation metrics to record (e.g.
        ``{"coherence_c_v": 0.54, "perplexity": 312.7}``).
        Numeric values are forwarded to ``mlflow.log_metric``; string
        values are logged as MLflow tags.

    Side-effects
    ------------
    - Writes a JSON log line at INFO level to stdout.
    - If MLflow is importable and a run is active, logs each numeric
      metric via ``mlflow.log_metric`` and each string value as a tag.
    - Flushes all handlers attached to every logger in the module registry
      to ensure no buffered output is lost.
    """
    _run_logger = get_logger("shared.logger")
    _run_logger.info(
        "Run completed.",
        extra={"event": "run_end", "run_id": run_id, "metrics": metrics},
    )

    if _MLFLOW_AVAILABLE and mlflow.active_run() is not None:
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))
                else:
                    mlflow.set_tag(key, str(value))
            mlflow.set_tag(
                "run_end_utc",
                datetime.now(tz=timezone.utc).strftime(_TS_FORMAT),
            )
        except Exception as exc:  # noqa: BLE001
            _run_logger.warning(
                "MLflow log_metric failed during run_end; continuing.",
                extra={"run_id": run_id, "error": str(exc)},
            )

    # Flush all registered handlers so buffered output is written before
    # the process exits or the run context is torn down.
    with _REGISTRY_LOCK:
        for logger in _LOGGERS.values():
            for handler in logger.handlers:
                try:
                    handler.flush()
                except Exception:  # noqa: BLE001
                    pass