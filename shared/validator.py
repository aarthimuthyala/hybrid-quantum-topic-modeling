"""
shared/validator.py
===================
Validation utilities for the HQC Topic Modeling project.

Implements the three public functions mandated by §8.2 of the Master Blueprint:
    - validate_corpus_doc(doc)              → Pydantic validation of CorpusDocument
    - validate_config(config, schema_name)  → JSON-Schema validation of config dicts
    - assert_qubit_feasibility(n_docs, n_topics) → quantum qubit-count guard

Data contracts (§5.2) enforced here:
    CorpusDocument: { doc_id: str, raw_text: str, metadata: dict, source: str }

Config JSON-schemas are loaded from the config/ directory at project root.
The project root is resolved relative to this file's location, following
the flat-imports convention of §2.

Quantum feasibility bound (§9.3):
    quantum.backend.max_qubits = 20 (from quantum_config.yaml)
    This module uses the DEFAULT_MAX_QUBITS constant; downstream code may
    override by passing an explicit limit to assert_qubit_feasibility().

Custom exception hierarchy:
    ValidationError      — base for all validation failures in this project
    CorpusValidationError — raised for malformed CorpusDocument payloads
    ConfigError          — raised when a config dict violates its schema
    QuantumFeasibilityError — raised when qubit demand exceeds backend limit
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, ValidationError as _PydanticValidationError

from shared.logger import get_logger

# ---------------------------------------------------------------------------
# Optional jsonschema import — required only for validate_config()
# ---------------------------------------------------------------------------
try:
    import jsonschema  # type: ignore[import]

    _JSONSCHEMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]
    _JSONSCHEMA_AVAILABLE = False

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Project root resolution (used to locate config/ schemas)
# ---------------------------------------------------------------------------
# shared/validator.py  →  shared/  →  project_root/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

# ---------------------------------------------------------------------------
# Quantum backend constant (mirrors quantum_config.yaml → quantum.backend.max_qubits)
# ---------------------------------------------------------------------------
DEFAULT_MAX_QUBITS: int = 20


# ---------------------------------------------------------------------------
# Custom exception hierarchy
# ---------------------------------------------------------------------------

class HQCValidationError(Exception):
    """
    Base class for all HQC project validation failures.

    Downstream code should catch this base class when a broad catch is
    acceptable, or one of the subclasses for targeted handling.
    """


class CorpusValidationError(HQCValidationError):
    """
    Raised when a dict or object does not conform to the CorpusDocument
    schema defined in §5.2.

    Attributes
    ----------
    doc_id : str | None
        The ``doc_id`` field of the offending document, if available.
    errors : list[dict]
        Pydantic error dicts describing each field failure.
    """

    def __init__(
        self,
        message: str,
        doc_id: str | None = None,
        errors: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(message)
        self.doc_id = doc_id
        self.errors: list[dict[str, Any]] = errors or []


class ConfigError(HQCValidationError):
    """
    Raised when a configuration dictionary fails JSON-Schema validation.

    Attributes
    ----------
    schema_name : str
        Name of the schema file that was being validated against.
    path : str
        JSON pointer to the failing config key.
    """

    def __init__(
        self,
        message: str,
        schema_name: str = "",
        path: str = "",
    ) -> None:
        super().__init__(message)
        self.schema_name = schema_name
        self.path = path


class QuantumFeasibilityError(HQCValidationError):
    """
    Raised when the qubit demand implied by (n_docs, n_topics) exceeds
    the configured backend limit (DEFAULT_MAX_QUBITS = 20).

    Attributes
    ----------
    required_qubits : int
        Number of qubits the experiment would require.
    max_qubits : int
        Hard limit imposed by the backend configuration.
    """

    def __init__(
        self,
        message: str,
        required_qubits: int = 0,
        max_qubits: int = DEFAULT_MAX_QUBITS,
    ) -> None:
        super().__init__(message)
        self.required_qubits = required_qubits
        self.max_qubits = max_qubits


# ---------------------------------------------------------------------------
# §5.2 — CorpusDocument Pydantic model
# ---------------------------------------------------------------------------

class CorpusDocument(BaseModel):
    """
    Pydantic model representing the canonical CorpusDocument contract (§5.2).

    Schema
    ------
    {
        "doc_id":   str,   # non-empty, unique identifier
        "raw_text": str,   # non-empty raw document text
        "metadata": dict,  # arbitrary key-value pairs (empty dict is valid)
        "source":   str    # origin identifier (file path, URL, dataset name)
    }

    Validators
    ----------
    - ``doc_id`` must be a non-empty string after stripping whitespace.
    - ``raw_text`` must be a non-empty string after stripping whitespace.
    - ``source`` must be a non-empty string after stripping whitespace.
    - ``metadata`` defaults to an empty dict if not supplied.
    """

    doc_id: str = Field(..., description="Unique, non-empty document identifier.")
    raw_text: str = Field(..., description="Raw document text (must be non-empty).")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata key-value pairs.",
    )
    source: str = Field(..., description="Origin of the document (path, URL, or dataset name).")

    @field_validator("doc_id", "raw_text", "source")
    @classmethod
    def _must_be_non_empty(cls, value: str) -> str:
        """Reject blank or whitespace-only strings for mandatory string fields."""
        if not value or not value.strip():
            raise ValueError("Field must not be empty or whitespace-only.")
        return value

    model_config = {"frozen": True}  # CorpusDocuments are immutable after creation


# ---------------------------------------------------------------------------
# Public API — §8.2
# ---------------------------------------------------------------------------

def validate_corpus_doc(doc: dict[str, Any] | Any) -> CorpusDocument:
    """
    Validate *doc* against the CorpusDocument schema (§5.2).

    Accepts either a plain ``dict`` or an object with the required
    attributes.  If *doc* is already a :class:`CorpusDocument` instance
    it is returned as-is (idempotent).

    Parameters
    ----------
    doc : dict[str, Any] | Any
        The document to validate.  Expected keys:
        ``doc_id``, ``raw_text``, ``metadata``, ``source``.

    Returns
    -------
    CorpusDocument
        A validated, immutable :class:`CorpusDocument` instance.

    Raises
    ------
    CorpusValidationError
        If any required field is missing, empty, or of the wrong type.
        The exception carries ``.errors`` (list of Pydantic error dicts)
        and ``.doc_id`` (the offending id, if extractable).

    Examples
    --------
    >>> from shared.validator import validate_corpus_doc
    >>> doc = validate_corpus_doc({
    ...     "doc_id": "doc_001",
    ...     "raw_text": "Quantum computing is fascinating.",
    ...     "metadata": {"category": "science"},
    ...     "source": "20ng",
    ... })
    >>> doc.doc_id
    'doc_001'
    """
    # Fast path — already validated
    if isinstance(doc, CorpusDocument):
        return doc

    # Coerce object → dict if needed (e.g. dataclasses, named tuples)
    if not isinstance(doc, dict):
        try:
            doc = doc.__dict__
        except AttributeError:
            raise CorpusValidationError(
                f"Cannot validate type {type(doc).__name__!r}: expected dict "
                "or object with __dict__.",
                doc_id=None,
            )

    doc_id_hint: str | None = doc.get("doc_id")  # for error reporting

    try:
        validated = CorpusDocument.model_validate(doc)
        logger.debug(
            "CorpusDocument validated successfully.",
            extra={"doc_id": validated.doc_id, "source": validated.source},
        )
        return validated

    except _PydanticValidationError as exc:
        error_list: list[dict[str, Any]] = exc.errors()
        logger.error(
            "CorpusDocument validation failed.",
            extra={"doc_id": doc_id_hint, "errors": error_list},
        )
        raise CorpusValidationError(
            f"CorpusDocument validation failed for doc_id={doc_id_hint!r}: "
            f"{len(error_list)} error(s). First: {error_list[0]['msg']}",
            doc_id=str(doc_id_hint) if doc_id_hint is not None else None,
            errors=error_list,
        ) from exc


def validate_config(config: dict[str, Any], schema_name: str) -> None:
    """
    Validate *config* against a JSON-Schema file loaded from ``config/``.

    The schema file is resolved as:
        ``{PROJECT_ROOT}/config/{schema_name}.schema.json``

    Follows the blueprint rule (§1.2): "single source of truth for all
    hyperparameters" — all config dicts must be explicitly validated
    before use.

    Parameters
    ----------
    config : dict[str, Any]
        The configuration dictionary to validate (typically loaded from
        a YAML file).
    schema_name : str
        Base name of the JSON-Schema file, **without** the
        ``.schema.json`` suffix.  Example: ``"base_config"``,
        ``"classical_config"``, ``"quantum_config"``.

    Returns
    -------
    None
        Returns silently on success.

    Raises
    ------
    ConfigError
        If the config dict violates the schema, or if the schema file
        cannot be found.
    ImportError
        If ``jsonschema`` is not installed (in practice this should be
        in ``requirements.txt``).

    Notes
    -----
    JSON-Schema files must live at ``config/{schema_name}.schema.json``.
    They are not included in this repository skeleton but must be created
    by the team following standard JSON-Schema draft-07 or later.
    """
    if not _JSONSCHEMA_AVAILABLE:
        raise ImportError(
            "The 'jsonschema' package is required for validate_config(). "
            "Run: pip install jsonschema"
        )

    schema_path = _CONFIG_DIR / f"{schema_name}.schema.json"

    if not schema_path.exists():
        logger.error(
            "JSON-Schema file not found.",
            extra={"schema_name": schema_name, "path": str(schema_path)},
        )
        raise ConfigError(
            f"Schema file not found: {schema_path}. "
            "Ensure the file exists under config/ and follows the naming "
            "convention {schema_name}.schema.json.",
            schema_name=schema_name,
        )

    try:
        with schema_path.open("r", encoding="utf-8") as fh:
            schema: dict[str, Any] = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"Schema file {schema_path} contains invalid JSON: {exc}",
            schema_name=schema_name,
        ) from exc

    try:
        jsonschema.validate(instance=config, schema=schema)
        logger.debug(
            "Config validated successfully.",
            extra={"schema_name": schema_name},
        )
    except jsonschema.ValidationError as exc:
        path_str = " → ".join(str(p) for p in exc.absolute_path) or "<root>"
        logger.error(
            "Config validation failed.",
            extra={
                "schema_name": schema_name,
                "path": path_str,
                "message": exc.message,
            },
        )
        raise ConfigError(
            f"Config validation failed for schema '{schema_name}' "
            f"at path '{path_str}': {exc.message}",
            schema_name=schema_name,
            path=path_str,
        ) from exc
    except jsonschema.SchemaError as exc:
        raise ConfigError(
            f"The schema file '{schema_name}.schema.json' is itself invalid: "
            f"{exc.message}",
            schema_name=schema_name,
        ) from exc


def assert_qubit_feasibility(
    n_docs: int,
    n_topics: int,
    max_qubits: int = DEFAULT_MAX_QUBITS,
) -> None:
    """
    Raise :class:`QuantumFeasibilityError` if the qubit demand implied by
    (*n_docs*, *n_topics*) exceeds *max_qubits*.

    Qubit estimation follows the QAOA graph-partitioning formulation used
    in this project (§3.3 / §5.1 stage 05):
        required_qubits = ceil(log2(n_docs)) + ceil(log2(n_topics))

    This is a conservative lower bound; the actual circuit may require
    more qubits depending on the ansatz depth and cost Hamiltonian
    encoding.  Teams should always call this guard before constructing
    quantum circuits.

    Parameters
    ----------
    n_docs : int
        Number of documents in the (sub)corpus being processed.
        Must be ≥ 1.
    n_topics : int
        Number of topics or clusters.  Must be ≥ 1.
    max_qubits : int, optional
        Backend qubit limit.  Defaults to ``DEFAULT_MAX_QUBITS`` (20),
        matching ``quantum_config.yaml → quantum.backend.max_qubits``.
        Override for tests or when targeting a different backend.

    Returns
    -------
    None
        Returns silently when the experiment is feasible.

    Raises
    ------
    ValueError
        If *n_docs* < 1 or *n_topics* < 1.
    QuantumFeasibilityError
        If ``required_qubits > max_qubits``.

    Examples
    --------
    >>> assert_qubit_feasibility(50, 5)          # toy subset → OK
    >>> assert_qubit_feasibility(200, 10)         # small subset → OK
    >>> assert_qubit_feasibility(18846, 20)       # full 20ng → raises
    QuantumFeasibilityError: ...
    """
    if n_docs < 1:
        raise ValueError(f"n_docs must be ≥ 1, got {n_docs}.")
    if n_topics < 1:
        raise ValueError(f"n_topics must be ≥ 1, got {n_topics}.")

    # Lower-bound qubit estimate for the QAOA encoding
    qubits_for_docs: int = math.ceil(math.log2(n_docs)) if n_docs > 1 else 1
    qubits_for_topics: int = math.ceil(math.log2(n_topics)) if n_topics > 1 else 1
    required_qubits: int = qubits_for_docs + qubits_for_topics

    logger.debug(
        "Qubit feasibility check.",
        extra={
            "n_docs": n_docs,
            "n_topics": n_topics,
            "required_qubits": required_qubits,
            "max_qubits": max_qubits,
        },
    )

    if required_qubits > max_qubits:
        msg = (
            f"Quantum feasibility check failed: experiment requires at least "
            f"{required_qubits} qubits (n_docs={n_docs}, n_topics={n_topics}) "
            f"but the backend limit is {max_qubits}. "
            f"Use a subset conforming to §7.2 (e.g. toy=50 docs / 5 topics, "
            f"small=200 docs / 10 topics, medium=500 docs / 20 topics)."
        )
        logger.error(
            "Quantum feasibility exceeded.",
            extra={
                "required_qubits": required_qubits,
                "max_qubits": max_qubits,
                "n_docs": n_docs,
                "n_topics": n_topics,
            },
        )
        raise QuantumFeasibilityError(
            msg,
            required_qubits=required_qubits,
            max_qubits=max_qubits,
        )