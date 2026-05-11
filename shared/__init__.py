"""shared — Cross-cutting utilities consumed by all layers."""
from shared.logger import get_logger, log_run_start, log_run_end
from shared.validator import (
    validate_corpus_doc,
    validate_config,
    assert_qubit_feasibility,
    CorpusDocument,
    CorpusValidationError,
    ConfigError,
    QuantumFeasibilityError,
)

__all__ = [
    "get_logger",
    "log_run_start",
    "log_run_end",
    "validate_corpus_doc",
    "validate_config",
    "assert_qubit_feasibility",
    "CorpusDocument",
    "CorpusValidationError",
    "ConfigError",
    "QuantumFeasibilityError",
]