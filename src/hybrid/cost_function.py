"""
src/hybrid/cost_function.py
============================
Team T-4: Hybrid Pipeline Team
MASTER BLUEPRINT v1.0 — §3.4

Responsibility:
    Map NLP similarity/clustering cost (DTMatrix or EmbeddingMatrix) to a
    QUBO / Ising Hamiltonian consumable by src/quantum/qaoa_optimizer.py and
    src/quantum/vqe_solver.py.

Inputs:  DTMatrix (np.ndarray) or EmbeddingMatrix (np.ndarray)
Outputs: CostHamiltonian dataclass

Integration points:
    - Consumes T-2 artifacts: DTMatrix produced by tfidf_vectorizer.py
    - Produces CostHamiltonian consumed by T-3 qaoa_optimizer.py / vqe_solver.py
    - Uses shared/math_utils.qubo_to_ising and shared/math_utils.cosine_similarity_matrix
    - Uses shared/logger.get_logger
    - Uses shared/serializer.save_artifact for §5 artifact naming

Blueprint data-flow stage: Stage 05 — Cost Hamiltonian Build
Artifact:  outputs/models/{run_id}_hamiltonian.pkl
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from shared.logger import get_logger
from shared.math_utils import (
    compute_graph_laplacian,
    cosine_similarity_matrix,
    normalize_rows,
    qubo_to_ising,
)
from shared.serializer import save_artifact
from shared.validator import assert_qubit_feasibility

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants (never hardcode numerics — all surfaced here for config override)
# ---------------------------------------------------------------------------
DEFAULT_N_TOPICS: int = 20
DEFAULT_PENALTY_STRENGTH: float = 1.0
DEFAULT_SIMILARITY_THRESHOLD: float = 0.0   # edges with weight ≤ this are pruned
DEFAULT_NORMALISE_WEIGHTS: bool = True
MAX_QUBO_VARIABLES: int = 500               # guard against accidentally huge problems


# ---------------------------------------------------------------------------
# Public data-type contracts (Blueprint §5.2)
# ---------------------------------------------------------------------------

class HamiltonianEncoding(str, Enum):
    """Supported problem encodings for the quantum back-end."""
    QUBO = "qubo"
    ISING = "ising"


@dataclass
class CostHamiltonian:
    """
    Frozen contract between the Hybrid layer (T-4) and the Quantum layer (T-3).

    Fields
    ------
    run_id          : Unique run identifier (Blueprint §6 naming: {run_id}_type.ext)
    encoding        : One of HamiltonianEncoding.QUBO | HamiltonianEncoding.ISING
    qubo            : QUBO coefficient dict {(i, j): float} — always populated
    ising_h         : Ising linear biases {i: float} — populated when encoding=ISING
    ising_J         : Ising quadratic couplings {(i,j): float} — populated when encoding=ISING
    n_variables     : Number of binary / spin variables
    n_topics        : Intended number of clusters / topics
    offset          : Constant energy offset from QUBO → Ising conversion
    metadata        : Provenance info (corpus_id, matrix shape, thresholds, …)
    """
    run_id: str
    encoding: HamiltonianEncoding
    qubo: Dict[Tuple[int, int], float]
    ising_h: Dict[int, float]
    ising_J: Dict[Tuple[int, int], float]
    n_variables: int
    n_topics: int
    offset: float = 0.0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

class CostFunctionBuilder:
    """
    Converts an NLP similarity / document-term matrix into a QUBO / Ising
    Hamiltonian suitable for QAOA or VQE circuits.

    The encoding follows a graph-cut / min-cut formulation:
        • Build an n×n cosine-similarity graph W from the document embeddings.
        • Derive the normalised graph Laplacian L.
        • Encode the k-way clustering objective as a QUBO:
              H = Σ_{i<j} W_ij (x_i - x_j)² + λ Σ_i penalty_i
          where x_i ∈ {0,1} represents cluster membership.
        • Optionally convert to Ising (spin) representation.

    Blueprint §8.5 utility usage:
        shared/math_utils.cosine_similarity_matrix
        shared/math_utils.compute_graph_laplacian
        shared/math_utils.qubo_to_ising
        shared/math_utils.normalize_rows
    """

    def __init__(
        self,
        n_topics: int = DEFAULT_N_TOPICS,
        penalty_strength: float = DEFAULT_PENALTY_STRENGTH,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        normalise_weights: bool = DEFAULT_NORMALISE_WEIGHTS,
        encoding: HamiltonianEncoding = HamiltonianEncoding.ISING,
    ) -> None:
        self.n_topics = n_topics
        self.penalty_strength = penalty_strength
        self.similarity_threshold = similarity_threshold
        self.normalise_weights = normalise_weights
        self.encoding = encoding

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(
        self,
        matrix: np.ndarray,
        run_id: str,
        corpus_id: str = "unknown",
        save: bool = True,
    ) -> CostHamiltonian:
        """
        Build a CostHamiltonian from a document-term or embedding matrix.

        Parameters
        ----------
        matrix    : np.ndarray, shape (n_docs, n_features).
                    Either a DTMatrix from tfidf_vectorizer.py or an
                    EmbeddingMatrix (dense float array).
        run_id    : Unique identifier for this pipeline run.
        corpus_id : Source corpus identifier for metadata provenance.
        save      : If True, persist the Hamiltonian via shared/serializer
                    to outputs/models/{run_id}_hamiltonian.pkl.

        Returns
        -------
        CostHamiltonian
        """
        n_docs, n_features = matrix.shape
        logger.info(
            "Building cost Hamiltonian",
            extra={
                "run_id": run_id,
                "n_docs": n_docs,
                "n_features": n_features,
                "n_topics": self.n_topics,
                "encoding": self.encoding.value,
            },
        )

        # Qubit feasibility guard (Blueprint §8.2)
        assert_qubit_feasibility(n_docs, self.n_topics)

        if n_docs > MAX_QUBO_VARIABLES:
            warnings.warn(
                f"n_docs={n_docs} exceeds MAX_QUBO_VARIABLES={MAX_QUBO_VARIABLES}. "
                "Consider using a quantum-feasibility subset (Blueprint §7.2).",
                ResourceWarning,
                stacklevel=2,
            )

        # Step 1 — Normalise rows so cosine similarity is well-defined
        if self.normalise_weights:
            matrix = normalize_rows(matrix)

        # Step 2 — Build similarity graph
        W = self._build_similarity_graph(matrix)

        # Step 3 — Compute graph Laplacian (used for penalty terms)
        L = compute_graph_laplacian(W)

        # Step 4 — Encode as QUBO
        qubo = self._build_qubo(W, L, n_docs)

        # Step 5 — Optionally convert to Ising representation
        ising_h, ising_J, offset = self._to_ising(qubo)

        hamiltonian = CostHamiltonian(
            run_id=run_id,
            encoding=self.encoding,
            qubo=qubo,
            ising_h=ising_h,
            ising_J=ising_J,
            n_variables=n_docs,
            n_topics=self.n_topics,
            offset=offset,
            metadata={
                "corpus_id": corpus_id,
                "matrix_shape": list(matrix.shape),
                "similarity_threshold": self.similarity_threshold,
                "penalty_strength": self.penalty_strength,
                "normalise_weights": self.normalise_weights,
                "n_qubo_terms": len(qubo),
                "n_ising_h_terms": len(ising_h),
                "n_ising_J_terms": len(ising_J),
            },
        )

        if save:
            artifact_path = f"outputs/models/{run_id}_hamiltonian.pkl"
            save_artifact(hamiltonian, artifact_path, fmt="pkl")
            logger.info("Hamiltonian saved", extra={"path": artifact_path, "run_id": run_id})

        return hamiltonian

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_similarity_graph(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the cosine-similarity weight matrix W (n_docs × n_docs).
        Entries below similarity_threshold are zeroed (graph pruning).
        """
        W = cosine_similarity_matrix(matrix)
        np.fill_diagonal(W, 0.0)  # no self-loops
        W[W < self.similarity_threshold] = 0.0
        W = np.clip(W, 0.0, 1.0)  # cosine similarity ∈ [0, 1]
        return W

    def _build_qubo(
        self,
        W: np.ndarray,
        L: np.ndarray,
        n_docs: int,
    ) -> Dict[Tuple[int, int], float]:
        """
        Encode the graph clustering problem as a QUBO dict.

        Objective (graph-cut formulation):
            minimize  -Σ_{i<j} W_ij · (x_i ⊕ x_j)
                      + λ · Σ_i L_ii · x_i      [degree penalty]

        Variables x_i ∈ {0,1} represent cluster-0 vs cluster-1 membership.
        For k>2 topics this is a single-layer binary relaxation; multi-layer
        decomposition is handled by hybrid_pipeline.py (recursive bisection).

        Returns QUBO dict where keys are (i, j) with i ≤ j.
        """
        qubo: Dict[Tuple[int, int], float] = {}

        # Quadratic terms: reward same-cluster pairs, penalise cross-cluster
        for i in range(n_docs):
            for j in range(i + 1, n_docs):
                w_ij = float(W[i, j])
                if w_ij == 0.0:
                    continue
                # Cross-cluster penalty (promotes similar docs into same cluster)
                qubo[(i, j)] = qubo.get((i, j), 0.0) - w_ij

        # Linear terms: degree-based penalty from graph Laplacian diagonal
        for i in range(n_docs):
            l_ii = float(L[i, i])
            if l_ii != 0.0:
                qubo[(i, i)] = qubo.get((i, i), 0.0) + self.penalty_strength * l_ii

        return qubo

    def _to_ising(
        self,
        qubo: Dict[Tuple[int, int], float],
    ) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
        """
        Delegate to shared/math_utils.qubo_to_ising.
        Returns (h, J, offset) — Ising linear biases, couplings, constant.
        """
        if self.encoding == HamiltonianEncoding.QUBO:
            # Return empty Ising dicts; caller uses qubo field directly
            return {}, {}, 0.0

        h, J, offset = qubo_to_ising(qubo)
        return h, J, offset


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def build_cost_hamiltonian(
    matrix: np.ndarray,
    run_id: str,
    n_topics: int = DEFAULT_N_TOPICS,
    penalty_strength: float = DEFAULT_PENALTY_STRENGTH,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    normalise_weights: bool = DEFAULT_NORMALISE_WEIGHTS,
    encoding: HamiltonianEncoding = HamiltonianEncoding.ISING,
    corpus_id: str = "unknown",
    save: bool = True,
) -> CostHamiltonian:
    """
    Functional convenience wrapper around CostFunctionBuilder.build().

    Consumed by hybrid_pipeline.py (T-4) and directly by qaoa_optimizer.py /
    vqe_solver.py (T-3) via the CostHamiltonian contract.

    Example
    -------
    >>> from src.hybrid.cost_function import build_cost_hamiltonian
    >>> hamiltonian = build_cost_hamiltonian(
    ...     matrix=dt_matrix,        # np.ndarray from tfidf_vectorizer.py
    ...     run_id="abc123",
    ...     n_topics=10,
    ...     encoding=HamiltonianEncoding.ISING,
    ...     corpus_id="20ng",
    ... )
    >>> print(hamiltonian.n_variables, len(hamiltonian.ising_J))
    """
    builder = CostFunctionBuilder(
        n_topics=n_topics,
        penalty_strength=penalty_strength,
        similarity_threshold=similarity_threshold,
        normalise_weights=normalise_weights,
        encoding=encoding,
    )
    return builder.build(matrix=matrix, run_id=run_id, corpus_id=corpus_id, save=save)