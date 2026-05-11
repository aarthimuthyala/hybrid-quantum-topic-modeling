"""
src/quantum/circuit_builder.py
==============================
Team T-3: Quantum Engine — Circuit Construction Module

Responsibility:
    Construct parameterised quantum circuits for QAOA and VQE ansatz variants.
    Input  : CircuitConfig (dataclass defined here; frozen contract per §10.3)
    Output : qiskit.circuit.QuantumCircuit

Qiskit compatibility: 1.x  (imports from qiskit, qiskit.circuit.library)
Naming conventions   : §6 of MASTER_BLUEPRINT v1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    EfficientSU2,
    QAOAAnsatz,
    RealAmplitudes,
    TwoLocal,
)
from qiskit.quantum_info import SparsePauliOp

from shared.logger import get_logger

logger: logging.Logger = get_logger(__name__, level="INFO")

# ---------------------------------------------------------------------------
# Constants  (§6 — UPPER_SNAKE_CASE)
# ---------------------------------------------------------------------------
DEFAULT_SHOTS: int = 1024
DEFAULT_REPS: int = 2
SUPPORTED_ANSATZ_TYPES: frozenset[str] = frozenset(
    {"RealAmplitudes", "EfficientSU2", "TwoLocal", "QAOA", "custom_ry"}
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class CircuitType(str, Enum):
    """Top-level circuit family selector."""

    QAOA = "QAOA"
    VQE = "VQE"


class AnsatzType(str, Enum):
    """Named ansatz variants supported by CircuitBuilder."""

    REAL_AMPLITUDES = "RealAmplitudes"
    EFFICIENT_SU2 = "EfficientSU2"
    TWO_LOCAL = "TwoLocal"
    QAOA = "QAOA"
    CUSTOM_RY = "custom_ry"


# ---------------------------------------------------------------------------
# Data contracts  (frozen per Blueprint §10.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CircuitConfig:
    """
    Immutable configuration object consumed by CircuitBuilder.

    All fields map directly to config/quantum_config.yaml keys (§9.3).

    Attributes
    ----------
    n_qubits:
        Number of qubits; must be ≤ quantum.backend.max_qubits (default 20).
    circuit_type:
        QAOA or VQE — selects the high-level circuit family.
    ansatz_type:
        Specific parameterised ansatz to construct.
    reps:
        Repetition depth for layered ansätze (maps to quantum.vqe.reps).
    p_layers:
        QAOA p-layers (maps to quantum.qaoa.p_layers).
    entanglement:
        Entanglement strategy passed to Qiskit library circuits
        (e.g. "linear", "full", "circular").
    cost_hamiltonian:
        SparsePauliOp encoding the cost function; required for QAOA circuits.
    insert_barriers:
        If True, barriers are inserted between layers for readability/debugging.
    seed:
        Optional integer seed for any stochastic circuit elements.
    """

    n_qubits: int
    circuit_type: CircuitType
    ansatz_type: AnsatzType = AnsatzType.REAL_AMPLITUDES
    reps: int = DEFAULT_REPS
    p_layers: int = 2
    entanglement: str = "linear"
    cost_hamiltonian: Optional[SparsePauliOp] = field(default=None, compare=False)
    insert_barriers: bool = True
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CircuitBuildResult:
    """
    Return type of CircuitBuilder.build().

    Attributes
    ----------
    circuit:
        The assembled QuantumCircuit (possibly with unbound Parameters).
    parameter_vector:
        Ordered ParameterVector; use this to bind values before execution.
    n_parameters:
        Total number of free parameters.
    circuit_depth:
        Transpiled (or pre-transpile) circuit depth.
    metadata:
        Arbitrary key-value pairs (config snapshot, ansatz name, etc.).
    """

    circuit: QuantumCircuit
    parameter_vector: ParameterVector
    n_parameters: int
    circuit_depth: int
    metadata: dict


# ---------------------------------------------------------------------------
# CircuitBuilder
# ---------------------------------------------------------------------------
class CircuitBuilder:
    """
    Construct parameterised quantum circuits for QAOA and VQE workflows.

    Usage
    -----
    >>> cfg = CircuitConfig(n_qubits=4, circuit_type=CircuitType.VQE)
    >>> builder = CircuitBuilder(cfg)
    >>> result = builder.build()
    >>> qc = result.circuit
    >>> bound = qc.assign_parameters(dict(zip(result.parameter_vector, values)))
    """

    def __init__(self, config: CircuitConfig) -> None:
        self._cfg = config
        self._validate_config()
        logger.info(
            "CircuitBuilder initialised",
            extra={
                "n_qubits": config.n_qubits,
                "circuit_type": config.circuit_type,
                "ansatz_type": config.ansatz_type,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> CircuitBuildResult:
        """
        Assemble and return the QuantumCircuit described by self._cfg.

        Returns
        -------
        CircuitBuildResult
            Contains circuit, parameter_vector, n_parameters, circuit_depth,
            and a metadata snapshot.

        Raises
        ------
        ValueError
            If the requested ansatz is not supported or configuration is
            internally inconsistent.
        """
        if self._cfg.circuit_type == CircuitType.QAOA:
            return self._build_qaoa()
        if self._cfg.circuit_type == CircuitType.VQE:
            return self._build_vqe()
        raise ValueError(f"Unsupported circuit_type: {self._cfg.circuit_type}")

    def bind_parameters(
        self,
        circuit: QuantumCircuit,
        parameter_vector: ParameterVector,
        values: Sequence[float],
    ) -> QuantumCircuit:
        """
        Return a copy of *circuit* with all parameters bound to *values*.

        Parameters
        ----------
        circuit:
            Parameterised QuantumCircuit from build().
        parameter_vector:
            The ParameterVector returned alongside the circuit.
        values:
            Numeric values to bind; must match len(parameter_vector).

        Returns
        -------
        QuantumCircuit
            A concrete (parameter-free) circuit ready for execution.
        """
        if len(values) != len(parameter_vector):
            raise ValueError(
                f"Expected {len(parameter_vector)} parameter values, "
                f"got {len(values)}."
            )
        binding = dict(zip(parameter_vector, values))
        return circuit.assign_parameters(binding)

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_qaoa(self) -> CircuitBuildResult:
        """Build a QAOA circuit using Qiskit's QAOAAnsatz or manual layers."""
        cfg = self._cfg

        if cfg.cost_hamiltonian is None:
            # Fall back to a placeholder ZZ-chain Hamiltonian for testing.
            logger.warning(
                "No cost_hamiltonian provided for QAOA; "
                "using placeholder ZZ-chain Hamiltonian."
            )
            pauli_list = [
                ("ZZ" + "I" * (cfg.n_qubits - 2), 1.0)
            ]
            cost_op = SparsePauliOp.from_list(pauli_list)
        else:
            cost_op = cfg.cost_hamiltonian

        qaoa = QAOAAnsatz(
            cost_operator=cost_op,
            reps=cfg.p_layers,
            insert_barriers=cfg.insert_barriers,
            name="QAOA_circuit",
        )

        param_vec = ParameterVector("θ", length=len(qaoa.parameters))
        # QAOAAnsatz already owns its parameters; expose them via ParameterVector
        # by building a mapping wrapper circuit only if needed.
        # For clean downstream usage we return the ansatz directly.
        qc = qaoa

        logger.info(
            "QAOA circuit built",
            extra={
                "p_layers": cfg.p_layers,
                "n_qubits": cfg.n_qubits,
                "n_parameters": len(qc.parameters),
                "depth": qc.depth(),
            },
        )

        return CircuitBuildResult(
            circuit=qc,
            parameter_vector=ParameterVector("θ", length=len(qc.parameters)),
            n_parameters=len(qc.parameters),
            circuit_depth=qc.depth(),
            metadata=self._build_metadata("QAOA"),
        )

    def _build_vqe(self) -> CircuitBuildResult:
        """Dispatch to the correct VQE ansatz factory."""
        dispatch = {
            AnsatzType.REAL_AMPLITUDES: self._build_real_amplitudes,
            AnsatzType.EFFICIENT_SU2: self._build_efficient_su2,
            AnsatzType.TWO_LOCAL: self._build_two_local,
            AnsatzType.CUSTOM_RY: self._build_custom_ry,
        }
        factory = dispatch.get(self._cfg.ansatz_type)
        if factory is None:
            raise ValueError(
                f"Unsupported VQE ansatz: {self._cfg.ansatz_type}. "
                f"Choose from {list(dispatch.keys())}."
            )
        return factory()

    def _build_real_amplitudes(self) -> CircuitBuildResult:
        cfg = self._cfg
        qc = RealAmplitudes(
            num_qubits=cfg.n_qubits,
            entanglement=cfg.entanglement,
            reps=cfg.reps,
            insert_barriers=cfg.insert_barriers,
        )
        return self._wrap_library_circuit(qc, "RealAmplitudes")

    def _build_efficient_su2(self) -> CircuitBuildResult:
        cfg = self._cfg
        qc = EfficientSU2(
            num_qubits=cfg.n_qubits,
            entanglement=cfg.entanglement,
            reps=cfg.reps,
            insert_barriers=cfg.insert_barriers,
        )
        return self._wrap_library_circuit(qc, "EfficientSU2")

    def _build_two_local(self) -> CircuitBuildResult:
        cfg = self._cfg
        qc = TwoLocal(
            num_qubits=cfg.n_qubits,
            rotation_blocks=["ry", "rz"],
            entanglement_blocks="cx",
            entanglement=cfg.entanglement,
            reps=cfg.reps,
            insert_barriers=cfg.insert_barriers,
        )
        return self._wrap_library_circuit(qc, "TwoLocal")

    def _build_custom_ry(self) -> CircuitBuildResult:
        """
        Hand-built layered Ry + CX ansatz for full parameter visibility.

        Layer structure (repeated `reps` times):
            1. Ry rotation on every qubit.
            2. Ladder of CX gates (q_i → q_{i+1}).
        Final layer: Ry rotation on every qubit.
        """
        cfg = self._cfg
        n = cfg.n_qubits
        total_params = n * (cfg.reps + 1)
        pv = ParameterVector("θ", length=total_params)
        param_idx = 0

        qr = QuantumRegister(n, name="q")
        qc = QuantumCircuit(qr, name="custom_ry_ansatz")

        for rep in range(cfg.reps):
            # Rotation layer
            for i in range(n):
                qc.ry(pv[param_idx], qr[i])
                param_idx += 1
            if cfg.insert_barriers:
                qc.barrier()
            # Entanglement layer
            for i in range(n - 1):
                qc.cx(qr[i], qr[i + 1])
            if cfg.insert_barriers:
                qc.barrier()

        # Final rotation layer
        for i in range(n):
            qc.ry(pv[param_idx], qr[i])
            param_idx += 1

        logger.info(
            "Custom RY ansatz built",
            extra={
                "n_qubits": n,
                "reps": cfg.reps,
                "n_parameters": total_params,
                "depth": qc.depth(),
            },
        )

        return CircuitBuildResult(
            circuit=qc,
            parameter_vector=pv,
            n_parameters=total_params,
            circuit_depth=qc.depth(),
            metadata=self._build_metadata("custom_ry"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _wrap_library_circuit(
        self, qc: QuantumCircuit, label: str
    ) -> CircuitBuildResult:
        """Package a Qiskit library ansatz into a CircuitBuildResult."""
        params = list(qc.parameters)
        pv = ParameterVector("θ", length=len(params))

        logger.info(
            "%s ansatz built",
            label,
            extra={
                "n_qubits": self._cfg.n_qubits,
                "reps": self._cfg.reps,
                "entanglement": self._cfg.entanglement,
                "n_parameters": len(params),
                "depth": qc.depth(),
            },
        )

        return CircuitBuildResult(
            circuit=qc,
            parameter_vector=pv,
            n_parameters=len(params),
            circuit_depth=qc.depth(),
            metadata=self._build_metadata(label),
        )

    def _build_metadata(self, label: str) -> dict:
        cfg = self._cfg
        return {
            "ansatz": label,
            "n_qubits": cfg.n_qubits,
            "circuit_type": cfg.circuit_type.value,
            "reps": cfg.reps,
            "p_layers": cfg.p_layers,
            "entanglement": cfg.entanglement,
            "insert_barriers": cfg.insert_barriers,
            "seed": cfg.seed,
        }

    def _validate_config(self) -> None:
        cfg = self._cfg
        MAX_QUBITS = 20  # quantum.backend.max_qubits per §9.3
        if not (1 <= cfg.n_qubits <= MAX_QUBITS):
            raise ValueError(
                f"n_qubits must be in [1, {MAX_QUBITS}]; got {cfg.n_qubits}."
            )
        if cfg.reps < 1:
            raise ValueError(f"reps must be ≥ 1; got {cfg.reps}.")
        if cfg.p_layers < 1:
            raise ValueError(f"p_layers must be ≥ 1; got {cfg.p_layers}.")
        if cfg.ansatz_type.value not in SUPPORTED_ANSATZ_TYPES:
            raise ValueError(
                f"Unsupported ansatz_type '{cfg.ansatz_type}'. "
                f"Supported: {SUPPORTED_ANSATZ_TYPES}."
            )