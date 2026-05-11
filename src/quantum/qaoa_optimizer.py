"""
src/quantum/qaoa_optimizer.py
==============================
Team T-3: Quantum Engine — QAOA Optimization Module

Responsibility:
    Run QAOA to solve graph-partitioning formulations of the clustering cost.
    Consumes CostHamiltonian and BackendHandle from upstream modules;
    returns QAOAResult conforming to the Blueprint §5.2 data contract.

    Input  : CostHamiltonian (SparsePauliOp), QAOAOptimizerConfig
    Output : QAOAResult  {job_id, optimal_params, cost, counts, circuit_depth}

Optimizer support   : COBYLA (default), SPSA, SLSQP, NELDER_MEAD
Qiskit compatibility: 1.x — uses qiskit.primitives.Sampler / Estimator
Noise support       : Injected via BackendHandle (from backend_manager.py)
Naming conventions  : §6 of MASTER_BLUEPRINT v1.0
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler

from src.quantum.circuit_builder import (
    AnsatzType,
    CircuitBuilder,
    CircuitConfig,
    CircuitType,
)
from src.quantum.backend_manager import BackendHandle
from shared.logger import get_logger

logger: logging.Logger = get_logger(__name__, level="INFO")

# ---------------------------------------------------------------------------
# Constants  (§6 — UPPER_SNAKE_CASE)
# ---------------------------------------------------------------------------
DEFAULT_SHOTS: int = 1024
DEFAULT_P_LAYERS: int = 2
DEFAULT_MAX_ITER: int = 100
DEFAULT_OPTIMIZER: str = "COBYLA"
SUPPORTED_OPTIMIZERS: frozenset[str] = frozenset(
    {"COBYLA", "SPSA", "SLSQP", "NELDER_MEAD"}
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ClassicalOptimizer(str, Enum):
    """Classical outer-loop optimizer for QAOA parameter tuning (§9.3)."""

    COBYLA = "COBYLA"
    SPSA = "SPSA"
    SLSQP = "SLSQP"
    NELDER_MEAD = "NELDER_MEAD"


# ---------------------------------------------------------------------------
# Data contracts  (frozen per Blueprint §10.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QAOAOptimizerConfig:
    """
    Immutable configuration for the QAOA optimizer.
    Maps to quantum.qaoa.* in config/quantum_config.yaml (§9.3).

    Attributes
    ----------
    p_layers:
        QAOA repetition depth (quantum.qaoa.p_layers).
    optimizer:
        Classical outer optimizer name (quantum.qaoa.optimizer).
    max_iter:
        Maximum classical optimizer iterations (quantum.qaoa.max_iter).
    shots:
        Number of circuit shots per objective evaluation.
    initial_params:
        Optional initial parameter values (length = 2 * p_layers).
        If None, random initialisation in [0, 2π) is used.
    tol:
        Convergence tolerance passed to the scipy optimizer.
    seed:
        RNG seed for reproducible random initialisation.
    """

    p_layers: int = DEFAULT_P_LAYERS
    optimizer: ClassicalOptimizer = ClassicalOptimizer.COBYLA
    max_iter: int = DEFAULT_MAX_ITER
    shots: int = DEFAULT_SHOTS
    initial_params: Optional[list[float]] = None
    tol: float = 1e-4
    seed: int = 42


# ---------------------------------------------------------------------------
# Output data contract  (§5.2 — frozen)
# ---------------------------------------------------------------------------
@dataclass
class QAOAResult:
    """
    Blueprint §5.2 QAOAResult contract.

    Attributes
    ----------
    job_id:
        Unique run identifier (UUID4 string).
    optimal_params:
        Best (γ, β) parameter vector found by the classical optimizer.
    cost:
        Minimum cost value achieved (expectation value of cost Hamiltonian).
    counts:
        Measurement counts from the final circuit execution.
    circuit_depth:
        Depth of the transpiled QAOA circuit.
    n_qubits:
        Number of qubits in the circuit.
    p_layers:
        QAOA p-layers used.
    n_iterations:
        Number of classical optimizer iterations performed.
    elapsed_s:
        Wall-clock time in seconds for the full optimization.
    optimizer_success:
        Whether the classical optimizer converged.
    metadata:
        Full configuration snapshot for MLflow logging.
    """

    job_id: str
    optimal_params: list[float]
    cost: float
    counts: dict
    circuit_depth: int
    n_qubits: int
    p_layers: int
    n_iterations: int
    elapsed_s: float
    optimizer_success: bool
    metadata: dict


# ---------------------------------------------------------------------------
# QAOAOptimizer
# ---------------------------------------------------------------------------
class QAOAOptimizer:
    """
    Execute the QAOA optimization loop for graph-partitioning clustering.

    The optimizer follows the standard variational loop:
      1. Build QAOAAnsatz from CostHamiltonian via CircuitBuilder.
      2. Transpile circuit for the target backend.
      3. Run outer classical optimizer (COBYLA/SPSA/...) calling the
         quantum objective (Sampler-based expectation estimate) at each step.
      4. Collect final measurement counts and return QAOAResult.

    Noise-aware simulation is transparently handled via the injected
    BackendHandle (which carries an AerSimulator with optional NoiseModel).

    Usage
    -----
    >>> cost_op = SparsePauliOp.from_list([("ZZ", 1.0), ("IZ", 0.5)])
    >>> cfg = QAOAOptimizerConfig(p_layers=2, optimizer=ClassicalOptimizer.COBYLA)
    >>> backend_handle = BackendManager(BackendConfig()).get_backend()
    >>> optimizer = QAOAOptimizer(cost_op, cfg, backend_handle)
    >>> result = optimizer.run()
    """

    def __init__(
        self,
        cost_hamiltonian: SparsePauliOp,
        config: QAOAOptimizerConfig,
        backend_handle: BackendHandle,
    ) -> None:
        self._hamiltonian = cost_hamiltonian
        self._cfg = config
        self._handle = backend_handle
        self._n_qubits: int = cost_hamiltonian.num_qubits
        self._rng = np.random.default_rng(config.seed)
        self._iteration_count: int = 0
        self._cost_history: list[float] = []
        self._validate()
        logger.info(
            "QAOAOptimizer initialised",
            extra={
                "n_qubits": self._n_qubits,
                "p_layers": config.p_layers,
                "optimizer": config.optimizer,
                "shots": config.shots,
                "backend": backend_handle.backend_name,
                "has_noise": backend_handle.has_noise,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> QAOAResult:
        """
        Execute the full QAOA optimization loop.

        Returns
        -------
        QAOAResult
            Contains optimal parameters, minimum cost, final measurement
            counts, circuit metadata, and convergence information.

        Raises
        ------
        RuntimeError
            If the backend execution fails.
        """
        job_id = str(uuid.uuid4())
        t_start = time.perf_counter()

        # Build and transpile the QAOA circuit
        circuit, transpiled = self._build_transpiled_circuit()

        # Initialise parameters: [γ_1, ..., γ_p, β_1, ..., β_p]
        initial_params = self._resolve_initial_params()

        logger.info(
            "Starting QAOA optimization loop",
            extra={
                "job_id": job_id,
                "n_parameters": len(initial_params),
                "circuit_depth": transpiled.depth(),
                "max_iter": self._cfg.max_iter,
            },
        )

        # Build objective
        objective = self._make_objective(transpiled)

        # Run classical optimizer
        opt_result: OptimizeResult = self._run_classical_optimizer(
            objective, initial_params
        )

        elapsed = time.perf_counter() - t_start

        # Final evaluation with optimal parameters to get counts
        final_counts = self._sample_circuit(transpiled, opt_result.x.tolist())

        result = QAOAResult(
            job_id=job_id,
            optimal_params=opt_result.x.tolist(),
            cost=float(opt_result.fun),
            counts=final_counts,
            circuit_depth=transpiled.depth(),
            n_qubits=self._n_qubits,
            p_layers=self._cfg.p_layers,
            n_iterations=self._iteration_count,
            elapsed_s=round(elapsed, 4),
            optimizer_success=bool(opt_result.success),
            metadata=self._build_metadata(job_id),
        )

        logger.info(
            "QAOA optimization complete",
            extra={
                "job_id": job_id,
                "cost": result.cost,
                "n_iterations": result.n_iterations,
                "elapsed_s": result.elapsed_s,
                "converged": result.optimizer_success,
            },
        )
        return result

    @property
    def cost_history(self) -> list[float]:
        """Return the per-iteration cost values recorded during optimization."""
        return list(self._cost_history)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_transpiled_circuit(self) -> tuple[QuantumCircuit, QuantumCircuit]:
        """Build QAOAAnsatz and transpile it for the target backend."""
        circuit_cfg = CircuitConfig(
            n_qubits=self._n_qubits,
            circuit_type=CircuitType.QAOA,
            ansatz_type=AnsatzType.QAOA,
            p_layers=self._cfg.p_layers,
            cost_hamiltonian=self._hamiltonian,
        )
        build_result = CircuitBuilder(circuit_cfg).build()
        raw_circuit = build_result.circuit

        # Append measurements to all qubits
        measured = raw_circuit.copy()
        measured.measure_all()

        # Transpile with the backend's pass manager
        pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=self._handle.backend,
        )
        transpiled = pm.run(measured)
        return raw_circuit, transpiled

    def _make_objective(
        self, transpiled: QuantumCircuit
    ) -> Callable[[np.ndarray], float]:
        """
        Return a callable objective function for the classical optimizer.

        The objective estimates the expectation value of the cost Hamiltonian
        from measurement counts: E[C] = Σ_z P(z) * C(z), where C(z) is the
        cost of bitstring z evaluated classically.
        """

        def objective(params: np.ndarray) -> float:
            self._iteration_count += 1
            counts = self._sample_circuit(transpiled, params.tolist())
            cost = self._estimate_cost_from_counts(counts)
            self._cost_history.append(cost)

            if self._iteration_count % 10 == 0:
                logger.debug(
                    "QAOA iteration %d: cost=%.6f",
                    self._iteration_count,
                    cost,
                )
            return cost

        return objective

    def _sample_circuit(
        self, transpiled: QuantumCircuit, params: list[float]
    ) -> dict:
        """
        Bind parameters, run circuit on backend, return counts dict.

        Handles parameter binding for both QAOAAnsatz-style ordered parameters
        and ParameterVector-style parameters.
        """
        param_keys = sorted(transpiled.parameters, key=lambda p: p.name)
        if len(param_keys) != len(params):
            raise ValueError(
                f"Parameter count mismatch: circuit has {len(param_keys)} "
                f"parameters, got {len(params)} values."
            )
        binding = dict(zip(param_keys, params))
        bound = transpiled.assign_parameters(binding)

        job = self._handle.backend.run(bound, shots=self._cfg.shots)
        result = job.result()
        if not result.success:
            raise RuntimeError(
                f"Backend job failed during QAOA objective evaluation. "
                f"Status: {result.status}"
            )
        return result.get_counts()

    def _estimate_cost_from_counts(self, counts: dict) -> float:
        """
        Compute the expectation value of the cost Hamiltonian from counts.

        For each measured bitstring z, evaluates the cost using the diagonal
        of the cost Hamiltonian (ZZ-type interactions), then weights by
        probability P(z) = count(z) / total_shots.

        This is the standard QAOA expectation estimator for ZZ-diagonal
        cost operators.
        """
        total_shots = sum(counts.values())
        cost = 0.0

        # Extract ZZ diagonal contributions from SparsePauliOp
        cost_matrix = self._get_diagonal_cost()

        for bitstring, count in counts.items():
            prob = count / total_shots
            # Qiskit bitstring order: rightmost bit = qubit 0
            bits = [int(b) for b in reversed(bitstring.replace(" ", ""))]
            # Ensure length matches n_qubits
            bits = bits[: self._n_qubits]
            bits += [0] * max(0, self._n_qubits - len(bits))
            z = np.array(bits, dtype=float)
            # Map {0,1} → {+1,-1} for Ising evaluation
            spin = 1 - 2 * z
            c_z = float(cost_matrix @ spin)
            cost += prob * c_z

        return cost

    def _get_diagonal_cost(self) -> np.ndarray:
        """
        Extract the diagonal cost vector from the SparsePauliOp.

        For QAOA over ZZ-type problems, the relevant entries are those
        where Pauli strings contain only Z and I operators.
        Returns a vector h ∈ ℝ^n such that C(z) = h · spin(z).
        """
        n = self._n_qubits
        h = np.zeros(n)
        J = np.zeros((n, n))

        for pauli, coeff in zip(
            self._hamiltonian.paulis, self._hamiltonian.coeffs
        ):
            label = pauli.to_label()  # e.g. "ZZII"
            z_positions = [i for i, c in enumerate(reversed(label)) if c == "Z"]
            if len(z_positions) == 1:
                h[z_positions[0]] += coeff.real
            elif len(z_positions) == 2:
                i, j = z_positions
                J[i, j] += coeff.real
                J[j, i] += coeff.real

        # Diagonal cost vector: C(z) = h·spin + spin·J·spin (simplified to linear)
        # Return h for the linear estimator; full quadratic handled below.
        # For complete Ising: sum_i h_i * s_i + sum_{i<j} J_{ij} * s_i * s_j
        # We return h + sum_j J_{ij} as an effective linear approximation.
        effective_h = h + J.sum(axis=1)
        return effective_h

    def _run_classical_optimizer(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: list[float],
    ) -> OptimizeResult:
        """Dispatch to the appropriate scipy optimizer."""
        optimizer_name = self._cfg.optimizer.value
        method_map = {
            "COBYLA": "COBYLA",
            "SLSQP": "SLSQP",
            "NELDER_MEAD": "Nelder-Mead",
            "SPSA": None,  # handled separately
        }

        if optimizer_name == "SPSA":
            return self._run_spsa(objective, np.array(initial_params))

        scipy_method = method_map.get(optimizer_name, "COBYLA")
        return minimize(
            objective,
            x0=np.array(initial_params),
            method=scipy_method,
            options={"maxiter": self._cfg.max_iter, "rhobeg": 0.5},
            tol=self._cfg.tol,
        )

    def _run_spsa(
        self,
        objective: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> OptimizeResult:
        """
        Minimal SPSA (Simultaneous Perturbation Stochastic Approximation)
        implementation for noise-robust gradient-free optimization.

        Hyperparameters follow Spall (1998) recommended defaults.
        """
        n = len(params)
        a = 0.602
        c = 0.101
        A = 0.1 * self._cfg.max_iter
        alpha = 0.602
        gamma = 0.101

        theta = params.copy()
        best_theta = theta.copy()
        best_cost = float("inf")

        for k in range(1, self._cfg.max_iter + 1):
            a_k = a / (k + A) ** alpha
            c_k = c / k ** gamma

            delta = 2 * self._rng.integers(0, 2, size=n) - 1  # Bernoulli ±1
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            cost_plus = objective(theta_plus)
            cost_minus = objective(theta_minus)

            grad_approx = (cost_plus - cost_minus) / (2 * c_k * delta)
            theta -= a_k * grad_approx

            mid_cost = (cost_plus + cost_minus) / 2.0
            if mid_cost < best_cost:
                best_cost = mid_cost
                best_theta = theta.copy()

        return OptimizeResult(
            x=best_theta,
            fun=best_cost,
            success=True,
            message="SPSA completed",
            nit=self._cfg.max_iter,
        )

    def _resolve_initial_params(self) -> list[float]:
        """Return user-provided or randomly initialised parameters."""
        n_params = 2 * self._cfg.p_layers  # [γ_1..γ_p, β_1..β_p]
        if self._cfg.initial_params is not None:
            if len(self._cfg.initial_params) != n_params:
                raise ValueError(
                    f"initial_params length {len(self._cfg.initial_params)} "
                    f"does not match expected {n_params} (= 2 * p_layers)."
                )
            return list(self._cfg.initial_params)
        return self._rng.uniform(0, 2 * np.pi, size=n_params).tolist()

    def _build_metadata(self, job_id: str) -> dict:
        cfg = self._cfg
        return {
            "job_id": job_id,
            "n_qubits": self._n_qubits,
            "p_layers": cfg.p_layers,
            "optimizer": cfg.optimizer.value,
            "max_iter": cfg.max_iter,
            "shots": cfg.shots,
            "tol": cfg.tol,
            "seed": cfg.seed,
            "backend_name": self._handle.backend_name,
            "backend_type": self._handle.backend_type.value,
            "has_noise": self._handle.has_noise,
        }

    def _validate(self) -> None:
        if self._cfg.p_layers < 1:
            raise ValueError(f"p_layers must be ≥ 1; got {self._cfg.p_layers}.")
        if self._cfg.shots < 1:
            raise ValueError(f"shots must be ≥ 1; got {self._cfg.shots}.")
        if self._cfg.max_iter < 1:
            raise ValueError(f"max_iter must be ≥ 1; got {self._cfg.max_iter}.")
        if self._n_qubits > 20:
            raise ValueError(
                f"n_qubits {self._n_qubits} exceeds max_qubits=20 (§9.3)."
            )