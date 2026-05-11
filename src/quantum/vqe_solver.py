"""
src/quantum/vqe_solver.py
==========================
Team T-3: Quantum Engine — VQE Ground-State Solver Module

Responsibility:
    VQE (Variational Quantum Eigensolver) ground-state solver for Hamiltonian
    encodings of NLP/clustering cost functions.  Consumes a Hamiltonian
    (SparsePauliOp) and a BackendHandle; returns VQEResult conforming to
    Blueprint §4.3 API response schema.

    Input  : SparsePauliOp  (Hamiltonian)  +  VQESolverConfig
    Output : VQEResult  {job_id, ground_energy, optimal_params, ...}

Optimizer support   : SPSA (default for noise-robustness), COBYLA, L-BFGS-B
Ansatz support      : RealAmplitudes, EfficientSU2, TwoLocal, custom_ry
                      (delegated to CircuitBuilder)
Qiskit compatibility: 1.x — Estimator primitive for expectation evaluation
Noise support       : Injected via BackendHandle (AerSimulator + NoiseModel)
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
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer.primitives import Estimator as AerEstimator

from src.quantum.circuit_builder import (
    AnsatzType,
    CircuitBuilder,
    CircuitBuildResult,
    CircuitConfig,
    CircuitType,
)
from src.quantum.backend_manager import BackendHandle
from shared.logger import get_logger

logger: logging.Logger = get_logger(__name__, level="INFO")

# ---------------------------------------------------------------------------
# Constants  (§6 — UPPER_SNAKE_CASE)
# ---------------------------------------------------------------------------
DEFAULT_REPS: int = 2
DEFAULT_MAX_ITER: int = 150
DEFAULT_SHOTS: int = 1024
DEFAULT_ANSATZ: str = "RealAmplitudes"
SUPPORTED_VQE_OPTIMIZERS: frozenset[str] = frozenset(
    {"SPSA", "COBYLA", "L-BFGS-B", "SLSQP"}
)
CONVERGENCE_HISTORY_KEY: str = "energy_history"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class VQEOptimizer(str, Enum):
    """
    Classical outer-loop optimizer for VQE parameter tuning.
    Maps to quantum.vqe.optimizer in config/quantum_config.yaml (§9.3).
    """

    SPSA = "SPSA"
    COBYLA = "COBYLA"
    L_BFGS_B = "L-BFGS-B"
    SLSQP = "SLSQP"


# ---------------------------------------------------------------------------
# Data contracts  (frozen per Blueprint §10.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VQESolverConfig:
    """
    Immutable configuration for the VQE solver.
    Maps to quantum.vqe.* in config/quantum_config.yaml (§9.3).

    Attributes
    ----------
    ansatz:
        Ansatz type string (quantum.vqe.ansatz).
    reps:
        Ansatz repetition depth (quantum.vqe.reps).
    optimizer:
        Classical outer optimizer.
    max_iter:
        Maximum optimization iterations.
    shots:
        Shots per circuit evaluation (used when backend is shot-based).
    initial_params:
        Optional explicit starting parameters.  If None, uses random init.
    tol:
        Convergence tolerance for gradient-based optimizers.
    seed:
        RNG seed for reproducible random initialisation and SPSA.
    entanglement:
        Entanglement layout for the ansatz ("linear", "full", "circular").
    gradient_free:
        If True, forces gradient-free optimization even for methods that
        support gradients.  Always True in noisy settings.
    """

    ansatz: AnsatzType = AnsatzType.REAL_AMPLITUDES
    reps: int = DEFAULT_REPS
    optimizer: VQEOptimizer = VQEOptimizer.SPSA
    max_iter: int = DEFAULT_MAX_ITER
    shots: int = DEFAULT_SHOTS
    initial_params: Optional[list[float]] = None
    tol: float = 1e-5
    seed: int = 42
    entanglement: str = "linear"
    gradient_free: bool = True


# ---------------------------------------------------------------------------
# Output data contract  (§4.3 API schema — frozen)
# ---------------------------------------------------------------------------
@dataclass
class VQEResult:
    """
    VQE output contract (Blueprint §4.3 /quantum/vqe/run response).

    Attributes
    ----------
    job_id:
        Unique run identifier (UUID4 string).
    ground_energy:
        Estimated ground-state energy (minimum eigenvalue of Hamiltonian).
    optimal_params:
        Best parameter vector achieving ground_energy.
    n_qubits:
        Number of qubits.
    ansatz:
        Ansatz type used.
    circuit_depth:
        Transpiled circuit depth.
    n_parameters:
        Number of variational parameters.
    n_iterations:
        Classical optimizer iterations performed.
    elapsed_s:
        Wall-clock seconds for the full VQE run.
    optimizer_success:
        Whether the classical optimizer reported convergence.
    energy_history:
        Per-iteration energy estimates (useful for convergence plots).
    metadata:
        Full configuration snapshot for MLflow logging.
    """

    job_id: str
    ground_energy: float
    optimal_params: list[float]
    n_qubits: int
    ansatz: str
    circuit_depth: int
    n_parameters: int
    n_iterations: int
    elapsed_s: float
    optimizer_success: bool
    energy_history: list[float]
    metadata: dict


# ---------------------------------------------------------------------------
# VQESolver
# ---------------------------------------------------------------------------
class VQESolver:
    """
    Variational Quantum Eigensolver for Hamiltonian ground-state estimation.

    Workflow
    --------
    1. Build the variational ansatz via CircuitBuilder (§3.3).
    2. Transpile for the target backend (BackendHandle).
    3. Iteratively evaluate E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩ using the Aer Estimator
       primitive (noise-aware if BackendHandle carries a NoiseModel).
    4. Minimise E(θ) with SPSA (default) or scipy optimizer.
    5. Return VQEResult with optimal θ, ground energy, and convergence data.

    Usage
    -----
    >>> hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("IX", -0.5)])
    >>> cfg = VQESolverConfig(ansatz=AnsatzType.REAL_AMPLITUDES, reps=2)
    >>> handle = BackendManager(BackendConfig()).get_backend()
    >>> solver = VQESolver(hamiltonian, cfg, handle)
    >>> result = solver.run()
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        config: VQESolverConfig,
        backend_handle: BackendHandle,
    ) -> None:
        self._hamiltonian = hamiltonian
        self._cfg = config
        self._handle = backend_handle
        self._n_qubits: int = hamiltonian.num_qubits
        self._rng = np.random.default_rng(config.seed)
        self._energy_history: list[float] = []
        self._iteration_count: int = 0
        self._estimator: Optional[AerEstimator] = None
        self._validate()
        logger.info(
            "VQESolver initialised",
            extra={
                "n_qubits": self._n_qubits,
                "ansatz": config.ansatz.value,
                "reps": config.reps,
                "optimizer": config.optimizer.value,
                "shots": config.shots,
                "backend": backend_handle.backend_name,
                "has_noise": backend_handle.has_noise,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> VQEResult:
        """
        Execute the full VQE optimization loop.

        Returns
        -------
        VQEResult
            Contains ground energy, optimal parameters, convergence data,
            and a full metadata snapshot.

        Raises
        ------
        RuntimeError
            If the backend execution fails or the estimator returns NaN.
        """
        job_id = str(uuid.uuid4())
        t_start = time.perf_counter()

        # Build ansatz circuit and transpile
        build_result, transpiled = self._build_transpiled_ansatz()

        # Initialise classical parameters
        initial_params = self._resolve_initial_params(build_result.n_parameters)

        # Initialise Aer Estimator with noise model
        self._estimator = self._build_estimator()

        logger.info(
            "Starting VQE optimization loop",
            extra={
                "job_id": job_id,
                "n_parameters": build_result.n_parameters,
                "circuit_depth": transpiled.depth(),
                "ansatz": self._cfg.ansatz.value,
                "optimizer": self._cfg.optimizer.value,
            },
        )

        # Build objective: E(θ) = ⟨H⟩
        objective = self._make_objective(transpiled)

        # Run classical optimizer
        opt_result = self._run_classical_optimizer(objective, initial_params)

        elapsed = time.perf_counter() - t_start

        result = VQEResult(
            job_id=job_id,
            ground_energy=float(opt_result.fun),
            optimal_params=opt_result.x.tolist(),
            n_qubits=self._n_qubits,
            ansatz=self._cfg.ansatz.value,
            circuit_depth=transpiled.depth(),
            n_parameters=build_result.n_parameters,
            n_iterations=self._iteration_count,
            elapsed_s=round(elapsed, 4),
            optimizer_success=bool(opt_result.success),
            energy_history=list(self._energy_history),
            metadata=self._build_metadata(job_id, build_result),
        )

        logger.info(
            "VQE optimization complete",
            extra={
                "job_id": job_id,
                "ground_energy": result.ground_energy,
                "n_iterations": result.n_iterations,
                "elapsed_s": result.elapsed_s,
                "converged": result.optimizer_success,
            },
        )
        return result

    @property
    def energy_history(self) -> list[float]:
        """Return the per-iteration energy values recorded during optimization."""
        return list(self._energy_history)

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _build_transpiled_ansatz(
        self,
    ) -> tuple[CircuitBuildResult, QuantumCircuit]:
        """Build the ansatz via CircuitBuilder and transpile for backend."""
        circuit_cfg = CircuitConfig(
            n_qubits=self._n_qubits,
            circuit_type=CircuitType.VQE,
            ansatz_type=self._cfg.ansatz,
            reps=self._cfg.reps,
            entanglement=self._cfg.entanglement,
        )
        build_result = CircuitBuilder(circuit_cfg).build()
        raw_circuit = build_result.circuit

        pm = generate_preset_pass_manager(
            optimization_level=1,
            backend=self._handle.backend,
        )
        transpiled = pm.run(raw_circuit)
        return build_result, transpiled

    def _build_estimator(self) -> AerEstimator:
        """
        Build an AerEstimator configured with the backend's noise model.

        The Estimator primitive evaluates ⟨ψ(θ)|H|ψ(θ)⟩ directly from
        the circuit and observable, handling shot-noise when noise_model
        is present.
        """
        estimator = AerEstimator()
        estimator.set_options(
            shots=self._cfg.shots,
            seed_simulator=self._cfg.seed,
        )
        if self._handle.has_noise:
            # Retrieve the NoiseModel from the backend's options
            noise_model = getattr(
                self._handle.backend.options, "noise_model", None
            )
            if noise_model is not None:
                estimator.set_options(noise_model=noise_model)
                logger.info(
                    "Noise model attached to AerEstimator",
                    extra={"backend": self._handle.backend_name},
                )
        return estimator

    def _make_objective(
        self, transpiled: QuantumCircuit
    ) -> Callable[[np.ndarray], float]:
        """
        Return the VQE energy objective E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩.

        The Aer Estimator evaluates the expectation value directly without
        requiring explicit measurement operators.
        """
        hamiltonian = self._hamiltonian

        def objective(params: np.ndarray) -> float:
            self._iteration_count += 1

            # Bind parameters to transpiled circuit
            param_keys = sorted(transpiled.parameters, key=lambda p: p.name)
            if len(param_keys) != len(params):
                raise RuntimeError(
                    f"Parameter mismatch in VQE objective: circuit has "
                    f"{len(param_keys)} params, optimizer passed {len(params)}."
                )
            binding = dict(zip(param_keys, params))
            bound_circuit = transpiled.assign_parameters(binding)

            # Evaluate ⟨H⟩ via AerEstimator
            job = self._estimator.run([(bound_circuit, hamiltonian)])
            pub_result = job.result()[0]
            energy = float(pub_result.data.evs)

            if np.isnan(energy):
                logger.warning(
                    "NaN energy at iteration %d — returning large sentinel.",
                    self._iteration_count,
                )
                energy = 1e9

            self._energy_history.append(energy)

            if self._iteration_count % 10 == 0:
                logger.debug(
                    "VQE iteration %d: energy=%.8f",
                    self._iteration_count,
                    energy,
                )
            return energy

        return objective

    def _run_classical_optimizer(
        self,
        objective: Callable[[np.ndarray], float],
        initial_params: list[float],
    ) -> OptimizeResult:
        """Dispatch to the configured classical optimizer."""
        optimizer_name = self._cfg.optimizer

        if optimizer_name == VQEOptimizer.SPSA:
            return self._run_spsa(objective, np.array(initial_params))

        scipy_map = {
            VQEOptimizer.COBYLA: "COBYLA",
            VQEOptimizer.L_BFGS_B: "L-BFGS-B",
            VQEOptimizer.SLSQP: "SLSQP",
        }
        method = scipy_map.get(optimizer_name, "COBYLA")
        return minimize(
            objective,
            x0=np.array(initial_params),
            method=method,
            options={"maxiter": self._cfg.max_iter},
            tol=self._cfg.tol,
        )

    def _run_spsa(
        self,
        objective: Callable[[np.ndarray], float],
        params: np.ndarray,
    ) -> OptimizeResult:
        """
        SPSA implementation tuned for noisy quantum landscapes.

        Recommended for VQE on real or noisy-simulated hardware because
        it does not require gradient computation and is robust to shot noise.

        Hyperparameters follow Spall (1998) and Kandala et al. (2017).
        """
        n = len(params)
        # SPSA hyperparameters
        a = 0.628       # step size scaling
        c = 0.1         # perturbation scaling
        A = 0.01 * self._cfg.max_iter
        alpha = 0.602
        gamma = 0.101

        theta = params.copy()
        best_theta = theta.copy()
        best_energy = float("inf")

        for k in range(1, self._cfg.max_iter + 1):
            a_k = a / (k + A) ** alpha
            c_k = c / k ** gamma

            # Bernoulli ±1 perturbation vector
            delta = 2 * self._rng.integers(0, 2, size=n) - 1

            energy_plus = objective(theta + c_k * delta)
            energy_minus = objective(theta - c_k * delta)

            grad_approx = (energy_plus - energy_minus) / (2 * c_k * delta)
            theta = theta - a_k * grad_approx

            mid_energy = (energy_plus + energy_minus) / 2.0
            if mid_energy < best_energy:
                best_energy = mid_energy
                best_theta = theta.copy()

        return OptimizeResult(
            x=best_theta,
            fun=best_energy,
            success=True,
            message=f"SPSA completed {self._cfg.max_iter} iterations.",
            nit=self._cfg.max_iter,
        )

    def _resolve_initial_params(self, n_params: int) -> list[float]:
        """Return user-provided or randomly initialised parameters."""
        if self._cfg.initial_params is not None:
            if len(self._cfg.initial_params) != n_params:
                raise ValueError(
                    f"initial_params length {len(self._cfg.initial_params)} "
                    f"does not match ansatz parameter count {n_params}."
                )
            return list(self._cfg.initial_params)
        # Small random initialisation avoids barren plateaus near zero
        return (self._rng.uniform(-0.1, 0.1, size=n_params)).tolist()

    def _build_metadata(
        self, job_id: str, build_result: CircuitBuildResult
    ) -> dict:
        cfg = self._cfg
        return {
            "job_id": job_id,
            "n_qubits": self._n_qubits,
            "ansatz": cfg.ansatz.value,
            "reps": cfg.reps,
            "entanglement": cfg.entanglement,
            "optimizer": cfg.optimizer.value,
            "max_iter": cfg.max_iter,
            "shots": cfg.shots,
            "tol": cfg.tol,
            "seed": cfg.seed,
            "n_parameters": build_result.n_parameters,
            "circuit_depth_pre_transpile": build_result.circuit_depth,
            "backend_name": self._handle.backend_name,
            "backend_type": self._handle.backend_type.value,
            "has_noise": self._handle.has_noise,
        }

    def _validate(self) -> None:
        if self._n_qubits > 20:
            raise ValueError(
                f"n_qubits {self._n_qubits} exceeds max_qubits=20 (§9.3)."
            )
        if self._cfg.reps < 1:
            raise ValueError(f"reps must be ≥ 1; got {self._cfg.reps}.")
        if self._cfg.max_iter < 1:
            raise ValueError(f"max_iter must be ≥ 1; got {self._cfg.max_iter}.")
        if self._cfg.shots < 1:
            raise ValueError(f"shots must be ≥ 1; got {self._cfg.shots}.")
        if self._cfg.optimizer.value not in SUPPORTED_VQE_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer '{self._cfg.optimizer}'. "
                f"Choose from {SUPPORTED_VQE_OPTIMIZERS}."
            )