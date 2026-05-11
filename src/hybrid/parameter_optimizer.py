"""
src/hybrid/parameter_optimizer.py
===================================
Team T-4: Hybrid Pipeline Team
MASTER BLUEPRINT v1.0 — §3.4

Responsibility:
    Classical outer-loop optimizer (COBYLA / SPSA / Adam) that tunes the
    variational parameters of a parameterized quantum circuit produced by
    src/quantum/circuit_builder.py.

Inputs:  ParameterizedCircuit (Qiskit QuantumCircuit), cost_fn (callable)
Outputs: OptimizedParams dataclass

Integration points:
    - Consumes T-3 artifacts: ParameterizedCircuit from circuit_builder.py
    - Wraps cost_fn supplied by qaoa_optimizer.py / vqe_solver.py expectation-value loops
    - Produces OptimizedParams consumed by hybrid_pipeline.py and benchmark_runner.py
    - Uses shared/logger.get_logger, shared/serializer.save_artifact
    - Config keys sourced from config/quantum_config.yaml (Blueprint §9.3)

Blueprint data-flow stage: Stage 06 — Quantum Optimization (classical outer loop)
Config:  quantum.qaoa.optimizer / quantum.vqe.optimizer
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from shared.logger import get_logger
from shared.serializer import save_artifact

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — all numeric defaults traceable to config (Blueprint §9 rule)
# ---------------------------------------------------------------------------
DEFAULT_OPTIMIZER: str = "COBYLA"          # matches quantum_config.yaml qaoa.optimizer
DEFAULT_MAX_ITER: int = 100                # matches quantum_config.yaml qaoa.max_iter
DEFAULT_TOL: float = 1e-6
DEFAULT_LEARNING_RATE: float = 0.01       # Adam / SPSA base learning rate
DEFAULT_SPSA_A: float = 0.602             # SPSA stability exponent α
DEFAULT_SPSA_C: float = 0.101             # SPSA noise exponent γ
DEFAULT_ADAM_BETA1: float = 0.9
DEFAULT_ADAM_BETA2: float = 0.999
DEFAULT_ADAM_EPSILON: float = 1e-8
CONVERGENCE_WINDOW: int = 10             # iterations to check convergence plateau


# ---------------------------------------------------------------------------
# Supported optimizer identifiers (Blueprint §9.3: COBYLA | SPSA | Adam)
# ---------------------------------------------------------------------------

class OptimizerType(str, Enum):
    COBYLA = "COBYLA"
    SPSA = "SPSA"
    ADAM = "Adam"


# ---------------------------------------------------------------------------
# Public output contract (Blueprint §5.2 style)
# ---------------------------------------------------------------------------

@dataclass
class OptimizedParams:
    """
    Result contract returned to hybrid_pipeline.py and T-3 quantum modules.

    Fields
    ------
    run_id          : Unique pipeline run identifier.
    optimizer_type  : Which optimizer was used.
    optimal_params  : List[float] — best variational parameters found.
    final_cost      : float — cost-function value at optimal_params.
    n_iterations    : int — number of cost-function evaluations performed.
    converged       : bool — whether a convergence criterion was met.
    cost_history    : List[float] — cost value at each iteration (for MLflow logging).
    wall_time_s     : float — wall-clock time in seconds.
    metadata        : Dict — optimizer hyperparams, circuit info, provenance.
    """
    run_id: str
    optimizer_type: OptimizerType
    optimal_params: List[float]
    final_cost: float
    n_iterations: int
    converged: bool
    cost_history: List[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer implementations
# ---------------------------------------------------------------------------

class _COBYLAOptimizer:
    """
    Constrained Optimisation BY Linear Approximations — derivative-free,
    well-suited for noisy quantum expectation-value landscapes.
    Uses scipy.optimize.minimize under the hood.
    """

    def __init__(self, max_iter: int = DEFAULT_MAX_ITER, tol: float = DEFAULT_TOL) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, bool, List[float]]:
        from scipy.optimize import minimize  # optional heavy import isolated here

        cost_history: List[float] = []

        def tracked_cost(params: np.ndarray) -> float:
            val = float(cost_fn(params))
            cost_history.append(val)
            return val

        result = minimize(
            tracked_cost,
            initial_params,
            method="COBYLA",
            options={"maxiter": self.max_iter, "rhobeg": self.tol},
        )

        converged = result.success or (
            len(cost_history) >= CONVERGENCE_WINDOW
            and np.std(cost_history[-CONVERGENCE_WINDOW:]) < self.tol
        )
        return result.x, float(result.fun), len(cost_history), converged, cost_history


class _SPSAOptimizer:
    """
    Simultaneous Perturbation Stochastic Approximation.
    Efficient for high-dimensional noisy landscapes; default for VQE
    (quantum_config.yaml: vqe.optimizer = SPSA).
    """

    def __init__(
        self,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
        a: float = DEFAULT_SPSA_A,
        c: float = DEFAULT_SPSA_C,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.a = a
        self.c = c
        self.learning_rate = learning_rate

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
    ) -> Tuple[np.ndarray, float, int, bool, List[float]]:
        params = initial_params.copy().astype(float)
        n_params = len(params)
        cost_history: List[float] = []
        best_params = params.copy()
        best_cost = float("inf")

        for k in range(1, self.max_iter + 1):
            # Gain sequences: a_k, c_k
            a_k = self.learning_rate / (k ** self.a)
            c_k = self.tol / (k ** self.c)

            # Random ±1 Bernoulli perturbation vector
            delta = np.where(np.random.rand(n_params) > 0.5, 1.0, -1.0)

            cost_plus = float(cost_fn(params + c_k * delta))
            cost_minus = float(cost_fn(params - c_k * delta))

            # Gradient estimate
            g_hat = (cost_plus - cost_minus) / (2.0 * c_k * delta)

            params = params - a_k * g_hat

            current_cost = float(cost_fn(params))
            cost_history.append(current_cost)

            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()

            # Convergence check over rolling window
            if (
                k >= CONVERGENCE_WINDOW
                and np.std(cost_history[-CONVERGENCE_WINDOW:]) < self.tol * 1e-2
            ):
                logger.info(
                    "SPSA converged",
                    extra={"iteration": k, "final_cost": best_cost, "n_evals": len(cost_history)},
                )
                return best_params, best_cost, len(cost_history), True, cost_history

        converged = np.std(cost_history[-CONVERGENCE_WINDOW:]) < self.tol
        return best_params, best_cost, len(cost_history), converged, cost_history


class _AdamOptimizer:
    """
    Adaptive Moment Estimation — gradient-based; suitable when analytic or
    finite-difference gradients are available (e.g. parameter-shift rule).
    """

    def __init__(
        self,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        beta1: float = DEFAULT_ADAM_BETA1,
        beta2: float = DEFAULT_ADAM_BETA2,
        epsilon: float = DEFAULT_ADAM_EPSILON,
    ) -> None:
        self.max_iter = max_iter
        self.tol = tol
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, float, int, bool, List[float]]:
        params = initial_params.copy().astype(float)
        n_params = len(params)
        m = np.zeros(n_params)   # 1st moment
        v = np.zeros(n_params)   # 2nd moment
        cost_history: List[float] = []
        best_params = params.copy()
        best_cost = float("inf")

        for k in range(1, self.max_iter + 1):
            # Gradient: use supplied grad_fn or finite-difference fallback
            if grad_fn is not None:
                g = grad_fn(params)
            else:
                g = self._finite_difference_grad(cost_fn, params)

            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g ** 2

            # Bias-corrected moments
            m_hat = m / (1 - self.beta1 ** k)
            v_hat = v / (1 - self.beta2 ** k)

            params = params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            current_cost = float(cost_fn(params))
            cost_history.append(current_cost)

            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()

            if (
                k >= CONVERGENCE_WINDOW
                and np.std(cost_history[-CONVERGENCE_WINDOW:]) < self.tol
            ):
                logger.info(
                    "Adam converged",
                    extra={"iteration": k, "final_cost": best_cost},
                )
                return best_params, best_cost, len(cost_history), True, cost_history

        converged = np.std(cost_history[-CONVERGENCE_WINDOW:]) < self.tol
        return best_params, best_cost, len(cost_history), converged, cost_history

    @staticmethod
    def _finite_difference_grad(
        cost_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
        epsilon: float = 1e-5,
    ) -> np.ndarray:
        """Central finite-difference gradient estimate."""
        grad = np.zeros_like(params)
        for i in range(len(params)):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += epsilon
            p_minus[i] -= epsilon
            grad[i] = (cost_fn(p_plus) - cost_fn(p_minus)) / (2.0 * epsilon)
        return grad


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------

class ParameterOptimizer:
    """
    Classical outer-loop optimizer for variational quantum circuit parameters.

    Selects and runs COBYLA, SPSA, or Adam based on config / explicit argument,
    then wraps the result in OptimizedParams for downstream consumers.

    Typical caller: hybrid_pipeline.py, qaoa_optimizer.py (T-3)

    Example
    -------
    >>> from src.hybrid.parameter_optimizer import ParameterOptimizer, OptimizerType
    >>> optimizer = ParameterOptimizer(
    ...     optimizer_type=OptimizerType.COBYLA,
    ...     max_iter=100,
    ...     run_id="abc123",
    ... )
    >>> result = optimizer.optimize(
    ...     cost_fn=expectation_value_fn,   # callable: params -> float
    ...     initial_params=np.random.uniform(-np.pi, np.pi, n_params),
    ... )
    >>> print(result.optimal_params, result.final_cost)
    """

    def __init__(
        self,
        optimizer_type: OptimizerType | str = OptimizerType.COBYLA,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float = DEFAULT_TOL,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        spsa_a: float = DEFAULT_SPSA_A,
        spsa_c: float = DEFAULT_SPSA_C,
        adam_beta1: float = DEFAULT_ADAM_BETA1,
        adam_beta2: float = DEFAULT_ADAM_BETA2,
        run_id: str = "unknown",
    ) -> None:
        self.optimizer_type = OptimizerType(optimizer_type)
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.spsa_a = spsa_a
        self.spsa_c = spsa_c
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.run_id = run_id

    def optimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        save: bool = False,
    ) -> OptimizedParams:
        """
        Run the selected classical optimizer against the provided cost function.

        Parameters
        ----------
        cost_fn        : Callable mapping a 1-D parameter array to a scalar cost.
                         Typically the QAOA / VQE expectation-value function from T-3.
        initial_params : Starting point in parameter space (np.ndarray, 1-D float).
        grad_fn        : Optional gradient oracle (used by Adam only).
                         If None and Adam is selected, finite differences are used.
        save           : If True, persist OptimizedParams via shared/serializer.

        Returns
        -------
        OptimizedParams
        """
        logger.info(
            "Starting parameter optimization",
            extra={
                "run_id": self.run_id,
                "optimizer": self.optimizer_type.value,
                "n_params": len(initial_params),
                "max_iter": self.max_iter,
            },
        )

        t_start = time.perf_counter()

        optimal_params, final_cost, n_iters, converged, history = self._dispatch(
            cost_fn=cost_fn,
            initial_params=np.asarray(initial_params, dtype=float),
            grad_fn=grad_fn,
        )

        wall_time = time.perf_counter() - t_start

        result = OptimizedParams(
            run_id=self.run_id,
            optimizer_type=self.optimizer_type,
            optimal_params=optimal_params.tolist(),
            final_cost=final_cost,
            n_iterations=n_iters,
            converged=converged,
            cost_history=history,
            wall_time_s=round(wall_time, 4),
            metadata={
                "optimizer": self.optimizer_type.value,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "learning_rate": self.learning_rate,
                "n_initial_params": len(initial_params),
            },
        )

        logger.info(
            "Parameter optimization complete",
            extra={
                "run_id": self.run_id,
                "final_cost": final_cost,
                "n_iterations": n_iters,
                "converged": converged,
                "wall_time_s": result.wall_time_s,
            },
        )

        if save:
            artifact_path = f"outputs/models/{self.run_id}_optimized_params.pkl"
            save_artifact(result, artifact_path, fmt="pkl")

        return result

    # ------------------------------------------------------------------
    # Internal dispatcher
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        cost_fn: Callable[[np.ndarray], float],
        initial_params: np.ndarray,
        grad_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    ) -> Tuple[np.ndarray, float, int, bool, List[float]]:
        """Instantiate and run the chosen optimizer backend."""
        if self.optimizer_type == OptimizerType.COBYLA:
            opt = _COBYLAOptimizer(max_iter=self.max_iter, tol=self.tol)
            return opt.minimize(cost_fn, initial_params)

        elif self.optimizer_type == OptimizerType.SPSA:
            opt = _SPSAOptimizer(
                max_iter=self.max_iter,
                tol=self.tol,
                a=self.spsa_a,
                c=self.spsa_c,
                learning_rate=self.learning_rate,
            )
            return opt.minimize(cost_fn, initial_params)

        elif self.optimizer_type == OptimizerType.ADAM:
            opt = _AdamOptimizer(
                max_iter=self.max_iter,
                tol=self.tol,
                learning_rate=self.learning_rate,
                beta1=self.adam_beta1,
                beta2=self.adam_beta2,
            )
            return opt.minimize(cost_fn, initial_params, grad_fn=grad_fn)

        else:  # pragma: no cover
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def optimize_circuit_parameters(
    cost_fn: Callable[[np.ndarray], float],
    initial_params: np.ndarray,
    run_id: str,
    optimizer_type: str = DEFAULT_OPTIMIZER,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    grad_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    save: bool = False,
) -> OptimizedParams:
    """
    Functional convenience wrapper around ParameterOptimizer.optimize().

    Consumed by hybrid_pipeline.py (T-4) and optionally by qaoa_optimizer.py /
    vqe_solver.py (T-3) when they delegate outer-loop optimization to T-4.

    Parameters
    ----------
    cost_fn        : Callable — expectation-value / energy function.
    initial_params : np.ndarray — initial variational parameters.
    run_id         : str — unique pipeline run ID (Blueprint §6).
    optimizer_type : str — "COBYLA" | "SPSA" | "Adam" (from quantum_config.yaml).
    max_iter       : int — maximum optimizer iterations.
    tol            : float — convergence tolerance.
    learning_rate  : float — step size (SPSA / Adam only).
    grad_fn        : Optional gradient oracle (Adam only).
    save           : bool — persist result artifact.

    Returns
    -------
    OptimizedParams

    Example
    -------
    >>> from src.hybrid.parameter_optimizer import optimize_circuit_parameters
    >>> result = optimize_circuit_parameters(
    ...     cost_fn=lambda p: expectation_value(circuit, p, backend),
    ...     initial_params=np.zeros(2 * p_layers * n_qubits),
    ...     run_id="abc123",
    ...     optimizer_type="COBYLA",
    ...     max_iter=100,
    ... )
    >>> print(result.converged, result.final_cost)
    """
    optimizer = ParameterOptimizer(
        optimizer_type=optimizer_type,
        max_iter=max_iter,
        tol=tol,
        learning_rate=learning_rate,
        run_id=run_id,
    )
    return optimizer.optimize(
        cost_fn=cost_fn,
        initial_params=initial_params,
        grad_fn=grad_fn,
        save=save,
    )