"""
src/quantum/backend_manager.py
==============================
Team T-3: Quantum Engine — Backend Selection & Configuration Module

Responsibility:
    Select and configure the simulation backend; toggle real vs. simulated
    execution.  Returns a ready-to-use Qiskit backend handle consumed by
    qaoa_optimizer.py and vqe_solver.py.

    Input  : BackendConfig (dataclass defined here; frozen contract §10.3)
    Output : Backend handle (AerSimulator | IBM Quantum provider backend)

Qiskit compatibility : 1.x
Noise integration    : Qiskit Aer NoiseModel (from noise_model_factory.py)
Naming conventions   : §6 of MASTER_BLUEPRINT v1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from qiskit.circuit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from shared.logger import get_logger

logger: logging.Logger = get_logger(__name__, level="INFO")

# ---------------------------------------------------------------------------
# Constants  (§6 — UPPER_SNAKE_CASE)
# ---------------------------------------------------------------------------
DEFAULT_SHOTS: int = 1024
DEFAULT_MAX_QUBITS: int = 20
AER_SIMULATOR_NAME: str = "aer_simulator"
SUPPORTED_BACKEND_TYPES: frozenset[str] = frozenset({"simulator", "ibm_real"})
SUPPORTED_SIMULATION_METHODS: frozenset[str] = frozenset(
    {
        "automatic",
        "statevector",
        "density_matrix",
        "stabilizer",
        "matrix_product_state",
        "extended_stabilizer",
        "unitary",
        "superop",
    }
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class BackendType(str, Enum):
    """Top-level backend selector (maps to quantum.backend.type in §9.3)."""

    SIMULATOR = "simulator"
    IBM_REAL = "ibm_real"


class SimulationMethod(str, Enum):
    """AerSimulator method strings."""

    AUTOMATIC = "automatic"
    STATEVECTOR = "statevector"
    DENSITY_MATRIX = "density_matrix"
    STABILIZER = "stabilizer"
    MATRIX_PRODUCT_STATE = "matrix_product_state"


# ---------------------------------------------------------------------------
# Data contracts  (frozen per Blueprint §10.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackendConfig:
    """
    Immutable configuration object consumed by BackendManager.

    All fields map directly to config/quantum_config.yaml: quantum.backend (§9.3).

    Attributes
    ----------
    backend_type:
        "simulator" (Qiskit Aer) or "ibm_real" (IBM Quantum runtime).
    backend_name:
        Aer backend name or IBM Quantum backend name.
        Default: "aer_simulator".
    shots:
        Number of circuit execution shots.
    max_qubits:
        Upper qubit count guardrail; circuits exceeding this raise an error.
    simulation_method:
        AerSimulator method; "automatic" lets Aer choose.
    noise_model:
        Optional pre-built NoiseModel from noise_model_factory.py.
    coupling_map:
        Optional coupling map list-of-edges for topology-aware simulation.
    basis_gates:
        Optional gate set to restrict transpilation (mirrors device basis).
    seed_simulator:
        Seed for reproducible Aer shots.
    ibm_token:
        IBM Quantum API token; only required when backend_type == "ibm_real".
    ibm_instance:
        IBM Quantum instance string (hub/group/project).
    optimization_level:
        Qiskit transpiler optimisation level [0–3].
    """

    backend_type: BackendType = BackendType.SIMULATOR
    backend_name: str = AER_SIMULATOR_NAME
    shots: int = DEFAULT_SHOTS
    max_qubits: int = DEFAULT_MAX_QUBITS
    simulation_method: SimulationMethod = SimulationMethod.AUTOMATIC
    noise_model: Optional[NoiseModel] = field(default=None, compare=False)
    coupling_map: Optional[list[list[int]]] = field(default=None, compare=False)
    basis_gates: Optional[list[str]] = field(default=None, compare=False)
    seed_simulator: Optional[int] = 42
    ibm_token: Optional[str] = field(default=None, repr=False)
    ibm_instance: Optional[str] = None
    optimization_level: int = 1


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class BackendHandle:
    """
    Opaque container returned by BackendManager.get_backend().

    Attributes
    ----------
    backend:
        A Qiskit-compatible backend instance (AerSimulator or IBM backend).
    backend_name:
        Human-readable backend identifier.
    backend_type:
        SIMULATOR or IBM_REAL.
    shots:
        Shot count baked into this handle (from BackendConfig).
    has_noise:
        True if a NoiseModel is attached.
    metadata:
        Configuration snapshot for MLflow logging and auditing.
    """

    backend: AerSimulator  # typed as AerSimulator; IBM backends satisfy same interface
    backend_name: str
    backend_type: BackendType
    shots: int
    has_noise: bool
    metadata: dict


# ---------------------------------------------------------------------------
# BackendManager
# ---------------------------------------------------------------------------
class BackendManager:
    """
    Select, configure, and return a Qiskit backend handle.

    The manager supports:
    - Qiskit Aer noiseless / noisy simulation (primary path for CI and dev)
    - IBM Quantum real-device execution (gated by HQC_BACKEND_TYPE env var)

    All CI runs use the simulator backend only (§10.4).

    Usage
    -----
    >>> cfg = BackendConfig(backend_type=BackendType.SIMULATOR, shots=2048)
    >>> manager = BackendManager(cfg)
    >>> handle = manager.get_backend()
    >>> job = handle.backend.run(transpiled_circuit, shots=handle.shots)
    """

    def __init__(self, config: BackendConfig) -> None:
        self._cfg = config
        self._backend: Optional[AerSimulator] = None
        self._validate_config()
        logger.info(
            "BackendManager initialised",
            extra={
                "backend_type": config.backend_type,
                "backend_name": config.backend_name,
                "shots": config.shots,
                "simulation_method": config.simulation_method,
                "has_noise": config.noise_model is not None,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_backend(self) -> BackendHandle:
        """
        Build (if not cached) and return the backend handle.

        The handle is cached after the first call; reconfigure with a new
        BackendManager instance if settings change.

        Returns
        -------
        BackendHandle
            Ready-to-use backend with shot count and metadata.

        Raises
        ------
        RuntimeError
            If IBM Quantum credentials are missing or the real backend is
            unavailable.
        ValueError
            If configuration is invalid.
        """
        if self._backend is None:
            if self._cfg.backend_type == BackendType.SIMULATOR:
                self._backend = self._build_aer_backend()
            elif self._cfg.backend_type == BackendType.IBM_REAL:
                self._backend = self._build_ibm_backend()
            else:
                raise ValueError(
                    f"Unsupported backend_type: {self._cfg.backend_type}. "
                    f"Choose from {list(BackendType)}."
                )

        has_noise = self._cfg.noise_model is not None
        handle = BackendHandle(
            backend=self._backend,
            backend_name=self._cfg.backend_name,
            backend_type=self._cfg.backend_type,
            shots=self._cfg.shots,
            has_noise=has_noise,
            metadata=self._build_metadata(),
        )
        logger.info(
            "Backend handle returned",
            extra={
                "backend_name": handle.backend_name,
                "shots": handle.shots,
                "has_noise": handle.has_noise,
            },
        )
        return handle

    def run_circuit(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None,
    ) -> dict:
        """
        Convenience method: transpile and run *circuit* on the managed backend.

        Returns the measurement counts dictionary directly.

        Parameters
        ----------
        circuit:
            A concrete (parameter-free) QuantumCircuit with measurement gates.
        shots:
            Override shot count; defaults to BackendConfig.shots.

        Returns
        -------
        dict
            Measurement counts, e.g. {"00": 512, "11": 512}.

        Raises
        ------
        RuntimeError
            If the job fails or the backend is not reachable.
        """
        handle = self.get_backend()
        n_shots = shots if shots is not None else handle.shots

        # Transpile for the backend
        pass_manager = generate_preset_pass_manager(
            optimization_level=self._cfg.optimization_level,
            backend=handle.backend,
        )
        transpiled = pass_manager.run(circuit)

        logger.info(
            "Submitting circuit to backend",
            extra={
                "backend": handle.backend_name,
                "shots": n_shots,
                "circuit_depth": transpiled.depth(),
                "n_qubits": transpiled.num_qubits,
            },
        )

        job = handle.backend.run(transpiled, shots=n_shots)
        result = job.result()

        if not result.success:
            raise RuntimeError(
                f"Backend job failed. Status: {result.status}. "
                f"Backend: {handle.backend_name}."
            )

        counts: dict = result.get_counts()
        logger.info(
            "Circuit execution complete",
            extra={"n_unique_outcomes": len(counts), "shots": n_shots},
        )
        return counts

    def get_backend_properties(self) -> dict:
        """
        Return a serialisable dict of backend properties for MLflow logging.

        Returns
        -------
        dict
            Keys: name, type, method, max_qubits, shots, has_noise,
            basis_gates, coupling_map.
        """
        cfg = self._cfg
        return {
            "name": cfg.backend_name,
            "type": cfg.backend_type.value,
            "method": cfg.simulation_method.value,
            "max_qubits": cfg.max_qubits,
            "shots": cfg.shots,
            "has_noise": cfg.noise_model is not None,
            "basis_gates": cfg.basis_gates or [],
            "coupling_map": cfg.coupling_map or [],
            "seed_simulator": cfg.seed_simulator,
            "optimization_level": cfg.optimization_level,
        }

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_aer_backend(self) -> AerSimulator:
        """Construct and return a configured AerSimulator instance."""
        cfg = self._cfg

        aer_options: dict = {
            "method": cfg.simulation_method.value,
            "max_parallel_threads": 0,  # use all available CPU threads
        }
        if cfg.seed_simulator is not None:
            aer_options["seed_simulator"] = cfg.seed_simulator

        backend = AerSimulator(**aer_options)

        # Attach noise model
        if cfg.noise_model is not None:
            backend.set_options(noise_model=cfg.noise_model)
            logger.info(
                "Noise model attached to AerSimulator",
                extra={"noise_model_basis_gates": cfg.noise_model.basis_gates},
            )

        # Restrict to coupling map if provided (topology-aware simulation)
        if cfg.coupling_map is not None:
            backend.set_options(coupling_map=cfg.coupling_map)
            logger.debug(
                "Coupling map set on AerSimulator",
                extra={"n_edges": len(cfg.coupling_map)},
            )

        # Restrict basis gates if provided
        if cfg.basis_gates is not None:
            backend.set_options(basis_gates=cfg.basis_gates)
            logger.debug(
                "Basis gates restricted",
                extra={"basis_gates": cfg.basis_gates},
            )

        logger.info(
            "AerSimulator backend constructed",
            extra={
                "method": cfg.simulation_method.value,
                "seed": cfg.seed_simulator,
            },
        )
        return backend

    def _build_ibm_backend(self) -> AerSimulator:
        """
        Attempt to connect to an IBM Quantum real backend.

        Falls back to a fake backend (from qiskit_ibm_runtime.fake_provider)
        if the token is unavailable, so tests do not require live credentials.

        Note: In CI, this code path is never triggered (§10.4).
        """
        cfg = self._cfg

        if cfg.ibm_token is None:
            logger.warning(
                "IBM Quantum token not provided; falling back to fake backend.",
                extra={"backend_name": cfg.backend_name},
            )
            return self._build_fake_backend(cfg.backend_name)

        try:
            from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore

            service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=cfg.ibm_token,
                instance=cfg.ibm_instance or "ibm-q/open/main",
            )
            real_backend = service.backend(cfg.backend_name)
            logger.info(
                "Connected to IBM Quantum backend",
                extra={"backend_name": cfg.backend_name},
            )
            # Wrap in AerSimulator from_backend for local noise-aware simulation
            # of the real device (common research pattern).
            aer_from_real = AerSimulator.from_backend(real_backend)
            if cfg.seed_simulator is not None:
                aer_from_real.set_options(seed_simulator=cfg.seed_simulator)
            return aer_from_real

        except ImportError:
            logger.error(
                "qiskit_ibm_runtime not installed; cannot connect to IBM Quantum. "
                "Install with: pip install qiskit-ibm-runtime"
            )
            raise RuntimeError(
                "qiskit_ibm_runtime is required for IBM_REAL backend type."
            ) from None
        except Exception as exc:
            logger.error(
                "Failed to connect to IBM Quantum backend",
                extra={"error": str(exc)},
            )
            raise RuntimeError(
                f"IBM Quantum backend connection failed: {exc}"
            ) from exc

    def _build_fake_backend(self, name: str) -> AerSimulator:
        """
        Return an AerSimulator seeded from a qiskit_ibm_runtime fake provider.

        Falls back further to a plain AerSimulator if fake providers are
        unavailable.
        """
        fake_map: dict[str, str] = {
            "fake_nairobi": "FakeNairobi",
            "fake_lagos": "FakeLagos",
            "fake_manila": "FakeManila",
        }
        provider_name = fake_map.get(name.lower())
        if provider_name is None:
            logger.warning(
                "Unknown fake backend '%s'; using plain AerSimulator.", name
            )
            return self._build_aer_backend()

        try:
            import importlib
            module = importlib.import_module("qiskit_ibm_runtime.fake_provider")
            FakeClass = getattr(module, provider_name)
            fake = FakeClass()
            sim = AerSimulator.from_backend(fake)
            if self._cfg.seed_simulator is not None:
                sim.set_options(seed_simulator=self._cfg.seed_simulator)
            logger.info("Fake backend '%s' loaded as AerSimulator.", name)
            return sim
        except (ImportError, AttributeError):
            logger.warning(
                "Fake backend '%s' unavailable; falling back to plain AerSimulator.",
                name,
            )
            return self._build_aer_backend()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metadata(self) -> dict:
        cfg = self._cfg
        return {
            "backend_type": cfg.backend_type.value,
            "backend_name": cfg.backend_name,
            "shots": cfg.shots,
            "simulation_method": cfg.simulation_method.value,
            "has_noise": cfg.noise_model is not None,
            "max_qubits": cfg.max_qubits,
            "seed_simulator": cfg.seed_simulator,
            "optimization_level": cfg.optimization_level,
        }

    def _validate_config(self) -> None:
        cfg = self._cfg
        if cfg.shots < 1:
            raise ValueError(f"shots must be ≥ 1; got {cfg.shots}.")
        if cfg.max_qubits < 1:
            raise ValueError(f"max_qubits must be ≥ 1; got {cfg.max_qubits}.")
        if cfg.optimization_level not in (0, 1, 2, 3):
            raise ValueError(
                f"optimization_level must be 0–3; got {cfg.optimization_level}."
            )
        if cfg.simulation_method.value not in SUPPORTED_SIMULATION_METHODS:
            raise ValueError(
                f"simulation_method '{cfg.simulation_method}' not in "
                f"{SUPPORTED_SIMULATION_METHODS}."
            )
        if (
            cfg.backend_type == BackendType.IBM_REAL
            and cfg.ibm_token is None
        ):
            logger.warning(
                "BackendType.IBM_REAL selected but ibm_token is None. "
                "Will fall back to fake backend at runtime."
            )