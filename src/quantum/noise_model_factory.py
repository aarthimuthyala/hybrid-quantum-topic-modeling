"""
src/quantum/noise_model_factory.py
===================================
Team T-3: Quantum Engine — Noise Model Construction Module

Responsibility:
    Build Qiskit Aer NoiseModel instances from device specifications or
    synthetic profiles defined in config/noise_config.yaml (§9.4).

    Input  : NoiseConfig (frozen dataclass defined here)
    Output : qiskit_aer.noise.NoiseModel

Supported profiles (noise.profile §9.4):
    - depolarizing   : Pauli depolarizing channel on 1Q and 2Q gates
    - thermal        : Thermal relaxation (T1/T2) + depolarizing
    - device_fake    : Noise model extracted from a qiskit_ibm_runtime
                       fake backend (e.g. FakeNairobi)

Qiskit compatibility : 1.x + qiskit-aer >= 0.13
Naming conventions   : §6 of MASTER_BLUEPRINT v1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)
from qiskit_aer.noise.errors import QuantumError

from shared.logger import get_logger

logger: logging.Logger = get_logger(__name__, level="INFO")

# ---------------------------------------------------------------------------
# Constants  (§6 — UPPER_SNAKE_CASE)
# ---------------------------------------------------------------------------
DEFAULT_SINGLE_QUBIT_ERROR: float = 0.001   # noise.depolarizing.single_qubit_error
DEFAULT_TWO_QUBIT_ERROR: float = 0.01       # noise.depolarizing.two_qubit_error
DEFAULT_T1_NS: float = 50_000.0             # noise.thermal.t1  (nanoseconds)
DEFAULT_T2_NS: float = 70_000.0             # noise.thermal.t2  (nanoseconds)
DEFAULT_GATE_TIME_1Q_NS: float = 50.0       # typical single-qubit gate duration
DEFAULT_GATE_TIME_2Q_NS: float = 300.0      # typical two-qubit gate duration
DEFAULT_FAKE_BACKEND_NAME: str = "fake_nairobi"

# Gates to which 1Q depolarizing / thermal noise is applied
SINGLE_QUBIT_GATES: list[str] = ["u1", "u2", "u3", "rx", "ry", "rz", "h", "x", "y", "z", "s", "sdg", "t", "tdg"]
# Gates to which 2Q depolarizing / thermal noise is applied
TWO_QUBIT_GATES: list[str] = ["cx", "cz", "ecr", "rzz"]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class NoiseProfile(str, Enum):
    """
    Top-level noise profile selector.
    Maps to noise.profile in config/noise_config.yaml (§9.4).
    """

    DEPOLARIZING = "depolarizing"
    THERMAL = "thermal"
    DEVICE_FAKE = "device_fake"
    NONE = "none"  # Noiseless — returns an empty NoiseModel


# ---------------------------------------------------------------------------
# Data contracts  (frozen per Blueprint §10.3)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DepolarizingParams:
    """
    Parameters for the depolarizing noise profile.
    Maps to noise.depolarizing.* in noise_config.yaml.
    """

    single_qubit_error: float = DEFAULT_SINGLE_QUBIT_ERROR
    two_qubit_error: float = DEFAULT_TWO_QUBIT_ERROR


@dataclass(frozen=True)
class ThermalParams:
    """
    Parameters for the thermal relaxation noise profile.
    Maps to noise.thermal.* in noise_config.yaml.
    T1, T2 values are in nanoseconds.
    """

    t1: float = DEFAULT_T1_NS
    t2: float = DEFAULT_T2_NS
    gate_time_1q: float = DEFAULT_GATE_TIME_1Q_NS
    gate_time_2q: float = DEFAULT_GATE_TIME_2Q_NS
    # Readout error probabilities: p(0|1) and p(1|0)
    readout_error_p01: float = 0.02
    readout_error_p10: float = 0.02


@dataclass(frozen=True)
class DeviceFakeParams:
    """
    Parameters for the device_fake noise profile.
    Maps to noise.device_fake.* in noise_config.yaml.
    """

    backend_name: str = DEFAULT_FAKE_BACKEND_NAME


@dataclass(frozen=True)
class NoiseConfig:
    """
    Immutable configuration consumed by NoiseModelFactory.

    All fields map directly to config/noise_config.yaml (§9.4).

    Attributes
    ----------
    profile:
        Which noise model to construct.
    depolarizing:
        Parameters for the depolarizing profile (used when profile == DEPOLARIZING
        or as a component of the thermal profile).
    thermal:
        Parameters for the thermal relaxation profile.
    device_fake:
        Parameters for device-extracted noise models.
    n_qubits:
        Number of qubits the noise model is applied to; used for readout
        error registration across all qubits.
    """

    profile: NoiseProfile = NoiseProfile.DEPOLARIZING
    depolarizing: DepolarizingParams = field(default_factory=DepolarizingParams)
    thermal: ThermalParams = field(default_factory=ThermalParams)
    device_fake: DeviceFakeParams = field(default_factory=DeviceFakeParams)
    n_qubits: int = 4


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class NoiseModelResult:
    """
    Return type of NoiseModelFactory.build().

    Attributes
    ----------
    noise_model:
        The assembled Qiskit Aer NoiseModel, ready to be injected into
        BackendConfig.noise_model.
    profile:
        Which profile was used.
    basis_gates:
        Gates carrying noise channels in this model.
    metadata:
        Configuration snapshot for MLflow logging (§8.1 log_run_start).
    """

    noise_model: NoiseModel
    profile: NoiseProfile
    basis_gates: list[str]
    metadata: dict


# ---------------------------------------------------------------------------
# NoiseModelFactory
# ---------------------------------------------------------------------------
class NoiseModelFactory:
    """
    Build Qiskit Aer NoiseModel instances for noise-aware simulation.

    Usage
    -----
    >>> cfg = NoiseConfig(profile=NoiseProfile.DEPOLARIZING, n_qubits=4)
    >>> factory = NoiseModelFactory(cfg)
    >>> result = factory.build()
    >>> noise_model = result.noise_model
    # Inject into BackendConfig:
    >>> backend_cfg = BackendConfig(noise_model=noise_model)
    """

    def __init__(self, config: NoiseConfig) -> None:
        self._cfg = config
        self._validate_config()
        logger.info(
            "NoiseModelFactory initialised",
            extra={"profile": config.profile, "n_qubits": config.n_qubits},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> NoiseModelResult:
        """
        Construct and return the NoiseModel described by self._cfg.

        Returns
        -------
        NoiseModelResult
            Contains the NoiseModel, profile, basis_gates, and metadata.

        Raises
        ------
        ValueError
            For invalid parameter combinations.
        RuntimeError
            If a fake backend cannot be loaded (device_fake profile).
        """
        dispatch = {
            NoiseProfile.NONE: self._build_noiseless,
            NoiseProfile.DEPOLARIZING: self._build_depolarizing,
            NoiseProfile.THERMAL: self._build_thermal,
            NoiseProfile.DEVICE_FAKE: self._build_device_fake,
        }
        factory = dispatch.get(self._cfg.profile)
        if factory is None:
            raise ValueError(f"Unsupported noise profile: {self._cfg.profile}")
        return factory()

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_noiseless(self) -> NoiseModelResult:
        """Return an empty (noiseless) NoiseModel for baseline comparisons."""
        nm = NoiseModel()
        logger.info("Noiseless (empty) NoiseModel built.")
        return NoiseModelResult(
            noise_model=nm,
            profile=NoiseProfile.NONE,
            basis_gates=[],
            metadata={"profile": "none"},
        )

    def _build_depolarizing(self) -> NoiseModelResult:
        """
        Build a symmetric Pauli depolarizing noise model.

        Applies:
          - 1Q depolarizing channel to all single-qubit gates.
          - 2Q depolarizing channel to all two-qubit gates.
        """
        params = self._cfg.depolarizing
        nm = NoiseModel()

        err_1q: QuantumError = depolarizing_error(params.single_qubit_error, 1)
        err_2q: QuantumError = depolarizing_error(params.two_qubit_error, 2)

        nm.add_all_qubit_quantum_error(err_1q, SINGLE_QUBIT_GATES)
        nm.add_all_qubit_quantum_error(err_2q, TWO_QUBIT_GATES)

        basis = SINGLE_QUBIT_GATES + TWO_QUBIT_GATES
        logger.info(
            "Depolarizing NoiseModel built",
            extra={
                "single_qubit_error": params.single_qubit_error,
                "two_qubit_error": params.two_qubit_error,
            },
        )
        return NoiseModelResult(
            noise_model=nm,
            profile=NoiseProfile.DEPOLARIZING,
            basis_gates=basis,
            metadata={
                "profile": "depolarizing",
                "single_qubit_error": params.single_qubit_error,
                "two_qubit_error": params.two_qubit_error,
            },
        )

    def _build_thermal(self) -> NoiseModelResult:
        """
        Build a combined thermal relaxation + depolarizing noise model.

        For each gate duration, thermal_relaxation_error is computed from
        T1, T2, and gate time. A readout error is registered on all qubits.

        T2 is clamped to 2*T1 if T2 > 2*T1 (physical constraint).
        """
        tp = self._cfg.thermal
        dp = self._cfg.depolarizing
        n = self._cfg.n_qubits

        # Physical constraint: T2 <= 2 * T1
        t2_effective = min(tp.t2, 2.0 * tp.t1)
        if t2_effective < tp.t2:
            logger.warning(
                "T2 (%.1f ns) > 2*T1 (%.1f ns); clamping T2 to %.1f ns.",
                tp.t2, tp.t1, t2_effective,
            )

        nm = NoiseModel()

        # Thermal relaxation errors
        err_1q_thermal = thermal_relaxation_error(
            tp.t1, t2_effective, tp.gate_time_1q
        )
        err_2q_thermal = thermal_relaxation_error(
            tp.t1, t2_effective, tp.gate_time_2q
        ).expand(
            thermal_relaxation_error(tp.t1, t2_effective, tp.gate_time_2q)
        )

        nm.add_all_qubit_quantum_error(err_1q_thermal, SINGLE_QUBIT_GATES)
        nm.add_all_qubit_quantum_error(err_2q_thermal, TWO_QUBIT_GATES)

        # Depolarizing component on top of thermal
        err_1q_dep = depolarizing_error(dp.single_qubit_error, 1)
        err_2q_dep = depolarizing_error(dp.two_qubit_error, 2)
        nm.add_all_qubit_quantum_error(err_1q_dep, SINGLE_QUBIT_GATES)
        nm.add_all_qubit_quantum_error(err_2q_dep, TWO_QUBIT_GATES)

        # Readout error applied per qubit
        readout = ReadoutError(
            [
                [1 - tp.readout_error_p01, tp.readout_error_p01],
                [tp.readout_error_p10, 1 - tp.readout_error_p10],
            ]
        )
        for qubit in range(n):
            nm.add_readout_error(readout, [qubit])

        basis = SINGLE_QUBIT_GATES + TWO_QUBIT_GATES
        logger.info(
            "Thermal NoiseModel built",
            extra={
                "t1_ns": tp.t1,
                "t2_ns": t2_effective,
                "gate_time_1q_ns": tp.gate_time_1q,
                "gate_time_2q_ns": tp.gate_time_2q,
                "n_qubits": n,
            },
        )
        return NoiseModelResult(
            noise_model=nm,
            profile=NoiseProfile.THERMAL,
            basis_gates=basis,
            metadata={
                "profile": "thermal",
                "t1_ns": tp.t1,
                "t2_ns": t2_effective,
                "gate_time_1q_ns": tp.gate_time_1q,
                "gate_time_2q_ns": tp.gate_time_2q,
                "readout_error_p01": tp.readout_error_p01,
                "readout_error_p10": tp.readout_error_p10,
                "single_qubit_error": dp.single_qubit_error,
                "two_qubit_error": dp.two_qubit_error,
                "n_qubits": n,
            },
        )

    def _build_device_fake(self) -> NoiseModelResult:
        """
        Extract a NoiseModel from a qiskit_ibm_runtime fake backend.

        The noise model captures gate errors, readout errors, and T1/T2
        parameters from the fake device's backend properties.

        Falls back to a depolarizing model if the fake provider is unavailable
        (e.g. in minimal CI environments).
        """
        backend_name = self._cfg.device_fake.backend_name

        fake_map: dict[str, str] = {
            "fake_nairobi": "FakeNairobi",
            "fake_lagos": "FakeLagos",
            "fake_manila": "FakeManila",
            "fake_cairo": "FakeCairo",
            "fake_kolkata": "FakeKolkata",
            "fake_montreal": "FakeMontreal",
        }

        provider_class_name = fake_map.get(backend_name.lower())
        if provider_class_name is None:
            logger.warning(
                "Unknown fake backend '%s'; falling back to depolarizing model.",
                backend_name,
            )
            return self._build_depolarizing()

        try:
            import importlib
            module = importlib.import_module("qiskit_ibm_runtime.fake_provider")
            FakeClass = getattr(module, provider_class_name)
            fake_backend = FakeClass()
            nm = NoiseModel.from_backend(fake_backend)
            basis = nm.basis_gates

            logger.info(
                "Device-fake NoiseModel built from '%s'",
                backend_name,
                extra={"basis_gates": basis},
            )
            return NoiseModelResult(
                noise_model=nm,
                profile=NoiseProfile.DEVICE_FAKE,
                basis_gates=basis,
                metadata={
                    "profile": "device_fake",
                    "backend_name": backend_name,
                    "provider_class": provider_class_name,
                },
            )

        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Could not load fake backend '%s' (%s); "
                "falling back to depolarizing noise model.",
                backend_name,
                exc,
            )
            return self._build_depolarizing()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        cfg = self._cfg
        if cfg.n_qubits < 1:
            raise ValueError(f"n_qubits must be ≥ 1; got {cfg.n_qubits}.")
        if cfg.depolarizing.single_qubit_error < 0 or cfg.depolarizing.single_qubit_error > 1:
            raise ValueError(
                f"single_qubit_error must be in [0, 1]; "
                f"got {cfg.depolarizing.single_qubit_error}."
            )
        if cfg.depolarizing.two_qubit_error < 0 or cfg.depolarizing.two_qubit_error > 1:
            raise ValueError(
                f"two_qubit_error must be in [0, 1]; "
                f"got {cfg.depolarizing.two_qubit_error}."
            )
        if cfg.thermal.t1 <= 0:
            raise ValueError(f"T1 must be > 0 ns; got {cfg.thermal.t1}.")
        if cfg.thermal.t2 <= 0:
            raise ValueError(f"T2 must be > 0 ns; got {cfg.thermal.t2}.")