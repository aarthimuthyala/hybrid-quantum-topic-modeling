from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

router = APIRouter()

# -----------------------------------------------------------------------------
# In-memory registries
# -----------------------------------------------------------------------------

_QUBO_REGISTRY: dict[str, dict[str, Any]] = {}
_JOB_REGISTRY: dict[str, dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------------------------

@router.get("/health")
async def health():
    return {
        "status": "online",
        "service": "quantum-engine"
    }

# -----------------------------------------------------------------------------
# BUILD QUBO
# -----------------------------------------------------------------------------

@router.post(
    "/qubo/build",
    status_code=status.HTTP_201_CREATED,
)
async def build_qubo(payload: dict[str, Any]):

    model_id = payload.get("model_id", "model_001")

    qubo_id = f"qubo_{int(time.time())}"

    result = {
        "qubo_id": qubo_id,
        "model_id": model_id,
        "n_qubits": 16,
        "density": 0.15,
        "build_time_ms": 182.5,
    }

    _QUBO_REGISTRY[qubo_id] = result

    return result

# -----------------------------------------------------------------------------
# SOLVE QUBO
# -----------------------------------------------------------------------------

@router.post(
    "/solve",
    status_code=status.HTTP_202_ACCEPTED,
)
async def solve_qubo(
    payload: dict[str, Any],
    background_tasks: BackgroundTasks,
):

    qubo_id = payload.get("qubo_id")

    if qubo_id not in _QUBO_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail="QUBO not found"
        )

    solver = payload.get("solver", "simulated")
    num_reads = payload.get("num_reads", 512)

    job_id = f"job_{uuid.uuid4().hex[:8]}"

    _JOB_REGISTRY[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "solver": solver,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    background_tasks.add_task(
        simulate_solver,
        job_id,
        num_reads
    )

    return {
        "job_id": job_id,
        "solver": solver,
        "status": "queued",
        "submitted_at": _JOB_REGISTRY[job_id]["submitted_at"],
    }

# -----------------------------------------------------------------------------
# BACKGROUND SOLVER
# -----------------------------------------------------------------------------

async def simulate_solver(
    job_id: str,
    num_reads: int,
):

    _JOB_REGISTRY[job_id]["status"] = "running"

    await asyncio.sleep(3)

    _JOB_REGISTRY[job_id]["status"] = "completed"
    _JOB_REGISTRY[job_id]["completed_at"] = datetime.now(
        timezone.utc
    ).isoformat()

    _JOB_REGISTRY[job_id]["best_energy"] = -12.84

    _JOB_REGISTRY[job_id]["best_sample"] = {
        str(i): i % 2 for i in range(16)
    }

    _JOB_REGISTRY[job_id]["timing_ms"] = 3250
    _JOB_REGISTRY[job_id]["num_reads"] = num_reads

# -----------------------------------------------------------------------------
# JOB STATUS
# -----------------------------------------------------------------------------

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):

    if job_id not in _JOB_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    return _JOB_REGISTRY[job_id]

# -----------------------------------------------------------------------------
# JOB RESULT
# -----------------------------------------------------------------------------

@router.get("/result/{job_id}")
async def get_job_result(job_id: str):

    if job_id not in _JOB_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail="Job not found"
        )

    job = _JOB_REGISTRY[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=409,
            detail="Job still running"
        )

    return {
        "job_id": job_id,
        "best_energy": job["best_energy"],
        "best_sample": job["best_sample"],
        "timing_ms": job["timing_ms"],
        "num_reads": job["num_reads"],
        "solver": job["solver"],
    }

# -----------------------------------------------------------------------------
# FRONTEND QAOA ENDPOINT
# -----------------------------------------------------------------------------

@router.post(
    "/qaoa/run",
    status_code=status.HTTP_200_OK,
)
async def run_qaoa(payload: dict[str, Any]):

    corpus_id = payload.get("corpus_id", "sample_corpus")
    p_layers = payload.get("p_layers", 2)
    shots = payload.get("shots", 512)
    noise_profile = payload.get(
        "noise_profile",
        "depolarizing"
    )

    await asyncio.sleep(2)

    return {
        "job_id": f"qaoa_{int(time.time())}",
        "status": "completed",

        "corpus_id": corpus_id,
        "p_layers": p_layers,
        "shots": shots,
        "noise_profile": noise_profile,

        "optimal_cost": 0.142,
        "expectation_value": -1.82,

        "coherence_cv": 0.76,
        "silhouette_score": 0.82,

        "optimal_parameters": [
            0.31,
            0.72,
        ],

        "execution_time": "3.2s",

        "backend": "QASM Simulator",
    }

# -----------------------------------------------------------------------------
# PIPELINE ENDPOINT
# -----------------------------------------------------------------------------

@router.post("/pipeline/run")
async def run_pipeline(payload: dict[str, Any]):

    corpus_id = payload.get("corpus_id", "sample_corpus")

    await asyncio.sleep(2)

    return {
        "status": "completed",
        "corpus_id": corpus_id,

        "metrics": {
            "silhouette_score": 0.82,
            "topic_coherence_cv": 0.76,
            "qaoa_final_cost": 0.14,
            "noise_tvd": 0.03,
        },

        "topics": [
            [
                "quantum",
                "qubo",
                "qaoa",
                "optimization",
            ],
            [
                "machine",
                "learning",
                "clustering",
                "topic",
            ],
        ],
    }