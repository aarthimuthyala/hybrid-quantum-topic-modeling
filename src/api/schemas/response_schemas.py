"""
src/api/schemas/response_schemas.py
======================================
Team T-5 — API & Backend Layer (L4)
HQC Topic Modeling Project · Master Blueprint v1.0

Responsibility:
    Single source of truth for ALL Pydantic v2 response body schemas
    used across the entire API surface (Blueprint §10.3).

    Route modules import from here; they do NOT define their own
    response models. This ensures:
        1. Every field name, type, and example is declared once.
        2. OpenAPI schema names are stable across all route files.
        3. Downstream consumers (auto-generated SDKs, frontend TypeScript
           clients, integration tests) all reference one canonical source.

Blueprint coverage:
    §4 preamble  : ErrorResponse
    §4.1 Ingest  : UploadResponse, CorpusMetaResponse, PreprocessResponse
    §4.2 Classical: TopicModelTrainResponse, NMFTrainResponse,
                    ClusterResponse, ModelMetaResponse
    §4.3 Quantum  : QUBOBuildResponse, QuantumSolveResponse,
                    JobStatusResponse, QuantumResultResponse
    §4.4 Eval     : CoherenceResponse, ClusterMetricsResponse,
                    BenchmarkResponse, EvalReportResponse

Design rules (Blueprint §6, §10.3):
    - PascalCase class names matching the resource + "Response" suffix.
    - All fields have explicit Field() with description and examples.
    - Optional fields typed as T | None with default=None.
    - No business logic — only serialisation structure.
    - model_config = ConfigDict(populate_by_name=True) on every class
      for forward-compatibility with SDK generation.

Dependencies:
    pydantic>=2.7
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ===========================================================================
# §4 preamble — Universal error envelope (Blueprint §4 preamble)
# ===========================================================================

class ErrorResponse(BaseModel):
    """
    Canonical error envelope returned by all exception handlers.

    FROZEN — field names are contractually guaranteed to clients.
    Changing them requires a versioned API increment (Blueprint §10.3).

    Fields:
        error  : Machine-readable error code slug, e.g. 'model_not_found'.
        detail : Human-readable explanation suitable for display.
        code   : HTTP status code mirrored in the body for client convenience.
    """

    model_config = ConfigDict(populate_by_name=True)

    error: str = Field(
        ...,
        description="Machine-readable error code slug.",
        examples=["model_not_found"],
    )
    detail: str = Field(
        ...,
        description="Human-readable error explanation.",
        examples=["model_id 'lda_20ng_xxx' not found."],
    )
    code: int = Field(
        ...,
        ge=400,
        le=599,
        description="HTTP status code mirrored in the response body.",
        examples=[404],
    )


# ===========================================================================
# §4.1 — Ingestion layer response schemas
# ===========================================================================

class UploadResponse(BaseModel):
    """
    Response for POST /api/v1/ingest/upload (Blueprint §4.1).

    Returned with HTTP 201 Created on successful corpus ingestion.

    Fields:
        corpus_id : Unique identifier assigned to the uploaded corpus.
                    Use this in all subsequent /ingest and /classical calls.
        doc_count : Number of documents detected in the uploaded file.
        status    : Lifecycle status — always 'uploaded' from this endpoint.
    """

    model_config = ConfigDict(populate_by_name=True)

    corpus_id: str = Field(
        ...,
        description="Unique corpus identifier for subsequent API calls.",
        examples=["20ng_a3f2c1"],
    )
    doc_count: int = Field(
        ...,
        ge=0,
        description="Number of documents detected in the uploaded file.",
        examples=[11314],
    )
    status: Literal["uploaded"] = Field(
        default="uploaded",
        description="Corpus lifecycle status.",
    )


class CorpusMetaResponse(BaseModel):
    """
    Response for GET /api/v1/ingest/{corpus_id} (Blueprint §4.1).

    Fields:
        corpus_id  : Unique corpus identifier.
        doc_count  : Total number of documents in the raw corpus.
        vocab_size : Vocabulary size after preprocessing (0 if preprocessing
                     has not yet been run via /ingest/preprocess).
        created_at : ISO-8601 UTC timestamp of initial upload.
    """

    model_config = ConfigDict(populate_by_name=True)

    corpus_id: str = Field(
        ...,
        description="Unique corpus identifier.",
        examples=["20ng_a3f2c1"],
    )
    doc_count: int = Field(
        ...,
        ge=0,
        description="Raw document count.",
        examples=[11314],
    )
    vocab_size: int = Field(
        ...,
        ge=0,
        description="Vocabulary size (0 before preprocessing).",
        examples=[42000],
    )
    created_at: str = Field(
        ...,
        description="ISO-8601 UTC upload timestamp.",
        examples=["2025-05-10T12:00:00+00:00"],
    )


class PreprocessResponse(BaseModel):
    """
    Response for POST /api/v1/ingest/preprocess (Blueprint §4.1).

    Fields:
        corpus_id      : Corpus that was preprocessed.
        clean_doc_count: Documents remaining after preprocessing
                         (may be lower than doc_count if empty docs
                          were filtered out by the pipeline).
        status         : Always 'preprocessed' from this endpoint.
    """

    model_config = ConfigDict(populate_by_name=True)

    corpus_id: str = Field(
        ...,
        description="Preprocessed corpus identifier.",
        examples=["20ng_a3f2c1"],
    )
    clean_doc_count: int = Field(
        ...,
        ge=0,
        description="Documents remaining after preprocessing.",
        examples=[11200],
    )
    status: Literal["preprocessed"] = Field(
        default="preprocessed",
        description="Corpus lifecycle status.",
    )


# ===========================================================================
# §4.2 — Classical NLP layer response schemas
# ===========================================================================

class TopicModelTrainResponse(BaseModel):
    """
    Response for POST /api/v1/classical/lda/train (Blueprint §4.2).

    Fields:
        model_id   : Unique model identifier — use in /eval/coherence and
                     /quantum/qubo/build.
        coherence  : C_V topic coherence (None if T-4 not yet integrated).
        perplexity : LDA per-word held-out perplexity (None for NMF).
        run_id     : MLflow run identifier for experiment tracking.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(
        ...,
        description="Unique model identifier.",
        examples=["lda_20ng_1715350000"],
    )
    coherence: float | None = Field(
        default=None,
        description="C_V topic coherence score; None before T-4 integration.",
        examples=[0.512],
    )
    perplexity: float | None = Field(
        default=None,
        description="LDA per-word perplexity; None for NMF.",
        examples=[1842.3],
    )
    run_id: str = Field(
        ...,
        description="MLflow run identifier for experiment tracking.",
        examples=["lda_20ng_1715350000"],
    )


class NMFTrainResponse(BaseModel):
    """
    Response for POST /api/v1/classical/nmf/train (Blueprint §4.2).

    NMF does not produce a perplexity metric, so the field is omitted
    (unlike TopicModelTrainResponse). This keeps the response minimal
    and avoids confusing clients with a null perplexity field.

    Fields:
        model_id  : Unique model identifier.
        coherence : C_V topic coherence (None before T-4 integration).
        run_id    : MLflow run identifier.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(
        ...,
        description="Unique NMF model identifier.",
        examples=["nmf_20ng_1715350001"],
    )
    coherence: float | None = Field(
        default=None,
        description="C_V topic coherence score; None before T-4 integration.",
        examples=[0.498],
    )
    run_id: str = Field(
        ...,
        description="MLflow run identifier.",
        examples=["nmf_20ng_1715350001"],
    )


class ClusterResponse(BaseModel):
    """
    Response for POST /api/v1/classical/cluster (Blueprint §4.2).

    Fields:
        cluster_id : Unique identifier for this clustering run.
                     Use in /eval/cluster/metrics and /quantum/qubo/build.
        silhouette : Silhouette coefficient [-1, 1]. None if n_docs > 5 000
                     and compute_silhouette=False in config, or if only
                     one cluster was populated.
        labels     : Per-document cluster assignment. labels[i] is the
                     zero-based cluster index for document i.
    """

    model_config = ConfigDict(populate_by_name=True)

    cluster_id: str = Field(
        ...,
        description="Unique clustering run identifier.",
        examples=["kmeans_20ng_1715350200"],
    )
    silhouette: float | None = Field(
        default=None,
        description="Silhouette coefficient [-1, 1]; None if not computed.",
        examples=[0.312],
    )
    labels: list[int] = Field(
        ...,
        description="Per-document cluster index (len = n_docs).",
        examples=[[0, 3, 7, 2, 19]],
    )


class ModelMetaResponse(BaseModel):
    """
    Response for GET /api/v1/classical/model/{model_id} (Blueprint §4.2).

    Unified schema covering LDA, NMF, and K-Means model records.
    The 'type' field disambiguates which model class produced the record.

    Fields:
        model_id : Unique model identifier.
        type     : Model class — 'lda', 'nmf', or 'kmeans'.
        params   : Full hyperparameter snapshot (mirrors classical_config.yaml).
        metrics  : Primary evaluation metrics captured at training time.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(
        ...,
        description="Unique model identifier.",
        examples=["lda_20ng_1715350000"],
    )
    type: Literal["lda", "nmf", "kmeans"] = Field(
        ...,
        description="Model class identifier.",
        examples=["lda"],
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Full hyperparameter snapshot from training.",
        examples=[{"n_topics": 20, "max_iter": 100, "model": "lda"}],
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Primary metrics captured at training time.",
        examples=[{"coherence": 0.512, "perplexity": 1842.3}],
    )


# ===========================================================================
# §4.3 — Quantum engine layer response schemas
# ===========================================================================

class QUBOBuildResponse(BaseModel):
    """
    Response for POST /api/v1/quantum/qubo/build (Blueprint §4.3).

    Fields:
        qubo_id      : Unique identifier for the built QUBO matrix.
                       Use in POST /quantum/solve.
        n_qubits     : Number of binary variables (QUBO dimension).
        density      : Fraction of non-zero off-diagonal QUBO entries.
        build_time_ms: QUBO construction elapsed time in milliseconds.
    """

    model_config = ConfigDict(populate_by_name=True)

    qubo_id: str = Field(
        ...,
        description="Unique QUBO identifier for solver submission.",
        examples=["qubo_lda_20ng_1715350000_1715350100"],
    )
    n_qubits: int = Field(
        ...,
        ge=1,
        description="Number of binary variables (QUBO matrix dimension).",
        examples=[190],
    )
    density: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of non-zero off-diagonal QUBO entries [0, 1].",
        examples=[0.152],
    )
    build_time_ms: float = Field(
        ...,
        ge=0.0,
        description="QUBO construction elapsed time in milliseconds.",
        examples=[843.2],
    )


class QuantumSolveResponse(BaseModel):
    """
    Response for POST /api/v1/quantum/solve (Blueprint §4.3).

    HTTP 202 Accepted — the solve is asynchronous. Poll
    GET /quantum/job/{job_id} for status; retrieve the result from
    GET /quantum/result/{job_id} when status='completed'.

    Fields:
        job_id       : Unique solver job identifier.
        solver       : Solver backend selected for this job.
        status       : Always 'queued' when first submitted.
        submitted_at : ISO-8601 UTC timestamp of submission.
    """

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(
        ...,
        description="Unique solver job identifier.",
        examples=["job_simulated_1715350150_a3c2f1"],
    )
    solver: Literal["dwave", "simulated", "hybrid"] = Field(
        ...,
        description="Solver backend for this job.",
        examples=["simulated"],
    )
    status: Literal["queued"] = Field(
        default="queued",
        description="Initial job status — always 'queued' at submission.",
    )
    submitted_at: str = Field(
        ...,
        description="ISO-8601 UTC submission timestamp.",
        examples=["2025-05-10T12:05:00+00:00"],
    )


class JobStatusResponse(BaseModel):
    """
    Response for GET /api/v1/quantum/job/{job_id} (Blueprint §4.3).

    Fields:
        job_id       : Unique solver job identifier.
        status       : Current lifecycle status.
        solver       : Solver backend for this job.
        submitted_at : ISO-8601 UTC submission timestamp.
        completed_at : ISO-8601 UTC completion timestamp (None if pending).
    """

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(
        ...,
        description="Unique solver job identifier.",
        examples=["job_simulated_1715350150_a3c2f1"],
    )
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = Field(
        ...,
        description="Current job lifecycle status.",
        examples=["completed"],
    )
    solver: Literal["dwave", "simulated", "hybrid"] = Field(
        ...,
        description="Solver backend for this job.",
        examples=["simulated"],
    )
    submitted_at: str = Field(
        ...,
        description="ISO-8601 UTC submission timestamp.",
        examples=["2025-05-10T12:05:00+00:00"],
    )
    completed_at: str | None = Field(
        default=None,
        description="ISO-8601 UTC completion timestamp; None while pending.",
        examples=["2025-05-10T12:05:04+00:00"],
    )


class QuantumResultResponse(BaseModel):
    """
    Response for GET /api/v1/quantum/result/{job_id} (Blueprint §4.3).

    Only available once GET /quantum/job/{job_id} returns status='completed'.

    Fields:
        job_id      : Solver job identifier.
        best_energy : Lowest Hamiltonian energy found across all reads.
        best_sample : Qubit spin assignments for the lowest-energy sample.
                      Keys are qubit index strings; values are 0 or 1.
        timing_ms   : Total solver wall-clock time in milliseconds.
        num_reads   : Number of annealing reads that were performed.
        solver      : Solver backend used.
    """

    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(
        ...,
        description="Solver job identifier.",
        examples=["job_simulated_1715350150_a3c2f1"],
    )
    best_energy: float = Field(
        ...,
        description="Lowest Hamiltonian energy found across all annealing reads.",
        examples=[-47.25],
    )
    best_sample: dict[str, int] = Field(
        ...,
        description=(
            "Spin assignments for the lowest-energy sample. "
            "Keys are qubit index strings; values are 0 or 1."
        ),
        examples=[{"0": 1, "1": 0, "2": 1}],
    )
    timing_ms: float = Field(
        ...,
        ge=0.0,
        description="Total solver wall-clock time in milliseconds.",
        examples=[3842.7],
    )
    num_reads: int = Field(
        ...,
        ge=0,
        description="Number of annealing reads performed.",
        examples=[1000],
    )
    solver: Literal["dwave", "simulated", "hybrid"] = Field(
        ...,
        description="Solver backend used.",
        examples=["simulated"],
    )


# ===========================================================================
# §4.4 — Evaluation layer response schemas
# ===========================================================================

class CoherenceResponse(BaseModel):
    """
    Response for POST /api/v1/eval/coherence (Blueprint §4.4).

    Fields:
        model_id : Model that was evaluated.
        metric   : Coherence metric used.
        score    : Computed coherence score.
        run_id   : Evaluation run identifier for report retrieval.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_id: str = Field(
        ...,
        description="Evaluated model identifier.",
        examples=["lda_20ng_1715350000"],
    )
    metric: Literal["c_v", "u_mass", "c_uci"] = Field(
        ...,
        description="Coherence metric used.",
        examples=["c_v"],
    )
    score: float = Field(
        ...,
        description="Computed coherence score (higher is better for c_v / c_uci).",
        examples=[0.512],
    )
    run_id: str = Field(
        ...,
        description="Evaluation run identifier — use in GET /eval/report/{run_id}.",
        examples=["coherence_lda_20ng_1715350000_1715350300"],
    )


class ClusterMetricsResponse(BaseModel):
    """
    Response for POST /api/v1/eval/cluster/metrics (Blueprint §4.4).

    Fields:
        cluster_id : K-Means clustering run that was evaluated.
        scores     : Mapping of metric name → computed score.
        run_id     : Evaluation run identifier.
    """

    model_config = ConfigDict(populate_by_name=True)

    cluster_id: str = Field(
        ...,
        description="Evaluated clustering run identifier.",
        examples=["kmeans_20ng_1715350200"],
    )
    scores: dict[str, float] = Field(
        ...,
        description="Metric name → computed score mapping.",
        examples=[{"silhouette": 0.312, "davies_bouldin": 1.45}],
    )
    run_id: str = Field(
        ...,
        description="Evaluation run identifier — use in GET /eval/report/{run_id}.",
        examples=["cluster_metrics_kmeans_20ng_1715350200_1715350400"],
    )


class BenchmarkResponse(BaseModel):
    """
    Response for POST /api/v1/eval/benchmark (Blueprint §4.4).

    Fields:
        run_id    : Unique benchmark run identifier.
        corpus_id : Corpus both models were evaluated on.
        classical : Classical model metric scores (metric → float).
        hybrid    : Hybrid model metric scores (metric → float).
        delta     : Per-metric score difference (hybrid − classical).
                    Positive means hybrid outperforms on higher-is-better
                    metrics; negative means classical wins.
        winner    : 'classical', 'hybrid', or 'tie'.
    """

    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(
        ...,
        description="Unique benchmark run identifier.",
        examples=["benchmark_20ng_1715350500"],
    )
    corpus_id: str = Field(
        ...,
        description="Corpus used for evaluation.",
        examples=["20ng_a3f2c1"],
    )
    classical: dict[str, float] = Field(
        ...,
        description="Classical model scores per metric.",
        examples=[{"c_v": 0.503, "silhouette": 0.291}],
    )
    hybrid: dict[str, float] = Field(
        ...,
        description="Hybrid quantum-classical model scores per metric.",
        examples=[{"c_v": 0.541, "silhouette": 0.318}],
    )
    delta: dict[str, float] = Field(
        ...,
        description=(
            "Per-metric delta (hybrid − classical). "
            "Positive favours hybrid on higher-is-better metrics."
        ),
        examples=[{"c_v": 0.038, "silhouette": 0.027}],
    )
    winner: Literal["classical", "hybrid", "tie"] = Field(
        ...,
        description=(
            "Overall winner determined by majority vote across all metrics "
            "(accounting for metric polarity)."
        ),
        examples=["hybrid"],
    )


class EvalReportResponse(BaseModel):
    """
    Response for GET /api/v1/eval/report/{run_id} (Blueprint §4.4).

    Generic container for any evaluation run — coherence, cluster metrics,
    or benchmark. The 'type' field tells the client how to interpret 'results'.

    Fields:
        run_id     : Unique evaluation run identifier.
        type       : Report type — 'coherence', 'cluster_metrics', or 'benchmark'.
        created_at : ISO-8601 UTC timestamp when the evaluation completed.
        results    : Full results dict (schema depends on report type).
    """

    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(
        ...,
        description="Unique evaluation run identifier.",
        examples=["coherence_lda_20ng_1715350000_1715350300"],
    )
    type: Literal["coherence", "cluster_metrics", "benchmark"] = Field(
        ...,
        description="Report type — determines the structure of 'results'.",
        examples=["coherence"],
    )
    created_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp of evaluation completion.",
        examples=["2025-05-10T12:05:30+00:00"],
    )
    results: dict[str, Any] = Field(
        ...,
        description=(
            "Full evaluation results dict. Structure depends on 'type': "
            "coherence → {model_id, metric, score, elapsed_ms}; "
            "cluster_metrics → {cluster_id, metrics, scores, elapsed_ms}; "
            "benchmark → {classical, hybrid, delta, winner, elapsed_ms}."
        ),
    )