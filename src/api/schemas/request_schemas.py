"""
src/api/schemas/request_schemas.py
=====================================
Team T-5 — API & Backend Layer (L4)
HQC Topic Modeling Project · Master Blueprint v1.0

Responsibility:
    Single source of truth for ALL Pydantic v2 request body schemas
    used across the entire API surface (Blueprint §10.3 — no schema
    duplication across route files).

    Route modules import from here; they do NOT define their own
    request models. This file is the canonical record of every
    field constraint, validator, and default value for incoming data.

Blueprint coverage:
    §4.1  Ingestion    : PreprocessRequest
    §4.2  Classical    : LDATrainRequest, NMFTrainRequest, ClusterRequest
    §4.3  Quantum      : QUBOBuildRequest, QuantumSolveRequest
    §4.4  Evaluation   : CoherenceRequest, ClusterMetricsRequest,
                          BenchmarkRequest

Design rules (Blueprint §6, §10.3):
    - PascalCase class names matching the action + "Request" suffix.
    - All fields have explicit Field() with description and examples.
    - field_validator decorators raise ValueError with human-readable
      messages (FastAPI converts these to 422 responses automatically).
    - No business logic here — only structural validation.

Dependencies:
    pydantic>=2.7
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Shared domain constants (mirrored from route files for validator use)
# ---------------------------------------------------------------------------
_SUPPORTED_PIPELINE_STEPS: frozenset[str] = frozenset(
    {"lowercase", "stopwords", "lemmatize", "html_strip", "punct_remove"}
)
_VALID_COHERENCE_METRICS: frozenset[str] = frozenset({"c_v", "u_mass", "c_uci"})
_VALID_CLUSTER_METRICS: frozenset[str] = frozenset(
    {"silhouette", "davies_bouldin", "nmi", "ari"}
)
_VALID_SOLVERS: frozenset[str] = frozenset({"dwave", "simulated", "hybrid"})
_VALID_BENCHMARK_METRICS: frozenset[str] = (
    _VALID_COHERENCE_METRICS
    | _VALID_CLUSTER_METRICS
    | frozenset({"perplexity", "reconstruction_err"})
)
_MIN_TOPICS: int = 2
_MAX_TOPICS: int = 200
_MIN_K: int = 2
_MAX_K: int = 500
_MIN_NUM_READS: int = 1
_MAX_NUM_READS: int = 10_000
_MIN_ANNEALING_US: int = 1
_MAX_ANNEALING_US: int = 2_000          # D-Wave Leap hard ceiling (μs)
_ALPHA_RANGE: tuple[float, float] = (0.0, 10.0)
_BETA_RANGE:  tuple[float, float] = (0.0, 10.0)
_GAMMA_RANGE: tuple[float, float] = (0.0, 10.0)


# ===========================================================================
# §4.1 — Ingestion layer request schemas
# ===========================================================================

class PreprocessRequest(BaseModel):
    """
    POST /api/v1/ingest/preprocess (Blueprint §4.1).

    Triggers the T-1 text preprocessing pipeline on a previously
    uploaded and registered corpus. The pipeline field specifies an
    ordered sequence of transformation steps; order matters
    (e.g. html_strip should precede lowercase).

    Fields:
        corpus_id : Must reference a corpus registered via /ingest/upload.
        pipeline  : Ordered list of preprocessing step names; each step
                    must be one of the supported values.
    """

    corpus_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="ID of a previously uploaded corpus.",
        examples=["20ng_a3f2c1"],
    )
    pipeline: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered preprocessing steps. "
            f"Allowed: {sorted(_SUPPORTED_PIPELINE_STEPS)}"
        ),
        examples=[["lowercase", "html_strip", "stopwords", "lemmatize"]],
    )

    @field_validator("pipeline")
    @classmethod
    def validate_pipeline_steps(cls, steps: list[str]) -> list[str]:
        """Reject unknown pipeline step names with a clear error message."""
        unknown = set(steps) - _SUPPORTED_PIPELINE_STEPS
        if unknown:
            raise ValueError(
                f"Unknown pipeline steps: {sorted(unknown)}. "
                f"Supported steps: {sorted(_SUPPORTED_PIPELINE_STEPS)}"
            )
        if len(steps) != len(set(steps)):
            raise ValueError("Duplicate pipeline steps are not allowed.")
        return steps


# ===========================================================================
# §4.2 — Classical NLP layer request schemas
# ===========================================================================

class LDATrainRequest(BaseModel):
    """
    POST /api/v1/classical/lda/train (Blueprint §4.2).

    Triggers LDA training on the TF-IDF document-term matrix of the
    specified corpus. The model is saved to outputs/models/ and
    registered for retrieval via GET /classical/model/{model_id}.

    Fields:
        corpus_id : Must reference a preprocessed corpus.
        n_topics  : Number of latent topics to discover [2, 200].
        config_id : Config profile key in classical_config.yaml.
    """

    corpus_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Preprocessed corpus identifier.",
        examples=["20ng_a3f2c1"],
    )
    n_topics: int = Field(
        ...,
        ge=_MIN_TOPICS,
        le=_MAX_TOPICS,
        description=f"Number of LDA topics in [{_MIN_TOPICS}, {_MAX_TOPICS}].",
        examples=[20],
    )
    config_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        description="Config profile key in classical_config.yaml.",
        examples=["default"],
    )


class NMFTrainRequest(BaseModel):
    """
    POST /api/v1/classical/nmf/train (Blueprint §4.2).

    Identical field structure to LDATrainRequest — kept as a distinct
    class so OpenAPI generates separate schema names, improving
    auto-generated client code and documentation clarity.

    Fields:
        corpus_id : Must reference a preprocessed corpus.
        n_topics  : Number of NMF components (rank k) [2, 200].
        config_id : Config profile key in classical_config.yaml.
    """

    corpus_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Preprocessed corpus identifier.",
        examples=["20ng_a3f2c1"],
    )
    n_topics: int = Field(
        ...,
        ge=_MIN_TOPICS,
        le=_MAX_TOPICS,
        description=f"Number of NMF components (rank k) in [{_MIN_TOPICS}, {_MAX_TOPICS}].",
        examples=[20],
    )
    config_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        description="Config profile key in classical_config.yaml.",
        examples=["default"],
    )


class ClusterRequest(BaseModel):
    """
    POST /api/v1/classical/cluster (Blueprint §4.2).

    Runs K-Means clustering on the TF-IDF document-term matrix
    (method='tfidf') or a dense embedding matrix produced by the
    quantum engine (method='embed').

    Fields:
        corpus_id : Must reference a preprocessed (and vectorised) corpus.
        k         : Number of K-Means clusters [2, 500].
        method    : Input matrix type — 'tfidf' or 'embed'.
    """

    corpus_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Preprocessed corpus identifier.",
        examples=["20ng_a3f2c1"],
    )
    k: int = Field(
        ...,
        ge=_MIN_K,
        le=_MAX_K,
        description=f"Number of K-Means clusters in [{_MIN_K}, {_MAX_K}].",
        examples=[20],
    )
    method: Literal["tfidf", "embed"] = Field(
        default="tfidf",
        description=(
            "Input matrix type. 'tfidf' clusters the TF-IDF DTMatrix; "
            "'embed' clusters a dense quantum embedding (requires T-3)."
        ),
    )


# ===========================================================================
# §4.3 — Quantum engine layer request schemas
# ===========================================================================

class QUBOBuildRequest(BaseModel):
    """
    POST /api/v1/quantum/qubo/build (Blueprint §4.3).

    Constructs a QUBO Hamiltonian from the topic-word distribution of a
    trained classical model. The three Lagrangian multipliers (alpha,
    beta, gamma) control the trade-off between topic diversity, vocabulary
    coverage, and sparsity respectively (Blueprint §3.3).

    Fields:
        model_id : model_id returned by /classical/lda/train or /nmf/train.
        alpha    : Lagrangian weight for topic-diversity penalty [0, 10].
        beta     : Lagrangian weight for vocabulary-coverage reward [0, 10].
        gamma    : Lagrangian weight for sparsity regularisation [0, 10].
    """

    model_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="model_id of a trained LDA or NMF model.",
        examples=["lda_20ng_1715350000"],
    )
    alpha: float = Field(
        default=1.0,
        ge=_ALPHA_RANGE[0],
        le=_ALPHA_RANGE[1],
        description="Topic-diversity Lagrangian multiplier α ∈ [0, 10].",
        examples=[1.0],
    )
    beta: float = Field(
        default=1.0,
        ge=_BETA_RANGE[0],
        le=_BETA_RANGE[1],
        description="Vocabulary-coverage Lagrangian multiplier β ∈ [0, 10].",
        examples=[1.0],
    )
    gamma: float = Field(
        default=0.5,
        ge=_GAMMA_RANGE[0],
        le=_GAMMA_RANGE[1],
        description="Sparsity regularisation multiplier γ ∈ [0, 10].",
        examples=[0.5],
    )

    @model_validator(mode="after")
    def validate_nonzero_weights(self) -> "QUBOBuildRequest":
        """
        Warn when all three Lagrangian multipliers are zero — this
        produces a trivial all-zero QUBO and is almost certainly a
        client configuration error.
        """
        if self.alpha == 0.0 and self.beta == 0.0 and self.gamma == 0.0:
            raise ValueError(
                "At least one of alpha, beta, gamma must be non-zero. "
                "An all-zero QUBO Hamiltonian is trivially solved and meaningless."
            )
        return self


class QuantumSolveRequest(BaseModel):
    """
    POST /api/v1/quantum/solve (Blueprint §4.3).

    Submits a QUBO to the selected solver backend. The call is
    asynchronous: 202 is returned immediately with a job_id. Poll
    GET /quantum/job/{job_id} for completion status.

    Fields:
        qubo_id          : qubo_id returned by /quantum/qubo/build.
        solver           : Backend — 'dwave', 'simulated', or 'hybrid'.
        num_reads        : Annealing samples to draw [1, 10 000].
        annealing_time_us: Annealing time in μs (D-Wave only) [1, 2 000].
    """

    qubo_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="qubo_id from POST /quantum/qubo/build.",
        examples=["qubo_lda_20ng_1715350000_1715350100"],
    )
    solver: Literal["dwave", "simulated", "hybrid"] = Field(
        default="simulated",
        description=(
            "Solver backend. 'dwave' requires D-Wave Leap credentials; "
            "'simulated' uses CPU simulated annealing; "
            "'hybrid' uses D-Wave's Hybrid Solver Service."
        ),
    )
    num_reads: int = Field(
        default=1000,
        ge=_MIN_NUM_READS,
        le=_MAX_NUM_READS,
        description=f"Annealing reads (samples) [{_MIN_NUM_READS}, {_MAX_NUM_READS}].",
        examples=[1000],
    )
    annealing_time_us: int = Field(
        default=20,
        ge=_MIN_ANNEALING_US,
        le=_MAX_ANNEALING_US,
        description=(
            f"Annealing time in microseconds (D-Wave only) "
            f"[{_MIN_ANNEALING_US}, {_MAX_ANNEALING_US}]. "
            "Ignored for 'simulated' and 'hybrid' solvers."
        ),
        examples=[20],
    )

    @field_validator("num_reads")
    @classmethod
    def validate_num_reads(cls, v: int) -> int:
        """Ensure num_reads is a strictly positive integer."""
        if v < 1:
            raise ValueError("num_reads must be at least 1.")
        return v


# ===========================================================================
# §4.4 — Evaluation layer request schemas
# ===========================================================================

class CoherenceRequest(BaseModel):
    """
    POST /api/v1/eval/coherence (Blueprint §4.4).

    Evaluates the semantic quality of topics using a coherence metric.
    C_V (sliding-window normalised PMI) is the recommended default and
    the best predictor of human topic judgements.

    Fields:
        model_id : model_id of a trained LDA or NMF model.
        metric   : Coherence metric — 'c_v', 'u_mass', or 'c_uci'.
    """

    model_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="model_id of a trained classical topic model.",
        examples=["lda_20ng_1715350000"],
    )
    metric: Literal["c_v", "u_mass", "c_uci"] = Field(
        default="c_v",
        description=(
            "Coherence metric. 'c_v' (recommended, correlates best with human "
            "judgements); 'u_mass' (fast, intrinsic); 'c_uci' (PMI-based)."
        ),
    )


class ClusterMetricsRequest(BaseModel):
    """
    POST /api/v1/eval/cluster/metrics (Blueprint §4.4).

    Computes one or more clustering quality metrics for a K-Means result.

    Internal metrics (no ground truth needed):
        silhouette      — mean Silhouette coefficient, higher is better
        davies_bouldin  — Davies-Bouldin index, lower is better

    External / supervised metrics (require ground_truth_labels):
        nmi — Normalised Mutual Information
        ari — Adjusted Rand Index

    Fields:
        cluster_id          : cluster_id from POST /classical/cluster.
        metrics             : List of metric names to compute.
        ground_truth_labels : Per-document true class labels (required for
                              nmi and ari). Length must match cluster doc count.
    """

    cluster_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="cluster_id from POST /classical/cluster.",
        examples=["kmeans_20ng_1715350200"],
    )
    metrics: list[Literal["silhouette", "davies_bouldin", "nmi", "ari"]] = Field(
        ...,
        min_length=1,
        description=(
            "Metrics to compute. Supervised metrics (nmi, ari) require "
            "ground_truth_labels."
        ),
        examples=[["silhouette", "davies_bouldin"]],
    )
    ground_truth_labels: list[int] | None = Field(
        default=None,
        description=(
            "Per-document ground-truth class labels. Required when 'nmi' "
            "or 'ari' are requested. Must have the same length as the "
            "number of documents in the clustering run."
        ),
        examples=[None],
    )

    @field_validator("metrics")
    @classmethod
    def validate_metric_names(cls, metrics: list[str]) -> list[str]:
        """Reject unknown metric names early."""
        unknown = set(metrics) - _VALID_CLUSTER_METRICS
        if unknown:
            raise ValueError(
                f"Unknown cluster metrics: {sorted(unknown)}. "
                f"Supported: {sorted(_VALID_CLUSTER_METRICS)}"
            )
        if len(metrics) != len(set(metrics)):
            raise ValueError("Duplicate metric names are not allowed.")
        return metrics

    @model_validator(mode="after")
    def check_supervised_labels(self) -> "ClusterMetricsRequest":
        """
        Enforce that supervised metrics (nmi, ari) always have ground
        truth labels. Checked here so the error fires at schema validation
        time (422) rather than inside the route handler.
        """
        supervised_requested = {"nmi", "ari"} & set(self.metrics)
        if supervised_requested and self.ground_truth_labels is None:
            raise ValueError(
                f"Metrics {sorted(supervised_requested)} require "
                "'ground_truth_labels' in the request body."
            )
        return self


class BenchmarkRequest(BaseModel):
    """
    POST /api/v1/eval/benchmark (Blueprint §4.4).

    Runs a structured head-to-head evaluation comparing a classical
    topic model against a hybrid quantum-classical model on the same
    corpus. Any combination of coherence, clustering, and model-fit
    metrics may be requested.

    Fields:
        corpus_id          : Corpus both models were trained on.
        classical_model_id : model_id of the LDA or NMF classical model.
        hybrid_model_id    : model_id of the hybrid quantum-classical model.
        metrics            : Evaluation metrics to include in the comparison.
    """

    corpus_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Corpus both models were trained on.",
        examples=["20ng_a3f2c1"],
    )
    classical_model_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="model_id of the classical (LDA or NMF) baseline model.",
        examples=["lda_20ng_1715350000"],
    )
    hybrid_model_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="model_id of the hybrid quantum-classical comparison model.",
        examples=["hybrid_20ng_1715351000"],
    )
    metrics: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Evaluation metrics to include in the comparison. "
            f"Valid: {sorted(_VALID_BENCHMARK_METRICS)}"
        ),
        examples=[["c_v", "silhouette"]],
    )

    @field_validator("metrics")
    @classmethod
    def validate_benchmark_metrics(cls, metrics: list[str]) -> list[str]:
        """Reject any metric not in the union of coherence + cluster sets."""
        unknown = set(metrics) - _VALID_BENCHMARK_METRICS
        if unknown:
            raise ValueError(
                f"Unknown benchmark metrics: {sorted(unknown)}. "
                f"Valid: {sorted(_VALID_BENCHMARK_METRICS)}"
            )
        if len(metrics) != len(set(metrics)):
            raise ValueError("Duplicate metric names are not allowed.")
        return metrics

    @model_validator(mode="after")
    def validate_distinct_models(self) -> "BenchmarkRequest":
        """Prevent comparing a model against itself."""
        if self.classical_model_id == self.hybrid_model_id:
            raise ValueError(
                "classical_model_id and hybrid_model_id must be different models. "
                "Comparing a model against itself produces meaningless results."
            )
        return self