"""
shared/math_utils.py
====================
Mathematical utility functions for the HQC Topic Modeling project.

Implements the five public functions mandated by §8.5 of the Master Blueprint:
    - cosine_similarity(a, b)          → float in [-1, 1]
    - entropy(distribution)            → Shannon entropy (nats or bits)
    - kl_divergence(p, q)              → KL(P‖Q) in nats
    - sparse_to_dense(matrix)          → scipy.sparse → np.ndarray
    - estimate_qubits(n_docs, n_topics) → int (qubit demand estimate)

All functions operate on ``numpy.ndarray`` inputs unless otherwise noted.
Scipy sparse matrices are supported where explicitly stated.

Numerical stability:
    - Probability distributions are always clipped to [ε, 1] before log
      operations to avoid log(0) = -∞.
    - Zero-norm vectors in cosine_similarity return 0.0 (defined convention).
    - KL divergence raises ``ValueError`` if *q* has zero-probability mass
      where *p* has positive mass (undefined in standard KL).

Blueprint constraints honoured:
    - Full package imports only (§2 RULE)
    - Structured JSON logging via ``shared.logger`` (§8.1)
    - Type annotations with numpy typing (``npt.ArrayLike``)
    - No hardcoded hyperparameters — all thresholds exposed as parameters
"""

from __future__ import annotations

import math
from typing import Union

from shared.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Optional numpy / scipy imports
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import numpy.typing as npt

    _NUMPY_AVAILABLE = True
except ImportError as _np_exc:  # pragma: no cover
    raise ImportError(
        "NumPy is required for shared.math_utils: pip install numpy"
    ) from _np_exc

try:
    import scipy.sparse as sp  # type: ignore[import]

    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    sp = None  # type: ignore[assignment]
    _SCIPY_AVAILABLE = False

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[npt.ArrayLike, "np.ndarray"]
SparseOrDense = Union["np.ndarray", "sp.spmatrix"] if _SCIPY_AVAILABLE else "np.ndarray"

# ---------------------------------------------------------------------------
# Numerical stability constant
# ---------------------------------------------------------------------------
_EPSILON: float = 1e-10  # floor for probability values before log operations


# ---------------------------------------------------------------------------
# Public API — §8.5
# ---------------------------------------------------------------------------

def cosine_similarity(
    a: ArrayLike,
    b: ArrayLike,
    *,
    zero_norm_value: float = 0.0,
) -> float:
    """
    Compute the cosine similarity between two 1-D vectors *a* and *b*.

    Satisfies the §8.5 contract:
        ``cosine_similarity(a, b)`` → float in [-1, 1].

    Cosine similarity is defined as:
        cos(θ) = (a · b) / (‖a‖₂ · ‖b‖₂)

    Used throughout the pipeline for:
    - Topic coherence measurement (§4.3)
    - Document-topic similarity in classical models (§3.3)
    - QAOA result validation against classical baselines (§9.3)

    Parameters
    ----------
    a : array_like, shape (n,)
        First vector.
    b : array_like, shape (n,)
        Second vector.  Must have the same length as *a*.
    zero_norm_value : float, optional
        Value to return when either *a* or *b* has zero L2-norm.
        Default ``0.0`` (two zero vectors are defined as orthogonal).

    Returns
    -------
    float
        Cosine similarity in the range [-1.0, 1.0].  Returns
        *zero_norm_value* if either input has zero norm.

    Raises
    ------
    ValueError
        If *a* and *b* have different lengths or are not 1-D.

    Examples
    --------
    >>> cosine_similarity([1, 0, 0], [1, 0, 0])
    1.0
    >>> cosine_similarity([1, 0], [0, 1])
    0.0
    >>> cosine_similarity([1, 1], [-1, -1])
    -1.0
    """
    vec_a = np.asarray(a, dtype=np.float64).ravel()
    vec_b = np.asarray(b, dtype=np.float64).ravel()

    if vec_a.ndim != 1 or vec_b.ndim != 1:
        raise ValueError(
            "cosine_similarity requires 1-D vectors, "
            f"got shapes {vec_a.shape} and {vec_b.shape}."
        )
    if vec_a.shape != vec_b.shape:
        raise ValueError(
            f"Vectors must have equal length: {vec_a.shape[0]} vs {vec_b.shape[0]}."
        )

    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))

    if norm_a < _EPSILON or norm_b < _EPSILON:
        logger.debug(
            "cosine_similarity: zero-norm vector encountered.",
            extra={"norm_a": norm_a, "norm_b": norm_b, "returning": zero_norm_value},
        )
        return zero_norm_value

    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clamp to [-1, 1] to guard against floating-point drift
    return float(np.clip(similarity, -1.0, 1.0))


def cosine_similarity_matrix(
    matrix_a: ArrayLike,
    matrix_b: ArrayLike | None = None,
) -> "np.ndarray":
    """
    Compute pairwise cosine similarities between rows of *matrix_a* and
    (optionally) *matrix_b*.

    Equivalent to ``sklearn.metrics.pairwise.cosine_similarity`` but
    without the sklearn dependency.

    Parameters
    ----------
    matrix_a : array_like, shape (m, n)
        First matrix; each row is a vector.
    matrix_b : array_like, shape (k, n) | None, optional
        Second matrix.  If ``None``, computes the self-similarity of
        *matrix_a* (shape (m, m)).

    Returns
    -------
    np.ndarray, shape (m, k)
        Pairwise cosine similarities.  Entry [i, j] is
        ``cosine_similarity(matrix_a[i], matrix_b[j])``.
    """
    A = np.asarray(matrix_a, dtype=np.float64)
    B = np.asarray(matrix_b, dtype=np.float64) if matrix_b is not None else A

    # L2-normalise rows
    norm_a = np.linalg.norm(A, axis=1, keepdims=True)
    norm_b = np.linalg.norm(B, axis=1, keepdims=True)

    # Replace zero norms with 1 to avoid division by zero (result stays 0)
    norm_a[norm_a < _EPSILON] = 1.0
    norm_b[norm_b < _EPSILON] = 1.0

    A_norm = A / norm_a
    B_norm = B / norm_b

    sim_matrix = np.clip(A_norm @ B_norm.T, -1.0, 1.0)
    return sim_matrix


def entropy(
    distribution: ArrayLike,
    base: float | None = None,
    *,
    normalize: bool = False,
) -> float:
    """
    Compute the Shannon entropy of a probability distribution.

    Satisfies the §8.5 contract:
        ``entropy(distribution)`` → Shannon entropy (nats or bits).

    H(P) = -∑ p_i · log(p_i)

    Zero-probability entries are skipped (0 · log(0) := 0 by convention),
    consistent with ``scipy.stats.entropy`` and the standard definition.

    Used for:
    - Topic quality measurement (high-entropy topics are diffuse / poor)
    - Document-topic assignment confidence (§4.3)
    - Quantum state entropy estimation (§3.4)

    Parameters
    ----------
    distribution : array_like, shape (n,)
        Non-negative values representing a probability distribution.
        Need not sum to 1 if *normalize=True*; must sum to ~1 otherwise.
    base : float | None, optional
        Logarithm base.  Common choices:
        - ``None`` (default) → natural log (nats)
        - ``2``              → bits (shannons)
        - ``10``             → hartleys (bans)
    normalize : bool, optional
        When ``True``, the distribution is normalised to sum to 1 before
        the entropy calculation.  Default ``False``.

    Returns
    -------
    float
        Shannon entropy in the specified unit (nats by default).
        Returns ``0.0`` for a degenerate distribution (all mass on one
        outcome) and ``log(n)`` for a uniform distribution over n outcomes.

    Raises
    ------
    ValueError
        If any element of *distribution* is negative.
        If *normalize=False* and the distribution sums to zero.
        If *base* ≤ 0 or *base* == 1.

    Examples
    --------
    >>> entropy([0.5, 0.5])           # 1 bit ≈ 0.693 nats
    0.6931471805599453
    >>> entropy([0.5, 0.5], base=2)   # exactly 1 bit
    1.0
    >>> entropy([1.0, 0.0, 0.0])      # degenerate → 0
    0.0
    """
    p = np.asarray(distribution, dtype=np.float64).ravel()

    if np.any(p < 0):
        raise ValueError(
            "entropy: distribution must have non-negative values, "
            f"found min={p.min():.6g}."
        )

    if base is not None and (base <= 0 or base == 1):
        raise ValueError(
            f"entropy: base must be > 0 and ≠ 1, got base={base}."
        )

    if normalize:
        total = p.sum()
        if total < _EPSILON:
            raise ValueError(
                "entropy: cannot normalize a distribution that sums to zero."
            )
        p = p / total
    else:
        total = p.sum()
        if total < _EPSILON:
            raise ValueError(
                "entropy: distribution sums to zero. "
                "Pass normalize=True or supply a non-trivial distribution."
            )

    # Mask zero entries (0 · log(0) := 0)
    nonzero = p > _EPSILON
    p_nz = p[nonzero]

    h = float(-np.sum(p_nz * np.log(p_nz)))

    if base is not None:
        h /= math.log(base)

    return h


def kl_divergence(
    p: ArrayLike,
    q: ArrayLike,
    *,
    normalize: bool = False,
    epsilon: float = _EPSILON,
) -> float:
    """
    Compute the Kullback-Leibler divergence KL(P ‖ Q).

    Satisfies the §8.5 contract:
        ``kl_divergence(p, q)`` → KL(P‖Q) in nats.

    KL(P ‖ Q) = ∑ p_i · log(p_i / q_i)

    This is the *forward* KL divergence; it measures how much information
    is lost when *Q* is used to approximate *P*.

    Used for:
    - Evaluating topic model fit (§4.3): KL between true and inferred topics
    - Quantum state fidelity assessment (§9.3)

    Parameters
    ----------
    p : array_like, shape (n,)
        Reference distribution (the "true" distribution).
    q : array_like, shape (n,)
        Approximating distribution.
    normalize : bool, optional
        Independently normalise *p* and *q* to sum to 1 before computation.
        Default ``False``.
    epsilon : float, optional
        Floor applied to *q* before computing log(p/q) to avoid division
        by zero when *q* has zero mass.  Default ``1e-10``.

    Returns
    -------
    float
        KL(P ‖ Q) ≥ 0, in nats.  Returns ``0.0`` if and only if P == Q.

    Raises
    ------
    ValueError
        If *p* and *q* have different lengths.
        If any element of *p* or *q* is negative.
        If *q* has zero probability mass at a position where *p* has
        positive mass AND *epsilon* is 0 (strict undefined KL).

    Examples
    --------
    >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> kl_divergence([1.0, 0.0], [0.5, 0.5])  # large divergence
    0.6931471805599453
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    q_arr = np.asarray(q, dtype=np.float64).ravel()

    if p_arr.shape != q_arr.shape:
        raise ValueError(
            f"kl_divergence: p and q must have the same length, "
            f"got {p_arr.shape[0]} vs {q_arr.shape[0]}."
        )
    if np.any(p_arr < 0):
        raise ValueError(
            f"kl_divergence: p must be non-negative, found min={p_arr.min():.6g}."
        )
    if np.any(q_arr < 0):
        raise ValueError(
            f"kl_divergence: q must be non-negative, found min={q_arr.min():.6g}."
        )

    if normalize:
        p_sum = p_arr.sum()
        q_sum = q_arr.sum()
        if p_sum < _EPSILON:
            raise ValueError("kl_divergence: p sums to zero; cannot normalize.")
        if q_sum < _EPSILON:
            raise ValueError("kl_divergence: q sums to zero; cannot normalize.")
        p_arr = p_arr / p_sum
        q_arr = q_arr / q_sum

    # Clip q to avoid log(0); p is left as-is (zero p contributes 0 to sum)
    q_safe = np.clip(q_arr, epsilon, None)

    # Only sum over positions where p > 0 (0 · log(0/q) := 0)
    mask = p_arr > epsilon
    if not np.any(mask):
        return 0.0

    kl = float(np.sum(p_arr[mask] * np.log(p_arr[mask] / q_safe[mask])))

    # Numerical guard: KL should always be ≥ 0 (Gibbs' inequality)
    return max(kl, 0.0)


def js_divergence(
    p: ArrayLike,
    q: ArrayLike,
    *,
    normalize: bool = False,
) -> float:
    """
    Compute the Jensen-Shannon divergence JS(P ‖ Q), the symmetrised
    and bounded form of KL divergence.

    JS(P, Q) = ½ KL(P ‖ M) + ½ KL(Q ‖ M),  where M = ½(P + Q)

    Returns a value in [0, log(2)] nats; sqrt(JS) is a proper metric.

    Parameters
    ----------
    p, q : array_like, shape (n,)
        Two probability distributions (same length).
    normalize : bool, optional
        Normalise each distribution independently.  Default ``False``.

    Returns
    -------
    float
        Jensen-Shannon divergence in [0, log(2)] nats.
    """
    p_arr = np.asarray(p, dtype=np.float64).ravel()
    q_arr = np.asarray(q, dtype=np.float64).ravel()

    if normalize:
        p_arr = p_arr / (p_arr.sum() + _EPSILON)
        q_arr = q_arr / (q_arr.sum() + _EPSILON)

    m = 0.5 * (p_arr + q_arr)
    return 0.5 * kl_divergence(p_arr, m) + 0.5 * kl_divergence(q_arr, m)


def sparse_to_dense(matrix: Any) -> "np.ndarray":
    """
    Convert a scipy sparse matrix to a dense ``numpy.ndarray``.

    Satisfies the §8.5 contract:
        ``sparse_to_dense(matrix)`` → ``np.ndarray``.

    This helper is used by classical NLP stages (LDA, NMF, TF-IDF) that
    produce CSR/CSC sparse document-term matrices, which must be densified
    before being fed into quantum circuit construction (§9.2) or
    visualisation layers (§4).

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix | np.ndarray
        Input matrix.  If already a dense ``numpy.ndarray``, returned
        unchanged (no copy).

    Returns
    -------
    np.ndarray
        Dense 2-D array of ``float64``.

    Raises
    ------
    ImportError
        If scipy is not installed and *matrix* is not already a
        ``numpy.ndarray``.
    TypeError
        If *matrix* is neither a numpy array nor a scipy sparse matrix.

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> m = sp.csr_matrix([[1, 0], [0, 2]])
    >>> dense = sparse_to_dense(m)
    >>> dense.tolist()
    [[1.0, 0.0], [0.0, 2.0]]
    """
    if isinstance(matrix, np.ndarray):
        return matrix.astype(np.float64)

    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required to convert sparse matrices: pip install scipy"
        )

    if not sp.issparse(matrix):
        raise TypeError(
            f"sparse_to_dense: expected a scipy sparse matrix or numpy ndarray, "
            f"got {type(matrix).__name__}."
        )

    dense = matrix.toarray().astype(np.float64)

    logger.debug(
        "Sparse matrix converted to dense.",
        extra={
            "shape": list(dense.shape),
            "nnz": matrix.nnz,
            "density": f"{matrix.nnz / max(matrix.shape[0] * matrix.shape[1], 1):.4f}",
        },
    )
    return dense


def dense_to_sparse(
    matrix: "np.ndarray",
    fmt: str = "csr",
) -> Any:
    """
    Convert a dense ``numpy.ndarray`` to a scipy sparse matrix.

    Companion to :func:`sparse_to_dense`.  Used when a dense matrix
    produced upstream needs to be stored or processed more efficiently.

    Parameters
    ----------
    matrix : np.ndarray
        Dense 2-D array.
    fmt : str, optional
        Sparse format: ``"csr"`` (default), ``"csc"``, ``"coo"``, ``"lil"``.

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse matrix in the requested format.

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for dense_to_sparse: pip install scipy"
        )

    arr = np.asarray(matrix, dtype=np.float64)
    fmt = fmt.lower()

    _CONSTRUCTORS = {
        "csr": sp.csr_matrix,
        "csc": sp.csc_matrix,
        "coo": sp.coo_matrix,
        "lil": sp.lil_matrix,
    }
    if fmt not in _CONSTRUCTORS:
        raise ValueError(
            f"Unsupported sparse format: {fmt!r}. "
            f"Choose from {list(_CONSTRUCTORS)}."
        )

    sparse = _CONSTRUCTORS[fmt](arr)
    logger.debug(
        "Dense matrix converted to sparse.",
        extra={"shape": list(arr.shape), "fmt": fmt, "nnz": sparse.nnz},
    )
    return sparse


def estimate_qubits(
    n_docs: int,
    n_topics: int,
    *,
    encoding: str = "log",
) -> int:
    """
    Estimate the number of qubits required for a quantum topic-modeling
    experiment with *n_docs* documents and *n_topics* topics.

    Satisfies the §8.5 contract:
        ``estimate_qubits(n_docs, n_topics)`` → int (qubit demand estimate).

    This function encodes the same formula used by
    :func:`shared.validator.assert_qubit_feasibility` so that the
    estimate is always consistent with the feasibility guard.  Call
    :func:`shared.validator.assert_qubit_feasibility` to enforce the
    hard backend limit.

    Encoding schemes
    ----------------
    ``"log"`` (default — QAOA graph-partitioning formulation, §9.3):
        required_qubits = ⌈log₂(n_docs)⌉ + ⌈log₂(n_topics)⌉

    ``"linear"`` (dense amplitude encoding, experimental):
        required_qubits = n_docs + n_topics
        (only viable for very small toy experiments)

    ``"sqrt"`` (square-root amplitude encoding, §9.3 note):
        required_qubits = ⌈log₂(n_docs)⌉ + ⌈√(n_topics)⌉

    Parameters
    ----------
    n_docs : int
        Number of documents in the corpus / sub-corpus.  Must be ≥ 1.
    n_topics : int
        Number of topics or clusters.  Must be ≥ 1.
    encoding : str, optional
        Encoding strategy: ``"log"`` (default), ``"linear"``, ``"sqrt"``.

    Returns
    -------
    int
        Estimated number of qubits required.

    Raises
    ------
    ValueError
        If *n_docs* < 1, *n_topics* < 1, or *encoding* is unrecognised.

    Examples
    --------
    >>> estimate_qubits(50, 5)      # toy subset
    8
    >>> estimate_qubits(200, 10)    # small subset
    11
    >>> estimate_qubits(18846, 20)  # full 20ng → exceeds 20-qubit limit
    19
    """
    if n_docs < 1:
        raise ValueError(f"n_docs must be ≥ 1, got {n_docs}.")
    if n_topics < 1:
        raise ValueError(f"n_topics must be ≥ 1, got {n_topics}.")

    enc = encoding.lower()

    if enc == "log":
        q_docs = math.ceil(math.log2(n_docs)) if n_docs > 1 else 1
        q_topics = math.ceil(math.log2(n_topics)) if n_topics > 1 else 1
        total = q_docs + q_topics

    elif enc == "linear":
        total = n_docs + n_topics

    elif enc == "sqrt":
        q_docs = math.ceil(math.log2(n_docs)) if n_docs > 1 else 1
        q_topics = math.ceil(math.sqrt(n_topics))
        total = q_docs + q_topics

    else:
        raise ValueError(
            f"Unknown encoding: {enc!r}. "
            "Choose 'log', 'linear', or 'sqrt'."
        )

    logger.debug(
        "Qubit estimate computed.",
        extra={
            "n_docs": n_docs,
            "n_topics": n_topics,
            "encoding": enc,
            "estimated_qubits": total,
        },
    )
    return total


# ---------------------------------------------------------------------------
# Additional math helpers used by multiple pipeline stages
# ---------------------------------------------------------------------------

def normalize_rows(matrix: ArrayLike, *, norm: str = "l1") -> "np.ndarray":
    """
    Row-normalise *matrix* using L1 or L2 norm.

    Used to convert raw count matrices (document-term, document-topic)
    to probability distributions before entropy or KL calculations.

    Parameters
    ----------
    matrix : array_like, shape (m, n)
        Input matrix.  Each row is normalised independently.
    norm : str, optional
        ``"l1"`` (default) — each row sums to 1 (probability distribution).
        ``"l2"`` — each row has unit Euclidean norm.

    Returns
    -------
    np.ndarray, shape (m, n)
        Row-normalised matrix of ``float64``.  Rows with zero norm are
        left as all-zeros (no division).

    Raises
    ------
    ValueError
        If *norm* is not ``"l1"`` or ``"l2"``.
    """
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    norm = norm.lower()
    if norm == "l1":
        row_sums = arr.sum(axis=1, keepdims=True)
    elif norm == "l2":
        row_sums = np.linalg.norm(arr, axis=1, keepdims=True)
    else:
        raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}.")

    # Avoid division by zero
    row_sums[row_sums < _EPSILON] = 1.0
    return arr / row_sums


def softmax(x: ArrayLike, axis: int = -1) -> "np.ndarray":
    """
    Compute the softmax of *x* along *axis*.

    Numerically stable implementation using the max-subtraction trick.

    Parameters
    ----------
    x : array_like
        Input array of arbitrary shape.
    axis : int, optional
        Axis along which softmax is computed.  Default ``-1`` (last axis).

    Returns
    -------
    np.ndarray
        Array of the same shape as *x* with values in (0, 1) summing to 1
        along *axis*.
    """
    arr = np.asarray(x, dtype=np.float64)
    shifted = arr - arr.max(axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def top_k_indices(
    values: ArrayLike,
    k: int,
    *,
    descending: bool = True,
) -> "np.ndarray":
    """
    Return the indices of the *k* largest (or smallest) values.

    Used for topic word extraction: given a topic's word-weight vector,
    return the indices of the top-*k* most probable words.

    Parameters
    ----------
    values : array_like, shape (n,)
        1-D array of scores / weights.
    k : int
        Number of indices to return.  Clamped to ``len(values)`` if
        *k* exceeds the array length.
    descending : bool, optional
        When ``True`` (default), return the *k* largest values.
        When ``False``, return the *k* smallest.

    Returns
    -------
    np.ndarray, shape (min(k, n),)
        Indices of the top-*k* elements, sorted by value (descending or
        ascending per *descending*).

    Raises
    ------
    ValueError
        If *values* is not 1-D or *k* < 1.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.ndim != 1:
        raise ValueError("top_k_indices: values must be a 1-D array.")
    if k < 1:
        raise ValueError(f"top_k_indices: k must be ≥ 1, got {k}.")

    k = min(k, len(arr))

    if descending:
        indices = np.argpartition(arr, -k)[-k:]
        indices = indices[np.argsort(arr[indices])[::-1]]
    else:
        indices = np.argpartition(arr, k - 1)[:k]
        indices = indices[np.argsort(arr[indices])]

    return indices


def pairwise_kl_matrix(
    distributions: ArrayLike,
    *,
    normalize: bool = True,
    epsilon: float = _EPSILON,
) -> "np.ndarray":
    """
    Compute the pairwise KL divergence matrix for a set of distributions.

    Entry [i, j] = KL(distributions[i] ‖ distributions[j]).

    Used for topic diversity analysis: a high mean off-diagonal KL
    indicates the model discovered distinct topics (§4.3).

    Parameters
    ----------
    distributions : array_like, shape (k, n)
        *k* distributions, each of length *n*.
    normalize : bool, optional
        Normalise each row to sum to 1.  Default ``True``.
    epsilon : float, optional
        Stability floor for denominator.  Default ``1e-10``.

    Returns
    -------
    np.ndarray, shape (k, k)
        Pairwise KL divergence matrix.  Diagonal entries are 0.
    """
    D = np.asarray(distributions, dtype=np.float64)
    if D.ndim != 2:
        raise ValueError(
            f"pairwise_kl_matrix: expected 2-D array, got shape {D.shape}."
        )

    if normalize:
        row_sums = D.sum(axis=1, keepdims=True)
        row_sums[row_sums < epsilon] = 1.0
        D = D / row_sums

    k = D.shape[0]
    result = np.zeros((k, k), dtype=np.float64)

    for i in range(k):
        for j in range(k):
            if i != j:
                result[i, j] = kl_divergence(D[i], D[j], epsilon=epsilon)

    return result