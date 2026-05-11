"""
src/evaluation/topic_coherence.py
====================================
Team T-4: Hybrid Pipeline Team
MASTER BLUEPRINT v1.0 — §3.5

Responsibility:
    Compute C_V, NPMI, and UMass coherence scores for a trained topic model.

Inputs:  TopicModelResult (from src/classical/lda_model.py or nmf_model.py),
         corpus (List[CleanDocument])
Outputs: CoherenceScores dataclass

Integration points:
    - Consumes T-2 artifacts: TopicModelResult, CleanDocument list
    - Results consumed by benchmark_runner.py and /eval/coherence/{model_id} endpoint
    - Uses shared/logger.get_logger
    - MLflow logging via mlflow.log_metrics (Blueprint §10.4)

Blueprint API endpoint: GET /eval/coherence/{model_id} → { model_id, c_v, npmi, umass }
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from shared.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOPN: int = 10          # top-N words per topic used for coherence
DEFAULT_EPSILON: float = 1e-12  # smoothing to avoid log(0)
MIN_DOCS_FOR_COHERENCE: int = 5


# ---------------------------------------------------------------------------
# Output contract (Blueprint §5.2 / §4.5)
# ---------------------------------------------------------------------------

@dataclass
class CoherenceScores:
    """
    Coherence result contract consumed by benchmark_runner and the eval API.

    Fields
    ------
    model_id    : Identifier of the evaluated topic model.
    c_v         : float — C_V coherence (sliding window + cosine similarity).
    npmi        : float — Normalised Pointwise Mutual Information (macro-avg).
    umass       : float — UMass coherence (log co-document frequency).
    per_topic   : Dict — per-topic breakdown for all three metrics.
    n_topics    : int — number of topics evaluated.
    topn        : int — number of top words used.
    metadata    : Dict — provenance (corpus_id, model type, …).
    """
    model_id: str
    c_v: float
    npmi: float
    umass: float
    per_topic: Dict[int, Dict[str, float]] = field(default_factory=dict)
    n_topics: int = 0
    topn: int = DEFAULT_TOPN
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core evaluator
# ---------------------------------------------------------------------------

class TopicCoherenceEvaluator:
    """
    Computes C_V, NPMI, and UMass coherence from topic-word distributions
    and a reference corpus of tokenised documents.

    All three measures are computed internally without Gensim so the module
    is always importable in CI. When Gensim is present, the Gensim pipeline
    is used as a higher-fidelity C_V estimator (toggled via use_gensim flag).

    Example
    -------
    >>> evaluator = TopicCoherenceEvaluator(topn=10)
    >>> scores = evaluator.evaluate(
    ...     topic_result=topic_model_result,   # TopicModelResult dict / object
    ...     documents=clean_doc_list,          # List[CleanDocument]
    ...     model_id="lda_20ng_abc123",
    ...     corpus_id="20ng",
    ... )
    >>> print(scores.c_v, scores.npmi, scores.umass)
    """

    def __init__(
        self,
        topn: int = DEFAULT_TOPN,
        epsilon: float = DEFAULT_EPSILON,
        use_gensim: bool = True,
    ) -> None:
        self.topn = topn
        self.epsilon = epsilon
        self.use_gensim = use_gensim

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        topic_result: Dict,
        documents: List[Dict],
        model_id: str,
        corpus_id: str = "unknown",
        log_to_mlflow: bool = True,
    ) -> CoherenceScores:
        """
        Compute coherence scores for a TopicModelResult against a document corpus.

        Parameters
        ----------
        topic_result : TopicModelResult serialised as dict. Must contain 'topics'
                       — list of {word: weight} dicts or (word, weight) lists.
        documents    : List[CleanDocument] dicts with 'tokens' key (List[str]).
        model_id     : Unique model identifier for logging.
        corpus_id    : Source corpus identifier.
        log_to_mlflow: Emit scores to the active MLflow run.

        Returns
        -------
        CoherenceScores
        """
        topics = self._extract_top_words(topic_result)
        n_topics = len(topics)

        if len(documents) < MIN_DOCS_FOR_COHERENCE:
            logger.warning(
                "Too few documents for reliable coherence",
                extra={"n_docs": len(documents), "model_id": model_id},
            )

        token_lists = self._to_token_lists(documents)
        doc_freq, cooc_freq, n_docs = self._build_frequency_index(token_lists)

        logger.info(
            "Computing coherence scores",
            extra={"model_id": model_id, "n_topics": n_topics, "topn": self.topn},
        )

        per_topic: Dict[int, Dict[str, float]] = {}
        umass_scores, npmi_scores, cv_scores = [], [], []

        for t_idx, top_words in enumerate(topics):
            u = self._umass_topic(top_words, doc_freq, cooc_freq, n_docs)
            n = self._npmi_topic(top_words, doc_freq, cooc_freq, n_docs)
            c = self._cv_topic(top_words, doc_freq, cooc_freq, n_docs)
            per_topic[t_idx] = {"umass": u, "npmi": n, "c_v": c}
            umass_scores.append(u)
            npmi_scores.append(n)
            cv_scores.append(c)

        # Macro-average across topics
        c_v_mean = float(np.mean(cv_scores)) if cv_scores else 0.0
        npmi_mean = float(np.mean(npmi_scores)) if npmi_scores else 0.0
        umass_mean = float(np.mean(umass_scores)) if umass_scores else 0.0

        # Gensim override for C_V if available and requested
        if self.use_gensim:
            c_v_mean = self._gensim_cv(topics, token_lists, c_v_mean)

        scores = CoherenceScores(
            model_id=model_id,
            c_v=round(c_v_mean, 6),
            npmi=round(npmi_mean, 6),
            umass=round(umass_mean, 6),
            per_topic=per_topic,
            n_topics=n_topics,
            topn=self.topn,
            metadata={
                "corpus_id": corpus_id,
                "n_docs": len(documents),
                "use_gensim": self.use_gensim,
            },
        )

        if log_to_mlflow:
            self._log_mlflow(model_id, scores)

        logger.info(
            "Coherence computed",
            extra={"model_id": model_id, "c_v": scores.c_v,
                   "npmi": scores.npmi, "umass": scores.umass},
        )
        return scores

    # ------------------------------------------------------------------
    # Frequency index construction
    # ------------------------------------------------------------------

    def _to_token_lists(self, documents: List[Dict]) -> List[List[str]]:
        return [doc.get("tokens", doc.get("lemmas", [])) for doc in documents]

    def _build_frequency_index(
        self, token_lists: List[List[str]]
    ) -> Tuple[Counter, Dict[Tuple[str, str], int], int]:
        """
        Build:
            doc_freq  : {word: n_docs_containing_word}
            cooc_freq : {(w_i, w_j): n_docs_both_present}
            n_docs    : total document count
        """
        n_docs = len(token_lists)
        doc_freq: Counter = Counter()
        cooc_freq: Dict[Tuple[str, str], int] = defaultdict(int)

        for tokens in token_lists:
            unique = set(tokens)
            for w in unique:
                doc_freq[w] += 1
            unique_list = sorted(unique)
            for i, w_i in enumerate(unique_list):
                for w_j in unique_list[i + 1:]:
                    cooc_freq[(w_i, w_j)] += 1

        return doc_freq, cooc_freq, n_docs

    # ------------------------------------------------------------------
    # UMass coherence  (Mimno et al. 2011)
    # ------------------------------------------------------------------

    def _umass_topic(
        self,
        words: List[str],
        doc_freq: Counter,
        cooc_freq: Dict,
        n_docs: int,
    ) -> float:
        score = 0.0
        for m in range(1, len(words)):
            w_m = words[m]
            for l in range(m):
                w_l = words[l]
                key = (min(w_l, w_m), max(w_l, w_m))
                d_ml = cooc_freq.get(key, 0)
                d_l = doc_freq.get(w_l, 0)
                score += math.log((d_ml + self.epsilon) / max(d_l, 1))
        return score

    # ------------------------------------------------------------------
    # NPMI coherence  (Bouma 2009, normalised)
    # ------------------------------------------------------------------

    def _npmi_topic(
        self,
        words: List[str],
        doc_freq: Counter,
        cooc_freq: Dict,
        n_docs: int,
    ) -> float:
        scores = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                w_i, w_j = words[i], words[j]
                key = (min(w_i, w_j), max(w_i, w_j))
                p_i = doc_freq.get(w_i, 0) / n_docs
                p_j = doc_freq.get(w_j, 0) / n_docs
                p_ij = cooc_freq.get(key, 0) / n_docs
                if p_ij < self.epsilon or p_i < self.epsilon or p_j < self.epsilon:
                    scores.append(-1.0)
                    continue
                pmi = math.log(p_ij / (p_i * p_j))
                norm = -math.log(p_ij + self.epsilon)
                scores.append(pmi / norm if norm != 0 else 0.0)
        return float(np.mean(scores)) if scores else 0.0

    # ------------------------------------------------------------------
    # C_V coherence  (Röder et al. 2015 approximation)
    # ------------------------------------------------------------------

    def _cv_topic(
        self,
        words: List[str],
        doc_freq: Counter,
        cooc_freq: Dict,
        n_docs: int,
    ) -> float:
        """
        Sliding-window NPMI aggregated into a pseudo-cosine similarity.
        Full C_V uses a 110-word window; here we use the full word-pair set
        as a lightweight approximation consistent with the NPMI implementation.
        """
        npmi_vec = []
        for i in range(len(words)):
            row = []
            for j in range(len(words)):
                if i == j:
                    row.append(1.0)
                    continue
                w_i, w_j = words[i], words[j]
                key = (min(w_i, w_j), max(w_i, w_j))
                p_i = doc_freq.get(w_i, 0) / n_docs
                p_j = doc_freq.get(w_j, 0) / n_docs
                p_ij = cooc_freq.get(key, 0) / n_docs
                if p_ij < self.epsilon or p_i < self.epsilon or p_j < self.epsilon:
                    row.append(0.0)
                    continue
                pmi = math.log(p_ij / (p_i * p_j))
                norm = -math.log(p_ij + self.epsilon)
                row.append(pmi / norm if norm != 0 else 0.0)
            npmi_vec.append(row)

        # Mean pairwise cosine (word vectors are their NPMI rows)
        mat = np.array(npmi_vec, dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + self.epsilon
        mat_normed = mat / norms
        cos_sim = float(np.mean(mat_normed @ mat_normed.T))
        return cos_sim

    # ------------------------------------------------------------------
    # Gensim override (higher-fidelity C_V when available)
    # ------------------------------------------------------------------

    def _gensim_cv(
        self,
        topics: List[List[str]],
        token_lists: List[List[str]],
        fallback: float,
    ) -> float:
        try:
            from gensim.models.coherencemodel import CoherenceModel  # type: ignore
            from gensim.corpora import Dictionary  # type: ignore

            dictionary = Dictionary(token_lists)
            cm = CoherenceModel(
                topics=topics,
                texts=token_lists,
                dictionary=dictionary,
                coherence="c_v",
                topn=self.topn,
            )
            return float(cm.get_coherence())
        except Exception as exc:
            logger.debug("Gensim C_V unavailable — using internal approximation", extra={"reason": str(exc)})
            return fallback

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_top_words(self, topic_result: Dict) -> List[List[str]]:
        """
        Extract the top-N word list per topic from a TopicModelResult dict.
        Handles both {word: weight} dict format and [(word, weight)] list format.
        """
        raw_topics = topic_result.get("topics", [])
        top_words_per_topic: List[List[str]] = []
        for topic in raw_topics:
            if isinstance(topic, dict):
                words = sorted(topic, key=topic.get, reverse=True)[: self.topn]
            elif isinstance(topic, (list, tuple)):
                words = [w for w, _ in sorted(topic, key=lambda x: x[1], reverse=True)][: self.topn]
            else:
                words = []
            top_words_per_topic.append(words)
        return top_words_per_topic

    def _log_mlflow(self, model_id: str, scores: CoherenceScores) -> None:
        try:
            import mlflow  # type: ignore
            mlflow.log_metrics({
                "coherence_c_v": scores.c_v,
                "coherence_npmi": scores.npmi,
                "coherence_umass": scores.umass,
            })
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level convenience function (Blueprint §4.5 endpoint backing)
# ---------------------------------------------------------------------------

def compute_coherence(
    topic_result: Dict,
    documents: List[Dict],
    model_id: str,
    topn: int = DEFAULT_TOPN,
    corpus_id: str = "unknown",
    use_gensim: bool = True,
    log_to_mlflow: bool = True,
) -> CoherenceScores:
    """
    Compute topic coherence scores.

    Consumed by:
        - benchmark_runner.py (T-4)
        - GET /eval/coherence/{model_id} API route (src/api/routes/eval_routes.py)

    Example
    -------
    >>> from src.evaluation.topic_coherence import compute_coherence
    >>> scores = compute_coherence(
    ...     topic_result=lda_result,
    ...     documents=clean_docs,
    ...     model_id="lda_20ng_abc123",
    ...     topn=10,
    ...     corpus_id="20ng",
    ... )
    >>> print(scores.c_v, scores.npmi, scores.umass)
    """
    evaluator = TopicCoherenceEvaluator(topn=topn, use_gensim=use_gensim)
    return evaluator.evaluate(
        topic_result=topic_result,
        documents=documents,
        model_id=model_id,
        corpus_id=corpus_id,
        log_to_mlflow=log_to_mlflow,
    )