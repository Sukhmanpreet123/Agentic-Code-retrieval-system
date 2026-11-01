"""
reranker.py
===========

WHAT THIS FILE DOES:
    Takes the fused top-N candidates (e.g. top-50) and re-sorts them more
    carefully, by looking at the QUERY and EACH SNIPPET TOGETHER at the
    same time -- something BM25 and embedding search can't do, because
    they score the query and each document SEPARATELY and then compare.

    Why does looking at them together help? A cross-encoder can notice
    things like "this query asks about ERROR HANDLING and this specific
    snippet's only error handling is a bare `except: pass`" -- a subtle
    interaction that two independent vectors can't capture.

TWO OPTIONS, SAME INTERFACE (`.rerank(query, candidates) -> sorted candidates`):

    1. CrossEncoderReranker (real option, used for the actual submission)
       - Wraps a small pretrained cross-encoder via `sentence-transformers`.
       - Requires: pip install sentence-transformers
       - Only ever called on a SHORT list (~50 items), never the whole
         corpus -- that's what keeps it CPU-fast (strategy doc section 13).

    2. FeatureBasedReranker (offline fallback / lightweight alternative)
       - No model download needed.
       - Combines a handful of hand-built features (BM25 score, cosine
         similarity, whether the query's literal identifier appears
         verbatim in the snippet, etc.) with a simple weighted sum.
       - Less accurate than a cross-encoder, but instant, fully
         interpretable, and a good "explain this to a judge" fallback if
         the cross-encoder proves too slow on your machine.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Candidate:
    """One candidate snippet being considered for the final top-10."""
    doc_id: str
    text: str                 # the chunk's indexable text (see chunking.py)
    bm25_score: float = 0.0
    embedding_score: float = 0.0
    fusion_score: float = 0.0
    final_score: float = 0.0  # filled in by the reranker


class CrossEncoderReranker:
    """
    Real option. Small pretrained cross-encoder checkpoint suggestion:
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    (swap for a code-specific cross-encoder checkpoint if you find/train one).
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name or self.MODEL_NAME)

    def rerank(self, query: str, candidates: List[Candidate]) -> List[Candidate]:
        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)
        for candidate, score in zip(candidates, scores):
            candidate.final_score = float(score)
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates


class FeatureBasedReranker:
    """
    Offline fallback. A transparent, hand-weighted combination of signals
    we already computed earlier in the pipeline -- no model, no download.
    """

    def __init__(self, bm25_weight: float = 0.4, embedding_weight: float = 0.4,
                 exact_identifier_weight: float = 0.2):
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        self.exact_identifier_weight = exact_identifier_weight

    def rerank(self, query: str, candidates: List[Candidate],
               literal_identifiers: List[str] = None) -> List[Candidate]:
        literal_identifiers = literal_identifiers or []

        # normalize bm25/embedding scores to 0-1 range across THIS candidate
        # list only, so the weights above stay meaningful regardless of
        # each score's raw scale.
        bm25_values = [c.bm25_score for c in candidates] or [0.0]
        embedding_values = [c.embedding_score for c in candidates] or [0.0]
        bm25_max = max(bm25_values) or 1.0
        embedding_max = max(embedding_values) or 1.0

        for candidate in candidates:
            exact_hit = any(
                identifier.lower() in candidate.text.lower()
                for identifier in literal_identifiers
            )
            candidate.final_score = (
                self.bm25_weight * (candidate.bm25_score / bm25_max)
                + self.embedding_weight * (candidate.embedding_score / embedding_max)
                + self.exact_identifier_weight * (1.0 if exact_hit else 0.0)
            )

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates


def get_reranker(backend: str = "feature"):
    if backend == "feature":
        return FeatureBasedReranker()
    if backend in ("cross-encoder", "cross_encoder"):
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker backend: {backend}")
