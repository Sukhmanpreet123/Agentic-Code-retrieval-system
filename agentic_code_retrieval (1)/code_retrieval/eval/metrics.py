"""
metrics.py
==========

Implements the two official scoring metrics (NDCG@10 and MRR) from
scratch, in plain Python, so you can:
  1. Fully explain how your self-evaluation works to a judge.
  2. Run quick ablations (baseline vs hybrid vs +reranker) on your own
     dev set BEFORE running the full, slower official MTEB evaluation.

Both functions take the same simple shape:
    ranked_doc_ids: the list of doc_ids YOUR system returned, best first
    relevant_doc_ids: the set of doc_ids that are ACTUALLY correct for
                       this query (the "ground truth" / qrels)
"""

import math
from typing import Iterable, List, Set


def reciprocal_rank(ranked_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """
    MRR ingredient for ONE query: 1 / (rank of the first relevant result).
    If no relevant result was retrieved at all, this is 0.
    """
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_doc_ids: List[str], relevant_doc_ids: Set[str], k: int = 10) -> float:
    """
    NDCG@k for ONE query. Rewards relevant documents appearing EARLY in
    the top-k, and normalizes by the best possible ranking so the score
    is always between 0 and 1.
    """
    def dcg(doc_ids: Iterable[str]) -> float:
        return sum(
            (1.0 if doc_id in relevant_doc_ids else 0.0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(doc_ids, start=1)
        )

    actual_dcg = dcg(ranked_doc_ids[:k])
    ideal_ranking = list(relevant_doc_ids)[:k]  # best possible order = all relevant docs first
    ideal_dcg = dcg(ideal_ranking)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_system(query_results: List[List[str]], query_relevant_sets: List[Set[str]], k: int = 10):
    """
    Averages MRR and NDCG@k across many queries -- this is what the
    "Experiment Table" in the strategy doc (section 19) compares row by
    row (baseline vs hybrid vs +reranker, etc.).
    """
    mrr_scores = [
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(query_results, query_relevant_sets)
    ]
    ndcg_scores = [
        ndcg_at_k(ranked, relevant, k=k)
        for ranked, relevant in zip(query_results, query_relevant_sets)
    ]
    return {
        "MRR": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        f"NDCG@{k}": sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0,
        "num_queries": len(query_results),
    }
