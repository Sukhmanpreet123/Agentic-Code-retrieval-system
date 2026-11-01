"""
fusion.py
=========

WHAT THIS FILE DOES:
    Combines TWO separate ranked lists (one from BM25, one from the
    embedding search) into ONE ranked list.

WHY NOT JUST AVERAGE THE RAW SCORES?
    BM25 scores and cosine-similarity scores live on completely different
    scales (BM25 might be 0-30, cosine similarity is always -1 to 1).
    Averaging them directly would let whichever one happens to have
    bigger numbers dominate, for no principled reason.

RECIPROCAL RANK FUSION (RRF) SIDESTEPS THAT PROBLEM ENTIRELY:
    Instead of looking at the SCORES, it only looks at the RANK POSITION
    (1st place, 2nd place, ...) a document got in each list. A document's
    fused score is:

        fused_score(doc) = sum over each ranked list of  1 / (k + rank_in_that_list)

    A document that shows up near the top of BOTH lists gets a high fused
    score. A document that only one retriever found still gets SOME
    credit (instead of being thrown away), because BM25 and embeddings
    catch different kinds of relevant results (see strategy doc section 9).

    `k` (usually ~60) is a smoothing constant: it stops rank #1 from
    completely dominating rank #2, so the fusion is more robust to one
    retriever being slightly noisy for a particular query.
"""

from collections import defaultdict
from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float]]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """
    `ranked_lists` is a list of ranked results, e.g.:
        [
            [(doc_id, bm25_score), ...],       # from BM25Index.search()
            [(doc_id, cosine_score), ...],     # from VectorIndex.search()
        ]

    Returns one fused, re-sorted list: [(doc_id, fused_score), ...]
    """
    fused_scores: Dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (doc_id, _original_score) in enumerate(ranked_list, start=1):
            fused_scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return fused
