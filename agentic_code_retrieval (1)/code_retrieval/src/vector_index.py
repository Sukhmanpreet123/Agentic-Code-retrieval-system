"""
vector_index.py
================

WHAT THIS FILE DOES:
    Stores every chunk's embedding vector in one big numpy matrix, and
    answers "which vectors are most similar to this query vector?" by
    doing a single matrix multiplication.

WHY NOT FAISS?
    FAISS is the industry-standard tool for searching MILLIONS of vectors
    fast using approximate search. At hackathon scale (thousands of code
    chunks), an exact brute-force search with numpy is:
      - Fast enough (a few milliseconds),
      - Exact (no approximation error),
      - Zero extra dependency,
      - Much easier to explain to a judge ("we multiply the query vector
        by the corpus matrix and sort") than "we use an HNSW graph".

    If your corpus grows into the hundreds of thousands, swap this class
    for `faiss.IndexFlatIP` (same "inner product on normalized vectors =
    cosine similarity" idea, just implemented in C++ for speed).
"""

from typing import List, Tuple

import numpy as np


class VectorIndex:
    def __init__(self):
        self.doc_ids: List[str] = []
        self.matrix: np.ndarray = None  # shape: (num_docs, embedding_dim)

    def build(self, doc_ids: List[str], vectors: np.ndarray) -> None:
        """
        `vectors` must already be L2-normalized (embedder.py does this),
        so that a dot product IS a cosine similarity.
        """
        self.doc_ids = list(doc_ids)
        self.matrix = vectors

    def search(self, query_vector: np.ndarray, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Compute cosine similarity between the query and every stored
        vector in one shot, then take the top_k highest.

            similarities = matrix @ query_vector      # shape: (num_docs,)
        """
        if self.matrix is None or len(self.doc_ids) == 0:
            return []

        similarities = self.matrix @ query_vector

        # argpartition is an O(n) way to find the top_k largest values
        # without fully sorting everything -- then we sort just those.
        k = min(top_k, len(similarities))
        top_indices = np.argpartition(-similarities, k - 1)[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [(self.doc_ids[i], float(similarities[i])) for i in top_indices]

    def add_or_update(self, doc_ids: List[str], vectors: np.ndarray) -> None:
        """
        Incrementally add new vectors / overwrite vectors for existing
        doc_ids -- used by versioning.py when only some chunks changed.
        """
        existing = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        for doc_id, vector in zip(doc_ids, vectors):
            if doc_id in existing:
                self.matrix[existing[doc_id]] = vector
            else:
                self.doc_ids.append(doc_id)
                self.matrix = (
                    vector.reshape(1, -1)
                    if self.matrix is None
                    else np.vstack([self.matrix, vector.reshape(1, -1)])
                )

    def remove(self, doc_ids_to_remove: List[str]) -> None:
        keep_mask = [doc_id not in set(doc_ids_to_remove) for doc_id in self.doc_ids]
        self.doc_ids = [d for d, keep in zip(self.doc_ids, keep_mask) if keep]
        if self.matrix is not None:
            self.matrix = self.matrix[keep_mask]
