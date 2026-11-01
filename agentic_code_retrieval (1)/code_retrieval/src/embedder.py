"""
embedder.py
===========

WHAT THIS FILE DOES:
    Turns a list of text strings (chunk text or a query) into a list of
    numeric vectors ("embeddings") that capture MEANING, not just exact
    words. Two documents about "the same idea" end up with vectors that
    point in a similar direction, even if they don't share a single word.

TWO BACKENDS, SAME INTERFACE:
    1. TfidfEmbedder
       - Pure scikit-learn, no internet, no model download.
       - This is a bag-of-words method dressed up with dimensionality
         reduction (SVD) -- NOT true semantic understanding, but good
         enough to develop and test the whole pipeline offline.
       - Use this while you're building/debugging the system.

    2. SentenceTransformerEmbedder
       - Wraps a real pretrained embedding model via the
         `sentence-transformers` library (requires `pip install
         sentence-transformers` and an internet connection the first
         time, to download the model).
       - This is what you should use for the ACTUAL hackathon
         submission, because it's what gives you real semantic
         retrieval quality (the strategy doc's P0 recommendation).
       - Recommended small, code-aware checkpoints (swap the name in
         MODEL_NAME below):
           "flax-sentence-embeddings/st-codesearch-distilroberta-base"
           "sentence-transformers/all-MiniLM-L6-v2"   (general-purpose, still small)

    Both classes expose the exact same two methods:
        .fit(texts)          -- call once, on the whole corpus (builds vocab/model)
        .encode(texts)        -- returns a numpy array, shape (len(texts), dim)

    Because the interface is identical, the rest of the pipeline
    (vector_index.py, pipeline.py) doesn't care which backend is plugged
    in. Swapping TfidfEmbedder for SentenceTransformerEmbedder is a
    ONE-LINE change in pipeline.py.
"""

from typing import List

import numpy as np


class TfidfEmbedder:
    """Offline, dependency-light embedder. Good for development/testing."""

    def __init__(self, n_components: int = 128):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        self.vectorizer = TfidfVectorizer(max_features=20000)
        self.n_components = n_components
        self.svd = None  # built during fit(), once we know corpus size

    def fit(self, texts: List[str]) -> None:
        from sklearn.decomposition import TruncatedSVD

        sparse_matrix = self.vectorizer.fit_transform(texts)
        # TruncatedSVD needs n_components < min(n_samples, n_features)
        safe_components = min(self.n_components, sparse_matrix.shape[1] - 1, sparse_matrix.shape[0] - 1)
        safe_components = max(safe_components, 2)
        self.svd = TruncatedSVD(n_components=safe_components, random_state=0)
        self.svd.fit(sparse_matrix)

    def encode(self, texts: List[str]) -> np.ndarray:
        sparse_matrix = self.vectorizer.transform(texts)
        dense = self.svd.transform(sparse_matrix)
        return _l2_normalize(dense.astype(np.float32))


class SentenceTransformerEmbedder:
    """
    Real semantic embedder. Requires:
        pip install sentence-transformers

    This is the backend you use for your actual NDCG@10 / MRR submission.
    """

    MODEL_NAME = "flax-sentence-embeddings/st-codesearch-distilroberta-base"

    def __init__(self, model_name: str = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name or self.MODEL_NAME)

    def fit(self, texts: List[str]) -> None:
        # Pretrained models need no fitting -- this exists only so the
        # interface matches TfidfEmbedder.
        pass

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # <- already L2-normalized here
        )
        return vectors.astype(np.float32)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """
    Rescale every row vector to length 1. This makes a plain dot product
    equivalent to cosine similarity, which is what vector_index.py assumes.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def get_embedder(backend: str = "tfidf"):
    """
    Factory function so the rest of the code can just say:
        embedder = get_embedder("tfidf")       # for offline dev/testing
        embedder = get_embedder("sentence-transformers")   # for real submission
    """
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend in ("sentence-transformers", "st", "semantic"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedder backend: {backend}")
