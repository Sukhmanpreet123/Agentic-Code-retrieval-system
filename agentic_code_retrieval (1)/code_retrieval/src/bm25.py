"""
bm25.py
=======

WHAT THIS FILE DOES:
    A small, from-scratch implementation of BM25 -- the classic lexical
    ("exact word match") search algorithm used by search engines for
    decades before embeddings existed.

WHY IMPLEMENT IT OURSELVES INSTEAD OF USING A LIBRARY?
    BM25 is genuinely simple (about 30 lines of real logic). Writing it
    ourselves means:
      1. Zero extra dependency.
      2. We can explain every line to a judge/interviewer, instead of
         saying "a library does it".
    (If you'd rather use a library, `rank_bm25` on PyPI implements the
    same formula -- this file is a drop-in-compatible alternative.)

HOW BM25 WORKS, IN ONE PARAGRAPH:
    For each word in the query, BM25 asks: "how rare is this word across
    the whole corpus, and how often/densely does it appear in THIS
    particular document?" Rare words that appear a lot in one document
    are a strong relevance signal. Very common words (like "the", "is")
    contribute almost nothing, because they appear everywhere. The two
    tuning constants (k1, b) control how quickly "more occurrences" stops
    mattering, and how much to penalize long documents for naturally
    containing more words.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def tokenize(text: str) -> List[str]:
    """Very simple tokenizer: pull out word-like tokens, lowercase them."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """
    Usage:
        index = BM25Index()
        index.build(doc_ids=["a", "b"], doc_texts=["hello world", "goodbye world"])
        results = index.search("hello", top_k=10)   # -> [(doc_id, score), ...]
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_term_freqs: List[Counter] = []   # term -> count, per document
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freq: Dict[str, int] = defaultdict(int)   # in how many docs does term appear
        self.num_docs: int = 0

    def build(self, doc_ids: List[str], doc_texts: List[str]) -> None:
        self.doc_ids = list(doc_ids)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_freq = defaultdict(int)

        for text in doc_texts:
            tokens = tokenize(text)
            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))
            for term in term_freq:
                self.doc_freq[term] += 1

        self.num_docs = len(doc_texts)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.num_docs if self.num_docs else 0.0
        )

    def _idf(self, term: str) -> float:
        """
        Inverse Document Frequency: rare terms get a high score,
        terms that appear in almost every document get a score near zero
        (or even slightly negative, which the +0.5/+1 smoothing avoids).
        """
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: List[str], doc_index: int) -> float:
        term_freq = self.doc_term_freqs[doc_index]
        doc_len = self.doc_lengths[doc_index]
        score = 0.0
        for term in query_tokens:
            f = term_freq.get(term, 0)
            if f == 0:
                continue
            idf = self._idf(term)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (
                1 - self.b + self.b * (doc_len / self.avg_doc_length)
            )
            score += idf * (numerator / denominator)
        return score

    def search(self, query_text: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Returns [(doc_id, bm25_score), ...] sorted best-first."""
        query_tokens = tokenize(query_text)
        scored = [
            (self.doc_ids[i], self.score(query_tokens, i))
            for i in range(self.num_docs)
        ]
        scored = [pair for pair in scored if pair[1] > 0]  # drop zero-score docs
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def has_exact_identifier_hit(self, identifier: str) -> bool:
        """
        Used by the pipeline to decide fusion weights: "does this literal
        identifier from the query appear verbatim in ANY document?"
        """
        token = identifier.lower()
        return self.doc_freq.get(token, 0) > 0
