"""
query_processing.py
====================

WHAT THIS FILE DOES:
    Takes the raw natural-language query the user typed and does three
    small, CHEAP things to it before retrieval even starts:

    1. Split camelCase / snake_case identifiers into normal words, so
       a query mentioning "getUserData" also matches code that contains
       tokens like "get", "user", "data" (and vice versa).
    2. Detect whether the query already contains a literal code identifier
       (e.g. the user typed a real function name). This is used later to
       decide how much to trust BM25 (exact match) vs. embeddings
       (semantic match) for THIS particular query.
    3. Nothing fancier than that. No LLM calls, no ML model, no network
       request -- this whole file runs in microseconds, which matters
       because the hackathon brief requires the system to stay fast and
       CPU-only.

WHY NOT MORE (query expansion, 10-category classifiers, etc.)?
    The strategy doc explains this in detail (section 10): fancier query
    understanding is easy to get wrong and hard to prove helps. We only
    keep what's simple, deterministic, and explainable.
"""

import re
from dataclasses import dataclass
from typing import List


_CAMEL_CASE_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_IDENTIFIER_LIKE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9]+)+|[a-z]+[A-Z][a-zA-Z0-9]*)\b")


def split_identifier(token: str) -> List[str]:
    """
    "getUserData"  -> ["get", "User", "Data"] -> lowercased -> ["get","user","data"]
    "get_user_data" -> ["get", "user", "data"]
    "trim"          -> ["trim"]   (already a normal word, unchanged)
    """
    # snake_case -> split on underscores first
    parts = token.split("_")
    words = []
    for part in parts:
        if not part:
            continue
        # camelCase -> split before each uppercase letter that follows a lowercase/digit
        words.extend(_CAMEL_CASE_SPLIT.split(part))
    return [w.lower() for w in words if w]


@dataclass
class ProcessedQuery:
    raw_text: str
    normalized_text: str      # identifiers split into plain words, everything lowercased
    has_literal_identifier: bool  # did the ORIGINAL query contain a camelCase/snake_case token?
    literal_identifiers: List[str]  # the actual identifier-looking tokens found


def preprocess_query(raw_query: str) -> ProcessedQuery:
    """
    Main entry point. Returns a ProcessedQuery with:
      - normalized_text: safe to feed into BOTH BM25 and the embedding model
      - has_literal_identifier: a cheap signal used by the retrieval pipeline
        to decide whether to lean more heavily on BM25 for this query
        (see pipeline.py -> `_decide_fusion_weights`)
    """
    literal_identifiers = _IDENTIFIER_LIKE.findall(raw_query)

    tokens = raw_query.split()
    expanded_tokens: List[str] = []
    for tok in tokens:
        clean_tok = re.sub(r"[^\w]", "", tok)
        if not clean_tok:
            expanded_tokens.append(tok)
            continue
        if clean_tok != clean_tok.lower() or "_" in clean_tok:
            # looks like an identifier -> split it, but KEEP the original
            # token too, so an exact match against code is still possible
            expanded_tokens.append(clean_tok)
            expanded_tokens.extend(split_identifier(clean_tok))
        else:
            expanded_tokens.append(clean_tok)

    normalized_text = " ".join(expanded_tokens).lower()

    return ProcessedQuery(
        raw_text=raw_query,
        normalized_text=normalized_text,
        has_literal_identifier=len(literal_identifiers) > 0,
        literal_identifiers=literal_identifiers,
    )
