"""
pipeline.py
===========

WHAT THIS FILE DOES:
    This is the "conductor" -- it wires together every other file in this
    project into the ONE flow described in the strategy doc:

        Code corpus --(chunking.py)--> chunks
        chunks --(bm25.py + embedder.py + vector_index.py)--> two indexes
        query --(query_processing.py)--> cleaned query
        cleaned query --(bm25.py)--> BM25 top-100
        cleaned query --(embedder.py + vector_index.py)--> embedding top-100
        both lists --(fusion.py)--> fused top-50
        fused top-50 --(reranker.py)--> final top-10
        [if low confidence]--> one more retrieval round with extra terms
        --(versioning.py)--> only re-index what changed, when the code updates

    Read this file top-to-bottom and you have read the entire retrieval
    system's control flow in one place.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from bm25 import BM25Index
from chunking import CodeChunk, chunk_repository
from embedder import get_embedder
from fusion import reciprocal_rank_fusion
from query_processing import preprocess_query
from reranker import Candidate, get_reranker
from vector_index import VectorIndex
from versioning import LineageStore, diff_versions


@dataclass
class RetrievalResult:
    doc_id: str
    function_name: str
    file_path: str
    code: str
    final_score: float
    lineage_versions: Optional[List[str]] = None  # for the Bonus: other versions of this function


class CodeRetrievalPipeline:
    def __init__(self, embedder_backend: str = "tfidf", reranker_backend: str = "feature",
                 initial_top_k: int = 100, fused_top_k: int = 50, final_top_k: int = 10,
                 low_confidence_threshold: float = 0.15):
        """
        embedder_backend: "tfidf" (offline/dev) or "sentence-transformers" (real submission)
        reranker_backend: "feature" (offline/dev) or "cross-encoder" (real submission)

        low_confidence_threshold: if the fused score of our #1 result is
        below this, we treat the query as "ambiguous" and run one extra
        retrieval pass (see `_run_second_pass_if_needed`). This threshold
        should be tuned on your own dev set -- see eval/run_experiments.py.
        """
        self.embedder = get_embedder(embedder_backend)
        self.reranker = get_reranker(reranker_backend)
        self.bm25_index = BM25Index()
        self.vector_index = VectorIndex()
        self.lineage_store = LineageStore()

        self.chunks_by_id: Dict[str, CodeChunk] = {}
        self.initial_top_k = initial_top_k
        self.fused_top_k = fused_top_k
        self.final_top_k = final_top_k
        self.low_confidence_threshold = low_confidence_threshold

    # ------------------------------------------------------------------
    # OFFLINE: building / updating the indexes
    # ------------------------------------------------------------------

    def build_from_files(self, file_path_to_source: Dict[str, str], version_label: str = "v1") -> None:
        """First-time build: chunk everything, embed everything, index everything."""
        chunks = chunk_repository(file_path_to_source)
        self._index_chunks(chunks, is_full_rebuild=True)
        self.lineage_store.record_version(version_label, chunks)

    def update_to_new_version(self, file_path_to_source: Dict[str, str], version_label: str) -> dict:
        """
        Incremental update (P1): figure out what changed since the last
        build, and ONLY re-embed/re-index those chunks. Returns a small
        report so you can show "we only touched N of M chunks" in your demo.
        """
        new_chunks = chunk_repository(file_path_to_source)
        previous_chunks = list(self.chunks_by_id.values())
        diff = diff_versions(previous_chunks, new_chunks)

        # remove chunks that no longer exist
        if diff.removed_chunk_ids:
            self.vector_index.remove(diff.removed_chunk_ids)
            for chunk_id in diff.removed_chunk_ids:
                self.chunks_by_id.pop(chunk_id, None)

        # re-index only added/modified chunks
        self._index_chunks(diff.chunks_needing_reindex, is_full_rebuild=False)

        # unchanged chunks: keep their existing entries, just make sure
        # they're recorded under the new version label too
        for chunk in diff.unchanged:
            self.chunks_by_id[chunk.chunk_id] = chunk

        self.lineage_store.record_version(version_label, new_chunks)

        # BM25 index has no cheap "incremental update" in this simple
        # implementation, so we rebuild it from all current chunks --
        # this is still fast because BM25 has no model to train.
        self._rebuild_bm25_index()

        return {
            "added": len(diff.added),
            "modified": len(diff.modified),
            "unchanged": len(diff.unchanged),
            "removed": len(diff.removed_chunk_ids),
        }

    def _index_chunks(self, chunks: List[CodeChunk], is_full_rebuild: bool) -> None:
        if not chunks and not is_full_rebuild:
            self._rebuild_bm25_index()
            return

        for chunk in chunks:
            self.chunks_by_id[chunk.chunk_id] = chunk

        texts = [c.to_indexable_text() for c in chunks]
        if texts:
            if is_full_rebuild:
                self.embedder.fit(texts)  # TF-IDF needs to see the corpus once; ST models don't
            vectors = self.embedder.encode(texts)
            doc_ids = [c.chunk_id for c in chunks]
            if is_full_rebuild:
                self.vector_index.build(doc_ids, vectors)
            else:
                self.vector_index.add_or_update(doc_ids, vectors)

        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self) -> None:
        all_chunks = list(self.chunks_by_id.values())
        doc_ids = [c.chunk_id for c in all_chunks]
        texts = [c.to_indexable_text() for c in all_chunks]
        self.bm25_index.build(doc_ids, texts)

    # ------------------------------------------------------------------
    # ONLINE: answering a query
    # ------------------------------------------------------------------

    def search(self, raw_query: str) -> List[RetrievalResult]:
        processed = preprocess_query(raw_query)

        candidates = self._retrieve_and_rerank(processed.normalized_text, processed.literal_identifiers)

        top_score = candidates[0].final_score if candidates else 0.0
        if top_score < self.low_confidence_threshold:
            candidates = self._run_second_pass(processed, candidates)

        return self._to_results(candidates[: self.final_top_k])

    def _retrieve_and_rerank(self, query_text: str, literal_identifiers: List[str]) -> List[Candidate]:
        bm25_hits = self.bm25_index.search(query_text, top_k=self.initial_top_k)

        query_vector = self.embedder.encode([query_text])[0]
        embedding_hits = self.vector_index.search(query_vector, top_k=self.initial_top_k)

        fused = reciprocal_rank_fusion([bm25_hits, embedding_hits])[: self.fused_top_k]

        bm25_scores = dict(bm25_hits)
        embedding_scores = dict(embedding_hits)

        candidates = []
        for doc_id, fusion_score in fused:
            chunk = self.chunks_by_id.get(doc_id)
            if chunk is None:
                continue
            candidates.append(Candidate(
                doc_id=doc_id,
                text=chunk.to_indexable_text(),
                bm25_score=bm25_scores.get(doc_id, 0.0),
                embedding_score=embedding_scores.get(doc_id, 0.0),
                fusion_score=fusion_score,
            ))

        if not candidates:
            return []

        try:
            return self.reranker.rerank(query_text, candidates, literal_identifiers=literal_identifiers)
        except TypeError:
            # CrossEncoderReranker.rerank() doesn't take literal_identifiers
            return self.reranker.rerank(query_text, candidates)

    def _run_second_pass(self, processed, first_pass_candidates: List[Candidate]) -> List[Candidate]:
        """
        Agentic multi-pass retrieval (strategy doc section 14): the first
        pass was low-confidence, so we pull extra vocabulary out of the
        TOP result of pass one (function names / docstring words the
        original query didn't use) and retrieve again with that richer
        query. We do this AT MOST once, to keep latency bounded.
        """
        if not first_pass_candidates:
            return first_pass_candidates

        top_chunk = self.chunks_by_id.get(first_pass_candidates[0].doc_id)
        if top_chunk is None:
            return first_pass_candidates

        extra_terms = [top_chunk.function_name] + top_chunk.docstring.split()[:8]
        enriched_query = processed.normalized_text + " " + " ".join(extra_terms)

        second_pass_candidates = self._retrieve_and_rerank(enriched_query, processed.literal_identifiers)

        # keep whichever pass produced a more confident #1 result
        first_top = first_pass_candidates[0].final_score
        second_top = second_pass_candidates[0].final_score if second_pass_candidates else -1
        return second_pass_candidates if second_top > first_top else first_pass_candidates

    def _to_results(self, candidates: List[Candidate]) -> List[RetrievalResult]:
        results = []
        for candidate in candidates:
            chunk = self.chunks_by_id.get(candidate.doc_id)
            if chunk is None:
                continue
            results.append(RetrievalResult(
                doc_id=chunk.chunk_id,
                function_name=chunk.function_name,
                file_path=chunk.file_path,
                code=chunk.code,
                final_score=candidate.final_score,
                lineage_versions=self.lineage_store.get_lineage_versions(chunk.chunk_id),
            ))
        return results
