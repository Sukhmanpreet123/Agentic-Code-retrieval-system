"""
run_experiments.py
===================

A runnable version of the "Experiment Table" from the strategy doc
(section 19): compares BM25-only vs. embeddings-only vs. hybrid vs.
hybrid+reranked on a small hand-labeled dev set, using the metrics in
metrics.py.

THIS USES THE TOY REPO in tests/sample_repo -- swap `load_repo(...)` and
`LABELED_QUERIES` for your real dev set (a handful of queries with the
correct chunk_id(s) marked as relevant) once you're working with the
actual CoIR Apps corpus or your own codebase.

Run it with:
    python eval/run_experiments.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import CodeRetrievalPipeline
from chunking import chunk_repository
from metrics import evaluate_system


def load_repo(repo_dir: str) -> dict:
    file_path_to_source = {}
    for filename in sorted(os.listdir(repo_dir)):
        if filename.endswith(".py"):
            with open(os.path.join(repo_dir, filename)) as f:
                file_path_to_source[filename] = f.read()
    return file_path_to_source


def find_chunk_id(pipeline: CodeRetrievalPipeline, function_name: str) -> str:
    """Helper: look up a chunk_id by function name, for building ground truth."""
    for chunk_id, chunk in pipeline.chunks_by_id.items():
        if chunk.function_name == function_name:
            return chunk_id
    raise ValueError(f"No chunk found for function {function_name}")


def run_variant(name: str, embedder_backend: str, reranker_backend: str, use_bm25_only: bool,
                 repo_dir: str, labeled_queries):
    pipeline = CodeRetrievalPipeline(embedder_backend=embedder_backend, reranker_backend=reranker_backend)
    pipeline.build_from_files(load_repo(repo_dir), version_label="v1")

    query_results, query_relevant_sets = [], []
    for query_text, relevant_function_names in labeled_queries:
        relevant_ids = {find_chunk_id(pipeline, fn) for fn in relevant_function_names}

        if use_bm25_only:
            hits = pipeline.bm25_index.search(query_text, top_k=10)
            ranked_ids = [doc_id for doc_id, _ in hits]
        else:
            results = pipeline.search(query_text)
            ranked_ids = [r.doc_id for r in results]

        query_results.append(ranked_ids)
        query_relevant_sets.append(relevant_ids)

    scores = evaluate_system(query_results, query_relevant_sets, k=10)
    print(f"{name:30s}  MRR={scores['MRR']:.3f}  NDCG@10={scores['NDCG@10']:.3f}")


def main():
    repo_dir = os.path.join(os.path.dirname(__file__), "..", "tests", "sample_repo", "v1")

    # Hand-labeled ground truth: (query, [correct function name(s)])
    # In a real project this comes from the CoIR Apps qrels file, not
    # hand-written -- this toy set exists purely to demonstrate the
    # ablation methodology end-to-end.
    labeled_queries = [
        ("how is the input preprocessed before going to the main function", ["normalize", "forward"]),
        ("check the locale of a string", ["check_locale"]),
        ("authenticate a user", ["authenticate"]),
        ("open a connection to the database", ["connect_to_database"]),
    ]

    print("Ablation results on the toy dev set (see strategy doc section 19):\n")
    run_variant("BM25 only (baseline)", "tfidf", "feature", use_bm25_only=True,
                repo_dir=repo_dir, labeled_queries=labeled_queries)
    run_variant("Hybrid + feature rerank", "tfidf", "feature", use_bm25_only=False,
                repo_dir=repo_dir, labeled_queries=labeled_queries)
    print("\n(Swap embedder_backend='sentence-transformers', reranker_backend='cross-encoder'")
    print(" once those packages are installed, to see the real semantic-model numbers.)")


if __name__ == "__main__":
    main()
