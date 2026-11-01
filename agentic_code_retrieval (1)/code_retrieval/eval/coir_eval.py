"""
coir_eval.py
============

Evaluates our pipeline against a REAL subset of the CoIR Apps benchmark
(Hugging Face: CoIR-Retrieval/apps), instead of hand-made toy examples.
This is what you show/explain if an interviewer asks "how did you
evaluate this?"

WHY DOCUMENTS ARE NOT CHUNKED HERE:
    src/chunking.py splits a Python FILE into per-FUNCTION chunks -- that
    makes sense for a real codebase. The CoIR Apps corpus is different:
    each corpus entry is already a single, self-contained competitive-
    programming SOLUTION (a whole small program), which is exactly the
    retrievable unit here. So we skip chunking.py entirely and wrap each
    corpus row directly as one CodeChunk.

Install:
    pip install datasets

Run:
    python eval/coir_eval.py --num-queries 50 --max-corpus 2000
    (start small -- see the --num-queries / --max-corpus flags below)
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from chunking import CodeChunk
from metrics import evaluate_system
from pipeline import CodeRetrievalPipeline


def _get(row, *keys, default=None):
    """Different HF dataset versions name columns slightly differently
    (e.g. '_id' vs 'id', 'query-id' vs 'query_id'). Try each in turn."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _load_config_robust(dataset_name: str, config: str, preferred_split: str = "test"):
    """
    Different CoIR/BEIR-style datasets on HF name their splits inconsistently
    -- sometimes "test", sometimes the config's own name (e.g. the "corpus"
    config's only split is literally called "corpus", not "test"). Instead
    of guessing, we ask for `preferred_split` first, and if that split
    doesn't exist, fall back to whatever split IS there.
    """
    from datasets import load_dataset

    try:
        return load_dataset(dataset_name, config, split=preferred_split)
    except ValueError:
        dataset_dict = load_dataset(dataset_name, config)
        if preferred_split in dataset_dict:
            return dataset_dict[preferred_split]
        # No "test" split -- just take whichever single split actually exists.
        only_split_name = list(dataset_dict.keys())[0]
        print(f"  (config '{config}' has no '{preferred_split}' split -- using '{only_split_name}' instead)")
        return dataset_dict[only_split_name]


def load_coir_apps(split: str = "test"):
    corpus = _load_config_robust("CoIR-Retrieval/apps", "corpus", preferred_split=split)
    queries = _load_config_robust("CoIR-Retrieval/apps", "queries", preferred_split=split)
    try:
        qrels = _load_config_robust("CoIR-Retrieval/apps", "default", preferred_split=split)
    except Exception:
        qrels = _load_config_robust("CoIR-Retrieval/apps", "qrels", preferred_split=split)
    return corpus, queries, qrels


def build_pipeline_from_rows(rows, embedder_backend: str, reranker_backend: str) -> CodeRetrievalPipeline:
    """Builds the index from an already-chosen list of corpus rows (no
    truncation logic here -- the caller decides exactly which rows to include,
    see the "gold docs + distractors" selection in main())."""
    pipeline = CodeRetrievalPipeline(embedder_backend=embedder_backend, reranker_backend=reranker_backend)

    chunks = []
    for row in rows:
        doc_id = str(_get(row, "_id", "id"))
        text = _get(row, "text", default="")
        title = _get(row, "title", default="") or ""
        chunks.append(CodeChunk(
            chunk_id=doc_id,
            function_name=title or f"doc_{doc_id}",
            file_path="coir_apps_corpus",
            class_name=None,
            imports=[],
            called_functions=[],
            docstring=title,
            code=text,
        ))

    # Reuse the pipeline's own internal indexing step directly, since we
    # already have CodeChunk objects and don't need chunking.py here.
    pipeline._index_chunks(chunks, is_full_rebuild=True)
    return pipeline


def build_relevant_map(qrels) -> dict:
    relevant_map = {}
    for row in qrels:
        qid = str(_get(row, "query-id", "query_id"))
        cid = str(_get(row, "corpus-id", "corpus_id"))
        relevant_map.setdefault(qid, set()).add(cid)
    return relevant_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=50,
                         help="How many queries to sample for evaluation (start small -- reranking every query costs time).")
    parser.add_argument("--max-corpus", type=int, default=2000,
                         help="Total corpus size to build the index over (gold answer docs + random distractor docs). The full test corpus is ~8,765 docs.")
    parser.add_argument("--embedder", choices=["tfidf", "sentence-transformers"], default="tfidf")
    parser.add_argument("--reranker", choices=["feature", "cross-encoder"], default="feature")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    print("Loading CoIR Apps dataset...")
    corpus, queries, qrels = load_coir_apps()
    relevant_map = build_relevant_map(qrels)
    print(f"Loaded {len(corpus)} corpus docs, {len(queries)} queries, {len(relevant_map)} queries with qrels.")

    # --- DIAGNOSTIC: check that qrels' corpus-ids actually match the
    # corpus config's own ids at all, BEFORE spending time building the
    # index. If this overlap is 0, nothing downstream can ever score
    # above 0 -- it means the two files use different ID formats. ---
    all_corpus_ids = {str(_get(row, "_id", "id")) for row in corpus}
    all_qrel_corpus_ids = set()
    for doc_ids in relevant_map.values():
        all_qrel_corpus_ids |= doc_ids
    overlap = all_corpus_ids & all_qrel_corpus_ids
    print(f"Sanity check: {len(overlap)} of {len(all_qrel_corpus_ids)} qrel corpus-ids match an actual corpus id.")
    if len(overlap) == 0:
        print("\n!! ID MISMATCH DETECTED !!")
        print(f"  Example corpus id:      {next(iter(all_corpus_ids))!r}")
        print(f"  Example qrels corpus-id: {next(iter(all_qrel_corpus_ids))!r}")
        return
    # --- end diagnostic ---

    # STEP 1: pick the queries we'll actually test FIRST (only from ones
    # that have a qrel at all).
    eligible_query_rows = [row for row in queries if str(_get(row, "_id", "id")) in relevant_map]
    random.shuffle(eligible_query_rows)
    sampled_rows = eligible_query_rows[: args.num_queries]
    print(f"Sampled {len(sampled_rows)} queries to evaluate (out of {len(eligible_query_rows)} with qrels).")

    # STEP 2: figure out exactly which corpus docs are the "gold" correct
    # answers for THOSE specific queries -- these MUST be in the index,
    # or the run is unwinnable no matter how good the retriever is
    # (this is exactly what went wrong when we truncated to "the first N
    # rows" earlier -- the corpus file turned out to be ordered such that
    # the first N rows were all training-split docs, containing none of
    # the test queries' correct answers at all).
    gold_doc_ids = set()
    for row in sampled_rows:
        gold_doc_ids |= relevant_map[str(_get(row, "_id", "id"))]
    print(f"These queries need {len(gold_doc_ids)} specific 'gold' documents to be present in the corpus.")

    if args.max_corpus < len(gold_doc_ids):
        args.max_corpus = len(gold_doc_ids) + 50
        print(f"--max-corpus was smaller than the number of needed gold docs -- raised it to {args.max_corpus}.")

    # STEP 3: build the corpus we'll index = ALL gold docs (guaranteed)
    # + random distractor docs to fill up to --max-corpus. This gives a
    # smaller, fast-to-run corpus that is still a FAIR test, because the
    # correct answers are always in there somewhere -- the system still
    # has to actually find them among the distractors.
    all_rows = list(corpus)
    gold_rows = [row for row in all_rows if str(_get(row, "_id", "id")) in gold_doc_ids]
    remaining_rows = [row for row in all_rows if str(_get(row, "_id", "id")) not in gold_doc_ids]
    random.shuffle(remaining_rows)
    distractor_budget = max(args.max_corpus - len(gold_rows), 0)
    corpus_subset_rows = gold_rows + remaining_rows[:distractor_budget]
    random.shuffle(corpus_subset_rows)

    print(f"Building index with embedder={args.embedder}, reranker={args.reranker} "
          f"over {len(corpus_subset_rows)} docs "
          f"({len(gold_rows)} gold answers + {len(corpus_subset_rows) - len(gold_rows)} random distractors)...")
    pipeline = build_pipeline_from_rows(corpus_subset_rows, args.embedder, args.reranker)

    query_results, query_relevant_sets = [], []
    for row in sampled_rows:
        qid = str(_get(row, "_id", "id"))
        query_text = _get(row, "text", default="")
        results = pipeline.search(query_text)
        query_results.append([r.doc_id for r in results])
        query_relevant_sets.append(relevant_map[qid])

    scores = evaluate_system(query_results, query_relevant_sets, k=10)
    print("\n=== Results on real CoIR Apps data ===")
    print(f"MRR:      {scores['MRR']:.3f}")
    print(f"NDCG@10:  {scores['NDCG@10']:.3f}")
    print(f"(evaluated on {scores['num_queries']} queries, corpus size {len(corpus_subset_rows)})")


if __name__ == "__main__":
    main()
