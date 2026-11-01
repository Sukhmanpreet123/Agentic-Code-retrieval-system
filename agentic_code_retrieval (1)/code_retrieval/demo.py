"""
demo.py
=======

Runs the WHOLE system end-to-end on a small toy codebase, entirely
offline (uses the TF-IDF embedder + feature-based reranker, so it needs
NO internet connection and NO model download -- good for quickly proving
the pipeline works before you swap in the real semantic model).

Run it with:
    python demo.py

What it demonstrates, matching the strategy doc's demo plan (section 23):
    1. Build the index from "version 1" of a toy codebase.
    2. Run a natural-language query and show ranked results + latency.
    3. Update to "version 2" (one function changed, one added) and show
       that only the changed/added chunks get re-indexed (P1).
    4. Run a query whose top result exists in both versions, and show the
       lineage information (Bonus).

To use the REAL semantic embedder + real cross-encoder reranker instead
of the offline versions, change the two lines marked "SWITCH HERE".
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import CodeRetrievalPipeline


def load_repo(repo_dir: str) -> dict:
    """
    IMPORTANT: keys are the file's path RELATIVE to the repo root (e.g.
    "auth.py"), not the absolute path including the "v1"/"v2" folder.
    chunk_id is a hash of (file_path + function_name) -- see chunking.py --
    so if we used the absolute, version-specific path, EVERY chunk would
    look "new" in every version, and versioning.py could never detect
    that a function is unchanged. Using the repo-relative path is what
    lets the same function be recognized as "the same function slot"
    across versions.
    """
    file_path_to_source = {}
    for filename in sorted(os.listdir(repo_dir)):
        if filename.endswith(".py"):
            full_path = os.path.join(repo_dir, filename)
            with open(full_path) as f:
                file_path_to_source[filename] = f.read()   # relative path as the key
    return file_path_to_source


def run_query(pipeline: CodeRetrievalPipeline, query: str) -> None:
    start = time.time()
    results = pipeline.search(query)
    latency_ms = (time.time() - start) * 1000

    print(f"\nQuery: {query!r}   (latency: {latency_ms:.1f} ms)")
    for rank, result in enumerate(results[:5], start=1):
        lineage = ""
        if result.lineage_versions and len(result.lineage_versions) > 1:
            lineage = f"  [seen in versions: {', '.join(result.lineage_versions)}]"
        print(f"  #{rank}  score={result.final_score:.3f}  {result.function_name}()"
              f"  ({os.path.basename(result.file_path)}){lineage}")


def main():
    here = os.path.dirname(__file__)
    v1_dir = os.path.join(here, "tests", "sample_repo", "v1")
    v2_dir = os.path.join(here, "tests", "sample_repo", "v2")

    # SWITCH HERE for the real submission:
    #   CodeRetrievalPipeline(embedder_backend="sentence-transformers", reranker_backend="cross-encoder")
    pipeline = CodeRetrievalPipeline(embedder_backend="tfidf", reranker_backend="feature")

    print("=== Building index from version 1 ===")
    pipeline.build_from_files(load_repo(v1_dir), version_label="v1")
    print(f"Indexed {len(pipeline.chunks_by_id)} chunks.")

    run_query(pipeline, "how is the input preprocessed before going to the main function")
    run_query(pipeline, "check the locale of a string")
    run_query(pipeline, "authenticate a user")

    print("\n=== Updating to version 2 (one function changed, one added) ===")
    start = time.time()
    report = pipeline.update_to_new_version(load_repo(v2_dir), version_label="v2")
    rebuild_ms = (time.time() - start) * 1000
    print(f"Rebuild report: {report}   (rebuild time: {rebuild_ms:.1f} ms)")

    run_query(pipeline, "authenticate a user")   # should show lineage across v1 and v2
    run_query(pipeline, "revoke an auth token")   # only exists in v2


if __name__ == "__main__":
    main()
