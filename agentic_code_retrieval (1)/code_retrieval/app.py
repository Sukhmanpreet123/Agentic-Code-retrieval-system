"""
app.py
======

A Streamlit UI where YOU upload your own .py file(s), and it builds a
live search index over them -- then you type a question and it shows
the most relevant functions from YOUR code.

This file adds NO new retrieval logic. It just:
  1. Takes uploaded files and hands them to chunking.py (via
     pipeline.build_from_files, which already knows how to parse
     Python files into function-level chunks -- see src/chunking.py).
  2. Calls pipeline.search(query) -- the exact same method used
     everywhere else in this project.
  3. Draws the results on screen instead of printing them to a terminal.

Run it with:
    pip install streamlit
    streamlit run app.py
"""

import hashlib
import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import CodeRetrievalPipeline


def fingerprint_uploads(uploaded_files) -> str:
    """
    A short hash representing "exactly these files with exactly this
    content". We use this to detect when the user has uploaded a
    DIFFERENT set of files, so we know when to rebuild the index --
    and, just as importantly, when NOT to (so we don't rebuild the
    whole index every time Streamlit reruns the script, e.g. every
    keystroke in the search box).
    """
    hasher = hashlib.sha256()
    for f in sorted(uploaded_files, key=lambda f: f.name):
        hasher.update(f.name.encode())
        hasher.update(f.getvalue())
    return hasher.hexdigest()


def build_pipeline_from_uploads(uploaded_files, embedder_backend: str, reranker_backend: str) -> CodeRetrievalPipeline:
    file_path_to_source = {}
    for f in uploaded_files:
        file_path_to_source[f.name] = f.getvalue().decode("utf-8", errors="ignore")

    pipeline = CodeRetrievalPipeline(embedder_backend=embedder_backend, reranker_backend=reranker_backend)
    pipeline.build_from_files(file_path_to_source, version_label="v1")
    return pipeline


st.set_page_config(page_title="Search Your Code", layout="centered")
st.title("Search Your Code")
st.caption("Upload one or more .py files, then ask a question in plain English to find the relevant function(s).")

with st.sidebar:
    st.header("Settings")
    embedder_backend = st.selectbox(
        "Embedder", ["tfidf", "sentence-transformers"],
        help="tfidf = fast, offline, bag-of-words. sentence-transformers = real semantic embeddings (needs internet the first time).",
    )
    reranker_backend = st.selectbox(
        "Reranker", ["feature", "cross-encoder"],
        help="feature = instant, hand-weighted. cross-encoder = real reranking model (needs internet the first time).",
    )

uploaded_files = st.file_uploader(
    "Upload .py file(s)", type=["py"], accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload at least one .py file to get started.")
    st.stop()

# --- Rebuild the index ONLY when the uploaded files actually changed
# (different files, or the same file edited/re-uploaded) or when the
# chosen embedder/reranker changed. Otherwise reuse what's already built,
# so typing in the search box below doesn't re-index everything on every
# keystroke (Streamlit reruns the whole script on any interaction). ---
current_fingerprint = fingerprint_uploads(uploaded_files) + f"::{embedder_backend}::{reranker_backend}"

if st.session_state.get("fingerprint") != current_fingerprint:
    with st.spinner(f"Indexing {len(uploaded_files)} file(s)..."):
        st.session_state.pipeline = build_pipeline_from_uploads(uploaded_files, embedder_backend, reranker_backend)
        st.session_state.fingerprint = current_fingerprint

pipeline = st.session_state.pipeline
num_chunks = len(pipeline.chunks_by_id)

if num_chunks == 0:
    st.warning("No functions were found in the uploaded file(s) -- make sure they contain valid Python function definitions.")
    st.stop()

st.success(f"Indexed {num_chunks} function(s) from {len(uploaded_files)} file(s).")

# --- Search ---
query = st.text_input("Ask a question about this code", placeholder="e.g. how is the input validated?")

if query:
    start = time.time()
    results = pipeline.search(query)
    latency_ms = (time.time() - start) * 1000

    st.caption(f"Found {len(results)} result(s) in {latency_ms:.1f} ms")

    if not results:
        st.warning("No relevant functions found for that query.")

    for rank, result in enumerate(results[:10], start=1):
        with st.expander(f"#{rank}  {result.function_name}()  \u2014  score {result.final_score:.3f}"):
            st.caption(result.file_path)
            st.code(result.code, language="python")