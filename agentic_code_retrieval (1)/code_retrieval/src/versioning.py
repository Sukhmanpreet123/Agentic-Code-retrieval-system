"""
versioning.py
=============

WHAT THIS FILE DOES (P1 requirement: "retrieval across versions"):
    Codebases change every commit. Re-embedding EVERY function from
    scratch every time a single line changes would be slow and wasteful.
    This file figures out exactly WHICH chunks are new/changed/deleted
    between two versions, so the pipeline only re-indexes what actually
    changed.

HOW WE DETECT CHANGES:
    Every CodeChunk already has:
      - chunk_id       = hash(file_path + class_name + function_name)
                         -> identifies "the same function slot" across versions
      - content_hash    = hash(the function's code)
                         -> identifies whether the CODE ITSELF changed

    Comparing chunk_id sets between two versions tells us ADDED / REMOVED.
    Comparing content_hash for chunk_ids present in BOTH versions tells us
    MODIFIED vs UNCHANGED.

THIS ALSO POWERS THE BONUS (evolutionary retrieval):
    Because every version of "the same function slot" shares the same
    chunk_id, we can group them into a "lineage" -- e.g. `authenticate()`
    across v1/v2/v3 -- to avoid showing near-duplicate results as if they
    were unrelated (see LineageStore below).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set

from chunking import CodeChunk


@dataclass
class VersionDiff:
    added: List[CodeChunk] = field(default_factory=list)
    modified: List[CodeChunk] = field(default_factory=list)
    unchanged: List[CodeChunk] = field(default_factory=list)
    removed_chunk_ids: List[str] = field(default_factory=list)

    @property
    def chunks_needing_reindex(self) -> List[CodeChunk]:
        """These are the ONLY chunks that need a fresh embedding computed."""
        return self.added + self.modified


def diff_versions(
    previous_chunks: List[CodeChunk],
    current_chunks: List[CodeChunk],
) -> VersionDiff:
    """
    Compares chunk lists from two versions of the codebase and classifies
    every chunk in the CURRENT version as added / modified / unchanged,
    and finds chunk_ids that existed before but don't anymore (removed).
    """
    previous_by_id: Dict[str, CodeChunk] = {c.chunk_id: c for c in previous_chunks}
    current_ids: Set[str] = {c.chunk_id for c in current_chunks}

    diff = VersionDiff()
    for chunk in current_chunks:
        previous_version = previous_by_id.get(chunk.chunk_id)
        if previous_version is None:
            diff.added.append(chunk)
        elif previous_version.content_hash != chunk.content_hash:
            diff.modified.append(chunk)
        else:
            diff.unchanged.append(chunk)

    diff.removed_chunk_ids = [
        chunk_id for chunk_id in previous_by_id if chunk_id not in current_ids
    ]
    return diff


class LineageStore:
    """
    Tracks, for every chunk_id ("function slot"), which versions it has
    appeared in. This is what makes "Evolutionary Retrieval" (Bonus)
    possible: at query time we can say "these 3 results are actually the
    same function evolving across versions 1, 2, and 3" instead of
    showing 3 confusing near-duplicates.
    """

    def __init__(self):
        # chunk_id -> list of version labels it has appeared in, in order
        self.chunk_id_to_versions: Dict[str, List[str]] = {}
        # chunk_id -> the latest content_hash seen, per version
        self.chunk_id_version_hash: Dict[str, Dict[str, str]] = {}

    def record_version(self, version_label: str, chunks: List[CodeChunk]) -> None:
        for chunk in chunks:
            self.chunk_id_to_versions.setdefault(chunk.chunk_id, [])
            if version_label not in self.chunk_id_to_versions[chunk.chunk_id]:
                self.chunk_id_to_versions[chunk.chunk_id].append(version_label)
            self.chunk_id_version_hash.setdefault(chunk.chunk_id, {})
            self.chunk_id_version_hash[chunk.chunk_id][version_label] = chunk.content_hash

    def get_lineage_versions(self, chunk_id: str) -> List[str]:
        """All versions a given function slot has existed in."""
        return self.chunk_id_to_versions.get(chunk_id, [])

    def did_change_between(self, chunk_id: str, version_a: str, version_b: str) -> bool:
        hashes = self.chunk_id_version_hash.get(chunk_id, {})
        return hashes.get(version_a) != hashes.get(version_b)
