"""Frozen semantic text representation for Day 3 chunks."""

from __future__ import annotations

from reporag.chunking.models import Chunk


def format_chunk_for_embedding(chunk: Chunk) -> str:
    """Return the version-1 metadata-plus-source representation.

    Missing optional symbols are represented by an empty value. The source
    content is appended byte-for-byte as decoded by the Day 3 chunker.
    """

    return (
        f"Repository: {chunk.repository}\n"
        f"File: {chunk.file_path}\n"
        f"Language: {chunk.language}\n"
        f"Type: {chunk.chunk_type}\n"
        f"Symbol: {chunk.symbol_name or ''}\n"
        f"Parent: {chunk.parent_symbol or ''}\n"
        "Code:\n"
        f"{chunk.content}"
    )
