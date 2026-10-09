"""Frozen deterministic document representation for BM25."""

from __future__ import annotations

from reporag.chunking.models import Chunk


def format_chunk_for_bm25(chunk: Chunk) -> str:
    return (
        f"File: {chunk.file_path}\n"
        f"Language: {chunk.language}\n"
        f"Type: {chunk.chunk_type}\n"
        f"Symbol: {chunk.symbol_name or ''}\n"
        f"Parent: {chunk.parent_symbol or ''}\n"
        "Code:\n"
        f"{chunk.content}"
    )
