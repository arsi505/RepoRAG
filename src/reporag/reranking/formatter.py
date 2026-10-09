"""Deterministic semantic text for Method D candidates."""

from __future__ import annotations

from reporag.retrieval.models import HybridSearchResult

from .config import FORMATTER_VERSION


def format_candidate(candidate: HybridSearchResult) -> str:
    """Format stable repository metadata and exact source, without retrieval scores."""

    return "\n".join(
        (
            f"File: {candidate.file_path}",
            f"Language: {candidate.language}",
            f"Type: {candidate.chunk_type}",
            f"Symbol: {candidate.symbol_name or ''}",
            f"Parent: {candidate.parent_symbol or ''}",
            "Code:",
            candidate.content,
        )
    )


__all__ = ["FORMATTER_VERSION", "format_candidate"]
