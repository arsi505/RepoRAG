"""Public vector-search result models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchResult:
    rank: int
    chunk_id: str
    score: float
    repository: str
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    content: str
    method: str = "vector"
    score_name: str = "cosine_similarity"
