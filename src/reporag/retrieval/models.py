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


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    rank: int
    chunk_id: str
    rrf_score: float
    vector_rank: int | None
    vector_raw_score: float | None
    vector_contribution: float
    bm25_rank: int | None
    bm25_raw_score: float | None
    bm25_contribution: float
    repository: str
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    content: str
    retrieval_method: str = "hybrid_rrf"
    score_type: str = "rrf"

    @property
    def score(self) -> float:
        return self.rrf_score

    @property
    def method(self) -> str:
        return self.retrieval_method

    @property
    def score_name(self) -> str:
        return self.score_type
