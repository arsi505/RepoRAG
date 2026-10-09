"""Inspectable Method D result and diagnostic models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RerankedResult:
    rank: int
    chunk_id: str
    reranker_score: float
    hybrid_rank: int
    hybrid_rrf_score: float
    vector_rank: int | None
    vector_raw_score: float | None
    bm25_rank: int | None
    bm25_raw_score: float | None
    repository: str
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    content: str
    retrieval_method: str = "hybrid_reranked"
    score_type: str = "cross_encoder_raw_logit"


@dataclass(frozen=True, slots=True)
class RerankingTimings:
    candidate_generation_seconds: float
    formatting_tokenization_seconds: float
    scoring_seconds: float
    sorting_finalization_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class RerankingDiagnostics:
    candidate_count: int
    candidates_over_configured_limit: int
    candidates_truncated: int
    pair_token_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RerankingOutcome:
    results: tuple[RerankedResult, ...]
    diagnostics: RerankingDiagnostics
    timings: RerankingTimings
