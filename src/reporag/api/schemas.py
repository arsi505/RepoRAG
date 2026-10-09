"""Validated HTTP request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reporag.qa.config import DEFAULT_QA_CONFIG


RetrievalMethod = Literal["vector", "bm25", "hybrid", "reranked"]
EvidenceScope = Literal["all", "code", "docs"]
GenerationProvider = Literal["deepseek", "gemini", "openai"]


class QARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository: str = Field(min_length=1)
    question: str = Field(min_length=1)
    retrieval_method: RetrievalMethod = DEFAULT_QA_CONFIG.default_retrieval_method
    evidence_scope: EvidenceScope = DEFAULT_QA_CONFIG.evidence_scope
    provider: GenerationProvider = DEFAULT_QA_CONFIG.generation_provider
    context_k: int = Field(default=DEFAULT_QA_CONFIG.default_context_k, gt=0)
    max_output_tokens: int = Field(
        default=DEFAULT_QA_CONFIG.max_output_tokens,
        gt=0,
    )


class SourceResponse(BaseModel):
    id: str
    file_path: str
    symbol: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    retrieval_rank: int
    content: str


class QAResponse(BaseModel):
    question: str
    repository: str
    retrieval_method: RetrievalMethod
    evidence_scope: EvidenceScope
    provider: GenerationProvider
    model: str
    answer: str
    citations_valid: bool
    cited_source_ids: list[str]
    generation_latency_seconds: float | None
    usage: dict[str, Any] | None
    sources: list[SourceResponse]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: Literal["RepoRAG"]


class ErrorResponse(BaseModel):
    detail: str
    code: str
