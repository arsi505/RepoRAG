"""Public repository Q&A evidence and answer models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Evidence:
    source_id: str
    chunk_id: str
    retrieval_rank: int
    repository: str
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    content: str


@dataclass(frozen=True, slots=True)
class CitationValidation:
    cited_source_ids: tuple[str, ...]
    unknown_source_ids: tuple[str, ...]
    citations_valid: bool


@dataclass(frozen=True, slots=True)
class QATimings:
    retrieval_seconds: float
    context_assembly_seconds: float
    generation_seconds: float | None
    citation_validation_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class QAResult:
    question: str
    answer_text: str
    retrieval_method: str
    context_k: int
    evidence_scope: str
    cited_source_ids: tuple[str, ...]
    unknown_source_ids: tuple[str, ...]
    citations_valid: bool
    evidence: tuple[Evidence, ...]
    context_text: str
    qa_fingerprint: str
    provider: str | None = None
    model: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    generation_latency_seconds: float | None = None
    dry_run: bool = False
    timings: QATimings | None = None
