"""Deterministic source-ID assignment and evidence formatting."""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Sequence

from .models import Evidence


def filter_by_evidence_scope(
    results: Sequence[Any], evidence_scope: str
) -> tuple[Any, ...]:
    """Filter ranked results without changing their original order."""

    if evidence_scope == "all":
        return tuple(results)
    if evidence_scope not in {"code", "docs"}:
        raise ValueError(f"Unsupported evidence scope: {evidence_scope}")

    filtered: list[Any] = []
    for result in results:
        path = PurePosixPath(result.file_path)
        is_documentation = (
            str(result.language).casefold() == "markdown"
            or path.suffix.casefold() == ".md"
        )
        if (evidence_scope == "docs") == is_documentation:
            filtered.append(result)
    return tuple(filtered)


def build_evidence(results: Sequence[Any], *, context_k: int) -> tuple[Evidence, ...]:
    if context_k <= 0:
        raise ValueError("context_k must be positive")
    evidence: list[Evidence] = []
    seen: set[str] = set()
    for result in results:
        if result.chunk_id in seen:
            continue
        if PurePosixPath(result.file_path).is_absolute() or PureWindowsPath(
            result.file_path
        ).is_absolute():
            raise ValueError("Repository evidence file paths must be relative")
        seen.add(result.chunk_id)
        evidence.append(
            Evidence(
                source_id=f"S{len(evidence) + 1}",
                chunk_id=result.chunk_id,
                retrieval_rank=result.rank,
                repository=result.repository,
                file_path=result.file_path,
                language=result.language,
                chunk_type=result.chunk_type,
                symbol_name=result.symbol_name,
                parent_symbol=result.parent_symbol,
                start_line=result.start_line,
                end_line=result.end_line,
                content=result.content,
            )
        )
        if len(evidence) == context_k:
            break
    return tuple(evidence)


def format_evidence(item: Evidence) -> str:
    return "\n".join(
        (
            f"[{item.source_id}]",
            f"File: {item.file_path}",
            f"Language: {item.language}",
            f"Type: {item.chunk_type}",
            f"Symbol: {item.symbol_name or ''}",
            f"Parent: {item.parent_symbol or ''}",
            f"Lines: {item.start_line}-{item.end_line}",
            f"Retrieval rank: {item.retrieval_rank}",
            "Code:",
            item.content,
        )
    )


def format_context(evidence: Sequence[Evidence]) -> str:
    return "\n\n".join(format_evidence(item) for item in evidence)
