"""Parse and validate generated RepoRAG source markers."""

from __future__ import annotations

import re
from typing import Sequence

from .models import CitationValidation, Evidence


_SOURCE_MARKER = re.compile(r"\[S([1-9][0-9]*)\]")


def parse_source_ids(text: str) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for number in _SOURCE_MARKER.findall(text):
        source_id = f"S{number}"
        if source_id not in seen:
            ordered.append(source_id)
            seen.add(source_id)
    return tuple(ordered)


def validate_citations(text: str, evidence: Sequence[Evidence]) -> CitationValidation:
    cited = parse_source_ids(text)
    available = {item.source_id for item in evidence}
    unknown = tuple(source_id for source_id in cited if source_id not in available)
    return CitationValidation(
        cited_source_ids=cited,
        unknown_source_ids=unknown,
        citations_valid=bool(cited) and not unknown,
    )


def cited_evidence(
    validation: CitationValidation, evidence: Sequence[Evidence]
) -> tuple[Evidence, ...]:
    by_id = {item.source_id: item for item in evidence}
    return tuple(
        by_id[source_id]
        for source_id in validation.cited_source_ids
        if source_id in by_id
    )
