"""Public and internal models for code chunking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    repository: str
    commit_hash: str | None
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    parent_symbol: str | None
    start_line: int
    end_line: int
    part_index: int | None
    part_count: int
    content_hash: str
    content: str
    parser_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StructuralUnit:
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: str | None
    parent_symbol: str | None


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    chunks: tuple[Chunk, ...]
    structural_chunk_count: int
    fallback_chunk_count: int
