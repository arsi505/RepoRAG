"""Orchestration for ingestion-backed structural and fallback chunking."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from reporag.ingestion.models import FileMetadata, IngestionManifest

from .extractors import extract_structures
from .fallback import (
    DEFAULT_FALLBACK_MAX_LINES,
    DEFAULT_FALLBACK_OVERLAP_LINES,
    DEFAULT_MAX_STRUCTURAL_LINES,
    split_lines,
)
from .languages import structural_language_for
from .models import Chunk, ChunkingResult, StructuralUnit
from .parser import ParserFailure, TreeSitterParser


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    fallback_max_lines: int = DEFAULT_FALLBACK_MAX_LINES
    fallback_overlap_lines: int = DEFAULT_FALLBACK_OVERLAP_LINES
    max_structural_lines: int = DEFAULT_MAX_STRUCTURAL_LINES


class ChunkingError(RuntimeError):
    """Raised when an accepted source file cannot be chunked safely."""


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _chunk_id(
    *,
    repository: str,
    commit_hash: str | None,
    file_path: str,
    chunk_type: str,
    symbol_name: str | None,
    parent_symbol: str | None,
    start_line: int,
    end_line: int,
    part_index: int | None,
    part_count: int,
    content_hash: str,
) -> str:
    stable_inputs = [
        repository,
        commit_hash,
        file_path,
        chunk_type,
        symbol_name,
        parent_symbol,
        start_line,
        end_line,
        part_index,
        part_count,
        content_hash,
    ]
    encoded = json.dumps(stable_inputs, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _make_chunk(
    manifest: IngestionManifest,
    file_metadata: FileMetadata,
    *,
    content: str,
    chunk_type: str,
    symbol_name: str | None,
    parent_symbol: str | None,
    start_line: int,
    end_line: int,
    part_index: int | None,
    part_count: int,
    parser_status: str,
) -> Chunk:
    content_hash = _content_hash(content)
    fields = {
        "repository": manifest.repository.name,
        "commit_hash": manifest.repository.commit_hash,
        "file_path": file_metadata.relative_path,
        "chunk_type": chunk_type,
        "symbol_name": symbol_name,
        "parent_symbol": parent_symbol,
        "start_line": start_line,
        "end_line": end_line,
        "part_index": part_index,
        "part_count": part_count,
        "content_hash": content_hash,
    }
    return Chunk(
        chunk_id=_chunk_id(**fields),
        language=file_metadata.language or "Unknown",
        content=content,
        parser_status=parser_status,
        **fields,
    )


def _fallback_chunks(
    manifest: IngestionManifest,
    file_metadata: FileMetadata,
    content: bytes,
    *,
    start_line: int,
    config: ChunkingConfig,
    parser_status: str,
    overlap_lines: int | None = None,
) -> list[Chunk]:
    overlap = config.fallback_overlap_lines if overlap_lines is None else overlap_lines
    return [
        _make_chunk(
            manifest,
            file_metadata,
            content=part.content,
            chunk_type="file",
            symbol_name=None,
            parent_symbol=None,
            start_line=part.start_line,
            end_line=part.end_line,
            part_index=part.part_index,
            part_count=part.part_count,
            parser_status=parser_status,
        )
        for part in split_lines(
            content,
            start_line=start_line,
            max_lines=config.fallback_max_lines,
            overlap_lines=overlap,
            expose_parts=True,
        )
    ]


def _structural_chunks(
    manifest: IngestionManifest,
    file_metadata: FileMetadata,
    source: bytes,
    unit: StructuralUnit,
    *,
    config: ChunkingConfig,
    parser_status: str,
) -> list[Chunk]:
    content = source[unit.start_byte : unit.end_byte]
    return [
        _make_chunk(
            manifest,
            file_metadata,
            content=part.content,
            chunk_type=unit.chunk_type,
            symbol_name=unit.symbol_name,
            parent_symbol=unit.parent_symbol,
            start_line=part.start_line,
            end_line=part.end_line,
            part_index=part.part_index,
            part_count=part.part_count,
            parser_status=parser_status,
        )
        for part in split_lines(
            content,
            start_line=unit.start_line,
            max_lines=config.max_structural_lines,
            overlap_lines=0,
            expose_parts=True,
        )
    ]


def _file_context_chunks(
    manifest: IngestionManifest,
    file_metadata: FileMetadata,
    source: bytes,
    units: tuple[StructuralUnit, ...],
    *,
    config: ChunkingConfig,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    cursor = 0
    for unit in units:
        if unit.start_byte > cursor:
            gap = source[cursor : unit.start_byte]
            if gap.strip():
                start_line = source.count(b"\n", 0, cursor) + 1
                chunks.extend(
                    _fallback_chunks(
                        manifest,
                        file_metadata,
                        gap,
                        start_line=start_line,
                        config=config,
                        parser_status="file_context",
                        overlap_lines=0,
                    )
                )
        cursor = max(cursor, unit.end_byte)
    if cursor < len(source):
        gap = source[cursor:]
        if gap.strip():
            start_line = source.count(b"\n", 0, cursor) + 1
            chunks.extend(
                _fallback_chunks(
                    manifest,
                    file_metadata,
                    gap,
                    start_line=start_line,
                    config=config,
                    parser_status="file_context",
                    overlap_lines=0,
                )
            )
    return chunks


def _chunk_file(
    manifest: IngestionManifest,
    file_metadata: FileMetadata,
    source: bytes,
    *,
    config: ChunkingConfig,
    parser: TreeSitterParser,
) -> list[Chunk]:
    language = structural_language_for(file_metadata)
    if language is None:
        return _fallback_chunks(
            manifest,
            file_metadata,
            source,
            start_line=1,
            config=config,
            parser_status="fallback_unsupported_language",
        )

    try:
        outcome = parser.parse(language.grammar, source)
    except ParserFailure:
        return _fallback_chunks(
            manifest,
            file_metadata,
            source,
            start_line=1,
            config=config,
            parser_status="fallback_parser_failure",
        )

    units = extract_structures(language.key, outcome.nodes, source)
    if not units:
        return _fallback_chunks(
            manifest,
            file_metadata,
            source,
            start_line=1,
            config=config,
            parser_status=(
                "fallback_syntax_errors" if outcome.has_errors else "fallback_no_structures"
            ),
        )

    parser_status = "parsed_with_errors" if outcome.has_errors else "parsed"
    chunks: list[Chunk] = []
    for unit in units:
        chunks.extend(
            _structural_chunks(
                manifest,
                file_metadata,
                source,
                unit,
                config=config,
                parser_status=parser_status,
            )
        )
    chunks.extend(
        _file_context_chunks(
            manifest,
            file_metadata,
            source,
            units,
            config=config,
        )
    )
    return chunks


def chunk_repository(
    manifest: IngestionManifest,
    *,
    config: ChunkingConfig = ChunkingConfig(),
    parser: TreeSitterParser | None = None,
) -> ChunkingResult:
    """Chunk only files accepted by the existing ingestion manifest."""

    repository_root = Path(manifest.repository.local_path)
    parser = parser or TreeSitterParser()
    chunks: list[Chunk] = []

    for file_metadata in manifest.files:
        source_path = repository_root / Path(file_metadata.relative_path)
        try:
            source = source_path.read_bytes()
            source.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ChunkingError(
                f"Accepted file became unreadable after ingestion: {file_metadata.relative_path}"
            ) from exc
        chunks.extend(
            _chunk_file(
                manifest,
                file_metadata,
                source,
                config=config,
                parser=parser,
            )
        )

    chunks.sort(
        key=lambda chunk: (
            chunk.file_path.casefold(),
            chunk.file_path,
            chunk.start_line,
            chunk.end_line,
            chunk.chunk_id,
        )
    )
    structural_count = sum(chunk.chunk_type != "file" for chunk in chunks)
    return ChunkingResult(
        chunks=tuple(chunks),
        structural_chunk_count=structural_count,
        fallback_chunk_count=len(chunks) - structural_count,
    )
