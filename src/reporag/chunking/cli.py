"""Command-line interface for Day 3 code-aware chunking."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from reporag.ingestion.repository import IngestionError, ingest_repository

from .chunker import ChunkingConfig, ChunkingError, chunk_repository
from .fallback import (
    DEFAULT_FALLBACK_MAX_LINES,
    DEFAULT_FALLBACK_OVERLAP_LINES,
    DEFAULT_MAX_STRUCTURAL_LINES,
)
from .writer import write_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest a repository and write deterministic code chunks as JSONL."
    )
    parser.add_argument("source", help="Local directory or public GitHub repository URL")
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/repository_cache"))
    parser.add_argument("--fallback-max-lines", type=int, default=DEFAULT_FALLBACK_MAX_LINES)
    parser.add_argument(
        "--fallback-overlap-lines",
        type=int,
        default=DEFAULT_FALLBACK_OVERLAP_LINES,
    )
    parser.add_argument(
        "--max-structural-lines", type=int, default=DEFAULT_MAX_STRUCTURAL_LINES
    )
    return parser


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    config = ChunkingConfig(
        fallback_max_lines=options.fallback_max_lines,
        fallback_overlap_lines=options.fallback_overlap_lines,
        max_structural_lines=options.max_structural_lines,
    )
    try:
        manifest = ingest_repository(options.source, cache_directory=options.cache_dir)
        result = chunk_repository(manifest, config=config)
        output_path = write_chunks(result, manifest.repository.name, options.output_dir)
    except (IngestionError, ChunkingError, OSError, ValueError) as exc:
        print(f"Chunking failed: {exc}", file=sys.stderr)
        return 1

    repository = manifest.repository
    print(f"Repository: {repository.name}")
    print(f"Commit: {repository.commit_hash or 'unavailable'}")
    print(f"Accepted files: {repository.accepted_file_count}")
    print(f"Chunks generated: {len(result.chunks)}")
    print(f"Structural chunks: {result.structural_chunk_count}")
    print(f"Fallback chunks: {result.fallback_chunk_count}")
    print("Languages:")
    for language, count in sorted(Counter(chunk.language for chunk in result.chunks).items()):
        print(f"  {language}: {count}")
    print(f"Output: {_display_path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
