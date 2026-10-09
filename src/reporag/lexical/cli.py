"""Build and search transparent BM25 indexes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from reporag.chunking.chunker import ChunkingError, chunk_repository
from reporag.chunking.writer import chunk_filename, write_chunks
from reporag.embeddings.index import load_chunks
from reporag.ingestion.repository import IngestionError, ingest_repository

from .bm25 import BM25SearchError, search
from .config import DEFAULT_BM25_CONFIG
from .index import BM25IndexError, build_index, load_index, save_index


def _slug(repository_name: str) -> str:
    return Path(chunk_filename(repository_name)).stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or search a RepoRAG BM25 index.")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build BM25 over frozen Day 3 chunks")
    build.add_argument("source", help="Local repository or public GitHub URL")
    build.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    build.add_argument("--index-root", type=Path, default=Path("data/bm25"))
    build.add_argument("--repository-cache", type=Path, default=Path("data/repository_cache"))

    query = commands.add_parser("search", help="Search an existing BM25 index")
    query.add_argument("index", type=Path, help="BM25 index directory")
    query.add_argument("query", help="Natural-language or identifier query")
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--show-content", action="store_true")
    return parser


def _frozen_chunks(source: str, chunks_dir: Path, repository_cache: Path):
    manifest = ingest_repository(source, cache_directory=repository_cache)
    generated = chunk_repository(manifest)
    chunk_path = chunks_dir / chunk_filename(manifest.repository.name)
    if chunk_path.exists():
        frozen = load_chunks(chunk_path)
        if [chunk.to_dict() for chunk in frozen] != [
            chunk.to_dict() for chunk in generated.chunks
        ]:
            raise ChunkingError(
                f"Current repository does not match frozen Day 3 chunks at {chunk_path}"
            )
        return manifest, frozen, chunk_path.resolve()
    output = write_chunks(generated, manifest.repository.name, chunks_dir)
    return manifest, generated.chunks, output


def _display_symbol(parent: str | None, symbol: str | None) -> str:
    if parent and symbol:
        return f"{parent}.{symbol}"
    return symbol or parent or "(file context)"


def _build(options: argparse.Namespace) -> int:
    try:
        manifest, chunks, chunk_path = _frozen_chunks(
            options.source, options.chunks_dir, options.repository_cache
        )
        index = build_index(chunks)
        destination = options.index_root / _slug(manifest.repository.name)
        index = save_index(index, destination)
    except (IngestionError, ChunkingError, BM25IndexError, OSError, ValueError) as exc:
        print(f"BM25 index build failed: {exc}", file=sys.stderr)
        return 1

    metadata = index.metadata
    print(f"Repository: {metadata.repository}")
    print(f"Commit: {metadata.commit_hash or 'unavailable'}")
    print(f"Chunks: {metadata.chunk_count}")
    print(f"Frozen chunks: {chunk_path}")
    print(f"Tokenizer: {metadata.tokenizer_version}")
    print(f"Formatter: {metadata.formatter_version}")
    print(f"BM25 k1: {metadata.k1}")
    print(f"BM25 b: {metadata.b}")
    print(f"Total lexical tokens: {metadata.total_tokens}")
    print(f"Average document length: {metadata.average_document_length:.6f}")
    print(f"Minimum document length: {metadata.minimum_document_length}")
    print(f"Median document length: {metadata.median_document_length:.6f}")
    print(f"Maximum document length: {metadata.maximum_document_length}")
    print(f"Vocabulary size: {metadata.vocabulary_size}")
    print(f"Source fingerprint: {metadata.source_fingerprint}")
    print(f"Build/tokenization seconds: {index.build_seconds:.6f}")
    print(f"Index save seconds: {index.save_seconds:.6f}")
    print("Ten longest tokenized documents:")
    longest = sorted(
        zip(index.chunks, index.documents, strict=True),
        key=lambda item: (-len(item[1]), item[0].chunk_id),
    )[:10]
    for chunk, document in longest:
        print(
            f"  {len(document)} tokens | {chunk.file_path} | {chunk.chunk_type} | "
            f"{chunk.symbol_name or '-'} | lines {chunk.start_line}-{chunk.end_line}"
        )
    print(f"Index: {destination.resolve()}")
    return 0


def _search(options: argparse.Namespace) -> int:
    try:
        index = load_index(options.index)
        started = perf_counter()
        results = search(index, options.query, top_k=options.top_k)
        scoring_seconds = perf_counter() - started
    except (BM25IndexError, BM25SearchError, OSError, ValueError) as exc:
        print(f"BM25 search failed: {exc}", file=sys.stderr)
        return 1
    for result in results:
        print(f"{result.rank}. bm25={result.score:.6f}")
        print(f"   {result.file_path}")
        print(f"   {_display_symbol(result.parent_symbol, result.symbol_name)}")
        print(f"   lines {result.start_line}-{result.end_line}")
        if options.show_content:
            print("--- source ---")
            print(result.content)
            print("--- end source ---")
    if not results:
        print("No positive-scoring BM25 results.")
    print(f"Index load seconds: {index.load_seconds:.6f}")
    print(f"Query scoring seconds: {scoring_seconds:.6f}")
    return 0


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    return _build(options) if options.command == "build" else _search(options)


if __name__ == "__main__":
    raise SystemExit(main())
