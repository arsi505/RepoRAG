"""CLI for building the frozen Day 4 vector index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from reporag.chunking.chunker import ChunkingError, chunk_repository
from reporag.chunking.writer import chunk_filename, write_chunks
from reporag.ingestion.repository import IngestionError, ingest_repository

from .config import DEFAULT_EMBEDDING_CONFIG
from .index import IndexErrorBase, build_index, load_chunks, save_index
from .model import SentenceTransformerEmbeddingModel


def _slug(repository_name: str) -> str:
    return Path(chunk_filename(repository_name)).stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a persistent exact vector index.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build embeddings for frozen Day 3 chunks")
    build.add_argument("source", help="Local repository or public GitHub URL")
    build.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    build.add_argument("--index-root", type=Path, default=Path("data/embeddings"))
    build.add_argument("--repository-cache", type=Path, default=Path("data/repository_cache"))
    build.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    build.add_argument("--batch-size", type=int, default=16)
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


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    total_started = perf_counter()
    try:
        manifest, chunks, chunk_path = _frozen_chunks(
            options.source, options.chunks_dir, options.repository_cache
        )
        model = SentenceTransformerEmbeddingModel(
            config=DEFAULT_EMBEDDING_CONFIG,
            cache_directory=options.model_cache,
            batch_size=options.batch_size,
            show_progress=True,
        )
        index = build_index(chunks, model)
        index_directory = options.index_root / _slug(manifest.repository.name)
        index = save_index(index, index_directory)
    except (IngestionError, ChunkingError, IndexErrorBase, OSError, ValueError) as exc:
        print(f"Index build failed: {exc}", file=sys.stderr)
        return 1

    config = index.metadata.embedding_config
    print(f"Repository: {index.metadata.repository}")
    print(f"Commit: {index.metadata.commit_hash or 'unavailable'}")
    print(f"Chunks: {index.metadata.chunk_count}")
    print(f"Frozen chunks: {chunk_path}")
    print(f"Model: {config.model_name}")
    print(f"Model revision: {config.model_revision}")
    print(f"Trusted code revision: {config.trusted_code_revision or 'not applicable'}")
    print(f"Dimension: {config.embedding_dimension}")
    print(f"Maximum sequence length: {config.max_sequence_length}")
    print(
        "Maximum observed document tokens: "
        f"{index.metadata.maximum_observed_document_tokens}"
    )
    print(f"Documents over model limit: {index.metadata.documents_over_model_limit}")
    print(f"Documents truncated: {index.metadata.documents_truncated}")
    if index.metadata.truncated_chunk_ids:
        print("Truncated chunk IDs:")
        for chunk_id in index.metadata.truncated_chunk_ids:
            print(f"  {chunk_id}")
    print(f"Normalized: {'yes' if config.normalize_embeddings else 'no'}")
    print(f"Source fingerprint: {index.metadata.source_fingerprint}")
    print(f"Model load seconds: {model.load_seconds:.3f}")
    print(f"Embedding seconds: {index.embedding_seconds:.3f}")
    print(f"Index save seconds: {index.save_seconds:.3f}")
    print(f"Total seconds: {perf_counter() - total_started:.3f}")
    print(f"Index: {index_directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
