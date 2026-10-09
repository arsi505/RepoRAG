"""Command-line interface for Method C hybrid RRF search."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reporag.embeddings.index import IndexErrorBase, load_index as load_vector_index
from reporag.embeddings.model import SentenceTransformerEmbeddingModel
from reporag.lexical.index import BM25IndexError, load_index as load_bm25_index

from .hybrid import (
    HybridCompatibilityError,
    create_hybrid_metadata,
    save_hybrid_metadata,
    search,
)
from .vector import VectorSearchError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse complete RepoRAG Vector and BM25 rankings with RRF."
    )
    parser.add_argument("vector_index", type=Path)
    parser.add_argument("bm25_index", type=Path)
    parser.add_argument("query", help="Original query sent unchanged to both retrievers")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-content", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/hybrid"))
    return parser


def _symbol(parent: str | None, symbol: str | None) -> str:
    if parent and symbol:
        return f"{parent}.{symbol}"
    return symbol or parent or "(file context)"


def _rank(value: int | None) -> str:
    return str(value) if value is not None else "absent"


def _score(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "absent"


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    try:
        vector_index = load_vector_index(options.vector_index)
        bm25_index = load_bm25_index(options.bm25_index)
        metadata = create_hybrid_metadata(vector_index, bm25_index)
        metadata_directory = options.metadata_root / metadata.repository.lower()
        metadata_path = save_hybrid_metadata(metadata, metadata_directory)
        model = SentenceTransformerEmbeddingModel(
            config=vector_index.metadata.embedding_config,
            cache_directory=options.model_cache,
            local_files_only=True,
        )
        outcome = search(
            vector_index,
            bm25_index,
            model,
            options.query,
            top_k=options.top_k,
        )
    except (
        IndexErrorBase,
        BM25IndexError,
        HybridCompatibilityError,
        VectorSearchError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Hybrid search failed: {exc}", file=sys.stderr)
        return 1

    for result in outcome.results:
        print(f"{result.rank}. RRF={result.rrf_score:.8f}")
        print(f"   {result.file_path}")
        print(f"   {_symbol(result.parent_symbol, result.symbol_name)}")
        print(f"   lines {result.start_line}-{result.end_line}")
        print(
            f"   Vector: rank={_rank(result.vector_rank)} "
            f"cosine={_score(result.vector_raw_score)}"
        )
        print(
            f"   BM25: rank={_rank(result.bm25_rank)} "
            f"score={_score(result.bm25_raw_score)}"
        )
        if options.verbose:
            print(
                "   Contributions: "
                f"vector={result.vector_contribution:.8f} "
                f"bm25={result.bm25_contribution:.8f}"
            )
        if options.show_content:
            print("--- source ---")
            print(result.content)
            print("--- end source ---")

    print(f"RRF version: {metadata.rrf_version}; k={metadata.rrf_k}")
    print(f"Hybrid fingerprint: {metadata.hybrid_fingerprint}")
    print(f"Hybrid metadata: {metadata_path.resolve()}")
    print(f"Vector ranking seconds: {outcome.timings.vector_ranking_seconds:.6f}")
    print(f"BM25 ranking seconds: {outcome.timings.bm25_ranking_seconds:.6f}")
    print(f"RRF fusion seconds: {outcome.timings.rrf_fusion_seconds:.6f}")
    print(f"Total hybrid query seconds: {outcome.timings.total_seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
