"""Minimal semantic repository-search CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

from reporag.embeddings.index import IndexErrorBase, load_index
from reporag.embeddings.model import SentenceTransformerEmbeddingModel

from .vector import VectorSearchError, search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search an existing RepoRAG vector index.")
    parser.add_argument("index", type=Path, help="Index directory containing metadata.json")
    parser.add_argument("query", help="Natural-language software-engineering question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--show-content", action="store_true")
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    return parser


def _symbol(parent: str | None, symbol: str | None) -> str:
    if parent and symbol:
        return f"{parent}.{symbol}"
    return symbol or parent or "(file context)"


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    try:
        index = load_index(options.index)
        model = SentenceTransformerEmbeddingModel(
            config=index.metadata.embedding_config,
            cache_directory=options.model_cache,
            local_files_only=True,
        )
        started = perf_counter()
        results = search(index, model, options.query, top_k=options.top_k)
        retrieval_seconds = perf_counter() - started
    except (IndexErrorBase, VectorSearchError, OSError, ValueError) as exc:
        print(f"Search failed: {exc}", file=sys.stderr)
        return 1

    for result in results:
        print(f"{result.rank}. score={result.score:.6f}")
        print(f"   {result.file_path}")
        print(f"   {_symbol(result.parent_symbol, result.symbol_name)}")
        print(f"   lines {result.start_line}-{result.end_line}")
        if options.show_content:
            print("--- source ---")
            print(result.content)
            print("--- end source ---")
    print(f"Query retrieval seconds: {retrieval_seconds:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
