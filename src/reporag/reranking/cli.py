"""CLI for Method D Hybrid RRF plus cross-encoder reranking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reporag.embeddings.index import IndexErrorBase, load_index as load_vector_index
from reporag.embeddings.model import SentenceTransformerEmbeddingModel
from reporag.lexical.index import BM25IndexError, load_index as load_bm25_index
from reporag.retrieval.hybrid import HybridCompatibilityError, create_hybrid_metadata

from .config import DEFAULT_RERANKER_CONFIG
from .metadata import create_metadata, save_metadata
from .model import CrossEncoderRerankerModel
from .reranker import search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RepoRAG Method D reranking.")
    parser.add_argument("vector_index", type=Path)
    parser.add_argument("bm25_index", type=Path)
    parser.add_argument("query", help="Original query used unchanged at both stages")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument("--metadata-root", type=Path, default=Path("data/reranking"))
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--show-content", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _symbol(parent: str | None, symbol: str | None) -> str:
    if parent and symbol:
        return f"{parent}.{symbol}"
    return symbol or parent or "(file context)"


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    try:
        vector = load_vector_index(options.vector_index)
        bm25 = load_bm25_index(options.bm25_index)
        hybrid_metadata = create_hybrid_metadata(vector, bm25)
        metadata = create_metadata(hybrid_metadata)
        metadata_path = save_metadata(
            metadata, options.metadata_root / metadata.repository.lower()
        )
        embedding_model = SentenceTransformerEmbeddingModel(
            config=vector.metadata.embedding_config,
            cache_directory=options.model_cache,
            local_files_only=True,
        )
        reranker_model = CrossEncoderRerankerModel(
            config=DEFAULT_RERANKER_CONFIG,
            cache_directory=options.model_cache,
            local_files_only=not options.allow_download,
        )
        outcome = search(
            vector,
            bm25,
            embedding_model,
            reranker_model,
            options.query,
            final_top_k=options.top_k,
        )
    except (IndexErrorBase, BM25IndexError, HybridCompatibilityError, OSError, ValueError) as exc:
        print(f"Reranking failed: {exc}", file=sys.stderr)
        return 1

    for result in outcome.results:
        print(f"{result.rank}. reranker_logit={result.reranker_score:.6f}")
        print(f"   {result.file_path}")
        print(f"   {_symbol(result.parent_symbol, result.symbol_name)}")
        print(f"   lines {result.start_line}-{result.end_line}")
        print(f"   Hybrid: previous_rank={result.hybrid_rank} RRF={result.hybrid_rrf_score:.8f}")
        if options.verbose:
            print(
                f"   Vector: rank={result.vector_rank} cosine={result.vector_raw_score}"
            )
            print(f"   BM25: rank={result.bm25_rank} score={result.bm25_raw_score}")
        if options.show_content:
            print("--- source ---")
            print(result.content)
            print("--- end source ---")

    diagnostics = outcome.diagnostics
    timings = outcome.timings
    print(f"Candidates: {diagnostics.candidate_count}")
    print(f"Candidates over {metadata.max_length}: {diagnostics.candidates_over_configured_limit}")
    print(f"Candidates truncated: {diagnostics.candidates_truncated}")
    print(f"Device: {reranker_model.device}; dtype: {reranker_model.dtype}")
    print(f"Batch size: {metadata.batch_size}")
    print(f"Model load seconds: {reranker_model.load_seconds:.6f}")
    print(f"Candidate generation seconds: {timings.candidate_generation_seconds:.6f}")
    print(f"Formatting/tokenization seconds: {timings.formatting_tokenization_seconds:.6f}")
    print(f"Cross-encoder scoring seconds: {timings.scoring_seconds:.6f}")
    print(f"Sorting/finalization seconds: {timings.sorting_finalization_seconds:.6f}")
    print(f"Total Method D query seconds: {timings.total_seconds:.6f}")
    print(f"Method D fingerprint: {metadata.reranking_fingerprint}")
    print(f"Method D metadata: {metadata_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
