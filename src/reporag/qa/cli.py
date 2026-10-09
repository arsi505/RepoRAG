"""Repository Q&A CLI over frozen RepoRAG retrieval methods."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reporag.embeddings.index import IndexErrorBase, load_index as load_vector
from reporag.generation.fake import FakeGenerator
from reporag.generation.openai_provider import OpenAIResponsesProvider
from reporag.lexical.index import BM25IndexError, load_index as load_bm25
from reporag.retrieval.hybrid import HybridCompatibilityError

from .config import DEFAULT_QA_CONFIG, SUPPORTED_METHODS
from .models import Evidence, QAResult
from .retrieval import RepositoryRetriever
from .service import QAServiceError, answer_question


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask one grounded repository question.")
    parser.add_argument("vector_index", type=Path)
    parser.add_argument("bm25_index", type=Path)
    parser.add_argument("question")
    parser.add_argument(
        "--method", choices=sorted(SUPPORTED_METHODS), default="hybrid"
    )
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    return parser


def _symbol(item: Evidence) -> str:
    if item.parent_symbol and item.symbol_name:
        return f"{item.parent_symbol}.{item.symbol_name}"
    return item.symbol_name or item.parent_symbol or "(file context)"


def _print_sources(result: QAResult) -> None:
    print("Sources:")
    for item in result.evidence:
        cited = " cited" if item.source_id in result.cited_source_ids else ""
        print(f"[{item.source_id}]{cited}")
        print(item.file_path)
        print(_symbol(item))
        print(f"lines {item.start_line}-{item.end_line}")


def main(arguments: list[str] | None = None) -> int:
    options = build_parser().parse_args(arguments)
    try:
        vector = load_vector(options.vector_index)
        bm25 = load_bm25(options.bm25_index)
        retriever = RepositoryRetriever(
            vector, bm25, model_cache=options.model_cache
        )
        generator = (
            FakeGenerator()
            if options.dry_run
            else OpenAIResponsesProvider(
                model=DEFAULT_QA_CONFIG.generation_model,
                max_output_tokens=DEFAULT_QA_CONFIG.max_output_tokens,
            )
        )
        result = answer_question(
            options.question,
            retriever,
            generator,
            retrieval_method=options.method,
            context_k=options.context_k,
            dry_run=options.dry_run,
        )
    except (
        IndexErrorBase,
        BM25IndexError,
        HybridCompatibilityError,
        QAServiceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Repository Q&A failed: {exc}", file=sys.stderr)
        return 1

    print(result.answer_text)
    print()
    _print_sources(result)
    print(f"Retrieval method: {result.retrieval_method}")
    print(f"Context chunks: {len(result.evidence)}")
    print(f"Citations valid: {result.citations_valid}")
    if result.unknown_source_ids:
        print(f"Unknown citations: {', '.join(result.unknown_source_ids)}")
    if result.provider:
        print(f"Provider/model: {result.provider}/{result.model}")
    if result.usage is not None:
        print(f"Usage: {json.dumps(result.usage, sort_keys=True)}")
    if options.show_context or options.dry_run:
        print("--- generation context ---")
        print(result.context_text)
        print("--- end generation context ---")
    if result.timings is not None:
        timings = result.timings
        print(f"Retrieval seconds: {timings.retrieval_seconds:.6f}")
        print(f"Context assembly seconds: {timings.context_assembly_seconds:.6f}")
        if timings.generation_seconds is not None:
            print(f"Generation seconds: {timings.generation_seconds:.6f}")
        print(f"Citation validation seconds: {timings.citation_validation_seconds:.6f}")
        print(f"Total Q&A seconds: {timings.total_seconds:.6f}")
    print(f"Q&A fingerprint: {result.qa_fingerprint}")
    return 0 if result.citations_valid or result.dry_run or not result.evidence else 2


if __name__ == "__main__":
    raise SystemExit(main())
