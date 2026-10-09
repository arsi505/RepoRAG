"""Repository Q&A CLI over frozen RepoRAG retrieval methods."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from reporag.embeddings.index import IndexErrorBase, load_index as load_vector
from reporag.generation.deepseek_provider import DeepSeekProvider
from reporag.generation.environment import load_local_environment
from reporag.generation.fake import FakeGenerator
from reporag.generation.gemini_provider import GeminiProvider
from reporag.generation.openai_provider import OpenAIResponsesProvider
from reporag.lexical.index import BM25IndexError, load_index as load_bm25
from reporag.retrieval.hybrid import HybridCompatibilityError

from .config import (
    DEFAULT_QA_CONFIG,
    SUPPORTED_EVIDENCE_SCOPES,
    SUPPORTED_METHODS,
    SUPPORTED_PROVIDERS,
    model_for_provider,
)
from .models import Evidence, QAResult
from .retrieval import RepositoryRetriever
from .service import QAServiceError, answer_question


def _configure_utf8_output() -> None:
    """Keep generated Unicode answers printable on Windows legacy consoles."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask one grounded repository question.")
    parser.add_argument("vector_index", type=Path)
    parser.add_argument("bm25_index", type=Path)
    parser.add_argument("question")
    parser.add_argument(
        "--method", choices=sorted(SUPPORTED_METHODS), default="hybrid"
    )
    parser.add_argument("--context-k", type=int, default=5)
    parser.add_argument(
        "--evidence-scope",
        choices=sorted(SUPPORTED_EVIDENCE_SCOPES),
        default=DEFAULT_QA_CONFIG.evidence_scope,
    )
    parser.add_argument(
        "--provider", choices=sorted(SUPPORTED_PROVIDERS), default="deepseek"
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_QA_CONFIG.max_output_tokens,
        help="maximum generated answer tokens (default: %(default)s)",
    )
    parser.add_argument("--model-cache", type=Path, default=Path("data/model_cache"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-context", action="store_true")
    return parser


def create_generator(
    provider: str,
    *,
    dry_run: bool,
    max_output_tokens: int = DEFAULT_QA_CONFIG.max_output_tokens,
):
    if dry_run:
        return FakeGenerator()
    model = model_for_provider(provider)
    if provider == "deepseek":
        return DeepSeekProvider(
            model=model,
            max_output_tokens=max_output_tokens,
        )
    if provider == "gemini":
        return GeminiProvider(
            model=model,
            max_output_tokens=max_output_tokens,
        )
    if provider == "openai":
        return OpenAIResponsesProvider(
            model=model,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError(f"Unsupported generation provider: {provider}")


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
    _configure_utf8_output()
    options = build_parser().parse_args(arguments)
    try:
        load_local_environment()
        vector = load_vector(options.vector_index)
        bm25 = load_bm25(options.bm25_index)
        retriever = RepositoryRetriever(
            vector, bm25, model_cache=options.model_cache
        )
        qa_config = replace(
            DEFAULT_QA_CONFIG,
            max_output_tokens=options.max_output_tokens,
            evidence_scope=options.evidence_scope,
        )
        generation_model = model_for_provider(options.provider)
        generator = create_generator(
            options.provider,
            dry_run=options.dry_run,
            max_output_tokens=qa_config.max_output_tokens,
        )
        result = answer_question(
            options.question,
            retriever,
            generator,
            retrieval_method=options.method,
            context_k=options.context_k,
            dry_run=options.dry_run,
            config=qa_config,
            generation_provider=options.provider,
            generation_model=generation_model,
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
    print(f"Evidence scope: {result.evidence_scope}")
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
