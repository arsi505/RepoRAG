"""Two-stage Method D candidate generation and cross-encoder reranking."""

from __future__ import annotations

from time import perf_counter
from typing import Sequence

from reporag.embeddings.index import EmbeddingIndex
from reporag.embeddings.model import EmbeddingModel
from reporag.lexical.index import BM25Index
from reporag.retrieval.hybrid import search as hybrid_search
from reporag.retrieval.models import HybridSearchResult

from .config import DEFAULT_RERANKER_CONFIG, RerankerConfig
from .formatter import format_candidate
from .model import RerankerModel, validate_scores
from .models import RerankedResult, RerankingDiagnostics, RerankingOutcome, RerankingTimings


def rerank_candidates(
    query: str,
    candidates: Sequence[HybridSearchResult],
    model: RerankerModel,
    *,
    final_top_k: int = 5,
    config: RerankerConfig = DEFAULT_RERANKER_CONFIG,
    candidate_generation_seconds: float = 0.0,
) -> RerankingOutcome:
    """Rerank at most the frozen candidate depth without mutating inputs."""

    config.validate_final_top_k(final_top_k)
    if model.config != config:
        raise ValueError("Reranker model configuration does not match Method D")
    identifiers = [candidate.chunk_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate chunk IDs in Hybrid candidate ranking")
    eligible = tuple(candidates[: config.candidate_k])
    if not eligible:
        return RerankingOutcome(
            results=(),
            diagnostics=RerankingDiagnostics(0, 0, 0, ()),
            timings=RerankingTimings(
                candidate_generation_seconds, 0.0, 0.0, 0.0, candidate_generation_seconds
            ),
        )

    total_started = perf_counter()
    formatting_started = perf_counter()
    documents = tuple(format_candidate(candidate) for candidate in eligible)
    pairs = tuple((query, document) for document in documents)
    lengths = model.pair_token_lengths(pairs)
    if len(lengths) != len(pairs) or any(length <= 0 for length in lengths):
        raise ValueError("Reranker returned invalid pair token lengths")
    over_limit = sum(length > config.max_length for length in lengths)
    formatting_seconds = perf_counter() - formatting_started

    scoring_started = perf_counter()
    scores = validate_scores(model.score_pairs(pairs), len(eligible))
    scoring_seconds = perf_counter() - scoring_started

    sorting_started = perf_counter()
    scored = list(zip(eligible, scores, strict=True))
    scored.sort(
        key=lambda item: (
            -round(item[1], 12),
            item[0].rank,
            item[0].chunk_id,
        )
    )
    selected = scored[: min(final_top_k, len(scored))]
    results = tuple(
        RerankedResult(
            rank=rank,
            chunk_id=candidate.chunk_id,
            reranker_score=score,
            hybrid_rank=candidate.rank,
            hybrid_rrf_score=candidate.rrf_score,
            vector_rank=candidate.vector_rank,
            vector_raw_score=candidate.vector_raw_score,
            bm25_rank=candidate.bm25_rank,
            bm25_raw_score=candidate.bm25_raw_score,
            repository=candidate.repository,
            file_path=candidate.file_path,
            language=candidate.language,
            chunk_type=candidate.chunk_type,
            symbol_name=candidate.symbol_name,
            parent_symbol=candidate.parent_symbol,
            start_line=candidate.start_line,
            end_line=candidate.end_line,
            content=candidate.content,
        )
        for rank, (candidate, score) in enumerate(selected, start=1)
    )
    sorting_seconds = perf_counter() - sorting_started
    reranking_seconds = perf_counter() - total_started
    return RerankingOutcome(
        results=results,
        diagnostics=RerankingDiagnostics(
            candidate_count=len(eligible),
            candidates_over_configured_limit=over_limit,
            candidates_truncated=over_limit,
            pair_token_lengths=lengths,
        ),
        timings=RerankingTimings(
            candidate_generation_seconds=candidate_generation_seconds,
            formatting_tokenization_seconds=formatting_seconds,
            scoring_seconds=scoring_seconds,
            sorting_finalization_seconds=sorting_seconds,
            total_seconds=candidate_generation_seconds + reranking_seconds,
        ),
    )


def search(
    vector_index: EmbeddingIndex,
    bm25_index: BM25Index,
    embedding_model: EmbeddingModel,
    reranker_model: RerankerModel,
    query: str,
    *,
    final_top_k: int = 5,
    config: RerankerConfig = DEFAULT_RERANKER_CONFIG,
) -> RerankingOutcome:
    """Reuse Method C for top-50 generation, then rerank only those candidates."""

    config.validate_final_top_k(final_top_k)
    started = perf_counter()
    hybrid = hybrid_search(
        vector_index,
        bm25_index,
        embedding_model,
        query,
        top_k=config.candidate_k,
    )
    candidate_seconds = perf_counter() - started
    return rerank_candidates(
        query,
        hybrid.results,
        reranker_model,
        final_top_k=final_top_k,
        config=config,
        candidate_generation_seconds=candidate_seconds,
    )
