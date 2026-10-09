"""Standard exhaustive BM25 scoring and deterministic ranking."""

from __future__ import annotations

import math
from collections import Counter

from reporag.retrieval.models import SearchResult

from .index import BM25Index
from .tokenizer import tokenize


class BM25SearchError(ValueError):
    """Raised for an invalid lexical query or search parameter."""


def score_document(
    query_tokens: tuple[str, ...],
    document: tuple[str, ...],
    index: BM25Index,
) -> float:
    frequencies = Counter(document)
    length = len(document)
    config = index.metadata.config
    normalization = config.k1 * (
        1.0 - config.b + config.b * length / index.metadata.average_document_length
    )
    score = 0.0
    for term in query_tokens:
        term_frequency = frequencies.get(term, 0)
        if term_frequency:
            numerator = term_frequency * (config.k1 + 1.0)
            denominator = term_frequency + normalization
            score += index.inverse_document_frequencies.get(term, 0.0) * (
                numerator / denominator
            )
    if not math.isfinite(score):
        raise BM25SearchError("BM25 scoring produced NaN or infinity")
    return score


def search(index: BM25Index, query: str, *, top_k: int = 5) -> tuple[SearchResult, ...]:
    if not query.strip():
        raise BM25SearchError("Query must not be empty")
    if top_k <= 0:
        raise BM25SearchError("top_k must be positive")
    if not index.chunks:
        raise BM25SearchError("Cannot search an empty BM25 index")
    query_tokens = tokenize(query)
    if not query_tokens:
        raise BM25SearchError("Query contains no searchable lexical tokens")
    scores = [
        score_document(query_tokens, document, index) for document in index.documents
    ]
    positive = [position for position, score in enumerate(scores) if score > 0.0]
    order = sorted(
        positive,
        key=lambda position: (
            -round(scores[position], 12),
            index.chunks[position].chunk_id,
        ),
    )[:top_k]
    results = []
    for rank, position in enumerate(order, start=1):
        chunk = index.chunks[position]
        results.append(
            SearchResult(
                rank=rank,
                chunk_id=chunk.chunk_id,
                score=scores[position],
                repository=chunk.repository,
                file_path=chunk.file_path,
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                symbol_name=chunk.symbol_name,
                parent_symbol=chunk.parent_symbol,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                method="bm25",
                score_name="bm25",
            )
        )
    return tuple(results)
