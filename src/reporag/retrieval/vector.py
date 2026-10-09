"""Deterministic exhaustive cosine-similarity retrieval."""

from __future__ import annotations

import numpy as np

from reporag.embeddings.index import EmbeddingIndex
from reporag.embeddings.model import EmbeddingModel, normalize_rows

from .models import SearchResult


class VectorSearchError(ValueError):
    """Raised for invalid query/search state."""


def search(
    index: EmbeddingIndex,
    model: EmbeddingModel,
    query: str,
    *,
    top_k: int = 5,
) -> tuple[SearchResult, ...]:
    if not query.strip():
        raise VectorSearchError("Query must not be empty")
    if top_k <= 0:
        raise VectorSearchError("top_k must be positive")
    if not index.chunks:
        raise VectorSearchError("Cannot search an empty index")
    if model.config != index.metadata.embedding_config:
        raise VectorSearchError("Embedding model configuration does not match the index")
    try:
        query_matrix = normalize_rows(model.encode_queries([query]))
    except ValueError as exc:
        raise VectorSearchError(f"Invalid query embedding: {exc}") from exc
    expected_shape = (1, index.metadata.embedding_dimension)
    if query_matrix.shape != expected_shape:
        raise VectorSearchError(
            f"Query embedding shape mismatch: expected {expected_shape}, got {query_matrix.shape}"
        )
    scores = np.asarray(index.embeddings @ query_matrix[0], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise VectorSearchError("Cosine similarity contains NaN or infinity")
    # Scores equal at 12 decimal places are treated as effectively tied.
    order = sorted(
        range(len(index.chunks)),
        key=lambda position: (-round(float(scores[position]), 12), index.chunks[position].chunk_id),
    )[: min(top_k, len(index.chunks))]
    results: list[SearchResult] = []
    for rank, position in enumerate(order, start=1):
        chunk = index.chunks[position]
        results.append(
            SearchResult(
                rank=rank,
                chunk_id=chunk.chunk_id,
                score=float(scores[position]),
                repository=chunk.repository,
                file_path=chunk.file_path,
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                symbol_name=chunk.symbol_name,
                parent_symbol=chunk.parent_symbol,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
            )
        )
    return tuple(results)
