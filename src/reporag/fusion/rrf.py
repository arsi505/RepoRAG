"""Pure deterministic Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


class RRFFusionError(ValueError):
    """Raised when a ranked input or RRF parameter is malformed."""


@dataclass(frozen=True, slots=True)
class RankedItem:
    chunk_id: str
    rank: int
    raw_score: float


@dataclass(frozen=True, slots=True)
class FusedRank:
    rank: int
    chunk_id: str
    rrf_score: float
    vector_rank: int | None
    vector_raw_score: float | None
    vector_contribution: float
    bm25_rank: int | None
    bm25_raw_score: float | None
    bm25_contribution: float


def _validated(items: Iterable[RankedItem], ranker: str) -> dict[str, RankedItem]:
    by_id: dict[str, RankedItem] = {}
    ranks: set[int] = set()
    for item in items:
        if not item.chunk_id:
            raise RRFFusionError(f"{ranker} chunk_id must not be empty")
        if isinstance(item.rank, bool) or not isinstance(item.rank, int) or item.rank < 1:
            raise RRFFusionError(f"{ranker} rank must be a positive 1-based integer")
        if not math.isfinite(item.raw_score):
            raise RRFFusionError(f"{ranker} raw score must be finite")
        if item.chunk_id in by_id:
            raise RRFFusionError(f"Duplicate chunk_id in {ranker} ranking: {item.chunk_id}")
        if item.rank in ranks:
            raise RRFFusionError(f"Duplicate rank in {ranker} ranking: {item.rank}")
        by_id[item.chunk_id] = item
        ranks.add(item.rank)
    return by_id


def reciprocal_rank_fusion(
    vector_ranking: Iterable[RankedItem],
    bm25_ranking: Iterable[RankedItem],
    *,
    k: int = 60,
    top_k: int | None = None,
) -> tuple[FusedRank, ...]:
    """Fuse complete Vector and positive-score BM25 rankings by exact chunk ID.

    Scores are ``sum(1 / (k + rank_i))``. Values equal at 12 decimal
    places are treated as effectively tied, then ordered by chunk ID.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RRFFusionError("RRF k must be a positive integer")
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
    ):
        raise RRFFusionError("top_k must be a positive integer when provided")

    vector = _validated(vector_ranking, "Vector")
    bm25 = _validated(bm25_ranking, "BM25")
    candidates = set(vector) | set(bm25)
    scored: list[tuple[str, float, float, float]] = []
    for chunk_id in candidates:
        vector_contribution = (
            1.0 / (k + vector[chunk_id].rank) if chunk_id in vector else 0.0
        )
        bm25_contribution = (
            1.0 / (k + bm25[chunk_id].rank) if chunk_id in bm25 else 0.0
        )
        scored.append(
            (chunk_id, vector_contribution + bm25_contribution, vector_contribution, bm25_contribution)
        )

    # Raw retriever scores are deliberately not used as a tie-breaker.
    scored.sort(key=lambda item: (-round(item[1], 12), item[0]))
    if top_k is not None:
        scored = scored[:top_k]

    return tuple(
        FusedRank(
            rank=rank,
            chunk_id=chunk_id,
            rrf_score=score,
            vector_rank=vector[chunk_id].rank if chunk_id in vector else None,
            vector_raw_score=vector[chunk_id].raw_score if chunk_id in vector else None,
            vector_contribution=vector_contribution,
            bm25_rank=bm25[chunk_id].rank if chunk_id in bm25 else None,
            bm25_raw_score=bm25[chunk_id].raw_score if chunk_id in bm25 else None,
            bm25_contribution=bm25_contribution,
        )
        for rank, (chunk_id, score, vector_contribution, bm25_contribution) in enumerate(
            scored, start=1
        )
    )
