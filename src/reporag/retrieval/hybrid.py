"""Method C: full-ranking Vector/BM25 fusion with RRF."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from reporag.chunking.models import Chunk
from reporag.embeddings.config import DEFAULT_EMBEDDING_CONFIG
from reporag.embeddings.index import EmbeddingIndex
from reporag.embeddings.model import EmbeddingModel
from reporag.fusion.rrf import RankedItem, reciprocal_rank_fusion
from reporag.lexical.bm25 import search as bm25_search
from reporag.lexical.config import DEFAULT_BM25_CONFIG
from reporag.lexical.index import BM25Index

from .models import HybridSearchResult
from .vector import search as vector_search


METHOD_NAME = "hybrid_rrf"
RRF_VERSION = "reporag-rrf-v1"
RRF_K = 60
VECTOR_METHOD_VERSION = "reporag-exact-cosine-v1"
BM25_METHOD_VERSION = "reporag-bm25-v1"


class HybridCompatibilityError(ValueError):
    """Raised when frozen Vector and BM25 indexes cannot be fused."""


@dataclass(frozen=True, slots=True)
class HybridConfig:
    method_name: str = METHOD_NAME
    rrf_version: str = RRF_VERSION
    rrf_k: int = RRF_K
    vector_method_version: str = VECTOR_METHOD_VERSION
    bm25_method_version: str = BM25_METHOD_VERSION

    def __post_init__(self) -> None:
        if self.method_name != METHOD_NAME:
            raise ValueError(f"Hybrid method_name must be {METHOD_NAME}")
        if not self.rrf_version:
            raise ValueError("RRF version must not be empty")
        if isinstance(self.rrf_k, bool) or not isinstance(self.rrf_k, int) or self.rrf_k <= 0:
            raise ValueError("RRF k must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_HYBRID_CONFIG = HybridConfig()


@dataclass(frozen=True, slots=True)
class HybridMetadata:
    repository: str
    commit_hash: str | None
    chunk_count: int
    corpus_fingerprint: str
    vector_fingerprint: str
    bm25_fingerprint: str
    vector_method_version: str
    vector_config: dict[str, Any]
    bm25_method_version: str
    bm25_config: dict[str, Any]
    method_name: str
    rrf_version: str
    rrf_k: int
    hybrid_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HybridTimings:
    vector_ranking_seconds: float
    bm25_ranking_seconds: float
    rrf_fusion_seconds: float
    total_seconds: float


@dataclass(frozen=True, slots=True)
class HybridSearchOutcome:
    results: tuple[HybridSearchResult, ...]
    timings: HybridTimings
    vector_ranking: tuple[Any, ...]
    bm25_ranking: tuple[Any, ...]


def corpus_fingerprint(chunks: Iterable[Chunk]) -> str:
    """Identify an ordered Day 3 corpus independently of retrieval formatting."""

    ordered = tuple(chunks)
    repository = ordered[0].repository if ordered else None
    commit_hash = ordered[0].commit_hash if ordered else None
    digest = hashlib.sha256()
    digest.update(
        json.dumps([repository, commit_hash], separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")
    for chunk in ordered:
        digest.update(
            json.dumps([chunk.chunk_id, chunk.content_hash], separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_index_compatibility(
    vector_index: EmbeddingIndex, bm25_index: BM25Index
) -> str:
    vector_metadata = vector_index.metadata
    bm25_metadata = bm25_index.metadata
    if vector_metadata.repository != bm25_metadata.repository:
        raise HybridCompatibilityError("Vector/BM25 repository mismatch")
    if vector_metadata.commit_hash != bm25_metadata.commit_hash:
        raise HybridCompatibilityError("Vector/BM25 commit hash mismatch")
    if vector_metadata.chunk_count != bm25_metadata.chunk_count:
        raise HybridCompatibilityError("Vector/BM25 chunk count mismatch")
    if vector_metadata.ordered_chunk_ids != bm25_metadata.ordered_chunk_ids:
        raise HybridCompatibilityError("Vector/BM25 ordered chunk IDs mismatch")
    if len(vector_index.chunks) != vector_metadata.chunk_count:
        raise HybridCompatibilityError("Vector index chunk count contradicts its metadata")
    if len(bm25_index.chunks) != bm25_metadata.chunk_count:
        raise HybridCompatibilityError("BM25 index chunk count contradicts its metadata")
    vector_corpus = corpus_fingerprint(vector_index.chunks)
    bm25_corpus = corpus_fingerprint(bm25_index.chunks)
    if vector_corpus != bm25_corpus:
        raise HybridCompatibilityError("Vector/BM25 source Day 3 corpus fingerprint mismatch")
    if vector_metadata.embedding_config != DEFAULT_EMBEDDING_CONFIG:
        raise HybridCompatibilityError("Vector index does not use the frozen Method A configuration")
    if bm25_metadata.config != DEFAULT_BM25_CONFIG:
        raise HybridCompatibilityError("BM25 index does not use the frozen Method B configuration")
    return vector_corpus


def hybrid_fingerprint(
    *,
    repository: str,
    commit_hash: str | None,
    corpus_identity: str,
    vector_fingerprint: str,
    bm25_fingerprint: str,
    config: HybridConfig = DEFAULT_HYBRID_CONFIG,
) -> str:
    payload = [
        repository,
        commit_hash,
        corpus_identity,
        vector_fingerprint,
        bm25_fingerprint,
        config.method_name,
        config.rrf_version,
        config.rrf_k,
        config.vector_method_version,
        config.bm25_method_version,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def create_hybrid_metadata(
    vector_index: EmbeddingIndex,
    bm25_index: BM25Index,
    *,
    config: HybridConfig = DEFAULT_HYBRID_CONFIG,
) -> HybridMetadata:
    corpus_identity = validate_index_compatibility(vector_index, bm25_index)
    vector_metadata = vector_index.metadata
    bm25_metadata = bm25_index.metadata
    fingerprint = hybrid_fingerprint(
        repository=vector_metadata.repository,
        commit_hash=vector_metadata.commit_hash,
        corpus_identity=corpus_identity,
        vector_fingerprint=vector_metadata.source_fingerprint,
        bm25_fingerprint=bm25_metadata.source_fingerprint,
        config=config,
    )
    return HybridMetadata(
        repository=vector_metadata.repository,
        commit_hash=vector_metadata.commit_hash,
        chunk_count=vector_metadata.chunk_count,
        corpus_fingerprint=corpus_identity,
        vector_fingerprint=vector_metadata.source_fingerprint,
        bm25_fingerprint=bm25_metadata.source_fingerprint,
        vector_method_version=config.vector_method_version,
        vector_config=vector_metadata.embedding_config.to_dict(),
        bm25_method_version=config.bm25_method_version,
        bm25_config=bm25_metadata.config.to_dict(),
        method_name=config.method_name,
        rrf_version=config.rrf_version,
        rrf_k=config.rrf_k,
        hybrid_fingerprint=fingerprint,
    )


def save_hybrid_metadata(metadata: HybridMetadata, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "metadata.json"
    destination.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination


def search(
    vector_index: EmbeddingIndex,
    bm25_index: BM25Index,
    model: EmbeddingModel,
    query: str,
    *,
    top_k: int = 5,
    config: HybridConfig = DEFAULT_HYBRID_CONFIG,
) -> HybridSearchOutcome:
    """Rank both complete indexes independently, fuse, then apply final top-k."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    validate_index_compatibility(vector_index, bm25_index)
    total_started = perf_counter()

    vector_started = perf_counter()
    vector_results = vector_search(
        vector_index, model, query, top_k=vector_index.metadata.chunk_count
    )
    vector_seconds = perf_counter() - vector_started

    bm25_started = perf_counter()
    bm25_results = bm25_search(
        bm25_index, query, top_k=bm25_index.metadata.chunk_count
    )
    bm25_seconds = perf_counter() - bm25_started

    fusion_started = perf_counter()
    fused = reciprocal_rank_fusion(
        (RankedItem(item.chunk_id, item.rank, item.score) for item in vector_results),
        (RankedItem(item.chunk_id, item.rank, item.score) for item in bm25_results),
        k=config.rrf_k,
        top_k=top_k,
    )
    chunks_by_id = {chunk.chunk_id: chunk for chunk in vector_index.chunks}
    results = tuple(
        HybridSearchResult(
            rank=item.rank,
            chunk_id=item.chunk_id,
            rrf_score=item.rrf_score,
            vector_rank=item.vector_rank,
            vector_raw_score=item.vector_raw_score,
            vector_contribution=item.vector_contribution,
            bm25_rank=item.bm25_rank,
            bm25_raw_score=item.bm25_raw_score,
            bm25_contribution=item.bm25_contribution,
            repository=chunks_by_id[item.chunk_id].repository,
            file_path=chunks_by_id[item.chunk_id].file_path,
            language=chunks_by_id[item.chunk_id].language,
            chunk_type=chunks_by_id[item.chunk_id].chunk_type,
            symbol_name=chunks_by_id[item.chunk_id].symbol_name,
            parent_symbol=chunks_by_id[item.chunk_id].parent_symbol,
            start_line=chunks_by_id[item.chunk_id].start_line,
            end_line=chunks_by_id[item.chunk_id].end_line,
            content=chunks_by_id[item.chunk_id].content,
        )
        for item in fused
    )
    fusion_seconds = perf_counter() - fusion_started
    total_seconds = perf_counter() - total_started
    return HybridSearchOutcome(
        results=results,
        timings=HybridTimings(
            vector_ranking_seconds=vector_seconds,
            bm25_ranking_seconds=bm25_seconds,
            rrf_fusion_seconds=fusion_seconds,
            total_seconds=total_seconds,
        ),
        vector_ranking=vector_results,
        bm25_ranking=bm25_results,
    )
