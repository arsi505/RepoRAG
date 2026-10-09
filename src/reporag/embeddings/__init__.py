"""Deterministic chunk embeddings and persistent exact-search indexes."""

from .config import DEFAULT_EMBEDDING_CONFIG, EmbeddingConfig
from .formatter import format_chunk_for_embedding
from .index import EmbeddingIndex, IndexMismatchError, build_index, load_index, save_index

__all__ = [
    "DEFAULT_EMBEDDING_CONFIG",
    "EmbeddingConfig",
    "EmbeddingIndex",
    "IndexMismatchError",
    "build_index",
    "format_chunk_for_embedding",
    "load_index",
    "save_index",
]
