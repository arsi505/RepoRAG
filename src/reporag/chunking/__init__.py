"""Code-aware parsing and deterministic source chunking."""

from .chunker import ChunkingConfig, chunk_repository
from .models import Chunk, ChunkingResult

__all__ = ["Chunk", "ChunkingConfig", "ChunkingResult", "chunk_repository"]
