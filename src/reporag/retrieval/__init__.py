"""Exact vector retrieval over frozen RepoRAG chunks."""

from .models import SearchResult
from .vector import VectorSearchError, search

__all__ = ["SearchResult", "VectorSearchError", "search"]
