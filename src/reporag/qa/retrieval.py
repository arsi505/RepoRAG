"""Read-only adapter over frozen Methods A-D."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from reporag.embeddings.index import EmbeddingIndex
from reporag.embeddings.model import SentenceTransformerEmbeddingModel
from reporag.lexical.bm25 import search as bm25_search
from reporag.lexical.index import BM25Index
from reporag.reranking.metadata import create_metadata as create_reranking_metadata
from reporag.reranking.model import CrossEncoderRerankerModel
from reporag.reranking.reranker import search as reranked_search
from reporag.retrieval.hybrid import create_hybrid_metadata, search as hybrid_search
from reporag.retrieval.vector import search as vector_search


class QARetriever(Protocol):
    def retrieve(self, question: str, method: str, top_k: int) -> tuple[Any, ...]: ...

    def fingerprint(self, method: str) -> str: ...


class RepositoryRetriever:
    def __init__(
        self,
        vector_index: EmbeddingIndex,
        bm25_index: BM25Index,
        *,
        model_cache: Path = Path("data/model_cache"),
    ) -> None:
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.model_cache = model_cache
        self.hybrid_metadata = create_hybrid_metadata(vector_index, bm25_index)
        self.reranking_metadata = create_reranking_metadata(self.hybrid_metadata)
        self._embedding_model = None
        self._reranker_model = None

    def _embedding(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformerEmbeddingModel(
                config=self.vector_index.metadata.embedding_config,
                cache_directory=self.model_cache,
                local_files_only=True,
            )
        return self._embedding_model

    def _reranker(self):
        if self._reranker_model is None:
            self._reranker_model = CrossEncoderRerankerModel(
                cache_directory=self.model_cache,
                local_files_only=True,
            )
        return self._reranker_model

    def retrieve(self, question: str, method: str, top_k: int) -> tuple[Any, ...]:
        if method == "vector":
            return vector_search(self.vector_index, self._embedding(), question, top_k=top_k)
        if method == "bm25":
            return bm25_search(self.bm25_index, question, top_k=top_k)
        if method == "hybrid":
            return hybrid_search(
                self.vector_index,
                self.bm25_index,
                self._embedding(),
                question,
                top_k=top_k,
            ).results
        if method == "reranked":
            return reranked_search(
                self.vector_index,
                self.bm25_index,
                self._embedding(),
                self._reranker(),
                question,
                final_top_k=top_k,
            ).results
        raise ValueError(f"Unsupported retrieval method: {method}")

    def fingerprint(self, method: str) -> str:
        values = {
            "vector": self.vector_index.metadata.source_fingerprint,
            "bm25": self.bm25_index.metadata.source_fingerprint,
            "hybrid": self.hybrid_metadata.hybrid_fingerprint,
            "reranked": self.reranking_metadata.reranking_fingerprint,
        }
        try:
            return values[method]
        except KeyError as exc:
            raise ValueError(f"Unsupported retrieval method: {method}") from exc
