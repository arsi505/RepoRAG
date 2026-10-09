"""Embedding-model abstraction and the frozen Jina implementation."""

from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_EMBEDDING_CONFIG, EmbeddingConfig


FloatMatrix = NDArray[np.float32]


class EmbeddingModel(Protocol):
    """Small protocol that keeps unit tests offline."""

    @property
    def config(self) -> EmbeddingConfig: ...

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix: ...

    def token_lengths(self, texts: Sequence[str]) -> tuple[int, ...]: ...


def normalize_rows(vectors: NDArray[np.floating]) -> FloatMatrix:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embedding output must be a two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding output contains NaN or infinity")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Embedding output contains a zero-length vector")
    normalized = matrix / norms
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized embeddings contain NaN or infinity")
    return np.asarray(normalized, dtype=np.float32)


class SentenceTransformerEmbeddingModel:
    """Pinned local adapter for jinaai/jina-embeddings-v2-base-code."""

    def __init__(
        self,
        *,
        config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
        cache_directory: Path = Path("data/model_cache"),
        batch_size: int = 16,
        show_progress: bool = False,
        local_files_only: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_directory = cache_directory.resolve()
        os.environ.setdefault("HF_HOME", str(cache_directory / "huggingface"))
        os.environ.setdefault("HF_MODULES_CACHE", str(cache_directory / "modules"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_directory))
        model_source = config.model_name
        revision = config.model_revision
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            cached_name = "models--" + config.model_name.replace("/", "--")
            snapshot = cache_directory / cached_name / "snapshots" / config.model_revision
            if not snapshot.is_dir():
                raise ValueError(
                    f"Pinned model snapshot is missing from local cache: {snapshot}"
                )
            model_source = str(snapshot)
            revision = None
        started = perf_counter()
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_source,
            revision=revision,
            cache_folder=str(cache_directory),
            trust_remote_code=True,
            local_files_only=local_files_only,
            model_kwargs={"code_revision": config.trusted_code_revision},
            config_kwargs={
                "code_revision": config.trusted_code_revision,
                "local_files_only": local_files_only,
            },
        )
        self.load_seconds = perf_counter() - started
        self._config = config
        self._batch_size = batch_size
        self._show_progress = show_progress
        tokenizer_limit = int(self._model.tokenizer.model_max_length)
        transformer = self._model[0].auto_model
        architecture_limit = int(transformer.config.max_position_embeddings)
        if tokenizer_limit < config.max_sequence_length:
            raise ValueError(
                "Tokenizer does not support configured maximum sequence length: "
                f"{tokenizer_limit} < {config.max_sequence_length}"
            )
        if architecture_limit < config.max_sequence_length:
            raise ValueError(
                "Model architecture does not support configured maximum sequence length: "
                f"{architecture_limit} < {config.max_sequence_length}"
            )
        self._model.max_seq_length = config.max_sequence_length
        self.tokenizer_supported_max = tokenizer_limit
        self.architecture_supported_max = architecture_limit
        actual_dimension = self._model.get_embedding_dimension()
        if actual_dimension != config.embedding_dimension:
            raise ValueError(
                "Model embedding dimension mismatch: "
                f"expected {config.embedding_dimension}, got {actual_dimension}"
            )

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def _encode(self, texts: Sequence[str]) -> FloatMatrix:
        values = list(texts)
        if not values:
            return np.empty((0, self.config.embedding_dimension), dtype=np.float32)
        encoded = self._model.encode(
            values,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=self._show_progress,
        )
        return normalize_rows(encoded)

    def encode_documents(self, texts: Sequence[str]) -> FloatMatrix:
        return self._encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> FloatMatrix:
        # The model card specifies no asymmetric task prefixes.
        return self._encode(texts)

    def token_lengths(self, texts: Sequence[str]) -> tuple[int, ...]:
        """Count tokenizer input IDs before model truncation."""

        return tuple(
            len(
                self._model.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for text in texts
        )
