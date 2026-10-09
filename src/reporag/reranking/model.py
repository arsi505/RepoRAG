"""Offline-testable reranker abstraction and pinned BGE CrossEncoder adapter."""

from __future__ import annotations

import math
import os
from pathlib import Path
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np

from .config import DEFAULT_RERANKER_CONFIG, RerankerConfig


class RerankerModel(Protocol):
    @property
    def config(self) -> RerankerConfig: ...

    def pair_token_lengths(self, pairs: Sequence[tuple[str, str]]) -> tuple[int, ...]: ...

    def document_token_lengths(self, documents: Sequence[str]) -> tuple[int, ...]: ...

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]: ...


class CrossEncoderRerankerModel:
    """Pinned raw-logit adapter for BAAI/bge-reranker-v2-m3."""

    def __init__(
        self,
        *,
        config: RerankerConfig = DEFAULT_RERANKER_CONFIG,
        cache_directory: Path = Path("data/model_cache"),
        local_files_only: bool = True,
        device: str | None = None,
    ) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        cache_directory = cache_directory.resolve()
        os.environ.setdefault("HF_HOME", str(cache_directory / "huggingface"))
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(cache_directory))
        source = config.model_name
        revision: str | None = config.model_revision
        if local_files_only:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            snapshot = (
                cache_directory
                / ("models--" + config.model_name.replace("/", "--"))
                / "snapshots"
                / config.model_revision
            )
            if not snapshot.is_dir():
                raise ValueError(f"Pinned reranker snapshot is missing: {snapshot}")
            source = str(snapshot)
            revision = None
        else:
            # An explicit CLI --allow-download must undo offline flags set by
            # an earlier local-only embedding-model load in the same process.
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)

        from sentence_transformers import CrossEncoder

        started = perf_counter()
        self._model = CrossEncoder(
            source,
            revision=revision,
            cache_folder=str(cache_directory),
            max_length=config.max_length,
            device=device,
            local_files_only=local_files_only,
        )
        self.load_seconds = perf_counter() - started
        self._config = config
        architecture_positions = int(self._model.model.config.max_position_embeddings)
        self.architecture_supported_max = architecture_positions - 2
        if self.architecture_supported_max < config.model_supported_max_length:
            raise ValueError(
                "Reranker architecture maximum is below the frozen supported maximum: "
                f"{self.architecture_supported_max} < {config.model_supported_max_length}"
            )
        if self._model.max_length != config.max_length:
            raise ValueError("CrossEncoder did not accept the configured max_length")
        self.device = str(self._model.model.device)
        self.dtype = str(next(self._model.model.parameters()).dtype).removeprefix("torch.")

    @property
    def config(self) -> RerankerConfig:
        return self._config

    def pair_token_lengths(self, pairs: Sequence[tuple[str, str]]) -> tuple[int, ...]:
        if not pairs:
            return ()
        encoded = self._model.tokenizer(
            list(pairs), add_special_tokens=True, truncation=False, padding=False
        )
        return tuple(len(ids) for ids in encoded["input_ids"])

    def document_token_lengths(self, documents: Sequence[str]) -> tuple[int, ...]:
        if not documents:
            return ()
        encoded = self._model.tokenizer(
            list(documents), add_special_tokens=True, truncation=False, padding=False
        )
        return tuple(len(ids) for ids in encoded["input_ids"])

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> tuple[float, ...]:
        if not pairs:
            return ()
        import torch

        values = self._model.predict(
            list(pairs),
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            activation_fn=torch.nn.Identity(),
            convert_to_numpy=True,
        )
        scores = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(scores) != len(pairs) or not np.isfinite(scores).all():
            raise ValueError("Cross-encoder returned invalid or non-finite raw logits")
        return tuple(float(score) for score in scores)


def validate_scores(scores: Sequence[float], expected: int) -> tuple[float, ...]:
    values = tuple(float(score) for score in scores)
    if len(values) != expected:
        raise ValueError("Reranker score count does not match candidate count")
    if not all(math.isfinite(score) for score in values):
        raise ValueError("Reranker scores must be finite")
    return values
