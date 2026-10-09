"""Frozen Day 4 embedding configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
MODEL_REVISION = "516f4baf13dec4ddddda8631e019b5737c8bc250"
TRUSTED_CODE_REVISION = "3baf9e3ac750e76e8edd3019170176884695fb94"
EMBEDDING_DIMENSION = 768
EMBEDDING_TEXT_FORMAT_VERSION = "reporag-embedding-text-v1"
MAX_SEQUENCE_LENGTH = 8192


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    model_name: str
    model_revision: str
    embedding_dimension: int
    normalize_embeddings: bool
    text_format_version: str
    trusted_code_revision: str | None = None
    max_sequence_length: int = MAX_SEQUENCE_LENGTH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EmbeddingConfig":
        return cls(
            model_name=str(value["model_name"]),
            model_revision=str(value["model_revision"]),
            embedding_dimension=int(value["embedding_dimension"]),
            normalize_embeddings=bool(value["normalize_embeddings"]),
            text_format_version=str(value["text_format_version"]),
            trusted_code_revision=value.get("trusted_code_revision"),
            max_sequence_length=int(value.get("max_sequence_length", MAX_SEQUENCE_LENGTH)),
        )


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    model_name=MODEL_NAME,
    model_revision=MODEL_REVISION,
    embedding_dimension=EMBEDDING_DIMENSION,
    normalize_embeddings=True,
    text_format_version=EMBEDDING_TEXT_FORMAT_VERSION,
    trusted_code_revision=TRUSTED_CODE_REVISION,
    max_sequence_length=MAX_SEQUENCE_LENGTH,
)
