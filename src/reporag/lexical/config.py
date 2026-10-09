"""Frozen Method B configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


TOKENIZER_VERSION = "reporag-code-tokenizer-v1"
FORMATTER_VERSION = "reporag-bm25-document-v1"


@dataclass(frozen=True, slots=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    tokenizer_version: str = TOKENIZER_VERSION
    formatter_version: str = FORMATTER_VERSION

    def __post_init__(self) -> None:
        if self.k1 <= 0.0:
            raise ValueError("BM25 k1 must be positive")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError("BM25 b must be between zero and one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BM25Config":
        return cls(
            k1=float(value["k1"]),
            b=float(value["b"]),
            tokenizer_version=str(value["tokenizer_version"]),
            formatter_version=str(value["formatter_version"]),
        )


DEFAULT_BM25_CONFIG = BM25Config()
