"""Frozen Method D configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


METHOD_NAME = "hybrid_reranked"
METHOD_VERSION = "reporag-hybrid-reranker-v1"
CANDIDATE_METHOD = "hybrid_rrf"
CANDIDATE_K = 50
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_LICENSE = "Apache-2.0"
MODEL_SUPPORTED_MAX_LENGTH = 8192
MAX_LENGTH = 1024
FORMATTER_VERSION = "reporag-reranker-document-v1"
TIE_POLICY = "reranker-logit-desc-12dp_then-hybrid-rank-asc_then-chunk-id-asc-v1"
BATCH_SIZE = 4


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    method_name: str = METHOD_NAME
    method_version: str = METHOD_VERSION
    candidate_method: str = CANDIDATE_METHOD
    candidate_k: int = CANDIDATE_K
    model_name: str = MODEL_NAME
    model_revision: str = MODEL_REVISION
    model_license: str = MODEL_LICENSE
    model_supported_max_length: int = MODEL_SUPPORTED_MAX_LENGTH
    max_length: int = MAX_LENGTH
    formatter_version: str = FORMATTER_VERSION
    tie_policy: str = TIE_POLICY
    batch_size: int = BATCH_SIZE

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if not self.model_name:
            raise ValueError("Reranker model name is required")
        if not self.model_revision:
            raise ValueError("Immutable reranker model revision is required")
        if self.max_length <= 0:
            raise ValueError("Reranker max_length must be positive")
        if self.max_length > self.model_supported_max_length:
            raise ValueError("Reranker max_length exceeds the model-supported maximum")
        if not self.formatter_version:
            raise ValueError("Reranker formatter version is required")
        if self.batch_size <= 0:
            raise ValueError("Reranker batch size must be positive")

    def validate_final_top_k(self, final_top_k: int) -> None:
        if final_top_k <= 0:
            raise ValueError("final top-k must be positive")
        if final_top_k > self.candidate_k:
            raise ValueError("candidate_k must be greater than or equal to final top-k")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_RERANKER_CONFIG = RerankerConfig()
