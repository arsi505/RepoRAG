"""Versioned RepoRAG Day 8 Q&A configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


QA_VERSION = "reporag-qa-v1"
PROMPT_VERSION = "reporag-qa-prompt-v1"
CONTEXT_FORMATTER_VERSION = "reporag-qa-context-v1"
DEFAULT_RETRIEVAL_METHOD = "hybrid"
DEFAULT_CONTEXT_K = 5
DEFAULT_EVIDENCE_SCOPE = "all"
DEFAULT_GENERATION_PROVIDER = "deepseek"
DEEPSEEK_MODEL = "deepseek-flash"
GEMINI_MODEL = "gemini-3.8-flash"
OPENAI_MODEL = "gpt-5.4-mini-2026-03-17"
GENERATION_PROVIDER = DEFAULT_GENERATION_PROVIDER
GENERATION_MODEL = DEEPSEEK_MODEL
MAX_OUTPUT_TOKENS = 2000
SUPPORTED_METHODS = frozenset({"vector", "bm25", "hybrid", "reranked"})
SUPPORTED_EVIDENCE_SCOPES = frozenset({"all", "code", "docs"})
SUPPORTED_PROVIDERS = frozenset({"deepseek", "gemini", "openai"})
PROVIDER_MODELS = {
    "deepseek": DEEPSEEK_MODEL,
    "gemini": GEMINI_MODEL,
    "openai": OPENAI_MODEL,
}


def model_for_provider(provider: str) -> str:
    try:
        return PROVIDER_MODELS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported generation provider: {provider}") from exc


@dataclass(frozen=True, slots=True)
class QAConfig:
    qa_version: str = QA_VERSION
    prompt_version: str = PROMPT_VERSION
    context_formatter_version: str = CONTEXT_FORMATTER_VERSION
    default_retrieval_method: str = DEFAULT_RETRIEVAL_METHOD
    default_context_k: int = DEFAULT_CONTEXT_K
    evidence_scope: str = DEFAULT_EVIDENCE_SCOPE
    generation_provider: str = GENERATION_PROVIDER
    generation_model: str = GENERATION_MODEL
    max_output_tokens: int = MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if self.default_retrieval_method not in SUPPORTED_METHODS:
            raise ValueError("Unsupported default retrieval method")
        if self.default_context_k <= 0:
            raise ValueError("default context_k must be positive")
        if self.evidence_scope not in SUPPORTED_EVIDENCE_SCOPES:
            raise ValueError("Unsupported evidence scope")
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.generation_provider not in SUPPORTED_PROVIDERS:
            raise ValueError("Unsupported default generation provider")
        if not self.generation_model:
            raise ValueError("generation_model must not be empty")

    def validate_request(self, method: str, context_k: int) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported retrieval method: {method}")
        if context_k <= 0:
            raise ValueError("context_k must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_QA_CONFIG = QAConfig()


def qa_fingerprint(
    *,
    retrieval_method: str,
    retrieval_fingerprint: str,
    context_k: int,
    config: QAConfig = DEFAULT_QA_CONFIG,
    generation_provider: str | None = None,
    generation_model: str | None = None,
) -> str:
    config.validate_request(retrieval_method, context_k)
    provider = generation_provider or config.generation_provider
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported generation provider: {provider}")
    model = generation_model or (
        config.generation_model
        if provider == config.generation_provider
        else model_for_provider(provider)
    )
    if not model:
        raise ValueError("generation_model must not be empty")
    payload = [
        config.qa_version,
        config.prompt_version,
        config.context_formatter_version,
        retrieval_method,
        retrieval_fingerprint,
        context_k,
        config.evidence_scope,
        provider,
        model,
        config.max_output_tokens,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
