"""Deterministic Method D metadata and configuration fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from reporag.retrieval.hybrid import HybridMetadata

from .config import DEFAULT_RERANKER_CONFIG, RerankerConfig


@dataclass(frozen=True, slots=True)
class RerankingMetadata:
    repository: str
    commit_hash: str | None
    corpus_fingerprint: str
    hybrid_fingerprint: str
    candidate_method: str
    candidate_k: int
    model_name: str
    model_revision: str
    model_license: str
    model_supported_max_length: int
    max_length: int
    formatter_version: str
    method_name: str
    method_version: str
    tie_policy: str
    batch_size: int
    reranking_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def reranking_fingerprint(
    *,
    repository: str,
    commit_hash: str | None,
    corpus_fingerprint: str,
    hybrid_fingerprint: str,
    config: RerankerConfig = DEFAULT_RERANKER_CONFIG,
) -> str:
    payload = [
        repository,
        commit_hash,
        corpus_fingerprint,
        hybrid_fingerprint,
        config.candidate_method,
        config.candidate_k,
        config.model_name,
        config.model_revision,
        config.max_length,
        config.formatter_version,
        config.method_version,
        config.tie_policy,
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def create_metadata(
    hybrid: HybridMetadata,
    *,
    config: RerankerConfig = DEFAULT_RERANKER_CONFIG,
) -> RerankingMetadata:
    fingerprint = reranking_fingerprint(
        repository=hybrid.repository,
        commit_hash=hybrid.commit_hash,
        corpus_fingerprint=hybrid.corpus_fingerprint,
        hybrid_fingerprint=hybrid.hybrid_fingerprint,
        config=config,
    )
    return RerankingMetadata(
        repository=hybrid.repository,
        commit_hash=hybrid.commit_hash,
        corpus_fingerprint=hybrid.corpus_fingerprint,
        hybrid_fingerprint=hybrid.hybrid_fingerprint,
        candidate_method=config.candidate_method,
        candidate_k=config.candidate_k,
        model_name=config.model_name,
        model_revision=config.model_revision,
        model_license=config.model_license,
        model_supported_max_length=config.model_supported_max_length,
        max_length=config.max_length,
        formatter_version=config.formatter_version,
        method_name=config.method_name,
        method_version=config.method_version,
        tie_policy=config.tie_policy,
        batch_size=config.batch_size,
        reranking_fingerprint=fingerprint,
    )


def save_metadata(metadata: RerankingMetadata, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "metadata.json"
    destination.write_text(
        json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return destination
