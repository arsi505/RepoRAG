"""Build, validate, and persist transparent Day 4 embedding indexes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from reporag.chunking.models import Chunk

from .config import EmbeddingConfig
from .formatter import format_chunk_for_embedding
from .model import EmbeddingModel, FloatMatrix, normalize_rows


class IndexErrorBase(RuntimeError):
    """Base class for clear index failures."""


class IndexMismatchError(IndexErrorBase):
    """Raised instead of silently reusing a stale or incompatible index."""


@dataclass(frozen=True, slots=True)
class IndexMetadata:
    repository: str
    commit_hash: str | None
    chunk_count: int
    ordered_chunk_ids: tuple[str, ...]
    source_fingerprint: str
    model_name: str
    model_revision: str
    embedding_dimension: int
    normalize_embeddings: bool
    text_format_version: str
    trusted_code_revision: str | None
    max_sequence_length: int
    maximum_observed_document_tokens: int
    documents_over_model_limit: int
    documents_truncated: int
    truncated_chunk_ids: tuple[str, ...]
    created_at_utc: str

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_name=self.model_name,
            model_revision=self.model_revision,
            embedding_dimension=self.embedding_dimension,
            normalize_embeddings=self.normalize_embeddings,
            text_format_version=self.text_format_version,
            trusted_code_revision=self.trusted_code_revision,
            max_sequence_length=self.max_sequence_length,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ordered_chunk_ids"] = list(self.ordered_chunk_ids)
        value["truncated_chunk_ids"] = list(self.truncated_chunk_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "IndexMetadata":
        try:
            return cls(
                repository=str(value["repository"]),
                commit_hash=value["commit_hash"],
                chunk_count=int(value["chunk_count"]),
                ordered_chunk_ids=tuple(str(item) for item in value["ordered_chunk_ids"]),
                source_fingerprint=str(value["source_fingerprint"]),
                model_name=str(value["model_name"]),
                model_revision=str(value["model_revision"]),
                embedding_dimension=int(value["embedding_dimension"]),
                normalize_embeddings=bool(value["normalize_embeddings"]),
                text_format_version=str(value["text_format_version"]),
                trusted_code_revision=value.get("trusted_code_revision"),
                max_sequence_length=int(value["max_sequence_length"]),
                maximum_observed_document_tokens=int(
                    value["maximum_observed_document_tokens"]
                ),
                documents_over_model_limit=int(value["documents_over_model_limit"]),
                documents_truncated=int(value["documents_truncated"]),
                truncated_chunk_ids=tuple(
                    str(item) for item in value["truncated_chunk_ids"]
                ),
                created_at_utc=str(value["created_at_utc"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexMismatchError(f"Invalid index metadata: {exc}") from exc


@dataclass(frozen=True, slots=True)
class EmbeddingIndex:
    metadata: IndexMetadata
    chunks: tuple[Chunk, ...]
    embeddings: FloatMatrix
    embedding_seconds: float | None = None
    save_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class IndexExpectation:
    repository: str | None = None
    commit_hash: str | None = None
    ordered_chunk_ids: tuple[str, ...] | None = None
    config: EmbeddingConfig | None = None


def source_fingerprint(chunks: Iterable[Chunk], config: EmbeddingConfig) -> str:
    digest = hashlib.sha256()
    header = [
        config.text_format_version,
        config.model_name,
        config.model_revision,
        config.trusted_code_revision,
        config.max_sequence_length,
        config.normalize_embeddings,
    ]
    digest.update(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    for chunk in chunks:
        digest.update(
            json.dumps(
                [chunk.chunk_id, chunk.content_hash],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_chunks(chunks: tuple[Chunk, ...]) -> tuple[str, str | None]:
    if not chunks:
        raise ValueError("Cannot build an embedding index from an empty chunk dataset")
    repositories = {chunk.repository for chunk in chunks}
    commits = {chunk.commit_hash for chunk in chunks}
    if len(repositories) != 1 or len(commits) != 1:
        raise ValueError("All chunks in an index must belong to one repository snapshot")
    identifiers = [chunk.chunk_id for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Chunk IDs must be unique")
    return next(iter(repositories)), next(iter(commits))


def _validate_matrix(
    matrix: NDArray[np.floating], chunk_count: int, dimension: int
) -> FloatMatrix:
    value = np.asarray(matrix, dtype=np.float32)
    expected_shape = (chunk_count, dimension)
    if value.shape != expected_shape:
        raise IndexMismatchError(
            f"Embedding matrix shape mismatch: expected {expected_shape}, got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise IndexMismatchError("Embedding matrix contains NaN or infinity")
    norms = np.linalg.norm(value, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise IndexMismatchError("Embedding matrix rows are not L2-normalized")
    return value


def build_index(chunks: Iterable[Chunk], model: EmbeddingModel) -> EmbeddingIndex:
    ordered_chunks = tuple(chunks)
    repository, commit_hash = _validate_chunks(ordered_chunks)
    texts = [format_chunk_for_embedding(chunk) for chunk in ordered_chunks]
    token_lengths = model.token_lengths(texts)
    if len(token_lengths) != len(ordered_chunks):
        raise ValueError(
            "Embedding model returned a token-length count that does not match the chunks"
        )
    if any(length <= 0 for length in token_lengths):
        raise ValueError("Embedding document token lengths must be positive")
    truncated_chunk_ids = tuple(
        chunk.chunk_id
        for chunk, token_count in zip(ordered_chunks, token_lengths, strict=True)
        if token_count > model.config.max_sequence_length
    )
    started = perf_counter()
    matrix = normalize_rows(model.encode_documents(texts))
    elapsed = perf_counter() - started
    if matrix.shape != (len(ordered_chunks), model.config.embedding_dimension):
        raise ValueError(
            "Embedding model returned the wrong shape: "
            f"expected {(len(ordered_chunks), model.config.embedding_dimension)}, "
            f"got {matrix.shape}"
        )
    metadata = IndexMetadata(
        repository=repository,
        commit_hash=commit_hash,
        chunk_count=len(ordered_chunks),
        ordered_chunk_ids=tuple(chunk.chunk_id for chunk in ordered_chunks),
        source_fingerprint=source_fingerprint(ordered_chunks, model.config),
        model_name=model.config.model_name,
        model_revision=model.config.model_revision,
        embedding_dimension=model.config.embedding_dimension,
        normalize_embeddings=model.config.normalize_embeddings,
        text_format_version=model.config.text_format_version,
        trusted_code_revision=model.config.trusted_code_revision,
        max_sequence_length=model.config.max_sequence_length,
        maximum_observed_document_tokens=max(token_lengths),
        documents_over_model_limit=len(truncated_chunk_ids),
        documents_truncated=len(truncated_chunk_ids),
        truncated_chunk_ids=truncated_chunk_ids,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    return EmbeddingIndex(metadata, ordered_chunks, matrix, embedding_seconds=elapsed)


def save_index(index: EmbeddingIndex, directory: Path) -> EmbeddingIndex:
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    matrix_path = directory / "embeddings.npy"
    metadata_path = directory / "metadata.json"
    chunks_path = directory / "chunks.jsonl"
    np.save(matrix_path, index.embeddings, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(index.metadata.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in index.chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    elapsed = perf_counter() - started
    return EmbeddingIndex(
        index.metadata,
        index.chunks,
        index.embeddings,
        embedding_seconds=index.embedding_seconds,
        save_seconds=elapsed,
    )


def load_chunks(path: Path) -> tuple[Chunk, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return tuple(Chunk(**json.loads(line)) for line in lines if line.strip())
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise IndexMismatchError(f"Unable to load chunk records from {path}: {exc}") from exc


def _check_expectation(metadata: IndexMetadata, expected: IndexExpectation) -> None:
    mismatches: list[str] = []
    if expected.repository is not None and metadata.repository != expected.repository:
        mismatches.append("repository identity")
    if expected.commit_hash is not None and metadata.commit_hash != expected.commit_hash:
        mismatches.append("commit hash")
    if (
        expected.ordered_chunk_ids is not None
        and metadata.ordered_chunk_ids != expected.ordered_chunk_ids
    ):
        mismatches.append("ordered chunk IDs")
    if expected.config is not None:
        actual = metadata.embedding_config
        if actual.model_name != expected.config.model_name:
            mismatches.append("embedding model")
        if actual.model_revision != expected.config.model_revision:
            mismatches.append("model revision")
        if actual.embedding_dimension != expected.config.embedding_dimension:
            mismatches.append("embedding dimension")
        if actual.normalize_embeddings != expected.config.normalize_embeddings:
            mismatches.append("normalization policy")
        if actual.text_format_version != expected.config.text_format_version:
            mismatches.append("embedding text format")
        if actual.trusted_code_revision != expected.config.trusted_code_revision:
            mismatches.append("trusted model-code revision")
        if actual.max_sequence_length != expected.config.max_sequence_length:
            mismatches.append("maximum sequence length")
    if mismatches:
        raise IndexMismatchError(
            "Stale or incompatible embedding index (mismatch: "
            + ", ".join(mismatches)
            + "); regenerate it"
        )


def load_index(
    directory: Path, *, expected: IndexExpectation | None = None
) -> EmbeddingIndex:
    try:
        metadata = IndexMetadata.from_dict(
            json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        )
        chunks = load_chunks(directory / "chunks.jsonl")
        matrix = np.load(directory / "embeddings.npy", allow_pickle=False)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, IndexMismatchError):
            raise
        raise IndexMismatchError(f"Unable to load embedding index at {directory}: {exc}") from exc
    if metadata.chunk_count != len(chunks):
        raise IndexMismatchError("Chunk count does not match index metadata")
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    if chunk_ids != metadata.ordered_chunk_ids:
        raise IndexMismatchError("Stored chunks do not match ordered chunk IDs")
    if any(chunk.repository != metadata.repository for chunk in chunks):
        raise IndexMismatchError("Stored chunk repository does not match metadata")
    if any(chunk.commit_hash != metadata.commit_hash for chunk in chunks):
        raise IndexMismatchError("Stored chunk commit does not match metadata")
    if source_fingerprint(chunks, metadata.embedding_config) != metadata.source_fingerprint:
        raise IndexMismatchError("Embedding source fingerprint does not match stored chunks")
    if metadata.documents_over_model_limit != len(metadata.truncated_chunk_ids):
        raise IndexMismatchError("Document-over-limit count does not match truncated chunk IDs")
    if metadata.documents_truncated != len(metadata.truncated_chunk_ids):
        raise IndexMismatchError("Truncation count does not match truncated chunk IDs")
    validated = _validate_matrix(
        matrix, metadata.chunk_count, metadata.embedding_dimension
    )
    if expected is not None:
        _check_expectation(metadata, expected)
    return EmbeddingIndex(metadata, chunks, validated)
