"""Transparent persistent BM25 index over frozen Day 3 chunks."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from reporag.chunking.models import Chunk

from .config import BM25Config, DEFAULT_BM25_CONFIG
from .formatter import format_chunk_for_bm25
from .tokenizer import tokenize


class BM25IndexError(RuntimeError):
    """Base class for lexical-index errors."""


class BM25IndexMismatchError(BM25IndexError):
    """Raised instead of reusing a stale lexical index."""


@dataclass(frozen=True, slots=True)
class BM25IndexMetadata:
    repository: str
    commit_hash: str | None
    chunk_count: int
    ordered_chunk_ids: tuple[str, ...]
    source_fingerprint: str
    tokenizer_version: str
    formatter_version: str
    k1: float
    b: float
    total_tokens: int
    average_document_length: float
    minimum_document_length: int
    median_document_length: float
    maximum_document_length: int
    vocabulary_size: int
    created_at_utc: str

    @property
    def config(self) -> BM25Config:
        return BM25Config(
            k1=self.k1,
            b=self.b,
            tokenizer_version=self.tokenizer_version,
            formatter_version=self.formatter_version,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ordered_chunk_ids"] = list(self.ordered_chunk_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BM25IndexMetadata":
        try:
            return cls(
                repository=str(value["repository"]),
                commit_hash=value["commit_hash"],
                chunk_count=int(value["chunk_count"]),
                ordered_chunk_ids=tuple(str(item) for item in value["ordered_chunk_ids"]),
                source_fingerprint=str(value["source_fingerprint"]),
                tokenizer_version=str(value["tokenizer_version"]),
                formatter_version=str(value["formatter_version"]),
                k1=float(value["k1"]),
                b=float(value["b"]),
                total_tokens=int(value["total_tokens"]),
                average_document_length=float(value["average_document_length"]),
                minimum_document_length=int(value["minimum_document_length"]),
                median_document_length=float(value["median_document_length"]),
                maximum_document_length=int(value["maximum_document_length"]),
                vocabulary_size=int(value["vocabulary_size"]),
                created_at_utc=str(value["created_at_utc"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BM25IndexMismatchError(f"Invalid BM25 index metadata: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BM25Index:
    metadata: BM25IndexMetadata
    chunks: tuple[Chunk, ...]
    documents: tuple[tuple[str, ...], ...]
    document_frequencies: dict[str, int]
    inverse_document_frequencies: dict[str, float]
    build_seconds: float | None = None
    save_seconds: float | None = None
    load_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class BM25IndexExpectation:
    repository: str | None = None
    commit_hash: str | None = None
    ordered_chunk_ids: tuple[str, ...] | None = None
    config: BM25Config | None = None


def source_fingerprint(chunks: Iterable[Chunk], config: BM25Config) -> str:
    ordered = tuple(chunks)
    repository = ordered[0].repository if ordered else None
    commit_hash = ordered[0].commit_hash if ordered else None
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [
                repository,
                commit_hash,
                config.tokenizer_version,
                config.formatter_version,
                config.k1,
                config.b,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")
    for chunk in ordered:
        digest.update(
            json.dumps(
                [chunk.chunk_id, chunk.content_hash], separators=(",", ":")
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _corpus_statistics(
    documents: tuple[tuple[str, ...], ...]
) -> tuple[dict[str, int], dict[str, float], int, float, int, float, int, int]:
    if not documents:
        raise ValueError("Cannot build a BM25 index from an empty corpus")
    lengths = [len(document) for document in documents]
    if any(length == 0 for length in lengths):
        raise ValueError("BM25 documents must contain at least one token")
    frequencies: Counter[str] = Counter()
    for document in documents:
        frequencies.update(set(document))
    count = len(documents)
    idf = {
        term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
        for term, frequency in frequencies.items()
    }
    if not all(math.isfinite(value) and value > 0.0 for value in idf.values()):
        raise ValueError("BM25 IDF calculation produced an invalid value")
    total = sum(lengths)
    return (
        dict(frequencies),
        idf,
        total,
        total / count,
        min(lengths),
        float(statistics.median(lengths)),
        max(lengths),
        len(frequencies),
    )


def _validate_chunks(chunks: tuple[Chunk, ...]) -> tuple[str, str | None]:
    if not chunks:
        raise ValueError("Cannot build a BM25 index from an empty chunk dataset")
    repositories = {chunk.repository for chunk in chunks}
    commits = {chunk.commit_hash for chunk in chunks}
    if len(repositories) != 1 or len(commits) != 1:
        raise ValueError("All BM25 chunks must belong to one repository snapshot")
    identifiers = [chunk.chunk_id for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("BM25 chunk IDs must be unique")
    return next(iter(repositories)), next(iter(commits))


def build_index(
    chunks: Iterable[Chunk], *, config: BM25Config = DEFAULT_BM25_CONFIG
) -> BM25Index:
    started = perf_counter()
    ordered_chunks = tuple(chunks)
    repository, commit_hash = _validate_chunks(ordered_chunks)
    documents = tuple(tokenize(format_chunk_for_bm25(chunk)) for chunk in ordered_chunks)
    frequencies, idf, total, average, minimum, median, maximum, vocabulary = (
        _corpus_statistics(documents)
    )
    metadata = BM25IndexMetadata(
        repository=repository,
        commit_hash=commit_hash,
        chunk_count=len(ordered_chunks),
        ordered_chunk_ids=tuple(chunk.chunk_id for chunk in ordered_chunks),
        source_fingerprint=source_fingerprint(ordered_chunks, config),
        tokenizer_version=config.tokenizer_version,
        formatter_version=config.formatter_version,
        k1=config.k1,
        b=config.b,
        total_tokens=total,
        average_document_length=average,
        minimum_document_length=minimum,
        median_document_length=median,
        maximum_document_length=maximum,
        vocabulary_size=vocabulary,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    return BM25Index(
        metadata,
        ordered_chunks,
        documents,
        frequencies,
        idf,
        build_seconds=perf_counter() - started,
    )


def save_index(index: BM25Index, directory: Path) -> BM25Index:
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    (directory / "metadata.json").write_text(
        json.dumps(index.metadata.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with (directory / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in index.chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    with (directory / "corpus.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for chunk, document in zip(index.chunks, index.documents, strict=True):
            handle.write(
                json.dumps(
                    {"chunk_id": chunk.chunk_id, "tokens": document},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
    elapsed = perf_counter() - started
    return BM25Index(
        index.metadata,
        index.chunks,
        index.documents,
        index.document_frequencies,
        index.inverse_document_frequencies,
        build_seconds=index.build_seconds,
        save_seconds=elapsed,
    )


def _read_chunks(path: Path) -> tuple[Chunk, ...]:
    return tuple(
        Chunk(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _read_corpus(path: Path) -> tuple[tuple[str, tuple[str, ...]], ...]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            records.append(
                (str(value["chunk_id"]), tuple(str(token) for token in value["tokens"]))
            )
    return tuple(records)


def _check_expectation(metadata: BM25IndexMetadata, expected: BM25IndexExpectation) -> None:
    mismatches: list[str] = []
    if expected.repository is not None and metadata.repository != expected.repository:
        mismatches.append("repository identity")
    if expected.commit_hash is not None and metadata.commit_hash != expected.commit_hash:
        mismatches.append("commit hash")
    if expected.ordered_chunk_ids is not None and metadata.ordered_chunk_ids != expected.ordered_chunk_ids:
        mismatches.append("ordered chunk IDs")
    if expected.config is not None:
        actual = metadata.config
        if actual.tokenizer_version != expected.config.tokenizer_version:
            mismatches.append("tokenizer version")
        if actual.formatter_version != expected.config.formatter_version:
            mismatches.append("formatter version")
        if actual.k1 != expected.config.k1:
            mismatches.append("k1")
        if actual.b != expected.config.b:
            mismatches.append("b")
    if mismatches:
        raise BM25IndexMismatchError(
            "Stale or incompatible BM25 index (mismatch: "
            + ", ".join(mismatches)
            + "); regenerate it"
        )


def load_index(
    directory: Path, *, expected: BM25IndexExpectation | None = None
) -> BM25Index:
    started = perf_counter()
    try:
        metadata = BM25IndexMetadata.from_dict(
            json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        )
        chunks = _read_chunks(directory / "chunks.jsonl")
        corpus_records = _read_corpus(directory / "corpus.jsonl")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise BM25IndexMismatchError(f"Unable to load BM25 index at {directory}: {exc}") from exc
    if len(chunks) != metadata.chunk_count or len(corpus_records) != metadata.chunk_count:
        raise BM25IndexMismatchError("BM25 index chunk/corpus count mismatch")
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    corpus_ids = tuple(record[0] for record in corpus_records)
    if chunk_ids != metadata.ordered_chunk_ids or corpus_ids != metadata.ordered_chunk_ids:
        raise BM25IndexMismatchError("BM25 index ordered chunk IDs do not align")
    if any(chunk.repository != metadata.repository for chunk in chunks):
        raise BM25IndexMismatchError("BM25 chunk repository does not match metadata")
    if any(chunk.commit_hash != metadata.commit_hash for chunk in chunks):
        raise BM25IndexMismatchError("BM25 chunk commit does not match metadata")
    if source_fingerprint(chunks, metadata.config) != metadata.source_fingerprint:
        raise BM25IndexMismatchError("BM25 source/config fingerprint mismatch")
    documents = tuple(record[1] for record in corpus_records)
    frequencies, idf, total, average, minimum, median, maximum, vocabulary = (
        _corpus_statistics(documents)
    )
    expected_stats = (
        metadata.total_tokens,
        metadata.average_document_length,
        metadata.minimum_document_length,
        metadata.median_document_length,
        metadata.maximum_document_length,
        metadata.vocabulary_size,
    )
    actual_stats = (total, average, minimum, median, maximum, vocabulary)
    if not all(
        math.isclose(float(actual), float(stored), rel_tol=1e-12, abs_tol=1e-12)
        for actual, stored in zip(actual_stats, expected_stats, strict=True)
    ):
        raise BM25IndexMismatchError("BM25 corpus statistics do not match metadata")
    if expected is not None:
        _check_expectation(metadata, expected)
    return BM25Index(
        metadata,
        chunks,
        documents,
        frequencies,
        idf,
        load_seconds=perf_counter() - started,
    )
