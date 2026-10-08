"""Data models for repository ingestion manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FileMetadata:
    relative_path: str
    extension: str
    language: str | None
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SkippedFile:
    relative_path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class RepositoryMetadata:
    name: str
    original_source: str
    source_type: str
    local_path: str
    commit_hash: str | None
    branch: str | None
    accepted_file_count: int
    skipped_file_count: int
    ingestion_timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IngestionManifest:
    repository: RepositoryMetadata
    files: tuple[FileMetadata, ...]
    skipped_files: tuple[SkippedFile, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": self.repository.to_dict(),
            "files": [item.to_dict() for item in self.files],
            "skipped_files": [item.to_dict() for item in self.skipped_files],
        }
