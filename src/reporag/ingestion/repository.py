"""Repository source resolution, discovery, and metadata collection."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .filters import (
    DEFAULT_MAX_FILE_SIZE_BYTES,
    appears_binary,
    classify_supported_file,
    is_excluded_directory,
    normalized_relative_path,
)
from .models import FileMetadata, IngestionManifest, RepositoryMetadata, SkippedFile


class IngestionError(RuntimeError):
    """Raised when a repository source cannot be prepared for ingestion."""


def _run_git(arguments: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IngestionError(f"Git command failed: {exc}") from exc


def is_github_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "github.com",
        "www.github.com",
    }


def _github_identity(source: str) -> tuple[str, str, str]:
    if not is_github_url(source):
        raise IngestionError("Remote source must be a public GitHub HTTP(S) URL")
    parsed = urlparse(source)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2:
        raise IngestionError("GitHub source must identify one owner and repository")
    owner, repository = parts
    if repository.lower().endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository:
        raise IngestionError("GitHub source must identify one owner and repository")
    canonical = f"https://github.com/{owner}/{repository}.git"
    return owner, repository, canonical


def _normalize_git_url(value: str) -> str:
    return value.strip().rstrip("/").removesuffix(".git").lower()


def prepare_github_repository(source: str, cache_directory: Path) -> tuple[Path, str]:
    """Clone a public GitHub repository or reuse a matching clean cached clone."""

    owner, repository, canonical_url = _github_identity(source)
    cache_directory.mkdir(parents=True, exist_ok=True)
    local_path = (cache_directory / f"{owner.lower()}--{repository.lower()}").resolve()

    if local_path.exists():
        inside = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=local_path)
        origin = _run_git(["remote", "get-url", "origin"], cwd=local_path)
        status = _run_git(["status", "--porcelain"], cwd=local_path)
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            raise IngestionError(f"Cache path is not a Git repository: {local_path}")
        if origin.returncode != 0 or _normalize_git_url(origin.stdout) != _normalize_git_url(canonical_url):
            raise IngestionError(f"Cache path belongs to a different Git remote: {local_path}")
        if status.returncode != 0 or status.stdout.strip():
            raise IngestionError(f"Cached repository has local changes and cannot be reused: {local_path}")
        return local_path, repository

    clone = _run_git(["clone", "--quiet", canonical_url, str(local_path)])
    if clone.returncode != 0:
        message = clone.stderr.strip() or clone.stdout.strip() or "unknown Git error"
        raise IngestionError(f"Unable to clone public GitHub repository: {message}")
    return local_path, repository


def git_metadata(repository_path: Path) -> tuple[str | None, str | None]:
    """Return the current commit and branch, tolerating non-Git directories."""

    try:
        root_result = _run_git(["rev-parse", "--show-toplevel"], cwd=repository_path)
    except IngestionError:
        return None, None
    if root_result.returncode != 0 or not root_result.stdout.strip():
        return None, None
    detected_root = Path(root_result.stdout.strip()).resolve()
    if os.path.normcase(str(detected_root)) != os.path.normcase(str(repository_path.resolve())):
        return None, None

    try:
        commit_result = _run_git(["rev-parse", "--verify", "HEAD"], cwd=repository_path)
    except IngestionError:
        return None, None
    if commit_result.returncode != 0:
        return None, None

    commit_hash = commit_result.stdout.strip() or None
    try:
        branch_result = _run_git(
            ["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=repository_path
        )
    except IngestionError:
        return commit_hash, None
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    return commit_hash, branch or None


def discover_files(
    repository_path: Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[tuple[FileMetadata, ...], tuple[SkippedFile, ...]]:
    """Discover safe UTF-8 source/config/test files beneath a repository root."""

    accepted: list[FileMetadata] = []
    skipped: list[SkippedFile] = []

    def record_walk_error(error: OSError) -> None:
        error_path = Path(error.filename) if error.filename else repository_path
        try:
            relative_text = normalized_relative_path(error_path.relative_to(repository_path))
        except ValueError:
            relative_text = error_path.name
        skipped.append(
            SkippedFile(relative_text or ".", f"unreadable_directory: {error.__class__.__name__}")
        )

    for current_root, directory_names, file_names in os.walk(
        repository_path, followlinks=False, onerror=record_walk_error
    ):
        current_path = Path(current_root)
        kept_directories: list[str] = []
        for directory_name in sorted(directory_names, key=str.casefold):
            candidate = current_path / directory_name
            relative = candidate.relative_to(repository_path)
            if candidate.is_symlink() or is_excluded_directory(relative):
                continue
            kept_directories.append(directory_name)
        directory_names[:] = kept_directories

        for file_name in sorted(file_names, key=str.casefold):
            absolute_path = current_path / file_name
            relative_path = absolute_path.relative_to(repository_path)
            relative_text = normalized_relative_path(relative_path)

            if absolute_path.is_symlink():
                skipped.append(SkippedFile(relative_text, "symbolic_link"))
                continue

            classification = classify_supported_file(relative_path)
            if classification is None:
                skipped.append(SkippedFile(relative_text, "unsupported_extension"))
                continue

            try:
                size_bytes = absolute_path.stat().st_size
            except OSError as exc:
                skipped.append(SkippedFile(relative_text, f"unreadable: {exc.__class__.__name__}"))
                continue

            if size_bytes > max_file_size_bytes:
                skipped.append(
                    SkippedFile(relative_text, f"exceeds_size_limit:{max_file_size_bytes}")
                )
                continue

            try:
                content = absolute_path.read_bytes()
            except OSError as exc:
                skipped.append(SkippedFile(relative_text, f"unreadable: {exc.__class__.__name__}"))
                continue

            if appears_binary(content):
                skipped.append(SkippedFile(relative_text, "binary_content"))
                continue
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                skipped.append(SkippedFile(relative_text, "not_utf8"))
                continue

            extension, language = classification
            accepted.append(FileMetadata(relative_text, extension, language, size_bytes))

    accepted.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path))
    skipped.sort(key=lambda item: (item.relative_path.casefold(), item.relative_path, item.reason))
    return tuple(accepted), tuple(skipped)


def ingest_repository(
    source: str | Path,
    *,
    cache_directory: Path = Path("data/repository_cache"),
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> IngestionManifest:
    """Ingest a local directory or public GitHub URL into an in-memory manifest."""

    original_source = str(source)
    parsed_source = urlparse(original_source)
    if parsed_source.scheme in {"http", "https"}:
        if not is_github_url(original_source):
            raise IngestionError("Remote source must be a public GitHub HTTP(S) URL")
        repository_path, repository_name = prepare_github_repository(
            original_source, cache_directory
        )
        source_type = "github"
    else:
        repository_path = Path(source).expanduser().resolve()
        if not repository_path.is_dir():
            raise IngestionError(f"Local repository directory does not exist: {repository_path}")
        repository_name = repository_path.name
        source_type = "local"

    commit_hash, branch = git_metadata(repository_path)
    files, skipped_files = discover_files(
        repository_path, max_file_size_bytes=max_file_size_bytes
    )
    metadata = RepositoryMetadata(
        name=repository_name,
        original_source=original_source,
        source_type=source_type,
        local_path=str(repository_path),
        commit_hash=commit_hash,
        branch=branch,
        accepted_file_count=len(files),
        skipped_file_count=len(skipped_files),
        ingestion_timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    return IngestionManifest(metadata, files, skipped_files)
