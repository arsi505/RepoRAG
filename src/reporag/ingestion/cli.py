"""Command-line interface for repository ingestion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .filters import DEFAULT_MAX_FILE_SIZE_BYTES
from .manifest import write_manifest
from .repository import IngestionError, ingest_repository


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover repository files and write a deterministic JSON manifest."
    )
    parser.add_argument("source", help="Local directory or public GitHub repository URL")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data/manifests"),
        help="Manifest output directory (default: data/manifests)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/repository_cache"),
        help="GitHub clone cache directory (default: data/repository_cache)",
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=DEFAULT_MAX_FILE_SIZE_BYTES,
        help=f"Maximum accepted file size in bytes (default: {DEFAULT_MAX_FILE_SIZE_BYTES})",
    )
    return parser


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main(arguments: list[str] | None = None) -> int:
    parser = build_parser()
    options = parser.parse_args(arguments)
    if options.max_file_size < 0:
        parser.error("--max-file-size must be zero or greater")

    try:
        manifest = ingest_repository(
            options.source,
            cache_directory=options.cache_dir,
            max_file_size_bytes=options.max_file_size,
        )
        manifest_path = write_manifest(manifest, options.manifest_dir)
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Unable to write manifest: {exc}", file=sys.stderr)
        return 1

    repository = manifest.repository
    commit = repository.commit_hash or "unavailable"
    print(f"Repository: {repository.name}")
    print(f"Source type: {repository.source_type}")
    print(f"Commit: {commit}")
    print(f"Accepted files: {repository.accepted_file_count}")
    print(f"Skipped files: {repository.skipped_file_count}")
    print(f"Manifest: {_display_path(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
