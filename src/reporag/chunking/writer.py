"""Deterministic JSONL output for runtime chunk datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import ChunkingResult


def chunk_filename(repository_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", repository_name.lower()).strip("-.")
    return f"{normalized or 'repository'}.jsonl"


def write_chunks(
    result: ChunkingResult,
    repository_name: str,
    output_directory: Path = Path("data/chunks"),
) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / chunk_filename(repository_name)
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in result.chunks:
            handle.write(
                json.dumps(chunk.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
            handle.write("\n")
    temporary.replace(destination)
    return destination.resolve()
