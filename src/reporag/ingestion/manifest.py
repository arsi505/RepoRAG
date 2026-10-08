"""Deterministic JSON manifest writing."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .models import IngestionManifest


def manifest_filename(repository_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9._-]+", "-", repository_name.lower()).strip("-.")
    return f"{normalized or 'repository'}.json"


def write_manifest(manifest: IngestionManifest, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    destination = output_directory / manifest_filename(manifest.repository.name)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination.resolve()
