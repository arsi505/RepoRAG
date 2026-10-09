"""Central mapping from accepted files to Tree-sitter grammars."""

from __future__ import annotations

from dataclasses import dataclass

from reporag.ingestion.models import FileMetadata


@dataclass(frozen=True, slots=True)
class StructuralLanguage:
    key: str
    grammar: str


STRUCTURAL_LANGUAGES_BY_EXTENSION: dict[str, StructuralLanguage] = {
    ".c": StructuralLanguage("c", "c"),
    ".cc": StructuralLanguage("cpp", "cpp"),
    ".cpp": StructuralLanguage("cpp", "cpp"),
    ".h": StructuralLanguage("c", "c"),
    ".hpp": StructuralLanguage("cpp", "cpp"),
    ".js": StructuralLanguage("javascript", "javascript"),
    ".jsx": StructuralLanguage("javascript", "javascript"),
    ".py": StructuralLanguage("python", "python"),
    ".ts": StructuralLanguage("typescript", "typescript"),
    ".tsx": StructuralLanguage("tsx", "tsx"),
}


def structural_language_for(file_metadata: FileMetadata) -> StructuralLanguage | None:
    return STRUCTURAL_LANGUAGES_BY_EXTENSION.get(file_metadata.extension.lower())
