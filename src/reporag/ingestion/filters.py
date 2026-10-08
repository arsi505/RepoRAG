"""Centralized file-selection policy for repository ingestion."""

from __future__ import annotations

from pathlib import Path


DEFAULT_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024

SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".c": "C",
    ".cc": "C++",
    ".cpp": "C++",
    ".cs": "C#",
    ".css": "CSS",
    ".go": "Go",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".html": "HTML",
    ".java": "Java",
    ".js": "JavaScript",
    ".json": "JSON",
    ".jsx": "JavaScript JSX",
    ".kt": "Kotlin",
    ".kts": "Kotlin Script",
    ".md": "Markdown",
    ".php": "PHP",
    ".ps1": "PowerShell",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scss": "SCSS",
    ".sh": "Shell",
    ".sql": "SQL",
    ".svelte": "Svelte",
    ".swift": "Swift",
    ".toml": "TOML",
    ".ts": "TypeScript",
    ".tsx": "TypeScript TSX",
    ".vue": "Vue",
    ".xml": "XML",
    ".yaml": "YAML",
    ".yml": "YAML",
}

SUPPORTED_FILENAMES: dict[str, str] = {
    "dockerfile": "Dockerfile",
    "makefile": "Makefile",
}

EXCLUDED_DIRECTORY_NAMES: frozenset[str] = frozenset(
    {
        ".cache",
        ".git",
        ".idea",
        ".mypy_cache",
        ".next",
        ".nuxt",
        ".pytest_cache",
        ".ruff_cache",
        ".test_tmp",
        ".venv",
        ".vscode",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "env",
        "node_modules",
        "out",
        "target",
        "temp",
        "tmp",
        "vendor",
        "venv",
    }
)

# RepoRAG-owned generated/cache paths should not become ingestion inputs.
EXCLUDED_RELATIVE_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".github/artifacts",
        ".github/cache",
        "data/manifests",
        "data/repository_cache",
    }
)


def normalized_relative_path(path: Path) -> str:
    """Return a platform-independent relative path for manifests."""

    return path.as_posix()


def is_excluded_directory(relative_path: Path) -> bool:
    """Return whether a directory should be pruned without traversal."""

    if relative_path.name.lower() in EXCLUDED_DIRECTORY_NAMES:
        return True
    return normalized_relative_path(relative_path).lower() in EXCLUDED_RELATIVE_DIRECTORIES


def classify_supported_file(path: Path) -> tuple[str, str | None] | None:
    """Return normalized extension and language, or None if unsupported."""

    filename_language = SUPPORTED_FILENAMES.get(path.name.lower())
    if filename_language is not None:
        return "", filename_language

    extension = path.suffix.lower()
    language = SUPPORTED_EXTENSIONS.get(extension)
    if language is None:
        return None
    return extension, language


def appears_binary(data: bytes) -> bool:
    """Detect common binary content without attempting arbitrary decoding."""

    return b"\x00" in data
