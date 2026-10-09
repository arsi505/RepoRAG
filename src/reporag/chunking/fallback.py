"""Deterministic line-based fallback and structural splitting helpers."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_FALLBACK_MAX_LINES = 80
DEFAULT_FALLBACK_OVERLAP_LINES = 10
DEFAULT_MAX_STRUCTURAL_LINES = 200


@dataclass(frozen=True, slots=True)
class TextPart:
    content: str
    start_line: int
    end_line: int
    part_index: int | None
    part_count: int


def split_lines(
    content: bytes,
    *,
    start_line: int = 1,
    max_lines: int,
    overlap_lines: int = 0,
    expose_parts: bool = False,
) -> tuple[TextPart, ...]:
    if max_lines <= 0:
        raise ValueError("max_lines must be greater than zero")
    if overlap_lines < 0 or overlap_lines >= max_lines:
        raise ValueError("overlap_lines must be non-negative and less than max_lines")

    lines = content.splitlines(keepends=True)
    if not lines:
        lines = [b""]

    windows: list[tuple[int, int]] = []
    offset = 0
    while offset < len(lines):
        end = min(offset + max_lines, len(lines))
        windows.append((offset, end))
        if end == len(lines):
            break
        offset = end - overlap_lines

    part_count = len(windows)
    parts: list[TextPart] = []
    for index, (window_start, window_end) in enumerate(windows, start=1):
        part_content = b"".join(lines[window_start:window_end]).decode("utf-8")
        part_start_line = start_line + window_start
        part_end_line = part_start_line + (window_end - window_start) - 1
        parts.append(
            TextPart(
                content=part_content,
                start_line=part_start_line,
                end_line=part_end_line,
                part_index=index if expose_parts and part_count > 1 else None,
                part_count=part_count if expose_parts else 1,
            )
        )
    return tuple(parts)
