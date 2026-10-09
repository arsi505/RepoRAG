"""Vendor-neutral generation interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class GenerationError(RuntimeError):
    """Raised for actionable provider failures."""


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    provider: str
    model: str
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    latency_seconds: float | None = None


class GeneratorProvider(Protocol):
    def generate(self, instructions: str, user_input: str) -> GenerationResult: ...
