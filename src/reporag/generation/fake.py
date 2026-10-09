"""Deterministic offline generator for tests and mechanical demonstrations."""

from __future__ import annotations

from .protocol import GenerationError, GenerationResult


class FakeGenerator:
    def __init__(
        self,
        text: str = "The supplied evidence supports this answer. [S1]",
        *,
        error: str | None = None,
    ) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, instructions: str, user_input: str) -> GenerationResult:
        self.calls.append((instructions, user_input))
        if self.error is not None:
            raise GenerationError(self.error)
        return GenerationResult(
            text=self.text,
            provider="fake",
            model="deterministic-fake-v1",
            response_id="fake-response",
            usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            latency_seconds=0.0,
        )
