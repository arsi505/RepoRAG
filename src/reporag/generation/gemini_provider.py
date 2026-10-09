"""Official Google Gen AI provider without external tools."""

from __future__ import annotations

import os
from collections.abc import Callable
from time import perf_counter
from typing import Any

from google import genai
from google.genai import errors, types

from reporag.qa.config import GEMINI_MODEL, MAX_OUTPUT_TOKENS

from .protocol import GenerationError, GenerationResult


def _error_message(error: Exception) -> str:
    """Classify provider failures without reflecting sensitive exception text."""
    code = getattr(error, "code", None)
    name = type(error).__name__.lower()
    if code in {401, 403} or "auth" in name or "permission" in name:
        return "Gemini authentication failed; check GEMINI_API_KEY"
    if code == 429 or "ratelimit" in name or "resourceexhausted" in name:
        return "Gemini rate limit reached; retry later"
    if "timeout" in name:
        return "Gemini request timed out"
    if isinstance(error, (ConnectionError, OSError)) or any(
        marker in name for marker in ("connect", "network", "transport")
    ):
        return "Unable to connect to the Gemini API"
    return "Gemini provider request failed"


class GeminiProvider:
    """Generate grounded text with Google's Gemini API."""

    def __init__(
        self,
        *,
        model: str = GEMINI_MODEL,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError("Gemini model is required")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client_factory = client_factory or genai.Client

    def generate(self, instructions: str, user_input: str) -> GenerationResult:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise GenerationError(
                "GEMINI_API_KEY is not set; configure it before live generation"
            )

        try:
            client = self._client_factory(api_key=api_key)
            started = perf_counter()
            response = client.models.generate_content(
                model=self.model,
                contents=user_input,
                config=types.GenerateContentConfig(
                    system_instruction=instructions,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            elapsed = perf_counter() - started
        except errors.APIError as exc:
            raise GenerationError(_error_message(exc)) from exc
        except Exception as exc:
            raise GenerationError(_error_message(exc)) from exc

        text = (response.text or "").strip()
        if not text:
            raise GenerationError("Gemini returned an empty text response")
        usage_metadata = getattr(response, "usage_metadata", None)
        usage = (
            usage_metadata.model_dump(mode="json", exclude_none=True)
            if usage_metadata is not None
            else None
        )
        return GenerationResult(
            text=text,
            provider="gemini",
            model=self.model,
            response_id=getattr(response, "response_id", None),
            usage=usage,
            latency_seconds=elapsed,
        )
