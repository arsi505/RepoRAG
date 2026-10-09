"""Official OpenAI Responses API provider without external tools."""

from __future__ import annotations

import os
from time import perf_counter

from .protocol import GenerationError, GenerationResult


class OpenAIResponsesProvider:
    def __init__(self, *, model: str, max_output_tokens: int) -> None:
        if not model:
            raise ValueError("OpenAI model is required")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.model = model
        self.max_output_tokens = max_output_tokens

    def generate(self, instructions: str, user_input: str) -> GenerationResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise GenerationError(
                "OPENAI_API_KEY is not set; configure it in the environment before live generation"
            )
        try:
            from openai import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                AuthenticationError,
                OpenAI,
                RateLimitError,
            )

            client = OpenAI(api_key=api_key, timeout=60.0, max_retries=2)
            started = perf_counter()
            response = client.responses.create(
                model=self.model,
                instructions=instructions,
                input=user_input,
                max_output_tokens=self.max_output_tokens,
                tools=[],
                store=False,
            )
            elapsed = perf_counter() - started
        except AuthenticationError as exc:
            raise GenerationError("OpenAI authentication failed; check OPENAI_API_KEY") from exc
        except RateLimitError as exc:
            raise GenerationError("OpenAI rate limit reached; retry later") from exc
        except APITimeoutError as exc:
            raise GenerationError("OpenAI request timed out") from exc
        except APIConnectionError as exc:
            raise GenerationError("Unable to connect to the OpenAI API") from exc
        except APIError as exc:
            raise GenerationError(f"OpenAI API error: {exc}") from exc
        except (OSError, ValueError) as exc:
            raise GenerationError(f"OpenAI provider failed: {exc}") from exc

        text = (response.output_text or "").strip()
        if not text:
            raise GenerationError("OpenAI returned an empty text response")
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        return GenerationResult(
            text=text,
            provider="openai",
            model=response.model,
            response_id=response.id,
            usage=usage,
            latency_seconds=elapsed,
        )
