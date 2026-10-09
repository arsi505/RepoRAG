"""DeepSeek Responses API provider without external tools."""

from __future__ import annotations

import os
from collections.abc import Callable
from time import perf_counter
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from reporag.qa.config import DEEPSEEK_MODEL, MAX_OUTPUT_TOKENS

from .protocol import GenerationError, GenerationResult


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekProvider:
    """Generate grounded repository answers through DeepSeek's Responses API."""

    def __init__(
        self,
        *,
        model: str = DEEPSEEK_MODEL,
        max_output_tokens: int = MAX_OUTPUT_TOKENS,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not model:
            raise ValueError("DeepSeek model is required")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client_factory = client_factory or OpenAI

    def generate(self, instructions: str, user_input: str) -> GenerationResult:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise GenerationError(
                "DEEPSEEK_API_KEY is not set; configure it before live generation"
            )

        try:
            client = self._client_factory(
                api_key=api_key,
                base_url=DEEPSEEK_BASE_URL,
            )
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
            raise GenerationError(
                "DeepSeek authentication failed; check DEEPSEEK_API_KEY"
            ) from exc
        except RateLimitError as exc:
            raise GenerationError("DeepSeek rate limit reached; retry later") from exc
        except APITimeoutError as exc:
            raise GenerationError("DeepSeek request timed out") from exc
        except APIConnectionError as exc:
            raise GenerationError("Unable to connect to the DeepSeek API") from exc
        except APIStatusError as exc:
            if exc.status_code == 402:
                raise GenerationError(
                    "DeepSeek account has insufficient balance"
                ) from exc
            raise GenerationError(
                f"DeepSeek API request failed (HTTP {exc.status_code})"
            ) from exc
        except APIError as exc:
            raise GenerationError("DeepSeek API request failed") from exc
        except (OSError, ValueError) as exc:
            raise GenerationError("DeepSeek provider request failed") from exc

        status = getattr(response, "status", None)
        if status != "completed":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            if status == "incomplete" and reason:
                raise GenerationError(f"DeepSeek response was incomplete ({reason})")
            raise GenerationError(f"DeepSeek response did not complete (status={status})")

        text = (response.output_text or "").strip()
        if not text:
            raise GenerationError("DeepSeek returned an empty text response")
        usage = response.usage.model_dump(mode="json") if response.usage is not None else None
        return GenerationResult(
            text=text,
            provider="deepseek",
            model=response.model,
            response_id=response.id,
            usage=usage,
            latency_seconds=elapsed,
        )
