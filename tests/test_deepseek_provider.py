from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from reporag.generation.deepseek_provider import (
    DEEPSEEK_BASE_URL,
    DeepSeekProvider,
)
from reporag.generation.fake import FakeGenerator
from reporag.generation.gemini_provider import GeminiProvider
from reporag.generation.openai_provider import OpenAIResponsesProvider
from reporag.generation.protocol import GenerationError
from reporag.qa.cli import build_parser, create_generator
from reporag.qa.config import DEEPSEEK_MODEL
from reporag.qa.prompts import GROUNDING_INSTRUCTIONS, build_user_input


class Usage:
    def model_dump(self, **_kwargs):
        return {"input_tokens": 40, "output_tokens": 12, "total_tokens": 52}


class Response:
    def __init__(self, text="Grounded answer [S1].", status="completed"):
        self.output_text = text
        self.status = status
        self.model = DEEPSEEK_MODEL
        self.id = "deepseek-response"
        self.usage = Usage()
        self.incomplete_details = None


class RecordingResponses:
    def __init__(self, response=None, error=None):
        self.response = response or Response()
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class RecordingFactory:
    def __init__(self, responses=None):
        self.responses = responses or RecordingResponses()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(responses=self.responses)


class DeepSeekProviderOfflineTests(unittest.TestCase):
    def test_api_key_is_required_before_client_creation(self) -> None:
        factory = RecordingFactory()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GenerationError, "DEEPSEEK_API_KEY"):
                DeepSeekProvider(client_factory=factory).generate("instructions", "input")
        self.assertEqual(factory.calls, [])

    def test_default_model_and_output_limit(self) -> None:
        provider = DeepSeekProvider(client_factory=RecordingFactory())
        self.assertEqual(provider.model, "deepseek-flash")
        self.assertEqual(provider.max_output_tokens, 1200)

    def test_client_uses_required_base_url(self) -> None:
        factory = RecordingFactory()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True):
            DeepSeekProvider(client_factory=factory).generate("instructions", "input")
        self.assertEqual(factory.calls[0]["base_url"], DEEPSEEK_BASE_URL)

    def test_shared_instructions_and_evidence_are_sent_without_tools(self) -> None:
        factory = RecordingFactory()
        evidence = build_user_input("Where?", "[S1]\nCode:\nanswer")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True):
            DeepSeekProvider(client_factory=factory).generate(
                GROUNDING_INSTRUCTIONS, evidence
            )
        call = factory.responses.calls[0]
        self.assertEqual(call["instructions"], GROUNDING_INSTRUCTIONS)
        self.assertEqual(call["input"], evidence)
        self.assertIn("[S1]", call["input"])
        self.assertEqual(call["tools"], [])
        self.assertFalse(call["store"])

    def test_generated_output_and_usage_are_returned(self) -> None:
        factory = RecordingFactory()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True):
            result = DeepSeekProvider(client_factory=factory).generate("instructions", "input")
        self.assertEqual(result.text, "Grounded answer [S1].")
        self.assertEqual((result.provider, result.model), ("deepseek", DEEPSEEK_MODEL))
        self.assertEqual(result.usage["total_tokens"], 52)

    def test_empty_and_incomplete_responses_are_rejected(self) -> None:
        incomplete = Response(status="incomplete")
        incomplete.incomplete_details = SimpleNamespace(reason="max_output_tokens")
        for response, expected in (
            (Response(text="  "), "empty text response"),
            (incomplete, "incomplete.*max_output_tokens"),
        ):
            factory = RecordingFactory(RecordingResponses(response))
            with self.subTest(expected=expected):
                with patch.dict(
                    os.environ, {"DEEPSEEK_API_KEY": "unit-test-key"}, clear=True
                ):
                    with self.assertRaisesRegex(GenerationError, expected):
                        DeepSeekProvider(client_factory=factory).generate(
                            "instructions", "input"
                        )

    def test_provider_error_is_sanitized(self) -> None:
        secret = "unit-test-key-that-must-not-leak"
        factory = RecordingFactory(RecordingResponses(error=ValueError(secret)))
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": secret}, clear=True):
            with self.assertRaisesRegex(GenerationError, "provider request failed") as caught:
                DeepSeekProvider(client_factory=factory).generate("instructions", "input")
        self.assertNotIn(secret, str(caught.exception))

    def test_api_key_is_not_in_result(self) -> None:
        secret = "unit-test-key-that-must-not-leak"
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": secret}, clear=True):
            result = DeepSeekProvider(client_factory=RecordingFactory()).generate(
                "instructions", "input"
            )
        self.assertFalse(hasattr(result, "api_key"))
        self.assertNotIn(secret, repr(result))

    def test_provider_selection_keeps_all_providers_and_fake(self) -> None:
        options = build_parser().parse_args(["vector", "bm25", "question"])
        self.assertEqual(options.provider, "deepseek")
        self.assertIsInstance(create_generator("deepseek", dry_run=False), DeepSeekProvider)
        self.assertIsInstance(create_generator("gemini", dry_run=False), GeminiProvider)
        self.assertIsInstance(
            create_generator("openai", dry_run=False), OpenAIResponsesProvider
        )
        self.assertIsInstance(create_generator("deepseek", dry_run=True), FakeGenerator)


if __name__ == "__main__":
    unittest.main()
