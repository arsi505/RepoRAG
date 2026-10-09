from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from reporag.generation.environment import load_local_environment
from reporag.generation.fake import FakeGenerator
from reporag.generation.gemini_provider import GeminiProvider
from reporag.generation.openai_provider import OpenAIResponsesProvider
from reporag.generation.protocol import GenerationError
from reporag.qa.cli import build_parser, create_generator
from reporag.qa.config import GEMINI_MODEL, MAX_OUTPUT_TOKENS
from reporag.qa.prompts import GROUNDING_INSTRUCTIONS, build_user_input


class Usage:
    def model_dump(self, **_kwargs):
        return {
            "prompt_token_count": 40,
            "candidates_token_count": 12,
            "total_token_count": 52,
        }


class Response:
    def __init__(self, text="Grounded answer [S1]."):
        self.text = text
        self.response_id = "gemini-response"
        self.usage_metadata = Usage()


class RecordingModels:
    def __init__(self, response=None, error=None):
        self.response = response or Response()
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class RecordingFactory:
    def __init__(self, models=None):
        self.models = models or RecordingModels()
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(models=self.models)


class ProviderFailure(Exception):
    def __init__(self, code, message="sensitive provider detail"):
        super().__init__(message)
        self.code = code


class GeminiProviderOfflineTests(unittest.TestCase):
    def test_api_key_is_required_before_client_creation(self) -> None:
        factory = RecordingFactory()
        provider = GeminiProvider(client_factory=factory)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GenerationError, "GEMINI_API_KEY"):
                provider.generate("instructions", "input")
        self.assertEqual(factory.calls, [])

    def test_default_model_and_output_limit_are_frozen(self) -> None:
        provider = GeminiProvider(client_factory=RecordingFactory())
        self.assertEqual(provider.model, "gemini-3.8-flash")
        self.assertEqual(provider.model, GEMINI_MODEL)
        self.assertEqual(provider.max_output_tokens, MAX_OUTPUT_TOKENS)

    def test_grounding_instructions_and_evidence_are_sent(self) -> None:
        factory = RecordingFactory()
        provider = GeminiProvider(client_factory=factory)
        evidence_input = build_user_input("Where?", "[S1]\nCode:\nanswer")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}, clear=True):
            provider.generate(GROUNDING_INSTRUCTIONS, evidence_input)
        call = factory.models.calls[0]
        self.assertEqual(call["model"], GEMINI_MODEL)
        self.assertEqual(call["contents"], evidence_input)
        self.assertEqual(call["config"].system_instruction, GROUNDING_INSTRUCTIONS)
        self.assertIn("[S1]", call["contents"])

    def test_no_external_tools_are_enabled(self) -> None:
        factory = RecordingFactory()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}, clear=True):
            GeminiProvider(client_factory=factory).generate("instructions", "input")
        config = factory.models.calls[0]["config"]
        self.assertIsNone(config.tools)
        self.assertEqual(config.max_output_tokens, 1200)

    def test_generated_text_metadata_and_usage_are_returned(self) -> None:
        factory = RecordingFactory()
        with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}, clear=True):
            result = GeminiProvider(client_factory=factory).generate("instructions", "input")
        self.assertEqual(result.text, "Grounded answer [S1].")
        self.assertEqual((result.provider, result.model), ("gemini", GEMINI_MODEL))
        self.assertEqual(result.response_id, "gemini-response")
        self.assertEqual(result.usage["total_token_count"], 52)
        self.assertIsNotNone(result.latency_seconds)

    def test_key_is_not_stored_in_generation_result(self) -> None:
        secret = "unit-test-key-that-must-not-be-returned"
        factory = RecordingFactory()
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}, clear=True):
            result = GeminiProvider(client_factory=factory).generate("instructions", "input")
        self.assertFalse(hasattr(result, "api_key"))
        self.assertNotIn(secret, repr(result))

    def test_empty_response_is_rejected(self) -> None:
        factory = RecordingFactory(RecordingModels(Response("  ")))
        with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}, clear=True):
            with self.assertRaisesRegex(GenerationError, "empty text response"):
                GeminiProvider(client_factory=factory).generate("instructions", "input")

    def test_invalid_key_and_rate_limit_are_classified(self) -> None:
        for code, expected in ((401, "authentication failed"), (429, "rate limit")):
            factory = RecordingFactory(RecordingModels(error=ProviderFailure(code)))
            with self.subTest(code=code):
                with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}, clear=True):
                    with self.assertRaisesRegex(GenerationError, expected):
                        GeminiProvider(client_factory=factory).generate("instructions", "input")

    def test_network_and_generic_errors_are_safe_and_actionable(self) -> None:
        cases = (
            (ConnectionError("unit-test-key"), "connect to the Gemini API"),
            (RuntimeError("unit-test-key"), "provider request failed"),
        )
        for error, expected in cases:
            factory = RecordingFactory(RecordingModels(error=error))
            with self.subTest(error=type(error).__name__):
                with patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-key"}, clear=True):
                    with self.assertRaisesRegex(GenerationError, expected) as caught:
                        GeminiProvider(client_factory=factory).generate("instructions", "input")
                self.assertNotIn("unit-test-key", str(caught.exception))

    def test_provider_selection_still_supports_gemini(self) -> None:
        arguments = build_parser().parse_args(["vector", "bm25", "question"])
        self.assertEqual(arguments.provider, "deepseek")
        self.assertIsInstance(create_generator("gemini", dry_run=False), GeminiProvider)
        self.assertIsInstance(
            create_generator("openai", dry_run=False), OpenAIResponsesProvider
        )
        self.assertIsInstance(create_generator("gemini", dry_run=True), FakeGenerator)

    def test_dotenv_does_not_override_explicit_environment(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            dotenv_path = Path(directory) / ".env"
            dotenv_path.write_text("GEMINI_API_KEY=file-value\n", encoding="utf-8")
            with patch.dict(
                os.environ, {"GEMINI_API_KEY": "explicit-value"}, clear=True
            ):
                self.assertTrue(load_local_environment(dotenv_path))
                self.assertEqual(os.environ["GEMINI_API_KEY"], "explicit-value")

    def test_import_and_construction_do_not_require_dotenv_file(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            provider = GeminiProvider(client_factory=RecordingFactory())
        self.assertEqual(provider.model, GEMINI_MODEL)


if __name__ == "__main__":
    unittest.main()
