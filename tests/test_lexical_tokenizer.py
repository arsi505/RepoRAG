from __future__ import annotations

import unittest

from reporag.chunking.models import Chunk
from reporag.lexical.formatter import format_chunk_for_bm25
from reporag.lexical.tokenizer import tokenize


class LexicalTokenizerTests(unittest.TestCase):
    def test_camel_case_splitting_and_exact_preservation(self) -> None:
        tokens = tokenize("handleConnection")
        self.assertEqual(tokens, ("handleconnection", "handle", "connection"))

    def test_pascal_case_splitting(self) -> None:
        self.assertEqual(
            tokenize("PresenceService"),
            ("presenceservice", "presence", "service"),
        )

    def test_snake_case_splitting(self) -> None:
        self.assertEqual(
            tokenize("register_socket"),
            ("register_socket", "register", "socket"),
        )

    def test_kebab_case_splitting(self) -> None:
        self.assertEqual(tokenize("retry-job"), ("retry-job", "retry", "job"))

    def test_repository_path_splitting(self) -> None:
        tokens = tokenize("backend/src/auth/guards/jwt-auth.guard.ts")
        for expected in ("backend", "src", "auth", "guards", "jwt", "guard", "ts"):
            self.assertIn(expected, tokens)
        self.assertIn("jwt-auth", tokens)

    def test_acronym_splitting(self) -> None:
        self.assertEqual(
            tokenize("JWTAuthGuard"),
            ("jwtauthguard", "jwt", "auth", "guard"),
        )

    def test_case_normalization(self) -> None:
        self.assertEqual(tokenize("MiXeD"), ("mixed", "mi", "xe", "d"))

    def test_punctuation_and_numeric_components(self) -> None:
        tokens = tokenize("retry2Job(value); status=404")
        self.assertIn("retry2job", tokens)
        self.assertIn("retry", tokens)
        self.assertIn("2", tokens)
        self.assertIn("job", tokens)
        self.assertIn("404", tokens)
        self.assertNotIn("(", tokens)

    def test_tokenization_is_deterministic(self) -> None:
        value = "PresenceService.registerSocket"
        self.assertEqual(tokenize(value), tokenize(value))

    def test_empty_string(self) -> None:
        self.assertEqual(tokenize(""), ())
        self.assertEqual(tokenize("...///"), ())

    def test_formatter_uses_relative_metadata_without_machine_path(self) -> None:
        chunk = Chunk(
            chunk_id="one",
            repository="Repo",
            commit_hash="abc",
            file_path="src/service.py",
            language="Python",
            chunk_type="function",
            symbol_name="runTask",
            parent_symbol=None,
            start_line=1,
            end_line=2,
            part_index=None,
            part_count=1,
            content_hash="hash",
            content="def runTask():\n    pass\n",
            parser_status="parsed",
        )
        formatted = format_chunk_for_bm25(chunk)
        self.assertIn("File: src/service.py", formatted)
        self.assertIn("Symbol: runTask", formatted)
        self.assertNotIn("C:\\", formatted)
        self.assertNotIn("D:\\", formatted)


if __name__ == "__main__":
    unittest.main()
