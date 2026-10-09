from __future__ import annotations

import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from reporag.generation.fake import FakeGenerator
from reporag.generation.openai_provider import OpenAIResponsesProvider
from reporag.generation.protocol import GenerationError
from reporag.qa.citations import cited_evidence, parse_source_ids, validate_citations
from reporag.qa.config import (
    DEFAULT_QA_CONFIG,
    DEEPSEEK_MODEL,
    GEMINI_MODEL,
    OPENAI_MODEL,
    QAConfig,
    qa_fingerprint,
)
from reporag.qa.cli import build_parser
from reporag.qa.context import build_evidence, filter_by_evidence_scope, format_context
from reporag.qa.prompts import GROUNDING_INSTRUCTIONS, PROMPT_VERSION, build_user_input
from reporag.qa.service import (
    EMPTY_RETRIEVAL_ANSWER,
    QAServiceError,
    answer_question,
)
from reporag.retrieval.models import SearchResult


def result(
    chunk_id: str,
    rank: int,
    *,
    path: str | None = None,
    language: str = "Python",
) -> SearchResult:
    return SearchResult(
        rank=rank,
        chunk_id=chunk_id,
        score=1.0 / rank,
        repository="ExampleRepo",
        file_path=path or f"src/{chunk_id}.py",
        language=language,
        chunk_type="method",
        symbol_name=f"method_{chunk_id}",
        parent_symbol="Service",
        start_line=rank * 10,
        end_line=rank * 10 + 4,
        content=f"def method_{chunk_id}():\n    return '{chunk_id}'\n",
    )


class FakeRetriever:
    def __init__(self, results=()):
        self.results = tuple(results)
        self.calls = []

    def retrieve(self, question, method, top_k):
        self.calls.append((question, method, top_k))
        return self.results[:top_k]

    def fingerprint(self, method):
        return f"{method}-fingerprint"


class QAContextTests(unittest.TestCase):
    def test_evidence_scope_filters_without_reordering(self) -> None:
        values = (
            result("doc-1", 1, path="README.md", language="Markdown"),
            result("code-1", 2),
            result("doc-2", 3, path="docs/design.md", language="Markdown"),
            result("code-2", 4, path="tests/test_service.py"),
        )
        code = filter_by_evidence_scope(values, "code")
        docs = filter_by_evidence_scope(values, "docs")
        self.assertEqual([item.chunk_id for item in code], ["code-1", "code-2"])
        self.assertEqual([item.rank for item in code], [2, 4])
        self.assertEqual([item.chunk_id for item in docs], ["doc-1", "doc-2"])
        self.assertEqual([item.rank for item in docs], [1, 3])

    def test_ranks_map_to_source_ids_in_retrieval_order(self) -> None:
        evidence = build_evidence((result("a", 1), result("b", 2), result("c", 3)), context_k=3)
        self.assertEqual([item.source_id for item in evidence], ["S1", "S2", "S3"])
        self.assertEqual([item.retrieval_rank for item in evidence], [1, 2, 3])

    def test_context_is_deterministic_and_preserves_metadata_and_code(self) -> None:
        source = result("a", 1)
        evidence = build_evidence((source,), context_k=1)
        first = format_context(evidence)
        self.assertEqual(first, format_context(evidence))
        self.assertIn("File: src/a.py", first)
        self.assertIn("Symbol: method_a", first)
        self.assertIn("Parent: Service", first)
        self.assertIn("Lines: 10-14", first)
        self.assertIn(source.content, first)

    def test_context_has_no_injected_absolute_path(self) -> None:
        text = format_context(build_evidence((result("a", 1),), context_k=1))
        self.assertNotIn("D:\\", text)
        self.assertNotIn("C:\\", text)
        with self.assertRaisesRegex(ValueError, "relative"):
            build_evidence((result("bad", 1, path="C:\\repo\\bad.py"),), context_k=1)

    def test_duplicates_are_removed_preserving_first_occurrence(self) -> None:
        first = result("same", 1)
        duplicate = result("same", 2)
        other = result("other", 3)
        evidence = build_evidence((first, duplicate, other), context_k=3)
        self.assertEqual([item.chunk_id for item in evidence], ["same", "other"])
        self.assertEqual(evidence[0].retrieval_rank, 1)

    def test_context_k_is_enforced_without_reordering(self) -> None:
        values = tuple(result(str(position), position) for position in range(1, 6))
        evidence = build_evidence(values, context_k=3)
        self.assertEqual([item.chunk_id for item in evidence], ["1", "2", "3"])


class QAPromptTests(unittest.TestCase):
    def test_prompt_contains_exact_question_evidence_and_source_ids(self) -> None:
        question = "  Where exactly?  "
        context = "[S1]\nFile: src/a.py\nCode:\npass"
        prompt = build_user_input(question, context)
        self.assertIn(f"QUESTION:\n{question}", prompt)
        self.assertIn(context, prompt)
        self.assertIn("[S1]", prompt)

    def test_instructions_require_grounding_citations_and_insufficiency(self) -> None:
        self.assertIn("Use only the repository evidence", GROUNDING_INSTRUCTIONS)
        self.assertIn("[S1]", GROUNDING_INSTRUCTIONS)
        self.assertIn("don't have enough repository evidence", GROUNDING_INSTRUCTIONS)
        self.assertIn("Do not use web search", GROUNDING_INSTRUCTIONS)

    def test_prompt_version_and_output_are_stable(self) -> None:
        self.assertEqual(PROMPT_VERSION, "reporag-qa-prompt-v1")
        self.assertEqual(build_user_input("q", "c"), build_user_input("q", "c"))


class QACitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.evidence = build_evidence((result("a", 1), result("b", 2)), context_k=2)

    def test_single_multiple_and_repeated_citations_are_parsed(self) -> None:
        self.assertEqual(parse_source_ids("Claim [S1]."), ("S1",))
        self.assertEqual(parse_source_ids("[S2] and [S1] and [S2]"), ("S2", "S1"))

    def test_unknown_source_is_detected(self) -> None:
        validation = validate_citations("Claim [S1] [S9].", self.evidence)
        self.assertFalse(validation.citations_valid)
        self.assertEqual(validation.unknown_source_ids, ("S9",))

    def test_no_citations_is_invalid_for_an_evidence_answer(self) -> None:
        validation = validate_citations("Uncited answer", self.evidence)
        self.assertEqual(validation.cited_source_ids, ())
        self.assertFalse(validation.citations_valid)

    def test_valid_citations_and_file_mapping(self) -> None:
        validation = validate_citations("Supported [S2].", self.evidence)
        self.assertTrue(validation.citations_valid)
        mapped = cited_evidence(validation, self.evidence)
        self.assertEqual((mapped[0].source_id, mapped[0].file_path), ("S2", "src/b.py"))

    def test_malformed_markers_are_ignored_safely(self) -> None:
        self.assertEqual(parse_source_ids("[S] [S0] [S-1] S1 [s1]"), ())


class QAServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = tuple(result(str(position), position) for position in range(1, 7))

    def test_original_question_method_and_context_k_are_sent_unchanged(self) -> None:
        retriever = FakeRetriever(self.values)
        generator = FakeGenerator("Answer [S1].")
        question = "  original question?  "
        answer_question(
            question,
            retriever,
            generator,
            retrieval_method="bm25",
            context_k=3,
        )
        self.assertEqual(retriever.calls, [(question, "bm25", 3)])

    def test_code_scope_overretrieves_and_surfaces_code_below_top_five(self) -> None:
        docs = tuple(
            result(f"doc-{rank}", rank, path=f"docs/{rank}.md", language="Markdown")
            for rank in range(1, 6)
        )
        code = tuple(result(f"code-{rank}", rank) for rank in range(6, 11))
        retriever = FakeRetriever(docs + code)
        output = answer_question(
            "question",
            retriever,
            FakeGenerator("Answer [S1]."),
            config=replace(DEFAULT_QA_CONFIG, evidence_scope="code"),
        )
        self.assertEqual(retriever.calls, [("question", "hybrid", 50)])
        self.assertEqual(
            [item.chunk_id for item in output.evidence],
            ["code-6", "code-7", "code-8", "code-9", "code-10"],
        )
        self.assertEqual([item.retrieval_rank for item in output.evidence], [6, 7, 8, 9, 10])

    def test_docs_scope_excludes_code_and_handles_fewer_than_context_k(self) -> None:
        values = (
            result("code-1", 1),
            result("doc-2", 2, path="README.md", language="Markdown"),
            result("code-3", 3),
            result("doc-4", 4, path="docs/guide.md", language="Markdown"),
        )
        output = answer_question(
            "question",
            FakeRetriever(values),
            FakeGenerator("Answer [S1]."),
            config=replace(DEFAULT_QA_CONFIG, evidence_scope="docs"),
        )
        self.assertEqual([item.chunk_id for item in output.evidence], ["doc-2", "doc-4"])

    def test_scope_never_falls_back_when_no_chunks_match(self) -> None:
        docs = (
            result("doc", 1, path="README.md", language="Markdown"),
        )
        generator = FakeGenerator("must not run")
        output = answer_question(
            "question",
            FakeRetriever(docs),
            generator,
            config=replace(DEFAULT_QA_CONFIG, evidence_scope="code"),
        )
        self.assertEqual(output.answer_text, EMPTY_RETRIEVAL_ANSWER)
        self.assertEqual(output.evidence, ())
        self.assertEqual(generator.calls, [])

    def test_all_scope_preserves_existing_retrieval_behavior(self) -> None:
        retriever = FakeRetriever(self.values)
        output = answer_question(
            "question", retriever, FakeGenerator("Answer [S1].")
        )
        self.assertEqual(retriever.calls, [("question", "hybrid", 5)])
        self.assertEqual([item.chunk_id for item in output.evidence], ["1", "2", "3", "4", "5"])
        self.assertEqual(output.evidence_scope, "all")

    def test_evidence_is_passed_to_generator_and_answer_returned(self) -> None:
        generator = FakeGenerator("Generated answer [S1].")
        output = answer_question("question", FakeRetriever(self.values), generator)
        self.assertEqual(output.answer_text, "Generated answer [S1].")
        self.assertIn("[S1]", generator.calls[0][1])
        self.assertIn(self.values[0].content, generator.calls[0][1])
        self.assertTrue(output.citations_valid)

    def test_metadata_is_preserved_and_result_has_no_api_key(self) -> None:
        output = answer_question(
            "question", FakeRetriever(self.values), FakeGenerator("Answer [S1].")
        )
        self.assertEqual(output.evidence[0].file_path, self.values[0].file_path)
        self.assertEqual(output.evidence[0].content, self.values[0].content)
        self.assertFalse(hasattr(output, "api_key"))

    def test_fabricated_citation_is_reported(self) -> None:
        output = answer_question(
            "question", FakeRetriever(self.values), FakeGenerator("Answer [S99].")
        )
        self.assertFalse(output.citations_valid)
        self.assertEqual(output.unknown_source_ids, ("S99",))

    def test_empty_retrieval_does_not_call_generator(self) -> None:
        generator = FakeGenerator("must not run")
        output = answer_question("question", FakeRetriever(), generator)
        self.assertEqual(output.answer_text, EMPTY_RETRIEVAL_ANSWER)
        self.assertEqual(generator.calls, [])

    def test_provider_error_is_actionable(self) -> None:
        with self.assertRaisesRegex(QAServiceError, "Generation failed"):
            answer_question(
                "question", FakeRetriever(self.values), FakeGenerator(error="provider down")
            )

    def test_generator_input_has_no_machine_path(self) -> None:
        generator = FakeGenerator("Answer [S1].")
        answer_question("question", FakeRetriever(self.values), generator)
        self.assertNotIn("D:\\", generator.calls[0][1])
        self.assertNotIn("C:\\", generator.calls[0][1])

    def test_existing_results_are_not_mutated(self) -> None:
        before = tuple(self.values)
        answer_question("question", FakeRetriever(self.values), FakeGenerator("Answer [S1]."))
        self.assertEqual(self.values, before)

    def test_all_four_retrieval_modes_are_supported(self) -> None:
        for method in ("vector", "bm25", "hybrid", "reranked"):
            retriever = FakeRetriever(self.values)
            with self.subTest(method=method):
                output = answer_question(
                    "question",
                    retriever,
                    FakeGenerator("Answer [S1]."),
                    retrieval_method=method,
                )
                self.assertEqual(output.retrieval_method, method)
                self.assertEqual(retriever.calls[0][1], method)

    def test_dry_run_performs_retrieval_but_not_generation(self) -> None:
        generator = FakeGenerator("must not run")
        output = answer_question(
            "question", FakeRetriever(self.values), generator, dry_run=True
        )
        self.assertTrue(output.dry_run)
        self.assertEqual(len(output.evidence), 5)
        self.assertEqual(generator.calls, [])

    def test_insufficient_evidence_answer_can_be_represented(self) -> None:
        text = "The retrieved repository evidence is insufficient to answer this question confidently."
        output = answer_question(
            "question", FakeRetriever(self.values), FakeGenerator(text)
        )
        self.assertEqual(output.answer_text, text)
        self.assertFalse(output.citations_valid)


class QAConfigFingerprintTests(unittest.TestCase):
    def test_frozen_defaults(self) -> None:
        self.assertEqual(DEFAULT_QA_CONFIG.qa_version, "reporag-qa-v1")
        self.assertEqual(DEFAULT_QA_CONFIG.default_retrieval_method, "hybrid")
        self.assertEqual(DEFAULT_QA_CONFIG.default_context_k, 5)
        self.assertEqual(DEFAULT_QA_CONFIG.evidence_scope, "all")
        self.assertEqual(DEFAULT_QA_CONFIG.generation_provider, "deepseek")
        self.assertEqual(DEFAULT_QA_CONFIG.generation_model, DEEPSEEK_MODEL)
        self.assertEqual(DEFAULT_QA_CONFIG.max_output_tokens, 2000)

    def test_fingerprint_is_stable_and_configuration_sensitive(self) -> None:
        base = dict(retrieval_method="hybrid", retrieval_fingerprint="retrieval", context_k=5)
        original = qa_fingerprint(**base)
        self.assertEqual(original, qa_fingerprint(**base))
        for changed in (
            dict(base, retrieval_method="vector"),
            dict(base, retrieval_fingerprint="changed"),
            dict(base, context_k=3),
        ):
            self.assertNotEqual(original, qa_fingerprint(**changed))
        configs = (
            replace(DEFAULT_QA_CONFIG, generation_model="other-snapshot"),
            replace(DEFAULT_QA_CONFIG, prompt_version="prompt-v2"),
            replace(DEFAULT_QA_CONFIG, context_formatter_version="context-v2"),
            replace(DEFAULT_QA_CONFIG, max_output_tokens=600),
            replace(DEFAULT_QA_CONFIG, evidence_scope="code"),
        )
        for config in configs:
            self.assertNotEqual(original, qa_fingerprint(**base, config=config))

    def test_cli_parses_evidence_scope(self) -> None:
        defaults = build_parser().parse_args(["vector", "bm25", "question"])
        scoped = build_parser().parse_args(
            ["vector", "bm25", "question", "--evidence-scope", "docs"]
        )
        self.assertEqual(defaults.evidence_scope, "all")
        self.assertEqual(scoped.evidence_scope, "docs")

    def test_fingerprint_changes_between_gemini_and_openai(self) -> None:
        base = dict(
            retrieval_method="hybrid",
            retrieval_fingerprint="retrieval",
            context_k=5,
        )
        gemini = qa_fingerprint(
            **base,
            generation_provider="gemini",
            generation_model=GEMINI_MODEL,
        )
        openai = qa_fingerprint(
            **base,
            generation_provider="openai",
            generation_model=OPENAI_MODEL,
        )
        self.assertNotEqual(gemini, openai)

    def test_fingerprint_differs_across_all_live_providers(self) -> None:
        base = dict(
            retrieval_method="hybrid",
            retrieval_fingerprint="retrieval",
            context_k=5,
        )
        fingerprints = {
            qa_fingerprint(
                **base,
                generation_provider=provider,
                generation_model=model,
            )
            for provider, model in (
                ("deepseek", DEEPSEEK_MODEL),
                ("gemini", GEMINI_MODEL),
                ("openai", OPENAI_MODEL),
            )
        }
        self.assertEqual(len(fingerprints), 3)


class OpenAIProviderOfflineTests(unittest.TestCase):
    def test_missing_api_key_fails_before_network_access(self) -> None:
        provider = OpenAIResponsesProvider(model="snapshot", max_output_tokens=10)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GenerationError, "OPENAI_API_KEY"):
                provider.generate("instructions", "input")


if __name__ == "__main__":
    unittest.main()
