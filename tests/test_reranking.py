from __future__ import annotations

import math
import unittest
from dataclasses import replace

from reporag.reranking.config import (
    DEFAULT_RERANKER_CONFIG,
    FORMATTER_VERSION,
    MODEL_NAME,
    MODEL_REVISION,
    RerankerConfig,
)
from reporag.reranking.formatter import format_candidate
from reporag.reranking.metadata import reranking_fingerprint
from reporag.reranking.reranker import rerank_candidates
from reporag.retrieval.models import HybridSearchResult


def candidate(
    chunk_id: str,
    rank: int,
    *,
    rrf_score: float | None = None,
    content: str | None = None,
    symbol: str | None = None,
    parent: str | None = None,
) -> HybridSearchResult:
    return HybridSearchResult(
        rank=rank,
        chunk_id=chunk_id,
        rrf_score=rrf_score if rrf_score is not None else 1.0 / (60 + rank),
        vector_rank=rank,
        vector_raw_score=0.5 / rank,
        vector_contribution=1.0 / (60 + rank),
        bm25_rank=rank + 2,
        bm25_raw_score=10.0 / rank,
        bm25_contribution=1.0 / (62 + rank),
        repository="ExampleRepo",
        file_path=f"src/{chunk_id}.py",
        language="Python",
        chunk_type="method" if parent else "function",
        symbol_name=symbol or chunk_id,
        parent_symbol=parent,
        start_line=10,
        end_line=12,
        content=content or f"def {chunk_id}():\n    return {rank}\n",
    )


class FakeReranker:
    def __init__(self, scores=None, lengths=None, config=DEFAULT_RERANKER_CONFIG):
        self.config = config
        self.scores = scores
        self.lengths = lengths
        self.scored_pairs = ()

    def pair_token_lengths(self, pairs):
        return tuple(self.lengths or [20] * len(pairs))

    def document_token_lengths(self, documents):
        return tuple(len(document.split()) for document in documents)

    def score_pairs(self, pairs):
        self.scored_pairs = tuple(pairs)
        if self.scores is None:
            return tuple(float(position) for position in range(len(pairs)))
        return tuple(self.scores)


class RerankerConfigurationTests(unittest.TestCase):
    def test_frozen_defaults(self) -> None:
        self.assertEqual(DEFAULT_RERANKER_CONFIG.candidate_k, 50)
        self.assertEqual(DEFAULT_RERANKER_CONFIG.max_length, 1024)
        self.assertEqual(MODEL_NAME, "BAAI/bge-reranker-v2-m3")
        self.assertEqual(
            MODEL_REVISION, "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
        )
        self.assertEqual(FORMATTER_VERSION, "reporag-reranker-document-v1")

    def test_final_top_k_must_be_positive_and_within_candidate_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            DEFAULT_RERANKER_CONFIG.validate_final_top_k(0)
        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            DEFAULT_RERANKER_CONFIG.validate_final_top_k(51)

    def test_invalid_candidate_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_k"):
            RerankerConfig(candidate_k=0)

    def test_model_revision_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "revision"):
            RerankerConfig(model_revision="")

    def test_formatter_version_is_in_configuration(self) -> None:
        self.assertEqual(
            DEFAULT_RERANKER_CONFIG.to_dict()["formatter_version"], FORMATTER_VERSION
        )


class RerankerFormatterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.value = candidate(
            "method",
            1,
            content="def method():\n    return 'exact'\n",
            symbol="method",
            parent="Service",
        )

    def test_format_is_deterministic_and_contains_evidence(self) -> None:
        first = format_candidate(self.value)
        self.assertEqual(first, format_candidate(self.value))
        self.assertIn("File: src/method.py", first)
        self.assertIn("Language: Python", first)
        self.assertIn("Type: method", first)
        self.assertIn("Symbol: method", first)
        self.assertIn("Parent: Service", first)
        self.assertIn(self.value.content, first)

    def test_format_excludes_machine_and_retrieval_metadata(self) -> None:
        text = format_candidate(self.value)
        for forbidden in (
            "D:\\",
            "C:\\",
            "Hybrid",
            "Vector",
            "BM25",
            "RRF",
            "timestamp",
            "abc123",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)


class RerankingBehaviorTests(unittest.TestCase):
    def test_exactly_candidate_k_candidates_are_scored(self) -> None:
        model = FakeReranker()
        values = tuple(candidate(f"c{position:02}", position + 1) for position in range(60))
        rerank_candidates("query", values, model)
        self.assertEqual(len(model.scored_pairs), 50)

    def test_fewer_than_candidate_k_candidates_work(self) -> None:
        model = FakeReranker(scores=(0.2, 0.1))
        outcome = rerank_candidates("query", (candidate("a", 1), candidate("b", 2)), model)
        self.assertEqual(outcome.diagnostics.candidate_count, 2)
        self.assertEqual(len(outcome.results), 2)

    def test_reranker_reorders_and_final_top_k_follows_scores(self) -> None:
        values = (candidate("hybrid-first", 1), candidate("hybrid-second", 2))
        outcome = rerank_candidates(
            "query", values, FakeReranker(scores=(-5.0, 8.0)), final_top_k=1
        )
        self.assertEqual(outcome.results[0].chunk_id, "hybrid-second")
        self.assertEqual(outcome.results[0].hybrid_rank, 2)

    def test_raw_hybrid_score_does_not_determine_final_order(self) -> None:
        values = (
            candidate("large-rrf", 1, rrf_score=100.0),
            candidate("small-rrf", 2, rrf_score=0.001),
        )
        result = rerank_candidates(
            "query", values, FakeReranker(scores=(0.0, 1.0)), final_top_k=2
        )
        self.assertEqual([item.chunk_id for item in result.results], ["small-rrf", "large-rrf"])

    def test_repeated_ranking_is_deterministic(self) -> None:
        values = (candidate("a", 1), candidate("b", 2))
        first = rerank_candidates("query", values, FakeReranker(scores=(0.1, 0.2)))
        second = rerank_candidates("query", values, FakeReranker(scores=(0.1, 0.2)))
        self.assertEqual(first.results, second.results)

    def test_tie_uses_hybrid_rank_then_chunk_id(self) -> None:
        by_rank = rerank_candidates(
            "query",
            (candidate("later", 2), candidate("earlier", 1)),
            FakeReranker(scores=(1.0, 1.0)),
        )
        self.assertEqual([item.chunk_id for item in by_rank.results], ["earlier", "later"])
        by_id = rerank_candidates(
            "query",
            (candidate("z", 1), candidate("a", 1)),
            FakeReranker(scores=(1.0, 1.0)),
        )
        self.assertEqual([item.chunk_id for item in by_id.results], ["a", "z"])

    def test_metadata_and_vector_bm25_provenance_are_preserved(self) -> None:
        source = candidate("kept", 4, parent="Service")
        result = rerank_candidates("query", (source,), FakeReranker(scores=(3.0,))).results[0]
        self.assertEqual(result.retrieval_method, "hybrid_reranked")
        self.assertEqual(result.score_type, "cross_encoder_raw_logit")
        self.assertEqual((result.hybrid_rank, result.hybrid_rrf_score), (4, source.rrf_score))
        self.assertEqual((result.vector_rank, result.vector_raw_score), (4, source.vector_raw_score))
        self.assertEqual((result.bm25_rank, result.bm25_raw_score), (6, source.bm25_raw_score))
        self.assertEqual((result.file_path, result.parent_symbol, result.content), (source.file_path, "Service", source.content))

    def test_empty_candidates_are_handled(self) -> None:
        model = FakeReranker()
        outcome = rerank_candidates("query", (), model)
        self.assertEqual(outcome.results, ())
        self.assertEqual(outcome.diagnostics.candidate_count, 0)

    def test_invalid_scores_are_rejected(self) -> None:
        for score in (math.nan, math.inf, -math.inf):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "finite"):
                rerank_candidates("query", (candidate("bad", 1),), FakeReranker(scores=(score,)))

    def test_duplicate_chunk_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            rerank_candidates(
                "query",
                (candidate("same", 1), candidate("same", 2)),
                FakeReranker(),
            )

    def test_original_hybrid_results_are_not_mutated(self) -> None:
        values = (candidate("a", 1), candidate("b", 2))
        before = tuple(values)
        rerank_candidates("query", values, FakeReranker(scores=(0.0, 1.0)))
        self.assertEqual(values, before)

    def test_truncation_accounting_is_deterministic(self) -> None:
        values = (candidate("short", 1), candidate("long", 2))
        model = FakeReranker(scores=(2.0, 1.0), lengths=(100, 1025))
        first = rerank_candidates("query", values, model)
        second = rerank_candidates(
            "query", values, FakeReranker(scores=(2.0, 1.0), lengths=(100, 1025))
        )
        self.assertEqual(first.diagnostics.candidates_over_configured_limit, 1)
        self.assertEqual(first.diagnostics.candidates_truncated, 1)
        self.assertEqual(first.diagnostics, second.diagnostics)
        self.assertEqual([item.chunk_id for item in first.results], ["short", "long"])

    def test_below_limit_reports_no_truncation(self) -> None:
        outcome = rerank_candidates(
            "query", (candidate("short", 1),), FakeReranker(scores=(1.0,), lengths=(1024,))
        )
        self.assertEqual(outcome.diagnostics.candidates_truncated, 0)


class RerankingFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = dict(
            repository="Repo",
            commit_hash="commit",
            corpus_fingerprint="corpus",
            hybrid_fingerprint="hybrid",
        )

    def test_fingerprint_is_stable(self) -> None:
        self.assertEqual(reranking_fingerprint(**self.base), reranking_fingerprint(**self.base))

    def test_fingerprint_changes_for_method_d_inputs(self) -> None:
        original = reranking_fingerprint(**self.base)
        for field in ("commit_hash", "corpus_fingerprint", "hybrid_fingerprint"):
            changed = dict(self.base)
            changed[field] += "-changed"
            with self.subTest(field=field):
                self.assertNotEqual(original, reranking_fingerprint(**changed))
        config_changes = (
            replace(DEFAULT_RERANKER_CONFIG, candidate_k=40),
            replace(DEFAULT_RERANKER_CONFIG, model_name="other/model"),
            replace(DEFAULT_RERANKER_CONFIG, model_revision="other-revision"),
            replace(DEFAULT_RERANKER_CONFIG, max_length=512),
            replace(DEFAULT_RERANKER_CONFIG, formatter_version="formatter-v2"),
        )
        for changed_config in config_changes:
            with self.subTest(config=changed_config):
                self.assertNotEqual(
                    original,
                    reranking_fingerprint(**self.base, config=changed_config),
                )


if __name__ == "__main__":
    unittest.main()
