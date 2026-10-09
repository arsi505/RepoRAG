from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from reporag.chunking.models import Chunk
from reporag.lexical.bm25 import BM25SearchError, score_document, search
from reporag.lexical.config import BM25Config, DEFAULT_BM25_CONFIG
from reporag.lexical.index import (
    BM25IndexExpectation,
    BM25IndexMismatchError,
    build_index,
    load_index,
    save_index,
    source_fingerprint,
)


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    symbol: str | None = None,
    parent: str | None = None,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repository="ExampleRepo",
        commit_hash="abc123",
        file_path=f"src/{chunk_id}.py",
        language="Python",
        chunk_type="method" if parent else "function",
        symbol_name=symbol,
        parent_symbol=parent,
        start_line=10,
        end_line=12,
        part_index=None,
        part_count=1,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content=content,
        parser_status="parsed",
    )


class BM25RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path.cwd() / ".test_tmp"
        test_root.mkdir(exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temporary_directory.name)
        self.chunks = (
            make_chunk("connection", "def handleConnection():\n    register client\n", symbol="handleConnection"),
            make_chunk("presence", "def registerSocket():\n    update presence\n", symbol="registerSocket", parent="PresenceService"),
            make_chunk("unrelated", "def calculate_total():\n    return invoice\n", symbol="calculate_total"),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_frozen_configuration(self) -> None:
        self.assertEqual(DEFAULT_BM25_CONFIG.k1, 1.5)
        self.assertEqual(DEFAULT_BM25_CONFIG.b, 0.75)
        self.assertEqual(DEFAULT_BM25_CONFIG.tokenizer_version, "reporag-code-tokenizer-v1")
        self.assertEqual(DEFAULT_BM25_CONFIG.formatter_version, "reporag-bm25-document-v1")

    def test_exact_lexical_match_ranks_first(self) -> None:
        results = search(build_index(self.chunks), "invoice")
        self.assertEqual(results[0].chunk_id, "unrelated")

    def test_identifier_component_match(self) -> None:
        results = search(build_index(self.chunks), "connection")
        self.assertEqual(results[0].chunk_id, "connection")

    def test_term_frequency_increases_score_with_saturation(self) -> None:
        chunks = (
            make_chunk("once", "unicorn"),
            make_chunk("three", "unicorn unicorn unicorn"),
        )
        index = build_index(chunks, config=BM25Config(b=0.0))
        scores = {
            result.chunk_id: result.score for result in search(index, "unicorn", top_k=2)
        }
        self.assertGreater(scores["three"], scores["once"])
        self.assertLess(scores["three"], scores["once"] * 3)

    def test_idf_matches_documented_formula(self) -> None:
        index = build_index(self.chunks)
        expected = math.log(1.0 + (3 - 1 + 0.5) / (1 + 0.5))
        self.assertAlmostEqual(index.inverse_document_frequencies["invoice"], expected)
        self.assertGreater(
            index.inverse_document_frequencies["invoice"],
            index.inverse_document_frequencies["def"],
        )

    def test_document_length_normalization_favors_shorter_document(self) -> None:
        chunks = (
            make_chunk("short", "needle"),
            make_chunk("long", "needle " + "filler " * 100),
        )
        results = search(build_index(chunks, config=BM25Config(b=1.0)), "needle", top_k=2)
        self.assertEqual([result.chunk_id for result in results], ["short", "long"])

    def test_k1_configuration_affects_scoring(self) -> None:
        chunk = (make_chunk("repeated", "signal signal signal signal"),)
        low = search(build_index(chunk, config=BM25Config(k1=0.5)), "signal")[0].score
        high = search(build_index(chunk, config=BM25Config(k1=2.0)), "signal")[0].score
        self.assertNotEqual(low, high)

    def test_b_configuration_affects_scoring(self) -> None:
        chunks = (
            make_chunk("short", "signal"),
            make_chunk("long", "signal " + "filler " * 50),
        )
        without = search(build_index(chunks, config=BM25Config(b=0.0)), "signal", top_k=2)
        with_norm = search(build_index(chunks, config=BM25Config(b=1.0)), "signal", top_k=2)
        self.assertNotEqual(without[0].score, with_norm[0].score)

    def test_ranking_is_deterministic(self) -> None:
        index = build_index(self.chunks)
        self.assertEqual(search(index, "presence"), search(index, "presence"))

    def test_tie_breaking_uses_chunk_id(self) -> None:
        chunks = (make_chunk("z-last", "tie"), make_chunk("a-first", "tie"))
        results = search(build_index(chunks), "tie", top_k=2)
        self.assertEqual([result.chunk_id for result in results], ["a-first", "z-last"])

    def test_configurable_top_k(self) -> None:
        index = build_index(self.chunks)
        self.assertEqual(len(search(index, "def", top_k=1)), 1)
        self.assertEqual(len(search(index, "def", top_k=3)), 3)
        self.assertEqual(len(search(index, "def", top_k=5)), 3)

    def test_only_positive_scores_are_returned(self) -> None:
        results = search(build_index(self.chunks), "invoice", top_k=5)
        self.assertEqual([result.chunk_id for result in results], ["unrelated"])
        self.assertGreater(results[0].score, 0.0)

    def test_scores_are_finite(self) -> None:
        results = search(build_index(self.chunks), "def", top_k=5)
        self.assertTrue(all(math.isfinite(result.score) for result in results))

    def test_index_save_load_round_trip(self) -> None:
        original = build_index(self.chunks)
        save_index(original, self.root)
        loaded = load_index(self.root)
        self.assertEqual(loaded.metadata.to_dict(), original.metadata.to_dict())
        self.assertEqual(loaded.chunks, original.chunks)
        self.assertEqual(loaded.documents, original.documents)
        self.assertGreaterEqual(loaded.load_seconds, 0.0)

    def test_stale_index_detects_commit_change(self) -> None:
        save_index(build_index(self.chunks), self.root)
        with self.assertRaisesRegex(BM25IndexMismatchError, "commit hash"):
            load_index(self.root, expected=BM25IndexExpectation(commit_hash="other"))

    def test_stale_index_detects_chunk_ids(self) -> None:
        save_index(build_index(self.chunks), self.root)
        with self.assertRaisesRegex(BM25IndexMismatchError, "ordered chunk IDs"):
            load_index(self.root, expected=BM25IndexExpectation(ordered_chunk_ids=("other",)))

    def test_stale_index_detects_tokenizer_version(self) -> None:
        save_index(build_index(self.chunks), self.root)
        changed = replace(DEFAULT_BM25_CONFIG, tokenizer_version="tokenizer-v2")
        with self.assertRaisesRegex(BM25IndexMismatchError, "tokenizer version"):
            load_index(self.root, expected=BM25IndexExpectation(config=changed))

    def test_stale_index_detects_k1(self) -> None:
        save_index(build_index(self.chunks), self.root)
        with self.assertRaisesRegex(BM25IndexMismatchError, "k1"):
            load_index(
                self.root,
                expected=BM25IndexExpectation(config=BM25Config(k1=1.2)),
            )

    def test_stale_index_detects_b(self) -> None:
        save_index(build_index(self.chunks), self.root)
        with self.assertRaisesRegex(BM25IndexMismatchError, "b"):
            load_index(
                self.root,
                expected=BM25IndexExpectation(config=BM25Config(b=0.4)),
            )

    def test_result_contains_metadata_and_method(self) -> None:
        result = search(build_index(self.chunks), "connection", top_k=1)[0]
        chunk = self.chunks[0]
        self.assertEqual(result.method, "bm25")
        self.assertEqual(result.score_name, "bm25")
        self.assertEqual((result.file_path, result.symbol_name), (chunk.file_path, chunk.symbol_name))
        self.assertEqual((result.start_line, result.end_line, result.content), (10, 12, chunk.content))

    def test_empty_and_unsearchable_queries_are_rejected(self) -> None:
        index = build_index(self.chunks)
        with self.assertRaisesRegex(BM25SearchError, "must not be empty"):
            search(index, " ")
        with self.assertRaisesRegex(BM25SearchError, "no searchable"):
            search(index, "...///")
        with self.assertRaisesRegex(BM25SearchError, "positive"):
            search(index, "query", top_k=0)

    def test_unknown_term_returns_no_zero_score_fillers(self) -> None:
        self.assertEqual(search(build_index(self.chunks), "nonexistentxyz"), ())

    def test_fingerprint_is_deterministic_and_configuration_sensitive(self) -> None:
        first = source_fingerprint(self.chunks, DEFAULT_BM25_CONFIG)
        second = source_fingerprint(self.chunks, DEFAULT_BM25_CONFIG)
        changed = source_fingerprint(self.chunks, BM25Config(k1=1.2))
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_score_document_uses_raw_bm25_scale(self) -> None:
        index = build_index(self.chunks)
        score = score_document(("presence",), index.documents[1], index)
        self.assertGreater(score, 0.0)
        self.assertNotEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
