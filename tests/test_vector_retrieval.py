from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from reporag.chunking.models import Chunk
from reporag.embeddings.config import EmbeddingConfig
from reporag.embeddings.config import DEFAULT_EMBEDDING_CONFIG
from reporag.embeddings.formatter import format_chunk_for_embedding
from reporag.embeddings.index import (
    EmbeddingIndex,
    IndexExpectation,
    IndexMismatchError,
    build_index,
    load_index,
    save_index,
)
from reporag.embeddings.model import (
    SentenceTransformerEmbeddingModel,
    _adaptive_batch_indices,
    normalize_rows,
)
from reporag.retrieval.vector import VectorSearchError, search


CONFIG = EmbeddingConfig("fake/code-model", "revision-1", 3, True, "format-v1")


def make_chunk(
    chunk_id: str,
    *,
    symbol: str | None = None,
    parent: str | None = None,
    content: str = "return value\n",
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
        end_line=11,
        part_index=None,
        part_count=1,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content=content,
        parser_status="parsed",
    )


class FakeEmbeddingModel:
    def __init__(self, config: EmbeddingConfig = CONFIG) -> None:
        self._config = config

    @property
    def config(self) -> EmbeddingConfig:
        return self._config

    def encode_documents(self, texts):
        rows = []
        for position, _ in enumerate(texts):
            rows.append(([3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [1.0, 1.0, 0.0])[position % 3])
        return np.asarray(rows, dtype=np.float32)

    def encode_queries(self, texts):
        return np.asarray([[2.0, 0.0, 0.0] for _ in texts], dtype=np.float32)

    def token_lengths(self, texts):
        return tuple(len(text.split()) for text in texts)


class RecordingSentenceEncoder:
    def __init__(self) -> None:
        self.batches = []

    def encode(self, values, **_kwargs):
        self.batches.append(tuple(values))
        rows = {
            "long": [1.0, 0.0, 0.0],
            "short": [0.0, 1.0, 0.0],
            "medium": [0.0, 0.0, 1.0],
        }
        return np.asarray([rows[value] for value in values], dtype=np.float32)


class VectorRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        test_root = Path.cwd() / ".test_tmp"
        test_root.mkdir(exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temporary_directory.name)
        self.chunks = (
            make_chunk("chunk-b", symbol="alpha", parent="Service"),
            make_chunk("chunk-a", symbol="beta"),
            make_chunk("chunk-c", symbol="gamma"),
        )
        self.model = FakeEmbeddingModel()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_embedding_text_is_deterministic(self) -> None:
        self.assertEqual(
            format_chunk_for_embedding(self.chunks[0]),
            format_chunk_for_embedding(self.chunks[0]),
        )

    def test_frozen_maximum_sequence_length_is_8192(self) -> None:
        self.assertEqual(DEFAULT_EMBEDDING_CONFIG.max_sequence_length, 8192)

    def test_embedding_text_contains_repository_relative_path(self) -> None:
        text = format_chunk_for_embedding(self.chunks[0])
        self.assertIn("File: src/chunk-b.py", text)
        self.assertNotIn(str(self.root), text)

    def test_embedding_text_contains_symbol_and_parent(self) -> None:
        text = format_chunk_for_embedding(self.chunks[0])
        self.assertIn("Symbol: alpha", text)
        self.assertIn("Parent: Service", text)

    def test_embedding_text_never_contains_absolute_machine_path(self) -> None:
        text = format_chunk_for_embedding(self.chunks[0])
        self.assertNotIn("D:\\", text)
        self.assertNotIn("C:\\", text)

    def test_normalize_rows(self) -> None:
        matrix = normalize_rows(np.asarray([[3.0, 4.0, 0.0]], dtype=np.float32))
        self.assertTrue(np.allclose(np.linalg.norm(matrix, axis=1), 1.0))

    def test_adaptive_batches_bound_long_inputs_and_cover_every_index(self) -> None:
        batches = _adaptive_batch_indices(
            (3000, 20, 1000, 10),
            preferred_batch_size=16,
            max_sequence_length=8192,
        )
        self.assertEqual(batches[0], (0,))
        self.assertEqual(sorted(index for batch in batches for index in batch), [0, 1, 2, 3])
        self.assertTrue(all(len(batch) <= 16 for batch in batches))

    def test_adaptive_encoding_restores_original_document_order(self) -> None:
        model = SentenceTransformerEmbeddingModel.__new__(
            SentenceTransformerEmbeddingModel
        )
        model._config = CONFIG
        model._batch_size = 16
        model._show_progress = False
        model._model = RecordingSentenceEncoder()
        model.token_lengths = lambda texts: tuple(
            {"long": 3000, "short": 10, "medium": 1000}[text] for text in texts
        )

        encoded = model.encode_documents(("long", "short", "medium"))
        repeated = model.encode_documents(("long", "short", "medium"))

        self.assertEqual(encoded.shape, (3, CONFIG.embedding_dimension))
        self.assertTrue(np.array_equal(encoded, np.eye(3, dtype=np.float32)))
        self.assertTrue(np.array_equal(repeated, encoded))
        self.assertEqual(model.config, CONFIG)
        self.assertEqual(model._model.batches[0], ("long",))

    def test_matrix_rows_align_with_chunk_ids(self) -> None:
        index = build_index(self.chunks, self.model)
        self.assertEqual(index.metadata.ordered_chunk_ids, tuple(c.chunk_id for c in self.chunks))
        self.assertTrue(np.allclose(index.embeddings[0], [1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(index.embeddings[1], [0.0, 1.0, 0.0]))
        self.assertEqual(index.metadata.max_sequence_length, CONFIG.max_sequence_length)

    def test_index_save_load_round_trip(self) -> None:
        original = build_index(self.chunks, self.model)
        save_index(original, self.root)
        loaded = load_index(self.root)
        self.assertEqual(loaded.metadata.to_dict(), original.metadata.to_dict())
        self.assertEqual(loaded.chunks, original.chunks)
        self.assertTrue(np.array_equal(loaded.embeddings, original.embeddings))

    def test_stale_index_detects_commit_change(self) -> None:
        save_index(build_index(self.chunks, self.model), self.root)
        with self.assertRaisesRegex(IndexMismatchError, "commit hash"):
            load_index(self.root, expected=IndexExpectation(commit_hash="different"))

    def test_stale_index_detects_chunk_id_change(self) -> None:
        save_index(build_index(self.chunks, self.model), self.root)
        with self.assertRaisesRegex(IndexMismatchError, "ordered chunk IDs"):
            load_index(self.root, expected=IndexExpectation(ordered_chunk_ids=("other",)))

    def test_stale_index_detects_model_change(self) -> None:
        save_index(build_index(self.chunks, self.model), self.root)
        changed = replace(CONFIG, model_name="different/model")
        with self.assertRaisesRegex(IndexMismatchError, "embedding model"):
            load_index(self.root, expected=IndexExpectation(config=changed))

    def test_stale_index_detects_formatter_change(self) -> None:
        save_index(build_index(self.chunks, self.model), self.root)
        changed = replace(CONFIG, text_format_version="format-v2")
        with self.assertRaisesRegex(IndexMismatchError, "embedding text format"):
            load_index(self.root, expected=IndexExpectation(config=changed))

    def test_stale_index_detects_sequence_length_change(self) -> None:
        short = replace(CONFIG, max_sequence_length=512)
        save_index(build_index(self.chunks, FakeEmbeddingModel(short)), self.root)
        with self.assertRaisesRegex(IndexMismatchError, "maximum sequence length"):
            load_index(
                self.root,
                expected=IndexExpectation(
                    config=replace(short, max_sequence_length=8192)
                ),
            )

    def test_fingerprint_changes_with_sequence_length(self) -> None:
        short = build_index(
            self.chunks, FakeEmbeddingModel(replace(CONFIG, max_sequence_length=512))
        )
        long = build_index(
            self.chunks, FakeEmbeddingModel(replace(CONFIG, max_sequence_length=8192))
        )
        self.assertNotEqual(
            short.metadata.source_fingerprint, long.metadata.source_fingerprint
        )

    def test_no_truncation_is_recorded_within_limit(self) -> None:
        index = build_index(self.chunks, self.model)
        self.assertGreater(index.metadata.maximum_observed_document_tokens, 0)
        self.assertEqual(index.metadata.documents_over_model_limit, 0)
        self.assertEqual(index.metadata.documents_truncated, 0)
        self.assertEqual(index.metadata.truncated_chunk_ids, ())

    def test_over_limit_document_is_counted_deterministically(self) -> None:
        tiny = replace(CONFIG, max_sequence_length=5)
        first = build_index(self.chunks, FakeEmbeddingModel(tiny))
        second = build_index(self.chunks, FakeEmbeddingModel(tiny))
        self.assertEqual(first.metadata.documents_over_model_limit, 3)
        self.assertEqual(first.metadata.documents_truncated, 3)
        self.assertEqual(
            first.metadata.truncated_chunk_ids,
            tuple(chunk.chunk_id for chunk in self.chunks),
        )
        self.assertEqual(
            first.metadata.truncated_chunk_ids, second.metadata.truncated_chunk_ids
        )

    def test_exact_cosine_ranking(self) -> None:
        results = search(build_index(self.chunks, self.model), self.model, "query")
        self.assertEqual([item.chunk_id for item in results], ["chunk-b", "chunk-c", "chunk-a"])
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_configurable_top_k(self) -> None:
        index = build_index(self.chunks, self.model)
        self.assertEqual(len(search(index, self.model, "query", top_k=1)), 1)
        self.assertEqual(len(search(index, self.model, "query", top_k=3)), 3)
        self.assertEqual(len(search(index, self.model, "query", top_k=5)), 3)

    def test_deterministic_tie_breaking_uses_chunk_id(self) -> None:
        tied_chunks = (make_chunk("z-last"), make_chunk("a-first"))
        index = build_index(tied_chunks, self.model)
        index = replace(index, embeddings=np.asarray([[1.0, 0, 0], [1.0, 0, 0]], dtype=np.float32))
        results = search(index, self.model, "query")
        self.assertEqual([item.chunk_id for item in results], ["a-first", "z-last"])

    def test_search_returns_complete_chunk_metadata(self) -> None:
        result = search(build_index(self.chunks, self.model), self.model, "query", top_k=1)[0]
        chunk = self.chunks[0]
        self.assertEqual(
            (result.repository, result.file_path, result.language, result.chunk_type),
            (chunk.repository, chunk.file_path, chunk.language, chunk.chunk_type),
        )
        self.assertEqual((result.symbol_name, result.parent_symbol), ("alpha", "Service"))
        self.assertEqual((result.start_line, result.end_line, result.content), (10, 11, chunk.content))

    def test_empty_query_is_rejected(self) -> None:
        with self.assertRaisesRegex(VectorSearchError, "must not be empty"):
            search(build_index(self.chunks, self.model), self.model, "  ")

    def test_empty_index_build_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty chunk"):
            build_index((), self.model)

    def test_invalid_k_is_rejected(self) -> None:
        with self.assertRaisesRegex(VectorSearchError, "positive"):
            search(build_index(self.chunks, self.model), self.model, "query", top_k=0)

    def test_nan_or_infinite_query_embedding_is_rejected(self) -> None:
        model = FakeEmbeddingModel()
        model.encode_queries = lambda texts: np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float32)
        with self.assertRaisesRegex(VectorSearchError, "NaN or infinity"):
            search(build_index(self.chunks, model), model, "query")

    def test_same_fake_query_has_deterministic_ranking(self) -> None:
        index = build_index(self.chunks, self.model)
        first = search(index, self.model, "same query")
        second = search(index, self.model, "same query")
        self.assertEqual(first, second)

    def test_model_configuration_must_match_index(self) -> None:
        index = build_index(self.chunks, self.model)
        with self.assertRaisesRegex(VectorSearchError, "does not match"):
            search(index, FakeEmbeddingModel(replace(CONFIG, model_revision="other")), "query")


if __name__ == "__main__":
    unittest.main()
