from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from reporag.chunking.models import Chunk
from reporag.embeddings.config import DEFAULT_EMBEDDING_CONFIG
from reporag.embeddings.index import build_index as build_embedding_index
from reporag.lexical.index import build_index as build_bm25_index
from reporag.retrieval.hybrid import (
    DEFAULT_HYBRID_CONFIG,
    RRF_K,
    RRF_VERSION,
    HybridCompatibilityError,
    HybridConfig,
    corpus_fingerprint,
    create_hybrid_metadata,
    hybrid_fingerprint,
    save_hybrid_metadata,
    search,
    validate_index_compatibility,
)


def make_chunk(chunk_id: str, content: str, *, repository: str = "Repo") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        repository=repository,
        commit_hash="abc123",
        file_path=f"src/{chunk_id}.py",
        language="Python",
        chunk_type="function",
        symbol_name=chunk_id,
        parent_symbol=None,
        start_line=2,
        end_line=3,
        part_index=None,
        part_count=1,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content=content,
        parser_status="parsed",
    )


class FakeEmbeddingModel:
    config = DEFAULT_EMBEDDING_CONFIG

    def encode_documents(self, texts):
        rows = np.zeros((len(texts), self.config.embedding_dimension), dtype=np.float32)
        for position in range(len(texts)):
            rows[position, position] = 1.0
        return rows

    def encode_queries(self, texts):
        rows = np.zeros((len(texts), self.config.embedding_dimension), dtype=np.float32)
        rows[:, 0] = 1.0
        return rows

    def token_lengths(self, texts):
        return tuple(len(text.split()) for text in texts)


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = (
            make_chunk("connection", "def handle_connection(): realtime socket client"),
            make_chunk("presence", "def register_socket(): presence user socket"),
            make_chunk("other", "def total(): invoice amount"),
        )
        self.model = FakeEmbeddingModel()
        self.vector = build_embedding_index(self.chunks, self.model)
        self.bm25 = build_bm25_index(self.chunks)

    def test_frozen_rrf_configuration_is_visible(self) -> None:
        self.assertEqual(RRF_VERSION, "reporag-rrf-v1")
        self.assertEqual(RRF_K, 60)
        self.assertEqual(DEFAULT_HYBRID_CONFIG.method_name, "hybrid_rrf")
        self.assertIn("vector_method_version", DEFAULT_HYBRID_CONFIG.to_dict())
        self.assertIn("bm25_method_version", DEFAULT_HYBRID_CONFIG.to_dict())

    def test_compatible_indexes_succeed(self) -> None:
        self.assertEqual(
            validate_index_compatibility(self.vector, self.bm25),
            corpus_fingerprint(self.chunks),
        )

    def test_different_repository_is_rejected(self) -> None:
        bm25 = replace(
            self.bm25,
            metadata=replace(self.bm25.metadata, repository="OtherRepo"),
        )
        with self.assertRaisesRegex(HybridCompatibilityError, "repository"):
            validate_index_compatibility(self.vector, bm25)

    def test_different_commit_is_rejected(self) -> None:
        bm25 = replace(
            self.bm25, metadata=replace(self.bm25.metadata, commit_hash="different")
        )
        with self.assertRaisesRegex(HybridCompatibilityError, "commit"):
            validate_index_compatibility(self.vector, bm25)

    def test_different_chunk_count_is_rejected(self) -> None:
        bm25 = replace(
            self.bm25, metadata=replace(self.bm25.metadata, chunk_count=999)
        )
        with self.assertRaisesRegex(HybridCompatibilityError, "chunk count"):
            validate_index_compatibility(self.vector, bm25)

    def test_different_ordered_chunk_ids_are_rejected(self) -> None:
        bm25 = replace(
            self.bm25,
            metadata=replace(
                self.bm25.metadata,
                ordered_chunk_ids=tuple(reversed(self.bm25.metadata.ordered_chunk_ids)),
            ),
        )
        with self.assertRaisesRegex(HybridCompatibilityError, "ordered chunk IDs"):
            validate_index_compatibility(self.vector, bm25)

    def test_different_corpus_fingerprint_is_rejected(self) -> None:
        changed = replace(self.bm25.chunks[0], content_hash="0" * 64)
        bm25 = replace(self.bm25, chunks=(changed, *self.bm25.chunks[1:]))
        with self.assertRaisesRegex(HybridCompatibilityError, "corpus fingerprint"):
            validate_index_compatibility(self.vector, bm25)

    def test_search_retains_contributions_and_chunk_metadata(self) -> None:
        outcome = search(self.vector, self.bm25, self.model, "realtime socket", top_k=3)
        result = outcome.results[0]
        self.assertEqual(result.retrieval_method, "hybrid_rrf")
        self.assertEqual(result.score_type, "rrf")
        self.assertIsNotNone(result.vector_rank)
        self.assertIsNotNone(result.vector_raw_score)
        self.assertIsNotNone(result.bm25_rank)
        self.assertIsNotNone(result.bm25_raw_score)
        self.assertAlmostEqual(
            result.rrf_score,
            result.vector_contribution + result.bm25_contribution,
        )
        self.assertEqual(result.repository, "Repo")
        self.assertEqual(result.file_path, "src/connection.py")
        self.assertEqual((result.symbol_name, result.start_line, result.end_line), ("connection", 2, 3))
        self.assertFalse(Path(result.file_path).is_absolute())

    def test_search_is_deterministic_and_supports_top_k(self) -> None:
        first = search(self.vector, self.bm25, self.model, "socket", top_k=1)
        second = search(self.vector, self.bm25, self.model, "socket", top_k=1)
        self.assertEqual(first.results, second.results)
        self.assertEqual(len(first.results), 1)

    def test_metadata_is_deterministic_without_paths_or_timestamps(self) -> None:
        first = create_hybrid_metadata(self.vector, self.bm25)
        second = create_hybrid_metadata(self.vector, self.bm25)
        self.assertEqual(first, second)
        encoded = json.dumps(first.to_dict())
        self.assertNotIn("created_at", encoded)
        self.assertNotIn("D:\\\\", encoded)
        self.assertNotIn("C:\\\\", encoded)

    def test_metadata_save_is_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd() / ".test_tmp") as temporary:
            path = save_hybrid_metadata(
                create_hybrid_metadata(self.vector, self.bm25), Path(temporary)
            )
            first = path.read_bytes()
            save_hybrid_metadata(create_hybrid_metadata(self.vector, self.bm25), Path(temporary))
            self.assertEqual(path.read_bytes(), first)

    def test_fingerprint_is_stable_for_identical_configuration(self) -> None:
        self.assertEqual(
            create_hybrid_metadata(self.vector, self.bm25).hybrid_fingerprint,
            create_hybrid_metadata(self.vector, self.bm25).hybrid_fingerprint,
        )

    def test_fingerprint_changes_with_rrf_k(self) -> None:
        default = create_hybrid_metadata(self.vector, self.bm25).hybrid_fingerprint
        changed = create_hybrid_metadata(
            self.vector, self.bm25, config=HybridConfig(rrf_k=10)
        ).hybrid_fingerprint
        self.assertNotEqual(default, changed)

    def test_fingerprint_changes_with_each_input_identity(self) -> None:
        base = dict(
            repository="Repo",
            commit_hash="abc123",
            corpus_identity="corpus",
            vector_fingerprint="vector",
            bm25_fingerprint="bm25",
        )
        original = hybrid_fingerprint(**base)
        for field in (
            "commit_hash",
            "corpus_identity",
            "vector_fingerprint",
            "bm25_fingerprint",
        ):
            changed = dict(base)
            changed[field] += "-changed"
            with self.subTest(field=field):
                self.assertNotEqual(original, hybrid_fingerprint(**changed))


if __name__ == "__main__":
    unittest.main()
