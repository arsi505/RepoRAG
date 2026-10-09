from __future__ import annotations

import hashlib
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from reporag.chunking.chunker import ChunkingConfig, chunk_repository
from reporag.chunking.languages import STRUCTURAL_LANGUAGES_BY_EXTENSION
from reporag.chunking.writer import write_chunks
from reporag.ingestion.repository import ingest_repository


class ChunkingInvariantTests(unittest.TestCase):
    def setUp(self) -> None:
        test_temp_root = Path.cwd() / ".test_tmp"
        test_temp_root.mkdir(exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=test_temp_root)
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def chunk_files(self, files: dict[str, str], *, config: ChunkingConfig = ChunkingConfig()):
        for filename, content in files.items():
            path = self.root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content.encode("utf-8"))
        manifest = ingest_repository(self.root)
        return manifest, chunk_repository(manifest, config=config)

    def test_chunk_ids_and_content_hashes_are_deterministic(self) -> None:
        manifest, first = self.chunk_files({"main.py": "def run():\n    return 1\n"})
        second = chunk_repository(manifest)

        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunks],
            [chunk.chunk_id for chunk in second.chunks],
        )
        for chunk in first.chunks:
            expected_hash = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            self.assertEqual(chunk.content_hash, expected_hash)

    def test_required_language_pack_grammars_are_declared(self) -> None:
        with Path("language-pack.toml").open("rb") as handle:
            configuration = tomllib.load(handle)

        self.assertEqual(
            configuration["tree-sitter-language-pack"]["languages"],
            ["python", "typescript", "tsx", "javascript", "c", "cpp"],
        )
        self.assertEqual(
            STRUCTURAL_LANGUAGES_BY_EXTENSION[".jsx"].grammar,
            "javascript",
        )

    def test_output_order_is_deterministic(self) -> None:
        manifest, result = self.chunk_files(
            {
                "zeta.py": "def zed():\n    return 1\n",
                "Alpha.py": "def alpha():\n    return 1\n",
            }
        )
        output = write_chunks(result, manifest.repository.name, self.root / "output")
        records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual([record["file_path"] for record in records], ["Alpha.py", "zeta.py"])

    def test_unsupported_language_uses_overlapping_fallback_chunks(self) -> None:
        content = "".join(f'{{"line": {line}}}\n' for line in range(1, 96))
        _, result = self.chunk_files(
            {"settings.json": content},
            config=ChunkingConfig(fallback_max_lines=80, fallback_overlap_lines=10),
        )

        self.assertEqual(len(result.chunks), 2)
        first, second = result.chunks
        self.assertEqual((first.start_line, first.end_line), (1, 80))
        self.assertEqual((second.start_line, second.end_line), (71, 95))
        self.assertEqual(first.part_index, 1)
        self.assertEqual(second.part_index, 2)
        self.assertTrue(all(chunk.part_count == 2 for chunk in result.chunks))
        self.assertTrue(
            all(chunk.parser_status == "fallback_unsupported_language" for chunk in result.chunks)
        )

    def test_malformed_source_does_not_crash(self) -> None:
        _, result = self.chunk_files(
            {"broken.ts": "function incomplete( {\n  const value = ;\n"}
        )

        self.assertGreaterEqual(len(result.chunks), 1)
        self.assertTrue(any("syntax_errors" in chunk.parser_status for chunk in result.chunks))

    def test_source_without_symbols_is_preserved_by_fallback(self) -> None:
        source = "SETTING = 1\nOTHER = SETTING + 1\n"
        _, result = self.chunk_files({"settings.py": source})

        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].chunk_type, "file")
        self.assertEqual(result.chunks[0].content, source)
        self.assertEqual(result.chunks[0].parser_status, "fallback_no_structures")

    def test_chunk_records_do_not_include_absolute_machine_paths(self) -> None:
        _, result = self.chunk_files({"main.py": "def run():\n    return 1\n"})

        for chunk in result.chunks:
            record = chunk.to_dict()
            self.assertNotIn("local_path", record)
            self.assertFalse(Path(record["file_path"]).is_absolute())
            self.assertNotIn(str(self.root), json.dumps(record))


if __name__ == "__main__":
    unittest.main()
