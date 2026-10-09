from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reporag.chunking.chunker import ChunkingConfig, chunk_repository
from reporag.ingestion.repository import ingest_repository


class StructuralChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        test_temp_root = Path.cwd() / ".test_tmp"
        test_temp_root.mkdir(exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=test_temp_root)
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def chunk_source(
        self,
        filename: str,
        source: str,
        *,
        config: ChunkingConfig = ChunkingConfig(),
    ):
        source_path = self.root / filename
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(source.encode("utf-8"))
        manifest = ingest_repository(self.root)
        return chunk_repository(manifest, config=config).chunks

    def structural(self, chunks):
        return [chunk for chunk in chunks if chunk.chunk_type != "file"]

    def test_typescript_function_extraction_and_line_metadata(self) -> None:
        source = (
            "const version = 1;\n"
            "export function calculateBackoff(attempt: number) {\n"
            "  return attempt * 2;\n"
            "}\n"
        )
        chunks = self.structural(self.chunk_source("retry.ts", source))

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.chunk_type, "function")
        self.assertEqual(chunk.symbol_name, "calculateBackoff")
        self.assertEqual((chunk.start_line, chunk.end_line), (2, 4))
        self.assertEqual(
            chunk.content,
            "function calculateBackoff(attempt: number) {\n  return attempt * 2;\n}",
        )

    def test_typescript_named_arrow_function_extraction(self) -> None:
        chunks = self.structural(
            self.chunk_source(
                "backoff.ts",
                "const calculateBackoff = (attempt: number) => attempt * 2;\n",
            )
        )
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [("function", "calculateBackoff")],
        )

    def test_typescript_exported_named_arrow_function_extraction(self) -> None:
        chunks = self.structural(
            self.chunk_source(
                "exports.ts",
                "export const loadWorkspace = async (id: string) => ({ id });\n",
            )
        )
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [("function", "loadWorkspace")],
        )

    def test_javascript_named_function_expression_without_callback_fabrication(self) -> None:
        source = (
            "const buildJob = function () { return { ready: true }; };\n"
            "const results = items.map(item => item.id);\n"
        )
        chunks = self.structural(self.chunk_source("factory.js", source))
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [("function", "buildJob")],
        )

    def test_tsx_and_jsx_named_arrow_components(self) -> None:
        tsx_chunks = self.structural(
            self.chunk_source(
                "user-card.tsx",
                "export const UserCard: React.FC<Props> = (props) => <div>{props.name}</div>;\n",
            )
        )
        jsx_chunks = self.structural(
            self.chunk_source(
                "user-panel.jsx",
                "export const UserPanel = (props) => <section>{props.name}</section>;\n",
            )
        )
        self.assertTrue(any(chunk.symbol_name == "UserCard" for chunk in tsx_chunks))
        self.assertTrue(any(chunk.symbol_name == "UserPanel" for chunk in jsx_chunks))

    def test_typescript_class_methods_preserve_parent(self) -> None:
        source = (
            "class Worker {\n"
            "  constructor() {}\n"
            "  async retryJob(id: string) {\n"
            "    return id;\n"
            "  }\n"
            "  @Trace()\n"
            "  private async validatePayload<T extends object>(raw: T) {\n"
            "    return raw;\n"
            "  }\n"
            "}\n"
        )
        chunks = self.structural(self.chunk_source("worker.ts", source))

        self.assertEqual(
            [chunk.chunk_type for chunk in chunks], ["constructor", "method", "method"]
        )
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks],
            ["constructor", "retryJob", "validatePayload"],
        )
        self.assertTrue(all(chunk.parent_symbol == "Worker" for chunk in chunks))
        self.assertFalse(any(chunk.chunk_type == "class" for chunk in chunks))

    def test_multimethod_class_does_not_duplicate_full_class(self) -> None:
        source = (
            "class Scheduler {\n"
            "  start() { return this.worker.register(); }\n"
            "  stop() { return this.worker.disconnect(); }\n"
            "}\n"
        )
        chunks = self.structural(self.chunk_source("scheduler.ts", source))

        self.assertEqual([chunk.symbol_name for chunk in chunks], ["start", "stop"])
        self.assertTrue(all(chunk.chunk_type == "method" for chunk in chunks))
        self.assertFalse(any(chunk.chunk_type == "class" for chunk in chunks))

    def test_javascript_function_extraction(self) -> None:
        chunks = self.structural(
            self.chunk_source("jobs.js", "function enqueue(job) {\n  return job;\n}\n")
        )
        self.assertEqual([(chunk.chunk_type, chunk.symbol_name) for chunk in chunks], [("function", "enqueue")])

    def test_tsx_and_jsx_use_structural_parsers(self) -> None:
        tsx_chunks = self.structural(
            self.chunk_source("view.tsx", "export function View() {\n  return <div />;\n}\n")
        )
        jsx_chunks = self.structural(
            self.chunk_source("panel.jsx", "function Panel() {\n  return <section />;\n}\n")
        )

        self.assertTrue(any(chunk.symbol_name == "View" for chunk in tsx_chunks))
        self.assertTrue(any(chunk.symbol_name == "Panel" for chunk in jsx_chunks))

    def test_c_function_extraction(self) -> None:
        chunks = self.structural(
            self.chunk_source("math.c", "int add(int a, int b) {\n  return a + b;\n}\n")
        )
        self.assertEqual([(chunk.chunk_type, chunk.symbol_name) for chunk in chunks], [("function", "add")])

    def test_cpp_class_method_and_constructor_extraction(self) -> None:
        source = (
            "class Worker {\n"
            "public:\n"
            "  Worker() {}\n"
            "  int retry(int value) {\n"
            "    return value;\n"
            "  }\n"
            "};\n"
        )
        chunks = self.structural(self.chunk_source("worker.cpp", source))

        self.assertEqual([chunk.chunk_type for chunk in chunks], ["constructor", "method"])
        self.assertEqual([chunk.symbol_name for chunk in chunks], ["Worker", "retry"])
        self.assertTrue(all(chunk.parent_symbol == "Worker" for chunk in chunks))

    def test_python_function_class_and_method_extraction(self) -> None:
        source = (
            "class RecoveryManager:\n"
            "    def __init__(self):\n"
            "        self.ready = True\n"
            "\n"
            "    def recover(self, job):\n"
            "        return job\n"
            "\n"
            "def helper(value):\n"
            "    return value\n"
        )
        chunks = self.structural(self.chunk_source("recovery.py", source))

        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name, chunk.parent_symbol) for chunk in chunks],
            [
                ("constructor", "__init__", "RecoveryManager"),
                ("method", "recover", "RecoveryManager"),
                ("function", "helper", None),
            ],
        )

    def test_oversized_structural_chunk_is_split_with_stable_symbol_metadata(self) -> None:
        source = "def long_task():\n" + "".join(
            f"    value_{index} = {index}\n" for index in range(1, 8)
        )
        chunks = self.structural(
            self.chunk_source(
                "large.py",
                source,
                config=ChunkingConfig(max_structural_lines=3),
            )
        )

        self.assertEqual(len(chunks), 3)
        self.assertEqual([chunk.part_index for chunk in chunks], [1, 2, 3])
        self.assertTrue(all(chunk.part_count == 3 for chunk in chunks))
        self.assertTrue(all(chunk.symbol_name == "long_task" for chunk in chunks))
        self.assertEqual("".join(chunk.content for chunk in chunks), source.rstrip("\n"))


if __name__ == "__main__":
    unittest.main()
