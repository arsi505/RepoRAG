from __future__ import annotations

import unittest
from pathlib import Path

from reporag.ingestion.filters import (
    appears_binary,
    classify_supported_file,
    is_excluded_directory,
)


class IngestionFilterTests(unittest.TestCase):
    def test_supported_source_and_project_files_are_accepted(self) -> None:
        expected = {
            "main.py": (".py", "Python"),
            "component.tsx": (".tsx", "TypeScript TSX"),
            "settings.yaml": (".yaml", "YAML"),
            "Dockerfile": ("", "Dockerfile"),
            "Makefile": ("", "Makefile"),
        }
        for filename, classification in expected.items():
            with self.subTest(filename=filename):
                self.assertEqual(classify_supported_file(Path(filename)), classification)

    def test_unsupported_and_binary_files_are_rejected(self) -> None:
        self.assertIsNone(classify_supported_file(Path("program.exe")))
        self.assertTrue(appears_binary(b"text\x00binary"))
        self.assertFalse(appears_binary(b"plain UTF-8 text"))

    def test_excluded_directories_are_recognized(self) -> None:
        for directory in ("node_modules", ".git", "build", ".venv", ".test_tmp"):
            with self.subTest(directory=directory):
                self.assertTrue(is_excluded_directory(Path(directory)))
        self.assertTrue(is_excluded_directory(Path("data/manifests")))
        self.assertFalse(is_excluded_directory(Path("src/reporag")))


if __name__ == "__main__":
    unittest.main()
