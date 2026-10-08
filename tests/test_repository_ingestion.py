from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from reporag.ingestion.manifest import write_manifest
from reporag.ingestion.repository import discover_files, git_metadata, ingest_repository


class RepositoryIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        test_temp_root = Path.cwd() / ".test_tmp"
        test_temp_root.mkdir(exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=test_temp_root)
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_discovery_filters_directories_and_uses_relative_paths(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")
        (self.root / "node_modules").mkdir()
        (self.root / "node_modules" / "ignored.js").write_text("ignored", encoding="utf-8")
        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("ignored", encoding="utf-8")
        (self.root / "image.png").write_bytes(b"\x89PNG")

        files, skipped = discover_files(self.root)

        self.assertEqual([item.relative_path for item in files], ["src/main.py"])
        self.assertEqual([item.relative_path for item in skipped], ["image.png"])
        self.assertFalse(Path(files[0].relative_path).is_absolute())

    @unittest.skipUnless(shutil.which("git"), "Git executable is unavailable")
    def test_git_repository_metadata_is_obtained(self) -> None:
        subprocess.run(["git", "init", "--quiet"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "tests@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "RepoRAG Tests"], cwd=self.root, check=True)
        (self.root / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.py"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "test snapshot"], cwd=self.root, check=True)

        expected_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit_hash, branch = git_metadata(self.root)

        self.assertEqual(commit_hash, expected_commit)
        self.assertIsNotNone(branch)

    def test_non_git_directory_ingests_without_git_metadata(self) -> None:
        (self.root / "main.py").write_text("VALUE = 1\n", encoding="utf-8")
        manifest = ingest_repository(self.root)

        self.assertEqual(manifest.repository.source_type, "local")
        self.assertIsNone(manifest.repository.commit_hash)
        self.assertIsNone(manifest.repository.branch)
        self.assertEqual(manifest.repository.accepted_file_count, 1)

    def test_manifest_file_order_is_deterministic(self) -> None:
        (self.root / "z.py").write_text("Z = 1\n", encoding="utf-8")
        (self.root / "A.py").write_text("A = 1\n", encoding="utf-8")
        (self.root / "middle.py").write_text("M = 1\n", encoding="utf-8")

        manifest = ingest_repository(self.root)
        output_path = write_manifest(manifest, self.root / "manifests")
        payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            [item["relative_path"] for item in payload["files"]],
            ["A.py", "middle.py", "z.py"],
        )

    def test_oversized_and_non_utf8_files_are_skipped_with_reasons(self) -> None:
        (self.root / "large.py").write_text("12345", encoding="utf-8")
        (self.root / "encoded.py").write_bytes(b"\xff\xfe")

        files, skipped = discover_files(self.root, max_file_size_bytes=4)

        self.assertEqual(files, ())
        reasons = {item.relative_path: item.reason for item in skipped}
        self.assertEqual(reasons["large.py"], "exceeds_size_limit:4")
        self.assertEqual(reasons["encoded.py"], "not_utf8")

    def test_unreadable_file_is_skipped_without_aborting(self) -> None:
        unreadable = self.root / "unreadable.py"
        readable = self.root / "readable.py"
        unreadable.write_text("SECRET = True\n", encoding="utf-8")
        readable.write_text("OK = True\n", encoding="utf-8")
        original_read_bytes = Path.read_bytes

        def selective_read(path: Path) -> bytes:
            if path.name == "unreadable.py":
                raise PermissionError("test denial")
            return original_read_bytes(path)

        with patch.object(Path, "read_bytes", selective_read):
            files, skipped = discover_files(self.root)

        self.assertEqual([item.relative_path for item in files], ["readable.py"])
        self.assertEqual(skipped[0].relative_path, "unreadable.py")
        self.assertEqual(skipped[0].reason, "unreadable: PermissionError")


if __name__ == "__main__":
    unittest.main()
