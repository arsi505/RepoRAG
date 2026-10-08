from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

from reporag.ingestion.repository import (
    IngestionError,
    ingest_repository,
    is_github_url,
    prepare_github_repository,
)


def git_result(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["git"], returncode, stdout, stderr)


class GitHubIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        test_temp_root = Path.cwd() / ".test_tmp"
        test_temp_root.mkdir(exist_ok=True)
        self.temporary_directory = tempfile.TemporaryDirectory(dir=test_temp_root)
        self.root = Path(self.temporary_directory.name)
        self.cache_directory = self.root / "repository_cache"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_public_github_https_url_is_recognized(self) -> None:
        self.assertTrue(is_github_url("https://github.com/OpenAI/Example.git"))
        self.assertTrue(is_github_url("https://www.github.com/OpenAI/Example"))
        self.assertFalse(is_github_url("https://gitlab.com/OpenAI/Example.git"))

    @patch("reporag.ingestion.repository._run_git")
    def test_clone_uses_cache_path_and_derives_repository_name(self, run_git) -> None:
        run_git.return_value = git_result()
        source = "https://github.com/OpenAI/Example.git"
        expected_path = (self.cache_directory / "openai--example").resolve()

        local_path, repository_name = prepare_github_repository(
            source, self.cache_directory
        )

        self.assertEqual(repository_name, "Example")
        self.assertEqual(local_path, expected_path)
        self.assertTrue(self.cache_directory.is_dir())
        run_git.assert_called_once_with(
            [
                "clone",
                "--quiet",
                "https://github.com/OpenAI/Example.git",
                str(expected_path),
            ]
        )

    @patch("reporag.ingestion.repository._run_git")
    def test_clean_matching_cached_clone_is_reused(self, run_git) -> None:
        cached_path = self.cache_directory / "openai--example"
        cached_path.mkdir(parents=True)
        run_git.side_effect = [
            git_result(stdout="true\n"),
            git_result(stdout="https://github.com/OpenAI/Example.git\n"),
            git_result(stdout=""),
        ]

        local_path, repository_name = prepare_github_repository(
            "https://github.com/OpenAI/Example.git", self.cache_directory
        )

        self.assertEqual(local_path, cached_path.resolve())
        self.assertEqual(repository_name, "Example")
        self.assertEqual(
            run_git.call_args_list,
            [
                call(["rev-parse", "--is-inside-work-tree"], cwd=cached_path.resolve()),
                call(["remote", "get-url", "origin"], cwd=cached_path.resolve()),
                call(["status", "--porcelain"], cwd=cached_path.resolve()),
            ],
        )
        self.assertFalse(any("clone" in invocation.args[0] for invocation in run_git.call_args_list))

    @patch("reporag.ingestion.repository._run_git")
    def test_non_github_remote_url_is_rejected_without_git_or_network(self, run_git) -> None:
        with self.assertRaisesRegex(IngestionError, "public GitHub"):
            ingest_repository(
                "https://gitlab.com/OpenAI/Example.git",
                cache_directory=self.cache_directory,
            )

        run_git.assert_not_called()
        self.assertFalse(self.cache_directory.exists())


if __name__ == "__main__":
    unittest.main()
