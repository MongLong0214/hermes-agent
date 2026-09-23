#!/usr/bin/env python3
"""Regression test for contributor attribution PR base selection."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github/workflows/contributor-check.yml"


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def commit(cwd: Path, message: str, email: str) -> str:
    (cwd / "fixture.txt").write_text(message + "\n")
    git(cwd, "add", "fixture.txt")
    git(cwd, "-c", "user.name=Fixture", "-c", f"user.email={email}", "commit", "-m", message)
    return git(cwd, "rev-parse", "HEAD")


class ContributorCheckPrBaseTest(unittest.TestCase):
    def test_pull_request_uses_its_base_not_origin_main(self) -> None:
        """A release-based PR must not scan commits unique to its release base."""
        with tempfile.TemporaryDirectory() as tempdir:
            repo = Path(tempdir)
            git(repo, "init", "--initial-branch=main")
            root = commit(repo, "root", "root@example.invalid")
            commit(repo, "release base", "existing@example.invalid")
            release_base = commit(repo, "release rollup", "existing@example.invalid")
            git(repo, "branch", "release", "HEAD")
            git(repo, "checkout", "-B", "main", root)
            commit(repo, "main only", "main@example.invalid")
            git(repo, "checkout", "-b", "feature", "release")
            feature_head = commit(repo, "feature", "new@example.invalid")

            wrong_base = git(repo, "merge-base", "main", feature_head)
            wrong_emails = set(
                git(repo, "log", f"{wrong_base}..{feature_head}", "--format=%ae").splitlines()
            )
            right_emails = set(
                git(repo, "log", f"{release_base}..{feature_head}", "--format=%ae").splitlines()
            )

            self.assertIn("existing@example.invalid", wrong_emails)
            self.assertEqual(right_emails, {"new@example.invalid"})

        workflow = yaml.safe_load(WORKFLOW.read_text())
        check_step = workflow["jobs"]["check-attribution"]["steps"][1]
        self.assertEqual(
            check_step.get("env", {}).get("PR_BASE_SHA"),
            "${{ github.event.pull_request.base.sha || '' }}",
        )
        self.assertEqual(
            check_step["env"].get("EVENT_NAME"), "${{ github.event_name }}"
        )
        script = check_step["run"]
        self.assertIn('if [ "$EVENT_NAME" = "pull_request" ]; then', script)
        self.assertIn('[ -n "$PR_BASE_SHA" ] ||', script)
        self.assertIn('git cat-file -e "${PR_BASE_SHA}^{commit}"', script)
        self.assertIn('MERGE_BASE="$PR_BASE_SHA"', script)
        self.assertIn('MERGE_BASE=$(git merge-base origin/main HEAD)', script)


if __name__ == "__main__":
    unittest.main()
