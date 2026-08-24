"""Tests for ``scripts/check-whitespace.py`` and its CI enforcement site.

A whitespace defect reached a commit here because ``git diff --check`` had no
enforcement site at all: not a CI job, not a hook, not a script. These tests
pin the three properties that make the new gate real rather than decorative,
each written so that removing the property turns the test red:

* it is **range-scoped**, so pre-existing dirt in a file the change edits does
  not fail the change (the adoptability claim — without it the gate could not
  be switched on at blocking severity today);
* it inspects the **range, not the working tree** (bare ``git diff --check``
  reports clean on any CI checkout, which is the single easiest way for this
  gate to become a no-op that still shows green);
* an **unresolvable range fails**, it does not pass. A shallow clone with no
  merge base is the other way the gate silently degrades.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check-whitespace.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "whitespace-check.yml"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yaml"

EXIT_CLEAN = 0
EXIT_DEFECTS = 2
EXIT_UNRESOLVED = 3


def _env() -> dict[str, str]:
    """A git environment with no global/system config and a fixed identity.

    Without this the developer's own ``~/.gitconfig`` (``core.whitespace``,
    ``init.defaultBranch``, hooks) leaks into the fixture and the assertions
    stop describing the script.
    """
    env = dict(os.environ)
    env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "Whitespace Test",
            "GIT_AUTHOR_EMAIL": "whitespace@example.invalid",
            "GIT_COMMITTER_NAME": "Whitespace Test",
            "GIT_COMMITTER_EMAIL": "whitespace@example.invalid",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
        }
    )
    return env


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )


def _check(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """Invoke the gate script exactly as CI and contributors do."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(cwd),
        env=_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def _commit(root: Path, message: str) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo on ``main`` with one clean file, checked out on ``feature``.

    ``legacy.py`` deliberately carries BOTH defect shapes on ``main`` already —
    a trailing-whitespace line and a blank line at end of file — so every test
    below runs against a baseline that a tree-scoped linter would reject.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _write(root / "clean.py", "value = 1\n")
    _write(root / "legacy.py", "x = 1   \ny = 2\n\n")
    _commit(root, "baseline")
    _git(root, "checkout", "-b", "feature")
    return root


class TestVerdicts:
    def test_clean_range_passes(self, repo: Path) -> None:
        _write(repo / "clean.py", "value = 1\nother = 2\n")
        _commit(repo, "add a clean line")

        result = _check(repo)

        assert result.returncode == EXIT_CLEAN, result.stderr
        assert "No whitespace defects added" in result.stdout

    def test_trailing_whitespace_fails_and_names_file_and_line(
        self, repo: Path,
    ) -> None:
        _write(repo / "clean.py", "value = 1\nbad = 2   \n")
        _commit(repo, "add a line with trailing whitespace")

        result = _check(repo)

        assert result.returncode == EXIT_DEFECTS, result.stdout + result.stderr
        assert "clean.py:2: trailing whitespace." in result.stdout

    def test_blank_line_at_eof_fails(self, repo: Path) -> None:
        """The exact defect that motivated the gate: a heredoc-appended EOF newline."""
        _write(repo / "clean.py", "value = 1\nother = 2\n\n")
        _commit(repo, "append a blank line at EOF")

        result = _check(repo)

        assert result.returncode == EXIT_DEFECTS, result.stdout + result.stderr
        assert "clean.py:3: new blank line at EOF." in result.stdout

    def test_findings_exit_code_is_gits_own_two(self, repo: Path) -> None:
        """Pinning 2, not 1.

        ``git diff --check`` exits 2 on findings. The contract this gate is
        built on is that the code is PROPAGATED, never swallowed behind
        ``--exit-zero`` or ``|| true``; collapsing it to a generic 1 would hide
        a regression that starts reporting 1 for something else.
        """
        _write(repo / "clean.py", "value = 1\nbad = 2   \n")
        _commit(repo, "add a line with trailing whitespace")

        assert _check(repo).returncode == 2

    def test_failure_is_labelled_so_it_is_not_read_as_a_test_failure(
        self, repo: Path,
    ) -> None:
        _write(repo / "clean.py", "value = 1\nbad = 2   \n")
        _commit(repo, "add a line with trailing whitespace")

        result = _check(repo)

        assert "not a test failure" in result.stderr
        assert "scripts/check-whitespace.py" in result.stderr


class TestRangeScoped:
    def test_editing_a_file_that_is_already_dirty_still_passes(
        self, repo: Path,
    ) -> None:
        """The adoptability claim, as a test rather than an assertion.

        ``legacy.py`` already carries trailing whitespace on line 1 and a blank
        line at EOF. Adding a clean line to that very file must pass — if it did
        not, this gate could not be enabled without first reformatting the tree,
        which is precisely the change it exists to avoid.
        """
        _write(repo / "legacy.py", "x = 1   \ny = 2\nz = 3\n\n")
        _commit(repo, "add a clean line to a dirty file")

        result = _check(repo)

        assert result.returncode == EXIT_CLEAN, result.stdout + result.stderr

    def test_adding_a_dirty_line_to_a_dirty_file_fails(self, repo: Path) -> None:
        """The other half: the file's own history buys no amnesty for new dirt."""
        _write(repo / "legacy.py", "x = 1   \ny = 2\nz = 3\t \n\n")
        _commit(repo, "add a dirty line to a dirty file")

        result = _check(repo)

        assert result.returncode == EXIT_DEFECTS, result.stdout + result.stderr
        assert "legacy.py:3" in result.stdout

    def test_range_is_checked_even_though_the_working_tree_is_clean(
        self, repo: Path,
    ) -> None:
        """The trap that would make this gate a silent no-op.

        ``git diff --check`` with no arguments inspects UNSTAGED changes. On any
        CI checkout — and here, after committing — there are none, so the bare
        form reports clean while the range is dirty. This test fails the moment
        the script stops passing an explicit range.
        """
        _write(repo / "clean.py", "value = 1\nbad = 2   \n")
        _commit(repo, "add a line with trailing whitespace")

        bare = subprocess.run(
            ["git", "diff", "--check"],
            cwd=str(repo),
            env=_env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        assert bare.returncode == 0, "fixture is wrong: working tree is not clean"
        assert bare.stdout == ""

        assert _check(repo).returncode == EXIT_DEFECTS


class TestUnresolvableRangeFails:
    def test_missing_base_ref_exits_unresolved_not_clean(self, repo: Path) -> None:
        result = _check(repo, "--base-ref", "origin/does-not-exist")

        assert result.returncode == EXIT_UNRESOLVED, result.stdout + result.stderr
        assert "NOT a pass" in result.stderr

    def test_no_integration_branch_at_all_exits_unresolved(
        self, tmp_path: Path,
    ) -> None:
        """No ``main``, no ``master``, no remote: there is no range to check."""
        root = tmp_path / "orphan"
        root.mkdir()
        _git(root, "init", "-b", "wip")
        _write(root / "a.py", "a = 1\n")
        _commit(root, "only commit")

        result = _check(root)

        assert result.returncode == EXIT_UNRESOLVED, result.stdout + result.stderr
        assert "--base-ref" in result.stderr

    def test_unrelated_histories_exit_unresolved(self, repo: Path) -> None:
        _git(repo, "checkout", "--orphan", "disconnected")
        _git(repo, "rm", "-rf", "--cached", ".")
        _write(repo / "new.py", "n = 1\n")
        _commit(repo, "orphan root")

        result = _check(repo, "--base-ref", "main")

        assert result.returncode == EXIT_UNRESOLVED, result.stdout + result.stderr
        assert "no common ancestor" in result.stderr

    def test_shallow_clone_is_named_in_the_failure(self, tmp_path: Path) -> None:
        """A shallow clone has no merge base — the ``fetch-depth: 0`` trap.

        The message must say so, because the symptom (a gate that stops
        reporting) is indistinguishable from the gate simply passing.
        """
        origin = tmp_path / "origin"
        origin.mkdir()
        _git(origin, "init", "-b", "main")
        _write(origin / "a.py", "a = 1\n")
        _commit(origin, "one")
        _write(origin / "a.py", "a = 2\n")
        _commit(origin, "two")

        clone = tmp_path / "shallow"
        _git(tmp_path, "clone", "--depth", "1", origin.as_uri(), str(clone))
        assert (
            _git(clone, "rev-parse", "--is-shallow-repository").stdout.strip()
            == "true"
        )

        result = _check(clone, "--base-ref", "origin/does-not-exist")

        assert result.returncode == EXIT_UNRESOLVED, result.stdout + result.stderr
        assert "fetch-depth: 0" in result.stderr


class TestEnforcementSiteIsWired:
    """The script only bites if CI actually calls it, on a full history."""

    def test_workflow_runs_the_script(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "scripts/check-whitespace.py" in text

    def test_workflow_fetches_full_history(self) -> None:
        """Without this the merge base is unavailable and the gate degrades."""
        text = WORKFLOW.read_text(encoding="utf-8")
        assert "fetch-depth: 0" in text

    def test_workflow_does_not_swallow_the_exit_code(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        run_lines = [
            line
            for line in text.splitlines()
            if "check-whitespace.py" in line and not line.lstrip().startswith("#")
        ]
        assert run_lines, "no invocation of the gate script found in the workflow"
        for line in run_lines:
            assert "|| true" not in line
            assert "--exit-zero" not in line
            assert "continue-on-error" not in line

    def test_ci_requires_the_job_in_the_merge_gate(self) -> None:
        """Branch protection requires only ``all-checks-pass``; if the job is
        not in its ``needs``, a red whitespace check does not block a merge."""
        text = CI_WORKFLOW.read_text(encoding="utf-8")
        assert "uses: ./.github/workflows/whitespace-check.yml" in text

        gate = text.split("all-checks-pass:", 1)
        assert len(gate) == 2, "all-checks-pass job not found in ci.yaml"
        needs_block = gate[1].split("if: always()", 1)[0]
        assert "- whitespace-check" in needs_block
