"""Tests for tests/git_mutation_guard.py — the D5 git-mutation containment guard.

RED reproduction of the recorded incident: a destructive git command reaches
a live/shared worktree through a literal path that differs from the
worktree's canonical path (here: a symlink) but resolves to the identical
git identity (toplevel/git-dir/common-dir). The guard must refuse it, fail
closed, before the mutation runs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests import git_mutation_guard
from tools import self_repo_guard


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _init(path: Path) -> Path:
    """The only thing a test needs to create a disposable git repo: a plain
    ``git init``, exactly like any of the ~60 existing test files that
    aren't part of this guard's own test suite. No registration call of any
    kind -- that is the point of the protected-set redesign this module
    tests.
    """
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    return path


@pytest.fixture
def shared_repo_with_linked_worktree(tmp_path):
    """A repo with a second, *linked* worktree — the incident's exact shape.

    Returns ``(shared, linked, alias)``: ``shared`` is the "main" checkout
    -- a plain, unregistered ``git init`` -- ``linked`` is a worktree
    ``git worktree add`` registered to it (this is what made the original
    incident possible — another *active* worktree of the same repository,
    which plays the role of "a live/shared worktree" for these tests via
    ``reaimed_registry`` below), and ``alias`` is a symlink to ``linked``:
    same git identity, a different literal path — the shape the strict
    identity check exists for.
    """
    shared = tmp_path / "shared"
    _init(shared)
    (shared / "f.txt").write_text("base\n")
    _git(shared, "add", "f.txt")
    _git(shared, "commit", "-qm", "init")

    linked = tmp_path / "linked"
    _git(shared, "worktree", "add", str(linked), "-b", "feature")

    alias = tmp_path / "alias"
    alias.symlink_to(linked)

    return shared, linked, alias


@pytest.fixture
def reaimed_registry(monkeypatch, shared_repo_with_linked_worktree):
    """Re-aim the guard's protected set (``_compute_known_identities``) at
    the synthetic pair above.

    Mirrors exactly how the real protected set is built -- anchored on
    ``_CALLER_ROOT_HINT`` -- just pointed at a disposable repo instead of
    the real ambient checkout, so the RED test is self-contained and
    doesn't depend on (or risk) the real worktree list. ``shared`` and
    ``linked`` become "the caller's own worktree" and "a worktree linked to
    it" respectively -- both members of the protected set -- with no
    registration call anywhere.
    """
    shared, linked, alias = shared_repo_with_linked_worktree
    monkeypatch.setattr(git_mutation_guard, "_CALLER_ROOT_HINT", shared)
    monkeypatch.setattr(git_mutation_guard, "_known_identities_cache", None)
    return shared, linked, alias


class TestRedIncidentReproduction:
    """Reproduces the recorded incident: destructive git aimed at a live worktree."""

    def test_destructive_checkout_through_an_alias_path_is_refused_before_it_runs(
        self, reaimed_registry
    ):
        shared, linked, alias = reaimed_registry

        # Uncommitted work in the linked worktree — exactly what the
        # incident destroyed.
        (linked / "f.txt").write_text("UNCOMMITTED EDIT\n")

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(alias), "checkout", "--", "."],
                capture_output=True,
                text=True,
            )

        # Fail-closed, not fail-noticed: the mutation must never have run.
        assert (linked / "f.txt").read_text() == "UNCOMMITTED EDIT\n", (
            "the guard raised, but the checkout still executed and wiped "
            "the uncommitted edit — that would be fail-noticed, not "
            "fail-closed"
        )

    def test_destructive_reset_hard_through_the_alias_is_also_refused(
        self, reaimed_registry
    ):
        shared, linked, alias = reaimed_registry
        (linked / "f.txt").write_text("UNCOMMITTED EDIT 2\n")

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(alias), "reset", "--hard"],
                capture_output=True,
                text=True,
            )

        assert (linked / "f.txt").read_text() == "UNCOMMITTED EDIT 2\n"

    def test_destructive_git_dir_override_from_an_unrelated_cwd_is_refused(
        self, reaimed_registry, tmp_path
    ):
        """The flagship RED: no path in the argv or cwd names the victim.

        Verified live: `git --git-dir=<linked's .git> reset --hard`, run
        from an unrelated, non-git scratch directory with no `-C`/
        `--work-tree` at all, still moves `linked`'s branch pointer. This
        is blocked unconditionally: an explicit `--git-dir` disqualifies
        the "plain, standard-shape" allow path outright (see
        `_standard_shape_violation`), regardless of what it would have
        resolved to.
        """
        shared, linked, alias = reaimed_registry
        # A second commit on `feature` so a rollback is observable.
        (linked / "g.txt").write_text("second\n")
        _git(linked, "add", "g.txt")
        _git(linked, "commit", "-qm", "second")

        linked_git_dir = subprocess.run(
            ["git", "-C", str(linked), "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        before = _git(linked, "rev-parse", "feature").stdout.strip()

        scratch = tmp_path / "unrelated-scratch"
        scratch.mkdir()

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", f"--git-dir={linked_git_dir}", "reset", "--hard", "HEAD~1"],
                cwd=scratch,
                capture_output=True,
                text=True,
            )

        after = _git(linked, "rev-parse", "feature").stdout.strip()
        assert after == before, (
            "the guard raised, but `feature`'s branch pointer moved anyway "
            "-- fail-noticed, not fail-closed"
        )

    def test_c_flag_at_an_unrelated_non_repo_combined_with_git_dir_is_refused(
        self, reaimed_registry, tmp_path
    ):
        """Empirically-confirmed bypass #1 (sol round 4): `-C <non-repo>
        --git-dir=<live-git-dir>` -- an old top-level-flag walk consumed
        `-C`'s own value as if it had already reached the subcommand, so it
        never even looked far enough to see the `--git-dir=` that followed
        and let the mutation straight through (confirmed live: it moved
        `feature`'s branch pointer). This is closed without needing to fix
        that walk at all: `--git-dir` anywhere in the top-level flags
        disqualifies the plain-shape allow path outright, independent of
        `-C`.
        """
        shared, linked, alias = reaimed_registry
        (linked / "g.txt").write_text("second\n")
        _git(linked, "add", "g.txt")
        _git(linked, "commit", "-qm", "second")

        linked_git_dir = subprocess.run(
            ["git", "-C", str(linked), "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        before = _git(linked, "rev-parse", "feature").stdout.strip()

        non_repo = tmp_path / "unrelated-non-repo"
        non_repo.mkdir()

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(non_repo),
                    f"--git-dir={linked_git_dir}",
                    "reset",
                    "--hard",
                    "HEAD~1",
                ],
                capture_output=True,
                text=True,
            )

        after = _git(linked, "rev-parse", "feature").stdout.strip()
        assert after == before, (
            "the guard raised, but `feature`'s branch pointer moved anyway "
            "-- this is the exact bypass that forced the protected-set redesign"
        )

    def test_alias_wrapped_mutation_is_refused(self, reaimed_registry):
        """Empirically-confirmed bypass #2 (sol round 4): `-c
        alias.nuke='reset --hard' nuke` -- the shared parser extracts
        inline aliases but an older version of this module discarded that
        result; `nuke` isn't a recognized mutating subcommand, so the call
        sailed through unclassified (confirmed live: it moved `feature`'s
        branch pointer). This is closed by *resolving* the alias -- reusing
        `tools.self_repo_guard`'s own alias lookup -- rather than
        blanket-blocking every call that merely defines an inline alias:
        `nuke` resolves to `reset --hard`, which is then classified and
        target-checked exactly like a literal `git reset --hard` would be.
        """
        shared, linked, alias = reaimed_registry
        (linked / "g.txt").write_text("second\n")
        _git(linked, "add", "g.txt")
        _git(linked, "commit", "-qm", "second")

        before = _git(linked, "rev-parse", "feature").stdout.strip()

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(linked), "-c", "alias.nuke=reset --hard HEAD~1", "nuke"],
                capture_output=True,
                text=True,
            )

        after = _git(linked, "rev-parse", "feature").stdout.strip()
        assert after == before, (
            "the guard raised, but `feature`'s branch pointer moved anyway "
            "-- this is the exact bypass that forced the protected-set redesign"
        )

    def test_worktree_remove_aimed_at_the_linked_worktree_is_refused(
        self, reaimed_registry
    ):
        """`worktree remove/move` names its victim as an argument, not `cwd`."""
        shared, linked, alias = reaimed_registry

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(shared), "worktree", "remove", "--force", str(alias)],
                capture_output=True,
                text=True,
            )

        assert linked.exists(), "the linked worktree must still be on disk"

    def test_worktree_add_destination_aliasing_a_live_worktree_is_refused(
        self, reaimed_registry
    ):
        """`worktree add`'s destination is a separate argument from the
        source repo's identity -- it must also be checked, not just the
        source. Here the destination is `linked` itself, already a live
        worktree of `shared` (the protected set's live anchor under
        `reaimed_registry`) -- destination validation must reject it
        exactly like a fresh disposable repo's identity check would.
        """
        shared, linked, alias = reaimed_registry

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(shared), "worktree", "add", str(linked), "-b", "another"],
                capture_output=True,
                text=True,
            )

    def test_non_mutating_git_calls_through_the_same_alias_are_unaffected(
        self, reaimed_registry
    ):
        """Control: the guard targets mutation, not the alias/worktree itself."""
        shared, linked, alias = reaimed_registry
        result = subprocess.run(
            ["git", "-C", str(alias), "status", "--porcelain"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestAliasResolutionIsNotABlanketBlock:
    """The prior allowlist round blocked every call that merely *defined* an
    inline alias, regardless of what it resolved to or where it was aimed.
    The protected-set redesign resolves the alias instead and classifies
    the resolved command -- these are the controls proving that's really
    what happens, not just a narrower-looking blanket block.
    """

    def test_inline_alias_resolving_to_a_mutation_aimed_elsewhere_is_allowed(
        self, reaimed_registry, tmp_path
    ):
        """An inline `-c alias.*=` definition by itself must not disqualify
        a call whose *resolved* target isn't protected at all."""
        shared, linked, alias = reaimed_registry
        other = tmp_path / "unrelated-disposable"
        _init(other)
        (other / "f.txt").write_text("x\n")
        _git(other, "add", "f.txt")
        _git(other, "commit", "-qm", "init")
        _git(other, "checkout", "-b", "feature")

        result = subprocess.run(
            ["git", "-C", str(other), "-c", "alias.nuke=reset --hard", "nuke"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_inline_alias_resolving_to_a_safe_command_in_the_protected_repo_is_allowed(
        self, reaimed_registry
    ):
        """An alias that resolves to a non-mutating command must be allowed
        even when it's defined inline and aimed at the protected repo --
        proving the gate is about the *resolved* classification, not the
        mere presence of an inline alias."""
        shared, linked, alias = reaimed_registry
        result = subprocess.run(
            ["git", "-C", str(linked), "-c", "alias.st=status", "st", "--porcelain"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_configured_alias_resolving_to_a_mutation_is_refused(self, reaimed_registry):
        """Not just inline `-c alias.*=` -- an alias configured in the
        target repo's own git config (`git config alias.<name> <value>`) is
        resolved the same way, reusing
        `tools.self_repo_guard._read_git_alias`."""
        shared, linked, alias = reaimed_registry
        subprocess.run(
            ["git", "-C", str(linked), "config", "alias.nuke", "reset --hard"],
            check=True,
        )
        (linked / "g.txt").write_text("second\n")
        _git(linked, "add", "g.txt")
        _git(linked, "commit", "-qm", "second")
        before = _git(linked, "rev-parse", "feature").stdout.strip()

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(linked), "nuke"],
                capture_output=True,
                text=True,
            )

        after = _git(linked, "rev-parse", "feature").stdout.strip()
        assert after == before

    def test_shell_form_alias_is_refused_even_aimed_at_an_unprotected_repo(
        self, reaimed_registry, tmp_path
    ):
        """A shell-form (`!...`) alias can't be statically analyzed -- it is
        refused regardless of where it's aimed, not just when the target
        happens to be protected."""
        other = tmp_path / "unrelated-disposable-2"
        _init(other)

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(other), "-c", "alias.nuke=!rm -rf .", "nuke"],
                capture_output=True,
                text=True,
            )

    def test_unresolvable_alias_cycle_is_refused_by_the_recursion_limit(
        self, tmp_path
    ):
        """An alias chain that never bottoms out into a recognized
        subcommand -- here, two aliases pointing at each other -- must fail
        closed once it exceeds `tools.self_repo_guard._MAX_RECURSION` hops,
        even when the target is an ordinary, unprotected disposable repo.
        Ambiguity itself is refused, independent of target safety.
        """
        repo = tmp_path / "cycle-disposable"
        _init(repo)
        subprocess.run(
            ["git", "-C", str(repo), "config", "alias.loop1", "loop2"], check=True
        )
        subprocess.run(
            ["git", "-C", str(repo), "config", "alias.loop2", "loop1"], check=True
        )

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(repo), "loop1"],
                capture_output=True,
                text=True,
            )


class TestGuardAllowsOrdinaryUnregisteredRepos:
    """D5 protected-set redesign, item 3: an ordinary git operation against
    a fresh, unrelated temp repo must work exactly as it did before any of
    this guard's rounds, with no registration call of any kind -- this is
    what lets the ~60 existing test files keep creating and mutating their
    own disposable repos untouched.
    """

    def test_git_init_directly_against_a_fresh_path_is_allowed(self, tmp_path):
        """The exact shape the allowlist round refused outright: a test
        calling `git init` on its own, with no fixture/registration call at
        all. Must now be allowed -- this is the shape ~60 files use."""
        repo = tmp_path / "not-registered"
        repo.mkdir()
        result = subprocess.run(
            ["git", "init", "-q", "-b", "main", str(repo)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (repo / ".git").exists()

    def test_git_init_with_no_explicit_cwd_or_path_arg_is_allowed(self, tmp_path):
        """`git init` run via `cwd=`, no positional path argument -- the
        most common shape across the existing test suite's fixtures."""
        repo = tmp_path / "cwd-style"
        repo.mkdir()
        result = subprocess.run(
            ["git", "init"], cwd=repo, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    def test_checkout_in_a_fresh_disposable_repo_is_allowed(self, tmp_path):
        repo = tmp_path / "disposable"
        _init(repo)
        (repo / "a.txt").write_text("x\n")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-qm", "init")
        _git(repo, "checkout", "-b", "feature")

        result = subprocess.run(
            ["git", "-C", str(repo), "checkout", "main"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_worktree_add_from_a_fresh_disposable_repo_is_allowed(self, tmp_path):
        repo = tmp_path / "disposable2"
        _init(repo)
        (repo / "a.txt").write_text("x\n")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-qm", "init")

        extra = tmp_path / "disposable2-worktree"
        result = subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", str(extra), "-b", "wt"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_git_clone_of_a_fresh_disposable_repo_is_allowed(self, tmp_path):
        """`init`/`clone` resolve their destination from the subcommand's
        own positional argument (see `_bootstrap_target`), not from `cwd` --
        this is the shape that broke before that resolution existed."""
        src = tmp_path / "clone-src"
        _init(src)
        (src / "a.txt").write_text("x\n")
        _git(src, "add", "a.txt")
        _git(src, "commit", "-qm", "init")

        dest = tmp_path / "clone-dest"
        result = subprocess.run(
            ["git", "clone", "-q", str(src), str(dest)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (dest / ".git").exists()

    def test_unclassified_real_subcommand_is_allowed(self, tmp_path):
        """`update-ref` is a real, non-mutating-per-`_mutates_worktree` git
        subcommand this module does not specifically classify -- it must
        not be refused merely for being unrecognized (that regressed ~60
        existing test files under the allowlist round)."""
        repo = tmp_path / "update-ref-disposable"
        _init(repo)
        (repo / "a.txt").write_text("x\n")
        _git(repo, "add", "a.txt")
        _git(repo, "commit", "-qm", "init")

        result = subprocess.run(
            ["git", "-C", str(repo), "update-ref", "refs/heads/other", "HEAD"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


class TestGuardIsSessionScopedNotFunctionScoped:
    """D5 gap 1: the fixture must protect the whole session, not just a test body.

    A function-scoped autouse fixture only patches ``Popen.__init__`` during
    the *function*-scoped portion of a test's setup/call/teardown -- any
    session/module/class-scoped fixture elsewhere runs its own setup (and,
    at the end of the run, its teardown) entirely outside that window,
    unprotected. Session scope is the only scope guaranteed to set up before
    narrower-scoped fixtures (module/class/function) in the same session --
    not a guarantee that it runs before *every* other fixture: pytest does
    not order independent session-scoped fixtures against each other.
    """

    @pytest.fixture(scope="module")
    def _popen_patch_state_at_module_setup(self):
        # Module scope is broader than function scope, so this fixture's
        # setup runs -- for the first test in this module that requests it
        # -- before the (currently function-scoped) guard fixture has had a
        # chance to patch anything for that test. Only a session-scoped
        # guard would already be active by this point.
        return subprocess.Popen.__init__ is git_mutation_guard._guarded_popen_init

    def test_guard_is_already_active_before_a_module_scoped_fixture_runs(
        self, _popen_patch_state_at_module_setup
    ):
        assert _popen_patch_state_at_module_setup, (
            "Popen.__init__ was not yet patched when a module-scoped "
            "fixture's setup ran -- the guard only protects the inside of "
            "a single test function's own fixtures, not the whole session"
        )


class TestGuardFailsClosedOnItsOwnProbeFailure:
    """D5 gap 2: if the guard can't identify its own worktree, it must block."""

    def test_probe_failure_on_the_callers_own_worktree_fails_closed(
        self, monkeypatch, shared_repo_with_linked_worktree
    ):
        """Simulates the guard's own identity probe failing.

        `_probe_identity` is forced to return ``None`` specifically for
        `_CALLER_ROOT_HINT` (the caller's-own-worktree probe inside
        `_compute_known_identities`) -- e.g. a transient git/subprocess
        error -- while git itself keeps working fine everywhere else
        (including on the real, dangerous target `alias`/`linked`, which
        would otherwise be correctly recognized as a live worktree).
        Currently this silently degrades to an *empty* known-identities
        list (fail open): the guard can no longer tell a live worktree
        from a disposable one, so every mutating call -- even ones aimed
        squarely at a live worktree -- sails through unexamined. It must
        instead refuse every mutating call until the probe succeeds again.
        """
        shared, linked, alias = shared_repo_with_linked_worktree
        monkeypatch.setattr(git_mutation_guard, "_CALLER_ROOT_HINT", shared)
        monkeypatch.setattr(git_mutation_guard, "_known_identities_cache", None)

        real_probe = git_mutation_guard._probe_identity

        def _flaky_probe(path, label):
            if path == shared:
                return None  # the own-worktree probe fails
            return real_probe(path, label)

        monkeypatch.setattr(git_mutation_guard, "_probe_identity", _flaky_probe)

        (linked / "f.txt").write_text("UNCOMMITTED EDIT\n")

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(alias), "reset", "--hard"],
                capture_output=True,
                text=True,
            )

        assert (linked / "f.txt").read_text() == "UNCOMMITTED EDIT\n", (
            "the own-probe failure must fail closed (block the mutation), "
            "not fail open (let it run because nothing was 'known')"
        )


class TestTargetProbeExecutionErrorFailsClosed:
    """D5 gap 1 (sol review): a *target* probe failure must fail closed too.

    Previously only the guard's own-worktree probe (`_compute_known_identities`)
    failed closed on an execution error; the *target*-of-the-mutation probe in
    `_assert_target_not_protected` still conflated "probe execution failed"
    with "cleanly determined not to be a git repo" and let the mutation
    through in both cases. This simulates the actual subprocess call inside
    `_probe_identity` raising (not `_probe_identity` itself just handed a
    `None`), so the distinction the fix must draw -- error vs legitimate
    negative -- is exercised for real.
    """

    def test_target_probe_execution_error_is_refused_not_treated_as_no_repo(
        self, monkeypatch, shared_repo_with_linked_worktree
    ):
        shared, linked, alias = shared_repo_with_linked_worktree
        monkeypatch.setattr(git_mutation_guard, "_CALLER_ROOT_HINT", shared)
        monkeypatch.setattr(git_mutation_guard, "_known_identities_cache", None)

        real_run = subprocess.run
        seen = {"count": 0}

        def _flaky_run(args, *a, **kw):
            if isinstance(args, list) and "rev-parse" in args and str(linked) in args:
                seen["count"] += 1
                if seen["count"] > 1:
                    # The *first* rev-parse-on-`linked` call is the registry
                    # build (`_compute_known_identities`'s `git worktree
                    # list` loop) -- let it succeed for real, so `linked`'s
                    # identity is correctly known. Every call after that is
                    # the *target*-of-the-mutation probe in
                    # `_assert_target_not_protected`; that's the one this
                    # test simulates failing with a genuine execution error.
                    raise OSError("simulated execution error probing the target")
            return real_run(args, *a, **kw)

        monkeypatch.setattr(subprocess, "run", _flaky_run)

        (linked / "f.txt").write_text("UNCOMMITTED EDIT\n")

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(linked), "reset", "--hard"],
                capture_output=True,
                text=True,
            )

        assert (linked / "f.txt").read_text() == "UNCOMMITTED EDIT\n", (
            "a target probe execution error must fail closed (block the "
            "mutation), not be treated as 'target is not a git repo' and "
            "let it through"
        )


class TestLinkedWorktreeProbeFailureDuringListingFailsClosed:
    """D5 gap 2 (sol review): a linked worktree's own identity must not be
    silently dropped from the known-identities set when its probe fails
    while walking `git worktree list`'s output.

    Reuses the exact attack from
    ``test_destructive_git_dir_override_from_an_unrelated_cwd_is_refused``
    above -- an explicit ``--git-dir=<linked's git-dir>`` from an unrelated,
    non-git cwd, which only the *linked worktree's own* `git_dir` value (not
    the caller's/main worktree's `common_dir`) catches -- to prove the
    omission, not just any mutation aimed at the shared repository, is what
    breaks: a plain `-C <linked>` mutation is already caught via the main
    worktree's `common_dir`, so it wouldn't distinguish "omitted" from
    "present" here.
    """

    def test_git_dir_override_attack_succeeds_when_linked_worktree_is_omitted(
        self, monkeypatch, shared_repo_with_linked_worktree, tmp_path
    ):
        shared, linked, alias = shared_repo_with_linked_worktree
        monkeypatch.setattr(git_mutation_guard, "_CALLER_ROOT_HINT", shared)
        monkeypatch.setattr(git_mutation_guard, "_known_identities_cache", None)

        real_probe = git_mutation_guard._probe_identity

        def _flaky_probe(path, label):
            if label.startswith("a worktree registered to the same repository"):
                return None  # simulates that worktree's probe failing during listing
            return real_probe(path, label)

        monkeypatch.setattr(git_mutation_guard, "_probe_identity", _flaky_probe)

        (linked / "g.txt").write_text("second\n")
        _git(linked, "add", "g.txt")
        _git(linked, "commit", "-qm", "second")

        linked_git_dir = subprocess.run(
            ["git", "-C", str(linked), "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        before = _git(linked, "rev-parse", "feature").stdout.strip()

        scratch = tmp_path / "unrelated-scratch"
        scratch.mkdir()

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", f"--git-dir={linked_git_dir}", "reset", "--hard", "HEAD~1"],
                cwd=scratch,
                capture_output=True,
                text=True,
            )

        after = _git(linked, "rev-parse", "feature").stdout.strip()
        assert after == before, (
            "the linked worktree's identity was silently omitted from the "
            "known set (its probe 'failed' during `git worktree list` "
            "processing), which let a --git-dir override attack aimed at "
            "it through -- this must fail closed instead"
        )


class TestNonCleanNonZeroExitFailsClosed:
    """D5 gap (sol review, round 4): only git's specific, well-documented
    "not a git repository" fatal is a clean negative. `_probe_identity`
    currently treats *any* non-zero exit the same way -- rc=128 from an
    unrelated fatal error (permission denied, a corrupted repo, "cannot
    change to <dir>") is silently folded into the same "not a repo, nothing
    to alias" outcome as git's actual not-a-repository signal. This forces
    the *target* probe inside `_assert_target_not_protected` to return that
    unrelated rc=128 for a real, live worktree and proves the mutation
    sails through anyway.
    """

    def test_unrelated_fatal_error_at_rc_128_is_refused_not_treated_as_no_repo(
        self, monkeypatch, shared_repo_with_linked_worktree
    ):
        shared, linked, alias = shared_repo_with_linked_worktree
        monkeypatch.setattr(git_mutation_guard, "_CALLER_ROOT_HINT", shared)
        monkeypatch.setattr(git_mutation_guard, "_known_identities_cache", None)

        real_run = subprocess.run
        seen = {"count": 0}

        def _flaky_run(args, *a, **kw):
            if isinstance(args, list) and "rev-parse" in args and str(linked) in args:
                seen["count"] += 1
                if seen["count"] > 1:
                    # The first rev-parse-on-`linked` call is the registry
                    # build (`git worktree list` loop) -- let it succeed for
                    # real. Every call after that is the *target* probe in
                    # `_assert_target_not_protected`; simulate an unrelated
                    # rc=128 fatal there, not git's documented "not a git
                    # repository" fatal.
                    return subprocess.CompletedProcess(
                        args,
                        128,
                        stdout="",
                        stderr="fatal: cannot change to '...': Permission denied\n",
                    )
            return real_run(args, *a, **kw)

        monkeypatch.setattr(subprocess, "run", _flaky_run)

        (linked / "f.txt").write_text("UNCOMMITTED EDIT\n")

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", "-C", str(linked), "reset", "--hard"],
                capture_output=True,
                text=True,
            )

        assert (linked / "f.txt").read_text() == "UNCOMMITTED EDIT\n", (
            "an unrelated rc=128 fatal must fail closed (block the "
            "mutation), not be conflated with git's clean 'not a git "
            "repository' negative and let it through"
        )


class TestWorktreeListingFailureFailsClosed:
    """D5 gap (sol review, round 4): `git worktree list` itself failing must
    not leave `_compute_known_identities` proceeding with a partial
    (caller-only) registry -- that previously only covered a single
    *entry's* probe failing within an otherwise-successful listing, not the
    listing command itself failing outright.

    Reuses the `--git-dir` override attack (only the linked worktree's own
    `git_dir` catches it, not the main worktree's `common_dir`, which a
    partial caller-only registry still contains) -- a plain `-C <linked>`
    mutation would be caught regardless via `common_dir` aliasing, so it
    wouldn't distinguish "omitted from the registry" from "present".
    """

    def test_git_dir_override_attack_succeeds_when_worktree_list_fails_outright(
        self, monkeypatch, shared_repo_with_linked_worktree, tmp_path
    ):
        shared, linked, alias = shared_repo_with_linked_worktree
        monkeypatch.setattr(git_mutation_guard, "_CALLER_ROOT_HINT", shared)
        monkeypatch.setattr(git_mutation_guard, "_known_identities_cache", None)

        real_run = subprocess.run

        def _flaky_run(args, *a, **kw):
            if isinstance(args, list) and "worktree" in args and "list" in args:
                return subprocess.CompletedProcess(
                    args, 1, stdout="", stderr="fatal: simulated worktree list failure\n"
                )
            return real_run(args, *a, **kw)

        monkeypatch.setattr(subprocess, "run", _flaky_run)

        (linked / "g.txt").write_text("second\n")
        _git(linked, "add", "g.txt")
        _git(linked, "commit", "-qm", "second")

        linked_git_dir = real_run(
            ["git", "-C", str(linked), "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        before = _git(linked, "rev-parse", "feature").stdout.strip()

        scratch = tmp_path / "unrelated-scratch"
        scratch.mkdir()

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", f"--git-dir={linked_git_dir}", "reset", "--hard", "HEAD~1"],
                cwd=scratch,
                capture_output=True,
                text=True,
            )

        after = _git(linked, "rev-parse", "feature").stdout.strip()
        assert after == before, (
            "`git worktree list` failing outright must fail closed (block "
            "every mutating call), not silently degrade to a partial "
            "(caller-only) registry that omits the linked worktree and lets "
            "a --git-dir override attack aimed at it through"
        )


class TestMalformedIdentityFieldFailsClosed:
    """D5 gap (sol review, round 4): a probe that exits zero with the
    expected 3-line shape, but a field that fails to resolve to a real path
    (e.g. `Path.resolve()` raising), must not silently produce an identity
    with a `None` field -- that field then can never match anything in
    `_identity_overlaps`, which is functionally identical to that worktree
    never having been in the known-identities set at all. Reuses the
    `--git-dir` override attack (only the linked worktree's own `git_dir`
    catches it, not the main worktree's `common_dir`), so this proves the
    omission, not just any mutation aimed at the shared repository.
    """

    def test_git_dir_override_attack_succeeds_when_linked_worktree_has_an_unresolved_field(
        self, monkeypatch, shared_repo_with_linked_worktree, tmp_path
    ):
        shared, linked, alias = shared_repo_with_linked_worktree
        monkeypatch.setattr(git_mutation_guard, "_CALLER_ROOT_HINT", shared)
        monkeypatch.setattr(git_mutation_guard, "_known_identities_cache", None)

        real_resolve = git_mutation_guard._resolve_relative_to

        def _flaky_resolve(raw, base):
            if base == linked:
                # Simulate every field of *this* probe failing to resolve
                # (e.g. an OSError inside Path.resolve()) even though the
                # probe itself ran cleanly (zero exit, three lines).
                return None
            return real_resolve(raw, base)

        monkeypatch.setattr(git_mutation_guard, "_resolve_relative_to", _flaky_resolve)

        (linked / "g.txt").write_text("second\n")
        _git(linked, "add", "g.txt")
        _git(linked, "commit", "-qm", "second")

        linked_git_dir = subprocess.run(
            ["git", "-C", str(linked), "rev-parse", "--absolute-git-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        before = _git(linked, "rev-parse", "feature").stdout.strip()

        scratch = tmp_path / "unrelated-scratch"
        scratch.mkdir()

        with pytest.raises(git_mutation_guard.GitContainmentViolation):
            subprocess.run(
                ["git", f"--git-dir={linked_git_dir}", "reset", "--hard", "HEAD~1"],
                cwd=scratch,
                capture_output=True,
                text=True,
            )

        after = _git(linked, "rev-parse", "feature").stdout.strip()
        assert after == before, (
            "the linked worktree's identity had an unresolved (None) field, "
            "which let a --git-dir override attack aimed at it through -- "
            "this must fail closed instead of silently producing a "
            "non-matching identity"
        )


class TestReusesSelfRepoGuardDetection:
    """Pins that detection lives in one place — importing, not re-deriving it.

    Two independent definitions of "is this a mutating git command" (or "how
    is a git alias resolved") drifting apart is the exact defect shape this
    containment work has been fixing elsewhere; this test fails loudly the
    moment ``git_mutation_guard`` stops importing these objects and starts
    redefining them instead.
    """

    def test_mutation_classifier_is_the_same_object_not_a_copy(self):
        assert git_mutation_guard._mutates_worktree is self_repo_guard._mutates_worktree
        assert (
            git_mutation_guard._git_target_and_subcommand
            is self_repo_guard._git_target_and_subcommand
        )
        assert (
            git_mutation_guard._inspect_git_worktree
            is self_repo_guard._inspect_git_worktree
        )
        assert (
            git_mutation_guard.detect_self_repo_git_mutation
            is self_repo_guard.detect_self_repo_git_mutation
        )
        assert git_mutation_guard._read_git_alias is self_repo_guard._read_git_alias
        assert git_mutation_guard._MAX_RECURSION == self_repo_guard._MAX_RECURSION
