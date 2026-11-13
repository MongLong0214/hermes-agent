"""Local POSIX content/file search runs rg as a direct argv subprocess.

The shell path pays two bash spawns per search (``test -e`` probe + ``set -o
pipefail; rg ... | head``). On a ``LocalEnvironment`` the same rg argv can run
natively with a bounded stdout read; the parser and every argument builder are
shared, so the two transports must agree on results.
"""

import json

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.platforms("posix")  # native rg lane is POSIX-only


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.py").write_text("needle one\nplain\nneedle two\n")
    (tmp_path / "b.txt").write_text("needle three\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("needle four\n")
    return tmp_path


@pytest.fixture(scope="module")
def _local_env(tmp_path_factory):
    """One real LocalEnvironment per module (constructing one costs ~0.8 s)."""
    return LocalEnvironment(cwd=str(tmp_path_factory.mktemp("native-rg")))


@pytest.fixture
def ops_factory(_local_env):
    """``make(tree, spy)`` → ShellFileOperations over the shared env, every execute recorded in ``spy``."""
    real = type(_local_env).execute.__get__(_local_env, type(_local_env))

    def make(tree, spy):
        _local_env.cwd = str(tree)

        def recording(command, *a, **kw):
            spy.append(command)
            return real(command, *a, **kw)

        _local_env.execute = recording
        return ShellFileOperations(_local_env, cwd=str(tree))

    yield make
    _local_env.__dict__.pop("execute", None)


def _normalized(result):
    d = result.to_dict()
    for key in ("matches", "files"):
        if key in d:
            d[key] = sorted(json.dumps(item, sort_keys=True) for item in d[key])
    return d


def test_native_search_never_touches_the_shell_and_matches_shell_results(tree, ops_factory, monkeypatch):
    if not ops_factory(tree, [])._has_command("rg"):
        pytest.skip("native search parity requires ripgrep in the search environment")
    cases = [
        dict(pattern="needle", path=str(tree)),
        # (no offset/limit slicing here: rg's parallel walk orders files
        # nondeterministically, so a page differs run-to-run on either transport)
        dict(pattern="needle", path=str(tree), output_mode="count"),
        dict(pattern="needle", path=str(tree), output_mode="files_only", file_glob="*.py"),
        dict(pattern="NEEDLE_NOPE", path=str(tree)),  # zero-match probes
        dict(pattern="*.py", path=str(tree), target="files"),
        dict(pattern="needle", path=str(tree / "missing")),
    ]
    for case in cases:
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
        shell = _normalized(ops_factory(tree, []).search(**case))
        monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
        calls = []
        native = _normalized(ops_factory(tree, calls).search(**case))
        assert native == shell, case
        # rg resolution (``command -v rg``) still goes through the shell once; the
        # existence probe and the rg pipeline itself must not.
        assert not [c for c in calls if "pipefail" in c or c.startswith("test -e")], case


def test_native_runner_honours_deadline_and_interrupt_while_rg_is_silent(tree, ops_factory, monkeypatch):
    """A search producing no output must still stop at the deadline / on /stop
    (the shell path gets this from ``_wait_for_process``)."""
    import threading
    import time

    from tools import interrupt

    ops = ops_factory(tree, [])
    started = time.monotonic()
    result = ops._run_rg_native(["sh", "-c", "'sleep 30'"], 5, timeout=1)
    assert result.exit_code == 124 and time.monotonic() - started < 5

    tid = threading.get_ident()
    threading.Timer(0.3, lambda: interrupt.set_interrupt(True, tid)).start()
    try:
        started = time.monotonic()
        result = ops._run_rg_native(["sh", "-c", "'sleep 30'"], 5, timeout=30)
    finally:
        interrupt.set_interrupt(False, tid)
    assert result.exit_code == 130 and time.monotonic() - started < 5


def test_kill_switch_routes_search_back_to_the_shell(tree, ops_factory, monkeypatch):
    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "0")
    calls = []
    result = ops_factory(tree, calls).search(pattern="needle", path=str(tree))
    assert result.total_count == 4
    assert any(c.startswith("test -e") for c in calls)
    assert any(c.startswith("set -o pipefail; ") for c in calls)


def test_limit_hit_keeps_drained_matches_when_group_kill_is_refused(tree, ops_factory, monkeypatch):
    """Reaching ``limit`` takes the early-stop branch that TERMs rg's group. macOS answers
    ``killpg`` with EPERM (not ESRCH) for a zombie-only group; that must not surface as a tool
    error nor discard the matches already drained (#116855)."""
    import os

    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
    ops = ops_factory(tree, [])
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (_ for _ in ()).throw(PermissionError(1, "Operation not permitted")))
    result = ops.search(pattern="needle", path=str(tree), limit=2)
    assert not result.error and len(result.matches) == 2, result.to_dict()


def test_local_search_runs_the_pinned_ripgrep_not_the_first_on_path(_local_env, tmp_path, monkeypatch):
    """The terminal PATH puts the user's tools first; Hermes's own search still runs PM's rg."""
    import pm
    from pm.install import InstalledPackage

    pinned = tmp_path / "store" / "ripgrep" / "rg"
    pinned.parent.mkdir(parents=True)
    pinned.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(pm, "installed_package",
                        lambda name: InstalledPackage(pinned.parent, "15", pinned) if name == "ripgrep" else None)
    ops = ShellFileOperations(_local_env, cwd=str(tmp_path))
    assert ops._resolve_command("rg") == str(pinned).replace("\\", "/")


def _wait_dead(pid, within):
    import os
    import time

    deadline = time.monotonic() + within
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _kill_quietly(pidfile):
    """Reap a leftover sleeper through its group: once reparented to init it is outside the
    test's process tree, where the live-system guard refuses a per-PID kill."""
    import contextlib
    import os

    with contextlib.suppress(Exception):
        pgid = os.getpgid(int(pidfile.read_text().strip()))
        if pgid != os.getpgrp():
            os.killpg(pgid, 9)


@pytest.mark.parametrize("merge_stderr", [False, True], ids=["stdout", "stdout+stderr"])
def test_native_runner_returns_within_the_bound_when_a_grandchild_holds_the_pipe(
        tree, ops_factory, monkeypatch, tmp_path, merge_stderr):
    """rg exits but a descendant the group kill cannot reach keeps its pipes open: the search
    returns with the output already drained instead of waiting for the descendant to exit."""
    import time

    from tools.environments import local as local_env

    monkeypatch.setattr(local_env, "_kill_process_group_posix", lambda proc: None)  # unreachable survivor
    ops = ops_factory(tree, [])
    pidfile = tmp_path / "bg.pid"
    started = time.monotonic()
    try:
        result = ops._run_rg_native(["sh", "-c", f"'echo needle; (sleep 12 & echo $! > {pidfile})'"],
                                    10, timeout=1, merge_stderr=merge_stderr)
        elapsed = time.monotonic() - started
    finally:
        _kill_quietly(pidfile)
    assert "needle" in result.stdout, result.to_dict()
    assert elapsed < 8.0, f"a surviving descendant must not hang the search, took {elapsed:.2f}s"


def test_native_runner_kills_surviving_group_when_leader_already_exited(tree, ops_factory, tmp_path):
    """The leader (``sh``) backgrounds ``sleep`` and exits well before the deadline (after the
    spawn has recorded its group). The pgid recorded at spawn still reaches the descendant left
    in its group; skipping the kill because the leader is gone leaves it running."""
    import time

    ops = ops_factory(tree, [])
    pidfile = tmp_path / "bg.pid"
    started = time.monotonic()
    try:
        result = ops._run_rg_native(
            ["sh", "-c", f"'echo needle; (sleep 12 & echo $! > {pidfile}); sleep 0.3'"], 10, timeout=1)
        elapsed = time.monotonic() - started
        assert "needle" in result.stdout, result.to_dict()
        assert elapsed < 8.0, f"a surviving descendant must not hang the search, took {elapsed:.2f}s"
        assert _wait_dead(int(pidfile.read_text().strip()), within=5.0), "the descendant must be killed"
    finally:
        _kill_quietly(pidfile)


def test_native_runner_tolerates_pgid_lookup_racing_rg_exit(tree, ops_factory, monkeypatch):
    """rg can exit before its group is recorded at spawn and be gone by the kill; with no
    leader and no recorded pgid the lookup's ``ProcessLookupError`` must not escape and
    discard the output already drained."""
    import os
    import subprocess

    monkeypatch.setattr(subprocess.Popen, "poll", lambda self: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError(3, "No such process")))
    ops = ops_factory(tree, [])
    result = ops._run_rg_native(["sh", "-c", "'echo needle-one; echo needle-two'"], 10, timeout=5)
    assert result.exit_code == 0
    assert "needle-one" in result.stdout and "needle-two" in result.stdout, result.stdout
