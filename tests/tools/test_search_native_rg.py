"""Local POSIX content/file search runs rg as a direct argv subprocess.

The shell path pays two bash spawns per search (``test -e`` probe + ``set -o
pipefail; rg ... | head``). On a ``LocalEnvironment`` the same rg argv can run
natively with a bounded stdout read; the parser and every argument builder are
shared, so the two transports must agree on results.
"""

import contextlib
import json
import sys

import pytest

from tools.environments.local import LocalEnvironment
from tools.file_operations import ShellFileOperations

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="native rg lane is POSIX-only")


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
    assert any("pipefail" in c and "rg" in c for c in calls)


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


def test_native_runner_tolerates_pgid_lookup_racing_rg_exit(tree, ops_factory, monkeypatch):
    """rg can exit (and be reaped by an unrelated ``Popen`` cleanup elsewhere in the same
    process) between the cleanup's ``proc.poll()`` check and its ``os.getpgid`` lookup; unlike
    the ``killpg`` EPERM case above, this earlier lookup has no cached pgid to fall back to, so
    the ``ProcessLookupError`` must not escape and discard the already-drained output (18/day
    ``search_files`` '[Errno 3] No such process' failures in production)."""
    import os
    import subprocess

    monkeypatch.setattr(subprocess.Popen, "poll", lambda self: None)
    monkeypatch.setattr(os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError(3, "No such process")))
    ops = ops_factory(tree, [])
    result = ops._run_rg_native(["sh", "-c", "'echo needle-one; echo needle-two'"], 10, timeout=5)
    assert result.exit_code == 0
    assert "needle-one" in result.stdout and "needle-two" in result.stdout, result.stdout


def test_native_runner_kills_surviving_group_via_cached_pgid_after_leader_reaped(tree, ops_factory, monkeypatch, tmp_path):
    """PR64-03: a backgrounded descendant (plain ``&``, no ``setsid``) stays in the leader's
    process group and keeps rg's stdout pipe open even after the leader itself exits and is
    reaped — so ``os.getpgid(leader_pid)`` (the lookup inside ``_kill_process_group_posix``)
    then raises ESRCH with no leader left to ask. The pgid cached right after ``Popen`` (mirrors
    ``tools/environments/local.py``'s ``_run_bash``) must let the kill still reach the survivor
    via its numeric pgid, instead of silently doing nothing and leaving it running."""
    import os
    import subprocess as subprocess_mod
    import time

    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
    ops = ops_factory(tree, [])

    pidfile = tmp_path / "bg.pid"
    real_getpgid = os.getpgid
    calls = {"n": 0}

    def _getpgid(pid):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_getpgid(pid)  # the spawn-time caching call: leader is still alive
        raise ProcessLookupError(3, "No such process")  # every later lookup: leader already reaped

    monkeypatch.setattr(os, "getpgid", _getpgid)
    monkeypatch.setattr(subprocess_mod.Popen, "poll", lambda self: None)  # pre-kill check still sees "alive"

    started = time.monotonic()
    result = ops._run_rg_native(
        ["sh", "-c", f"'echo needle; (sleep 20 & echo $! > {pidfile})'"], 10, timeout=1,
    )
    elapsed = time.monotonic() - started

    assert "needle" in result.stdout, result.to_dict()
    assert calls["n"] >= 2, "the kill-path lookup must actually run (and see the simulated ESRCH)"
    # Generous bound: outer 1s deadline + _kill_process_group_posix's own TERM/KILL grace
    # (up to 1s + 2s) is already ~4s in the legitimate worst case, before any scheduling
    # slack under a loaded parallel test run — still far short of the 20s a left-running
    # descendant (the bug) would take.
    assert elapsed < 10.0, f"the cached pgid must let the kill reach the survivor, took {elapsed:.2f}s"

    bg_pid = int(pidfile.read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(bg_pid, 0)  # confirms the survivor was actually killed, not just timed out past


def test_native_runner_bounds_final_wait_when_group_cannot_be_killed(tree, ops_factory, monkeypatch, tmp_path):
    """PR64-03: when the surviving descendant truly cannot be reached (both the cached and
    the live pgid lookups fail — the already-gone-entirely case the ``suppress`` is for), the
    final ``drainer.join()`` must still be bounded so the search returns with the output
    already captured instead of hanging until the descendant exits on its own."""
    import os
    import time

    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
    ops = ops_factory(tree, [])

    pidfile = tmp_path / "bg.pid"
    monkeypatch.setattr(os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError(3, "No such process")))

    import subprocess as subprocess_mod
    monkeypatch.setattr(subprocess_mod.Popen, "poll", lambda self: None)

    started = time.monotonic()
    try:
        result = ops._run_rg_native(
            ["sh", "-c", f"'echo needle; (sleep 20 & echo $! > {pidfile})'"], 10, timeout=1,
        )
        elapsed = time.monotonic() - started
        assert "needle" in result.stdout, result.to_dict()
        assert elapsed < 10.0, f"a surviving descendant must not hang the search, took {elapsed:.2f}s"
    finally:
        # Nothing in this test's fault injection can kill the survivor; clean it up ourselves
        # so it doesn't keep running for the rest of the suite.
        with contextlib.suppress(Exception):
            os.kill(int(pidfile.read_text().strip()), 9)


def test_native_runner_kills_surviving_group_when_leader_already_exited(tree, ops_factory, monkeypatch, tmp_path):
    """PR64-R02: no fault injection here — the leader (``sh``) backgrounds ``sleep`` and
    writes its PID almost instantly, so by the time the drain loop gives up at the 1s
    deadline, ``proc.poll()`` genuinely reports the leader as already exited. The
    group-kill helper must still run in that case: the cached pgid can reach the live
    descendant even though the leader itself is gone, and skipping the helper just
    because ``proc.poll()`` is not None leaves that descendant (and its pipe) running
    forever instead of being reaped."""
    import os
    import time

    monkeypatch.setenv("HERMES_NATIVE_FILE_READ", "1")
    ops = ops_factory(tree, [])

    pidfile = tmp_path / "bg.pid"
    started = time.monotonic()
    try:
        result = ops._run_rg_native(
            ["sh", "-c", f"'echo needle; (sleep 20 & echo $! > {pidfile})'"], 10, timeout=1,
        )
        elapsed = time.monotonic() - started
        assert "needle" in result.stdout, result.to_dict()
        assert elapsed < 10.0, f"a surviving descendant must not hang the search, took {elapsed:.2f}s"

        bg_pid = int(pidfile.read_text().strip())

        def _dead():
            try:
                os.kill(bg_pid, 0)
            except ProcessLookupError:
                return True
            return False

        # The group-kill TERM/KILL grace (up to 1s + 2s inside _kill_process_group_posix)
        # runs after the outer deadline fires; give it room to land before judging.
        settle_deadline = time.monotonic() + 5.0
        while time.monotonic() < settle_deadline and not _dead():
            time.sleep(0.05)

        assert _dead(), "the live descendant (sleep) must actually be terminated, not leaked"
    finally:
        with contextlib.suppress(Exception):
            os.kill(int(pidfile.read_text().strip()), 9)
