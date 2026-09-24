"""``hermes target bind --json`` run the way the external controller runs it.

The controller spawns the CLI with a three-variable env and no PATH, writes one JSON request to
stdin, and compares stdout byte for byte: the 8-key receipt on exit 0, or exactly one closed error
object on exit 1. Every test here spawns a real process from this checkout against a store seeded
through the store API.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_STALE_KEY, FTS_STORAGE_VERSION
from tests.hermes_state.fork_store_fixture import build_store

REPO_ROOT = Path(__file__).resolve().parents[2]

_BIND = ("target", "bind", "--json")
_LINEAGE_DOMAIN = b"hermes.target-bind:lineage-root\0"
_RECEIPT_KEYS = [
    "domain",
    "version",
    "actor_id",
    "binding_generation",
    "executor_runtime_identity",
    "requested_session_id",
    "lineage_root_digest",
    "receipt_digest",
]
_INVALID = b'{"error":"target_bind_preflight_invalid"}'
_CONFLICT = b'{"error":"target_bind_preflight_conflict"}'
_UNAVAILABLE = b'{"error":"target_bind_preflight_unavailable"}'
# What one CLI start creates in a scaffolded home that has never run the CLI: the profile logs and the
# scratch-prune marker. Bind does not fork its startup path, so these are the whole allowance.
_STARTUP_WRITES = {"logs/agent.log", "logs/errors.log", "cache", "cache/scratch", "cache/scratch/.last_prune"}
# Journal sidecars exist only while a connection is open, so a clean close may or may not leave them.
_STORE_SIDECARS = {"state.db-wal", "state.db-shm", "state.db-journal"}
# The controller kills the child after this long (its ``runHermesTargetBind`` default timeout).
_CALLER_DEADLINE_S = 5.0


def _lineage_digest(root_id: str) -> str:
    return "sha256:" + hashlib.sha256(_LINEAGE_DOMAIN + root_id.encode("utf-8")).hexdigest()


def _canonical_digest(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scaffold(home: Path, monkeypatch) -> Path:
    """A home as a real install leaves it: the skeleton exists, the CLI has never run in it."""
    from hermes_cli.config import ensure_hermes_home

    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    ensure_hermes_home()
    return home


def _seed(home: Path, *chains: tuple[str, ...]) -> None:
    """Create each chain root-first, every session parented on the one before it."""
    db = SessionDB(home / "state.db")
    try:
        for chain in chains:
            parent = None
            for session_id in chain:
                db.create_session(session_id, source="cli", parent_session_id=parent)
                parent = session_id
    finally:
        db.close()


@pytest.fixture
def home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    # "tip" sits three parents below "root"; "other" is an unrelated lineage.
    _seed(home, ("root", "mid-1", "mid-2", "tip"), ("other",))
    return home


def _request(**overrides) -> bytes:
    request = {
        "domain": "hermes.target-bind",
        "version": 1,
        "session_id": "tip",
        "expected_lineage_root_digest": _lineage_digest("root"),
        "actor_id": "actor:a",
        "binding_generation": 1,
        "executor_runtime_identity": "runtime:a",
    }
    request.update(overrides)
    return json.dumps(request).encode("utf-8")


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = _scaffold(tmp_path / "home", monkeypatch)
    _seed(home, ("root", "mid-1", "mid-2", "tip"), ("other",))
    return home


def _child_env(home: Path) -> dict[str, str]:
    # The controller's three variables, plus PYTEST_CURRENT_TEST: it arms the guards that stop a child of
    # this suite from running startup recovery (a real reinstall) against the checkout it imports.
    return {
        "HOME": str(home),
        "HERMES_HOME": str(home),
        "HERMES_PROFILE": "default",
        "PYTEST_CURRENT_TEST": os.environ["PYTEST_CURRENT_TEST"],
    }


def _run(
    home: Path,
    stdin: bytes,
    argv: tuple[str, ...] = _BIND,
    *,
    python_args=("-m", "hermes_cli.main"),
    extra_env: dict[str, str] | None = None,
):
    env = {**_child_env(home), **(extra_env or {})}
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, *python_args, *argv],
        input=stdin,
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=60,
    )
    return proc, time.monotonic() - started


# ``-m hermes_cli.main``, reporting wall and CPU seconds from the moment bind reserves stdout for its reply
# until exit.
_TIMED_MAIN = (
    "import atexit, os, runpy, sys, time\n"
    "import hermes_cli.target_bind as target_bind\n"
    "reserve, reserved = target_bind.reserve_stdout_for_reply, []\n"
    "def timed():\n"
    "    reserved.append((time.monotonic(), time.process_time()))\n"
    "    reserve()\n"
    "target_bind.reserve_stdout_for_reply = timed\n"
    "atexit.register(lambda: os.write(2, '\\nSINCE_RESERVED={},{}'.format("
    "time.monotonic() - reserved[0][0], time.process_time() - reserved[0][1]).encode()))\n"
    "sys.argv = ['hermes', *sys.argv[1:]]\n"
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)\n"
)


def _run_timed(home: Path, stdin: bytes):
    proc, _ = _run(home, stdin, python_args=("-c", _TIMED_MAIN))
    return proc, [float(s) for s in proc.stderr.decode("utf-8").rsplit("SINCE_RESERVED=", 1)[1].split(",")]


def _assert_within_budget(since_reserved: list[float]) -> None:
    # A loaded runner stretches wall time, not work: at 0.5 s of CPU, bind's run past the reservation has
    # taken 12 s of wall under a double suite. So the work is held to the caller's window, and the wall to
    # the store's own patience, which a bind that waited on the store instead of its budget would outlast.
    wall, cpu = since_reserved
    assert cpu < _CALLER_DEADLINE_S and wall < SessionDB._WRITE_PATIENCE_S, since_reserved


def _state_meta(home: Path) -> dict[str, str]:
    conn = sqlite3.connect(home / "state.db")
    try:
        return dict(conn.execute("SELECT key, value FROM state_meta").fetchall())
    finally:
        conn.close()


def _receipt_rows(home: Path) -> list[tuple[str, str]]:
    conn = sqlite3.connect(home / "state.db")
    try:
        return conn.execute(
            "SELECT key, value FROM state_meta WHERE key GLOB 'target_bind_receipt:*' ORDER BY key"
        ).fetchall()
    finally:
        conn.close()


def _snapshot(root: Path) -> dict[str, tuple]:
    snap: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            snap[rel] = ("link", str(path.readlink()))
        elif path.is_dir():
            snap[rel] = ("dir", path.stat().st_mode)
        else:
            snap[rel] = ("file", path.stat().st_mode, path.read_bytes())
    return snap


def _assert_receipt(stdout: bytes, request: bytes) -> dict:
    receipt = json.loads(stdout.decode("utf-8"))
    sent = json.loads(request)
    assert list(receipt) == _RECEIPT_KEYS
    public = {key: receipt[key] for key in _RECEIPT_KEYS if key != "receipt_digest"}
    assert receipt["receipt_digest"] == _canonical_digest(public)
    assert receipt["lineage_root_digest"] == sent["expected_lineage_root_digest"]
    assert receipt["requested_session_id"] == sent["session_id"]
    assert receipt["actor_id"] == sent["actor_id"]
    assert receipt["binding_generation"] == sent["binding_generation"]
    assert receipt["executor_runtime_identity"] == sent["executor_runtime_identity"]
    return receipt


def _changed(before: dict[str, tuple], after: dict[str, tuple]) -> set[str]:
    return {rel for rel in before.keys() | after.keys() if before.get(rel) != after.get(rel)}


def _receipt_key(request: bytes) -> str:
    sent = json.loads(request)
    fields = ("domain", "version", "actor_id", "binding_generation", "executor_runtime_identity")
    identity = _canonical_digest({key: sent[key] for key in fields})
    return "target_bind_receipt:" + identity.removeprefix("sha256:")


def test_bind_prints_the_receipt_and_changes_only_the_store(fresh_home):
    before, meta_before = _snapshot(fresh_home), _state_meta(fresh_home)
    request = _request()
    proc, since_reserved = _run_timed(fresh_home, request)
    after, meta_after = _snapshot(fresh_home), _state_meta(fresh_home)

    assert proc.returncode == 0, proc.stderr
    _assert_within_budget(since_reserved)
    _assert_receipt(proc.stdout, request)
    # The store changes, each startup path that was absent appears, and nothing else moves.
    assert _changed(before, after) - _STORE_SIDECARS == {"state.db", *(_STARTUP_WRITES - before.keys())}
    # The first open of a seeded store stamps the FTS layout version; the receipt is the only other row.
    assert set(meta_after) - set(meta_before) - {"fts_storage_version"} == {_receipt_key(request)}
    assert {key: meta_after[key] for key in meta_before} == meta_before


def test_stdout_is_only_the_reply_when_startup_prints(home, tmp_path):
    # An installed entry-point plugin enabled in config is imported during startup, before bind runs.
    site = tmp_path / "site"
    dist = site / "noisy_ep-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: noisy-ep\nVersion: 1.0\n", encoding="utf-8")
    (dist / "entry_points.txt").write_text("[hermes_agent.plugins]\nnoisy-ep = noisy_ep_mod\n", encoding="utf-8")
    (site / "noisy_ep_mod.py").write_text(
        "import os\nprint('NOISY-EP-PRINT')\nos.write(1, b'NOISY-EP-FD1\\n')\n", encoding="utf-8"
    )
    (home / "config.yaml").write_text("plugins:\n  enabled: [noisy-ep]\n", encoding="utf-8")
    request = _request()

    proc, _ = _run(home, request, extra_env={"PYTHONPATH": str(site)})

    assert proc.returncode == 0, proc.stderr
    _assert_receipt(proc.stdout, request)
    # The plugin did run; its Python and fd-level writes both went to stderr.
    assert b"NOISY-EP-PRINT" in proc.stderr and b"NOISY-EP-FD1" in proc.stderr


def test_child_keeps_the_live_checkout_guards_armed(home):
    probe = (
        "import json, sys\n"
        "from hermes_cli import _early_recovery, main_install_repair\n"
        "root = _early_recovery._project_root()\n"
        "sys.stdout.write(json.dumps([str(root), _early_recovery._pytest_owns_live_checkout(root),"
        " main_install_repair._pytest_owns_live_checkout(root)]))\n"
    )
    proc, _ = _run(home, b"", (), python_args=("-c", probe))

    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == [str(REPO_ROOT), True, True]


@pytest.mark.parametrize("settled", [False, True], ids=["open-waits", "write-waits"])
def test_held_write_lock_is_unavailable_within_the_deadline(home, settled):
    if settled:
        # A store every Hermes open has already stamped: the open reads only, and the write waits.
        SessionDB(home / "state.db").close()
    holder = sqlite3.connect(home / "state.db", isolation_level=None)
    try:
        holder.execute("BEGIN IMMEDIATE")
        proc, since_reserved = _run_timed(home, _request())
    finally:
        holder.rollback()
        holder.close()

    assert (proc.returncode, proc.stdout) == (1, _UNAVAILABLE), proc.stderr
    _assert_within_budget(since_reserved)
    assert _receipt_rows(home) == []


def test_child_runs_this_checkout(home):
    # The venv's editable install may point at another checkout; cwd on sys.path must win.
    probe = (
        "import atexit, json, runpy, sys\n"
        "names = ('hermes_cli', 'hermes_cli.target_bind', 'hermes_state', 'hermes_state_target_bind')\n"
        "atexit.register(lambda: sys.stderr.write('\\nMODULES=' + json.dumps("
        "{n: sys.modules[n].__file__ for n in names if n in sys.modules})))\n"
        "sys.argv = ['hermes', *sys.argv[1:]]\n"
        "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)\n"
    )
    request = _request()
    proc, _ = _run(home, request, python_args=("-c", probe))

    assert proc.returncode == 0, proc.stderr
    _assert_receipt(proc.stdout, request)
    modules = json.loads(proc.stderr.decode("utf-8").rsplit("MODULES=", 1)[1])
    assert sorted(modules) == ["hermes_cli", "hermes_cli.target_bind", "hermes_state", "hermes_state_target_bind"]
    for name, path in modules.items():
        assert Path(path).resolve().is_relative_to(REPO_ROOT), (name, path)


def test_replay_is_byte_identical_and_adds_no_row(home):
    request = _request()
    first, _ = _run(home, request)
    rows = _receipt_rows(home)
    second, _ = _run(home, request)

    assert (first.returncode, second.returncode) == (0, 0), (first.stderr, second.stderr)
    assert second.stdout == first.stdout
    assert len(rows) == 1
    assert _receipt_rows(home) == rows


def test_same_identity_bound_to_another_session_is_a_conflict(home):
    first, _ = _run(home, _request())
    rows = _receipt_rows(home)
    proc, _ = _run(home, _request(session_id="other", expected_lineage_root_digest=_lineage_digest("other")))

    assert first.returncode == 0, first.stderr
    assert (proc.returncode, proc.stdout) == (1, _CONFLICT)
    assert _receipt_rows(home) == rows


def _with_key(**extra) -> bytes:
    return json.dumps({**json.loads(_request()), **extra}).encode("utf-8")


def _without_key(key: str) -> bytes:
    request = json.loads(_request())
    del request[key]
    return json.dumps(request).encode("utf-8")


_INVALID_REQUESTS = {
    "extra key": _with_key(lineage_root_id="root"),
    "duplicate key": _request()[:-1] + b',"actor_id":"actor:b"}',
    "missing key": _without_key("executor_runtime_identity"),
    "version 2": _request(version=2),
    "version true": _request(version=True),
    "generation 0": _request(binding_generation=0),
    "negative generation": _request(binding_generation=-1),
    "float generation": _request(binding_generation=1.0),
    # The controller holds the generation as a JS number; past 2**53 - 1 it cannot reproduce the digest.
    "unsafe generation": _request(binding_generation=2**53),
    "uppercase digest": _request(expected_lineage_root_digest=_lineage_digest("root").upper()),
    "unprefixed digest": _request(expected_lineage_root_digest=_lineage_digest("root")[len("sha256:"):]),
    "lineage mismatch": _request(expected_lineage_root_digest=_lineage_digest("mid-1")),
    "unknown session": _request(session_id="missing"),
    "control character": _request(actor_id="actor:\x1b"),
    "malformed json": b'{"domain":',
    "not an object": b"[]",
    "not utf-8": b"\xff\xfe",
    "empty stdin": b"",
}


@pytest.mark.parametrize("request_bytes", list(_INVALID_REQUESTS.values()), ids=list(_INVALID_REQUESTS))
def test_invalid_request_is_one_closed_error(home, request_bytes):
    proc, _ = _run(home, request_bytes)

    assert (proc.returncode, proc.stdout) == (1, _INVALID), proc.stderr
    assert _receipt_rows(home) == []


def test_missing_json_flag_is_invalid(home):
    proc, _ = _run(home, _request(), ("target", "bind"))

    assert (proc.returncode, proc.stdout) == (1, _INVALID), proc.stderr
    assert _receipt_rows(home) == []


def test_unopenable_store_is_unavailable(tmp_path):
    home = tmp_path / "home"
    (home / "state.db").mkdir(parents=True)

    proc, _ = _run(home, _request())

    assert (proc.returncode, proc.stdout) == (1, _UNAVAILABLE), proc.stderr


def _empty_sqlite(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 1")  # writes a real header; no Hermes tables
    conn.close()


_FOREIGN_STORES = {
    "missing": lambda path: None,
    "garbage": lambda path: path.write_bytes(b"not a database\n" * 64),
    "empty": lambda path: path.write_bytes(b""),
    "no hermes tables": _empty_sqlite,
}


@pytest.mark.parametrize("make_store", list(_FOREIGN_STORES.values()), ids=list(_FOREIGN_STORES))
def test_absent_or_foreign_store_is_unavailable_and_left_alone(tmp_path, monkeypatch, make_store):
    home = _scaffold(tmp_path / "home", monkeypatch)
    make_store(home / "state.db")
    before = _snapshot(home)

    proc, _ = _run(home, _request())

    assert (proc.returncode, proc.stdout) == (1, _UNAVAILABLE), proc.stderr
    # Bind creates, quarantines and scaffolds nothing: only the CLI's own startup writes remain.
    assert _changed(before, _snapshot(home)) <= _STARTUP_WRITES


def _seeded_store_with_meta(key: str, value: str):
    def make(path: Path) -> None:
        _seed(path.parent, ("root", "mid-1", "mid-2", "tip"), ("other",))
        conn = sqlite3.connect(path, isolation_level=None)
        try:
            conn.execute("INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)", (key, value))
        finally:
            conn.close()

    return make


# Stores the writer open would migrate, past any bound bind can keep: the fork's generation (fence swap
# and lineage stamp, then the data migrations), and a store this build stamped whose FTS it would rebuild.
_UNSETTLED_STORES = {
    "fork generation 29": lambda path: build_store(path, "fork29"),
    "stale fts index": _seeded_store_with_meta(FTS_STALE_KEY, "1"),
    "older fts storage": _seeded_store_with_meta("fts_storage_version", str(FTS_STORAGE_VERSION - 1)),
}


@pytest.mark.parametrize("make_store", list(_UNSETTLED_STORES.values()), ids=list(_UNSETTLED_STORES))
def test_unsettled_store_is_unavailable_and_byte_identical(tmp_path, monkeypatch, make_store):
    home = _scaffold(tmp_path / "home", monkeypatch)
    make_store(home / "state.db")
    before = _snapshot(home)

    proc, _ = _run(home, _request())

    assert (proc.returncode, proc.stdout) == (1, _UNAVAILABLE), proc.stderr
    # state.db keeps every byte (stamp, fences, sqlite_master, schema cookie) and gains no sidecar.
    assert _changed(before, _snapshot(home)) <= _STARTUP_WRITES


def test_fork_store_this_build_migrated_binds(tmp_path, monkeypatch):
    home = _scaffold(tmp_path / "home", monkeypatch)
    build_store(home / "state.db", "fork29")
    # The seeding open is the store's migration; bind's own open finds it settled.
    _seed(home, ("root", "mid-1", "mid-2", "tip"), ("other",))
    request = _request()

    proc, _ = _run(home, request)

    assert proc.returncode == 0, proc.stderr
    _assert_receipt(proc.stdout, request)


@pytest.fixture
def instrumented_home(home, tmp_path):
    """A home whose config runs a command secret helper and loads a plugin that prints at import."""
    sentinel = tmp_path / "secret-helper-ran"
    plugin = home / "plugins" / "noisy"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: noisy\nversion: 0.1.0\ndescription: noisy\n", encoding="utf-8")
    (plugin / "__init__.py").write_text(
        "print('NOISY-PLUGIN-IMPORTED')\n\n\ndef register(ctx):\n    pass\n", encoding="utf-8"
    )
    # ``:`` and the redirect are shell builtins, so the helper needs no PATH.
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [noisy]\n"
        f"secrets:\n  command:\n    enabled: true\n    command: ': > {sentinel}'\n",
        encoding="utf-8",
    )
    return home, sentinel


@pytest.mark.parametrize(
    "prefix",
    [(), ("--safe-mode",), ("--reasoning", "high"), ("-r", "some-session")],
    ids=["bare", "flag", "value-flag", "resume-value"],
)
def test_bind_runs_no_secret_helper_and_no_plugin(instrumented_home, prefix):
    home, sentinel = instrumented_home
    request = _request()
    proc, _ = _run(home, request, (*prefix, *_BIND))

    assert proc.returncode == 0, proc.stderr
    _assert_receipt(proc.stdout, request)
    assert not sentinel.exists()


def test_instrumentation_is_live_outside_bind(instrumented_home):
    home, sentinel = instrumented_home
    # ``target`` alone is not the bind preflight, so the helper runs.
    group, _ = _run(home, b"", ("target",))
    assert sentinel.exists(), group.stderr
    sentinel.unlink()

    # The parser consumes ``target`` as the --reasoning value: argv merely CONTAINING the words is not
    # the preflight, so the helper runs and the plugin (discovered for the unknown ``bind``) prints.
    lookalike, _ = _run(home, _request(), ("--reasoning", *_BIND))
    assert sentinel.exists(), lookalike.stderr
    assert b"NOISY-PLUGIN-IMPORTED" in lookalike.stdout
    assert _receipt_rows(home) == []


# Cross-language vector. ``receipt_digest`` below was computed by ``digestOf`` from the controller's
# ``src/core/digest.ts`` (run read-only under node over exactly these seven fields). The controller
# receives ``lineage_root_digest`` as an input; its value is the domain digest of the root id.
_VECTOR_ROOT = "root-Ω"
_VECTOR_FIELDS = {
    "domain": "hermes.target-bind",
    "version": 1,
    "actor_id": "actor:café-한글-\U0001f98a",
    "binding_generation": 9007199254740991,
    "executor_runtime_identity": 'rt:"quoted"\\path',
    "requested_session_id": "tip line",
    "lineage_root_digest": "sha256:f66b69bd39b60d71549f0e7d90b3b7f20a87a913b3f9ecedf8e1ce955f3bab57",
}
_VECTOR_RECEIPT_DIGEST = "sha256:2f0658a894d9b1c45774d56369a3e7067201c1fcab1a6bcd8d8a3257aee79f91"


def test_receipt_digest_matches_the_controller_vector(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    _seed(home, (_VECTOR_ROOT, "mid", _VECTOR_FIELDS["requested_session_id"]))
    # The controller sends ``JSON.stringify`` output: raw UTF-8, no ASCII escaping.
    request = json.dumps(
        {
            "domain": _VECTOR_FIELDS["domain"],
            "version": _VECTOR_FIELDS["version"],
            "session_id": _VECTOR_FIELDS["requested_session_id"],
            "expected_lineage_root_digest": _VECTOR_FIELDS["lineage_root_digest"],
            "actor_id": _VECTOR_FIELDS["actor_id"],
            "binding_generation": _VECTOR_FIELDS["binding_generation"],
            "executor_runtime_identity": _VECTOR_FIELDS["executor_runtime_identity"],
        },
        ensure_ascii=False,
    ).encode("utf-8")

    proc, _ = _run(home, request)

    assert proc.returncode == 0, proc.stderr
    assert _lineage_digest(_VECTOR_ROOT) == _VECTOR_FIELDS["lineage_root_digest"]
    receipt = json.loads(proc.stdout.decode("utf-8"))
    assert receipt == {**_VECTOR_FIELDS, "receipt_digest": _VECTOR_RECEIPT_DIGEST}
    assert _VECTOR_FIELDS["actor_id"].encode("utf-8") in proc.stdout
