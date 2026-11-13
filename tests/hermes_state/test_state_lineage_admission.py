"""Pre-DDL lineage admission: a state.db this build cannot migrate is refused before any write.

The fork line stamps ``schema_version`` as 1000 + its SCHEMA_VERSION and guards governed tables with
turn-fence triggers. This build has no fence function and no migration for those stores, so opening
one must raise ``IncompatibleSchemaError`` and leave the file exactly as it was: same bytes, same
mode, same inode, no sidecar, no quarantine copy. Fresh and pure-upstream stores open as before.

Fixtures are synthesized here; none of them is a copy of a real store.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path

import hermes_yaml as yaml
import pytest

import hermes_state
from hermes_state import SessionDB
from hermes_state_common import SCHEMA_VERSION

GOVERNED = (
    "async_delegations", "compression_locks", "gateway_routing", "messages",
    "session_model_usage", "session_turn_leases", "sessions", "system_prompts",
)
OPERATIONS = ("INSERT", "UPDATE", "DELETE")


def _fence():
    # Imported per test, so each counterexample fails on its own where the module is absent.
    return importlib.import_module("hermes_state_fence")


def live_fence_trigger_sql(table: str, operation: str, generation: int) -> str:
    """The fork's persisted fence declaration, character for character."""
    return (
        f"CREATE TRIGGER turn_fence_{table}_{operation.lower()} BEFORE {operation} ON {table} "
        "BEGIN "
        "SELECT CASE "
        "WHEN typeof(hermes_turn_fence_generation()) != 'integer' "
        f"OR hermes_turn_fence_generation() != {generation} "
        "THEN RAISE(ABORT, 'state DB generation incompatible') "
        "END; "
        "END"
    )


def make_store(path: Path, *, versions=(1030,), version_decl="INTEGER NOT NULL", fences=1030,
               journal="delete", extra_sql=()) -> Path:
    """schema_version + sessions + messages (one row each) and, unless ``fences`` is None, the 8x3
    fence triggers at that generation over stub governed tables."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        assert conn.execute(f"PRAGMA journal_mode={journal}").fetchone()[0].lower() == journal
        conn.execute(f"CREATE TABLE schema_version (version {version_decl})")
        conn.executemany("INSERT INTO schema_version (version) VALUES (?)", [(v,) for v in versions])
        conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT NOT NULL, started_at REAL)")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, "
                     "role TEXT NOT NULL, content TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('s1', 'cli', 1.0)")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES ('s1', 'user', 'hello')")
        for sql in extra_sql:
            conn.execute(sql)
        if fences is not None:
            for table in GOVERNED:
                conn.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY)")
                for operation in OPERATIONS:
                    conn.execute(live_fence_trigger_sql(table, operation, fences))
    finally:
        conn.close()
    os.chmod(path, 0o644)  # not 0600, so a chmod by the opener would show
    return path


def snapshot(path: Path) -> dict:
    st = os.stat(path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mode": stat.S_IMODE(st.st_mode),
        "inode": st.st_ino,
        "listing": sorted(os.listdir(path.parent)),
    }


def assert_untouched(path: Path, before: dict) -> None:
    assert snapshot(path) == before
    for suffix in ("-wal", "-shm", "-journal"):
        assert not Path(f"{path}{suffix}").exists(), suffix


def _upstream_store(path: Path, version: int, *extra_sql: str) -> Path:
    """A real upstream-shaped store from this build, re-stamped to ``version``."""
    db = SessionDB(db_path=path)
    try:
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="hello")
        db._conn.execute("UPDATE schema_version SET version = ?", (version,))
        for sql in extra_sql:
            db._conn.execute(sql)
        db._conn.commit()
    finally:
        db.close()
    os.chmod(path, 0o644)
    return path


# ── Refused fixtures: (builder, cause) ──

_AUTHORITIES = "CREATE TABLE session_process_authorities (session_id TEXT PRIMARY KEY)"
_STATE_FAMILY = ("INSERT OR REPLACE INTO state_meta (key, value) "
                 "VALUES ('session_process_state_family', 'v1')")

# name: (builder, cause, the lineage the probe decodes, or None when the decode itself refuses)
REFUSED = {
    "fenced_1030": (lambda p: make_store(p), "FENCE_GENERATION_MISMATCH", "fenced"),
    "fenced_1033": (lambda p: make_store(p, versions=(1033,), fences=1033), "FENCE_GENERATION_MISMATCH", "fenced"),
    "fenced_1000_unfenced": (
        lambda p: make_store(p, versions=(1000,), fences=None), "FENCE_GENERATION_MISMATCH", "fenced"),
    "fork_legacy_literal_28": (
        lambda p: make_store(p, versions=(29,), fences=28), "FENCE_GENERATION_MISMATCH", "fork"),
    "fork_authorities_at_30": (lambda p: _upstream_store(p, 30, _AUTHORITIES), "FENCE_GENERATION_MISMATCH", "fork"),
    "fork_state_family_at_30": (lambda p: _upstream_store(p, 30, _STATE_FAMILY), "FENCE_GENERATION_MISMATCH", "fork"),
    "upstream_stamp_with_fences": (
        lambda p: make_store(p, versions=(30,), fences=1030), "FENCE_GENERATION_MISMATCH", "upstream"),
    "unowned_fence_body": (
        lambda p: make_store(p, versions=(30,), fences=None, extra_sql=(
            "CREATE TRIGGER turn_fence_sessions_insert BEFORE INSERT ON sessions BEGIN SELECT 1; END",)),
        "FENCE_GENERATION_MISMATCH", None),
    "fenced_1034": (lambda p: make_store(p, versions=(1034,), fences=None), "BUILD_TOO_OLD", None),
    "fence_literal_1034": (lambda p: make_store(p, versions=(1033,), fences=1034), "BUILD_TOO_OLD", None),
    "upstream_34": (lambda p: _upstream_store(p, 34), "BUILD_TOO_OLD", None),
    "two_version_rows": (
        lambda p: make_store(p, versions=(30, 31), fences=None), "SCHEMA_VERSION_UNREADABLE", None),
    "text_version": (
        lambda p: make_store(p, versions=("30",), version_decl="", fences=None), "SCHEMA_VERSION_UNREADABLE", None),
}


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_refused_lineages_raise_and_leave_the_file_byte_identical(tmp_path, name):
    fence = _fence()
    build, cause, lineage = REFUSED[name]
    db = build(tmp_path / "store" / "state.db")
    before = snapshot(db)

    if lineage is None:
        with pytest.raises(fence.IncompatibleSchemaError) as decoded:
            fence.probe_store_lineage(db)
        assert decoded.value.cause == cause
    else:
        assert fence.probe_store_lineage(db).lineage == lineage
    with pytest.raises(fence.IncompatibleSchemaError) as caught:
        fence.refuse_unadmitted_lineage(db)

    assert caught.value.cause == cause
    assert caught.value.code == "STATE_DB_SCHEMA_INCOMPATIBLE"
    assert isinstance(caught.value, RuntimeError) and not isinstance(caught.value, sqlite3.DatabaseError)
    assert_untouched(db, before)


def test_live_1030_refusal_message_names_the_unadmitted_migration(tmp_path):
    fence = _fence()
    db = make_store(tmp_path / "state.db")

    with pytest.raises(fence.IncompatibleSchemaError) as caught:
        fence.refuse_unadmitted_lineage(db)

    assert (caught.value.expected_generation, caught.value.actual_generation) == (1033, 1030)
    assert str(caught.value) == (
        "Session state turn-fence generation does not match this Hermes build "
        "(build generation 1033, stored generation 1030) "
        "(fenced lineage at schema_version 1030; migration not admitted in this build); "
        "refusing to open it, and nothing was changed."
    )


def test_constants_follow_this_builds_schema_version():
    fence = _fence()
    assert fence.FENCE_LINEAGE_BASE == 1000
    assert fence.STORED_SCHEMA_VERSION == 1000 + SCHEMA_VERSION == 1033
    assert fence.FORK_LEGACY_GENERATIONS == {27, 28, 29}
    assert fence.FORK_BASE_UPSTREAM_GATE == 26


def test_live_fence_fixture_decodes_as_fenced_1030(tmp_path):
    fence = _fence()
    db = make_store(tmp_path / "state.db")
    with sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as conn:
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0] == 24
        literals = fence.owned_turn_fence_literals(conn.cursor())
    conn.close()
    assert len(literals) == 24 and set(literals.values()) == {1030}
    lineage = fence.probe_store_lineage(db)
    assert (lineage.lineage, lineage.stored, lineage.gate) == ("fenced", 1030, 30)


# ── Admitted fixtures ──

def test_missing_and_empty_paths_are_fresh_and_not_created(tmp_path):
    fence = _fence()
    missing = tmp_path / "missing" / "state.db"
    assert fence.refuse_unadmitted_lineage(missing).lineage == "fresh"
    assert not missing.parent.exists()
    empty = tmp_path / "empty" / "state.db"
    empty.parent.mkdir()
    empty.touch()
    assert fence.refuse_unadmitted_lineage(empty).lineage == "fresh"
    assert empty.stat().st_size == 0 and os.listdir(empty.parent) == ["state.db"]


@pytest.mark.parametrize("version", [30, 33])
def test_pure_upstream_stores_are_admitted_untouched(tmp_path, version):
    fence = _fence()
    db = _upstream_store(tmp_path / "store" / "state.db", version)
    before = snapshot(db)
    lineage = fence.refuse_unadmitted_lineage(db)
    assert (lineage.lineage, lineage.stored, lineage.fence_literals) == ("upstream", version, frozenset())
    assert_untouched(db, before)


def _corrupt_schema_with_raw_bytes(db_path: Path) -> None:
    """sqlite_master text that is not valid UTF-8 (the #98924 fixture)."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("PRAGMA writable_schema=ON")
        badname = b"tbl_\x81\x82"
        conn.execute(
            "UPDATE sqlite_master SET name=CAST(? AS TEXT), tbl_name=CAST(? AS TEXT), sql=CAST(? AS TEXT) "
            "WHERE name='messages_fts_trigram'",
            (badname, badname, b"CREATE GARBAGE \x81\x82"),
        )
        ver = conn.execute("PRAGMA schema_version").fetchone()[0]
        conn.execute(f"PRAGMA schema_version = {ver + 1}")
        conn.execute("PRAGMA writable_schema=OFF")
    finally:
        conn.close()


def test_invalid_utf8_schema_is_damage_handed_to_the_existing_probes(tmp_path):
    fence = _fence()
    db = _upstream_store(tmp_path / "store" / "state.db", SCHEMA_VERSION)
    _corrupt_schema_with_raw_bytes(db)
    before = snapshot(db)
    assert fence.refuse_unadmitted_lineage(db).lineage == "damaged"
    assert_untouched(db, before)
    # The target's own read-only probes still survive it (#98924).
    reader = SessionDB(db_path=db, read_only=True)
    reader.close()


# ── SessionDB: writable and read-only opens ──

@pytest.fixture
def open_spies(monkeypatch):
    calls = {"connect_and_init": 0, "repair": 0, "claim_repair": 0, "read_connect": 0}
    real_init = SessionDB._connect_and_init
    real_read = SessionDB._connect_read_only

    def counting_init(self):
        calls["connect_and_init"] += 1
        return real_init(self)

    def counting_read(self, *args, **kwargs):
        calls["read_connect"] += 1
        return real_read(self, *args, **kwargs)

    def no_repair(path):
        calls["repair"] += 1
        return {"repaired": False}

    def no_claim(path):
        calls["claim_repair"] += 1
        return False

    monkeypatch.setattr(SessionDB, "_connect_and_init", counting_init)
    monkeypatch.setattr(SessionDB, "_connect_read_only", counting_read)
    monkeypatch.setattr(hermes_state, "repair_state_db_schema", no_repair)
    monkeypatch.setattr(hermes_state, "_claim_repair_attempt", no_claim)
    return calls


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_writable_session_db_refuses_before_any_write(tmp_path, open_spies, name):
    build, cause, _ = REFUSED[name]
    db = build(tmp_path / "store" / "state.db")
    open_spies.update(dict.fromkeys(open_spies, 0))  # some builders open a SessionDB themselves
    before = snapshot(db)

    with pytest.raises(RuntimeError) as caught:
        SessionDB(db_path=db)

    assert type(caught.value).__name__ == "IncompatibleSchemaError"
    assert caught.value.cause == cause
    assert_untouched(db, before)
    # Not a DatabaseError, so the malformed-schema repair never sees it and lock patience never retries it.
    assert open_spies["connect_and_init"] == 1
    assert open_spies["repair"] == open_spies["claim_repair"] == 0
    assert hermes_state.get_last_init_error_exc() is caught.value
    assert hermes_state.get_last_init_error().startswith("IncompatibleSchemaError: ")


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_read_only_session_db_refuses_before_serving_a_read(tmp_path, open_spies, name):
    build, cause, _ = REFUSED[name]
    db = build(tmp_path / "store" / "state.db")
    open_spies.update(dict.fromkeys(open_spies, 0))
    before = snapshot(db)

    with pytest.raises(RuntimeError) as caught:
        SessionDB(db_path=db, read_only=True)

    assert type(caught.value).__name__ == "IncompatibleSchemaError"
    assert caught.value.cause == cause
    assert open_spies["read_connect"] == 0
    assert_untouched(db, before)


def test_wal_fixture_without_sidecars_is_probed_immutably_and_leaves_none(tmp_path, monkeypatch):
    fence = _fence()
    db = make_store(tmp_path / "store" / "state.db", journal="wal")
    assert db.read_bytes()[18:20] == b"\x02\x02"  # WAL-mode header, no sidecar on disk
    before = snapshot(db)
    uris = []
    real_decode = fence._decode_at

    def recording_decode(uri, path):
        uris.append(uri)
        return real_decode(uri, path)

    monkeypatch.setattr(fence, "_decode_at", recording_decode)
    for read_only in (False, True):
        with pytest.raises(fence.IncompatibleSchemaError):
            SessionDB(db_path=db, read_only=read_only)
    assert uris and all(uri.endswith("?mode=ro&immutable=1") for uri in uris)
    assert_untouched(db, before)


# ── No regression for the stores this build owns ──

def test_upstream_store_at_30_still_migrates_to_33(tmp_path):
    db = _upstream_store(tmp_path / "state.db", 30)
    migrated = SessionDB(db_path=db)
    try:
        assert migrated._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
        assert migrated.get_session("s1") is not None
    finally:
        migrated.close()


def test_fresh_path_still_creates_and_stamps_33(tmp_path):
    db = tmp_path / "fresh" / "state.db"
    fresh = SessionDB(db_path=db)
    try:
        assert fresh._conn.execute("SELECT version FROM schema_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        fresh.close()


# ── The probe waits on the read budget (live a648eafdfa) ──

# A holder that has written a page leaves a -journal, so the open's lineage probe cannot read the
# file immutably and takes a lock like any reader. A 1 s lock budget gives up at about 3 s of such
# a hold; the read budget outlasts 10 s.
_JOURNAL_HOLD_S = 5.0

_WRITING_HOLDER = """
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN EXCLUSIVE")
conn.execute("INSERT OR REPLACE INTO state_meta(key, value) VALUES ('busy-holder', 'x')")
print("held", flush=True)
time.sleep(float(sys.argv[2]))
conn.execute("COMMIT")
"""


@pytest.fixture()
def delete_mode_db(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"database": {"journal_mode": "delete"}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    db = SessionDB(db_path=home / "state.db")
    assert db._wal_active is False
    assert db._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    db.create_session("s1", source="cli")
    db.append_message("s1", role="user", content="hello")
    yield db
    db.close()


def test_read_only_open_waits_out_a_commit_that_left_a_journal(delete_mode_db):
    holder = subprocess.Popen(
        [sys.executable, "-c", _WRITING_HOLDER, str(delete_mode_db.db_path), str(_JOURNAL_HOLD_S)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert Path(f"{delete_mode_db.db_path}-journal").exists()
        reader = SessionDB(db_path=delete_mode_db.db_path, read_only=True)
        try:
            assert [s["id"] for s in reader.list_sessions_rich(limit=10)] == ["s1"]
        finally:
            reader.close()
    finally:
        assert holder.wait(timeout=60) == 0
