"""Tests for recovery-tooling gaps: issue #80205 (range-query budget can
omit a recoverable tail row) and the lost_and_found last-resort lane for
sources whose table schemas are unreadable.

The corrupted fixtures here are REAL physical SQLite page damage (flipped
b-tree/schema header bytes), not mocked cursor exceptions.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_state import (
    SessionDB,
    SessionTurnLeaseLostError,
    register_turn_fence_generation,
)
from hermes_cli import session_recovery
from hermes_cli.session_lost_and_found import (
    classify_lost_and_found_row,
    map_lost_and_found_rows,
    rebuild_fts_indexes,
    stub_missing_parent_sessions,
)
from hermes_cli.session_recovery import (
    SessionRecoveryDestinationError,
    SessionRecoverySafetyError,
    SessionRecoverySourceError,
    _probe_populated_edge,
    recover_session_database,
)

from tests.hermes_cli.test_session_recovery import (
    _arm_seal_authority_interleaving,
    _assert_seal_authority_interleaving_blocked,
    _btree_leaf_pages,
    _make_page_spanning_source,
)


from hermes_cli.session_lost_and_found import find_sqlite3_cli

# .recover needs a sqlite3 shell built with sqlite_dbpage — PATH presence
# alone is not enough (Ubuntu CI ships a build without it).
HAVE_SQLITE3_CLI = find_sqlite3_cli() is not None


# ── physical corruption helpers ─────────────────────────────────────────────


def _page_size(data: bytes) -> int:
    size = int.from_bytes(data[16:18], "big")
    return 65_536 if size == 1 else size


def _leaf_cell_count(path: Path, page_number: int) -> int:
    data = path.read_bytes()
    page_size = _page_size(data)
    header = (page_number - 1) * page_size + (100 if page_number == 1 else 0)
    assert data[header] in {0x0A, 0x0D}
    return int.from_bytes(data[header + 3 : header + 5], "big")


def _corrupt_leaf(path: Path, page_number: int) -> None:
    data = bytearray(path.read_bytes())
    page_size = _page_size(bytes(data))
    header = (page_number - 1) * page_size + (100 if page_number == 1 else 0)
    assert data[header] in {0x0A, 0x0D}
    data[header + 3 : header + 5] = b"\xff\xff"
    path.write_bytes(data)


def _corrupt_schema_page(path: Path) -> None:
    """Damage the sqlite_master b-tree so no table schema is readable.

    Page 1 holds the schema table root. An impossible cell count in its
    header makes every ``PRAGMA table_info`` / schema read raise
    'database disk image is malformed' while the file still opens and the
    data pages of every table remain physically intact.
    """
    data = bytearray(path.read_bytes())
    assert data[:16] == b"SQLite format 3\x00"
    header = 100
    assert data[header] in {0x02, 0x05, 0x0A, 0x0D}
    data[header + 3 : header + 5] = b"\xff\xff"
    path.write_bytes(data)


def _make_schema_unreadable_source(path: Path) -> dict[str, int]:
    db = SessionDB(db_path=path)
    try:
        for session_number in range(3):
            session_id = f"20260812_1353{session_number:02d}_abc{session_number:03x}"
            db.create_session(session_id, "cli", cwd=f"/tmp/laf-{session_number}")
            db.set_session_title(session_id, f"LAF {session_number}")
            for message_number in range(9):
                db.append_message(
                    session_id,
                    "user" if message_number % 2 == 0 else "assistant",
                    f"lost-and-found payload {session_number} {message_number}",
                )
    finally:
        db.close()
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
    finally:
        conn.close()
    _corrupt_schema_page(path)
    return {"sessions": 3, "messages": 27}


# ── issue #80205: recoverable tail row next to a damaged rowid edge ─────────


def test_exact_lookup_recovers_tail_row_next_to_damaged_high_edge(
    tmp_path: Path,
) -> None:
    """Regression for #80205: a readable boundary row must not be omitted.

    Damaging the RIGHTMOST messages leaf makes the ordered high-edge probe
    fail, so salvage falls back to the full rowid domain. The last readable
    row (the final cell of the last healthy leaf) can only be reached through
    a singleton range once bisection narrows down — and a singleton *range*
    scan must advance past the row into the damaged sibling page to prove the
    range is exhausted, discarding the already-produced row. The fix performs
    an exact ``rowid = ?`` lookup for singleton ranges, which stops at the
    hit and recovers the row exactly as SQLite's page-level ``.recover``
    does.
    """
    source = tmp_path / "tail-damaged.db"
    output = tmp_path / "tail-recovered.db"
    message_count = 320
    messages_root, count_index_root = _make_page_spanning_source(
        source, message_count
    )

    _, leaf_pages = _btree_leaf_pages(source, messages_root)
    assert len(leaf_pages) >= 3
    rightmost_leaf = leaf_pages[-1]
    lost_rows = _leaf_cell_count(source, rightmost_leaf)
    assert 0 < lost_rows < message_count
    boundary_rowid = message_count - lost_rows
    _corrupt_leaf(source, rightmost_leaf)
    if count_index_root is not None:
        _, index_leaves = _btree_leaf_pages(source, count_index_root)
        _corrupt_leaf(source, index_leaves[-1])

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=8,
        allow_partial=True,
    )

    copied = report["copy"]["messages"]
    bounds = copied["rowid_bounds"]
    # Premise check: the high edge probe really failed and fell back.
    assert any("high rowid" in error for error in bounds["errors"]), bounds
    assert "high" in bounds["fallback_edges"]

    conn = sqlite3.connect(str(output))
    try:
        recovered_ids = {
            int(row[0]) for row in conn.execute("SELECT id FROM messages")
        }
    finally:
        conn.close()

    assert 1 in recovered_ids
    # The headline regression: the last readable row before the damage.
    assert boundary_rowid in recovered_ids, (
        f"boundary row {boundary_rowid} was omitted; max recovered "
        f"{max(recovered_ids)}; exact_lookup_recovered="
        f"{copied.get('exact_lookup_recovered')}"
    )
    assert copied["exact_lookup_recovered"] >= 1
    assert recovered_ids == set(range(1, boundary_rowid + 1))
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verified"] is True


def test_probe_populated_edge_caps_synthetic_domain(tmp_path: Path) -> None:
    """The gallop converges on a finite bound in O(log) probes when the
    region beyond the data is cleanly seekable."""
    db_path = tmp_path / "clean.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany(
            "INSERT INTO t (id, v) VALUES (?, ?)",
            [(i, f"value {i}") for i in range(1, 101)],
        )
        probe = _probe_populated_edge(conn, "t", edge="high", anchor=1)
        assert probe["capped"] is True
        assert probe["bound"] >= 100
        assert probe["bound"] < 10_000
        assert probe["probes"] <= 64

        probe_low = _probe_populated_edge(conn, "t", edge="low", anchor=100)
        assert probe_low["capped"] is True
        assert probe_low["bound"] <= 1
    finally:
        conn.close()


# ── lost_and_found lane: unreadable table schemas ───────────────────────────


def test_unreadable_schema_without_cli_names_the_sqlite3_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a sqlite3 CLI the refusal must say exactly what to install."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "schemaless-recovered.db"
    _make_schema_unreadable_source(source)

    import hermes_cli.session_lost_and_found as laf

    monkeypatch.setattr(laf, "find_sqlite3_cli", lambda: None)
    with pytest.raises(SessionRecoverySourceError) as excinfo:
        recover_session_database(
            source,
            output,
            work_dir=tmp_path,
            allow_partial=True,
        )
    message = str(excinfo.value)
    assert "sessions" in message and "messages" in message
    assert "sqlite3" in message
    assert ".recover" in message
    assert not output.exists()


@pytest.mark.parametrize(
    "fault",
    (
        "no-registration",
        "registration-raises",
        "stale-int",
        "string",
        "bytes",
        "float",
        "none",
        "bool",
        "udf-raises",
    ),
)
def test_lost_and_found_destination_fence_failures_leave_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    source = tmp_path / "schemaless.db"
    output = tmp_path / "recovered.db"
    _make_schema_unreadable_source(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    schema_ref = tmp_path / "schema-ref.db"
    SessionDB(db_path=schema_ref).close()

    import hermes_cli.session_lost_and_found as laf

    monkeypatch.setattr(laf, "find_sqlite3_cli", lambda: "synthetic-sqlite3")

    def run_synthetic_recover(
        _snapshot: Path,
        lost_and_found: Path,
        _sqlite3_bin: str,
    ) -> dict[str, str]:
        _make_synthetic_lost_and_found(lost_and_found, schema_ref)
        return {"mode": "synthetic"}

    monkeypatch.setattr(laf, "run_cli_lost_and_found_recover", run_synthetic_recover)

    def register_fault(conn: sqlite3.Connection) -> None:
        if fault == "no-registration":
            return
        if fault == "registration-raises":
            raise sqlite3.OperationalError("private setup failure")
        values = {
            "stale-int": 26,
            "string": "27",
            "bytes": b"27",
            "float": 27.0,
            "none": None,
            "bool": True,
        }
        if fault == "udf-raises":
            def raises_generation() -> None:
                raise RuntimeError("private callback failure")

            callback = raises_generation
        else:
            callback = lambda: values[fault]
        conn.create_function("hermes_turn_fence_generation", 0, callback)

    monkeypatch.setattr(session_recovery, "register_turn_fence_generation", register_fault)
    if fault == "bool":
        monkeypatch.setattr(
            session_recovery,
            "_is_current_turn_fence_generation",
            lambda _value, _sqlite_type: False,
        )

    with pytest.raises(SessionRecoveryDestinationError) as excinfo:
        recover_session_database(source, output, work_dir=tmp_path, allow_partial=True)

    assert str(excinfo.value) == (
        "Recovery destination turn-fence setup is unavailable or incompatible."
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    for suffix in ("", "-wal", "-shm", "-journal"):
        assert not (output.with_name(output.name + suffix)).exists()


@pytest.mark.parametrize(
    ("takeover_marker", "expected_marker"),
    (
        (
            b"\x00foreign-lost-found-seal-owner\xff",
            b"\x00foreign-lost-found-seal-owner\xff",
        ),
        (None, None),
    ),
    ids=("foreign", "null"),
)
def test_lost_and_found_seal_refuses_marker_after_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    takeover_marker: bytes | None,
    expected_marker: bytes | None,
) -> None:
    """The page-level recovery lane shares sealing's no-maintenance fence."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "recovered.db"
    _make_schema_unreadable_source(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    schema_ref = tmp_path / "schema-ref.db"
    SessionDB(db_path=schema_ref).close()

    import hermes_cli.session_lost_and_found as laf

    monkeypatch.setattr(laf, "find_sqlite3_cli", lambda: "synthetic-sqlite3")

    def run_synthetic_recover(
        _snapshot: Path,
        lost_and_found: Path,
        _sqlite3_bin: str,
    ) -> dict[str, str]:
        _make_synthetic_lost_and_found(lost_and_found, schema_ref)
        return {"mode": "synthetic"}

    monkeypatch.setattr(laf, "run_cli_lost_and_found_recover", run_synthetic_recover)

    candidate: Path | None = None
    marker_key = session_recovery._OFFLINE_REBUILD_EPOCH_KEY
    raw_maintenance: list[str] = []
    takeover_complete = False
    sealing_started = False
    original_create_stage = session_recovery._create_destination_stage
    original_seal = session_recovery._seal_staged_database
    original_refresh = session_recovery._refresh_stage_children
    original_connect = sqlite3.connect

    def capture_stage(path: Path):
        nonlocal candidate
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    def trace_sealing_connection(database, *args, **kwargs):
        connection = original_connect(database, *args, **kwargs)
        if takeover_complete and candidate is not None and Path(database) == candidate:
            connection.set_trace_callback(
                lambda sql: raw_maintenance.append(sql)
                if "WAL_CHECKPOINT(TRUNCATE)" in sql.upper()
                or "JOURNAL_MODE=DELETE" in sql.upper()
                else None
            )
        return connection

    def take_over_after_seal_refresh(stage, **kwargs):
        nonlocal takeover_complete
        result = original_refresh(stage, **kwargs)
        if not sealing_started or takeover_complete:
            return result
        assert candidate == stage.candidate
        with original_connect(str(stage.candidate), isolation_level=None) as writer:
            assert writer.execute(
                "SELECT value FROM state_meta WHERE key = ?", (marker_key,)
            ).fetchone() is None
            writer.execute(
                "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                (
                    marker_key,
                    None
                    if takeover_marker is None
                    else sqlite3.Binary(takeover_marker),
                ),
            )
        takeover_complete = True
        return result

    def run_seal_after_takeover_seam(stage):
        nonlocal sealing_started
        assert candidate == stage.candidate
        sealing_started = True
        return original_seal(stage)

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_cleanup_destination_stage", lambda _stage: None)
    monkeypatch.setattr(session_recovery.sqlite3, "connect", trace_sealing_connection)
    monkeypatch.setattr(
        session_recovery, "_refresh_stage_children", take_over_after_seal_refresh
    )
    monkeypatch.setattr(session_recovery, "_seal_staged_database", run_seal_after_takeover_seam)

    caught: BaseException | None = None
    try:
        recover_session_database(source, output, work_dir=tmp_path, allow_partial=True)
    except BaseException as exc:
        caught = exc

    assert raw_maintenance == []
    assert isinstance(caught, SessionTurnLeaseLostError)
    assert takeover_complete is True
    assert candidate is not None
    assert not output.exists()
    with original_connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT CAST(value AS BLOB) FROM state_meta WHERE key = ?",
            (marker_key,),
        ).fetchone() == (expected_marker,)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


@pytest.mark.parametrize(
    ("authority_select", "takeover_marker"),
    (
        (1, b"\x00foreign-lost-found-checkpoint-owner\xff"),
        (1, None),
        (2, b"\x00foreign-lost-found-journal-owner\xff"),
        (2, None),
    ),
    ids=("checkpoint-foreign", "checkpoint-null", "journal-foreign", "journal-null"),
)
def test_lost_and_found_seal_blocks_takeover_at_pragma_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_select: int,
    takeover_marker: bytes | None,
) -> None:
    """The lost-and-found lane shares the seal's continuous SQLite authority."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "recovered.db"
    _make_schema_unreadable_source(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    schema_ref = tmp_path / "schema-ref.db"
    SessionDB(db_path=schema_ref).close()

    import hermes_cli.session_lost_and_found as laf

    monkeypatch.setattr(laf, "find_sqlite3_cli", lambda: "synthetic-sqlite3")

    def run_synthetic_recover(
        _snapshot: Path,
        lost_and_found: Path,
        _sqlite3_bin: str,
    ) -> dict[str, str]:
        _make_synthetic_lost_and_found(lost_and_found, schema_ref)
        return {"mode": "synthetic"}

    monkeypatch.setattr(laf, "run_cli_lost_and_found_recover", run_synthetic_recover)
    state = _arm_seal_authority_interleaving(
        monkeypatch,
        authority_select=authority_select,
        takeover_marker=takeover_marker,
    )

    report = None
    recovery_error: BaseException | None = None
    try:
        report = recover_session_database(
            source, output, work_dir=tmp_path, allow_partial=True
        )
    except BaseException as exc:
        recovery_error = exc

    assert state["authority_selects"] >= authority_select
    _assert_seal_authority_interleaving_blocked(state)
    assert recovery_error is None
    assert report is not None
    assert report["verified"] is True
    assert output.exists()
    with sqlite3.connect(str(output)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (session_recovery._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() is None
        assert destination.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_lost_and_found_publication_refuses_marker_after_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The page-level recovery lane uses publication's final no-owner proof."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "recovered.db"
    _make_schema_unreadable_source(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    schema_ref = tmp_path / "schema-ref.db"
    SessionDB(db_path=schema_ref).close()

    import hermes_cli.session_lost_and_found as laf

    monkeypatch.setattr(laf, "find_sqlite3_cli", lambda: "synthetic-sqlite3")

    def run_synthetic_recover(
        _snapshot: Path,
        lost_and_found: Path,
        _sqlite3_bin: str,
    ) -> dict[str, str]:
        _make_synthetic_lost_and_found(lost_and_found, schema_ref)
        return {"mode": "synthetic"}

    monkeypatch.setattr(laf, "run_cli_lost_and_found_recover", run_synthetic_recover)

    candidate: Path | None = None
    foreign_marker = "foreign-lost-found-publication-owner"
    original_create_stage = session_recovery._create_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    def take_over_after_barrier(staged_candidate: Path, _output: Path) -> None:
        assert candidate == staged_candidate
        with sqlite3.connect(str(staged_candidate), isolation_level=None) as writer:
            assert writer.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (session_recovery._OFFLINE_REBUILD_EPOCH_KEY,),
            ).fetchone() is None
            writer.execute(
                "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                (session_recovery._OFFLINE_REBUILD_EPOCH_KEY, foreign_marker),
            )

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_publication_barrier", take_over_after_barrier)

    with pytest.raises(SessionTurnLeaseLostError):
        recover_session_database(source, output, work_dir=tmp_path, allow_partial=True)

    assert candidate is not None and candidate.exists()
    assert not output.exists()
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (session_recovery._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (foreign_marker,)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash


def test_lost_and_found_copy_refusal_retains_the_real_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Page-level copy loss preserves its stage and cannot reach publication."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "recovered.db"
    _make_schema_unreadable_source(source)
    schema_ref = tmp_path / "schema-ref.db"
    SessionDB(db_path=schema_ref).close()

    import hermes_cli.session_lost_and_found as laf

    monkeypatch.setattr(laf, "find_sqlite3_cli", lambda: "synthetic-sqlite3")

    def run_synthetic_recover(
        _snapshot: Path,
        lost_and_found: Path,
        _sqlite3_bin: str,
    ) -> dict[str, str]:
        _make_synthetic_lost_and_found(lost_and_found, schema_ref)
        return {"mode": "synthetic"}

    candidate: Path | None = None
    stage = None
    copy_started = False
    cleanup_calls = 0
    original_create_stage = session_recovery._create_destination_stage
    original_map_rows = laf.map_lost_and_found_rows
    original_cleanup = session_recovery._cleanup_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate, stage
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    def take_over_during_real_mapping(lf_conn, destination):
        nonlocal copy_started
        assert candidate is not None
        copy_started = True
        with sqlite3.connect(str(candidate), isolation_level=None) as writer:
            writer.execute(
                "UPDATE state_meta SET value = ? WHERE key = ?",
                (
                    "foreign-lost-and-found-copy-owner",
                    session_recovery._OFFLINE_REBUILD_EPOCH_KEY,
                ),
            )
        assert stage is not None
        session_recovery._refresh_stage_children(stage, require_main=True)
        return original_map_rows(lf_conn, destination)

    def record_real_cleanup(destination_stage) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        original_cleanup(destination_stage)

    monkeypatch.setattr(laf, "run_cli_lost_and_found_recover", run_synthetic_recover)
    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(laf, "map_lost_and_found_rows", take_over_during_real_mapping)
    monkeypatch.setattr(
        session_recovery, "_cleanup_destination_stage", record_real_cleanup
    )
    monkeypatch.setattr(
        session_recovery,
        "_seal_staged_database",
        lambda _stage: pytest.fail("copy refusal reached staged-database sealing"),
    )
    monkeypatch.setattr(
        session_recovery,
        "_publish_staged_database",
        lambda _stage, _output: pytest.fail("copy refusal reached publication"),
    )

    with pytest.raises(SessionTurnLeaseLostError):
        recover_session_database(source, output, work_dir=tmp_path, allow_partial=True)

    assert copy_started is True
    assert stage is not None and stage.retain_on_authority_refusal is True
    assert cleanup_calls == 0
    assert candidate is not None and candidate.exists()
    assert not output.exists()
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (session_recovery._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == ("foreign-lost-and-found-copy-owner",)


@pytest.mark.skipif(
    not HAVE_SQLITE3_CLI,
    reason="sqlite3 CLI not on PATH; .recover is a shell-only feature",
)
def test_lost_and_found_lane_recovers_schema_unreadable_source(
    tmp_path: Path,
) -> None:
    """The last-resort lane must salvage rows SQL-level recovery cannot."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "schemaless-recovered.db"
    expected = _make_schema_unreadable_source(source)

    # Premise: the schema really is unreadable at the SQL level.
    probe = sqlite3.connect(str(source))
    try:
        with pytest.raises(sqlite3.DatabaseError):
            probe.execute("SELECT COUNT(*) FROM messages").fetchone()
    finally:
        probe.close()

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        allow_partial=True,
    )

    assert report["mode"] == "lost_and_found_salvage"
    assert report["best_effort"] is True
    assert report["partial"] is True
    assert report["complete"] is False
    assert report["installed"] is False
    assert report["unreadable_schemas"] == ["sessions", "messages"]
    assert any(
        "BEST-EFFORT" in warning
        for warning in report["verification"]["warnings"]
    )
    assert report["output"] == str(output)
    assert ".hermes-session-recovery-" not in json.dumps(report)
    assert output.exists()
    for suffix in ("-wal", "-shm", "-journal"):
        assert not os.path.lexists(output.with_name(output.name + suffix))

    conn = sqlite3.connect(str(output))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        orphans = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id NOT IN "
            "(SELECT id FROM sessions)"
        ).fetchone()[0]
        fts_matches = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            ("payload",),
        ).fetchone()[0]
    finally:
        conn.close()

    assert session_count == expected["sessions"]
    assert message_count == expected["messages"]
    assert orphans == 0
    assert fts_matches == expected["messages"]

    # The output must open as a regular current-schema session database.
    recovered_db = SessionDB(db_path=output)
    try:
        sessions = recovered_db.list_sessions_rich(limit=10)
        assert len(sessions) == expected["sessions"]
    finally:
        recovered_db.close()


@pytest.mark.skipif(
    not HAVE_SQLITE3_CLI,
    reason="sqlite3 CLI not on PATH; .recover is a shell-only feature",
)
def test_page_level_salvage_stops_later_dml_when_marker_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page-level map commit must not authorize stubs or derived writes."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "recovered.db"
    _make_schema_unreadable_source(source)
    candidate: Path | None = None
    original_create_stage = session_recovery._create_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_cleanup_destination_stage", lambda _stage: None)
    monkeypatch.setattr(
        session_recovery,
        "_refresh_stage_children",
        lambda _stage, **_kwargs: None,
    )
    monkeypatch.setattr(session_recovery, "_seal_staged_database", lambda _stage: None)
    monkeypatch.setattr(
        session_recovery,
        "_publish_staged_database",
        lambda _stage, _output: None,
    )

    import hermes_cli.session_lost_and_found as laf

    original_map = laf.map_lost_and_found_rows
    original_rebuild = laf.rebuild_fts_indexes
    foreign_marker = "foreign-page-level-owner"
    adversarial_fts_value = "foreign-page-level-fts-storage-version"
    adversarial_meta_value = "foreign-page-level-adversarial-value"
    foreign_session_id = "foreign-page-level-orphan"
    rebuilt_fts = False
    replaced = False

    def replace_marker_after_map(lf_conn, destination):
        nonlocal replaced
        report = original_map(lf_conn, destination)
        assert candidate is not None
        with sqlite3.connect(str(candidate), isolation_level=None) as writer:
            marker = writer.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (session_recovery._OFFLINE_REBUILD_EPOCH_KEY,),
            ).fetchone()
            if marker is None:
                writer.execute(
                    "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                    (session_recovery._OFFLINE_REBUILD_EPOCH_KEY, foreign_marker),
                )
            else:
                cursor = writer.execute(
                    "UPDATE state_meta SET value = ? WHERE key = ?",
                    (foreign_marker, session_recovery._OFFLINE_REBUILD_EPOCH_KEY),
                )
                assert cursor.rowcount == 1
            writer.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                ("fts_storage_version", adversarial_fts_value),
            )
            writer.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                ("adversarial-after-page-level-marker", adversarial_meta_value),
            )
            register_turn_fence_generation(writer)
            writer.execute(
                "INSERT INTO messages(id, session_id, role, content, timestamp) "
                "VALUES (999999, ?, 'user', 'foreign orphan', 1.0)",
                (foreign_session_id,),
            )
        replaced = True
        return report

    def observe_fts_rebuild(destination):
        nonlocal rebuilt_fts
        rebuilt_fts = True
        return original_rebuild(destination)

    monkeypatch.setattr(laf, "map_lost_and_found_rows", replace_marker_after_map)
    monkeypatch.setattr(laf, "rebuild_fts_indexes", observe_fts_rebuild)

    with pytest.raises(SessionTurnLeaseLostError):
        recover_session_database(source, output, work_dir=tmp_path, allow_partial=True)

    assert replaced is True
    assert rebuilt_fts is False
    assert candidate is not None
    assert output.exists() is False
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (session_recovery._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (foreign_marker,)
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_storage_version'"
        ).fetchone() == (adversarial_fts_value,)
        assert destination.execute(
            "SELECT value FROM state_meta "
            "WHERE key = 'adversarial-after-page-level-marker'"
        ).fetchone() == (adversarial_meta_value,)
        assert destination.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?",
            (foreign_session_id,),
        ).fetchone() == (1,)
        assert destination.execute(
            "SELECT id FROM sessions WHERE id = ?", (foreign_session_id,)
        ).fetchone() is None


# ── mapper unit tests (no sqlite3 CLI required) ─────────────────────────────


def _make_synthetic_lost_and_found(
    path: Path,
    dest_schema_db: Path,
) -> dict[str, int]:
    """Build a .recover-shaped lost_and_found DB directly, no CLI needed."""
    schema = sqlite3.connect(str(dest_schema_db))
    try:
        sessions_columns = [
            str(row[1]) for row in schema.execute("PRAGMA table_info(sessions)")
        ]
        messages_columns = [
            str(row[1]) for row in schema.execute("PRAGMA table_info(messages)")
        ]
        usage_columns = [
            str(row[1])
            for row in schema.execute("PRAGMA table_info(session_model_usage)")
        ]
    finally:
        schema.close()
    # Width is derived from the live schema so ordinary column additions
    # don't break this test (it pinned 54, then 55, then 56 in one week).
    # The floor guards against accidentally reading an empty/old schema.
    current_width = len(sessions_columns)
    assert current_width >= 55
    assert len(usage_columns) == 18

    max_fields = current_width
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        cells = ", ".join(f"c{i}" for i in range(max_fields))
        conn.execute(
            f"CREATE TABLE lost_and_found (rootpgno INTEGER, pgno INTEGER, "
            f"nfield INTEGER, id INTEGER, {cells})"
        )

        def insert(nfield: int, rowid, values: list) -> None:
            padded = list(values) + [None] * (max_fields - len(values))
            placeholders = ", ".join("?" for _ in range(4 + max_fields))
            conn.execute(
                f"INSERT INTO lost_and_found VALUES ({placeholders})",
                [2, 5, nfield, rowid, *padded],
            )

        def session_row(session_id: str, ncols: int) -> list:
            base = {
                "id": session_id,
                "source": "telegram",
                "started_at": 1_754_000_000.0,
                "message_count": 2,
                # A current 57-column source must carry a real physical
                # generation at its current-layout slot; literal 52 ignores it.
                "session_generation": 2,
                "title": f"synthetic {session_id}",
            }
            source_columns = (
                _REAL_52_SESSION_COLUMNS
                if ncols == len(_REAL_52_SESSION_COLUMNS)
                else sessions_columns
            )
            return [base.get(column) for column in source_columns[:ncols]]

        # Current layout (dynamic width) and literal real 52-column layout.
        insert(max_fields, 1, session_row("20260101_010101_aaa001", max_fields))
        insert(52, 2, session_row("20260202_020202_bbb002", 52))
        # 14-column legacy layout: identity + a plausible epoch timestamp.
        legacy = ["20250303_030303_ccc003", "cli", 1_741_000_000.0] + [None] * 11
        insert(14, 3, legacy)

        # messages rows: NULL first cell (rowid alias), session id second,
        # role third.
        for index, (session_id, role, content) in enumerate(
            [
                ("20260101_010101_aaa001", "user", "hello from user"),
                ("20260101_010101_aaa001", "assistant", "hello from assistant"),
                ("20261111_111111_ddd004", "user", "orphaned message payload"),
                ("20261111_111111_ddd004", "tool", "orphaned tool payload"),
            ]
        ):
            row = {
                "id": None,
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": 1_754_000_100.0 + index,
            }
            insert(
                23,
                100 + index,
                [row.get(column) for column in messages_columns[:23]],
            )

        # session_model_usage: 18 columns, orphaned session id on purpose.
        usage = {
            "session_id": "20261212_121212_eee005",
            "model": "test/model",
            "billing_provider": "",
            "billing_base_url": "",
            "billing_mode": "",
            "task": "",
            "api_call_count": 4,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": 0.01,
            "actual_cost_usd": 0.01,
            "first_seen": 1_754_000_000.0,
            "last_seen": 1_754_000_500.0,
        }
        insert(18, 200, [usage.get(column) for column in usage_columns])

        # Junk that must NOT be classified into canonical tables.
        insert(3, 300, ["random", "noise", 42])
        insert(max_fields, 301, ["not-a-session-id", "cli"] + [None] * (max_fields - 2))
        insert(23, 302, [None, "sess-x", "not-a-role", "junk"])
    finally:
        conn.close()
    return {
        "sessions": 3,
        "messages": 4,
        "session_model_usage": 1,
        "junk": 3,
    }


def test_classify_lost_and_found_row_sentinels() -> None:
    assert (
        classify_lost_and_found_row(
            23, (None, "20260101_010101_aaa001", "user", "hi")
        )
        == "messages"
    )
    assert (
        classify_lost_and_found_row(
            55, ("20260101_010101_aaa001", "cli") + (None,) * 53
        )
        == "sessions"
    )
    assert (
        classify_lost_and_found_row(
            52, ("20260101_010101_aaa001", "discord") + (None,) * 50
        )
        == "sessions"
    )
    assert (
        classify_lost_and_found_row(
            14, ("20250101_010101_zzz999", "cli") + (None,) * 12
        )
        == "sessions"
    )
    assert (
        classify_lost_and_found_row(
            18, ("20260101_010101_aaa001", "gpt-x") + (None,) * 16
        )
        == "session_model_usage"
    )
    # Junk shapes.
    assert classify_lost_and_found_row(3, ("random", "noise", 42)) is None
    assert (
        classify_lost_and_found_row(55, ("not-a-session-id", "cli") + (None,) * 53)
        is None
    )
    assert (
        classify_lost_and_found_row(23, (None, "sess", "not-a-role", "x")) is None
    )
    assert classify_lost_and_found_row(0, ()) is None


def test_mapper_rebuilds_sessiondb_from_synthetic_lost_and_found(
    tmp_path: Path,
) -> None:
    """Binary-independent: mapper + stubbing + FTS rebuild end to end."""
    schema_ref = tmp_path / "schema-ref.db"
    SessionDB(db_path=schema_ref).close()

    lf_path = tmp_path / "lost_and_found.db"
    expected = _make_synthetic_lost_and_found(lf_path, schema_ref)

    output = tmp_path / "mapped.db"
    SessionDB(db_path=output).close()

    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        # ``output`` was created via SessionDB then closed, same as the real
        # lost-and-found salvage path in session_recovery.py: this reopen
        # needs its own hermes_turn_fence_generation() registration or the
        # turn-fence triggers on the canonical tables reject the first write.
        register_turn_fence_generation(dest)
        dest.execute("PRAGMA foreign_keys=OFF")
        mapping = map_lost_and_found_rows(lf_conn, dest)
        stubbing = stub_missing_parent_sessions(dest)
        fts = rebuild_fts_indexes(dest)

        assert mapping["mapped"]["sessions"] == expected["sessions"]
        assert mapping["mapped"]["messages"] == expected["messages"]
        assert (
            mapping["mapped"]["session_model_usage"]
            == expected["session_model_usage"]
        )
        assert mapping["legacy_minimal_sessions"] == 1
        assert mapping["unmapped_rows"] == expected["junk"]

        # Orphaned children got stub parents — never deleted.
        assert stubbing["sessions_stubbed"] == 2  # ddd004 + eee005
        assert stubbing["messages_retained"] == 2
        message_count = dest.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert message_count == expected["messages"]
        usage_count = dest.execute(
            "SELECT COUNT(*) FROM session_model_usage"
        ).fetchone()[0]
        assert usage_count == expected["session_model_usage"]

        stub_titles = [
            str(row[0])
            for row in dest.execute(
                "SELECT title FROM sessions WHERE source = 'recovered'"
            )
        ]
        assert len(stub_titles) == 2
        assert all(title.startswith("[best-effort recovered") for title in stub_titles)

        # The 52-col row landed with its real metadata preserved.
        row = dest.execute(
            "SELECT source, title FROM sessions WHERE id = ?",
            ("20260202_020202_bbb002",),
        ).fetchone()
        assert row == ("telegram", "synthetic 20260202_020202_bbb002")

        assert fts.get("messages_fts") == "rebuilt"
        fts_hits = dest.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            ("payload OR hello",),
        ).fetchone()[0]
        assert fts_hits == expected["messages"]

        assert dest.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert dest.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        lf_conn.close()
        dest.close()

    # And the mapped output opens through the normal SessionDB path.
    db = SessionDB(db_path=output)
    try:
        assert len(db.list_sessions_rich(limit=20)) == 5
    finally:
        db.close()


def test_mapper_schema_less_current_session_mints_destination_authority(
    tmp_path: Path,
) -> None:
    """A current page-level row must not import its source generation/capabilities."""
    source_session_id = "20260831_235959_abc123"
    source_authority_token = "a" * 64
    source_reservation_token = "b" * 64
    source_reservation_id = "source-reservation-id"

    lf_path = tmp_path / "current-layout-lost-and-found.db"
    SessionDB(db_path=lf_path).close()
    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    output = tmp_path / "mapped.db"
    SessionDB(db_path=output).close()
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        register_turn_fence_generation(lf_conn)
        session_columns = [
            str(row[1]) for row in dest.execute("PRAGMA table_info(sessions)")
        ]
        generation_index = session_columns.index("session_generation")
        assert len(session_columns) == 57
        assert generation_index == 29

        source_state_db_id = lf_conn.execute(
            "SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'"
        ).fetchone()[0]
        lf_conn.execute(
            "INSERT INTO session_process_authorities "
            "(session_id, session_generation, state_db_id, state_family, "
            "authority_token, status, issued_at) VALUES (?, 2, ?, 'sessiondb-v1', "
            "?, 'ISSUED', 1)",
            (source_session_id, source_state_db_id, source_authority_token),
        )
        lf_conn.execute(
            "INSERT INTO session_process_authority_events "
            "(session_id, session_generation, state_db_id, state_family, "
            "event_type, reservation_id, occurred_at) "
            "VALUES (?, 2, ?, 'sessiondb-v1', 'SESSION_ISSUED', ?, 1)",
            (source_session_id, source_state_db_id, source_reservation_id),
        )
        lf_conn.execute(
            "INSERT INTO session_process_reservations "
            "(reservation_id, reservation_token_sha256, session_id, "
            "session_generation, state_db_id, state_family, status, reserved_at, "
            "expires_at) VALUES (?, ?, ?, 2, ?, 'sessiondb-v1', 'RESERVED', 1, 2)",
            (
                source_reservation_id,
                source_reservation_token,
                source_session_id,
                source_state_db_id,
            ),
        )

        cells = ", ".join(f"c{index}" for index in range(len(session_columns)))
        lf_conn.execute(
            "CREATE TABLE lost_and_found "
            f"(rootpgno INTEGER, pgno INTEGER, nfield INTEGER, id INTEGER, {cells})"
        )
        source_values = {
            "id": source_session_id,
            "source": "telegram",
            "started_at": 1_754_000_000.0,
            "message_count": 7,
            "session_generation": 2,
            "billing_provider": "source-billing-provider",
            "billing_base_url": "https://source.invalid/api",
            "billing_mode": "source-billing-mode",
            "estimated_cost_usd": 12.5,
            "actual_cost_usd": 9.25,
            "cost_status": "source-cost-status",
            "cost_source": "source-cost-source",
            "pricing_version": "source-pricing-version",
            "title": "schema-less current-layout title",
            "title_source": "source-title",
            "last_activity_at": 1_754_000_001.0,
            "last_activity_description": "aligned later field",
            "last_activity_provenance": "source-provenance",
            "api_call_count": 11,
            "handoff_state": "source-handoff-state",
            "handoff_platform": "source-handoff-platform",
            "handoff_error": "source-handoff-error",
            "profile_name": "source-profile",
            "rewind_count": 3,
        }
        recovered_row = [source_values.get(column) for column in session_columns]
        assert recovered_row[generation_index] == 2
        placeholders = ", ".join("?" for _ in range(4 + len(recovered_row)))
        lf_conn.execute(
            f"INSERT INTO lost_and_found VALUES ({placeholders})",
            [2, 5, len(recovered_row), 1, *recovered_row],
        )

        # The real recovery path reopens a SessionDB destination and registers
        # the generation UDF before mapping page-level lost-and-found rows.
        register_turn_fence_generation(dest)
        mapping = map_lost_and_found_rows(lf_conn, dest)

        assert mapping["mapped"]["sessions"] == 1
        assert mapping["unmapped_rows"] == 0
        assert dest.execute(
            "SELECT session_generation FROM sessions WHERE id = ?",
            (source_session_id,),
        ).fetchone() == (1,)
        assert dest.execute(
            "SELECT billing_provider, billing_base_url, billing_mode, "
            "estimated_cost_usd, actual_cost_usd, cost_status, cost_source, "
            "pricing_version, title, title_source, last_activity_at, "
            "last_activity_description, last_activity_provenance, api_call_count, "
            "handoff_state, handoff_platform, handoff_error, profile_name, "
            "rewind_count FROM sessions WHERE id = ?",
            (source_session_id,),
        ).fetchone() == (
            "source-billing-provider",
            "https://source.invalid/api",
            "source-billing-mode",
            12.5,
            9.25,
            "source-cost-status",
            "source-cost-source",
            "source-pricing-version",
            "schema-less current-layout title",
            "source-title",
            1_754_000_001.0,
            "aligned later field",
            "source-provenance",
            11,
            "source-handoff-state",
            "source-handoff-platform",
            "source-handoff-error",
            "source-profile",
            3,
        )

        destination_state_db_id = dest.execute(
            "SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'"
        ).fetchone()[0]
        authority = dest.execute(
            "SELECT session_generation, state_db_id, state_family, authority_token, "
            "status FROM session_process_authorities WHERE session_id = ?",
            (source_session_id,),
        ).fetchone()
        assert authority[:3] == (1, destination_state_db_id, "sessiondb-v1")
        assert authority[3] != source_authority_token
        assert len(authority[3]) == 64
        assert authority[4] == "ISSUED"
        assert dest.execute(
            "SELECT session_generation, state_db_id, event_type, reservation_id "
            "FROM session_process_authority_events WHERE session_id = ?",
            (source_session_id,),
        ).fetchall() == [(1, destination_state_db_id, "SESSION_ISSUED", None)]
        assert dest.execute(
            "SELECT COUNT(*) FROM session_process_reservations WHERE session_id = ?",
            (source_session_id,),
        ).fetchone() == (0,)
    finally:
        lf_conn.close()
        dest.close()

    recovered_db = SessionDB(db_path=output)
    try:
        assert len(recovered_db.list_sessions_rich(limit=10)) == 1
    finally:
        recovered_db.close()


# Literal layouts from the immutable schema DDLs, deliberately independent of
# the current destination schema and production mapper layout declarations.
_REAL_52_SESSION_COLUMNS = (
    "id", "source", "user_id", "session_key", "chat_id", "chat_type",
    "thread_id", "display_name", "origin_json", "expiry_finalized", "model",
    "model_config", "system_prompt", "system_prompt_hash", "parent_session_id",
    "started_at", "ended_at", "end_reason", "message_count", "tool_call_count",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "cwd", "git_branch", "git_repo_root",
    "billing_provider", "billing_base_url", "billing_mode", "estimated_cost_usd",
    "actual_cost_usd", "cost_status", "cost_source", "pricing_version", "title",
    "last_activity_at", "last_activity_description", "last_activity_provenance",
    "api_call_count", "handoff_state", "handoff_platform", "handoff_error",
    "compression_failure_cooldown_until", "compression_failure_error",
    "compression_fallback_streak", "compression_ineffective_count", "profile_name",
    "rewind_count", "archived", "pinned",
)

_IMMEDIATE_PRE_V28_56_SESSION_COLUMNS = (
    "id", "source", "user_id", "session_key", "chat_id", "chat_type",
    "thread_id", "display_name", "origin_json", "expiry_finalized", "model",
    "model_config", "system_prompt", "system_prompt_hash", "parent_session_id",
    "started_at", "ended_at", "end_reason", "message_count", "tool_call_count",
    "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    "reasoning_tokens", "cwd", "git_branch", "git_repo_root",
    "git_metadata_generation", "billing_provider", "billing_base_url", "billing_mode",
    "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source",
    "pricing_version", "title", "title_source", "last_activity_at",
    "last_activity_description", "last_activity_provenance", "api_call_count",
    "handoff_state", "handoff_platform", "handoff_error",
    "compression_failure_cooldown_until", "compression_failure_error",
    "compression_fallback_streak", "compression_ineffective_count", "profile_name",
    "rewind_count", "archived", "pinned", "hidden", "last_read_at",
)


@pytest.mark.parametrize(
    ("source_columns", "layout_name"),
    [
        pytest.param(
            _REAL_52_SESSION_COLUMNS,
            "real-52",
            id="real-52-7d066c3c-layout",
        ),
        pytest.param(
            _IMMEDIATE_PRE_V28_56_SESSION_COLUMNS,
            "pre-v28-56",
            id="pre-v28-56-30a0f23-layout",
        ),
    ],
)
def test_mapper_schema_less_historical_session_layout_preserves_field_meanings(
    tmp_path: Path,
    source_columns: tuple[str, ...],
    layout_name: str,
) -> None:
    """Historical schema-less rows retain each source field's named meaning."""
    width = len(source_columns)
    assert width in {52, 56}
    assert "session_generation" not in source_columns
    if width == 52:
        assert "git_metadata_generation" not in source_columns
    else:
        assert source_columns[28] == "git_metadata_generation"

    source_session_id = f"20260901_1200{width:02d}_abc{width:03d}"
    source_authority_token = ("a" if width == 52 else "b") * 64
    source_reservation_token = ("c" if width == 52 else "d") * 64
    source_reservation_id = f"source-reservation-{layout_name}"
    source_values = {
        "id": source_session_id,
        "source": "telegram" if width == 52 else "discord",
        "user_id": f"source-user-{layout_name}",
        "session_key": f"source-key-{layout_name}",
        "chat_id": f"source-chat-{layout_name}",
        "chat_type": "group",
        "thread_id": f"source-thread-{layout_name}",
        "display_name": f"source-display-{layout_name}",
        "origin_json": f'{{"layout": "{layout_name}"}}',
        "expiry_finalized": 1,
        "model": f"source-model-{layout_name}",
        "model_config": f'{{"model": "{layout_name}"}}',
        "system_prompt": None,
        "system_prompt_hash": None,
        "parent_session_id": None,
        "started_at": 1_754_000_000.0 + width,
        "ended_at": 1_754_000_100.0 + width,
        "end_reason": f"source-end-{layout_name}",
        "message_count": width,
        "tool_call_count": width + 1,
        "input_tokens": width + 2,
        "output_tokens": width + 3,
        "cache_read_tokens": width + 4,
        "cache_write_tokens": width + 5,
        "reasoning_tokens": width + 6,
        "cwd": f"/source/{layout_name}",
        "git_branch": f"source-branch-{layout_name}",
        "git_repo_root": f"/source/repo-{layout_name}",
        "git_metadata_generation": width + 7,
        "billing_provider": f"source-provider-{layout_name}",
        "billing_base_url": f"https://{layout_name}.invalid/api",
        "billing_mode": f"source-billing-mode-{layout_name}",
        "estimated_cost_usd": width + 0.125,
        "actual_cost_usd": width + 0.25,
        "cost_status": f"source-cost-status-{layout_name}",
        "cost_source": f"source-cost-source-{layout_name}",
        "pricing_version": f"source-pricing-{layout_name}",
        "title": f"source-title-{layout_name}",
        "title_source": f"source-title-source-{layout_name}",
        "last_activity_at": 1_754_000_200.0 + width,
        "last_activity_description": f"source-activity-{layout_name}",
        "last_activity_provenance": f"source-provenance-{layout_name}",
        "api_call_count": width + 8,
        "handoff_state": f"source-handoff-state-{layout_name}",
        "handoff_platform": f"source-handoff-platform-{layout_name}",
        "handoff_error": f"source-handoff-error-{layout_name}",
        "compression_failure_cooldown_until": 1_754_000_300.0 + width,
        "compression_failure_error": f"source-compression-error-{layout_name}",
        "compression_fallback_streak": width + 9,
        "compression_ineffective_count": width + 10,
        "profile_name": f"source-profile-{layout_name}",
        "rewind_count": width + 11,
        "archived": 1 if width == 52 else 0,
        "pinned": 0 if width == 52 else 1,
        "hidden": 1,
        "last_read_at": 1_754_000_400.0 + width,
    }
    recovered_row = [source_values[column] for column in source_columns]

    lf_path = tmp_path / f"{layout_name}-lost-and-found.db"
    SessionDB(db_path=lf_path).close()
    output = tmp_path / f"{layout_name}-mapped.db"
    SessionDB(db_path=output).close()
    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        register_turn_fence_generation(lf_conn)
        source_state_db_id = lf_conn.execute(
            "SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'"
        ).fetchone()[0]
        lf_conn.execute(
            "INSERT INTO session_process_authorities "
            "(session_id, session_generation, state_db_id, state_family, "
            "authority_token, status, issued_at) VALUES (?, 2, ?, 'sessiondb-v1', "
            "?, 'ISSUED', 1)",
            (source_session_id, source_state_db_id, source_authority_token),
        )
        lf_conn.execute(
            "INSERT INTO session_process_authority_events "
            "(session_id, session_generation, state_db_id, state_family, "
            "event_type, reservation_id, occurred_at) "
            "VALUES (?, 2, ?, 'sessiondb-v1', 'SESSION_ISSUED', ?, 1)",
            (source_session_id, source_state_db_id, source_reservation_id),
        )
        lf_conn.execute(
            "INSERT INTO session_process_reservations "
            "(reservation_id, reservation_token_sha256, session_id, "
            "session_generation, state_db_id, state_family, status, reserved_at, "
            "expires_at) VALUES (?, ?, ?, 2, ?, 'sessiondb-v1', 'RESERVED', 1, 2)",
            (
                source_reservation_id,
                source_reservation_token,
                source_session_id,
                source_state_db_id,
            ),
        )
        cells = ", ".join(f"c{index}" for index in range(width))
        lf_conn.execute(
            "CREATE TABLE lost_and_found "
            f"(rootpgno INTEGER, pgno INTEGER, nfield INTEGER, id INTEGER, {cells})"
        )
        placeholders = ", ".join("?" for _ in range(4 + width))
        lf_conn.execute(
            f"INSERT INTO lost_and_found VALUES ({placeholders})",
            [2, 5, width, 1, *recovered_row],
        )

        register_turn_fence_generation(dest)
        mapping = map_lost_and_found_rows(lf_conn, dest)

        assert mapping["mapped"]["sessions"] == 1
        assert mapping["unmapped_rows"] == 0
        quoted = ", ".join(f'"{column}"' for column in source_columns)
        assert dest.execute(
            f"SELECT {quoted} FROM sessions WHERE id = ?", (source_session_id,)
        ).fetchone() == tuple(source_values[column] for column in source_columns)
        assert dest.execute(
            "SELECT session_generation FROM sessions WHERE id = ?",
            (source_session_id,),
        ).fetchone() == (1,)
        if width == 52:
            assert dest.execute(
                "SELECT git_metadata_generation, title_source, hidden, last_read_at "
                "FROM sessions WHERE id = ?",
                (source_session_id,),
            ).fetchone() == (0, None, 0, None)

        destination_state_db_id = dest.execute(
            "SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'"
        ).fetchone()[0]
        assert destination_state_db_id != source_state_db_id
        authority = dest.execute(
            "SELECT session_generation, state_db_id, state_family, authority_token, "
            "status FROM session_process_authorities WHERE session_id = ?",
            (source_session_id,),
        ).fetchone()
        assert authority[:3] == (1, destination_state_db_id, "sessiondb-v1")
        assert authority[3] != source_authority_token
        assert len(authority[3]) == 64
        assert authority[4] == "ISSUED"
        assert dest.execute(
            "SELECT session_generation, state_db_id, event_type, reservation_id "
            "FROM session_process_authority_events WHERE session_id = ?",
            (source_session_id,),
        ).fetchall() == [(1, destination_state_db_id, "SESSION_ISSUED", None)]
        assert dest.execute(
            "SELECT COUNT(*) FROM session_process_reservations WHERE session_id = ?",
            (source_session_id,),
        ).fetchone() == (0,)
    finally:
        lf_conn.close()
        dest.close()

    recovered_db = SessionDB(db_path=output)
    try:
        assert recovered_db._conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0] == 1
    finally:
        recovered_db.close()


def test_mapper_schema_less_pre_v28_56_with_appended_generation_preserves_named_fields(
    tmp_path: Path,
) -> None:
    """Generic ALTER appends generation after the immutable pre-v28 layout."""
    source_columns = (
        *_IMMEDIATE_PRE_V28_56_SESSION_COLUMNS,
        "session_generation",
    )
    assert len(source_columns) == 57
    assert source_columns[28] == "git_metadata_generation"
    assert source_columns[29] == "billing_provider"
    assert source_columns[-2:] == ("last_read_at", "session_generation")

    source_session_id = "20260901_120057_abc057"
    source_authority_token = "e" * 64
    source_reservation_token = "f" * 64
    source_reservation_id = "source-appended-generation-reservation"
    source_values = {
        "id": source_session_id,
        "source": "discord",
        "user_id": "appended-source-user",
        "session_key": "appended-source-key",
        "chat_id": "appended-source-chat",
        "chat_type": "group",
        "thread_id": "appended-source-thread",
        "display_name": "appended-source-display",
        "origin_json": '{"layout": "pre-v28-56-appended-generation"}',
        "expiry_finalized": 1,
        "model": "appended-source-model",
        "model_config": '{"model": "appended-source"}',
        "system_prompt": None,
        "system_prompt_hash": None,
        "parent_session_id": None,
        "started_at": 1_754_000_057.0,
        "ended_at": 1_754_000_157.0,
        "end_reason": "appended-source-end",
        "message_count": 57,
        "tool_call_count": 58,
        "input_tokens": 59,
        "output_tokens": 60,
        "cache_read_tokens": 61,
        "cache_write_tokens": 62,
        "reasoning_tokens": 63,
        "cwd": "/source/pre-v28-appended",
        "git_branch": "source-appended-branch",
        "git_repo_root": "/source/appended/repo",
        "git_metadata_generation": 64,
        "billing_provider": "appended-billing-provider",
        "billing_base_url": "https://appended.invalid/api",
        "billing_mode": "appended-billing-mode",
        "estimated_cost_usd": 57.125,
        "actual_cost_usd": 57.25,
        "cost_status": "appended-cost-status",
        "cost_source": "appended-cost-source",
        "pricing_version": "appended-pricing-version",
        "title": "appended-source-title",
        "title_source": "appended-source-title-source",
        "last_activity_at": 1_754_000_257.0,
        "last_activity_description": "appended-source-activity",
        "last_activity_provenance": "appended-source-provenance",
        "api_call_count": 65,
        "handoff_state": "appended-handoff-state",
        "handoff_platform": "appended-handoff-platform",
        "handoff_error": "appended-handoff-error",
        "compression_failure_cooldown_until": 1_754_000_357.0,
        "compression_failure_error": "appended-compression-error",
        "compression_fallback_streak": 66,
        "compression_ineffective_count": 67,
        "profile_name": "appended-source-profile",
        "rewind_count": 68,
        "archived": 1,
        "pinned": 0,
        "hidden": 1,
        "last_read_at": 1_754_000_457.0,
        # This is physical cell 56 because generic ALTER TABLE ADD COLUMN
        # appends it; it is not current-layout physical cell 29.
        "session_generation": 2,
    }
    recovered_row = [source_values[column] for column in source_columns]
    assert recovered_row[-1] == 2

    lf_path = tmp_path / "pre-v28-appended-generation-lost-and-found.db"
    SessionDB(db_path=lf_path).close()
    output = tmp_path / "pre-v28-appended-generation-mapped.db"
    SessionDB(db_path=output).close()
    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        register_turn_fence_generation(lf_conn)
        source_state_db_id = lf_conn.execute(
            "SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'"
        ).fetchone()[0]
        lf_conn.execute(
            "INSERT INTO session_process_authorities "
            "(session_id, session_generation, state_db_id, state_family, "
            "authority_token, status, issued_at) VALUES (?, 2, ?, 'sessiondb-v1', "
            "?, 'ISSUED', 1)",
            (source_session_id, source_state_db_id, source_authority_token),
        )
        lf_conn.execute(
            "INSERT INTO session_process_authority_events "
            "(session_id, session_generation, state_db_id, state_family, "
            "event_type, reservation_id, occurred_at) "
            "VALUES (?, 2, ?, 'sessiondb-v1', 'SESSION_ISSUED', ?, 1)",
            (source_session_id, source_state_db_id, source_reservation_id),
        )
        lf_conn.execute(
            "INSERT INTO session_process_reservations "
            "(reservation_id, reservation_token_sha256, session_id, "
            "session_generation, state_db_id, state_family, status, reserved_at, "
            "expires_at) VALUES (?, ?, ?, 2, ?, 'sessiondb-v1', 'RESERVED', 1, 2)",
            (
                source_reservation_id,
                source_reservation_token,
                source_session_id,
                source_state_db_id,
            ),
        )
        cells = ", ".join(f"c{index}" for index in range(len(recovered_row)))
        lf_conn.execute(
            "CREATE TABLE lost_and_found "
            f"(rootpgno INTEGER, pgno INTEGER, nfield INTEGER, id INTEGER, {cells})"
        )
        placeholders = ", ".join("?" for _ in range(4 + len(recovered_row)))
        lf_conn.execute(
            f"INSERT INTO lost_and_found VALUES ({placeholders})",
            [2, 5, len(recovered_row), 1, *recovered_row],
        )

        register_turn_fence_generation(dest)
        mapping = map_lost_and_found_rows(lf_conn, dest)

        assert mapping["mapped"]["sessions"] == 1
        assert mapping["unmapped_rows"] == 0
        assert dest.execute(
            "SELECT billing_provider, billing_base_url, billing_mode, title, "
            "title_source, last_activity_at, last_activity_description, "
            "last_activity_provenance, handoff_state, handoff_platform, "
            "handoff_error, profile_name, rewind_count, last_read_at "
            "FROM sessions WHERE id = ?",
            (source_session_id,),
        ).fetchone() == (
            "appended-billing-provider",
            "https://appended.invalid/api",
            "appended-billing-mode",
            "appended-source-title",
            "appended-source-title-source",
            1_754_000_257.0,
            "appended-source-activity",
            "appended-source-provenance",
            "appended-handoff-state",
            "appended-handoff-platform",
            "appended-handoff-error",
            "appended-source-profile",
            68,
            1_754_000_457.0,
        )
        # The trailing source generation is excluded instead of being mapped
        # to last_read_at, while the fresh destination mints generation 1.
        assert dest.execute(
            "SELECT session_generation, last_read_at FROM sessions WHERE id = ?",
            (source_session_id,),
        ).fetchone() == (1, 1_754_000_457.0)

        destination_state_db_id = dest.execute(
            "SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'"
        ).fetchone()[0]
        assert destination_state_db_id != source_state_db_id
        authority = dest.execute(
            "SELECT session_generation, state_db_id, state_family, authority_token, "
            "status FROM session_process_authorities WHERE session_id = ?",
            (source_session_id,),
        ).fetchone()
        assert authority[:3] == (1, destination_state_db_id, "sessiondb-v1")
        assert authority[3] != source_authority_token
        assert len(authority[3]) == 64
        assert authority[4] == "ISSUED"
        assert dest.execute(
            "SELECT session_generation, state_db_id, event_type, reservation_id "
            "FROM session_process_authority_events WHERE session_id = ?",
            (source_session_id,),
        ).fetchall() == [(1, destination_state_db_id, "SESSION_ISSUED", None)]
        assert dest.execute(
            "SELECT COUNT(*) FROM session_process_reservations WHERE session_id = ?",
            (source_session_id,),
        ).fetchone() == (0,)
    finally:
        lf_conn.close()
        dest.close()


def test_mapper_schema_less_ambiguous_57_layout_refuses_without_partial_insert(
    tmp_path: Path,
) -> None:
    """A row whose two physical generation slots both validate is unmapped."""
    source_session_id = "20260901_120058_abc058"
    lf_path = tmp_path / "ambiguous-57-lost-and-found.db"
    SessionDB(db_path=lf_path).close()
    output = tmp_path / "ambiguous-57-mapped.db"
    SessionDB(db_path=output).close()
    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        session_columns = [
            str(row[1]) for row in dest.execute("PRAGMA table_info(sessions)")
        ]
        assert len(session_columns) == 57
        assert session_columns[29] == "session_generation"
        assert session_columns[-1] == "last_read_at"
        cells = ", ".join(f"c{index}" for index in range(len(session_columns)))
        lf_conn.execute(
            "CREATE TABLE lost_and_found "
            f"(rootpgno INTEGER, pgno INTEGER, nfield INTEGER, id INTEGER, {cells})"
        )
        recovered_row = [None] * len(session_columns)
        recovered_row[0] = source_session_id
        recovered_row[1] = "cli"
        recovered_row[15] = 1_754_000_058.0
        recovered_row[29] = 2
        # A damaged current-layout last_read_at can resemble the appended
        # historical generation slot. Both candidates must therefore be
        # rejected rather than silently preferring current order.
        recovered_row[-1] = 2
        placeholders = ", ".join("?" for _ in range(4 + len(recovered_row)))
        lf_conn.execute(
            f"INSERT INTO lost_and_found VALUES ({placeholders})",
            [2, 5, len(recovered_row), 1, *recovered_row],
        )

        register_turn_fence_generation(dest)
        mapping = map_lost_and_found_rows(lf_conn, dest)

        assert mapping["mapped"]["sessions"] == 0
        assert mapping["unmapped_rows"] == 1
        assert dest.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
        assert dest.execute(
            "SELECT COUNT(*) FROM session_process_authorities WHERE session_id = ?",
            (source_session_id,),
        ).fetchone() == (0,)
    finally:
        lf_conn.close()
        dest.close()


def test_mapper_schema_less_unknown_or_short_57_layouts_refuse_without_insert(
    tmp_path: Path,
) -> None:
    """Neither an unproven 57 layout nor a truncated one reaches INSERT."""
    unknown_session_id = "20260901_120059_abc059"
    short_session_id = "20260901_120100_abc100"
    lf_path = tmp_path / "unknown-short-57-lost-and-found.db"
    SessionDB(db_path=lf_path).close()
    output = tmp_path / "unknown-short-57-mapped.db"
    SessionDB(db_path=output).close()
    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        session_columns = [
            str(row[1]) for row in dest.execute("PRAGMA table_info(sessions)")
        ]
        assert len(session_columns) == 57
        full_cells = ", ".join(f"c{index}" for index in range(57))
        short_cells = ", ".join(f"c{index}" for index in range(56))
        lf_conn.execute(
            "CREATE TABLE lost_and_found_unknown "
            f"(rootpgno INTEGER, pgno INTEGER, nfield INTEGER, id INTEGER, {full_cells})"
        )
        lf_conn.execute(
            "CREATE TABLE lost_and_found_short "
            f"(rootpgno INTEGER, pgno INTEGER, nfield INTEGER, id INTEGER, {short_cells})"
        )
        unknown_row = [None] * 57
        unknown_row[0] = unknown_session_id
        unknown_row[1] = "cli"
        # Neither physical generation cell is a positive SQLite INTEGER.
        unknown_row[29] = "not-a-generation"
        unknown_row[-1] = None
        short_row = [None] * 56
        short_row[0] = short_session_id
        short_row[1] = "cli"
        short_row[29] = 2
        placeholders = ", ".join("?" for _ in range(4 + 57))
        lf_conn.execute(
            f"INSERT INTO lost_and_found_unknown VALUES ({placeholders})",
            [2, 5, 57, 1, *unknown_row],
        )
        placeholders = ", ".join("?" for _ in range(4 + 56))
        lf_conn.execute(
            f"INSERT INTO lost_and_found_short VALUES ({placeholders})",
            [2, 5, 57, 2, *short_row],
        )

        register_turn_fence_generation(dest)
        mapping = map_lost_and_found_rows(lf_conn, dest)

        assert mapping["mapped"]["sessions"] == 0
        assert mapping["unmapped_rows"] == 2
        assert dest.execute("SELECT COUNT(*) FROM sessions").fetchone() == (0,)
    finally:
        lf_conn.close()
        dest.close()


# ── issue #72291: source-fingerprint error must name the parent CLI ─────────


def test_fingerprint_error_enumerates_parent_cli_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "busy.db"
    SessionDB(db_path=source).close()

    fingerprints = iter([{"main": {"size": 1, "mtime_ns": 1}},
                         {"main": {"size": 2, "mtime_ns": 2}},
                         {"main": {"size": 3, "mtime_ns": 3}}])
    monkeypatch.setattr(
        session_recovery,
        "_source_fingerprint",
        lambda _source: next(fingerprints),
    )
    with pytest.raises(SessionRecoverySafetyError) as excinfo:
        session_recovery.inspect_session_database(source, work_dir=tmp_path)
    message = str(excinfo.value)
    assert "Stop every Hermes process" in message
    # The gap from #72291: the parent CLI session itself must be enumerated.
    assert "CLI session" in message
    assert "fresh shell" in message
    assert "snapshot" in message
