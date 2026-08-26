from __future__ import annotations

import hashlib
import json
import os
import select
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import (
    FTS_STORAGE_VERSION,
    SCHEMA_VERSION,
    SQLiteSerializationError,
    SessionDB,
)
from hermes_state_common import (
    TURN_FENCE_FUNCTION_NAME,
    TURN_FENCE_GENERATION,
)
from hermes_cli import session_recovery
from hermes_cli.session_recovery import (
    SessionRecoverySafetyError,
    SessionRecoverySourceError,
    inspect_session_database,
    recover_session_database,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_source(path: Path) -> dict[str, int]:
    db = SessionDB(db_path=path)
    try:
        for session_number in range(3):
            session_id = f"recovery-session-{session_number}"
            db.create_session(
                session_id,
                "cli",
                cwd=f"/tmp/recovery-{session_number}",
            )
            db.set_session_title(session_id, f"Recovery {session_number}")
            for message_number in range(7):
                db.append_message(
                    session_id,
                    "user" if message_number % 2 == 0 else "assistant",
                    f"recoverable payload {session_number} {message_number}",
                )

        db.set_meta("goal:recovery-session-0", '{"status":"active"}')
        db.apply_telegram_topic_migration()
        db._conn.execute(
            """
            INSERT INTO telegram_dm_topic_mode (
                chat_id, user_id, enabled, activated_at, updated_at
            ) VALUES (?, ?, 1, ?, ?)
            """,
            ("chat-1", "user-1", 1.0, 2.0),
        )
        db._conn.execute(
            """
            INSERT INTO telegram_dm_topic_bindings (
                chat_id, thread_id, user_id, session_key, session_id,
                managed_mode, linked_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chat-1",
                "thread-1",
                "user-1",
                "telegram:user-1:chat-1",
                "recovery-session-0",
                "auto",
                1.0,
                2.0,
            ),
        )
        db._conn.execute(
            """
            INSERT INTO gateway_routing (
                scope, session_key, entry_json, updated_at
            ) VALUES (?, ?, ?, ?)
            """,
            ("telegram", "telegram:user-1:chat-1", "{}", 2.0),
        )
        db._conn.execute(
            """
            INSERT INTO async_delegations (
                delegation_id, origin_session, state, dispatched_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("delegation-1", "recovery-session-0", "completed", 1.0, 2.0),
        )
        # These are derived transition markers and must not reach the new DB.
        db.set_meta("fts_rebuild_high_water", "999")
        db.set_meta("fts_rebuild_progress", "500")
    finally:
        db.close()
    return {"sessions": 3, "messages": 21}


def _orphan_fts_schema(path: Path) -> None:
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "DELETE FROM sqlite_master "
            "WHERE type='table' "
            "AND name IN ('messages_fts', 'messages_fts_trigram')"
        )
        conn.execute("PRAGMA writable_schema=OFF")
    finally:
        conn.close()
def _make_page_spanning_source(
    path: Path,
    message_count: int = 320,
) -> tuple[int, int | None]:
    db = SessionDB(db_path=path)
    try:
        db.create_session(
            "partial-recovery-session",
            "cli",
            cwd="/tmp/partial-recovery",
        )
        for message_number in range(message_count):
            db.append_message(
                "partial-recovery-session",
                "user" if message_number % 2 == 0 else "assistant",
                (
                    f"partial recovery payload {message_number:04d} "
                    + chr(65 + message_number % 26) * 1_500
                ),
            )
    finally:
        db.close()

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
        plan = " ".join(
            str(row[3])
            for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT COUNT(*) FROM messages"
            ).fetchall()
        )
        count_index = next(
            (
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'index' AND tbl_name = 'messages'"
                ).fetchall()
                if plan.endswith(str(row[0]))
            ),
            None,
        )
        names = ["messages"]
        if count_index is not None:
            names.append(count_index)
        placeholders = ", ".join("?" for _ in names)
        roots = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT name, rootpage FROM sqlite_master "
                f"WHERE name IN ({placeholders})",
                tuple(names),
            ).fetchall()
        }
        return roots["messages"], (
            roots[count_index] if count_index is not None else None
        )
    finally:
        conn.close()


def _make_many_sessions_source(
    path: Path,
    session_count: int = 180,
) -> int:
    db = SessionDB(db_path=path)
    try:
        for session_number in range(session_count):
            session_id = f"partial-session-{session_number:04d}"
            db.create_session(
                session_id,
                "cli",
                cwd=f"/tmp/partial-session-{session_number:04d}",
                system_prompt=(
                    f"session payload {session_number:04d} "
                    + chr(65 + session_number % 26) * 1_500
                ),
            )
            db.append_message(session_id, "user", f"message {session_number}")
    finally:
        db.close()

    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
        row = conn.execute(
            "SELECT rootpage FROM sqlite_master "
            "WHERE type = 'table' AND name = 'sessions'"
        ).fetchone()
        assert row is not None
        return int(row[0])
    finally:
        conn.close()


def _btree_leaf_pages(path: Path, root_page: int) -> tuple[int, list[int]]:
    data = path.read_bytes()
    page_size = int.from_bytes(data[16:18], "big")
    if page_size == 1:
        page_size = 65_536
    leaf_pages: list[int] = []
    visited: set[int] = set()

    def visit(page_number: int) -> None:
        if page_number in visited:
            return
        visited.add(page_number)
        page_start = (page_number - 1) * page_size
        header_offset = page_start + (100 if page_number == 1 else 0)
        page_type = data[header_offset]
        cell_count = int.from_bytes(
            data[header_offset + 3 : header_offset + 5],
            "big",
        )
        if page_type in {0x0A, 0x0D}:
            leaf_pages.append(page_number)
            return
        assert page_type in {0x02, 0x05}, (
            f"unexpected table b-tree page type {page_type:#x} "
            f"on page {page_number}"
        )

        pointer_array = header_offset + 12
        for cell_number in range(cell_count):
            pointer_offset = pointer_array + cell_number * 2
            cell_offset = int.from_bytes(
                data[pointer_offset : pointer_offset + 2],
                "big",
            )
            child_offset = page_start + cell_offset
            child_page = int.from_bytes(
                data[child_offset : child_offset + 4],
                "big",
            )
            visit(child_page)
        rightmost_page = int.from_bytes(
            data[header_offset + 8 : header_offset + 12],
            "big",
        )
        visit(rightmost_page)

    visit(root_page)
    return page_size, leaf_pages


def _corrupt_middle_table_leaf(
    path: Path,
    root_page: int,
    *,
    require_interior: bool = True,
) -> int:
    page_size, leaf_pages = _btree_leaf_pages(path, root_page)
    assert leaf_pages
    if require_interior:
        assert len(leaf_pages) >= 3
    leaf_page = leaf_pages[len(leaf_pages) // 2]
    page_start = (leaf_page - 1) * page_size
    header_offset = page_start + (100 if leaf_page == 1 else 0)

    data = bytearray(path.read_bytes())
    assert data[header_offset] in {0x0A, 0x0D}
    # An impossible cell count damages this one middle leaf while preserving
    # the table root and leaves on both sides. This is a physical SQLite page
    # failure, not a mocked cursor exception.
    data[header_offset + 3 : header_offset + 5] = b"\xff\xff"
    path.write_bytes(data)
    return leaf_page


def _corrupt_table_root(path: Path, root_page: int) -> None:
    data = bytearray(path.read_bytes())
    page_size = int.from_bytes(data[16:18], "big")
    if page_size == 1:
        page_size = 65_536
    page_start = (root_page - 1) * page_size
    header_offset = page_start + (100 if root_page == 1 else 0)
    assert data[header_offset] in {0x02, 0x05, 0x0A, 0x0D}
    # Damage the root enough that no rowid bounds can be read. This reproduces
    # a fully failed sessions copy while leaving the messages b-tree intact.
    data[header_offset + 3 : header_offset + 5] = b"\xff\xff"
    path.write_bytes(data)


def test_snapshot_blocks_connections_opened_during_the_copy(
    tmp_path: Path,
) -> None:
    """A connection must not be able to open while raw copy descriptors exist.

    Checking has_live_connection() and then copying leaves a window: a
    connection can open between the two, and the copy's close() cancels its
    POSIX advisory locks. The guard must hold the lifecycle lock across the
    whole bundle copy.

    Runs the copy in a worker thread and pauses it inside the patched copy, so
    the assertion is about lock ordering rather than which thread the
    scheduler happens to resume first: while the copy is parked, a
    connect_tracked() attempt must NOT complete; once released, it must.
    """
    import threading

    from hermes_cli import session_recovery as recovery_module
    from hermes_cli.sqlite_safe_read import connect_tracked

    source = tmp_path / "racy-state.db"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    _make_source(source)

    inside_copy = threading.Event()
    release_copy = threading.Event()
    connect_attempted = threading.Event()
    connection_opened = threading.Event()
    errors: list[str] = []
    real_copy2 = recovery_module.shutil.copy2

    def slow_copy2(src, dst, *args, **kwargs):
        result = real_copy2(src, dst, *args, **kwargs)
        if str(src).endswith("racy-state.db"):
            inside_copy.set()
            release_copy.wait(30)
        return result

    def do_copy():
        try:
            recovery_module._copy_source_bundle(source, snapshot_dir)
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(f"copy failed: {exc}")

    def do_connect():
        # Signal immediately before the blocking call so a timed "still
        # blocked" assertion cannot pass merely because this thread had not
        # been scheduled yet.
        connect_attempted.set()
        try:
            conn = connect_tracked(source, isolation_level=None, timeout=30.0)
            connection_opened.set()
            conn.close()
        except Exception as exc:  # pragma: no cover - surfaced via errors
            errors.append(f"connect failed: {exc}")

    recovery_module.shutil.copy2 = slow_copy2
    copier = threading.Thread(target=do_copy, daemon=True)
    connector = threading.Thread(target=do_connect, daemon=True)
    try:
        copier.start()
        assert inside_copy.wait(30), "copy never reached the patched operation"

        connector.start()
        assert connect_attempted.wait(30), "connector thread never started"
        # The connector is at the lock. While the copy holds it, the
        # connection must not open.
        assert not connection_opened.wait(1.0), (
            "connect_tracked() completed while raw copy descriptors were open "
            "— the guard is not holding the lifecycle lock across the copy"
        )

        release_copy.set()
        # Once the copy finishes and releases the lock, it must open promptly.
        assert connection_opened.wait(30), (
            "connect_tracked() never completed after the copy released the lock"
        )
    finally:
        release_copy.set()
        recovery_module.shutil.copy2 = real_copy2
        copier.join(30)
        connector.join(30)

    assert not errors, errors[0]


def test_partial_recovery_keeps_messages_when_sessions_are_unsalvageable(
    tmp_path: Path,
) -> None:
    """Salvaged messages must survive even when NO session row is recoverable.

    Reported July 2026: a user's recovery copied 20,817 of 20,824 messages,
    then orphan cleanup deleted every one of them because the sessions b-tree
    was damaged worse than the messages b-tree. The output had 0 sessions and
    0 messages — the salvage worked and then threw the result away, which is
    the exact opposite of what --allow-partial is for.

    Messages must be retained under reconstructed placeholder sessions, and
    the placeholder-ness must be reported as loss rather than passed off as a
    clean recovery.
    """
    source = tmp_path / "sessions-destroyed.db"
    output = tmp_path / "sessions-destroyed-recovered.db"

    messages_per_session = {
        "doomed-session-a": 40,
        "doomed-session-b": 35,
        "doomed-session-c": 45,
    }
    db = SessionDB(db_path=source)
    try:
        for session_id, message_count in messages_per_session.items():
            db.create_session(session_id, "cli", cwd=f"/tmp/{session_id}")
            for index in range(message_count):
                db.append_message(
                    session_id,
                    "user",
                    f"irreplaceable {session_id} {index}",
                )
    finally:
        db.close()

    # sessions unrecoverable, messages intact — the reported shape.
    # The marker is minted HERE, in the fixture, because the damage this test
    # needs is not reachable through any SessionDB method: every one of them
    # cascades the transcript away with its owner, which is the opposite of
    # the reported failure. A test may mint it; a production writer may not,
    # and that difference is what the writer census exists to hold.
    conn = sqlite3.connect(str(source), isolation_level=None)
    conn.create_function(
        TURN_FENCE_FUNCTION_NAME, 0, lambda: TURN_FENCE_GENERATION
    )
    try:
        conn.execute("DELETE FROM sessions")
    finally:
        conn.close()

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=16,
        allow_partial=True,
    )

    cleanup = report["orphan_cleanup"]
    assert cleanup["messages_removed"] == 0, (
        "salvaged messages were deleted for lack of a session row"
    )
    assert cleanup["sessions_reconstructed"] == len(messages_per_session)
    assert cleanup["messages_retained"] == 120

    with sqlite3.connect(str(output)) as verify:
        recovered_sessions = verify.execute(
            "SELECT id, source, title, message_count FROM sessions ORDER BY id"
        ).fetchall()
        messages = verify.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert messages == 120, f"expected all 120 messages retained, got {messages}"
    assert len(recovered_sessions) == len(messages_per_session)

    # Fabricated sessions must be identifiable and carry collision-safe titles.
    assert {row[0] for row in recovered_sessions} == set(messages_per_session)
    assert {row[1] for row in recovered_sessions} == {"recovered"}
    recovered_titles = [str(row[2]) for row in recovered_sessions]
    assert all(title.startswith("[recovered ") for title in recovered_titles)
    assert len(set(recovered_titles)) == len(recovered_titles)
    assert {
        str(row[0]): int(row[3]) for row in recovered_sessions
    } == messages_per_session

    # Retaining the data is still a lossy outcome and must say so.
    assert report["verification"]["loss_detected"] is True
    assert report["partial"] is True
    assert report["complete"] is False
    assert any(
        "reconstructed as placeholders" in warning
        for warning in report["verification"]["warnings"]
    ), report["verification"]["warnings"]

    # The output must remain structurally sound.
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["foreign_key_check"] == []
    assert report["verified"] is True
    assert report["installed"] is False










def test_cli_allow_partial_salvages_rows_across_a_corrupt_leaf(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corrupt-state.db"
    rejected_output = tmp_path / "rejected.db"
    output = tmp_path / "partial-recovered.db"
    message_count = 320
    messages_root, count_index_root = _make_page_spanning_source(
        source,
        message_count,
    )
    corrupt_page = _corrupt_middle_table_leaf(source, messages_root)
    if count_index_root is not None:
        _corrupt_middle_table_leaf(
            source,
            count_index_root,
            require_interior=False,
        )
    source_hash = _sha256(source)

    inspection = inspect_session_database(source, work_dir=tmp_path)
    assert inspection["recoverable"] is False
    assert inspection["tables"]["messages"]["rows"] is None
    with pytest.raises(SessionRecoverySourceError, match="messages"):
        recover_session_database(
            source,
            rejected_output,
            work_dir=tmp_path,
        )
    assert not rejected_output.exists()

    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "isolated-hermes-home")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "sessions",
            "recover",
            "--source",
            str(source),
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path),
            "--chunk-size",
            "8",
            "--allow-partial",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Partial recovery output verified" in result.stdout
    assert "active session database was not changed" in result.stdout
    assert _sha256(source) == source_hash

    report_path = output.with_name(output.name + ".recovery.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["allow_partial"] is True
    assert report["verified"] is True
    assert report["complete"] is False
    assert report["partial"] is True
    assert report["installed"] is False
    assert report["source_unchanged"] is True
    assert report["verification"]["healthy"] is True
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verification"]["foreign_key_check"] == []
    assert report["verification"]["table_counts"]["sessions"] == 1

    copied_messages = report["copy"]["messages"]
    assert copied_messages["status"] == "partial"
    assert copied_messages["copied_rows"] < message_count
    assert copied_messages["copied_rows"] > 0
    assert copied_messages["skipped_rowid_ranges"]
    assert any(
        item["low"] <= message_count and item["high"] >= 1
        for item in copied_messages["skipped_rowid_ranges"]
    )
    assert copied_messages["query_limit_reached"] is False

    conn = sqlite3.connect(str(output))
    try:
        recovered_ids = {
            int(row[0]) for row in conn.execute("SELECT id FROM messages")
        }
        assert 1 in recovered_ids
        assert message_count in recovered_ids
        assert len(recovered_ids) == copied_messages["copied_rows"]
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        conn.close()

    # Prove the helper damaged an interior data leaf, so successful recovery of
    # the first and last message IDs really crossed the corrupted region.
    assert corrupt_page not in {
        min(_btree_leaf_pages(source, messages_root)[1]),
        max(_btree_leaf_pages(source, messages_root)[1]),
    }


def test_partial_recovery_clears_only_unreadable_system_prompt_refs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "corrupt-system-prompts.db"
    output = tmp_path / "partial-system-prompts.db"
    session_count = 180
    _make_many_sessions_source(source, session_count)

    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        row = conn.execute(
            "SELECT rootpage FROM sqlite_master "
            "WHERE type = 'table' AND name = 'system_prompts'"
        ).fetchone()
        assert row is not None
        prompt_root = int(row[0])
    finally:
        conn.close()
    _corrupt_middle_table_leaf(source, prompt_root)

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=8,
        allow_partial=True,
    )

    assert report["verified"] is True
    assert report["partial"] is True
    assert report["copy"]["sessions"]["status"] == "complete"
    assert report["copy"]["messages"]["status"] == "complete"
    assert report["copy"]["system_prompts"]["status"] == "partial"
    cleared = report["orphan_cleanup"]["session_prompt_refs_cleared"]
    assert 0 < cleared < session_count
    assert report["verification"]["foreign_key_check"] == []

    conn = sqlite3.connect(str(output))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == session_count
        retained = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE system_prompt_hash IS NOT NULL"
        ).fetchone()[0]
        assert retained == session_count - cleared
        assert (
            conn.execute("SELECT COUNT(*) FROM system_prompts").fetchone()[0]
            == retained
        )
    finally:
        conn.close()


_OFFLINE_REBUILD_EPOCH_KEY = "_hermes_offline_rebuild_epoch_v1"
_E2_BARRIER_TIMEOUT_SECONDS = 15


def _read_offline_rebuild_marker(path: Path) -> str | None:
    with sqlite3.connect(str(path)) as conn:
        row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (_OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone()
    return None if row is None else str(row[0])


def _make_stale_source_rebuild_marker(path: Path) -> str:
    root = Path(__file__).resolve().parents[2]
    program = """
import os
import pathlib
import sys

from hermes_state import SessionDB

marker_key = "_hermes_offline_rebuild_epoch_v1"
db = SessionDB(db_path=pathlib.Path(sys.argv[1]))
with db.offline_rebuild(reason="e2 stale recovery source fixture"):
    marker = db.get_meta(marker_key)
    assert marker is not None
    print(marker, flush=True)
    assert sys.stdin.readline() == "RELEASE\\n"
    os._exit(0)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(path)],
        cwd=str(root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdin is not None
    assert child.stdout is not None
    assert child.stderr is not None
    ready, _, _ = select.select(
        [child.stdout], [], [], _E2_BARRIER_TIMEOUT_SECONDS
    )
    assert ready, "stale-marker child did not publish its live marker"
    marker = child.stdout.readline().strip()
    assert marker
    assert child.poll() is None
    stdout, stderr = child.communicate(
        "RELEASE\n", timeout=_E2_BARRIER_TIMEOUT_SECONDS
    )
    assert child.returncode == 0, stdout + stderr
    assert _read_offline_rebuild_marker(path) == marker
    return marker


class _RecoveryBarrier:
    def __init__(self) -> None:
        self._arrived = {
            "canonical": threading.Event(),
            "state_meta": threading.Event(),
        }
        self._released = {
            "canonical": threading.Event(),
            "state_meta": threading.Event(),
        }
        self.events: dict[str, dict[str, object]] = {}

    def __call__(self, event: dict[str, object]) -> None:
        table = str(event.get("table", ""))
        phase = None
        if table in session_recovery._CANONICAL_TABLES:
            phase = "canonical"
        elif table == "state_meta":
            phase = "state_meta"
        if phase is None or self._arrived[phase].is_set():
            return
        self.events[phase] = dict(event)
        self._arrived[phase].set()
        assert self._released[phase].wait(_E2_BARRIER_TIMEOUT_SECONDS), (
            f"recovery {phase} barrier was not released"
        )

    def wait(self, phase: str) -> None:
        assert self._arrived[phase].wait(_E2_BARRIER_TIMEOUT_SECONDS), (
            f"recovery did not reach the {phase} barrier"
        )

    def release(self, phase: str) -> None:
        self._released[phase].set()

    def release_all(self) -> None:
        for event in self._released.values():
            event.set()


def _run_recovery_contender_child(path: Path) -> tuple[subprocess.Popen[str], dict[str, str]]:
    root = Path(__file__).resolve().parents[2]
    program = """
import json
import pathlib
import sys

from hermes_state import SessionDB, SessionTurnLeaseLostError

db = SessionDB(db_path=pathlib.Path(sys.argv[1]))
try:
    token = db.try_acquire_session_turn_lease(
        "e2-source-session",
        "e2-contender",
        ttl_seconds=30.0,
    )
    lease = "REFUSED" if token is None else "ACQUIRED"
    try:
        db.create_session("e2-child-created", "child")
    except SessionTurnLeaseLostError:
        write = "REFUSED"
    else:
        write = "COMMITTED"
    print(json.dumps({"lease": lease, "write": write}, sort_keys=True), flush=True)
    assert sys.stdin.readline() == "RELEASE\\n"
finally:
    db.close()
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(path)],
        cwd=str(root),
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdin is not None
    assert child.stdout is not None
    assert child.stderr is not None
    ready, _, _ = select.select(
        [child.stdout], [], [], _E2_BARRIER_TIMEOUT_SECONDS
    )
    assert ready, "recovery contender did not report its outcome"
    line = child.stdout.readline()
    assert line, "recovery contender exited without an outcome"
    outcome = json.loads(line)
    assert set(outcome) == {"lease", "write"}
    assert child.poll() is None
    return child, outcome


@pytest.mark.parametrize("copy_mode", ["normal", "salvage"])
def test_state_meta_copy_preserves_destination_offline_rebuild_epoch_and_user_rows(
    tmp_path: Path,
    copy_mode: str,
) -> None:
    source = tmp_path / f"e2-helper-{copy_mode}-source.db"
    destination = tmp_path / f"e2-helper-{copy_mode}-destination.db"
    source_db = SessionDB(db_path=source)
    try:
        source_db.set_meta("e2:user", "source-user-value")
        source_db.set_meta("e2:ordinary", "source-ordinary-value")
    finally:
        source_db.close()
    source_marker = _make_stale_source_rebuild_marker(source)

    source_conn = sqlite3.connect(str(source), isolation_level=None)
    destination_db = SessionDB(db_path=destination)
    destination_db.set_meta("e2:user", "destination-before")
    marker_observations: list[str | None] = []
    try:
        with destination_db.offline_rebuild(reason=f"e2 helper {copy_mode}"):
            destination_marker = _read_offline_rebuild_marker(destination)
            assert destination_marker is not None
            assert destination_marker != source_marker

            def observe_progress(event: dict[str, object]) -> None:
                if event.get("table") != "state_meta":
                    return
                marker_observations.append(
                    _read_offline_rebuild_marker(destination)
                )
                assert marker_observations[-1] == destination_marker

            copy_function = (
                session_recovery._copy_state_meta_salvage
                if copy_mode == "salvage"
                else session_recovery._copy_state_meta
            )
            source_rows = int(
                source_conn.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0]
            )
            report = copy_function(
                source_conn,
                destination_db,
                chunk_size=1,
                progress_cb=observe_progress,
                source_rows=source_rows,
            )

            assert marker_observations
            assert _read_offline_rebuild_marker(destination) == destination_marker
            assert _OFFLINE_REBUILD_EPOCH_KEY in report["excluded_keys"]
            placeholders = ", ".join("?" for _ in report["excluded_keys"])
            expected_copied = int(
                source_conn.execute(
                    f"SELECT COUNT(*) FROM state_meta WHERE key NOT IN ({placeholders})",
                    tuple(report["excluded_keys"]),
                ).fetchone()[0]
            )
            assert report["copied_rows"] == expected_copied
            assert report["status"] == "complete"
            assert destination_db.get_meta("e2:user") == "source-user-value"
            assert destination_db.get_meta("e2:ordinary") == "source-ordinary-value"
            assert source_marker not in json.dumps(report, sort_keys=True)

        assert _read_offline_rebuild_marker(destination) is None
    finally:
        source_conn.close()
        destination_db.close()


@pytest.mark.parametrize("copy_mode", ["normal", "salvage"])
def test_recover_session_database_preserves_destination_offline_rebuild_epoch_and_blocks_child(
    tmp_path: Path,
    copy_mode: str,
) -> None:
    source = tmp_path / f"e2-caller-{copy_mode}-source.db"
    output = tmp_path / f"e2-caller-{copy_mode}-output.db"
    source_db = SessionDB(db_path=source)
    try:
        source_db.create_session("e2-source-session", "cli")
        source_db.set_session_title("e2-source-session", "E2 copied session")
        source_db.set_meta("e2:user", "source-user-value")
    finally:
        source_db.close()
    source_marker = _make_stale_source_rebuild_marker(source)

    barrier = _RecoveryBarrier()
    reports: list[dict[str, object]] = []
    recovery_errors: list[BaseException] = []
    recovery_done = threading.Event()

    def run_recovery() -> None:
        try:
            reports.append(
                recover_session_database(
                    source,
                    output,
                    work_dir=tmp_path,
                    chunk_size=1,
                    progress_cb=barrier,
                    allow_partial=copy_mode == "salvage",
                )
            )
        except BaseException as exc:
            recovery_errors.append(exc)
        finally:
            recovery_done.set()

    worker = threading.Thread(target=run_recovery, daemon=True)
    contender = None
    worker.start()
    try:
        barrier.wait("canonical")
        destination_marker = _read_offline_rebuild_marker(output)
        assert destination_marker is not None
        assert destination_marker != source_marker
        barrier.release("canonical")

        barrier.wait("state_meta")
        assert _read_offline_rebuild_marker(output) == destination_marker
        contender, outcome = _run_recovery_contender_child(output)
        assert outcome == {"lease": "REFUSED", "write": "REFUSED"}
        with sqlite3.connect(str(output)) as verifier:
            lease_rows = int(
                verifier.execute(
                    "SELECT COUNT(*) FROM session_turn_leases "
                    "WHERE conversation_id = 'e2-source-session'"
                ).fetchone()[0]
            )
            child_rows = int(
                verifier.execute(
                    "SELECT COUNT(*) FROM sessions WHERE id = 'e2-child-created'"
                ).fetchone()[0]
            )
        assert lease_rows == 0
        assert child_rows == 0
        assert contender.poll() is None
        assert _read_offline_rebuild_marker(output) == destination_marker

        barrier.release("state_meta")
        assert recovery_done.wait(_E2_BARRIER_TIMEOUT_SECONDS), (
            "recovery did not finish after the state_meta barrier"
        )
        worker.join(_E2_BARRIER_TIMEOUT_SECONDS)
        assert not worker.is_alive()
        assert not recovery_errors, repr(recovery_errors[0]) if recovery_errors else ""
        assert len(reports) == 1
        report = reports[0]
        assert _read_offline_rebuild_marker(output) is None
        assert report["copy"]["state_meta"]["status"] == "complete"
        assert _OFFLINE_REBUILD_EPOCH_KEY in report["copy"]["state_meta"][
            "excluded_keys"
        ]
        assert source_marker not in json.dumps(report, sort_keys=True)
        with sqlite3.connect(str(output)) as verifier:
            copied = verifier.execute(
                "SELECT title FROM sessions WHERE id = 'e2-source-session'"
            ).fetchone()
            user_meta = verifier.execute(
                "SELECT value FROM state_meta WHERE key = 'e2:user'"
            ).fetchone()
            child_rows = int(
                verifier.execute(
                    "SELECT COUNT(*) FROM sessions WHERE id = 'e2-child-created'"
                ).fetchone()[0]
            )
        assert copied == ("E2 copied session",)
        assert user_meta == ("source-user-value",)
        assert child_rows == 0
    finally:
        barrier.release_all()
        worker.join(_E2_BARRIER_TIMEOUT_SECONDS)
        if contender is not None:
            assert contender.stdin is not None
            stdout, stderr = contender.communicate(
                "RELEASE\n", timeout=_E2_BARRIER_TIMEOUT_SECONDS
            )
            assert contender.returncode == 0, stdout + stderr


@pytest.mark.parametrize(
    "epoch_change",
    ["missing", "malformed", "nonfinite", "foreign"],
)
def test_state_meta_copy_fails_closed_before_target_mutation_if_destination_epoch_changes(
    tmp_path: Path,
    epoch_change: str,
) -> None:
    source = tmp_path / f"e2-generation-{epoch_change}-source.db"
    source_db = SessionDB(db_path=source)
    try:
        source_db.set_meta("e2:user", "source-must-not-land")
    finally:
        source_db.close()
    source_marker = _make_stale_source_rebuild_marker(source)
    source_conn = sqlite3.connect(str(source), isolation_level=None)
    source_rows = int(
        source_conn.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0]
    )

    control_child = None
    if epoch_change in {"nonfinite", "foreign"}:
        control_child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys; print('READY', flush=True); "
                "assert sys.stdin.readline() == 'RELEASE\\n'",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert control_child.stdin is not None
        assert control_child.stdout is not None
        ready, _, _ = select.select(
            [control_child.stdout], [], [], _E2_BARRIER_TIMEOUT_SECONDS
        )
        assert ready
        assert control_child.stdout.readline().strip() == "READY"
        assert control_child.poll() is None

    try:
        for copy_mode in ("normal", "salvage"):
            destination = tmp_path / (
                f"e2-generation-{epoch_change}-{copy_mode}-destination.db"
            )
            destination_db = SessionDB(db_path=destination)
            destination_db.set_meta("e2:user", "destination-before")
            destination_db.set_meta("e2:untouched", "must-remain-identical")
            try:
                with pytest.raises(
                    (SQLiteSerializationError, hermes_state.SessionTurnLeaseLostError)
                ):
                    with destination_db.offline_rebuild(
                        reason=f"e2 generation {epoch_change} {copy_mode}"
                    ):
                        original_marker = _read_offline_rebuild_marker(destination)
                        assert original_marker is not None
                        with sqlite3.connect(str(destination)) as verifier:
                            before_rows = verifier.execute(
                                "SELECT key, value FROM state_meta "
                                "WHERE key != ? ORDER BY key",
                                (_OFFLINE_REBUILD_EPOCH_KEY,),
                            ).fetchall()

                        if epoch_change == "missing":
                            adversarial_marker = None
                        elif epoch_change == "malformed":
                            adversarial_marker = "{malformed-e2-marker"
                        else:
                            assert control_child is not None
                            owner_start = hermes_state._process_start_time(
                                control_child.pid
                            )
                            assert owner_start is not None
                            if epoch_change == "nonfinite":
                                adversarial_marker = (
                                    '{"nonce":"e2-nonfinite","owner_pid":'
                                    f"{control_child.pid},"
                                    '"owner_pid_start":Infinity,'
                                    '"reason":"e2-control"}'
                                )
                            else:
                                adversarial_marker = json.dumps(
                                    {
                                        "nonce": "e2-foreign",
                                        "owner_pid": control_child.pid,
                                        "owner_pid_start": owner_start,
                                        "reason": "e2-control",
                                    },
                                    sort_keys=True,
                                    separators=(",", ":"),
                                )

                        with sqlite3.connect(
                            str(destination), isolation_level=None
                        ) as adversary:
                            if adversarial_marker is None:
                                adversary.execute(
                                    "DELETE FROM state_meta WHERE key = ?",
                                    (_OFFLINE_REBUILD_EPOCH_KEY,),
                                )
                            else:
                                adversary.execute(
                                    "UPDATE state_meta SET value = ? WHERE key = ?",
                                    (
                                        adversarial_marker,
                                        _OFFLINE_REBUILD_EPOCH_KEY,
                                    ),
                                )

                        copy_function = (
                            session_recovery._copy_state_meta_salvage
                            if copy_mode == "salvage"
                            else session_recovery._copy_state_meta
                        )
                        with pytest.raises(
                            SQLiteSerializationError,
                            match=(
                                "^offline rebuild exclusion changed before its "
                                "owner released it$"
                            ),
                        ) as caught:
                            copy_function(
                                source_conn,
                                destination_db,
                                chunk_size=1,
                                progress_cb=None,
                                source_rows=source_rows,
                            )
                        assert str(caught.value) == (
                            "offline rebuild exclusion changed before its owner "
                            "released it"
                        )
                        assert source_marker not in str(caught.value)

                        with sqlite3.connect(str(destination)) as verifier:
                            after_rows = verifier.execute(
                                "SELECT key, value FROM state_meta "
                                "WHERE key != ? ORDER BY key",
                                (_OFFLINE_REBUILD_EPOCH_KEY,),
                            ).fetchall()
                        assert after_rows == before_rows
                        assert (
                            _read_offline_rebuild_marker(destination)
                            == adversarial_marker
                        )
            finally:
                destination_db.close()
    finally:
        source_conn.close()
        if control_child is not None:
            assert control_child.stdin is not None
            stdout, stderr = control_child.communicate(
                "RELEASE\n", timeout=_E2_BARRIER_TIMEOUT_SECONDS
            )
            assert control_child.returncode == 0, stdout + stderr



