from __future__ import annotations

import hashlib
import json
import os
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
    SessionDB,
    register_turn_fence_generation,
)
from hermes_cli import session_recovery
from hermes_cli.session_recovery import (
    SessionRecoveryDestinationError,
    SessionRecoverySafetyError,
    SessionRecoverySourceError,
    _is_current_turn_fence_generation,
    _require_destination_fenced,
    inspect_session_database,
    recover_session_database,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_seal_authority_interleaving(
    monkeypatch: pytest.MonkeyPatch,
    *,
    authority_select: int,
    takeover_marker: bytes | None,
) -> dict[str, object]:
    """Attempt a second-connection takeover immediately after a seal proof."""
    state: dict[str, object] = {
        "authority_selects": 0,
        "candidate": None,
        "raw_after_takeover": [],
        "sealing_pragmas": [],
        "sealing_active": False,
        "takeover_attempted": False,
        "takeover_committed": False,
        "takeover_error": None,
    }
    original_connect = sqlite3.connect
    original_create_stage = session_recovery._create_destination_stage
    original_seal = session_recovery._seal_staged_database

    def take_over(candidate: Path) -> None:
        state["takeover_attempted"] = True
        writer = original_connect(str(candidate), isolation_level=None, timeout=0)
        try:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                (
                    hermes_state._OFFLINE_REBUILD_EPOCH_KEY,
                    None
                    if takeover_marker is None
                    else sqlite3.Binary(takeover_marker),
                ),
            )
            writer.execute("COMMIT")
            state["takeover_committed"] = True
        except sqlite3.OperationalError as exc:
            state["takeover_error"] = exc
        finally:
            if writer.in_transaction:
                writer.execute("ROLLBACK")
            writer.close()

    class _AuthorityCursor:
        def __init__(self, cursor: sqlite3.Cursor, candidate: Path) -> None:
            self._cursor = cursor
            self._candidate = candidate

        def fetchall(self):
            rows = self._cursor.fetchall()
            assert rows == []
            take_over(self._candidate)
            return rows

        def __getattr__(self, name: str):
            return getattr(self._cursor, name)

    class _SealingConnection:
        def __init__(self, connection: sqlite3.Connection, candidate: Path) -> None:
            self._connection = connection
            self._candidate = candidate

        def execute(self, sql: str, parameters=()):
            normalized = sql.lstrip().upper()
            if normalized.startswith("PRAGMA WAL_CHECKPOINT(TRUNCATE)") or normalized.startswith(
                "PRAGMA JOURNAL_MODE=DELETE"
            ):
                state["sealing_pragmas"].append(normalized)
                if state["takeover_committed"]:
                    state["raw_after_takeover"].append(normalized)
            cursor = self._connection.execute(sql, parameters)
            if normalized.startswith("SELECT VALUE FROM STATE_META"):
                state["authority_selects"] += 1
                if state["authority_selects"] == authority_select:
                    return _AuthorityCursor(cursor, self._candidate)
            return cursor

        def __getattr__(self, name: str):
            return getattr(self._connection, name)

    def capture_stage(path: Path):
        stage = original_create_stage(path)
        state["candidate"] = stage.candidate
        return stage

    def sealing_connection(database, *args, **kwargs):
        connection = original_connect(database, *args, **kwargs)
        candidate = state["candidate"]
        if (
            state["sealing_active"]
            and isinstance(candidate, Path)
            and Path(database) == candidate
        ):
            return _SealingConnection(connection, candidate)
        return connection

    def seal_with_interleaving(stage):
        state["sealing_active"] = True
        try:
            return original_seal(stage)
        finally:
            state["sealing_active"] = False

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_seal_staged_database", seal_with_interleaving)
    monkeypatch.setattr(session_recovery.sqlite3, "connect", sealing_connection)
    return state


def _assert_seal_authority_interleaving_blocked(state: dict[str, object]) -> None:
    """Assert the attempted takeover never became authority for a raw pragma."""
    assert state["takeover_attempted"] is True
    assert state["raw_after_takeover"] == []
    assert state["takeover_committed"] is False
    assert isinstance(state["takeover_error"], sqlite3.OperationalError)
    pragmas = state["sealing_pragmas"]
    assert any("WAL_CHECKPOINT(TRUNCATE)" in sql for sql in pragmas)
    assert any("JOURNAL_MODE=DELETE" in sql for sql in pragmas)


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
    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        # The source was built via SessionDB, whose turn-fence triggers on
        # `sessions` need hermes_turn_fence_generation() registered on
        # whichever connection issues the DELETE that simulates corruption.
        register_turn_fence_generation(conn)
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


def test_recover_exact_path_copies_rows_into_a_real_v27_store(
    tmp_path: Path,
) -> None:
    """The exact (non-``--allow-partial``) path must copy into a real store.

    Regression: the destination is opened with a bare ``sqlite3.connect()``
    after the initializing ``SessionDB`` handle is closed. A v27 store's
    canonical tables (``sessions``, ``messages``, ...) carry BEFORE INSERT/
    UPDATE/DELETE turn-fence triggers that call ``hermes_turn_fence_
    generation()`` — a UDF only ever registered on the connection that
    created it. Reopening raw and never re-registering it makes the very
    first insert into any canonical table fail with
    ``OperationalError: no such function: hermes_turn_fence_generation``,
    which ``_copy_table`` swallows per-table as ``status: "failed"``. Every
    row must instead land, exactly.
    """
    source = tmp_path / "healthy-source.db"
    output = tmp_path / "healthy-recovered.db"
    expected = _make_source(source)

    report = recover_session_database(source, output, work_dir=tmp_path, chunk_size=4)

    for table in ("sessions", "messages"):
        assert report["copy"][table]["status"] == "complete", report["copy"][table]
        assert "error" not in report["copy"][table], report["copy"][table]

    assert report["complete"] is True
    assert report["partial"] is False
    assert report["verified"] is True
    assert report["output"] == str(output)
    assert ".hermes-session-recovery-" not in json.dumps(report)
    assert output.exists()
    for suffix in ("-wal", "-shm", "-journal"):
        assert not os.path.lexists(output.with_name(output.name + suffix))

    with sqlite3.connect(str(output)) as verify:
        verify.row_factory = sqlite3.Row
        session_count = verify.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = verify.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert session_count == expected["sessions"]
    assert message_count == expected["messages"]


def test_normal_recovery_regenerates_destination_session_process_authority(
    tmp_path: Path,
) -> None:
    """Recovery must not import another database's session capability state."""
    source = tmp_path / "authority-source.db"
    output = tmp_path / "authority-recovered.db"
    session_id = "authority-recovery-session"
    source_db = SessionDB(db_path=source)
    try:
        source_db.create_session(session_id, "cli", cwd="/tmp/authority-recovery")
        source_db.end_session(session_id, "test-authority-generation")
        source_db.reopen_session(session_id)
        source_authority = source_db.issue_session_process_authority(session_id)
        assert source_authority is not None
        assert source_authority["session_generation"] == 2
        source_reservation = source_db.reserve_session_process_authority(
            source_authority
        )
        assert source_reservation is not None
        source_events = source_db._conn.execute(
            "SELECT event_type FROM session_process_authority_events "
            "WHERE state_db_id = ? ORDER BY id",
            (source_authority["state_db_id"],),
        ).fetchall()
        assert any(row[0] == "PROCESS_RESERVATION" for row in source_events)
    finally:
        source_db.close()

    report = recover_session_database(source, output, work_dir=tmp_path)

    assert report["complete"] is True, report["copy"]["sessions"]
    assert report["verified"] is True
    recovered_db = SessionDB(db_path=output)
    try:
        destination_authority = recovered_db.issue_session_process_authority(session_id)
        assert destination_authority is not None
        destination_meta = {
            str(row[0]): row[1]
            for row in recovered_db._conn.execute(
                "SELECT key, value FROM state_meta "
                "WHERE key IN (?, ?)",
                (
                    "session_process_state_db_id",
                    "session_process_state_family",
                ),
            ).fetchall()
        }
        assert (
            destination_meta["session_process_state_db_id"],
            destination_meta["session_process_state_family"],
        ) == (
            destination_authority["state_db_id"],
            destination_authority["state_family"],
        )
        assert destination_authority["session_generation"] == 1
        assert (
            destination_authority["state_db_id"],
            destination_authority["state_family"],
        ) != (
            source_authority["state_db_id"],
            source_authority["state_family"],
        )
        assert destination_authority["authority_token"] != source_authority["authority_token"]
        assert recovered_db.consume_session_process_reservation(source_reservation) is False
        assert (
            recovered_db._conn.execute(
                "SELECT COUNT(*) FROM session_process_authorities "
                "WHERE state_db_id = ? OR authority_token = ?",
                (
                    source_authority["state_db_id"],
                    source_authority["authority_token"],
                ),
            ).fetchone()[0]
            == 0
        )
        assert (
            recovered_db._conn.execute(
                "SELECT COUNT(*) FROM session_process_reservations "
                "WHERE reservation_id = ? OR state_db_id = ?",
                (
                    source_reservation["reservation_id"],
                    source_authority["state_db_id"],
                ),
            ).fetchone()[0]
            == 0
        )
        assert (
            recovered_db._conn.execute(
                "SELECT COUNT(*) FROM session_process_authority_events "
                "WHERE state_db_id = ? OR reservation_id = ?",
                (
                    source_authority["state_db_id"],
                    source_reservation["reservation_id"],
                ),
            ).fetchone()[0]
            == 0
        )
    finally:
        recovered_db.close()


def test_normal_recovery_stops_before_later_dml_when_marker_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed chunk must not authorize the next destination write."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    candidate: Path | None = None
    original_create_stage = session_recovery._create_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_cleanup_destination_stage", lambda _stage: None)

    foreign_marker = "foreign-recovery-owner"
    adversarial_fts_value = "foreign-fts-storage-version"
    adversarial_meta_value = "foreign-adversarial-value"
    replaced = False

    def replace_marker_after_committed_chunk(progress: dict[str, object]) -> None:
        nonlocal replaced
        if replaced or progress["table"] != "sessions":
            return
        assert candidate is not None
        with sqlite3.connect(str(candidate), isolation_level=None) as writer:
            marker = writer.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
            ).fetchone()
            assert marker is not None, "recovery must own a durable marker"
            cursor = writer.execute(
                "UPDATE state_meta SET value = ? WHERE key = ?",
                (foreign_marker, hermes_state._OFFLINE_REBUILD_EPOCH_KEY),
            )
            assert cursor.rowcount == 1
            writer.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                ("fts_storage_version", adversarial_fts_value),
            )
            writer.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                ("adversarial-after-marker", adversarial_meta_value),
            )
        replaced = True

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        recover_session_database(
            source,
            output,
            work_dir=tmp_path,
            chunk_size=1,
            progress_cb=replace_marker_after_committed_chunk,
        )

    assert replaced is True
    assert candidate is not None
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (foreign_marker,)
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_storage_version'"
        ).fetchone() == (adversarial_fts_value,)
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = 'adversarial-after-marker'"
        ).fetchone() == (adversarial_meta_value,)
        assert destination.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
        assert destination.execute("SELECT COUNT(*) FROM messages").fetchone() == (0,)
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = 'goal:recovery-session-0'"
        ).fetchone() is None


def test_recovery_health_probe_stops_before_dml_when_marker_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real recovery health probe cannot write after a claim takeover."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    candidate: Path | None = None
    original_create_stage = session_recovery._create_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_cleanup_destination_stage", lambda _stage: None)

    foreign_marker = "foreign-health-probe-owner"
    adversarial_key = "adversarial-health-probe-marker"
    adversarial_value = "foreign-health-probe-bytes"
    takeover_complete = False
    health_probe_dml: list[str] = []
    original_connect_repair = hermes_state._connect_repair_durable

    def trace_health_probe_connection(path: Path) -> sqlite3.Connection:
        connection = original_connect_repair(path)

        def record_dml(sql: str) -> None:
            if takeover_complete and sql.lstrip().upper().startswith(
                ("INSERT ", "UPDATE ", "DELETE ", "REPLACE ")
            ):
                health_probe_dml.append(sql)

        connection.set_trace_callback(record_dml)
        return connection

    monkeypatch.setattr(
        hermes_state, "_connect_repair_durable", trace_health_probe_connection
    )
    real_health_probe = hermes_state._db_opens_cleanly

    def replace_marker_before_health_probe(path: Path, **kwargs: object):
        nonlocal takeover_complete
        assert candidate is not None
        with sqlite3.connect(str(candidate), isolation_level=None) as writer:
            cursor = writer.execute(
                "UPDATE state_meta SET value = ? WHERE key = ?",
                (foreign_marker, hermes_state._OFFLINE_REBUILD_EPOCH_KEY),
            )
            assert cursor.rowcount == 1
            writer.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                (adversarial_key, adversarial_value),
            )
            # The current schema's turn-fence triggers are a second line of
            # defense. Remove them here so this regression proves the health
            # probe itself validates the no-owner contract before it tries a
            # write, including against a legacy candidate without those
            # triggers.
            trigger_names = [
                row[0]
                for row in writer.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            ]
            for trigger_name in trigger_names:
                writer.execute(f'DROP TRIGGER "{trigger_name}"')
        takeover_complete = True
        return real_health_probe(path, **kwargs)

    monkeypatch.setattr(
        session_recovery, "_db_opens_cleanly", replace_marker_before_health_probe
    )

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        recover_session_database(source, output, work_dir=tmp_path, chunk_size=1)

    assert takeover_complete is True
    assert health_probe_dml == []
    assert candidate is not None
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (foreign_marker,)
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (adversarial_key,),
        ).fetchone() == (adversarial_value,)


def test_seal_health_probe_refuses_foreign_marker_without_turn_fence_triggers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seal's own health write probe fences legacy candidates before DML."""
    output = tmp_path / "recovered.db"
    stage = session_recovery._create_destination_stage(output)
    db = SessionDB(db_path=stage.candidate)
    db.close()
    session_recovery._refresh_stage_children(stage, require_main=True)
    foreign_marker = "foreign-seal-health-probe-owner"
    dml: list[str] = []
    original_connect = hermes_state._connect_repair_durable
    real_health_probe = hermes_state._db_opens_cleanly

    def traced_connect(path: Path) -> sqlite3.Connection:
        connection = original_connect(path)
        connection.set_trace_callback(
            lambda sql: dml.append(sql)
            if sql.lstrip().upper().startswith(("INSERT ", "UPDATE ", "DELETE "))
            else None
        )
        return connection

    def take_over_before_health_probe(path: Path, **kwargs: object):
        with sqlite3.connect(str(path), isolation_level=None) as writer:
            writer.execute(
                "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                (hermes_state._OFFLINE_REBUILD_EPOCH_KEY, foreign_marker),
            )
            trigger_names = [
                row[0]
                for row in writer.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                )
            ]
            for trigger_name in trigger_names:
                writer.execute(f'DROP TRIGGER "{trigger_name}"')
        return real_health_probe(path, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_repair_durable", traced_connect)
    monkeypatch.setattr(session_recovery, "_db_opens_cleanly", take_over_before_health_probe)

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        session_recovery._seal_staged_database(stage)

    assert dml == []
    with sqlite3.connect(str(stage.candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (foreign_marker,)


def test_normal_recovery_excludes_the_source_offline_rebuild_marker(
    tmp_path: Path,
) -> None:
    """A stale source claim must never displace recovery's own claim."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_marker = "foreign-source-rebuild-marker"
    source_db = SessionDB(db_path=source)
    try:
        source_db.set_meta(hermes_state._OFFLINE_REBUILD_EPOCH_KEY, source_marker)
    finally:
        source_db.close()

    report = recover_session_database(source, output, work_dir=tmp_path, chunk_size=1)

    assert hermes_state._OFFLINE_REBUILD_EPOCH_KEY in report["copy"]["state_meta"][
        "excluded_keys"
    ]
    with sqlite3.connect(str(output)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() is None


@pytest.mark.parametrize("takeover", ("replace", "delete"))
def test_normal_recovery_topic_migration_stops_after_authority_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    takeover: str,
) -> None:
    """Topic migration must keep its scripts inside recovery's owner transaction."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)

    candidate: Path | None = None
    original_create_stage = session_recovery._create_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_cleanup_destination_stage", lambda _stage: None)

    foreign_marker = "foreign-topic-migration-owner-\u03bb"
    takeover_started = threading.Event()
    takeover_complete = threading.Event()
    migration_script_seen = threading.Event()
    takeover_errors: list[BaseException] = []
    later_topic_mutations: list[str] = []
    authority_checks = 0
    waited_after_migration = False

    def is_candidate_db(db_path: object) -> bool:
        return candidate is not None and Path(str(db_path)).resolve() == candidate.resolve()

    def take_over_marker() -> None:
        takeover_started.set()
        try:
            assert candidate is not None
            with sqlite3.connect(str(candidate), isolation_level=None, timeout=3.0) as writer:
                marker = writer.execute(
                    "SELECT value FROM state_meta WHERE key = ?",
                    (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
                ).fetchone()
                assert marker is not None, "recovery must own a durable marker"
                if takeover == "replace":
                    cursor = writer.execute(
                        "UPDATE state_meta SET value = ? WHERE key = ?",
                        (foreign_marker, hermes_state._OFFLINE_REBUILD_EPOCH_KEY),
                    )
                    assert cursor.rowcount == 1
                else:
                    cursor = writer.execute(
                        "DELETE FROM state_meta WHERE key = ?",
                        (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
                    )
                    assert cursor.rowcount == 1
        except BaseException as exc:
            takeover_errors.append(exc)
        finally:
            takeover_complete.set()

    original_connect = hermes_state._connect_tracked_db

    def connect_with_topic_migration_trace(path: object, *args: object, **kwargs: object):
        conn = original_connect(path, *args, **kwargs)
        if not is_candidate_db(path):
            return conn

        def trace_topic_migration(sql: str) -> None:
            normalized = sql.lstrip().upper()
            if normalized.startswith(
                "CREATE TABLE IF NOT EXISTS TELEGRAM_DM_TOPIC_MODE"
            ) and not migration_script_seen.is_set():
                # _execute_write already made its exact-owner comparison.
                assert authority_checks >= 2
                migration_script_seen.set()
                if conn.in_transaction:
                    threading.Thread(target=take_over_marker, daemon=True).start()
                else:
                    take_over_marker()
            elif takeover_complete.is_set() and (
                "TELEGRAM_DM_TOPIC" in normalized
                or normalized.startswith("INSERT INTO STATE_META")
            ) and normalized.startswith(("CREATE ", "DROP ", "ALTER ", "INSERT ", "UPDATE ")):
                later_topic_mutations.append(sql)

        conn.set_trace_callback(trace_topic_migration)
        return conn

    monkeypatch.setattr(
        hermes_state, "_connect_tracked_db", connect_with_topic_migration_trace
    )
    original_assert_authority = SessionDB._assert_offline_rebuild_write_authority

    def count_migration_authority_checks(self: SessionDB, conn: sqlite3.Connection) -> None:
        nonlocal authority_checks
        original_assert_authority(self, conn)
        if is_candidate_db(self.db_path) and self._offline_rebuild_marker is not None:
            authority_checks += 1

    monkeypatch.setattr(
        SessionDB, "_assert_offline_rebuild_write_authority", count_migration_authority_checks
    )
    original_execute_write = SessionDB._execute_write

    def wait_for_takeover_after_migration(self: SessionDB, *args: object, **kwargs: object):
        nonlocal waited_after_migration
        result = original_execute_write(self, *args, **kwargs)
        if (
            is_candidate_db(self.db_path)
            and migration_script_seen.is_set()
            and not waited_after_migration
        ):
            waited_after_migration = True
            assert takeover_complete.wait(3.0), "foreign takeover did not finish"
        return result

    monkeypatch.setattr(SessionDB, "_execute_write", wait_for_takeover_after_migration)

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        recover_session_database(source, output, work_dir=tmp_path, chunk_size=1)

    assert migration_script_seen.is_set()
    assert takeover_started.is_set()
    assert takeover_complete.is_set()
    assert takeover_errors == []
    assert later_topic_mutations == []
    assert candidate is not None
    with sqlite3.connect(str(candidate)) as destination:
        marker = destination.execute(
            "SELECT CAST(value AS BLOB) FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone()
        if takeover == "replace":
            assert marker == (foreign_marker.encode("utf-8"),)
        else:
            assert marker is None


def test_sql_salvage_stops_before_later_dml_when_marker_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Salvage must recheck the claim after every committed rowid chunk."""
    source = tmp_path / "source.db"
    filtered_output = tmp_path / "filtered.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_db = SessionDB(db_path=source)
    try:
        source_db.set_meta(
            hermes_state._OFFLINE_REBUILD_EPOCH_KEY,
            "foreign-source-rebuild-marker",
        )
    finally:
        source_db.close()

    filtered_report = recover_session_database(
        source,
        filtered_output,
        work_dir=tmp_path,
        chunk_size=1,
        allow_partial=True,
    )
    assert hermes_state._OFFLINE_REBUILD_EPOCH_KEY in filtered_report["copy"][
        "state_meta"
    ]["excluded_keys"]
    with sqlite3.connect(str(filtered_output)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() is None

    candidate: Path | None = None
    original_create_stage = session_recovery._create_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_cleanup_destination_stage", lambda _stage: None)

    foreign_marker = "foreign-salvage-owner"
    adversarial_fts_value = "foreign-salvage-fts-storage-version"
    adversarial_meta_value = "foreign-salvage-adversarial-value"
    foreign_session_id = "foreign-salvage-orphan"
    replaced = False

    def replace_marker_after_committed_chunk(progress: dict[str, object]) -> None:
        nonlocal replaced
        if replaced or progress["table"] != "sessions":
            return
        assert candidate is not None
        with sqlite3.connect(str(candidate), isolation_level=None) as writer:
            marker = writer.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
            ).fetchone()
            assert marker is not None, "recovery must own a durable marker"
            cursor = writer.execute(
                "UPDATE state_meta SET value = ? WHERE key = ?",
                (foreign_marker, hermes_state._OFFLINE_REBUILD_EPOCH_KEY),
            )
            assert cursor.rowcount == 1
            writer.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                ("fts_storage_version", adversarial_fts_value),
            )
            writer.execute(
                "INSERT OR REPLACE INTO state_meta(key, value) VALUES (?, ?)",
                ("adversarial-after-salvage-marker", adversarial_meta_value),
            )
            register_turn_fence_generation(writer)
            writer.execute(
                "INSERT INTO messages(id, session_id, role, content, timestamp) "
                "VALUES (999999, ?, 'user', 'foreign orphan', 1.0)",
                (foreign_session_id,),
            )
        replaced = True

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        recover_session_database(
            source,
            output,
            work_dir=tmp_path,
            chunk_size=1,
            progress_cb=replace_marker_after_committed_chunk,
            allow_partial=True,
        )

    assert replaced is True
    assert candidate is not None
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (foreign_marker,)
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = 'fts_storage_version'"
        ).fetchone() == (adversarial_fts_value,)
        assert destination.execute(
            "SELECT value FROM state_meta "
            "WHERE key = 'adversarial-after-salvage-marker'"
        ).fetchone() == (adversarial_meta_value,)
        assert destination.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)
        assert destination.execute("SELECT COUNT(*) FROM messages").fetchone() == (1,)
        assert destination.execute(
            "SELECT id FROM sessions WHERE id = ?", (foreign_session_id,)
        ).fetchone() is None


def test_recover_into_destination_without_fence_triggers_still_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix must not depend on the destination having fence triggers.

    Simulates an older-shaped destination schema (canonical tables present,
    turn-fence trigger barrier never installed) by disabling the delta that
    installs it while ``recover_session_database`` builds the destination.
    ``register_turn_fence_generation`` is harmless either way — it only
    registers a scalar function, and nothing in the destination references
    it when there are no triggers to call it — so recovery must still
    succeed.
    """
    def _stamp_schema_version_without_triggers(self, cursor) -> None:
        # Stamp schema_version as the real delta does, but skip installing
        # the turn-fence triggers themselves — reproducing the shape of an
        # older store that is otherwise fully migrated.
        cursor.execute("DELETE FROM schema_version")
        cursor.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (hermes_state.SCHEMA_VERSION,),
        )
        self._conn.commit()

    monkeypatch.setattr(
        hermes_state.SessionDB,
        "_apply_turn_fence_generation_delta",
        _stamp_schema_version_without_triggers,
    )

    source = tmp_path / "healthy-source-notriggers.db"
    output = tmp_path / "healthy-recovered-notriggers.db"
    expected = _make_source(source)

    report = recover_session_database(source, output, work_dir=tmp_path, chunk_size=4)

    with sqlite3.connect(str(output)) as verify:
        trigger_count = verify.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type = 'trigger' AND name LIKE 'turn_fence_%'"
        ).fetchone()[0]
        session_count = verify.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = verify.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    assert trigger_count == 0, "fixture must genuinely lack the fence triggers"
    assert report["copy"]["sessions"]["status"] == "complete", report["copy"]["sessions"]
    assert report["copy"]["messages"]["status"] == "complete", report["copy"]["messages"]
    assert report["complete"] is True
    assert report["verified"] is True
    assert session_count == expected["sessions"]
    assert message_count == expected["messages"]


def test_destination_fence_probe_rejects_bool_even_with_integer_sqlite_type(
) -> None:
    """SQLite considers bool an integer, but the recovery fence must not."""

    assert not _is_current_turn_fence_generation(True, "integer")
    assert _is_current_turn_fence_generation(hermes_state.TURN_FENCE_GENERATION, "integer")


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
def test_exact_recovery_destination_fence_failures_leave_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_hash = _sha256(source)

    def register_fault(conn: sqlite3.Connection) -> None:
        if fault == "no-registration":
            return
        if fault == "registration-raises":
            raise sqlite3.OperationalError("private setup failure")
        values = {
            "stale-int": hermes_state.TURN_FENCE_GENERATION - 1,
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
        recover_session_database(source, output, work_dir=tmp_path)

    assert str(excinfo.value) == (
        "Recovery destination turn-fence setup is unavailable or incompatible."
    )
    assert _sha256(source) == source_hash


def test_normal_recovery_publication_refuses_marker_after_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication re-proves no-owner authority after the final testable seam."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_hash = _sha256(source)
    candidate: Path | None = None
    foreign_marker = "foreign-normal-publication-owner"
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
                (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
            ).fetchone() is None
            writer.execute(
                "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                (hermes_state._OFFLINE_REBUILD_EPOCH_KEY, foreign_marker),
            )

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_publication_barrier", take_over_after_barrier)

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        recover_session_database(source, output, work_dir=tmp_path, chunk_size=1)

    assert candidate is not None and candidate.exists()
    assert not output.exists()
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (foreign_marker,)
    assert _sha256(source) == source_hash
    for suffix in ("", "-wal", "-shm", "-journal"):
        assert not os.path.lexists(output.with_name(output.name + suffix))


def test_normal_recovery_copy_refusal_retains_the_real_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copy-time authority loss reaches cleanup but retains staged evidence."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    candidate: Path | None = None
    stage = None
    copy_started = False
    cleanup_calls = 0
    original_create_stage = session_recovery._create_destination_stage
    original_copy_table = session_recovery._copy_table
    original_cleanup = session_recovery._cleanup_destination_stage

    def capture_stage(path: Path):
        nonlocal candidate, stage
        stage = original_create_stage(path)
        candidate = stage.candidate
        return stage

    def take_over_during_real_copy(source_conn, destination, table, **kwargs):
        nonlocal copy_started
        if not copy_started:
            assert candidate is not None
            copy_started = True
            with sqlite3.connect(str(candidate), isolation_level=None) as writer:
                writer.execute(
                    "UPDATE state_meta SET value = ? WHERE key = ?",
                    (
                        "foreign-normal-copy-owner",
                        hermes_state._OFFLINE_REBUILD_EPOCH_KEY,
                    ),
                )
            assert stage is not None
            session_recovery._refresh_stage_children(stage, require_main=True)
        return original_copy_table(source_conn, destination, table, **kwargs)

    def record_real_cleanup(destination_stage) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        original_cleanup(destination_stage)

    monkeypatch.setattr(session_recovery, "_create_destination_stage", capture_stage)
    monkeypatch.setattr(session_recovery, "_copy_table", take_over_during_real_copy)
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

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        recover_session_database(source, output, work_dir=tmp_path, chunk_size=1)

    assert copy_started is True
    assert stage is not None and stage.retain_on_authority_refusal is True
    assert cleanup_calls == 0
    assert candidate is not None and candidate.exists()
    assert not output.exists()
    with sqlite3.connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == ("foreign-normal-copy-owner",)


def test_sessions_recover_cli_maps_destination_error_without_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from hermes_cli.sessions_cmd import cmd_sessions

    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_hash = _sha256(source)

    def fail_destination_setup(_connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("private setup failure")

    monkeypatch.setattr(
        session_recovery,
        "register_turn_fence_generation",
        fail_destination_setup,
    )

    args = SimpleNamespace(
        sessions_action="recover",
        source=source,
        output=output,
        inspect_only=False,
        allow_partial=False,
        report=None,
        work_dir=tmp_path,
        chunk_size=4,
    )
    assert cmd_sessions(args) == 1

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.splitlines() == [
        "Recovering canonical session data into a new database…",
        "Error: session recovery failed: "
        "Recovery destination turn-fence setup is unavailable or incompatible.",
        "The supplied source database was not replaced or deleted.",
    ]
    for forbidden in ("✓", "Partial recovery", "BEST-EFFORT", "Recovery report"):
        assert forbidden not in captured.out
    assert "private setup failure" not in captured.out

    report_path = output.with_name(output.name + ".recovery.json")
    assert not report_path.exists()
    for suffix in ("", "-wal", "-shm", "-journal"):
        assert not os.path.lexists(output.with_name(output.name + suffix))
    assert _sha256(source) == source_hash


@pytest.mark.parametrize("target", ("main", "wal", "shm", "journal"))
def test_recovery_publication_collision_preserves_existing_sentinel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_hash = _sha256(source)
    sentinel = output.with_name(
        output.name if target == "main" else f"{output.name}-{target}"
    )
    sentinel_bytes = f"sentinel-{target}".encode()
    sentinel_identity: tuple[int, int] | None = None

    def insert_competitor(_candidate: Path, _final: Path) -> None:
        nonlocal sentinel_identity
        sentinel.write_bytes(sentinel_bytes)
        created = sentinel.stat()
        sentinel_identity = (created.st_dev, created.st_ino)

    monkeypatch.setattr(
        session_recovery,
        "_publication_barrier",
        insert_competitor,
    )

    with pytest.raises(SessionRecoverySafetyError):
        recover_session_database(source, output, work_dir=tmp_path)

    sentinel_stat = sentinel.stat()
    assert sentinel_identity is not None
    assert sentinel.read_bytes() == sentinel_bytes
    assert (sentinel_stat.st_dev, sentinel_stat.st_ino) == sentinel_identity
    assert not output.exists() or target == "main"
    assert _sha256(source) == source_hash


def test_recovery_publication_refuses_substituted_stage_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_hash = _sha256(source)
    replacement_bytes = b"substituted stage candidate"
    replacement: Path | None = None
    replacement_identity: tuple[int, int] | None = None

    def substitute(candidate: Path, _final: Path) -> None:
        nonlocal replacement, replacement_identity
        candidate.unlink()
        candidate.write_bytes(replacement_bytes)
        replacement = candidate
        created = candidate.stat()
        replacement_identity = (created.st_dev, created.st_ino)

    monkeypatch.setattr(session_recovery, "_publication_barrier", substitute)

    with pytest.raises(SessionRecoverySafetyError):
        recover_session_database(source, output, work_dir=tmp_path)

    assert replacement is not None
    assert replacement_identity is not None
    assert replacement.read_bytes() == replacement_bytes
    current = replacement.stat()
    assert (current.st_dev, current.st_ino) == replacement_identity
    assert not output.exists()
    assert _sha256(source) == source_hash


@pytest.mark.parametrize(
    ("takeover_marker", "expected_marker"),
    (
        (b"\x00foreign-normal-seal-owner\xff", b"\x00foreign-normal-seal-owner\xff"),
        (None, None),
    ),
    ids=("foreign", "null"),
)
def test_normal_recovery_seal_refuses_marker_after_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    takeover_marker: bytes | None,
    expected_marker: bytes | None,
) -> None:
    """Sealing must not perform raw maintenance after recovery releases its claim."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_hash = _sha256(source)
    candidate: Path | None = None
    marker_key = hermes_state._OFFLINE_REBUILD_EPOCH_KEY
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
            # The local compare-delete release must be complete before this
            # independent owner arrives after sealing's stage validation.
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
        recover_session_database(source, output, work_dir=tmp_path, chunk_size=1)
    except BaseException as exc:
        caught = exc

    assert raw_maintenance == []
    assert isinstance(caught, hermes_state.SessionTurnLeaseLostError)
    assert takeover_complete is True
    assert candidate is not None
    assert not output.exists()
    with original_connect(str(candidate)) as destination:
        assert destination.execute(
            "SELECT CAST(value AS BLOB) FROM state_meta WHERE key = ?",
            (marker_key,),
        ).fetchone() == (expected_marker,)
    assert _sha256(source) == source_hash


@pytest.mark.parametrize(
    ("authority_select", "takeover_marker"),
    (
        (1, b"\x00foreign-normal-checkpoint-owner\xff"),
        (1, None),
        (2, b"\x00foreign-normal-journal-owner\xff"),
        (2, None),
    ),
    ids=("checkpoint-foreign", "checkpoint-null", "journal-foreign", "journal-null"),
)
def test_normal_recovery_seal_blocks_takeover_at_pragma_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    authority_select: int,
    takeover_marker: bytes | None,
) -> None:
    """The seal retains SQLite authority through each raw maintenance boundary."""
    source = tmp_path / "source.db"
    output = tmp_path / "recovered.db"
    _make_source(source)
    source_hash = _sha256(source)
    state = _arm_seal_authority_interleaving(
        monkeypatch,
        authority_select=authority_select,
        takeover_marker=takeover_marker,
    )

    report = None
    recovery_error: BaseException | None = None
    try:
        report = recover_session_database(source, output, work_dir=tmp_path, chunk_size=1)
    except BaseException as exc:
        recovery_error = exc

    assert state["authority_selects"] >= authority_select
    _assert_seal_authority_interleaving_blocked(state)
    assert recovery_error is None
    assert report is not None
    assert report["copy"]["sessions"]["copied_rows"] == 3
    assert output.exists()
    with sqlite3.connect(str(output)) as destination:
        assert destination.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() is None
        assert destination.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    assert _sha256(source) == source_hash
