"""Regression coverage for terminal SQLite transaction cleanup."""

from __future__ import annotations

import sqlite3

import pytest

import hermes_state
from hermes_cli.sqlite_safe_read import has_live_connection
from hermes_state import SessionDB


class _RollbackFailureConnection(sqlite3.Connection):
    fail_rollback = False

    def rollback(self):
        if self.fail_rollback:
            raise sqlite3.OperationalError("forced rollback failure")
        return super().rollback()


class _CommitThenFailConnection(sqlite3.Connection):
    fail_commit = False

    def commit(self):
        result = super().commit()
        if self.fail_commit:
            raise sqlite3.OperationalError("ambiguous post-commit failure")
        return result


class _CommitOnceThenRaiseConnection(sqlite3.Connection):
    """Durably commit once, then report an ambiguous SQLite failure."""

    commit_error_type = sqlite3.InterfaceError
    fail_next_commit = False

    def commit(self):
        result = super().commit()
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise self.commit_error_type("no more rows available after commit")
        return result


class _ForeignKeyRestoreFailureConnection(hermes_state._SerializedConnection):
    fail_foreign_key_restore = False

    def execute(self, sql, parameters=()):
        if (
            self.fail_foreign_key_restore
            and " ".join(sql.upper().split()) == "PRAGMA FOREIGN_KEYS=ON"
        ):
            raise sqlite3.OperationalError("forced foreign-key restoration failure")
        return super().execute(sql, parameters)


def test_full_rollback_failure_retires_tracked_connection_once(tmp_path, monkeypatch):
    """A failed rollback detaches the handle before another writer proceeds."""
    db_path = tmp_path / "state.db"
    original_connect = hermes_state._connect_tracked_db

    def connect_with_rollback_failure(*args, **kwargs):
        kwargs["factory"] = _RollbackFailureConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        hermes_state, "_connect_tracked_db", connect_with_rollback_failure
    )
    db = SessionDB(db_path=db_path)
    try:
        db._conn.execute("CREATE TABLE transaction_recovery_probe (value TEXT)")
        db._conn.commit()
        owner = db._conn
        owner.fail_rollback = True

        def body(conn):
            conn.execute(
                "INSERT INTO transaction_recovery_probe(value) VALUES ('owner')"
            )
            raise RuntimeError("body failed")

        with pytest.raises(RuntimeError, match="body failed") as exc_info:
            db._execute_write(body)

        assert isinstance(exc_info.value.__cause__, sqlite3.OperationalError)
        assert db._conn is None
        assert not has_live_connection(db_path)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            sqlite3.Connection.execute(owner, "SELECT 1")

        foreign = sqlite3.connect(db_path, isolation_level=None)
        try:
            foreign.execute("BEGIN IMMEDIATE")
            foreign.execute(
                "INSERT INTO transaction_recovery_probe(value) VALUES ('foreign')"
            )
            foreign.execute("COMMIT")
            assert foreign.execute(
                "SELECT value FROM transaction_recovery_probe"
            ).fetchall() == [("foreign",)]
        finally:
            foreign.close()

        db.close()
        assert not has_live_connection(db_path)
    finally:
        db.close()


def test_write_transaction_does_not_replay_ambiguous_post_commit_failure(
    tmp_path, monkeypatch
):
    """A commit error after durable work escapes without replaying the body."""
    db_path = tmp_path / "state.db"
    original_connect = hermes_state._connect_tracked_db

    def connect_with_commit_failure(*args, **kwargs):
        kwargs["factory"] = _CommitThenFailConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        hermes_state, "_connect_tracked_db", connect_with_commit_failure
    )
    db = SessionDB(db_path=db_path)
    try:
        db._conn.execute("CREATE TABLE transaction_recovery_commit_probe (value TEXT)")
        db._conn.commit()
        db._conn.fail_commit = True
        attempts = {"n": 0}

        with pytest.raises(sqlite3.OperationalError, match="ambiguous post-commit"):
            with db.write_transaction() as conn:
                attempts["n"] += 1
                conn.execute(
                    "INSERT INTO transaction_recovery_commit_probe(value) VALUES ('once')"
                )

        assert attempts["n"] == 1
        assert [row[0] for row in db._conn.execute(
            "SELECT value FROM transaction_recovery_commit_probe"
        ).fetchall()] == ["once"]
    finally:
        db.close()


def _execute_write_after_durable_commit_error(
    tmp_path, monkeypatch, error_type
):
    db_path = tmp_path / "state.db"
    original_connect = hermes_state._connect_tracked_db

    def connect_with_commit_failure(*args, **kwargs):
        kwargs["factory"] = _CommitOnceThenRaiseConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        hermes_state, "_connect_tracked_db", connect_with_commit_failure
    )
    db = SessionDB(db_path=db_path)
    try:
        db._conn.execute("CREATE TABLE transaction_recovery_replay_probe (value TEXT)")
        db._conn.commit()
        db._conn.commit_error_type = error_type
        db._conn.fail_next_commit = True
        attempts = {"n": 0}
        raised = None

        def body(conn):
            attempts["n"] += 1
            conn.execute(
                "INSERT INTO transaction_recovery_replay_probe(value) VALUES ('once')"
            )

        try:
            db._execute_write(body)
        except error_type as exc:
            raised = exc

        assert attempts["n"] == 1
        assert [row[0] for row in db._conn.execute(
            "SELECT value FROM transaction_recovery_replay_probe"
        ).fetchall()] == ["once"]
        assert isinstance(raised, error_type)
        assert str(raised) == "no more rows available after commit"
    finally:
        db.close()


def test_execute_write_does_not_replay_post_commit_interface_error(
    tmp_path, monkeypatch
):
    """An ambiguous InterfaceError after commit must preserve one durable write."""
    _execute_write_after_durable_commit_error(
        tmp_path, monkeypatch, sqlite3.InterfaceError
    )


def test_execute_write_does_not_replay_post_commit_database_error(
    tmp_path, monkeypatch
):
    """An ambiguous DatabaseError after commit must preserve one durable write."""
    _execute_write_after_durable_commit_error(
        tmp_path, monkeypatch, sqlite3.DatabaseError
    )


@pytest.mark.parametrize(
    "error_type",
    (sqlite3.DatabaseError, sqlite3.InterfaceError),
    ids=("database-error", "interface-error"),
)
def test_execute_write_does_not_replay_no_more_rows_callback_after_begin(
    tmp_path, monkeypatch, error_type
):
    """A post-BEGIN SQLite callback error rolls back without recovery or replay."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db._conn.execute("CREATE TABLE callback_recovery_probe (value TEXT)")
        db._conn.commit()
        attempts = {"n": 0}
        retries = {"n": 0}
        fts_recovery = {"n": 0}
        original_errors = []

        def fail_after_real_mutation(conn):
            attempts["n"] += 1
            conn.execute(
                "INSERT INTO callback_recovery_probe(value) VALUES ('rolled-back')"
            )
            error = error_type("no more rows available after callback mutation")
            original_errors.append(error)
            raise error

        def record_patience_retry(*_args):
            retries["n"] += 1
            return retries["n"] == 1

        def record_fts_recovery(*_args):
            fts_recovery["n"] += 1
            return False

        monkeypatch.setattr(db, "_sleep_before_write_retry", record_patience_retry)
        monkeypatch.setattr(db, "_try_runtime_fts_rebuild", record_fts_recovery)
        monkeypatch.setattr(db, "_enter_fts_fail_open", record_fts_recovery)

        with pytest.raises(error_type) as exc_info:
            db._execute_write(fail_after_real_mutation)

        assert attempts["n"] == 1
        assert exc_info.value is original_errors[0]
        assert retries["n"] == 0
        assert fts_recovery["n"] == 0
        assert db._conn.execute(
            "SELECT value FROM callback_recovery_probe"
        ).fetchall() == []
    finally:
        db.close()


def test_execute_write_does_not_retry_archive_callback_after_begin(
    tmp_path, monkeypatch
):
    """A lost compaction lease after mutation rolls back without replaying it."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(session_id="compaction", source="cli")
        db.append_message(
            session_id="compaction", role="user", content="original turn"
        )
        assert db.try_acquire_compression_lock("compaction", "compressor")

        original_insert = db._insert_message_rows
        callbacks = {"n": 0}
        retries = {"n": 0}

        def raise_after_real_archive_callback(conn, session_id, messages):
            callbacks["n"] += 1
            original_insert(conn, session_id, messages)
            raise hermes_state.SessionCompressionInProgressError(
                "compression lease changed after archive mutation"
            )

        def allow_only_one_patience_retry(*_args):
            retries["n"] += 1
            return retries["n"] == 1

        monkeypatch.setattr(
            db, "_insert_message_rows", raise_after_real_archive_callback
        )
        monkeypatch.setattr(db, "_sleep_before_write_retry", allow_only_one_patience_retry)

        with pytest.raises(
            hermes_state.SessionCompressionInProgressError,
            match="after archive mutation",
        ):
            db.archive_and_compact(
                "compaction",
                [{"role": "assistant", "content": "must not publish"}],
                lock_holder="compressor",
            )

        assert callbacks["n"] == 1
        assert retries["n"] == 0
        rows = db.get_messages("compaction", include_inactive=True)
        assert [(row["content"], row["active"], row["compacted"]) for row in rows] == [
            ("original turn", 1, 0)
        ]
    finally:
        db.close()


def test_init_schema_preserves_body_error_after_retired_rollback_connection(
    tmp_path, monkeypatch
):
    """Schema FK cleanup never masks a body error after rollback retires it."""
    db_path = tmp_path / "state.db"
    original_connect = hermes_state._connect_tracked_db

    def connect_with_rollback_failure(*args, **kwargs):
        kwargs["factory"] = _RollbackFailureConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        hermes_state, "_connect_tracked_db", connect_with_rollback_failure
    )
    db = SessionDB(db_path=db_path)
    try:
        owner = db._conn
        owner.fail_rollback = True
        primary_error = sqlite3.DatabaseError("forced init-schema body failure")

        def fail_guarded_schema_body(_cursor):
            raise primary_error

        monkeypatch.setattr(
            db, "_heal_session_turn_leases_legacy_epoch", fail_guarded_schema_body
        )
        caught = None
        try:
            db._init_schema()
        except BaseException as exc:
            caught = exc

        assert caught is primary_error
        assert isinstance(caught.__cause__, sqlite3.OperationalError)
        assert db._conn is None
        assert not has_live_connection(db_path)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            sqlite3.Connection.execute(owner, "PRAGMA foreign_keys")
    finally:
        db.close()


def test_init_schema_retires_live_connection_when_fk_restore_fails(
    tmp_path, monkeypatch
):
    """A failed FK restore cannot leave a reusable connection in the wrong mode."""
    db_path = tmp_path / "state.db"
    original_connect = hermes_state._connect_tracked_db

    def connect_with_fk_restore_failure(*args, **kwargs):
        kwargs["factory"] = _ForeignKeyRestoreFailureConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        hermes_state, "_connect_tracked_db", connect_with_fk_restore_failure
    )
    db = SessionDB(db_path=db_path)
    try:
        owner = db._conn
        primary_error = sqlite3.DatabaseError("forced init-schema body failure")

        def fail_guarded_schema_body(_cursor):
            raise primary_error

        monkeypatch.setattr(
            db, "_heal_session_turn_leases_legacy_epoch", fail_guarded_schema_body
        )
        owner.fail_foreign_key_restore = True
        caught = None
        try:
            db._init_schema()
        except BaseException as exc:
            caught = exc

        assert caught is primary_error
        assert isinstance(caught.__cause__, sqlite3.OperationalError)
        assert "foreign-key restoration" in str(caught.__cause__)
        assert db._conn is None
        assert not has_live_connection(db_path)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            sqlite3.Connection.execute(owner, "PRAGMA foreign_keys")
    finally:
        db.close()


@pytest.mark.parametrize(
    "marker_value",
    (
        "not-json",
        '{"owner_pid":0,"owner_pid_start":1.0,"nonce":"zero"}',
        '{"owner_pid":1,"owner_pid_start":"nan","nonce":"text"}',
    ),
)
def test_malformed_offline_rebuild_marker_refuses_ordinary_write(
    tmp_path, marker_value
):
    """Malformed durable rebuild markers fence normal writes before DML."""
    db = SessionDB(db_path=tmp_path / "state.db")
    marker_key = "_hermes_offline_rebuild_epoch_v1"
    try:
        db._conn.execute("CREATE TABLE transaction_recovery_marker_probe (value TEXT)")
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, ?)",
            (marker_key, marker_value),
        )
        db._conn.commit()

        with pytest.raises(hermes_state.SessionTurnLeaseLostError):
            db._execute_write(
                lambda conn: conn.execute(
                    "INSERT INTO transaction_recovery_marker_probe(value) VALUES ('blocked')"
                )
            )

        assert db._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (marker_key,)
        ).fetchone()[0] == marker_value
        assert db._conn.execute(
            "SELECT value FROM transaction_recovery_marker_probe"
        ).fetchall() == []
    finally:
        db.close()


def test_null_offline_rebuild_marker_refuses_ordinary_write(tmp_path):
    """A present NULL claim is not the ordinary no-owner state."""
    db = SessionDB(db_path=tmp_path / "state.db")
    marker_key = hermes_state._OFFLINE_REBUILD_EPOCH_KEY
    try:
        db._conn.execute("CREATE TABLE transaction_recovery_null_marker_probe (value TEXT)")
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, NULL)",
            (marker_key,),
        )
        db._conn.commit()

        with pytest.raises(hermes_state.SessionTurnLeaseLostError):
            db._execute_write(
                lambda conn: conn.execute(
                    "INSERT INTO transaction_recovery_null_marker_probe(value) "
                    "VALUES ('blocked')"
                )
            )

        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (marker_key,)
            ).fetchall()
        ] == [(None,)]
        assert list(
            db._conn.execute(
                "SELECT value FROM transaction_recovery_null_marker_probe"
            ).fetchall()
        ) == []
    finally:
        db.close()


def test_offline_rebuild_authority_allows_absence_and_exact_local_marker(tmp_path):
    """Ordinary writes and the exact non-NULL owner remain authorized."""
    db = SessionDB(db_path=tmp_path / "state.db")
    marker_key = hermes_state._OFFLINE_REBUILD_EPOCH_KEY
    try:
        db._conn.execute("CREATE TABLE transaction_recovery_authority_probe (value TEXT)")
        db._conn.commit()

        db._execute_write(
            lambda conn: conn.execute(
                "INSERT INTO transaction_recovery_authority_probe(value) VALUES ('ordinary')"
            )
        )

        with db.offline_rebuild(reason="test exact local authority"):
            local_marker = db._offline_rebuild_marker
            assert isinstance(local_marker, str)
            assert local_marker
            assert db._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (marker_key,)
            ).fetchone()[0] == local_marker
            db._execute_write(
                lambda conn: conn.execute(
                    "INSERT INTO transaction_recovery_authority_probe(value) VALUES ('local')"
                )
            )

        assert db._conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (marker_key,)
        ).fetchone() is None
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT value FROM transaction_recovery_authority_probe ORDER BY value"
            ).fetchall()
        ] == [("local",), ("ordinary",)]
    finally:
        db.close()


def test_automatic_incremental_merge_does_not_mask_a_committed_write_after_takeover(
    tmp_path, monkeypatch
):
    """Routine FTS maintenance must not turn a committed primary write into failure."""
    db = SessionDB(db_path=tmp_path / "state.db")
    marker_key = hermes_state._OFFLINE_REBUILD_EPOCH_KEY
    foreign_marker = b"\x00foreign-post-commit-merge-owner\xff"
    primary_session = "post-commit-merge-primary"
    merge_commands: list[str] = []
    takeover_complete = False
    try:
        assert db._fts_enabled is True
        db._write_count = db._FTS_MERGE_EVERY_N_WRITES - 1
        original_merge = db._merge_fts_incrementally

        def acquire_foreign_marker_before_merge_claim(**kwargs):
            nonlocal takeover_complete
            assert takeover_complete is False
            with sqlite3.connect(str(db.db_path), isolation_level=None) as writer:
                # Seeing the primary row from a new connection proves the
                # ordinary callback committed before maintenance starts.
                assert writer.execute(
                    "SELECT id FROM sessions WHERE id = ?", (primary_session,)
                ).fetchone() == (primary_session,)
                writer.execute(
                    "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                    (marker_key, sqlite3.Binary(foreign_marker)),
                )
            takeover_complete = True
            return original_merge(**kwargs)

        def record_fts_maintenance(sql: str) -> None:
            upper = sql.upper()
            if "MESSAGES_FTS" in upper and (
                "VALUES('MERGE'" in upper or "VALUES('USERMERGE'" in upper
            ):
                merge_commands.append(sql)

        monkeypatch.setattr(
            db, "_merge_fts_incrementally", acquire_foreign_marker_before_merge_claim
        )
        db._conn.set_trace_callback(record_fts_maintenance)

        assert db.create_session(primary_session, "cli") == primary_session

        assert takeover_complete is True
        assert db._conn.execute(
            "SELECT id FROM sessions WHERE id = ?", (primary_session,)
        ).fetchone()[0] == primary_session
        assert db._conn.execute(
            "SELECT CAST(value AS BLOB) FROM state_meta WHERE key = ?",
            (marker_key,),
        ).fetchone()[0] == foreign_marker
        assert merge_commands == []

        # Explicit maintenance keeps its refusal contract. Only the automatic
        # post-commit caller may demote this ownership loss.
        with pytest.raises(hermes_state.SessionTurnLeaseLostError):
            original_merge(max_pages=37, max_commands=1)
        assert merge_commands == []
    finally:
        db._conn.set_trace_callback(None)
        db.close()
