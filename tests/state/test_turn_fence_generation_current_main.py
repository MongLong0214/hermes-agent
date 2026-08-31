"""Behavioral coverage for the v27 state-store generation fence."""

import sqlite3

import pytest

import hermes_state
from hermes_state import SCHEMA_SQL, SessionDB
from hermes_state_common import TURN_FENCE_GENERATION, TURN_FENCE_GOVERNED_TABLES


class _FailAfterOneTurnFenceTriggerCursor(sqlite3.Cursor):
    trigger_creations = 0

    def execute(self, sql, parameters=()):
        if sql.lstrip().upper().startswith("CREATE TRIGGER TURN_FENCE_"):
            type(self).trigger_creations += 1
            if type(self).trigger_creations == 2:
                raise sqlite3.OperationalError("forced v27 trigger failure")
        return super().execute(sql, parameters)


class _FailAfterOneTurnFenceTriggerConnection(
    hermes_state._SerializedConnectionMixin, sqlite3.Connection
):
    def cursor(self, factory=None):
        return super().cursor(factory or _FailAfterOneTurnFenceTriggerCursor)


def test_turn_fence_generation_v27_trigger_failure_rolls_back_before_schema_publication(
    tmp_path, monkeypatch
):
    """A partial v27 trigger install leaves the complete v26 state intact."""
    db_path = tmp_path / "state.db"
    seed = sqlite3.connect(str(db_path))
    seed.executescript(SCHEMA_SQL)
    seed.execute("INSERT INTO schema_version (version) VALUES (26)")
    seed.commit()
    seed.close()

    _FailAfterOneTurnFenceTriggerCursor.trigger_creations = 0
    real_connect = hermes_state._connect_tracked_db

    def connect_with_v27_failure(*args, **kwargs):
        kwargs["factory"] = _FailAfterOneTurnFenceTriggerConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", connect_with_v27_failure)

    with pytest.raises(sqlite3.OperationalError, match="forced v27 trigger failure"):
        SessionDB(db_path=db_path)

    check = sqlite3.connect(str(db_path))
    try:
        assert check.execute("SELECT version FROM schema_version").fetchall() == [(26,)]
        assert check.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'trigger' AND name LIKE 'turn_fence_%'"
        ).fetchall() == []
    finally:
        check.close()


def _governed_write_cases(conn):
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('messages-parent', 'cli', 1)"
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES ('messages-parent', 'user', 'before', 1)"
    )
    message_id = conn.execute("SELECT id FROM messages").fetchone()[0]
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('sessions-update', 'cli', 1)"
    )
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('sessions-delete', 'cli', 1)"
    )
    conn.execute("INSERT INTO system_prompts (hash, prompt) VALUES ('prompts-update', 'before')")
    conn.execute("INSERT INTO system_prompts (hash, prompt) VALUES ('prompts-delete', 'before')")
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('usage-parent', 'cli', 1)"
    )
    conn.execute(
        "INSERT INTO session_model_usage (session_id, model) VALUES ('usage-parent', 'usage-update')"
    )
    conn.execute(
        "INSERT INTO session_model_usage (session_id, model) VALUES ('usage-parent', 'usage-delete')"
    )
    conn.execute(
        "INSERT INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) "
        "VALUES ('leases-update', 'before', 1, 2)"
    )
    conn.execute(
        "INSERT INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) "
        "VALUES ('leases-delete', 'before', 1, 2)"
    )
    conn.execute(
        "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) "
        "VALUES ('locks-update', 'before', 1, 2)"
    )
    conn.execute(
        "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) "
        "VALUES ('locks-delete', 'before', 1, 2)"
    )
    conn.execute(
        "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
        "VALUES ('', 'routing-update', '{}', 1)"
    )
    conn.execute(
        "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
        "VALUES ('', 'routing-delete', '{}', 1)"
    )
    conn.execute(
        "INSERT INTO async_delegations "
        "(delegation_id, origin_session, state, dispatched_at, updated_at) "
        "VALUES ('async-update', 'origin', 'pending', 1, 1)"
    )
    conn.execute(
        "INSERT INTO async_delegations "
        "(delegation_id, origin_session, state, dispatched_at, updated_at) "
        "VALUES ('async-delete', 'origin', 'pending', 1, 1)"
    )
    for session_id in ("authority-insert", "authority-update", "authority-delete"):
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', 1)",
            (session_id,),
        )
    conn.execute(
        "DELETE FROM session_process_authorities WHERE session_id = 'authority-insert'"
    )
    state_db_id = conn.execute(
        "SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'"
    ).fetchone()[0]
    for reservation_id in ("reservation-update", "reservation-delete"):
        conn.execute(
            "INSERT INTO session_process_reservations "
            "(reservation_id, reservation_token_sha256, session_id, session_generation, "
            "state_db_id, state_family, status, reserved_at, expires_at) "
            "VALUES (?, ?, 'authority-update', 1, ?, 'sessiondb-v1', 'RESERVED', 1, 2)",
            (reservation_id, "a" * 64, state_db_id),
        )
    return (
        ("messages", "INSERT", "INSERT INTO messages (session_id, role, content, timestamp) VALUES ('messages-parent', 'user', 'insert', 2)", "SELECT COUNT(*) FROM messages WHERE content = 'insert'"),
        ("messages", "UPDATE", f"UPDATE messages SET content = 'after' WHERE id = {message_id}", f"SELECT content FROM messages WHERE id = {message_id}"),
        ("messages", "DELETE", f"DELETE FROM messages WHERE id = {message_id}", f"SELECT COUNT(*) FROM messages WHERE id = {message_id}"),
        ("sessions", "INSERT", "INSERT INTO sessions (id, source, started_at) VALUES ('sessions-insert', 'cli', 1)", "SELECT COUNT(*) FROM sessions WHERE id = 'sessions-insert'"),
        ("sessions", "UPDATE", "UPDATE sessions SET source = 'other' WHERE id = 'sessions-update'", "SELECT source FROM sessions WHERE id = 'sessions-update'"),
        ("sessions", "DELETE", "DELETE FROM sessions WHERE id = 'sessions-delete'", "SELECT COUNT(*) FROM sessions WHERE id = 'sessions-delete'"),
        ("system_prompts", "INSERT", "INSERT INTO system_prompts (hash, prompt) VALUES ('prompts-insert', 'insert')", "SELECT COUNT(*) FROM system_prompts WHERE hash = 'prompts-insert'"),
        ("system_prompts", "UPDATE", "UPDATE system_prompts SET prompt = 'after' WHERE hash = 'prompts-update'", "SELECT prompt FROM system_prompts WHERE hash = 'prompts-update'"),
        ("system_prompts", "DELETE", "DELETE FROM system_prompts WHERE hash = 'prompts-delete'", "SELECT COUNT(*) FROM system_prompts WHERE hash = 'prompts-delete'"),
        ("session_model_usage", "INSERT", "INSERT INTO session_model_usage (session_id, model) VALUES ('usage-parent', 'usage-insert')", "SELECT COUNT(*) FROM session_model_usage WHERE model = 'usage-insert'"),
        ("session_model_usage", "UPDATE", "UPDATE session_model_usage SET api_call_count = 7 WHERE model = 'usage-update'", "SELECT api_call_count FROM session_model_usage WHERE model = 'usage-update'"),
        ("session_model_usage", "DELETE", "DELETE FROM session_model_usage WHERE model = 'usage-delete'", "SELECT COUNT(*) FROM session_model_usage WHERE model = 'usage-delete'"),
        ("session_turn_leases", "INSERT", "INSERT INTO session_turn_leases (conversation_id, holder, acquired_at, expires_at) VALUES ('leases-insert', 'holder', 1, 2)", "SELECT COUNT(*) FROM session_turn_leases WHERE conversation_id = 'leases-insert'"),
        ("session_turn_leases", "UPDATE", "UPDATE session_turn_leases SET holder = 'after' WHERE conversation_id = 'leases-update'", "SELECT holder FROM session_turn_leases WHERE conversation_id = 'leases-update'"),
        ("session_turn_leases", "DELETE", "DELETE FROM session_turn_leases WHERE conversation_id = 'leases-delete'", "SELECT COUNT(*) FROM session_turn_leases WHERE conversation_id = 'leases-delete'"),
        ("compression_locks", "INSERT", "INSERT INTO compression_locks (session_id, holder, acquired_at, expires_at) VALUES ('locks-insert', 'holder', 1, 2)", "SELECT COUNT(*) FROM compression_locks WHERE session_id = 'locks-insert'"),
        ("compression_locks", "UPDATE", "UPDATE compression_locks SET holder = 'after' WHERE session_id = 'locks-update'", "SELECT holder FROM compression_locks WHERE session_id = 'locks-update'"),
        ("compression_locks", "DELETE", "DELETE FROM compression_locks WHERE session_id = 'locks-delete'", "SELECT COUNT(*) FROM compression_locks WHERE session_id = 'locks-delete'"),
        ("gateway_routing", "INSERT", "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) VALUES ('', 'routing-insert', '{}', 1)", "SELECT COUNT(*) FROM gateway_routing WHERE session_key = 'routing-insert'"),
        ("gateway_routing", "UPDATE", "UPDATE gateway_routing SET entry_json = '{\"after\": true}' WHERE session_key = 'routing-update'", "SELECT entry_json FROM gateway_routing WHERE session_key = 'routing-update'"),
        ("gateway_routing", "DELETE", "DELETE FROM gateway_routing WHERE session_key = 'routing-delete'", "SELECT COUNT(*) FROM gateway_routing WHERE session_key = 'routing-delete'"),
        ("async_delegations", "INSERT", "INSERT INTO async_delegations (delegation_id, origin_session, state, dispatched_at, updated_at) VALUES ('async-insert', 'origin', 'pending', 1, 1)", "SELECT COUNT(*) FROM async_delegations WHERE delegation_id = 'async-insert'"),
        ("async_delegations", "UPDATE", "UPDATE async_delegations SET state = 'done' WHERE delegation_id = 'async-update'", "SELECT state FROM async_delegations WHERE delegation_id = 'async-update'"),
        ("async_delegations", "DELETE", "DELETE FROM async_delegations WHERE delegation_id = 'async-delete'", "SELECT COUNT(*) FROM async_delegations WHERE delegation_id = 'async-delete'"),
        ("session_process_authorities", "INSERT", f"INSERT INTO session_process_authorities (session_id, session_generation, state_db_id, state_family, authority_token, status, issued_at) VALUES ('authority-insert', 1, '{state_db_id}', 'sessiondb-v1', '{'b' * 64}', 'ISSUED', 1)", "SELECT COUNT(*) FROM session_process_authorities WHERE session_id = 'authority-insert'"),
        ("session_process_authorities", "UPDATE", f"UPDATE session_process_authorities SET authority_token = '{'b' * 64}' WHERE session_id = 'authority-update'", "SELECT authority_token FROM session_process_authorities WHERE session_id = 'authority-update'"),
        ("session_process_authorities", "DELETE", "DELETE FROM session_process_authorities WHERE session_id = 'authority-delete'", "SELECT COUNT(*) FROM session_process_authorities WHERE session_id = 'authority-delete'"),
        ("session_process_reservations", "INSERT", f"INSERT INTO session_process_reservations (reservation_id, reservation_token_sha256, session_id, session_generation, state_db_id, state_family, status, reserved_at, expires_at) VALUES ('reservation-insert', '{'c' * 64}', 'authority-update', 1, '{state_db_id}', 'sessiondb-v1', 'RESERVED', 1, 2)", "SELECT COUNT(*) FROM session_process_reservations WHERE reservation_id = 'reservation-insert'"),
        ("session_process_reservations", "UPDATE", "UPDATE session_process_reservations SET status = 'BOUND' WHERE reservation_id = 'reservation-update'", "SELECT status FROM session_process_reservations WHERE reservation_id = 'reservation-update'"),
        ("session_process_reservations", "DELETE", "DELETE FROM session_process_reservations WHERE reservation_id = 'reservation-delete'", "SELECT COUNT(*) FROM session_process_reservations WHERE reservation_id = 'reservation-delete'"),
    )


def test_turn_fence_generation_unregistered_connection_cannot_mutate_any_governed_operation(
    tmp_path,
):
    """Every governed INSERT, UPDATE, and DELETE fails before mutation."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        cases = _governed_write_cases(db._conn)
        db._conn.commit()
        assert {table for table, _operation, _sql, _probe in cases} == set(
            TURN_FENCE_GOVERNED_TABLES
        )
        assert len(cases) == len(TURN_FENCE_GOVERNED_TABLES) * 3

        raw = sqlite3.connect(str(db.db_path))
        try:
            for _table, _operation, sql, probe in cases:
                before = raw.execute(probe).fetchall()
                with pytest.raises(sqlite3.DatabaseError):
                    raw.execute(sql)
                assert raw.execute(probe).fetchall() == before
        finally:
            raw.close()
    finally:
        db.close()


def test_turn_fence_generation_wrong_or_throwing_function_fails_before_mutation(
    tmp_path,
):
    """The trigger requires the exact integer generation, not mere presence."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        for value in (
            TURN_FENCE_GENERATION - 1,
            "27",
            b"27",
            27.0,
            None,
            True,
        ):
            raw = sqlite3.connect(str(db.db_path))
            try:
                raw.create_function(
                    "hermes_turn_fence_generation", 0, lambda value=value: value
                )
                with pytest.raises(sqlite3.DatabaseError):
                    raw.execute(
                        "INSERT INTO gateway_routing "
                        "(scope, session_key, entry_json, updated_at) "
                        "VALUES ('', ?, '{}', 1)",
                        (f"wrong-{type(value).__name__}",),
                    )
                assert raw.execute(
                    "SELECT COUNT(*) FROM gateway_routing "
                    "WHERE session_key = ?",
                    (f"wrong-{type(value).__name__}",),
                ).fetchone() == (0,)
            finally:
                raw.close()

        raw = sqlite3.connect(str(db.db_path))
        try:
            def raises_generation():
                raise RuntimeError("generation unavailable")

            raw.create_function("hermes_turn_fence_generation", 0, raises_generation)
            with pytest.raises(sqlite3.DatabaseError):
                raw.execute(
                    "INSERT INTO gateway_routing "
                    "(scope, session_key, entry_json, updated_at) "
                    "VALUES ('', 'throws', '{}', 1)"
                )
            assert raw.execute(
                "SELECT COUNT(*) FROM gateway_routing WHERE session_key = 'throws'"
            ).fetchone() == (0,)
        finally:
            raw.close()
    finally:
        db.close()


_LEGACY_V27_GOVERNED_TABLES = tuple(
    table
    for table in TURN_FENCE_GOVERNED_TABLES
    if table not in {"session_process_authorities", "session_process_reservations"}
)
_LEGACY_TURN_FENCE_OPERATIONS = ("INSERT", "UPDATE", "DELETE")


def _create_populated_v27_database(db_path, session_ids):
    """Build the parent-v27 storage shape, including its generation-27 fence."""
    initial = SessionDB(db_path)
    initial.close()

    legacy = sqlite3.connect(str(db_path))
    try:
        for name, in legacy.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            "AND (name LIKE 'turn_fence_%' OR name LIKE 'session_process_%')"
        ):
            legacy.execute(f'DROP TRIGGER "{name}"')
        legacy.execute("PRAGMA foreign_keys = OFF")
        legacy.execute("DROP TABLE session_process_reservations")
        legacy.execute("DROP TABLE session_process_authority_events")
        legacy.execute("DROP TABLE session_process_authorities")
        legacy.execute("ALTER TABLE sessions DROP COLUMN session_generation")
        legacy.execute(
            "DELETE FROM state_meta WHERE key IN "
            "('session_process_state_db_id', 'session_process_state_family')"
        )
        legacy.execute("DELETE FROM schema_version")
        legacy.execute("INSERT INTO schema_version (version) VALUES (27)")
        legacy.create_function("hermes_turn_fence_generation", 0, lambda: 27)
        for session_id in session_ids:
            legacy.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', 1)",
                (session_id,),
            )
        for table in _LEGACY_V27_GOVERNED_TABLES:
            for operation in _LEGACY_TURN_FENCE_OPERATIONS:
                legacy.execute(
                    f"CREATE TRIGGER turn_fence_{table}_{operation.lower()} "
                    f"BEFORE {operation} ON {table} BEGIN "
                    "SELECT CASE "
                    "WHEN typeof(hermes_turn_fence_generation()) != 'integer' "
                    "OR hermes_turn_fence_generation() != 27 "
                    "THEN RAISE(ABORT, 'state DB generation incompatible') "
                    "END; END"
                )
        legacy.commit()
    finally:
        legacy.close()


@pytest.mark.parametrize("session_ids", [(), ("v27-a", "v27-b")])
def test_v28_migrates_empty_and_populated_v27_databases_before_authority_backfill(
    tmp_path, session_ids
):
    """The v28 fence is live before it backfills authority over v27 sessions."""
    db_path = tmp_path / "state.db"
    _create_populated_v27_database(db_path, session_ids)

    db = SessionDB(db_path)
    try:
        assert [
            tuple(row)
            for row in db._conn.execute("SELECT version FROM schema_version")
        ] == [(28,)]
        expected_sessions = [(session_id, 1) for session_id in session_ids]
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT id, session_generation FROM sessions ORDER BY id"
            )
        ] == expected_sessions
        expected_authorities = [(session_id, 1, "ISSUED") for session_id in session_ids]
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT session_id, session_generation, status "
                "FROM session_process_authorities ORDER BY session_id"
            )
        ] == expected_authorities
        assert [
            tuple(row)
            for row in db._conn.execute(
                "SELECT session_id, session_generation, event_type "
                "FROM session_process_authority_events ORDER BY session_id, id"
            )
        ] == [(session_id, 1, "SESSION_ISSUED") for session_id in session_ids]
    finally:
        db.close()

    reopened = SessionDB(db_path)
    try:
        assert tuple(
            reopened._conn.execute(
                "SELECT COUNT(*) FROM session_process_authorities"
            ).fetchone()
        ) == (len(session_ids),)
        assert tuple(
            reopened._conn.execute(
                "SELECT COUNT(*) FROM session_process_authority_events"
            ).fetchone()
        ) == (len(session_ids),)
    finally:
        reopened.close()

    for label, generation in (("unregistered", None), ("old-v27", 27)):
        external = sqlite3.connect(str(db_path))
        try:
            if generation is not None:
                external.create_function(
                    "hermes_turn_fence_generation", 0, lambda: generation
                )
            with pytest.raises(sqlite3.DatabaseError):
                external.execute(
                    "INSERT INTO gateway_routing "
                    "(scope, session_key, entry_json, updated_at) "
                    "VALUES ('', ?, '{}', 1)",
                    (label,),
                )
            assert external.execute(
                "SELECT COUNT(*) FROM gateway_routing WHERE session_key = ?", (label,)
            ).fetchone() == (0,)
        finally:
            external.close()


def test_v28_fence_install_failure_rolls_back_a_populated_v27_upgrade(tmp_path, monkeypatch):
    """A failed v28 fence install leaves no authority backfill or v28 DDL behind."""
    db_path = tmp_path / "state.db"
    _create_populated_v27_database(db_path, ("v27-a", "v27-b"))
    _FailAfterOneTurnFenceTriggerCursor.trigger_creations = 0
    real_connect = hermes_state._connect_tracked_db

    def connect_with_v28_failure(*args, **kwargs):
        kwargs["factory"] = _FailAfterOneTurnFenceTriggerConnection
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", connect_with_v28_failure)

    with pytest.raises(sqlite3.OperationalError, match="forced v27 trigger failure"):
        SessionDB(db_path=db_path)

    check = sqlite3.connect(str(db_path))
    try:
        assert check.execute("SELECT version FROM schema_version").fetchall() == [(27,)]
        assert {
            row[1] for row in check.execute("PRAGMA table_info(sessions)").fetchall()
        }.isdisjoint({"session_generation"})
        assert check.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE 'session_process_%'"
        ).fetchall() == []
        assert check.execute(
            "SELECT key FROM state_meta WHERE key LIKE 'session_process_state_%'"
        ).fetchall() == []
        assert [
            tuple(row)
            for row in check.execute("SELECT id, source, started_at FROM sessions ORDER BY id")
        ] == [("v27-a", "cli", 1.0), ("v27-b", "cli", 1.0)]
    finally:
        check.close()
