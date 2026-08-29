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
