"""Producer-side SessionDB session/process authority lifecycle."""

from __future__ import annotations

import math
import sqlite3
import time

import pytest

from hermes_state import SessionDB


def test_create_session_issues_one_persistent_authority_generation(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session-a", "cli")

        authority = db.issue_session_process_authority("session-a")
        assert authority is not None
        assert authority["state_family"] == "sessiondb-v1"
        assert len(authority["state_db_id"]) == 64
        assert authority["session_id"] == "session-a"
        assert authority["session_generation"] == 1
        assert authority["status"] == "ISSUED"
        assert len(authority["authority_token"]) == 64

        # Duplicate creation preserves the original authority and emits no
        # second issuance for the same session generation.
        db.create_session("session-a", "cli")
        assert db.issue_session_process_authority("session-a") == authority
        events = db._conn.execute(
            "SELECT event_type, session_generation FROM session_process_authority_events "
            "WHERE session_id = ? ORDER BY id",
            ("session-a",),
        ).fetchall()
        assert [tuple(event) for event in events] == [("SESSION_ISSUED", 1)]
    finally:
        db.close()


def test_revocation_closes_current_authority_once_and_records_event(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session-a", "cli")

        assert db.revoke_session_process_authority("session-a") is True
        assert db.revoke_session_process_authority("session-a") is False
        assert db.issue_session_process_authority("session-a") is None
        events = db._conn.execute(
            "SELECT event_type FROM session_process_authority_events "
            "WHERE session_id = ? ORDER BY id",
            ("session-a",),
        ).fetchall()
        assert [event[0] for event in events] == [
            "SESSION_ISSUED",
            "SESSION_REVOKED",
        ]
    finally:
        db.close()


def test_reservation_is_high_entropy_short_lived_and_consumed_once(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session-a", "cli")
        authority = db.issue_session_process_authority("session-a")
        assert authority is not None

        reservation = db.reserve_session_process_authority(authority, ttl_seconds=5.0)
        assert reservation is not None
        assert len(reservation["reservation_id"]) >= 32
        assert len(reservation["reservation_token"]) >= 32
        assert reservation["expires_at"] > reservation["reserved_at"]
        assert db.consume_session_process_reservation(reservation) is True
        assert db.consume_session_process_reservation(reservation) is False

        events = db._conn.execute(
            "SELECT event_type FROM session_process_authority_events "
            "WHERE session_id = ? ORDER BY id",
            ("session-a",),
        ).fetchall()
        assert [event[0] for event in events] == [
            "SESSION_ISSUED",
            "PROCESS_RESERVATION",
            "PROCESS_BOUND",
        ]

        malformed_generation = dict(authority, session_generation=True)
        malformed_time = dict(authority, issued_at=math.nan)
        unknown_session = dict(authority, session_id="unknown")
        assert db.reserve_session_process_authority(malformed_generation) is None
        assert db.reserve_session_process_authority(malformed_time) is None
        assert db.reserve_session_process_authority(unknown_session) is None
    finally:
        db.close()


def test_raw_session_writers_close_once_and_reopen_a_new_generation(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db._conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("raw-session", "cli", time.time()),
        )
        first = db.issue_session_process_authority("raw-session")
        assert first is not None
        assert first["session_generation"] == 1

        db._conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
            (time.time(), "agent_close", "raw-session"),
        )
        db._conn.execute(
            "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
            (time.time(), "duplicate_close", "raw-session"),
        )
        assert db.issue_session_process_authority("raw-session") is None

        db._conn.execute(
            "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
            ("raw-session",),
        )
        second = db.issue_session_process_authority("raw-session")
        assert second is not None
        assert second["session_generation"] == 2
        assert second["authority_token"] != first["authority_token"]
        events = db._conn.execute(
            "SELECT event_type, session_generation FROM session_process_authority_events "
            "WHERE session_id = ? ORDER BY id",
            ("raw-session",),
        ).fetchall()
        assert [tuple(event) for event in events] == [
            ("SESSION_ISSUED", 1),
            ("SESSION_CLOSED", 1),
            ("SESSION_ISSUED", 2),
        ]
    finally:
        db.close()


def test_expired_stale_closed_and_revoked_authority_is_rejected(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session-a", "cli")
        first = db.issue_session_process_authority("session-a")
        assert first is not None
        expired = db.reserve_session_process_authority(first, ttl_seconds=0.001)
        assert expired is not None
        time.sleep(0.01)
        assert db.consume_session_process_reservation(expired) is False

        stale = db.reserve_session_process_authority(first, ttl_seconds=5.0)
        assert stale is not None
        db.end_session("session-a", "agent_close")
        assert db.reserve_session_process_authority(first) is None
        assert db.consume_session_process_reservation(stale) is False

        db.reopen_session("session-a")
        second = db.issue_session_process_authority("session-a")
        assert second is not None
        assert second["session_generation"] == 2
        assert db.revoke_session_process_authority("session-a") is True
        assert db.reserve_session_process_authority(second) is None
    finally:
        db.close()


def test_process_events_are_append_only_and_cover_terminal_transitions(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session-a", "cli")
        authority = db.issue_session_process_authority("session-a")
        assert authority is not None
        bound = db.reserve_session_process_authority(authority, ttl_seconds=5.0)
        assert bound is not None
        assert db.consume_session_process_reservation(bound) is True
        db._conn.execute(
            "UPDATE session_process_reservations SET status = 'TERMINAL' "
            "WHERE reservation_id = ?",
            (bound["reservation_id"],),
        )

        aborted = db.reserve_session_process_authority(authority, ttl_seconds=5.0)
        assert aborted is not None
        db._conn.execute(
            "UPDATE session_process_reservations SET status = 'ABORTED' "
            "WHERE reservation_id = ?",
            (aborted["reservation_id"],),
        )
        events = db._conn.execute(
            "SELECT event_type FROM session_process_authority_events "
            "WHERE session_id = ? ORDER BY id",
            ("session-a",),
        ).fetchall()
        assert [event[0] for event in events] == [
            "SESSION_ISSUED",
            "PROCESS_RESERVATION",
            "PROCESS_BOUND",
            "PROCESS_TERMINAL",
            "PROCESS_RESERVATION",
            "PROCESS_ABORTED",
        ]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db._conn.execute(
                "DELETE FROM session_process_authority_events WHERE session_id = ?",
                ("session-a",),
            )
    finally:
        db.close()


def test_raw_sql_cannot_regress_session_authority_generation(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.create_session("session-a", "cli")
        with pytest.raises(sqlite3.IntegrityError, match="monotonic"):
            db._conn.execute(
                "UPDATE sessions SET session_generation = 0 WHERE id = ?",
                ("session-a",),
            )
        authority = db.issue_session_process_authority("session-a")
        assert authority is not None
        assert authority["session_generation"] == 1
    finally:
        db.close()


def test_authority_database_identity_is_immutable(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db._conn.execute(
                "UPDATE state_meta SET value = ? WHERE key = ?",
                ("0" * 64, "session_process_state_db_id"),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db._conn.execute(
                "DELETE FROM state_meta WHERE key = ?",
                ("session_process_state_family",),
            )
    finally:
        db.close()
