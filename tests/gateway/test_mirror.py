"""Tests for gateway/mirror.py — session mirroring."""

import json
from unittest.mock import patch, MagicMock

import gateway.mirror as mirror_mod
from gateway.mirror import (
    mirror_to_session,
    _find_session_id,
)


def _setup_sessions(tmp_path, sessions_data):
    """Helper to write a fake sessions.json and patch module-level paths."""
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    index_file = sessions_dir / "sessions.json"
    index_file.write_text(json.dumps(sessions_data))
    return sessions_dir, index_file


class TestFindSessionId:
    def test_finds_matching_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "agent:main:telegram:dm": {
                "session_id": "sess_abc",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            }
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_abc"

    def test_returns_most_recent(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "old": {
                "session_id": "sess_old",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "new": {
                "session_id": "sess_new",
                "origin": {"platform": "telegram", "chat_id": "12345"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "12345")

        assert result == "sess_new"

    def test_thread_id_disambiguates_same_chat(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "topic_a": {
                "session_id": "sess_topic_a",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "10"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "topic_b": {
                "session_id": "sess_topic_b",
                "origin": {"platform": "telegram", "chat_id": "-1001", "thread_id": "11"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = _find_session_id("telegram", "-1001", thread_id="10")

        assert result == "sess_topic_a"


class TestMirrorToSession:


    def test_successful_mirror_uses_user_id_for_group_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {
            "alice": {
                "session_id": "sess_alice",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "alice"},
                "updated_at": "2026-01-01T00:00:00",
            },
            "bob": {
                "session_id": "sess_bob",
                "origin": {"platform": "telegram", "chat_id": "-1001", "user_id": "bob"},
                "updated_at": "2026-02-01T00:00:00",
            },
        })

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file), \
             patch("gateway.mirror._append_to_sqlite", return_value=True) as mock_sqlite:
            result = mirror_to_session(
                "telegram",
                "-1001",
                "Hello group!",
                source_label="cli",
                user_id="alice",
            )

        assert result is True
        mock_sqlite.assert_called_once()
        assert mock_sqlite.call_args[0][0] == "sess_alice"

    def test_no_matching_session(self, tmp_path):
        sessions_dir, index_file = _setup_sessions(tmp_path, {})

        with patch.object(mirror_mod, "_SESSIONS_DIR", sessions_dir), \
             patch.object(mirror_mod, "_SESSIONS_INDEX", index_file):
            result = mirror_to_session("telegram", "99999", "Hello!")

        assert result is False


class TestAppendToSqlite:
    def test_connection_is_closed_after_use(self, tmp_path):
        """Verify _append_to_sqlite closes the SessionDB connection."""
        from gateway.mirror import _append_to_sqlite
        mock_db = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            _append_to_sqlite("sess_1", {"role": "assistant", "content": "hello"})

        mock_db.append_message.assert_called_once()
        mock_db.close.assert_called_once()


class TestMirrorRefusedWrite:
    """R-6: a refused mirror write must not be reported as mirrored."""

    def test_refused_write_does_not_report_mirrored_true(self, tmp_path, monkeypatch):
        """A refused/failed SQLite append must not come back as ``True``.

        ``_append_to_sqlite`` (gateway/mirror.py) wraps its
        ``SessionDB.append_message`` call in a bare ``except Exception``
        that only logs at DEBUG and never signals failure to its caller.
        ``mirror_to_session`` therefore falls through to its unconditional
        ``return True`` (line 96) even when the write was refused and the
        message was never persisted anywhere — the caller
        (``tools/send_message_tool.py:521`` sets ``result["mirrored"] =
        True`` straight from this return value, which is what the model
        sees) is told the mirror succeeded when it did not.

        This drives the REAL public entry point ``mirror_to_session`` (not
        ``_append_to_sqlite`` directly) against a REAL ``SessionDB``
        backed by a tmp_path SQLite file. Only the deepest dependency,
        ``SessionDB.append_message`` itself, is patched to raise —
        standing in for a real refusal a transcript write can hit when it
        does not hold the lock/lease it needs
        (``CompressionSessionClosedError``/``SessionTurnLeaseLostError``
        are the real exception classes ``hermes_state`` raises for exactly
        this "refused, not owned by the caller" case).
        """
        import hermes_state

        monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")

        def _refuse(self, *args, **kwargs):
            raise hermes_state.CompressionSessionClosedError("sess_refused")

        monkeypatch.setattr(hermes_state.SessionDB, "append_message", _refuse)

        result = mirror_to_session(
            "telegram",
            "12345",
            "This must not silently vanish",
            source_label="cli",
            session_id="sess_refused",
        )

        assert result is not True, (
            "mirror_to_session reported the mirror as successful for a "
            "write that was refused and never persisted — the message "
            f"was silently dropped instead of the caller being told. "
            f"got: {result!r}"
        )


class TestMirrorCloseFailureAfterCommit:
    """A close() failure after a successful append must not overturn the commit.

    ``_append_to_sqlite`` sets ``append_committed = True`` right after
    ``db.append_message`` returns, but its ``finally`` block called
    ``db.close()`` unguarded. If ``close()`` itself raised, that exception
    propagated out of ``_append_to_sqlite`` entirely — discarding the
    already-decided ``True`` — and was caught by ``mirror_to_session``'s
    outer ``except Exception``, which reported ``False``. That is the
    same class of lie as the original defect with the sign flipped: the
    row genuinely committed, but the caller is told it did not. Two of
    ``mirror_to_session``'s callers (``cron/scheduler.py``'s thread and
    in-channel seeders) log "did NOT land" on a falsy result — they would
    say that about a row that landed.
    """

    def test_close_failure_after_committed_append_still_reports_true(self):
        mock_db = MagicMock()
        mock_db.close.side_effect = RuntimeError("close boom")

        with patch("hermes_state.SessionDB", return_value=mock_db):
            result = mirror_to_session(
                "telegram",
                "55555",
                "Row commits even though close() blows up",
                source_label="cli",
                session_id="sess_committed",
            )

        assert result is True, (
            "a close() failure after a successful append flipped a "
            "committed write into a reported failure — the row IS on "
            f"disk, so the truthful answer is True. got: {result!r}"
        )
        mock_db.append_message.assert_called_once()

