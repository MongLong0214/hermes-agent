"""Regression tests: `hermes sessions` error paths return non-zero (SES-04).

Before this, delete/rename not-found, prune bad-arg, blank rename, and import
of a missing file all printed an error and returned exit 0 — a scripting/CI
hazard (a script pinning a bad id failed loudly via `pin` but deleting a bad
id "succeeded" silently). The subcommand dispatcher already maps an int
handler return to the process exit code; these tests pin the returns.
"""

from argparse import Namespace
import json
import sqlite3

import pytest

import hermes_cli.sessions_cmd as sc


def _args(action, **kw):
    base = dict(
        sessions_action=action,
        session_id=None, title=None, yes=True, source=None, path=None,
        from_source=None, dry_run=False, older_than=None, newer_than=None,
        before=None, after=None, limit=50,
    )
    base.update(kw)
    return Namespace(**base)


@pytest.fixture
def foreign_import_home(tmp_path, monkeypatch, _hermetic_environment):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    import hermes_state

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    token = set_hermes_home_override(str(tmp_path))
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def test_delete_missing_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    SessionDB(tmp_path / "state.db")  # initialize an empty store
    rc = sc.cmd_sessions(_args("delete", session_id="nope_xyz"))
    assert rc == 1
    assert "not found" in capsys.readouterr().out.lower()


def test_rename_missing_returns_1(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    SessionDB(tmp_path / "state.db")
    rc = sc.cmd_sessions(_args("rename", session_id="nope_xyz", title=["New"]))
    assert rc == 1


def test_import_missing_file_returns_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rc = sc.cmd_sessions(_args("import", path=str(tmp_path / "nope.jsonl")))
    assert rc == 1
    assert "file not found" in capsys.readouterr().out.lower()


def test_foreign_import_rolls_back_before_cli_failure(
    tmp_path, foreign_import_home, capsys
):
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.close()
    foreign_path = tmp_path / "claude-session.jsonl"
    foreign_path.write_text(
        "\n".join(
            json.dumps(line)
            for line in (
                {
                    "type": "user",
                    "sessionId": "foreign-claude-session",
                    "cwd": "/tmp/foreign-project",
                    "message": {"role": "user", "content": "Hello from Claude."},
                },
                {
                    "type": "assistant",
                    "sessionId": "foreign-claude-session",
                    "cwd": "/tmp/foreign-project",
                    "message": {"role": "assistant", "content": "Hello from Hermes."},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TRIGGER foreign_import_intrude_after_user
            AFTER INSERT ON messages
            WHEN NEW.role = 'user' AND EXISTS (
                SELECT 1 FROM sessions
                WHERE id = NEW.session_id AND source = 'claude-code'
            )
            BEGIN
                UPDATE session_turn_leases
                SET holder = 'deterministic-intruder'
                WHERE conversation_id = NEW.session_id;
            END;

            CREATE TRIGGER foreign_import_fail_before_assistant
            BEFORE INSERT ON messages
            WHEN NEW.role = 'assistant' AND EXISTS (
                SELECT 1 FROM sessions
                WHERE id = NEW.session_id AND source = 'claude-code'
            )
            BEGIN
                SELECT RAISE(ABORT, 'forced foreign import failure');
            END;
            """
        )

    rc = sc.cmd_sessions(
        _args("import", path=str(foreign_path), from_source="claude")
    )
    output = capsys.readouterr().out.lower()
    assert rc == 1
    assert "nothing was imported" in output
    assert "✓ imported" not in output

    with sqlite3.connect(db_path) as conn:
        imported_sessions = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE source = 'claude-code'"
        ).fetchone()[0]
        imported_messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert imported_sessions == 0
    assert imported_messages == 0


def _write_minimal_claude_session(tmp_path):
    foreign_path = tmp_path / "claude-session.jsonl"
    foreign_path.write_text(
        "\n".join(
            json.dumps(line)
            for line in (
                {
                    "type": "user",
                    "sessionId": "foreign-claude-session",
                    "cwd": "/tmp/foreign-project",
                    "message": {"role": "user", "content": "Hello from Claude."},
                },
                {
                    "type": "assistant",
                    "sessionId": "foreign-claude-session",
                    "cwd": "/tmp/foreign-project",
                    "message": {"role": "assistant", "content": "Hello from Hermes."},
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return foreign_path


def _initialize_temp_state_db(tmp_path):
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    db.close()
    return db_path


def _install_foreign_import_failure_triggers(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TRIGGER foreign_import_intrude_after_user
            AFTER INSERT ON messages
            WHEN NEW.role = 'user' AND EXISTS (
                SELECT 1 FROM sessions
                WHERE id = NEW.session_id AND source = 'claude-code'
            )
            BEGIN
                UPDATE session_turn_leases
                SET holder = 'deterministic-intruder'
                WHERE conversation_id = NEW.session_id;
            END;

            CREATE TRIGGER foreign_import_fail_before_assistant
            BEFORE INSERT ON messages
            WHEN NEW.role = 'assistant' AND EXISTS (
                SELECT 1 FROM sessions
                WHERE id = NEW.session_id AND source = 'claude-code'
            )
            BEGIN
                SELECT RAISE(ABORT, 'forced foreign import failure');
            END;
            """
        )


def test_foreign_import_success_control(tmp_path, foreign_import_home, capsys):
    db_path = _initialize_temp_state_db(tmp_path)
    foreign_path = _write_minimal_claude_session(tmp_path)

    rc = sc.cmd_sessions(
        _args("import", path=str(foreign_path), from_source="claude")
    )
    output = capsys.readouterr().out
    assert rc is None
    assert "✓ Imported Claude Code session as " in output

    with sqlite3.connect(db_path) as conn:
        session = conn.execute(
            """SELECT id, source, cwd, origin_json, title, title_source,
                      message_count
               FROM sessions WHERE source = 'claude-code'"""
        ).fetchone()
        assert session is not None
        session_id, source, cwd, origin_json, title, title_source, message_count = session
        messages = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()

    assert source == "claude-code"
    assert f"✓ Imported Claude Code session as {session_id}" in output
    assert cwd == "/tmp/foreign-project"
    assert json.loads(origin_json) == {
        "imported_from": {
            "tool": "claude-code",
            "path": str(foreign_path),
            "foreign_session_id": "foreign-claude-session",
        }
    }
    assert title == "Imported from Claude Code: Hello from Claude."
    assert title_source == "user"
    assert message_count == 2
    assert messages == [
        ("user", "Hello from Claude."),
        ("assistant", "Hello from Hermes."),
    ]


def test_foreign_import_retry_after_failure_is_not_duplicate(
    tmp_path, foreign_import_home, capsys
):
    db_path = _initialize_temp_state_db(tmp_path)
    foreign_path = _write_minimal_claude_session(tmp_path)
    _install_foreign_import_failure_triggers(db_path)

    failed_rc = sc.cmd_sessions(
        _args("import", path=str(foreign_path), from_source="claude")
    )
    failed_output = capsys.readouterr().out.lower()
    assert failed_rc == 1
    assert "nothing was imported" in failed_output

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TRIGGER foreign_import_intrude_after_user")
        conn.execute("DROP TRIGGER foreign_import_fail_before_assistant")

    success_rc = sc.cmd_sessions(
        _args("import", path=str(foreign_path), from_source="claude")
    )
    success_output = capsys.readouterr().out
    assert success_rc is None
    assert "✓ Imported Claude Code session as " in success_output

    with sqlite3.connect(db_path) as conn:
        sessions = conn.execute(
            "SELECT id, message_count FROM sessions WHERE source = 'claude-code'"
        ).fetchall()
        messages = conn.execute(
            "SELECT role, content FROM messages ORDER BY id"
        ).fetchall()
    assert len(sessions) == 1
    assert sessions[0][1] == 2
    assert messages == [
        ("user", "Hello from Claude."),
        ("assistant", "Hello from Hermes."),
    ]


def test_create_imported_session_collision_preserves_existing_session(tmp_path):
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path)
    session_id = "occupied-foreign-import-id"
    existing_origin = json.dumps({"existing": "origin"})
    db.create_session(
        session_id,
        source="existing-source",
        cwd="/tmp/existing-project",
        origin_json=existing_origin,
    )
    db.append_message(session_id, "user", "Keep this message.")
    holder = "pid=0:turn=import-collision-test"
    assert db.acquire_session_turn_lease(session_id, holder, ttl_seconds=30.0)
    before = db.get_session(session_id)
    before_messages = db.get_messages(session_id)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            db.create_imported_session(
                session_id,
                source="claude-code",
                messages=[
                    {"role": "user", "content": "Imported user."},
                    {"role": "assistant", "content": "Imported assistant."},
                ],
                cwd="/tmp/foreign-project",
                origin_json=json.dumps({"imported_from": "claude"}),
                turn_lease_holder=holder,
            )
        after = db.get_session(session_id)
        assert {
            key: after[key]
            for key in ("source", "origin_json", "message_count", "tool_call_count")
        } == {
            key: before[key]
            for key in ("source", "origin_json", "message_count", "tool_call_count")
        }
        assert db.get_messages(session_id) == before_messages
        with sqlite3.connect(db_path) as conn:
            current_holder = conn.execute(
                "SELECT holder FROM session_turn_leases WHERE conversation_id = ?",
                (session_id,),
            ).fetchone()[0]
        assert current_holder == holder
    finally:
        db.release_session_turn_lease(session_id, holder)
        db.close()


def test_foreign_import_postcommit_lease_release_failure_preserves_success(
    tmp_path, monkeypatch, foreign_import_home, capsys
):
    from hermes_state import SessionDB

    db_path = _initialize_temp_state_db(tmp_path)
    foreign_path = _write_minimal_claude_session(tmp_path)
    real_release = SessionDB.release_session_turn_lease

    def release_then_fail(self, session_id, holder):
        real_release(self, session_id, holder)
        raise RuntimeError("forced post-commit lease release failure")

    monkeypatch.setattr(SessionDB, "release_session_turn_lease", release_then_fail)

    rc = sc.cmd_sessions(
        _args("import", path=str(foreign_path), from_source="claude")
    )
    output = capsys.readouterr().out
    assert rc is None
    assert "✓ Imported Claude Code session as " in output
    assert "nothing was imported" not in output.lower()

    with sqlite3.connect(db_path) as conn:
        sessions = conn.execute(
            "SELECT id, message_count FROM sessions WHERE source = 'claude-code'"
        ).fetchall()
        messages = conn.execute(
            "SELECT role, content FROM messages ORDER BY id"
        ).fetchall()
    assert len(sessions) == 1
    assert f"✓ Imported Claude Code session as {sessions[0][0]}" in output
    assert sessions[0][1] == 2
    assert messages == [
        ("user", "Hello from Claude."),
        ("assistant", "Hello from Hermes."),
    ]
