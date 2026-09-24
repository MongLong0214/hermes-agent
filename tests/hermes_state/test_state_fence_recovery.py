"""``hermes sessions recover`` publishes a fenced store: the destination is refused before any
row is copied unless it carries this build's stamp and exactly this build's fences."""

from __future__ import annotations

import contextlib
import sqlite3

import pytest

from hermes_state_fence import STORED_SCHEMA_VERSION
from tests.hermes_state.fence_store_probe import expected_fences, fence_triggers, read_ro, stored
from tests.hermes_state.fork_store_fixture import build_store, isolate_home


def test_recovery_produces_a_fenced_store_that_verifies(tmp_path, monkeypatch):
    from hermes_cli.session_recovery import recover_session_database

    isolate_home(tmp_path, monkeypatch)
    source = build_store(tmp_path / "source" / "state.db", "fork29")
    output = tmp_path / "recovered.db"

    report = recover_session_database(source, output, work_dir=tmp_path)

    assert report["complete"] is True and report["verified"] is True, report["verification"]["errors"]
    assert stored(output) == STORED_SCHEMA_VERSION
    assert fence_triggers(output) == expected_fences(output, STORED_SCHEMA_VERSION)
    assert read_ro(output, "SELECT COUNT(*) FROM messages")[0][0] == read_ro(
        source, "SELECT COUNT(*) FROM messages")[0][0]


def test_recovery_refuses_a_destination_without_its_fences(tmp_path, monkeypatch):
    from hermes_cli import session_recovery
    from hermes_state import SessionDB

    isolate_home(tmp_path, monkeypatch)
    source = build_store(tmp_path / "source" / "state.db", "fork29")
    output = tmp_path / "recovered.db"

    @contextlib.contextmanager
    def fence_stripping_session_db(*, db_path):
        with SessionDB(db_path=db_path) as destination:
            yield destination
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("DROP TRIGGER IF EXISTS turn_fence_messages_insert")
        finally:
            conn.close()

    monkeypatch.setattr(session_recovery, "SessionDB", fence_stripping_session_db)
    with pytest.raises(session_recovery.SessionRecoveryError, match="turn-fence"):
        session_recovery.recover_session_database(source, output, work_dir=tmp_path)

    assert read_ro(output, "SELECT COUNT(*) FROM sessions")[0][0] == 0
    assert read_ro(output, "SELECT COUNT(*) FROM messages")[0][0] == 0
