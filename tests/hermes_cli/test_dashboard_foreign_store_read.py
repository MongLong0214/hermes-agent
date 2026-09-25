"""A dashboard read never migrates a store another Hermes build owns.

The read path heals a stale schema through one writable open. On a fork-lineage store that open
migrated it to this build's generation, and the fork build then refused its own store.
"""

from hermes_cli.web_server_sessions import _open_session_db_at_path
from tests.hermes_state.fork_store_fixture import build_store, file_fingerprint, isolate_home


def test_dashboard_read_serves_a_fork_store_without_migrating_it(tmp_path, monkeypatch):
    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fork29")
    before = file_fingerprint(db_path)

    db = _open_session_db_at_path(db_path, read_only=True)
    try:
        served = db.session_count()
    finally:
        db.close()

    after = file_fingerprint(db_path)
    assert served > 0
    assert (after["bytes"], after["sqlite_master"]) == (before["bytes"], before["sqlite_master"])
