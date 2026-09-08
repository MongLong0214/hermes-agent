"""The serialization mixin must be wired into every tracked connection.

WHY THIS TEST EXISTS
    Commit 923c292051 (2026-08-25) serialized every SQLite connection behind an
    RLock because the turn-fence UDF re-enters Python from inside sqlite3_step,
    and an unsynchronized shared connection turns that into a GIL/connection-
    mutex ABBA deadlock — the gateway froze for 80+ minutes.

    On 2026-08-28 a merge (3c87b0cece) silently dropped that fix, and the
    gateway froze three times in one day.  A host-side guard caught it in
    minutes, but a host-side guard does not travel with the code.  This test
    does: any future tree that loses the mixin loses it WITH this test present,
    and the test fails there.

    If you are deleting this because the mixin was renamed: wire the new name
    in below.  If you are deleting it because the serialization was removed:
    read reports/sqlite-udf-gil-deadlock-fix-20260825.md first — the deadlock
    it prevents is not theoretical, it happened four times.
"""
import sqlite3

import hermes_state


def test_serialization_symbols_exist():
    assert hasattr(hermes_state, "_SerializedCursor"), (
        "hermes_state._SerializedCursor is gone — the GIL-deadlock "
        "serialization fix (orig 923c292051) has been dropped from this tree"
    )
    assert hasattr(hermes_state, "_serialized_connection_factory")


def test_tracked_connection_is_serialized(tmp_path):
    conn = hermes_state._connect_tracked_db(tmp_path / "probe.db")
    try:
        assert hasattr(conn, "_hermes_serial_lock"), (
            "_connect_tracked_db returned an unserialized connection — the "
            "factory hook in _connect_tracked_db has been dropped"
        )
        cur = conn.cursor()
        assert isinstance(cur, hermes_state._SerializedCursor), (
            "cursor() did not produce a _SerializedCursor — fetch paths are "
            "not under the connection RLock"
        )
        cur.execute("SELECT 1")
        assert cur.fetchone() == (1,)
    finally:
        conn.close()
