"""FTS schema DDL must not commit its caller's savepoint (v30 migration)."""

import pytest

from hermes_state import SessionDB
from hermes_state_common import FTS_SQL


def test_ensure_fts_schema_keeps_caller_savepoint_rollbackable(tmp_path):
    """The empty v29->v30 alignment calls this helper inside a SAVEPOINT.

    ``Cursor.executescript`` first commits its caller's transaction, so the old
    implementation made ``ROLLBACK TO fts_align_empty`` fail with ``no such
    savepoint`` and left the migration transaction partially committed.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    if not db._fts_enabled:
        db.close()
        pytest.skip("SQLite FTS5 unavailable")
    try:
        assert db._conn is not None
        cursor = db._conn.cursor()
        cursor.execute("SAVEPOINT fts_schema_ensure")
        cursor.execute(
            "INSERT INTO state_meta (key, value) VALUES ('fts_savepoint_probe', '1')"
        )

        assert db._ensure_fts_schema(cursor, "messages_fts", FTS_SQL)

        cursor.execute("ROLLBACK TO SAVEPOINT fts_schema_ensure")
        cursor.execute("RELEASE SAVEPOINT fts_schema_ensure")
        assert cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_savepoint_probe'"
        ).fetchone() is None
    finally:
        db.close()
