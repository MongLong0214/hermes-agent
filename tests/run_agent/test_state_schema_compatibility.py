"""State-schema recheck behavior kept separate from turn propagation."""

import sqlite3

import pytest

from hermes_state import IncompatibleSchemaError, SCHEMA_VERSION, SessionDB


def test_ensure_compatible_schema_fails_closed_after_sibling_future_drift(tmp_path):
    """The already-open connection detects a later incompatible scalar."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sibling = sqlite3.connect(str(db_path))
    try:
        sibling.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION + 1,))
        sibling.commit()

        with pytest.raises(IncompatibleSchemaError):
            db.ensure_compatible_schema()
    finally:
        sibling.close()
        db.close()


def test_ensure_compatible_schema_is_select_only_for_current_and_malformed_state(
    tmp_path,
):
    """The recheck neither repairs nor publishes while reading current state."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sibling = sqlite3.connect(str(db_path))
    statements = []
    try:
        db._conn.set_trace_callback(statements.append)
        assert db.ensure_compatible_schema() is None
        assert statements
        assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)

        statements.clear()
        sibling.execute("UPDATE schema_version SET version = 'not-an-integer'")
        sibling.commit()
        with pytest.raises(IncompatibleSchemaError):
            db.ensure_compatible_schema()
        assert statements
        assert all(statement.lstrip().upper().startswith("SELECT") for statement in statements)
        assert sibling.execute("SELECT version FROM schema_version").fetchall() == [
            ("not-an-integer",)
        ]
    finally:
        db._conn.set_trace_callback(None)
        sibling.close()
        db.close()
