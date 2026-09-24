"""Read-only views of a state.db for the fence-migration contracts.

Expected fences come from the fixture's own trigger text (``fence_sql``), not from the
production builder, so an assertion here is an independent oracle for the migration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.hermes_state.fork_store_fixture import AUTHORITY_GOVERNED, CORE_GOVERNED, OPERATIONS, fence_sql


def read_ro(db, sql: str, params=()) -> list:
    conn = sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def stamp_rows(db) -> list:
    return read_ro(db, "SELECT version, typeof(version) FROM schema_version")


def stored(db) -> int:
    rows = read_ro(db, "SELECT version FROM schema_version")
    assert len(rows) == 1, rows
    return rows[0][0]


def fence_triggers(db) -> dict:
    return dict(read_ro(db, "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'turn_fence_%'"))


def fence_master_rows(db) -> list:
    return read_ro(db, "SELECT type, name, tbl_name, sql FROM sqlite_master "
                       "WHERE name LIKE 'turn_fence_%' ORDER BY name")


def governed_present(db) -> list:
    tables = {row[0] for row in read_ro(db, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    return [t for t in CORE_GOVERNED + AUTHORITY_GOVERNED if t in tables]


def expected_fences(db, literal: int, tables=None) -> dict:
    return {
        f"turn_fence_{table}_{op.lower()}": fence_sql(table, op, literal)
        for table in (tables if tables is not None else governed_present(db)) for op in OPERATIONS
    }


def schema_cookie(db) -> int:
    return read_ro(db, "PRAGMA schema_version")[0][0]
