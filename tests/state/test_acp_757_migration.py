"""Temporary SQLite migration ownership and publication contracts."""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import (
    SCHEMA_VERSION,
    TURN_FENCE_GENERATION,
    turn_fence_trigger_definitions,
)


def _snapshot(path):
    with sqlite3.connect(path) as conn:
        objects = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        rows = {}
        for kind, name, _table, _sql in objects:
            if kind == "table":
                quoted = name.replace('"', '""')
                rows[name] = sorted(
                    conn.execute(f'SELECT * FROM "{quoted}"').fetchall(), key=repr
                )
        return objects, rows


def _seed(path):
    db = SessionDB(path)
    with db.write_transaction() as conn:
        for session in ("open", "closed", "revoked"):
            conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, 'cli', 1)",
                (session,),
            )
        conn.execute("UPDATE sessions SET ended_at = 2 WHERE id = 'open'")
        conn.execute("UPDATE sessions SET ended_at = NULL WHERE id = 'open'")
        conn.execute("UPDATE sessions SET ended_at = 3 WHERE id = 'closed'")
        conn.execute(
            "UPDATE sessions SET end_reason = 'authority_revoked', ended_at = 4 "
            "WHERE id = 'revoked'"
        )
        conn.execute("UPDATE schema_version SET version = 28")
    db.close()


@pytest.mark.parametrize("generation", [27, 28, TURN_FENCE_GENERATION])
@pytest.mark.parametrize("keep_original", [False, True])
def test_owned_fence_is_replaced_independent_of_name(
    tmp_path, generation, keep_original
):
    path = tmp_path / "state.db"
    _seed(path)
    name, sql = turn_fence_trigger_definitions()[0]
    renamed = sql.replace(name, '"retained legacy fence"', 1).replace(
        f"!= {TURN_FENCE_GENERATION} ", f"!= {generation} "
    )
    inert = (
        "CREATE TRIGGER user_annotation AFTER INSERT ON state_meta BEGIN "
        "SELECT 'hermes_turn_fence_generation()', 'not a fence'; "
        "/* hermes_turn_fence_generation() */ SELECT 1; END"
    )
    with sqlite3.connect(path) as conn:
        if not keep_original:
            conn.execute(f'DROP TRIGGER "{name}"')
        conn.execute(renamed)
        conn.execute(inert)
    before = _snapshot(path)[1]
    db = SessionDB(path)
    db.close()
    objects, after = _snapshot(path)
    triggers = {name: sql for kind, name, _table, sql in objects if kind == "trigger"}
    assert "retained legacy fence" not in triggers
    assert triggers["user_annotation"] == inert
    for name, sql in turn_fence_trigger_definitions():
        assert triggers[name] == sql
    for table in (
        "sessions",
        "session_process_authorities",
        "session_process_authority_events",
        "session_process_reservations",
    ):
        assert after[table] == before[table]
    assert set(before["state_meta"]).issubset(after["state_meta"])
    assert after["schema_version"] == [(SCHEMA_VERSION,)]
    complete = _snapshot(path)
    SessionDB(path).close()
    assert _snapshot(path) == complete


def test_old_version_remains_visible_through_authority_initialization(
    tmp_path, monkeypatch
):
    path = tmp_path / "state.db"
    _seed(path)
    initialize = SessionDB._initialize_session_process_authority_state
    seen = []

    def observe(cursor):
        seen.append(cursor.execute("SELECT version FROM schema_version").fetchone()[0])
        assert seen[-1] == 28
        initialize(cursor)
        assert cursor.execute("SELECT version FROM schema_version").fetchone()[0] == 28

    monkeypatch.setattr(
        SessionDB, "_initialize_session_process_authority_state", staticmethod(observe)
    )
    SessionDB(path).close()
    assert seen == [28]
    assert _snapshot(path)[1]["schema_version"] == [(SCHEMA_VERSION,)]


@pytest.mark.parametrize("omitted", ["authority", "issued", "terminal"])
def test_incomplete_authority_initialization_refuses_and_restores_preimage(
    tmp_path, monkeypatch, omitted
):
    path = tmp_path / "state.db"
    _seed(path)
    # A valid newer session epoch with no corresponding authority yet models
    # a legacy store awaiting backfill; historical rows stay untouched.
    from hermes_state_common import register_turn_fence_generation

    with sqlite3.connect(path) as conn:
        register_turn_fence_generation(conn)
        conn.execute("UPDATE sessions SET session_generation = session_generation + 1")
    before = _snapshot(path)
    initialize = SessionDB._initialize_session_process_authority_state

    class OmitOneMigrationStatement(sqlite3.Cursor):
        def __init__(self, cursor):
            super().__init__(cursor.connection)
            self.cursor = cursor

        def execute(self, sql, parameters=()):
            compact = " ".join(sql.split())
            skip = (
                omitted == "authority"
                and compact.startswith(
                    "INSERT OR IGNORE INTO session_process_authorities"
                )
                or omitted == "issued"
                and compact.startswith("INSERT INTO session_process_authority_events")
                and "'SESSION_ISSUED'" in compact
                or omitted == "terminal"
                and compact.startswith("INSERT INTO session_process_authority_events")
                and "'SESSION_REVOKED'" in compact
            )
            return self.cursor.execute(
                "SELECT 1" if skip else sql, () if skip else parameters
            )

    monkeypatch.setattr(
        SessionDB,
        "_initialize_session_process_authority_state",
        staticmethod(lambda cursor: initialize(OmitOneMigrationStatement(cursor))),
    )
    with pytest.raises(sqlite3.DatabaseError, match="AUTHORITY_MIGRATION_INCOMPLETE"):
        SessionDB(path)
    assert _snapshot(path) == before
    monkeypatch.undo()
    SessionDB(path).close()
    complete = _snapshot(path)
    for table in (
        "sessions",
        "session_process_authorities",
        "session_process_authority_events",
    ):
        assert set(before[1][table]).issubset(complete[1][table])
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM sessions s LEFT JOIN session_process_authorities a "
            "ON a.session_id = s.id AND a.session_generation = s.session_generation "
            "WHERE a.session_id IS NULL"
        ).fetchone() == (0,)
    SessionDB(path).close()
    assert _snapshot(path) == complete


@pytest.mark.parametrize(
    "declaration",
    [
        "CREATE TRIGGER extra_fence AFTER INSERT ON state_meta BEGIN SELECT hermes_turn_fence_generation(); END",
        'CREATE TRIGGER extra_fence AFTER INSERT ON state_meta BEGIN SELECT "HeRmEs_TuRn_FeNcE_GeNeRaTiOn" /* call */ (); END',
        "CREATE TRIGGER extra_fence AFTER INSERT ON state_meta BEGIN SELECT [hermes_turn_fence_generation](); END",
        "CREATE TRIGGER extra_fence AFTER INSERT ON state_meta BEGIN SELECT `hermes_turn_fence_generation`(); END",
        "collision",
        "changed-body",
    ],
)
def test_unknown_declarations_are_refused_without_changing_any_rows(
    tmp_path, declaration
):
    path = tmp_path / "state.db"
    _seed(path)
    name, sql = turn_fence_trigger_definitions()[0]
    with sqlite3.connect(path) as conn:
        if declaration == "collision":
            conn.execute(f'DROP TRIGGER "{name}"')
            declaration = (
                f"CREATE TRIGGER {name} AFTER INSERT ON state_meta BEGIN SELECT 1; END"
            )
        elif declaration == "changed-body":
            declaration = sql.replace(name, "extra_fence", 1).replace(
                "state DB generation incompatible", "user-owned diagnostic"
            )
        conn.execute(declaration)
    before = _snapshot(path)
    with pytest.raises(sqlite3.DatabaseError, match="TURN_FENCE_MIGRATION_REFUSED"):
        SessionDB(path)
    assert _snapshot(path) == before


@pytest.mark.parametrize(
    "checkpoint", ["mid-authority", "final-validation", "extra-fence"]
)
def test_migration_abort_restores_every_row_and_declaration(
    tmp_path, monkeypatch, checkpoint
):
    path = tmp_path / "state.db"
    _seed(path)
    from hermes_state_common import register_turn_fence_generation

    with sqlite3.connect(path) as conn:
        register_turn_fence_generation(conn)
        conn.execute("UPDATE sessions SET session_generation = session_generation + 1")
    before = _snapshot(path)
    initialize = SessionDB._initialize_session_process_authority_state
    validate = SessionDB._validate_session_process_authority_state
    observed = []

    class FailDuringEvents(sqlite3.Cursor):
        def __init__(self, cursor):
            super().__init__(cursor.connection)
            self.cursor = cursor

        def execute(self, sql, parameters=()):
            if " ".join(sql.split()).startswith(
                "INSERT INTO session_process_authority_events"
            ):
                assert (
                    self.cursor.execute(
                        "SELECT version FROM schema_version"
                    ).fetchone()[0]
                    == 28
                )
                # Authority backfill already ran before this failure.
                assert self.cursor.execute(
                    "SELECT COUNT(*) FROM session_process_authorities"
                ).fetchone()[0] > len(before[1]["session_process_authorities"])
                observed.append(checkpoint)
                raise sqlite3.DatabaseError("injected migration abort")
            return self.cursor.execute(sql, parameters)

    def fail_validation(cursor):
        validate(cursor)
        assert cursor.execute("SELECT version FROM schema_version").fetchone()[0] == 28
        observed.append(checkpoint)
        raise sqlite3.DatabaseError("injected migration abort")

    def add_extra(cursor):
        initialize(cursor)
        name, sql = turn_fence_trigger_definitions()[0]
        cursor.execute(sql.replace(name, "late_extra_fence", 1))
        observed.append(checkpoint)

    if checkpoint == "mid-authority":
        monkeypatch.setattr(
            SessionDB,
            "_initialize_session_process_authority_state",
            staticmethod(lambda cursor: initialize(FailDuringEvents(cursor))),
        )
    elif checkpoint == "final-validation":
        monkeypatch.setattr(
            SessionDB,
            "_validate_session_process_authority_state",
            staticmethod(fail_validation),
        )
    else:
        monkeypatch.setattr(
            SessionDB,
            "_initialize_session_process_authority_state",
            staticmethod(add_extra),
        )
    with pytest.raises(
        (sqlite3.DatabaseError, RuntimeError),
        match="injected migration abort|turn-fence trigger verification failed",
    ):
        SessionDB(path)
    assert observed == [checkpoint]
    assert _snapshot(path) == before
