"""Unconditional session_turn_leases legacy-epoch heal (#84512).

Installs whose ``session_turn_leases`` table predates this module carry a
legacy ``epoch INTEGER NOT NULL`` column with no ``DEFAULT`` (plus the
nullable ``owner_pid`` / ``owner_pid_start``).
``try_acquire_session_turn_lease``'s
``INSERT OR IGNORE INTO session_turn_leases (conversation_id, holder,
acquired_at, expires_at) ...`` never populates ``epoch``, so every insert
violates the NOT NULL constraint -- and ``OR IGNORE`` swallows that
violation silently. The row is never created, the following
``SELECT holder ...`` returns no owner, and ``try_acquire_session_turn_lease``
returns False forever, with no error anywhere. Measured as a 10-hour total
outage on a live store: every acquire polled its full patience window and
reported "Another Hermes process is using this session" while nothing held
it.

There has never been a version-gated migration for this table (grep
confirms), so a store can be sitting at ``schema_version == SCHEMA_VERSION``
today and still be broken -- ``_heal_session_turn_leases_legacy_epoch`` has
to run unconditionally on every open, same pattern as
``_heal_gateway_routing_pk`` / ``_heal_session_model_usage_pk``.
"""

import sqlite3

import pytest

from hermes_state import SessionDB
from hermes_state_common import (
    SCHEMA_VERSION,
    register_turn_fence_generation,
    turn_fence_trigger_definitions,
)

LEGACY_SQL = """
    CREATE TABLE session_turn_leases (
        conversation_id TEXT PRIMARY KEY,
        holder TEXT NOT NULL,
        acquired_at REAL NOT NULL,
        expires_at REAL NOT NULL,
        epoch INTEGER NOT NULL,
        owner_pid INTEGER,
        owner_pid_start REAL
    )
"""

_GOVERNED_TRIGGER_NAMES = tuple(
    sorted(
        name
        for name, _sql in turn_fence_trigger_definitions()
        if name.startswith("turn_fence_session_turn_leases_")
    )
)


def _make_legacy_epoch_db(db_path, with_triggers=True):
    """Build a state.db from raw sqlite3 with the true legacy 7-column
    ``session_turn_leases`` shape, ``schema_version`` already at current
    (proving the heal cannot depend on the version gate), and -- unless
    ``with_triggers`` is False -- the real turn-fence triggers already
    attached, exactly as a prior open of post-v27 code would have installed
    them (the trigger body never references ``epoch``, so nothing about
    installing them required the modern column shape)."""
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute(
        "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
    )
    conn.execute(LEGACY_SQL)
    if with_triggers:
        for name, sql in turn_fence_trigger_definitions():
            if name.startswith("turn_fence_session_turn_leases_"):
                conn.execute(sql)
    conn.commit()
    conn.close()


def _cols(db):
    rows = db._conn.execute(
        'PRAGMA table_info("session_turn_leases")'
    ).fetchall()
    return sorted(r["name"] for r in rows)


def _governed_triggers(db):
    rows = db._conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='trigger' AND tbl_name='session_turn_leases'"
    ).fetchall()
    return sorted(r[0] for r in rows)


class TestSessionTurnLeasesEpochHeal:
    def test_RED_legacy_epoch_column_silently_breaks_every_acquire(
        self, tmp_path
    ):
        """RED, direct: build the true legacy shape by hand (no heal
        involved -- this exercises exactly what _reconcile_columns +
        SCHEMA_SQL do on a fresh connection with no heal call) and show
        try_acquire returns False with no row, no exception."""
        conn = sqlite3.connect(tmp_path / "raw.db")
        conn.execute(LEGACY_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO session_turn_leases "
            "(conversation_id, holder, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            ("conv-red", "holder-A", 1.0, 2.0),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM session_turn_leases WHERE conversation_id = ?",
            ("conv-red",),
        ).fetchone()
        assert row is None, (
            "the INSERT OR IGNORE silently no-ops on the legacy shape -- "
            "this is the exact mechanism, reproduced without any Hermes code"
        )
        conn.close()

    def test_legacy_shape_fails_before_and_succeeds_after_heal(
        self, tmp_path
    ):
        """The real try_acquire_session_turn_lease: False (no row) on a
        store carrying the legacy epoch column, True (row present) once
        SessionDB opens it and the heal runs -- with the pre-existing
        turn-fence triggers left intact throughout."""
        db_path = tmp_path / "state.db"
        _make_legacy_epoch_db(db_path, with_triggers=True)

        # RED: reproduce with the real acquire path directly against the
        # legacy table, bypassing SessionDB/the heal entirely. The governed
        # triggers fire on any raw connection, so register the same UDF
        # SessionDB registers on every connection it opens (see
        # hermes_state._connect_and_init) -- otherwise this fails with
        # "no such function" rather than exercising the NOT NULL swallow.
        raw = sqlite3.connect(db_path)
        raw.row_factory = sqlite3.Row
        register_turn_fence_generation(raw)
        raw.execute(
            "INSERT OR IGNORE INTO session_turn_leases "
            "(conversation_id, holder, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            ("conv-1", "holder-A", 1.0, 2.0),
        )
        raw.commit()
        pre_row = raw.execute(
            "SELECT * FROM session_turn_leases WHERE conversation_id = ?",
            ("conv-1",),
        ).fetchone()
        assert pre_row is None
        raw.close()

        # GREEN: open through SessionDB, which runs the heal unconditionally
        # on every open, then acquire for real.
        db = SessionDB(db_path=db_path)
        try:
            assert "epoch" not in _cols(db)
            assert _governed_triggers(db) == list(_GOVERNED_TRIGGER_NAMES)

            assert db.try_acquire_session_turn_lease(
                "conv-1", "holder-A", ttl_seconds=30
            )
            row = db._conn.execute(
                "SELECT holder FROM session_turn_leases "
                "WHERE conversation_id = 'conv-1'"
            ).fetchone()
            assert row is not None and row["holder"] == "holder-A"
        finally:
            db.close()

    def test_owner_pid_columns_are_left_in_place(self, tmp_path):
        """Nullable legacy columns are not the cause of the outage and are
        deliberately not removed by this heal."""
        db_path = tmp_path / "state.db"
        _make_legacy_epoch_db(db_path, with_triggers=True)
        db = SessionDB(db_path=db_path)
        try:
            cols = _cols(db)
            assert "owner_pid" in cols
            assert "owner_pid_start" in cols
        finally:
            db.close()

    def test_healthy_db_is_a_noop_idempotent(self, tmp_path):
        """A DB already at the correct 4-column shape is untouched and
        keeps working -- re-running the heal directly is also a no-op."""
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s1", "cli")
            assert _cols(db) == [
                "acquired_at",
                "conversation_id",
                "expires_at",
                "holder",
            ]
            assert db.try_acquire_session_turn_lease(
                "s1", "holder-A", ttl_seconds=30
            )
            db.release_session_turn_lease("s1", "holder-A")

            cur = db._conn.cursor()
            db._heal_session_turn_leases_legacy_epoch(cur)

            assert _cols(db) == [
                "acquired_at",
                "conversation_id",
                "expires_at",
                "holder",
            ]
            assert db.try_acquire_session_turn_lease(
                "s1", "holder-B", ttl_seconds=30
            )
        finally:
            db.close()

    def test_no_legacy_leftover_table_and_index_present(self, tmp_path):
        """Whichever path ran (DROP COLUMN or rebuild fallback), no
        *_legacy_epoch residue is left behind and the expires_at index
        exists."""
        db_path = tmp_path / "state.db"
        _make_legacy_epoch_db(db_path, with_triggers=True)
        db = SessionDB(db_path=db_path)
        try:
            left = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='session_turn_leases_legacy_epoch'"
            ).fetchone()
            assert left is None
            idx = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name='idx_session_turn_leases_expires'"
            ).fetchone()
            assert idx is not None
        finally:
            db.close()

    def test_pre_335_rebuild_fallback_also_heals_and_reinstalls_triggers(
        self, tmp_path, monkeypatch
    ):
        """Force the < 3.35 rebuild branch (no DROP COLUMN support) and
        confirm it heals the column AND reinstalls the fence triggers that
        the RENAME would otherwise have carried away from the new table."""
        db_path = tmp_path / "state.db"
        _make_legacy_epoch_db(db_path, with_triggers=True)

        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
        db = SessionDB(db_path=db_path)
        try:
            assert "epoch" not in _cols(db)
            assert _governed_triggers(db) == list(_GOVERNED_TRIGGER_NAMES)
            assert db.try_acquire_session_turn_lease(
                "conv-legacy-sqlite", "holder-A", ttl_seconds=30
            )
        finally:
            db.close()

    def test_guard_every_not_null_no_default_column_is_insert_populated(
        self,
    ):
        """The class of bug that caused the outage: a column that is NOT
        NULL with no DEFAULT, that the real insert statement does not
        populate, silently discarded by INSERT OR IGNORE. Assert this
        holds for every column session_turn_leases declares today -- so
        the *next* NOT NULL column added to SCHEMA_SQL without a matching
        INSERT update turns this red instead of shipping silent as this one
        did.

        Scoped to session_turn_leases rather than every table in SCHEMA_SQL:
        several governed tables (sessions, messages, ...) are written by
        dozens of call sites with dynamically constructed column lists
        (partial UPDATEs, optional fields threaded through kwargs), so
        "the insert statement" is not a single static string to grep --
        making a fully general version of this check either badly
        false-positive-prone or require executing every write path with
        every column combination. session_turn_leases has exactly one
        writer (try_acquire_session_turn_lease's INSERT) with a single
        static column list, which is what makes this check meaningful
        rather than vacuous.
        """
        import inspect

        import hermes_state

        source = inspect.getsource(hermes_state.SessionDB.try_acquire_session_turn_lease)
        # Extract the literal column list from the one INSERT this table has.
        insert_start = source.index("INSERT OR IGNORE INTO session_turn_leases")
        insert_stmt = source[insert_start:insert_start + 400]
        # Pull the parenthesized column list right after the table name.
        paren_start = insert_stmt.index("(", insert_stmt.index("session_turn_leases"))
        paren_end = insert_stmt.index(")", paren_start)
        inserted_cols = {
            c.strip().strip('"')
            for c in insert_stmt[paren_start + 1:paren_end].replace("\n", " ").split(",")
        }

        from hermes_state_common import SCHEMA_SQL

        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(SCHEMA_SQL)
            declared = ref.execute(
                'PRAGMA table_info("session_turn_leases")'
            ).fetchall()
        finally:
            ref.close()

        not_null_no_default = {
            row[1] for row in declared if row[3] and row[4] is None
        }
        uncovered = not_null_no_default - inserted_cols
        assert uncovered == set(), (
            f"session_turn_leases declares NOT NULL/no-DEFAULT columns "
            f"{uncovered} that try_acquire_session_turn_lease's INSERT does "
            f"not populate -- this is exactly the class of bug that caused "
            f"the 10-hour outage (epoch was such a column before the "
            f"schema_common declaration was corrected)"
        )
