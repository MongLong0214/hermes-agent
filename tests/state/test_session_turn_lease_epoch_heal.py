"""Unconditional repair for the legacy session_turn_leases epoch column.

Legacy ``session_turn_leases`` tables can carry ``epoch INTEGER NOT NULL``
with no ``DEFAULT``, plus nullable ``owner_pid`` and ``owner_pid_start``.
The acquisition ``INSERT OR IGNORE`` does not populate ``epoch``; SQLite
rejects the row and the conflict mode suppresses the NOT NULL violation.  The
row is not created, the following owner lookup finds no row, and acquisition
returns False.

No version-gated migration covers this table, so a database can report
``schema_version == SCHEMA_VERSION`` while retaining the legacy shape.  The
heal must therefore run unconditionally on every open, consistent with
``_heal_gateway_routing_pk`` and ``_heal_session_model_usage_pk``.  The
nullable owner fields are deliberately preserved.
"""

import sqlite3

import pytest

import hermes_state
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


def _lease_store_snapshot(db_path):
    """Exact observable schema/data state used by refusal/rollback tests."""
    conn = sqlite3.connect(db_path)
    try:
        return {
            "objects": conn.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
            ).fetchall(),
            "columns": conn.execute(
                'PRAGMA table_info("session_turn_leases")'
            ).fetchall(),
            "rows": conn.execute(
                'SELECT * FROM session_turn_leases ORDER BY rowid'
            ).fetchall(),
            "schema_version": conn.execute(
                "SELECT version FROM schema_version ORDER BY rowid"
            ).fetchall(),
        }
    finally:
        conn.close()


def _make_schema_current_epoch_variant(db_path, table_sql):
    """Replace one already-current lease table with a seeded epoch variant."""
    fresh = SessionDB(db_path=db_path)
    fresh.close()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE session_turn_leases")
        conn.execute(table_sql)
        conn.execute(
            "INSERT INTO session_turn_leases "
            "(conversation_id, holder, acquired_at, expires_at, epoch, "
            "owner_pid, owner_pid_start) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("seed", "holder", 1.0, 2.0, 7, 123, 456.0),
        )
        conn.execute(
            "CREATE INDEX idx_session_turn_leases_expires "
            "ON session_turn_leases(expires_at)"
        )
        for name, sql in turn_fence_trigger_definitions():
            if name.startswith("turn_fence_session_turn_leases_"):
                conn.execute(sql)
        conn.commit()
    finally:
        conn.close()


def _seed_compatible_fallback_objects(db_path):
    conn = sqlite3.connect(db_path)
    try:
        register_turn_fence_generation(conn)
        conn.execute(
            "INSERT INTO session_turn_leases "
            "(conversation_id, holder, acquired_at, expires_at, epoch, "
            "owner_pid, owner_pid_start) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("seed-second", "holder-second", 3.0, 4.0, 8, 456, 789.0),
        )
        conn.execute(
            "CREATE INDEX session_turn_leases_owner_pid_idx "
            "ON session_turn_leases(owner_pid)"
        )
        conn.execute(
            """CREATE TRIGGER session_turn_leases_owner_pid_guard
            BEFORE UPDATE OF owner_pid ON session_turn_leases
            BEGIN
                SELECT NEW.owner_pid_start;
            END"""
        )
        conn.commit()
    finally:
        conn.close()


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

    @pytest.mark.parametrize(
        "table_sql",
        (
            LEGACY_SQL.replace("epoch INTEGER NOT NULL,", "epoch INTEGER,"),
            LEGACY_SQL.replace(
                "epoch INTEGER NOT NULL,", "epoch INTEGER NOT NULL DEFAULT 1,"
            ),
            LEGACY_SQL.replace("epoch INTEGER NOT NULL,", "epoch TEXT NOT NULL,"),
            """
                CREATE TABLE session_turn_leases (
                    conversation_id TEXT,
                    holder TEXT NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    epoch INTEGER PRIMARY KEY,
                    owner_pid INTEGER,
                    owner_pid_start REAL
                )
            """,
            """
                CREATE TABLE session_turn_leases (
                    conversation_id TEXT PRIMARY KEY,
                    holder TEXT NOT NULL,
                    epoch INTEGER NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    owner_pid INTEGER,
                    owner_pid_start REAL
                )
            """,
            LEGACY_SQL.replace(
                "owner_pid_start REAL", "owner_pid_start REAL, unproven_extension TEXT"
            ),
            LEGACY_SQL.replace(
                "owner_pid_start REAL", "owner_pid_start REAL, CHECK (holder <> '')"
            ),
        ),
        ids=(
            "nullable",
            "default",
            "different-type",
            "epoch-pk",
            "wrong-order",
            "extension",
            "unknown-check",
        ),
    )
    def test_schema_current_unproven_epoch_descriptor_refuses_without_mutation(
        self, tmp_path, table_sql
    ):
        """Only the complete seven-column legacy descriptor authorizes DDL."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, table_sql)
        before = _lease_store_snapshot(db_path)

        with pytest.raises(
            sqlite3.DatabaseError, match="SESSION_TURN_LEASE_EPOCH_HEAL_REFUSED"
        ):
            SessionDB(db_path=db_path)

        assert _lease_store_snapshot(db_path) == before

    def test_modern_drop_failure_refuses_open_and_preserves_legacy_store(self, tmp_path):
        """A real epoch-dependent index makes DROP fail loud and atomic."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE INDEX session_turn_leases_epoch_blocker "
                "ON session_turn_leases(epoch)"
            )
            conn.commit()
        finally:
            conn.close()
        before = _lease_store_snapshot(db_path)

        with pytest.raises(
            sqlite3.DatabaseError, match="SESSION_TURN_LEASE_EPOCH_HEAL_DROP_FAILED"
        ):
            SessionDB(db_path=db_path)

        assert _lease_store_snapshot(db_path) == before

    def test_pre_335_rebuild_preserves_seeded_rows_and_compatible_objects(
        self, tmp_path, monkeypatch
    ):
        """Fallback retains owner values, explicit objects, and all fences."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        _seed_compatible_fallback_objects(db_path)
        before = _lease_store_snapshot(db_path)
        before_sql = {name: sql for _kind, name, _table, sql in before["objects"]}

        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
        db = SessionDB(db_path=db_path)
        try:
            assert _cols(db) == [
                "acquired_at",
                "conversation_id",
                "expires_at",
                "holder",
                "owner_pid",
                "owner_pid_start",
            ]
            assert [
                tuple(row)
                for row in db._conn.execute(
                    "SELECT conversation_id, holder, acquired_at, expires_at, "
                    "owner_pid, owner_pid_start FROM session_turn_leases "
                    "ORDER BY conversation_id"
                ).fetchall()
            ] == [
                ("seed", "holder", 1.0, 2.0, 123, 456.0),
                ("seed-second", "holder-second", 3.0, 4.0, 456, 789.0),
            ]
            after_sql = {
                name: sql
                for _kind, name, _table, sql in _lease_store_snapshot(db_path)[
                    "objects"
                ]
            }
            for name in (
                "idx_session_turn_leases_expires",
                "session_turn_leases_owner_pid_idx",
                "session_turn_leases_owner_pid_guard",
                *_GOVERNED_TRIGGER_NAMES,
            ):
                assert after_sql[name] == before_sql[name]
            assert db._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'session_turn_leases_legacy_epoch'"
            ).fetchone() is None
            assert db.try_acquire_session_turn_lease(
                "fallback-open", "holder", ttl_seconds=30
            )
        finally:
            db.close()

    def test_pre_335_rebuild_captures_late_compatible_objects_after_begin(
        self, tmp_path, monkeypatch
    ):
        """The fallback snapshots objects only after its write lock is held."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        late_index = "session_turn_leases_late_owner_pid_idx"
        late_trigger = "session_turn_leases_late_owner_pid_guard"
        injected = False
        expected_objects = {}
        original_execute = hermes_state._SerializedCursor.execute

        def create_objects_before_begin(cursor, sql, parameters=()):
            nonlocal injected, expected_objects
            if sql == "BEGIN IMMEDIATE" and not injected:
                injected = True
                writer = sqlite3.connect(db_path)
                try:
                    writer.execute(
                        f"CREATE INDEX {late_index} "
                        "ON session_turn_leases(owner_pid)"
                    )
                    writer.execute(
                        f"""CREATE TRIGGER {late_trigger}
                        BEFORE UPDATE OF owner_pid ON session_turn_leases
                        BEGIN
                            SELECT NEW.owner_pid_start;
                        END"""
                    )
                    writer.commit()
                    expected_objects = dict(
                        writer.execute(
                            "SELECT name, sql FROM sqlite_master "
                            "WHERE name IN (?, ?) ORDER BY name",
                            (late_index, late_trigger),
                        ).fetchall()
                    )
                finally:
                    writer.close()
            return original_execute(cursor, sql, parameters)

        monkeypatch.setattr(
            hermes_state._SerializedCursor,
            "execute",
            create_objects_before_begin,
        )
        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))

        db = SessionDB(db_path=db_path)
        try:
            assert injected
            actual_objects = dict(
                db._conn.execute(
                    "SELECT name, sql FROM sqlite_master "
                    "WHERE name IN (?, ?) ORDER BY name",
                    (late_index, late_trigger),
                ).fetchall()
            )
            assert actual_objects == expected_objects
            assert db._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name = 'session_turn_leases_legacy_epoch'"
            ).fetchone() is None
        finally:
            db.close()

    def test_pre_335_rebuild_refuses_epoch_dependent_objects_without_mutation(
        self, tmp_path, monkeypatch
    ):
        """The fallback preflights definitions it cannot safely recreate."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "CREATE INDEX session_turn_leases_epoch_blocker "
                "ON session_turn_leases(epoch)"
            )
            conn.commit()
        finally:
            conn.close()
        before = _lease_store_snapshot(db_path)

        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
        with pytest.raises(
            sqlite3.DatabaseError, match="SESSION_TURN_LEASE_EPOCH_REBUILD_REFUSED"
        ):
            SessionDB(db_path=db_path)

        assert _lease_store_snapshot(db_path) == before

    @pytest.mark.parametrize(
        "stage", ("rename", "create", "copy", "verify", "drop", "index", "trigger")
    )
    def test_pre_335_rebuild_rolls_back_every_stage_failure(
        self, tmp_path, monkeypatch, stage
    ):
        """Every rebuild stage either commits together or leaves no residue."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        _seed_compatible_fallback_objects(db_path)
        before = _lease_store_snapshot(db_path)

        def fail_selected_stage(_db, checkpoint):
            if checkpoint == stage:
                raise sqlite3.OperationalError(f"forced fallback {stage} failure")

        monkeypatch.setattr(
            SessionDB,
            "_session_turn_lease_epoch_rebuild_checkpoint",
            fail_selected_stage,
            raising=False,
        )
        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))

        with pytest.raises(sqlite3.OperationalError, match=f"forced fallback {stage}"):
            SessionDB(db_path=db_path)

        assert _lease_store_snapshot(db_path) == before

    def test_owner_pid_columns_are_left_in_place(self, tmp_path):
        """Nullable legacy columns do not cause the rejected insert and are
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
        self, tmp_path
    ):
        """Exercise the real acquisition path against a fresh current schema.

        A future NOT NULL/no-DEFAULT column must be populated by the
        acquisition path. Discover that contract from the table and inspect
        the row produced by the real ``try_acquire_session_turn_lease`` call,
        so a future required-column omission turns this red instead of
        creating an unpopulated lease.

        This remains scoped to session_turn_leases because the production
        acquisition path creates one row for this table. It is a behavior
        invariant over the actual schema and row, not a parser for production
        source or SQL spelling.
        """
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db.try_acquire_session_turn_lease(
                "guard-conversation", "guard-holder", ttl_seconds=30
            )
            row = db._conn.execute(
                "SELECT * FROM session_turn_leases WHERE conversation_id = ?",
                ("guard-conversation",),
            ).fetchone()
            assert row is not None

            declared = db._conn.execute(
                'PRAGMA table_info("session_turn_leases")'
            ).fetchall()
            not_null_no_default = {
                column[1]
                for column in declared
                if column[3] and column[4] is None
            }
            unpopulated = {
                column for column in not_null_no_default if row[column] is None
            }
            assert unpopulated == set(), (
                f"session_turn_leases declares NOT NULL/no-DEFAULT columns "
                f"{unpopulated} that the real acquisition path did not "
                "populate; acquisition must populate every required column"
            )
        finally:
            db.close()

    @pytest.mark.parametrize(
        ("name", "ddl"),
        (
            (
                "session_turn_leases_dependent_view",
                "CREATE VIEW session_turn_leases_dependent_view AS "
                "SELECT conversation_id FROM session_turn_leases",
            ),
            (
                "session_turn_leases_external_dependency",
                "CREATE TRIGGER session_turn_leases_external_dependency "
                "AFTER UPDATE ON schema_version BEGIN "
                "SELECT count(*) FROM session_turn_leases; END",
            ),
        ),
        ids=("view", "external-trigger"),
    )
    def test_pre_335_rebuild_refuses_unprovable_dependent_objects_without_mutation(
        self, tmp_path, monkeypatch, name, ddl
    ):
        """Views and external triggers cannot be proven safe for table replay."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(ddl)
            conn.commit()
        finally:
            conn.close()
        before = _lease_store_snapshot(db_path)

        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
        with pytest.raises(
            sqlite3.DatabaseError, match="SESSION_TURN_LEASE_EPOCH_REBUILD_REFUSED"
        ):
            SessionDB(db_path=db_path)

        assert _lease_store_snapshot(db_path) == before
        assert any(row[1] == name for row in before["objects"])

    def test_pre_335_rebuild_refuses_wrong_body_for_canonical_fence_name(
        self, tmp_path, monkeypatch
    ):
        """A canonical fence name is not authority for a different trigger body."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        fence_name = _GOVERNED_TRIGGER_NAMES[0]
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(f"DROP TRIGGER {fence_name}")
            conn.execute(
                f"CREATE TRIGGER {fence_name} BEFORE INSERT "
                "ON session_turn_leases BEGIN SELECT 1; END"
            )
            conn.commit()
        finally:
            conn.close()
        before = _lease_store_snapshot(db_path)

        monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
        with pytest.raises(
            sqlite3.DatabaseError, match="SESSION_TURN_LEASE_EPOCH_REBUILD_REFUSED"
        ):
            SessionDB(db_path=db_path)

        assert _lease_store_snapshot(db_path) == before

    def test_modern_drop_postcondition_refuses_sabotaged_epoch_removal(
        self, tmp_path, monkeypatch
    ):
        """A successful DROP that leaves epoch behind cannot publish a false heal."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        original_execute = hermes_state._SerializedCursor.execute
        sabotaged = False

        def restore_epoch_after_drop(cursor, sql, parameters=()):
            nonlocal sabotaged
            result = original_execute(cursor, sql, parameters)
            if (
                sql == 'ALTER TABLE "session_turn_leases" DROP COLUMN "epoch"'
                and not sabotaged
            ):
                sabotaged = True
                original_execute(
                    cursor,
                    'ALTER TABLE "session_turn_leases" ADD COLUMN "epoch" '
                    "INTEGER NOT NULL DEFAULT 0",
                )
            return result

        monkeypatch.setattr(
            hermes_state._SerializedCursor, "execute", restore_epoch_after_drop
        )
        with pytest.raises(
            sqlite3.DatabaseError, match="SESSION_TURN_LEASE_EPOCH_HEAL_INCOMPLETE"
        ):
            SessionDB(db_path=db_path)

        assert sabotaged
        after = _lease_store_snapshot(db_path)
        assert any(column[1] == "epoch" for column in after["columns"])
        assert after["rows"] == [("seed", "holder", 1.0, 2.0, 123, 456.0, 0)]
        assert not any(
            row[0] == "table" and row[1] == "session_turn_leases_legacy_epoch"
            for row in after["objects"]
        )

    def test_modern_locked_drop_retries_through_open_boundary(self, tmp_path, monkeypatch):
        """A real busy DROP reaches SessionDB's existing open-time retry loop."""
        db_path = tmp_path / "state.db"
        _make_schema_current_epoch_variant(db_path, LEGACY_SQL)
        original_connect = hermes_state._connect_tracked_db
        original_execute = hermes_state._SerializedCursor.execute
        connection_attempts = 0
        lock_holder = None
        busy_seen = False

        def count_connections(*args, **kwargs):
            nonlocal connection_attempts
            connection_attempts += 1
            return original_connect(*args, **kwargs)

        def lock_the_real_drop(cursor, sql, parameters=()):
            nonlocal busy_seen, lock_holder
            if (
                sql == 'ALTER TABLE "session_turn_leases" DROP COLUMN "epoch"'
                and lock_holder is None
            ):
                lock_holder = sqlite3.connect(db_path)
                lock_holder.execute("BEGIN IMMEDIATE")
                try:
                    return original_execute(cursor, sql, parameters)
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                        raise
                    busy_seen = True
                    lock_holder.rollback()
                    raise
            return original_execute(cursor, sql, parameters)

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", count_connections)
        monkeypatch.setattr(hermes_state._SerializedCursor, "execute", lock_the_real_drop)
        db = None
        try:
            db = SessionDB(db_path=db_path)
            assert busy_seen
            assert connection_attempts >= 2
            assert "epoch" not in _cols(db)
        finally:
            if db is not None:
                db.close()
            if lock_holder is not None:
                lock_holder.rollback()
                lock_holder.close()

    def test_pre_335_rebuild_rollback_leaves_same_connection_usable(
        self, tmp_path, monkeypatch
    ):
        """A failed fallback rolls back without poisoning its open connection."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            conn = db._conn
            conn.execute('DROP TABLE "session_turn_leases"')
            conn.execute(LEGACY_SQL)
            for name, sql in turn_fence_trigger_definitions():
                if name.startswith("turn_fence_session_turn_leases_"):
                    conn.execute(sql)
            conn.execute(
                "INSERT INTO session_turn_leases "
                "(conversation_id, holder, acquired_at, expires_at, epoch, "
                "owner_pid, owner_pid_start) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("seed", "holder", 1.0, 2.0, 7, 123, 456.0),
            )
            conn.commit()
            _seed_compatible_fallback_objects(db_path)
            before = _lease_store_snapshot(db_path)

            monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
            with monkeypatch.context() as scoped:
                def fail_copy(_db, checkpoint):
                    if checkpoint == "copy":
                        raise sqlite3.OperationalError("forced fallback copy failure")

                scoped.setattr(
                    SessionDB,
                    "_session_turn_lease_epoch_rebuild_checkpoint",
                    fail_copy,
                )
                with pytest.raises(
                    sqlite3.OperationalError, match="forced fallback copy failure"
                ):
                    db._heal_session_turn_leases_legacy_epoch(conn.cursor())

            assert _lease_store_snapshot(db_path) == before
            assert tuple(
                conn.execute(
                    "SELECT holder, owner_pid, owner_pid_start FROM session_turn_leases "
                    "WHERE conversation_id = 'seed'"
                ).fetchone()
            ) == ("holder", 123, 456.0)

            db._heal_session_turn_leases_legacy_epoch(conn.cursor())
            assert "epoch" not in _cols(db)
            assert db.try_acquire_session_turn_lease(
                "same-connection-fallback", "holder", ttl_seconds=30
            )
        finally:
            db.close()
