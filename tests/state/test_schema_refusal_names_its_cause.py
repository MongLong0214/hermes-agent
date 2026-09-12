"""Each damage shape must produce the cause whose advice actually works.

`gateway/run.py` picks the owner-facing remedy from
`IncompatibleSchemaError.cause`, so the mapping is only as good as the cause
each raise site states. A constructed exception proves the message renders; it
proves nothing about which cause a real database produces. These cases enter
through `_validate_schema_version_scalar` and `_validate_connection_schema`
with real SQLite connections and real damage.

The distinction that matters is not build-versus-damage. It is **which command
the owner should run**, and a blind review caught the first version of this
change getting that wrong in two places:

    STORE_DAMAGED              malformed schema text -> `hermes sessions repair`
    STORE_CORRUPT              corrupt page image    -> `hermes sessions recover`
    SCHEMA_VERSION_UNREADABLE  the scalar is wrong   -> `hermes sessions recover`
    STORE_UNREADABLE           locked / transient    -> retry; assert nothing

Measured, because the first version asserted rather than measured: `hermes
sessions repair --check-only` on a store with two `schema_version` rows prints
*"opens cleanly — no repair needed"* — `_db_opens_cleanly` never reads that
table. And a sibling holding `BEGIN EXCLUSIVE` on a non-WAL store raised the
same refusal as real corruption, so a healthy store was reported damaged.

The second round then caught the mistake in the other direction, which is the
one these cases exist to keep closed: narrowing to `is_malformed_schema_error`
sent `database disk image is malformed` — SQLITE_CORRUPT, real page damage — to
`STORE_UNREADABLE`, whose message says the failure is often temporary and to try
again. That predicate gates *unattended automatic surgery*, deliberately narrow
because a wrong automatic rewrite spreads damage; it is not a statement that the
owner has nothing to do. Reading a narrow gate as a broad diagnosis is how one
wrong remedy becomes the opposite wrong remedy.
"""

import sqlite3

import pytest

from hermes_state import IncompatibleSchemaError, SessionDB, _database_error_cause
from hermes_state import _db_opens_cleanly, is_malformed_db_error, is_malformed_schema_error
from hermes_state import _validate_connection_schema, _validate_schema_version_scalar
from hermes_state_common import (
    SCHEMA_VERSION,
    TURN_FENCE_GENERATION,
    register_turn_fence_generation,
)


def _conn(*, version=None, version_sql=None, rows=1, fence=TURN_FENCE_GENERATION):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    if fence is not None:
        if fence == TURN_FENCE_GENERATION:
            register_turn_fence_generation(conn)
        else:
            conn.create_function(
                "hermes_turn_fence_generation", 0, lambda: fence
            )
    if version is not None or version_sql is not None:
        conn.execute("CREATE TABLE schema_version (version)")
        for _ in range(rows):
            if version_sql is not None:
                conn.execute(f"INSERT INTO schema_version VALUES ({version_sql})")
            else:
                conn.execute("INSERT INTO schema_version VALUES (?)", (version,))
    return conn


@pytest.mark.parametrize(
    ("kwargs", "cause", "generations"),
    [
        # The one shape a newer build fixes, and the only one that may claim so.
        ({"version": SCHEMA_VERSION + 1}, IncompatibleSchemaError.BUILD_TOO_OLD,
         (SCHEMA_VERSION, SCHEMA_VERSION + 1)),
        # Two rows: there is no single stored version to compare against.
        # This is the shape `hermes sessions repair` reports as clean.
        ({"version": SCHEMA_VERSION, "rows": 2},
         IncompatibleSchemaError.SCHEMA_VERSION_UNREADABLE, (None, None)),
        # Zero rows: same.
        ({"version": SCHEMA_VERSION, "rows": 0},
         IncompatibleSchemaError.SCHEMA_VERSION_UNREADABLE, (None, None)),
        # Non-numeric TEXT. Production DDL is `version INTEGER NOT NULL`, which
        # coerces '29' to the integer 29 — so the numeric spelling is not a
        # reachable store and this uses a value no affinity can convert.
        ({"version_sql": "'twenty-nine'"},
         IncompatibleSchemaError.SCHEMA_VERSION_UNREADABLE, (None, None)),
        ({"version": -1},
         IncompatibleSchemaError.SCHEMA_VERSION_UNREADABLE, (None, None)),
    ],
)
def test_scalar_validation_names_its_cause(kwargs, cause, generations):
    conn = _conn(**kwargs)
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_schema_version_scalar(conn)
    finally:
        conn.close()
    assert excinfo.value.cause == cause
    assert (
        excinfo.value.expected_generation,
        excinfo.value.actual_generation,
    ) == generations


def test_a_current_version_is_accepted():
    """The control: without it every assertion above passes on a broken reader."""
    conn = _conn(version=SCHEMA_VERSION)
    try:
        assert _validate_schema_version_scalar(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_a_missing_schema_version_table_is_absent_not_damaged():
    conn = _conn()
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_connection_schema(conn, allow_uninitialized_schema=False)
        assert excinfo.value.cause == IncompatibleSchemaError.SCHEMA_ABSENT
        # And the caller that accepts one gets no refusal at all.
        assert _validate_connection_schema(
            conn, allow_uninitialized_schema=True
        ) is None
    finally:
        conn.close()


@pytest.mark.parametrize("stored", [TURN_FENCE_GENERATION - 1,
                                    TURN_FENCE_GENERATION + 1])
def test_a_fence_skew_carries_both_numbers_in_either_direction(stored):
    conn = _conn(version=SCHEMA_VERSION, fence=stored)
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_connection_schema(conn)
    finally:
        conn.close()
    error = excinfo.value
    assert error.cause == IncompatibleSchemaError.FENCE_GENERATION_MISMATCH
    assert error.expected_generation == TURN_FENCE_GENERATION
    assert error.actual_generation == stored
    # A lower stored generation is a skew too, so the text must not say the
    # store is newer — that was the wording the single message hardcoded.
    assert "newer" not in str(error)


def test_a_non_integer_fence_generation_is_unreadable_not_a_skew():
    """A fence scalar returning TEXT has no generation to name a build by."""
    conn = _conn(version=SCHEMA_VERSION, fence="twenty-nine")
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_connection_schema(conn)
    finally:
        conn.close()
    assert excinfo.value.cause == IncompatibleSchemaError.SCHEMA_VERSION_UNREADABLE
    assert excinfo.value.actual_generation is None
    assert "None" not in str(excinfo.value)


def test_a_database_error_that_is_not_malformed_schema_is_unreadable():
    """An OperationalError from the SELECT is not evidence of damage."""
    conn = _conn(version=SCHEMA_VERSION, fence=None)
    try:
        # The fence scalar is unregistered, so the SELECT raises
        # OperationalError — a sqlite3.DatabaseError, and not corruption.
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_connection_schema(conn)
    finally:
        conn.close()
    assert excinfo.value.cause == IncompatibleSchemaError.STORE_UNREADABLE
    # The wording must not claim damage: the owner acts on this sentence.
    assert "damaged" not in str(excinfo.value)
    assert "malformed" not in str(excinfo.value)


def test_a_locked_store_is_unreadable_not_damaged(tmp_path):
    """The blocking case from review: a healthy store held by a sibling.

    `sqlite3.OperationalError` is a `sqlite3.DatabaseError`, so `database is
    locked` reached the same refusal as real corruption. The old single message
    then told the owner the store was damaged and to run a rewriting repair —
    both false, and the store opens on the next attempt.
    """
    db = tmp_path / "state.db"
    seed = sqlite3.connect(db, isolation_level=None)
    seed.execute("PRAGMA journal_mode=DELETE")
    seed.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    seed.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    seed.close()

    holder = sqlite3.connect(db, isolation_level=None, timeout=0)
    holder.execute("BEGIN EXCLUSIVE")
    victim = sqlite3.connect(db, isolation_level=None, timeout=0)
    register_turn_fence_generation(victim)
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_connection_schema(victim)
    finally:
        victim.close()
        holder.rollback()
        holder.close()

    assert excinfo.value.cause == IncompatibleSchemaError.STORE_UNREADABLE
    assert "damaged" not in str(excinfo.value)

    # The control that makes this a measurement: the very same store validates
    # once the lock is gone. Without it, "unreadable" could be masking a store
    # that was genuinely broken all along.
    survivor = sqlite3.connect(db, isolation_level=None)
    register_turn_fence_generation(survivor)
    try:
        assert _validate_connection_schema(survivor) == SCHEMA_VERSION
    finally:
        survivor.close()


def test_sessions_repair_does_not_see_a_schema_version_defect(tmp_path):
    """Why `SCHEMA_VERSION_UNREADABLE` must not name `hermes sessions repair`.

    Both entry points for that command gate on `_db_opens_cleanly` and return
    early with "opens cleanly — no repair needed" when it answers `None`. That
    probe checks `journal_mode`, `integrity_check`, a `sessions` count and the
    FTS index — it never reads `schema_version`, and `repair_state_db_schema`
    never writes it.

    So this pins the measurement the owner message rests on. If someone teaches
    the repair path to cover this, the assertion below fails, and the message in
    `gateway/run.py` that says "does not cover this" has to change with it.
    """
    db = tmp_path / "state.db"
    SessionDB(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    conn.close()

    # The refusal the gateway will render.
    probe = sqlite3.connect(db, isolation_level=None)
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_schema_version_scalar(probe)
    finally:
        probe.close()
    assert excinfo.value.cause == IncompatibleSchemaError.SCHEMA_VERSION_UNREADABLE

    # And the command an earlier draft of that message named, which sees nothing.
    assert _db_opens_cleanly(db) is None


@pytest.mark.parametrize(
    ("message", "cause"),
    [
        # The one class `hermes sessions repair` repairs.
        ("malformed database schema (messages_fts)",
         IncompatibleSchemaError.STORE_DAMAGED),
        # SQLITE_CORRUPT. Runtime repair fails closed here by design, so this
        # must not be answered with in-place repair — and must not be answered
        # with "try again" either, which is the regression round 2 caught.
        ("database disk image is malformed", IncompatibleSchemaError.STORE_CORRUPT),
        # Genuinely transient. `OperationalError` is a `DatabaseError`, which is
        # why these ever reached a damage claim.
        ("database is locked", IncompatibleSchemaError.STORE_UNREADABLE),
        ("disk I/O error", IncompatibleSchemaError.STORE_UNREADABLE),
        ("unable to open database file", IncompatibleSchemaError.STORE_UNREADABLE),
        ("attempt to write a readonly database",
         IncompatibleSchemaError.STORE_UNREADABLE),
    ],
)
def test_a_database_error_is_classified_by_what_the_owner_must_do(message, cause):
    """One assertion per class, because each one names a different command.

    `_database_error_cause` is the seam every `DatabaseError` site goes through,
    so this is the whole classification in one place rather than six reproduced
    failures. The two predicates it consults already existed; what was wrong was
    reading the narrow one as an answer to the broad question.
    """
    assert _database_error_cause(sqlite3.DatabaseError(message)) == cause


def test_the_two_malformed_predicates_are_not_the_same_question():
    """The control for the case above: the predicates must actually differ here.

    If `_MALFORMED_SCHEMA_MARKERS` ever grew to include the corrupt-image
    string, `STORE_CORRUPT` would become unreachable.

    This docstring used to claim the parametrised case above "would still pass,
    every message simply landing one bucket over". Review measured it and that
    is wrong: widening the narrow marker routes the corrupt string to
    `STORE_DAMAGED`, so that case fails too. The control does more than it said
    — it holds the predicates apart as a statement in its own right, rather than
    only covering a gap.
    """
    corrupt = sqlite3.DatabaseError("database disk image is malformed")
    assert is_malformed_db_error(corrupt) is True
    assert is_malformed_schema_error(corrupt) is False

    schema = sqlite3.DatabaseError("malformed database schema (messages_fts)")
    assert is_malformed_db_error(schema) is True
    assert is_malformed_schema_error(schema) is True


def test_recover_rebuilds_a_store_whose_schema_version_is_unusable(tmp_path):
    """The positive half of the remedy, which review flagged as unpinned.

    `test_sessions_repair_does_not_see_a_schema_version_defect` pins that the
    command the message does *not* name is blind to this. Nothing pinned that
    the command it *does* name works, so the owner-facing sentence rested only
    on two manual measurements. If `_CANONICAL_TABLES` stops excluding
    `schema_version`, or the destination verification changes, this fails and
    the message has to change with it.
    """
    from hermes_cli.session_recovery import (
        inspect_session_database,
        recover_session_database,
    )

    db = tmp_path / "state.db"
    SessionDB(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
    conn.close()

    probe = sqlite3.connect(db, isolation_level=None)
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_schema_version_scalar(probe)
    finally:
        probe.close()
    assert excinfo.value.cause == IncompatibleSchemaError.SCHEMA_VERSION_UNREADABLE

    assert inspect_session_database(db).get("recoverable") is True
    output = tmp_path / "recovered.db"
    recover_session_database(db, output, work_dir=tmp_path)

    rebuilt = sqlite3.connect(output, isolation_level=None)
    try:
        rows = rebuilt.execute(
            "SELECT version, typeof(version) FROM schema_version"
        ).fetchall()
    finally:
        rebuilt.close()
    # One row, correct type, correct value — and the source is left alone.
    assert rows == [(SCHEMA_VERSION, "integer")]
    assert db.exists()


def _corrupt_store(tmp_path, *, rows=4000, span=500 * 1024, stride=4096, width=300):
    """A real store with real page damage, opened the way production opens it.

    `STORE_CORRUPT` was guarded by one comparison of two constants against a
    hand-written message string. `STORE_UNREADABLE` next to it has a real store,
    a real sibling lock and a control — review pointed out the asymmetry and
    that closing it costs a few lines, because page damage is constructible.

    `timestamp` is REAL, not an ISO string. My first fixture wrote text there
    and `--allow-partial` died on `could not convert string to float`, which
    looked like a product refusal and was mine.
    """
    db = tmp_path / "state.db"
    SessionDB(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    register_turn_fence_generation(conn)
    # DELETE journalling so the damage is in the main file rather than a WAL
    # that a checkpoint would rewrite.
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('s1','cli',?)",
        (1757635200.0,),
    )
    for index in range(rows):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES ('s1','user',?,?)",
            ("x" * 200, 1757635200.0 + index),
        )
    conn.close()

    size = db.stat().st_size
    with open(db, "r+b") as handle:
        offset = stride
        while offset < min(size, span):
            handle.seek(offset)
            handle.write(b"\x00" * width)
            offset += stride
    return db


def test_page_damage_reaches_store_corrupt_through_a_real_open(tmp_path):
    """The database-level witness the class was missing."""
    db = _corrupt_store(tmp_path)

    # Through `SessionDB`, the way production opens it — not through the
    # classifier with a string someone typed.
    with pytest.raises(IncompatibleSchemaError) as excinfo:
        SessionDB(db).close()

    assert excinfo.value.cause == IncompatibleSchemaError.STORE_CORRUPT
    # And SQLite's own account of it, so the marker tuple is not being trusted
    # on faith.
    assert "malformed" in str(_db_opens_cleanly(db))
    # The transient wording must be unreachable here: this never clears.
    assert "temporary" not in str(excinfo.value)


def test_recovering_a_corrupt_store_needs_allow_partial(tmp_path):
    """Why the owner message names `--allow-partial` rather than implying it.

    `recoverable` requires `sessions` and `messages` to be *completely*
    readable, and page damage is the class that breaks exactly that. A plain
    `--output` run refuses; the flag salvages what survives. The message
    promises "rebuild what it can", which is this flag's contract, so naming it
    is what makes the promise true.
    """
    from hermes_cli.session_recovery import (
        SessionRecoverySourceError,
        inspect_session_database,
        recover_session_database,
    )

    db = _corrupt_store(tmp_path)

    report = inspect_session_database(db, work_dir=tmp_path)
    assert report.get("recoverable") is False

    with pytest.raises(SessionRecoverySourceError) as refusal:
        recover_session_database(db, tmp_path / "plain.db", work_dir=tmp_path)
    # The CLI hands over the flag, which is why this costs a step rather than
    # stranding the owner — and why the message should have said it first.
    assert "--allow-partial" in str(refusal.value)

    recover_session_database(
        db, tmp_path / "salvaged.db", work_dir=tmp_path, allow_partial=True
    )
    rebuilt = sqlite3.connect(tmp_path / "salvaged.db", isolation_level=None)
    try:
        messages = rebuilt.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        sessions = rebuilt.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        rebuilt.close()

    # Partial means partial: some rows are gone, and the point is that most are
    # not. Measured at 3,433 of 4,000 on this damage; asserting a band rather
    # than a number, because the exact count is SQLite's business.
    assert sessions == 1
    assert 0 < messages < 4000
    assert messages > 2000
