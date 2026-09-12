"""Each damage shape must produce the cause whose advice actually works.

`gateway/run.py` picks the owner-facing remedy from
`IncompatibleSchemaError.cause`, so the mapping is only as good as the cause
each raise site states. A constructed exception proves the message renders; it
proves nothing about which cause a real database produces. These cases enter
through `_validate_schema_version_scalar` and `_validate_connection_schema`
with real SQLite connections and real damage.

The distinction that matters: `BUILD_TOO_OLD` and `FENCE_GENERATION_MISMATCH`
are skews a different build fixes. Every other shape here is damage that no
build opens, and answering it with build advice is what `#784` was about.
"""

import sqlite3

import pytest

from hermes_state import IncompatibleSchemaError
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
        ({"version": SCHEMA_VERSION, "rows": 2},
         IncompatibleSchemaError.STORE_DAMAGED, (None, None)),
        # Zero rows: same.
        ({"version": SCHEMA_VERSION, "rows": 0},
         IncompatibleSchemaError.STORE_DAMAGED, (None, None)),
        # Stored as TEXT. A build at that number would still not open it.
        ({"version_sql": "'29'"}, IncompatibleSchemaError.STORE_DAMAGED,
         (None, None)),
        ({"version": -1}, IncompatibleSchemaError.STORE_DAMAGED, (None, None)),
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


def test_a_non_integer_fence_generation_is_damage_not_a_skew():
    """A fence scalar returning TEXT has no generation to name a build by."""
    conn = _conn(version=SCHEMA_VERSION, fence="twenty-nine")
    try:
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_connection_schema(conn)
    finally:
        conn.close()
    assert excinfo.value.cause == IncompatibleSchemaError.STORE_DAMAGED
    assert excinfo.value.actual_generation is None
    assert "None" not in str(excinfo.value)


def test_a_database_error_from_the_select_is_damage():
    """No schema_master at all: the validating SELECT itself fails."""
    conn = _conn(version=SCHEMA_VERSION, fence=None)
    try:
        # The fence scalar is unregistered, so the SELECT raises
        # OperationalError — a sqlite3.DatabaseError, not a version skew.
        with pytest.raises(IncompatibleSchemaError) as excinfo:
            _validate_connection_schema(conn)
    finally:
        conn.close()
    assert excinfo.value.cause == IncompatibleSchemaError.STORE_DAMAGED
