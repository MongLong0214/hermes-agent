"""A generation refusal is its own persistence cause: callers must not tell the user the disk is
full, retry it as a lock, or route it into corruption repair."""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state_errors import PERSISTENCE_ERROR_CAUSES, classify_persistence_error
from tests.hermes_state.fork_store_fixture import CURRENT_STAMP, FUTURE_STAMP


def test_incompatible_schema_error_classifies_as_schema_incompatible():
    from hermes_state_errors import IncompatibleSchemaError

    exc = IncompatibleSchemaError(cause="BUILD_TOO_OLD", expected_generation=CURRENT_STAMP, actual_generation=FUTURE_STAMP)

    assert classify_persistence_error(exc) == "schema_incompatible"
    assert "schema_incompatible" in PERSISTENCE_ERROR_CAUSES
    assert not isinstance(exc, sqlite3.DatabaseError)


@pytest.mark.parametrize("exc", [
    sqlite3.IntegrityError("state DB generation incompatible"),
    sqlite3.OperationalError("no such function: hermes_turn_fence_generation"),
    "RPC error: sqlite3.IntegrityError: state DB generation incompatible",
])
def test_fence_abort_phrases_classify_as_schema_incompatible(exc):
    assert classify_persistence_error(exc) == "schema_incompatible"


def test_fence_mismatch_message_never_claims_the_store_is_newer():
    from hermes_state_errors import IncompatibleSchemaError

    exc = IncompatibleSchemaError(cause="FENCE_GENERATION_MISMATCH", expected_generation=CURRENT_STAMP, actual_generation=29)

    assert "newer" not in str(exc).lower()
