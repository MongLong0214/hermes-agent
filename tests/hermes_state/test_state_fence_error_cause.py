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
    "RPC error: sqlite3.IntegrityError: state DB generation incompatible",
])
def test_fence_abort_phrases_classify_as_schema_incompatible(exc):
    assert classify_persistence_error(exc) == "schema_incompatible"


def test_bare_missing_udf_string_classifies_as_unknown():
    """No exception object survives an RPC boundary, so there is no type/__cause__ chain left to
    confirm a refusal against — the missing-UDF phrase alone is exactly as true of this build's own
    store as of another build's (see ``fence_refusal_verdict``). Bare text that only quotes it must
    not be read as a confirmed version mismatch; "unknown" is the honest answer."""
    assert classify_persistence_error("no such function: hermes_turn_fence_generation") == "unknown"


@pytest.mark.parametrize("quoted", [
    "Session state schema is newer than this Hermes build",
    "Session state turn-fence generation does not match this Hermes build",
    "Session state does not record exactly one integer schema version",
])
def test_bare_text_quoting_a_schema_head_classifies_as_unknown(quoted):
    """Plain text (an init-error slot, an RPC-wrapped error) that merely QUOTES one of the three
    ``IncompatibleSchemaError`` head phrases is not proof a refusal happened — only a real typed
    exception on the cause chain (``IncompatibleSchemaError``, or a confirmed fence abort) earns the
    version-mismatch copy; unconfirmed text is "unknown"."""
    assert classify_persistence_error(quoted) == "unknown"
    assert classify_persistence_error(f"tool output contained: {quoted}") == "unknown"


def test_missing_udf_exception_without_a_store_to_probe_is_not_a_fence_refusal():
    """A bare ``sqlite3.OperationalError("no such function: ...")`` says only that THIS connection
    never registered the turn-fence UDF — it is this build's own store just as often as another
    build's. ``classify_persistence_error`` used to bucket any such EXCEPTION as
    "schema_incompatible" by blind substring match; it must now route exception-shaped input
    through the same typed check ``hermes_state_user_copy.schema_incompatibility_cause`` and
    ``fence_refusal_verdict`` already use, which only confirms a refusal from a real store probe
    (``db_path=...``). With no store to probe, the honest answer is "cannot determine" — never the
    confident-but-wrong fence-mismatch bucket (matches ``test_fence_refusal_typed_cause.py``)."""
    exc = sqlite3.OperationalError("no such function: hermes_turn_fence_generation")
    assert classify_persistence_error(exc) != "schema_incompatible"


def test_fence_mismatch_message_never_claims_the_store_is_newer():
    from hermes_state_errors import IncompatibleSchemaError

    exc = IncompatibleSchemaError(cause="FENCE_GENERATION_MISMATCH", expected_generation=CURRENT_STAMP, actual_generation=29)

    assert "newer" not in str(exc).lower()
