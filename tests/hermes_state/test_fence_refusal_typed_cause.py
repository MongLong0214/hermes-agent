"""Fence remediation is chosen from the typed SQLite error the fence raised, never from prose.

The gateway turn explainer hands ``schema_incompatibility_cause`` whatever the agent raised. It used to
search ``str(exc)`` for the fence phrases, so any exception that merely QUOTED "no such function:
hermes_turn_fence_generation" (a tool result, a log line, a wrapped provider error) told the owner the
store belongs to another Hermes version. A write made without the fence function on this build's OWN
store got the same copy: that error says the writing connection never registered the function, not
which build fenced the store — only a lineage probe can say that.

The fence errors here are real: raw writes against real fenced stores.
"""

from __future__ import annotations

import sqlite3

import pytest

from hermes_state_fence import fence_refusal_verdict, register_turn_fence_generation
from hermes_state_user_copy import schema_incompatibility_cause
from tests.hermes_state.fork_store_fixture import build_store

_FENCE_MISMATCH = "FENCE_GENERATION_MISMATCH"


def _raw_write_error(db, *, register: bool) -> sqlite3.DatabaseError:
    conn = sqlite3.connect(str(db), isolation_level=None)
    if register:
        register_turn_fence_generation(conn)  # this build's writer
    try:
        with pytest.raises(sqlite3.DatabaseError) as caught:
            conn.execute("UPDATE sessions SET title = 'refused' WHERE id = 'fx-alpha'")
    finally:
        conn.close()
    return caught.value


def test_a_foreign_fence_abort_is_a_generation_mismatch(tmp_path):
    db = build_store(tmp_path / "fork" / "state.db", "fork29")
    err = _raw_write_error(db, register=True)

    assert isinstance(err, sqlite3.IntegrityError), err
    assert fence_refusal_verdict(err) is not None
    assert schema_incompatibility_cause(err) == _FENCE_MISMATCH
    # The typed error survives being wrapped by the layer that hit it.
    try:
        try:
            raise err
        except sqlite3.DatabaseError as inner:
            raise RuntimeError("transcript append failed") from inner
    except RuntimeError as wrapped:
        assert schema_incompatibility_cause(wrapped) == _FENCE_MISMATCH


def test_a_missing_fence_function_on_this_builds_store_is_not_a_generation_mismatch(tmp_path):
    own = build_store(tmp_path / "own" / "state.db", "fenced1030")
    err = _raw_write_error(own, register=False)

    assert isinstance(err, sqlite3.OperationalError) and "no such function" in str(err), err
    assert schema_incompatibility_cause(err) is None, "an unregistered writer was blamed on another build"
    assert fence_refusal_verdict(err) is None
    # With the store at hand, its lineage decides: this build's own store is not refused...
    assert fence_refusal_verdict(err, db_path=own) is None
    # ...while a store another build fenced is.
    foreign = build_store(tmp_path / "fork" / "state.db", "fork29")
    assert fence_refusal_verdict(_raw_write_error(foreign, register=False), db_path=foreign) is not None


@pytest.mark.parametrize("quoted", [
    "no such function: hermes_turn_fence_generation",
    "state DB generation incompatible",
    "Session state turn-fence generation does not match this Hermes build",
])
def test_an_unrelated_exception_quoting_fence_text_is_not_a_fence_refusal(quoted):
    err = RuntimeError(f"tool output contained: {quoted}")

    assert fence_refusal_verdict(err) is None
    assert schema_incompatibility_cause(err) is None
