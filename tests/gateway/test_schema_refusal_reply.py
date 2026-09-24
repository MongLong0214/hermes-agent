"""A gateway turn that dies on a state.db this build refuses gets the refusal's own remedy, never
the generic "something went wrong, /retry" reply: a retry meets the same refusal every time.

The errors are the real ones: a SessionDB open of a store this build refuses, and a raw write
on a fenced store without (or with the wrong) fence function."""

from __future__ import annotations

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.hermes_state.fork_store_fixture import REFUSED_KINDS, build_store

# One store per refusal cause.
_KIND_BY_CAUSE = {cause: kind for kind, cause in reversed(list(REFUSED_KINDS.items()))}
# The action each cause's reply must name: what the owner can actually run.
_REMEDY_BY_CAUSE = {
    "BUILD_TOO_OLD": "`hermes update`",
    "FENCE_GENERATION_MISMATCH": "doctor`",
    "SCHEMA_VERSION_UNREADABLE": "sessions recover",
}


def _reply(err) -> str:
    runner = object.__new__(GatewayRunner)

    async def stop_typing(event, source):
        return None

    runner._hmwa_stop_typing_for_turn = stop_typing
    runner._session_state = lambda key: SimpleNamespace(turn=SimpleNamespace(agent=None))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="c", user_id="u")
    prepared = runner._PreparedTurn([], "", None, None, None, None)
    return asyncio.run(runner._hmwa_agent_error_reply(
        err, MessageEvent(text="x", source=source), source, None, "k", prepared,
    ))


def _refusal_from_open(tmp_path, cause):
    from hermes_state import SessionDB

    db = build_store(tmp_path / cause / "state.db", _KIND_BY_CAUSE[cause])
    with pytest.raises(Exception) as caught:
        SessionDB(db_path=db).close()
    assert getattr(caught.value, "cause", None) == cause, caught.value
    return caught.value


def _lead(reply: str) -> str:
    return reply.split("\n\n")[0]


def test_each_refusal_cause_gets_its_own_remedy_not_the_generic_reply(tmp_path):
    generic = _lead(_reply(RuntimeError("unrelated failure")))
    leads = {}
    for cause in _REMEDY_BY_CAUSE:
        err = _refusal_from_open(tmp_path, cause)
        lead = _lead(_reply(err))
        assert lead != generic and generic not in lead, (cause, lead)
        assert "/retry" not in lead, lead
        assert _REMEDY_BY_CAUSE[cause] in lead, (cause, lead)
        # Plain words for the owner: no store path, and no generation numbers from the refusal.
        assert str(tmp_path) not in lead and "state.db" not in lead.replace("<state.db>", "")
        for generation in (err.expected_generation, err.actual_generation):
            assert generation is None or str(generation) not in lead, (cause, lead)
        leads[cause] = lead
    assert len(set(leads.values())) == len(leads), leads


def test_a_fence_refused_write_gets_the_generation_mismatch_reply(tmp_path):
    db = build_store(tmp_path / "fenced" / "state.db", "fenced1030")
    mismatch = _reply(_refusal_from_open(tmp_path, "FENCE_GENERATION_MISMATCH"))
    assert _lead(mismatch) != _lead(_reply(RuntimeError("unrelated failure")))
    for generation in (None, 29):
        conn = sqlite3.connect(str(db), isolation_level=None)
        if generation is not None:
            conn.create_function("hermes_turn_fence_generation", 0, lambda: generation)
        try:
            with pytest.raises(sqlite3.DatabaseError) as caught:
                conn.execute("UPDATE sessions SET title = 'refused' WHERE id = 'fx-alpha'")
        finally:
            conn.close()
        assert _reply(caught.value) == mismatch, (generation, caught.value)
