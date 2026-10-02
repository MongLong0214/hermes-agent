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


def _raw_write_error(db, generation):
    conn = sqlite3.connect(str(db), isolation_level=None)
    if generation is not None:
        conn.create_function("hermes_turn_fence_generation", 0, lambda: generation)
    try:
        with pytest.raises(sqlite3.DatabaseError) as caught:
            conn.execute("UPDATE sessions SET title = 'refused' WHERE id = 'fx-alpha'")
    finally:
        conn.close()
    return caught.value


def test_a_fence_refused_write_gets_the_generation_mismatch_reply(tmp_path):
    db = build_store(tmp_path / "fenced" / "state.db", "fenced1030")
    mismatch = _reply(_refusal_from_open(tmp_path, "FENCE_GENERATION_MISMATCH"))
    assert _lead(mismatch) != _lead(_reply(RuntimeError("unrelated failure")))
    assert _reply(_raw_write_error(db, 29)) == mismatch


def test_a_write_without_the_fence_function_is_not_blamed_on_another_build(tmp_path):
    """This build's own store: the error names an unregistered writer, not the store's lineage."""
    db = build_store(tmp_path / "fenced" / "state.db", "fenced1030")
    generic = _lead(_reply(RuntimeError("unrelated failure")))
    assert _lead(_reply(_raw_write_error(db, None))) == generic


@pytest.mark.parametrize("quoted", [
    "no such function: hermes_turn_fence_generation",
    "state DB generation incompatible",
    "Session state turn-fence generation does not match this Hermes build",
])
def test_an_unrelated_error_quoting_fence_text_gets_the_generic_reply(quoted):
    generic = _lead(_reply(RuntimeError("unrelated failure")))
    assert _lead(_reply(RuntimeError(f"tool output contained: {quoted}"))) == generic


def test_a_wrapped_refusal_keeps_its_own_cause():
    """A layer that wraps the refusal (``raise ... from err``) keeps the typed cause on the chain; the
    classifier must read it there, not fall back to "unknown" and recommend `doctor --fix`."""
    from hermes_state_errors import IncompatibleSchemaError
    from hermes_state_user_copy import describe_schema_refusal, describe_storage_failure

    for cause in _REMEDY_BY_CAUSE:
        refusal = IncompatibleSchemaError(cause=cause, expected_generation=1030, actual_generation=29)
        try:
            try:
                raise refusal
            except IncompatibleSchemaError as inner:
                raise RuntimeError("SessionStore SQLite handle unavailable") from inner
        except RuntimeError as wrapped:
            assert describe_storage_failure(wrapped) == describe_schema_refusal(cause), cause


def _startup_broadcast(monkeypatch, startup_exc) -> str:
    """The gateway's startup-failure warning, recorded the way __init__ records it, as sent home."""
    runner = object.__new__(GatewayRunner)
    runner._session_db_handle_cache = None  # no pre-broadcast re-open: the startup failure stands
    runner._record_session_db_init_error(startup_exc)
    sent: list = []
    monkeypatch.setattr(runner, "_home_channel_transports", lambda: [("telegram", {}, "home", object())])

    async def _capture(_platform, _home, _transport, message, _fmt):
        sent.append(message)

    monkeypatch.setattr(runner, "_send_home_channel_message", _capture)
    asyncio.run(runner._send_session_db_warning_notifications())
    assert len(sent) == 1, sent
    return sent[0]


_OTHER_VERSION = "belongs to a different Hermes version"


def test_startup_warning_reads_the_exception_not_its_text(tmp_path, monkeypatch):
    """The startup path used to keep only ``str(exc)``, so the classifier saw bare text and matched
    phrases: a missing-UDF error on this build's OWN store, or any error quoting fence text, told the
    owner their history belongs to another Hermes version and to update/restart."""
    own = build_store(tmp_path / "own" / "state.db", "fenced1030")
    missing_udf = _raw_write_error(own, None)
    assert "no such function" in str(missing_udf)
    for exc in (missing_udf, RuntimeError("tool output contained: state DB generation incompatible")):
        message = _startup_broadcast(monkeypatch, exc)
        assert _OTHER_VERSION not in message, (exc, message)
    # A genuine refusal, raw or wrapped by the opener, still gets its own copy.
    refusal = _refusal_from_open(tmp_path, "FENCE_GENERATION_MISMATCH")
    assert _OTHER_VERSION in _startup_broadcast(monkeypatch, refusal)
    try:
        raise RuntimeError("SessionStore SQLite handle unavailable") from refusal
    except RuntimeError as wrapped:
        assert _OTHER_VERSION in _startup_broadcast(monkeypatch, wrapped)


def test_tui_store_failure_reads_the_exception_not_its_text(tmp_path, monkeypatch):
    """Same for the TUI/Desktop backend: its open failure is recorded at first use and every
    "storage unavailable" reply is built from it."""
    import hermes_state_registry
    from tui_gateway import server

    own = build_store(tmp_path / "own" / "state.db", "fenced1030")
    missing_udf = _raw_write_error(own, None)

    def _failing_acquire(_db_path=None):
        raise missing_udf

    monkeypatch.setattr(hermes_state_registry, "acquire", _failing_acquire)
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr(server, "_db_error", None)
    assert server._get_db() is None
    reply = server._db_unavailable_error("r1", code=5000)
    assert reply["error"]["data"]["cause"] != "schema_incompatible", reply
    assert _OTHER_VERSION not in reply["error"]["message"], reply
