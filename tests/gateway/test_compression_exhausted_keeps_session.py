"""A gateway turn that exhausts context compression keeps its conversation.

The auto-reset (a route rotation that re-homed queued messages + a conversation-scope clear) replaced
the chat's session with a fresh one, so the existing conversation and everything scoped to it were dropped.
Now the session stays: same ``session_id``, no new session row, nothing re-homed, queued messages
stay queued on it. The unanswered ask the agent flushed at turn start is retracted owner-side once
the turn lease is released (otherwise the next ask merges into it and it replays on every later
request), exactly once, and never for a lock-contended defer. The reply points at /compress.

Real GatewayRunner, DiscordAdapter, SessionAuthority and SessionDB; the agent run is a stand-in.
"""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from plugins.platforms.discord.adapter import DiscordAdapter

ASK = "summarise the whole repository history"
EXHAUSTED = {"final_response": "copy", "failed": True, "partial": True, "completed": False,
             "compression_exhausted": True, "failure_reason": "context_overflow", "messages": []}


@pytest_asyncio.fixture
async def gw(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="fixture-reply")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fixture")
    runner._wire_adapter_handlers(adapter)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    event = MessageEvent(text="hello", source=source, message_id="m-hello")
    with native_callback(runner, event, get_hermes_home()):
        receipt = await authority.admit_native(event)
    route = authority.sessions[receipt.ref.session_id].route
    entry = runner.session_store.lookup_by_session_key(route)
    db = authority.db
    db.append_message(entry.session_id, "user", "hello")
    db.append_message(entry.session_id, "assistant", "hi there")

    def no_rotation(*_a, **_k):
        raise AssertionError("compression exhaustion must not reset or re-home the session")

    monkeypatch.setattr(runner.session_store, "reset_session", no_rotation)
    try:
        yield SimpleNamespace(runner=runner, adapter=adapter, authority=authority, source=source,
                              key=entry.session_key, entry=entry, db=db)
    finally:
        from gateway.session_runtime_workers import mutation_tasks
        await asyncio.gather(*mutation_tasks(authority))
        db.close()


def _conversation(gw):
    return [(m["role"], m["content"]) for m in gw.db.get_messages_as_conversation(gw.entry.session_id)]


async def _exhausted_turn(gw, result, *, flush_ask=True, message_id="m-ask"):
    """The gateway's own post-turn order (gateway/run_turn.py): classify, keep/reset, persist."""
    runner = gw.runner
    if flush_ask:  # the agent's turn-start flush
        gw.db.append_message(gw.entry.session_id, "user", ASK)
    event = MessageEvent(text=ASK, source=gw.source, message_id=message_id)
    prepared = SimpleNamespace(history=[{"role": "user", "content": "hello"}], persist_user_message=None,
                               message_text=ASK, persist_user_timestamp=None, persist_user_display_kind=None,
                               persistence_owner=None)
    failed_early, hidden, overflow = runner._hmwa_classify_turn_failure(result, prepared.history, gw.entry)
    response, entry = await runner._hmwa_compression_exhaustion_reset(
        result, result["final_response"], gw.entry, gw.key, gw.source)
    await runner._hmwa_persist_turn_transcript(
        event=event, source=gw.source, session_entry=entry, session_key=gw.key, agent_result=result,
        agent_messages=[], prepared=prepared, response=response, agent_failed_early=failed_early,
        hidden_reasoning_incomplete=hidden, is_context_overflow_failure=overflow)
    return response, entry


@pytest.mark.asyncio
async def test_exhausted_turn_keeps_the_session_and_retracts_the_unanswered_ask(gw):
    sessions_before = gw.db.session_count(include_archived=True, include_subagents=True)

    _response, entry = await _exhausted_turn(gw, dict(EXHAUSTED))

    assert entry.session_id == gw.entry.session_id
    assert gw.runner.session_store.peek_session_id(gw.key) == gw.entry.session_id
    assert gw.db.session_count(include_archived=True, include_subagents=True) == sessions_before
    assert _conversation(gw) == [("user", "hello"), ("assistant", "hi there")]
    assert gw.db.get_session(gw.entry.session_id).get("end_reason") in (None, "")


@pytest.mark.asyncio
async def test_a_retry_of_the_same_turn_never_retracts_twice(gw):
    await _exhausted_turn(gw, dict(EXHAUSTED))
    await _exhausted_turn(gw, dict(EXHAUSTED), flush_ask=False)  # redelivery: nothing new was flushed
    assert _conversation(gw) == [("user", "hello"), ("assistant", "hi there")]

    await _exhausted_turn(gw, dict(EXHAUSTED))  # the user resends and it exhausts again
    assert _conversation(gw) == [("user", "hello"), ("assistant", "hi there")]


@pytest.mark.asyncio
async def test_an_exhausted_turn_that_ran_tools_keeps_its_record(gw):
    sid = gw.entry.session_id
    gw.db.append_message(sid, "user", ASK)
    gw.db.append_message(sid, "assistant", None, tool_calls=[
        {"id": "t1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}])
    gw.db.append_message(sid, "tool", "ran", tool_call_id="t1")
    await _exhausted_turn(gw, dict(EXHAUSTED), flush_ask=False)
    assert [role for role, _ in _conversation(gw)][-3:] == ["user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_the_reply_points_at_compress_and_claims_no_reset(gw):
    from agent.i18n import t
    from agent.turn_failure_copy import site_copy

    copy = site_copy("context_overflow", model="fixture-model")
    assert "/compress" in copy and "not answered" in copy and "send it again" in copy
    assert "reset" not in copy.lower()

    response, _entry = await _exhausted_turn(gw, dict(EXHAUSTED, final_response=copy))
    assert response == copy
    assert "fresh session" not in response.lower()
    # The auto-reset notice is gone from the catalog: ``t`` falls back to the bare key.
    key = "gateway.session.auto_reset_context_exhausted"
    assert t(key) == key


@pytest.mark.asyncio
async def test_queued_messages_stay_queued_on_the_same_session(gw):
    runner, adapter = gw.runner, gw.adapter
    head, second = (MessageEvent(text=text, source=gw.source, message_id=text) for text in ("Q1", "Q2"))
    runner._enqueue_fifo(gw.key, head, adapter)
    runner._enqueue_fifo(gw.key, second, adapter)
    runner._session_state(gw.key).conversation.goal_compression_recovery = {"attempts": 1}

    _response, entry = await _exhausted_turn(gw, dict(EXHAUSTED))

    assert entry.session_id == gw.entry.session_id
    assert adapter._pending_messages[gw.key] is head
    assert runner._overflow_queue(gw.key) == [second]
    # No conversation boundary ran: conversation-scoped state (the goal's bound) survives.
    assert runner._session_state(gw.key).conversation.goal_compression_recovery == {"attempts": 1}


@pytest.mark.asyncio
async def test_compression_deferred_never_retracts(gw):
    deferred = dict(EXHAUSTED, compression_deferred=True)
    _response, entry = await _exhausted_turn(gw, deferred)
    assert entry.session_id == gw.entry.session_id
    assert ("user", ASK) in _conversation(gw)
