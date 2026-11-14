"""The gateway's recovery copy when the agent reports ``agent_persisted=False``.

False means SOME write of the turn failed, not that every row is unwritten: the turn-start flush can
already hold the ask. The gateway writes only the rows the store does not hold yet, so a partial
failure leaves one ask and the missing answer, never a second ask.

Real GatewayRunner, DiscordAdapter, SessionAuthority and SessionDB, and a real AIAgent on the same
SessionDB; only the model and the failing appends are stand-ins.
"""
import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
from tests.agent.test_run_agent import _mock_response, _mock_tool_call

ASK = "summarise the whole repository history"
FINAL = "Three releases, one revert."


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
    entry = runner.session_store.lookup_by_session_key(authority.sessions[receipt.ref.session_id].route)
    db = authority.db
    db.append_message(entry.session_id, "user", "hello")
    db.append_message(entry.session_id, "assistant", "hi there")
    try:
        yield SimpleNamespace(runner=runner, source=source, key=entry.session_key, entry=entry, db=db)
    finally:
        from gateway.session_runtime_workers import mutation_tasks
        await asyncio.gather(*mutation_tasks(authority))
        db.close()


def _rows(gw):
    return [(m["role"], m["content"]) for m in gw.db.get_messages(gw.entry.session_id)]


def _agent(gw):
    from run_agent import AIAgent
    agent = AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True, session_db=gw.db, session_id=gw.entry.session_id)
    agent._session_db_created = True
    agent.client = MagicMock()
    agent._cached_system_prompt, agent._use_prompt_caching, agent._skip_mcp_refresh = "You are helpful.", False, True
    agent.compression_enabled = agent.save_trajectories = False
    agent.tool_delay = 0
    return agent


def _break_writes_after(db, landed):
    """The agent's appends after the first ``landed`` raise, as a locked or full disk does."""
    real, calls = db.append_messages_batch, []

    def append(*args, **kwargs):
        calls.append(None)
        if len(calls) > landed:
            raise sqlite3.OperationalError("database is locked")
        return real(*args, **kwargs)

    db.append_messages_batch = append
    return lambda: delattr(db, "append_messages_batch")


def _history(gw):
    return gw.db.get_messages_as_conversation(gw.entry.session_id)


def _run_agent_turn(gw, *responses, landed):
    """One agent turn on the shared store whose appends after the first ``landed`` fail."""
    agent = _agent(gw)
    agent.valid_tool_names = {"web_search"}
    agent.client.chat.completions.create.side_effect = list(responses)
    history = _history(gw)
    restore = _break_writes_after(gw.db, landed)
    try:
        with patch.object(agent, "_cleanup_task_resources"), patch("model_tools.handle_function_call", return_value="ok"):
            result = agent.run_conversation(ASK, conversation_history=history)
    finally:
        restore()
    return result, history


async def _gateway_persists(gw, result, history):
    """The gateway's own post-turn order (gateway/run_turn.py): classify, then persist."""
    runner = gw.runner
    event = MessageEvent(text=ASK, source=gw.source, message_id=None)
    prepared = SimpleNamespace(history=history, persist_user_message=None, message_text=ASK,
                               persist_user_timestamp=None, persist_user_display_kind=None, persistence_owner=None)
    failed_early, hidden, overflow = runner._hmwa_classify_turn_failure(result, history, gw.entry)
    response, entry = await runner._hmwa_compression_exhaustion_reset(
        result, result["final_response"], gw.entry, gw.key, gw.source)
    await runner._hmwa_persist_turn_transcript(
        event=event, source=gw.source, session_entry=entry, session_key=gw.key, agent_result=result,
        agent_messages=result["messages"], prepared=prepared, response=response, agent_failed_early=failed_early,
        hidden_reasoning_incomplete=hidden, is_context_overflow_failure=overflow)


@pytest.mark.asyncio
async def test_a_partial_failure_recovers_only_the_rows_the_agent_never_landed(gw):
    """The turn-start flush lands the ask, the answer's flush fails: one ask, and the answer recovered."""
    result, history = _run_agent_turn(gw, _mock_response(content=FINAL, finish_reason="stop"), landed=1)
    assert result["agent_persisted"] is False and _rows(gw)[-1] == ("user", ASK)  # reach

    await _gateway_persists(gw, result, history)

    assert _rows(gw) == [("user", "hello"), ("assistant", "hi there"), ("user", ASK), ("assistant", FINAL)]


@pytest.mark.asyncio
async def test_a_failed_turn_never_reinserts_an_ask_the_store_holds(gw):
    """The tool round's flush fails after the ask landed: a failed turn whose user row is already held."""
    tool_round = _mock_response(content="", finish_reason="tool_calls",
                                tool_calls=[_mock_tool_call(name="web_search", arguments="{}", call_id="c1")])
    result, history = _run_agent_turn(gw, tool_round, landed=1)
    assert result["failed"] is True and result["agent_persisted"] is False  # reach

    await _gateway_persists(gw, result, history)

    rows = _rows(gw)
    assert rows.count(("user", ASK)) == 1, rows
    assert rows[:3] == [("user", "hello"), ("assistant", "hi there"), ("user", ASK)]
    assert rows[-1][0] == "assistant"  # the failed turn is closed


@pytest.mark.asyncio
async def test_a_stored_row_awaiting_its_rewrite_is_not_inserted_again(gw):
    """The finalizer's blank-row fill drops the marker on a row the store holds (``_row_id``); its rewrite
    failed. Inserting it would add a second assistant row. A marked row is held whether or not it
    carries its row id."""
    sid = gw.entry.session_id
    history = _history(gw)
    gw.db.append_message(sid, "user", ASK)
    blank_id = gw.db.append_message(sid, "assistant", "")
    messages = [*history, {"role": "user", "content": ASK, "_db_persisted": True},
                {"role": "assistant", "content": FINAL, "_row_id": blank_id}]
    result = {"final_response": FINAL, "messages": messages, "completed": True, "agent_persisted": False}

    await _gateway_persists(gw, result, history)

    assert _rows(gw) == [("user", "hello"), ("assistant", "hi there"), ("user", ASK), ("assistant", "")]


# --- compression exhaustion when the agent's writes fail too -------------------------------------------

def _overflow_end(gw, messages, history):
    """The #106260 overflow-terminal exit: end_turn with compression_exhausted, through the real recovery."""
    from agent.turn_retry_state import TurnRetryState
    from agent.turn_truncation import recover_from_truncation

    agent = _agent(gw)
    stub = SimpleNamespace(id="overflow-stub", usage=None, _overflow_terminal=True, choices=[])
    with patch.object(agent, "_cleanup_task_resources"):
        return recover_from_truncation(
            agent, stub, "length", TurnRetryState(), messages=messages, conversation_history=history,
            api_kwargs=None, api_call_count=1, effective_task_id="t", current_turn_user_idx=len(history),
            length_continue_retries=0, truncated_response_parts=[], truncated_tool_call_retries=0,
            retry_count=0, compression_attempts=0,
        ).result


@pytest.mark.asyncio
async def test_exhaustion_after_a_failed_turn_start_writes_nothing_and_retracts_nothing(gw):
    """Both of the turn's flushes fail: the ask never reached the store, the gateway writes nothing for an
    overflow, and its retract (gated on agent_persisted) is skipped. The transcript is as before."""
    history = _history(gw)
    agent, messages = _agent(gw), [*history, {"role": "user", "content": ASK}]
    restore = _break_writes_after(gw.db, 0)
    try:
        assert agent._flush_messages_to_session_db(messages, history) is False  # turn-start fails
        result = _overflow_end(gw, messages, history)
    finally:
        restore()
    assert result["compression_exhausted"] is True and result["agent_persisted"] is False  # reach

    await _gateway_persists(gw, result, history)

    assert _rows(gw) == [("user", "hello"), ("assistant", "hi there")]


@pytest.mark.asyncio
async def test_exhaustion_whose_closing_row_fails_keeps_the_tool_record_unclosed(gw):
    """The ask and a tool round landed; only end_turn's closing row fails. The store's tail is the tool
    row, so no retract applies either way; the overflow branch writes nothing, so the tail stays open."""
    history = _history(gw)
    agent = _agent(gw)
    call = {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
    messages = [*history, {"role": "user", "content": ASK}, {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": "ok", "tool_call_id": "c1"}]
    assert agent._flush_messages_to_session_db(messages, history) is True
    restore = _break_writes_after(gw.db, 0)
    try:
        result = _overflow_end(gw, messages, history)
    finally:
        restore()
    assert result["compression_exhausted"] is True and result["agent_persisted"] is False  # reach

    await _gateway_persists(gw, result, history)

    assert [role for role, _ in _rows(gw)] == ["user", "assistant", "user", "assistant", "tool"]
