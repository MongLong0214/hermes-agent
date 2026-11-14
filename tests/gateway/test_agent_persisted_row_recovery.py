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


# --- agent_persisted missing or not a bool is unknown, never True -------------------------------------

def _turn_start(gw, agent, *, lands=True):
    """The agent's turn-start flush of this turn's ask: ``(messages, history)`` as the loop holds them."""
    history = _history(gw)
    messages = [*history, {"role": "user", "content": ASK}]
    if lands:
        assert agent._flush_messages_to_session_db(messages, history) is True
    return messages, history


@pytest.mark.asyncio
@pytest.mark.parametrize("ask_landed", [True, False])
async def test_retry_exhaustion_keeps_one_ask_and_recovers_a_lost_one(gw, ask_landed):
    """max_retries_exhausted_result reports no agent_persisted. A landed ask is not inserted again; an
    ask the agent never stored is the gateway's recovery copy, not a lost row."""
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import max_retries_exhausted_result

    class _Http(Exception):
        status_code = 503

    agent = _agent(gw)
    messages, history = _turn_start(gw, agent, lands=ask_landed)
    error = _Http("HTTP 503: upstream unavailable")
    restore = _break_writes_after(gw.db, 0)
    try:
        result = max_retries_exhausted_result(
            agent, error, classify_api_error(error, provider="openrouter", model="m"), attempts=3,
            is_rate_limited=False, error_msg=str(error).lower(), api_kwargs=None, api_messages=[], messages=messages,
            conversation_history=history, api_call_count=1, approx_tokens=10, provider="openrouter",
            base_url="https://openrouter.ai/api/v1", model="m")
    finally:
        restore()
    assert result["failed"] is True and "agent_persisted" not in result  # reach: the unknown case

    await _gateway_persists(gw, result, history)

    rows = _rows(gw)
    assert rows[:3] == [("user", "hello"), ("assistant", "hi there"), ("user", ASK)]
    assert rows.count(("user", ASK)) == 1 and [role for role, _ in rows[3:]] == ["assistant"]


@pytest.mark.asyncio
async def test_an_interrupt_recovers_its_unstored_closing_row_once(gw):
    """abort_turn_on_interrupt closes an open tool round, its flush fails, and it reports no
    agent_persisted. The rows the agent stored are not inserted again; the closing row is recovered."""
    from agent.turn_recovery import abort_turn_on_interrupt

    agent = _agent(gw)
    history = _history(gw)
    call = {"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
    messages = [*history, {"role": "user", "content": ASK}, {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "content": "ok", "tool_call_id": "c1"}]
    assert agent._flush_messages_to_session_db(messages, history) is True
    restore = _break_writes_after(gw.db, 0)
    try:
        result = abort_turn_on_interrupt(agent, messages, history, 1, abort_message="Interrupted during retry wait.",
                                         interrupt_text="Stopped before the next step.")
    finally:
        restore()
    assert result["interrupted"] is True and "agent_persisted" not in result  # reach

    await _gateway_persists(gw, result, history)

    assert _rows(gw) == [("user", "hello"), ("assistant", "hi there"), ("user", ASK), ("assistant", ""), ("tool", "ok"),
                         ("assistant", "Stopped before the next step.")]


def _refusal(gw, agent, messages, history):
    """The provider's safety refusal arriving as a 400, through the real non-retryable exit."""
    from agent.error_classifier import classify_api_error
    from agent.turn_recovery import nonretryable_client_error_result

    class _Http(Exception):
        status_code = 400

    error = _Http("HTTP 400: your request was flagged by our safety system")
    classified = classify_api_error(error, provider="openrouter", model="m")
    return nonretryable_client_error_result(
        agent, error, classified, status_code=400, api_kwargs=None, api_messages=[], messages=messages,
        conversation_history=history, api_call_count=1, approx_tokens=10, provider="openrouter",
        base_url="https://openrouter.ai/api/v1", model="m", current_turn_user_idx=len(history))


@pytest.mark.asyncio
async def test_a_refused_turn_whose_ask_landed_is_dropped_and_nothing_else(gw):
    """The refusal hands back the pre-refusal history and proves its stored opening row: the gateway
    drops exactly that turn, inserts nothing, and the earlier exchange survives."""
    agent = _agent(gw)
    messages, history = _turn_start(gw, agent)
    result = _refusal(gw, agent, messages, history)
    assert result["failure_reason"] == "content_policy_blocked" and result["messages"] == history  # reach
    assert result["agent_persisted"] is True

    await _gateway_persists(gw, result, history)

    model_view = [(m["role"], m["content"]) for m in gw.db.get_messages_as_conversation(gw.entry.session_id)]
    assert model_view == [("user", "hello"), ("assistant", "hi there")]
    assert [m["content"] for m in gw.db.get_messages(gw.entry.session_id, include_inactive=True)].count(ASK) == 1


@pytest.mark.asyncio
async def test_a_refused_turn_whose_ask_never_landed_retracts_nothing(gw):
    """No stored opening row: no proof, so no drop. Guessing the newest user turn would take the earlier
    accepted exchange instead."""
    agent = _agent(gw)
    restore = _break_writes_after(gw.db, 0)
    try:
        messages, history = _turn_start(gw, agent, lands=False)
        assert agent._flush_messages_to_session_db(messages, history) is False
        result = _refusal(gw, agent, messages, history)
    finally:
        restore()
    assert result["failure_reason"] == "content_policy_blocked" and result["agent_persisted"] is False  # reach

    await _gateway_persists(gw, result, history)

    assert _rows(gw) == [("user", "hello"), ("assistant", "hi there")]
    model_view = [(m["role"], m["content"]) for m in gw.db.get_messages_as_conversation(gw.entry.session_id)]
    assert model_view == [("user", "hello"), ("assistant", "hi there")]


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["absent", None, 1, "true"])
async def test_a_missing_or_non_bool_value_is_unknown(gw, value):
    """Not proven persisted: a held row is skipped, an unstored one recovered, and the True-only
    exhausted-turn retract does not run."""
    sid = gw.entry.session_id
    history = _history(gw)
    gw.db.append_message(sid, "user", ASK)
    messages = [*history, {"role": "user", "content": ASK, "_db_persisted": True}, {"role": "assistant", "content": FINAL}]
    result = {"final_response": FINAL, "messages": messages, "completed": True}
    exhausted = {"final_response": "copy", "failed": True, "compression_exhausted": True,
                 "failure_reason": "context_overflow", "messages": messages[:-1]}
    if value != "absent":
        result["agent_persisted"] = exhausted["agent_persisted"] = value

    await _gateway_persists(gw, result, history)
    assert _rows(gw) == [("user", "hello"), ("assistant", "hi there"), ("user", ASK), ("assistant", FINAL)]

    gw.db.append_message(sid, "user", "a second ask that exhausts")
    await _gateway_persists(gw, exhausted, history)
    assert _rows(gw)[-1] == ("user", "a second ask that exhausts")  # not retracted


@pytest.mark.parametrize(("final", "value"), [("an answer", "absent"), ("an answer", False), ("an answer", "yes"),
                                              ("", True), ("", False), ("", "absent")])
def test_the_turn_runner_forwards_only_a_bool(final, value):
    """The gateway envelope carries the agent's own bool, or nothing; never a default True."""
    from gateway.run_turn_runner import TurnRunner
    from gateway.session import SessionSource
    from gateway.session_results import execution_result
    from gateway.turn_context import TurnContext
    from tests.gateway.test_local_viewer_events_submit import _gateway_runner, _RecordingAgent

    class _Agent(_RecordingAgent):
        def run_conversation(self, message, **kwargs):
            return {"final_response": final, "messages": [], "completed": True,
                    **({} if value == "absent" else {"agent_persisted": value})}

    ctx = TurnContext(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="sid", user_id="u"),
        message="hi", history=[], session_id="sid", session_key="key", user_config={},
        AIAgent=_Agent, resolve_display_setting=lambda *_a: False,
        _run_still_current=lambda: True, _hooks_ref=SimpleNamespace(loaded_hooks=False))
    token = execution_result.set({})
    try:
        turn = TurnRunner(_gateway_runner(), ctx)
        turn._approval_owner = (SimpleNamespace(sessions={"sid": SimpleNamespace(tool_progress_mode=None)},
                                                profile_id="p", adopt_agent=lambda *a: None), "sid", 1)
        delivered = turn.run_sync()
    finally:
        execution_result.reset(token)
    if isinstance(value, bool):
        assert delivered["agent_persisted"] is value
    else:
        assert "agent_persisted" not in delivered
