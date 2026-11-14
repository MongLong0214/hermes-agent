"""A turn that exhausts context compression retracts its own unanswered ask (CLI, TUI, ACP).

The session is kept on exhaustion, so the user row the agent flushed at turn start would otherwise be
merged into the next ask (#107070) and replayed on every later request; a closing boundary would grow
an oversized session (#1630). Hosts that own a direct ``SessionDB`` hand ``result["messages"]`` straight
back as history, so the retraction runs in the agent (``_close_durable_failed_turn``), while the turn
still holds its own durable turn lease: ``SessionDB.rewind_user_turn(turn_lease_holder=...)``. A lease
held by anyone else still refuses the rewind. Real SessionDB throughout.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

PRIOR = [{"role": "user", "content": "old request"}, {"role": "assistant", "content": "old response"}]
ASK = "summarise the whole repository history"
EXHAUSTED = {"completed": False, "failed": True, "partial": True, "compression_exhausted": True,
             "failure_reason": "context_overflow"}


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    store.create_session(session_id="s1", source="cli", model="m")
    for message in PRIOR:
        store.append_message("s1", message["role"], message["content"])
    yield store
    store.close()


def _conversation(db):
    return [(m["role"], m["content"]) for m in db.get_messages_as_conversation("s1")]


def _turn(db, holder="this-turn", *, lease_owner=None, tail=()):
    """A turn that flushed its ask (and ``tail``) under the lease ``lease_owner`` holds (default: its own)."""
    owner = lease_owner or holder
    assert db.acquire_session_turn_lease("s1", owner, wait_seconds=0)
    messages = [dict(m) for m in PRIOR] + [{"role": "user", "content": ASK}, *tail]
    for message in messages[len(PRIOR):]:
        db.append_message("s1", message["role"], message.get("content"), tool_calls=message.get("tool_calls"),
                          tool_call_id=message.get("tool_call_id"), turn_lease_holder=owner)
    agent = SimpleNamespace(_session_db=db, session_id="s1", _active_session_turn_lease_holder=holder,
                            _flush_messages_to_session_db=MagicMock(name="flush"))
    return agent, messages


def test_an_exhausted_turn_retracts_its_own_ask_under_its_own_lease(db):
    from agent.conversation_loop import _close_durable_failed_turn
    agent, messages = _turn(db)
    result = dict(EXHAUSTED, messages=messages, turn_id="t-1", current_turn_user_idx=len(PRIOR))

    _close_durable_failed_turn(agent, result)

    assert _conversation(db) == [(m["role"], m["content"]) for m in PRIOR]
    # The host installs this as its history: the ask is gone there too, with its turn boundary.
    assert result["messages"] is messages and [m["content"] for m in messages] == ["old request", "old response"]
    assert "turn_id" not in result and "current_turn_user_idx" not in result
    assert result["compression_exhausted"] is True  # still reported to the host
    agent._flush_messages_to_session_db.assert_not_called()  # no closing boundary row
    # The rewind ran under the turn's lease and left it with the turn, which releases it on return.
    db.release_session_turn_lease("s1", "this-turn")
    assert db.acquire_session_turn_lease("s1", "next-turn", wait_seconds=0)


def test_a_lease_held_by_another_turn_refuses_the_retraction(db):
    from agent.conversation_loop import _close_durable_failed_turn
    agent, messages = _turn(db, lease_owner="another-process")
    result = dict(EXHAUSTED, messages=messages)

    _close_durable_failed_turn(agent, result)

    assert _conversation(db)[-1] == ("user", ASK)
    assert messages[-1]["content"] == ASK


def test_an_exhausted_turn_that_ran_tools_keeps_its_record(db):
    from agent.conversation_loop import _close_durable_failed_turn
    tool_round = ({"role": "assistant", "content": None, "tool_calls": [
        {"id": "t1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "content": "ran", "tool_call_id": "t1"})
    agent, messages = _turn(db, tail=tool_round)
    result = dict(EXHAUSTED, messages=messages)

    _close_durable_failed_turn(agent, result)

    assert [role for role, _ in _conversation(db)][-3:] == ["user", "assistant", "tool"]
    assert len(messages) == len(PRIOR) + 3
    agent._flush_messages_to_session_db.assert_not_called()


def test_a_deferred_compression_never_retracts(db):
    from agent.conversation_loop import _close_durable_failed_turn
    agent, messages = _turn(db)
    result = dict(EXHAUSTED, compression_deferred=True, messages=messages)

    _close_durable_failed_turn(agent, result)

    assert _conversation(db)[-1] == ("user", ASK)
    agent._flush_messages_to_session_db.assert_not_called()


def _make_agent(db):
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, session_db=db, session_id="s1", platform="cli")
    agent._session_db_created = True
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.save_trajectories = False
    agent.compression_enabled = True
    return agent


def test_a_cli_turn_that_exhausts_leaves_the_session_as_it_was(db, monkeypatch):
    """The whole turn through ``AIAgent.run_conversation``: the facade takes the durable turn lease,
    the turn-start flush writes the ask, the provider overflows and compression times out (#98722),
    and the ask is retracted before the lease is released, so the next turn admits normally."""
    import time as _time
    from agent.conversation_compression import (
        mark_context_compression_timed_out, reset_context_compression_timeout_outcome)
    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr("agent.retry_utils.jittered_backoff", lambda *a, **k: 0.0)
    agent = _make_agent(db)
    overflow = Exception("This model's maximum context length is 8192 tokens. However, your messages "
                         "resulted in 95000 tokens. Please reduce the length of the messages.")
    overflow.status_code = 400
    agent.client.chat.completions.create.side_effect = [overflow, overflow]
    flushed = []

    def timed_out(messages, _system_message, **_kwargs):
        flushed.append(_conversation(db)[-1])
        reset_context_compression_timeout_outcome(agent)
        mark_context_compression_timed_out(agent)
        return messages, agent._cached_system_prompt
    agent._compress_context = timed_out

    with patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources"):
        result = agent.run_conversation(ASK, conversation_history=[dict(m) for m in PRIOR])

    assert result["compression_exhausted"] is True and result["failed"] is True
    assert flushed == [("user", ASK)]  # the turn-start flush wrote the ask before the turn exhausted
    assert _conversation(db) == [(m["role"], m["content"]) for m in PRIOR]
    assert [(m["role"], m["content"]) for m in result["messages"]] == [(m["role"], m["content"]) for m in PRIOR]
    assert db.acquire_session_turn_lease("s1", "next-turn", wait_seconds=0)  # released after the rewind
