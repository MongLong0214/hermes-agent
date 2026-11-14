"""Tests for gateway /goal verdict-message delivery.

The judge verdict message ("✓ Goal achieved", "⏸ budget exhausted", etc.)
must reach the user after each turn. Before this fix the code checked
``hasattr(adapter, "send_message")`` — but adapters expose ``send()``,
never ``send_message``, so the check always evaluated False and users
never saw verdicts. This test locks in the fix.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, build_session_key


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals

    goals._DB_CACHE.clear()
    # Pre-warm the SessionDB cache from this SYNC context. The tests call
    # GoalManager.set() on the event-loop thread, where _get_session_db()
    # refuses to construct SessionDB inline (loop-liveness guard) and only
    # waits _DB_BOOTSTRAP_LOOP_WAIT_S for a background bootstrap. On a loaded
    # CI runner the init overruns that window, the goal write is silently
    # dropped by design, and the continuation path no-ops — the recurring
    # sends == [] flake. Warming here uses the direct construction path, so
    # the loop-thread set() always finds a cached DB.
    goals._get_session_db()
    yield home
    goals._DB_CACHE.clear()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


class _RecordingAdapter:
    """Minimal adapter that records send() invocations."""

    def __init__(self) -> None:
        self._pending_messages: dict = {}
        self.sends: list[dict] = []

    async def send(self, chat_id: str, content: str, reply_to=None, metadata=None):
        self.sends.append({"chat_id": chat_id, "content": content, "metadata": metadata})

        class _R:
            success = True
            message_id = "mock-msg"

        return _R()


def _make_runner_with_adapter(session_id: str | None = None):
    from gateway.run import GatewayRunner
    import uuid

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")},
    )
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._queued_events = {}

    src = _make_source()
    # Default to a unique session_id so parallel runs
    # don't see each other's GoalManager state (DEFAULT_DB_PATH gets frozen at
    # module-import time, defeating per-test HERMES_HOME monkeypatches).
    session_entry = SessionEntry(
        session_key=build_session_key(src),
        session_id=session_id or f"goal-sess-{uuid.uuid4().hex[:8]}",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store._generate_session_key.return_value = build_session_key(src)

    adapter = _RecordingAdapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    return runner, adapter, session_entry, src


async def _drain_until(condition, timeout=5.0):
    """Yield to the event loop until ``condition()`` is truthy (bounded).

    The goal-continuation path finishes its sends/enqueues on spawned tasks;
    a fixed 0.05s sleep raced them on loaded CI runners (#88975). Returns as
    soon as the condition holds — the asserts after the call stay exact.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while not condition() and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_goal_verdict_continue_enqueues_continuation(hermes_home):
    """When the judge says continue, both the 'continuing' status and the
    continuation-prompt event must be delivered. The continuation prompt is
    routed through the adapter's pending-messages FIFO so the goal loop
    proceeds on the next turn."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "still needs work", False, None, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="here's a partial edit",
        )
        await _drain_until(lambda: adapter.sends and adapter._pending_messages)

    # Status line sent back
    assert len(adapter.sends) == 1
    assert "Continuing toward goal" in adapter.sends[0]["content"]
    # Continuation prompt enqueued for next turn
    assert adapter._pending_messages, "continuation prompt must be enqueued in pending_messages"


@pytest.mark.asyncio
async def test_goal_verdict_budget_exhausted_sends_pause(hermes_home):
    """When the budget is exhausted, a '⏸ Goal paused' message must be sent
    and no further continuation enqueued."""
    runner, adapter, session_entry, src = _make_runner_with_adapter()

    from hermes_cli.goals import GoalManager, save_goal

    mgr = GoalManager(session_entry.session_id, default_max_turns=2)
    state = mgr.set("tiny goal", max_turns=2)
    state.turns_used = 2
    save_goal(session_entry.session_id, state)

    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False)):
        await runner._post_turn_goal_continuation(
            session_entry=session_entry,
            source=src,
            final_response="still partial",
        )
        await _drain_until(lambda: adapter.sends)

    assert len(adapter.sends) == 1
    content = adapter.sends[0]["content"]
    assert "paused" in content.lower()
    assert "turns used" in content.lower()
    # No continuation enqueued when budget is exhausted
    assert not adapter._pending_messages




# ── compression exhaustion: the bound the TUI and CLI share ─────────

_EXHAUSTED = {"final_response": "This conversation has grown too long.", "failed": True,
              "partial": True, "compression_exhausted": True, "completed": False}


def _fifo_recorder(runner):
    enqueued = []
    runner._enqueue_fifo = lambda key, event, _adapter: enqueued.append((key, event.text))
    return enqueued


@pytest.mark.asyncio
async def test_exhausted_turn_is_not_judged_and_schedules_one_continuation(hermes_home):
    runner, adapter, session_entry, src = _make_runner_with_adapter()
    enqueued = _fifo_recorder(runner)
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_entry.session_id)
    mgr.set("polish the docs")
    judge = MagicMock(return_value=("done", "looks finished", False, None, False))
    with patch("hermes_cli.goals.judge_goal", judge):
        await runner._run_post_turn_hooks(agent_result=dict(_EXHAUSTED), source=src, is_internal=False)
        await _drain_until(lambda: adapter.sends)

    judge.assert_not_called()
    assert enqueued == [(session_entry.session_key, mgr.next_continuation_prompt())]
    assert [s["content"] for s in adapter.sends] == [
        "Context compression was exhausted. Retrying the active goal once."]
    state = GoalManager(session_entry.session_id).state
    assert state.status == "active" and state.turns_used == 0
    conversation = runner._session_state(session_entry.session_key).conversation
    assert conversation.goal_compression_recovery["attempts"] == 1


@pytest.mark.asyncio
async def test_second_exhaustion_pauses_the_goal_and_posts_the_notice_once(hermes_home):
    runner, adapter, session_entry, src = _make_runner_with_adapter()
    enqueued = _fifo_recorder(runner)
    from gateway.session_results import execution_result
    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("polish the docs")
    judge = MagicMock(return_value=("continue", "keep going", False, None, False))
    reply = _EXHAUSTED["final_response"]
    with patch("hermes_cli.goals.judge_goal", judge):
        for turn in (1, 2):
            # The live path returns the reply text; the owner captured the result dict.
            token = execution_result.set({"result": dict(_EXHAUSTED)})
            try:
                await runner._run_post_turn_hooks(agent_result=reply, source=src, is_internal=False)
            finally:
                execution_result.reset(token)
            await _drain_until(lambda: len(adapter.sends) >= turn)

    judge.assert_not_called()
    assert len(enqueued) == 1 and "polish the docs" in enqueued[0][1], "exactly one retry"
    state = GoalManager(session_entry.session_id).state
    assert state.status == "paused"
    assert state.paused_reason == "context compression exhausted twice consecutively"
    pause = [s["content"] for s in adapter.sends if "paused" in s["content"].lower()]
    assert pause == ["Goal paused after context compression was exhausted twice. "
                     "Run /compress, then /goal resume to continue."]
    assert runner._session_state(session_entry.session_key).conversation.goal_compression_recovery is None


@pytest.mark.asyncio
async def test_compression_deferred_is_not_exhaustion(hermes_home):
    runner, adapter, session_entry, src = _make_runner_with_adapter()
    _fifo_recorder(runner)
    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("polish the docs")
    deferred = dict(_EXHAUSTED, compression_deferred=True)
    judge = MagicMock(return_value=("continue", "keep going", False, None, False))
    with patch("hermes_cli.goals.judge_goal", judge):
        for _ in range(2):
            await runner._run_post_turn_hooks(agent_result=dict(deferred), source=src, is_internal=False)
    await asyncio.sleep(0.05)
    assert GoalManager(session_entry.session_id).state.status == "active"
    assert not any("compression was exhausted" in s["content"] for s in adapter.sends)
    assert runner._session_state(session_entry.session_key).conversation.goal_compression_recovery is None


@pytest.mark.asyncio
async def test_recovery_state_is_conversation_scoped_and_cleared_by_new(hermes_home):
    from gateway.session_state import ConversationState

    assert ConversationState().goal_compression_recovery is None
    assert ConversationState().loop_compression_recovery is None
    runner, _adapter, session_entry, src = _make_runner_with_adapter()
    _fifo_recorder(runner)
    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("polish the docs")
    with patch("hermes_cli.goals.judge_goal", MagicMock()):
        await runner._run_post_turn_hooks(agent_result=dict(_EXHAUSTED), source=src, is_internal=False)
    conversation = runner._session_state(session_entry.session_key).conversation
    conversation.loop_compression_recovery = {"loop_created_at": 1.0, "ticks": 1}
    assert conversation.goal_compression_recovery is not None

    runner._clear_conversation_scope(session_entry.session_key, reason="session_reset")  # the /new funnel

    assert conversation.goal_compression_recovery is None
    assert conversation.loop_compression_recovery is None


@pytest.mark.asyncio
async def test_successful_turn_between_exhaustions_resets_the_bound(hermes_home):
    runner, _adapter, session_entry, src = _make_runner_with_adapter()
    enqueued = _fifo_recorder(runner)
    from hermes_cli.goals import GoalManager

    GoalManager(session_entry.session_id).set("polish the docs")
    with patch("hermes_cli.goals.judge_goal", return_value=("continue", "keep going", False, None, False)):
        await runner._run_post_turn_hooks(agent_result=dict(_EXHAUSTED), source=src, is_internal=False)
        await runner._run_post_turn_hooks(agent_result={"final_response": "real progress"}, source=src,
                                          is_internal=False)
        assert runner._session_state(session_entry.session_key).conversation.goal_compression_recovery is None
        await runner._run_post_turn_hooks(agent_result=dict(_EXHAUSTED), source=src, is_internal=False)
    assert len(enqueued) == 3
    assert GoalManager(session_entry.session_id).state.status == "active"
