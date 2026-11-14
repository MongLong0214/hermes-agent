"""The interactive CLI's /goal and /loop hooks after a compression-exhausted turn.

An exhausted turn is a failed turn that added no assistant reply, so the last reply in the
history is stale and must never be judged. Before, the goal hook returned silently on any failed
turn, which left the goal active with nothing queued: it stalled with no paused state. Now the
CLI applies the bound the TUI and gateway share (``plan_compression_exhaustion_recovery``): one
continuation, then the goal pauses with a notice. ``compression_deferred`` is not exhaustion, and
an ordinary failed turn keeps the old behaviour.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RETRY_NOTICE = "Context compression was exhausted. Retrying the active goal once."
PAUSE_NOTICE = (
    "Goal paused after context compression was exhausted twice. "
    "Run /compress, then /goal resume to continue."
)
EXHAUSTED = {"final_response": "This conversation has grown too long.", "failed": True,
             "partial": True, "compression_exhausted": True, "completed": False}


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture
def printed(monkeypatch):
    import cli

    lines = []
    monkeypatch.setattr(cli, "_cprint", lambda text="", *a, **k: lines.append(str(text)))
    return lines


@pytest.fixture
def judge():
    mock = MagicMock(return_value=("continue", "not yet", False, None, False))
    with patch("hermes_cli.goals.judge_goal", mock):
        yield mock


HISTORY = [{"role": "user", "content": "watch CI"}, {"role": "assistant", "content": "CI is green."}]


def _cli(sid: str):
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli._prompt_start_time = None
    cli._flush_stream = lambda: None
    cli.session_id = sid
    cli.agent = MagicMock(session_id=sid, _interrupt_requested=False)
    cli.conversation_history = list(HISTORY)
    return cli


def _settle(cli, result):
    """The two phases of chat() that read the turn result, in production order."""
    from cli import _ChatTurn

    turn = _ChatTurn()
    turn.result = dict(result, messages=list(HISTORY))
    with patch("time.sleep"):
        cli._chat_settle_turn(turn)
    thread = MagicMock()
    thread.is_alive.return_value = False
    cli._chat_resolve_interrupt(turn, thread, None, turn.result.get("final_response") or "")


def _drain(cli):
    items = []
    while not cli._pending_input.empty():
        items.append(cli._pending_input.get_nowait())
    return items


def _goal_cli(sid):
    from hermes_cli.goals import GoalManager

    cli = _cli(sid)
    mgr = GoalManager(session_id=sid, default_max_turns=20)
    mgr.set("get CI green")
    cli._goal_manager = mgr
    return cli, mgr


def test_first_exhausted_turn_queues_one_continuation(hermes_home, printed, judge):
    from hermes_cli.goals import GoalManager

    cli, mgr = _goal_cli("cli-ex-first")
    _settle(cli, EXHAUSTED)
    assert cli._last_turn_compression_exhausted is True

    cli._maybe_continue_goal_after_turn()

    judge.assert_not_called()
    assert _drain(cli) == [mgr.next_continuation_prompt()]
    state = GoalManager(session_id="cli-ex-first").state
    assert state.status == "active"
    assert state.turns_used == 0
    assert cli._goal_compression_recovery["attempts"] == 1
    assert any(RETRY_NOTICE in line for line in printed)


def test_second_exhausted_turn_pauses_goal_with_notice(hermes_home, printed, judge):
    from hermes_cli.goals import GoalManager

    cli, _mgr = _goal_cli("cli-ex-second")
    queued = []
    for _ in range(2):  # the REPL runs the queued continuation as the next turn
        _settle(cli, EXHAUSTED)
        cli._maybe_continue_goal_after_turn()
        queued.append(len(_drain(cli)))
    judge.assert_not_called()

    assert queued == [1, 0], "one retry after the first exhaustion, nothing after the second"
    state = GoalManager(session_id="cli-ex-second").state
    assert state.status == "paused"
    assert state.paused_reason == "context compression exhausted twice consecutively"
    assert cli._goal_compression_recovery is None
    assert any(PAUSE_NOTICE in line for line in printed)


def test_successful_turn_between_resets_the_bound(hermes_home, printed, judge):
    cli, mgr = _goal_cli("cli-ex-reset")
    _settle(cli, EXHAUSTED)
    cli._maybe_continue_goal_after_turn()
    _drain(cli)

    cli.conversation_history = list(HISTORY)
    _settle(cli, {"final_response": "made progress"})
    assert cli._last_turn_compression_exhausted is False
    cli._maybe_continue_goal_after_turn()
    assert judge.call_count == 1
    assert cli._goal_compression_recovery is None
    _drain(cli)

    _settle(cli, EXHAUSTED)
    cli._maybe_continue_goal_after_turn()
    assert judge.call_count == 1
    assert _drain(cli) == [mgr.next_continuation_prompt()]
    assert mgr.state.status == "active"


def test_compression_deferred_is_not_exhaustion(hermes_home, printed, judge):
    from hermes_cli.goals import GoalManager

    cli, _mgr = _goal_cli("cli-ex-deferred")
    deferred = dict(EXHAUSTED, compression_deferred=True)
    for _ in range(2):
        _settle(cli, deferred)
        assert cli._last_turn_compression_exhausted is False
        cli._maybe_continue_goal_after_turn()

    judge.assert_not_called()
    assert _drain(cli) == []
    assert GoalManager(session_id="cli-ex-deferred").state.status == "active"
    assert getattr(cli, "_goal_compression_recovery", None) is None
    assert not any(RETRY_NOTICE in line or PAUSE_NOTICE in line for line in printed)


def test_ordinary_failed_turn_keeps_target_behaviour(hermes_home, printed, judge):
    from hermes_cli.goals import GoalManager

    cli, _mgr = _goal_cli("cli-ex-plain-failure")
    for _ in range(3):
        _settle(cli, {"final_response": "provider error", "failed": True})
        assert cli._last_turn_failed is True
        assert cli._last_turn_compression_exhausted is False
        cli._maybe_continue_goal_after_turn()

    judge.assert_not_called()
    assert _drain(cli) == []
    assert GoalManager(session_id="cli-ex-plain-failure").state.status == "active"
    assert printed == []


def test_flag_resets_at_the_start_of_every_chat_turn(hermes_home):
    cli = _cli("cli-ex-flag")
    cli._last_turn_compression_exhausted = True
    cli._secret_capture_callback = None
    cli._ensure_runtime_credentials = lambda: False
    assert cli.chat("hello") is None
    assert cli._last_turn_compression_exhausted is False


def _loop_cli(sid, **kwargs):
    from hermes_cli.loops import LoopManager

    cli = _cli(sid)
    loop = LoopManager(session_id=sid)
    loop.set("poll CI", interval_seconds=300, **kwargs)
    cli._loop_manager = loop
    cli._subagent_monitor = None
    return cli, loop


def _fire(loop):
    loop.state.next_due_at = time.time() - 1
    assert loop.fire_tick() is not None


def test_exhausted_loop_ticks_are_unjudged_and_the_second_pauses(hermes_home, printed, judge):
    from hermes_cli.loops import load_loop

    cli, loop = _loop_cli("cli-loop-ex", until="CI is green")
    _fire(loop)
    _settle(cli, EXHAUSTED)
    cli._maybe_complete_loop_tick_after_turn()
    judge.assert_not_called()
    first = load_loop("cli-loop-ex")
    assert first.status == "active" and not first.awaiting_response
    assert cli._loop_compression_recovery["ticks"] == 1

    _fire(loop)
    _settle(cli, EXHAUSTED)
    cli._maybe_complete_loop_tick_after_turn()
    judge.assert_not_called()
    second = load_loop("cli-loop-ex")
    assert second.status == "paused" and not second.awaiting_response
    assert second.paused_reason == "context compression exhausted twice consecutively"
    assert cli._loop_compression_recovery is None
    assert any("Loop paused after context compression was exhausted twice" in line for line in printed)


def test_good_loop_tick_between_resets_the_bound(hermes_home, printed, judge):
    from hermes_cli.loops import load_loop

    cli, loop = _loop_cli("cli-loop-reset")
    for result in (EXHAUSTED, {"final_response": "still red"}, EXHAUSTED):
        _fire(loop)
        _settle(cli, result)
        cli.conversation_history = list(HISTORY)
        cli._maybe_complete_loop_tick_after_turn()
    assert load_loop("cli-loop-reset").status == "active"
    assert cli._loop_compression_recovery["ticks"] == 1


def test_goal_and_loop_on_one_session(hermes_home, printed, judge):
    """The fork's end-to-end witness: both hooks on one exhausted session, twice."""
    from hermes_cli.goals import GoalManager
    from hermes_cli.loops import LoopManager, load_loop

    sid = "cli-both"
    cli, _goal = _goal_cli(sid)
    loop = LoopManager(session_id=sid)
    loop.set("poll CI", interval_seconds=300, until="CI is green")
    cli._loop_manager = loop
    cli._subagent_monitor = None
    queued = []
    for _ in range(2):
        _fire(loop)
        _settle(cli, EXHAUSTED)
        cli._maybe_continue_goal_after_turn()
        cli._maybe_complete_loop_tick_after_turn()
        queued.append(len(_drain(cli)))
    judge.assert_not_called()
    assert queued == [1, 0]
    assert GoalManager(session_id=sid).state.status == "paused"
    assert load_loop(sid).status == "paused"
