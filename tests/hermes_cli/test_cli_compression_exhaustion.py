"""The interactive CLI's /goal and /loop hooks never act on a compression-exhausted turn's reply.

An exhausted turn adds no assistant reply, so the last one in the history is stale. Judging it (a
billed aux call) would restart the goal or loop, and the session is kept, so each retry re-sends
the same oversized request, up to ``goals.max_turns`` / ``loops.max_ticks`` times. The bound is the
one the gateway and TUI share: one retry, then pause.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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


def test_exhausted_turns_are_never_judged_and_pause_goal_and_loop_on_the_second(hermes_home):
    from cli import HermesCLI, _ChatTurn
    from hermes_cli.goals import GoalManager
    from hermes_cli.loops import LoopManager, load_loop

    sid = "sid-cli-exhausted"
    cli = HermesCLI.__new__(HermesCLI)
    cli._pending_input = queue.Queue()
    cli._last_turn_interrupted = False
    cli._prompt_start_time = None
    cli._flush_stream = lambda: None
    cli.session_id = sid
    cli.agent = MagicMock(session_id=sid)
    history = [{"role": "user", "content": "watch CI"}, {"role": "assistant", "content": "CI is green."}]
    cli.conversation_history = list(history)
    goal = GoalManager(session_id=sid, default_max_turns=20)
    goal.set("get CI green")
    cli._goal_manager = goal
    loop = LoopManager(session_id=sid)
    loop.set("poll CI", interval_seconds=300, until="CI is green")
    cli._loop_manager = loop
    judge = MagicMock(return_value=("done", "CI is green", False, None, False))

    queued = []
    with patch("hermes_cli.goals.judge_goal", judge):
        for _ in range(2):
            loop.state.next_due_at = time.time() - 1
            assert loop.fire_tick() is not None
            turn = _ChatTurn()
            turn.result = {"final_response": "This conversation has grown too long.", "failed": True,
                           "compression_exhausted": True, "messages": list(history)}
            cli._chat_settle_turn(turn)
            cli._maybe_continue_goal_after_turn()
            cli._maybe_complete_loop_tick_after_turn()
            judge.assert_not_called()
            queued.append(cli._pending_input.qsize())
            while not cli._pending_input.empty():
                cli._pending_input.get_nowait()

    assert queued == [1, 0], "one goal retry after the first exhaustion, nothing after the second"
    assert GoalManager(session_id=sid).state.status == "paused"
    reloaded = load_loop(sid)
    assert reloaded.status == "paused" and not reloaded.awaiting_response
