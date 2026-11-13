"""Shared bounded recovery after a compression-exhausted /goal turn.

The session is kept after exhaustion, so an unbounded continuation would re-send the same
oversized request every turn. ``plan_compression_exhaustion_recovery`` is the one bound every
host (TUI, CLI, gateway) uses: one fresh continuation per goal, then the goal pauses. The
exhausted turn is never judge input and never spends a goal turn.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

RETRY_NOTICE = "Context compression was exhausted. Retrying the active goal once."
PAUSE_NOTICE = (
    "Goal paused after context compression was exhausted twice. "
    "Run /compress, then /goal resume to continue."
)


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
def no_judge(monkeypatch):
    """Any judge call is a failure: an exhausted turn added no reply to judge."""
    from hermes_cli import goals

    calls = []

    def _judge(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("an exhausted turn must never be judged")

    monkeypatch.setattr(goals, "judge_goal", _judge)
    return calls


def _goal(session_id: str, text: str = "finish the current task"):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id)
    mgr.set(text)
    return mgr


def test_limit_is_one_retry():
    from hermes_cli.goals import COMPRESSION_RECOVERY_LIMIT

    assert COMPRESSION_RECOVERY_LIMIT == 1


def test_first_exhaustion_returns_one_continuation_and_attempts_1(hermes_home, no_judge):
    from hermes_cli.goals import GoalManager, plan_compression_exhaustion_recovery

    mgr = _goal("rec-first")
    expected = mgr.next_continuation_prompt()

    prompt, notice, state = plan_compression_exhaustion_recovery(mgr, None)

    assert prompt == expected
    assert notice == RETRY_NOTICE
    assert state == {
        "goal_created_at": float(mgr.state.created_at),
        "goal": "finish the current task",
        "attempts": 1,
    }
    reloaded = GoalManager("rec-first").state
    assert reloaded.status == "active"
    assert reloaded.turns_used == 0
    assert no_judge == []


def test_second_consecutive_exhaustion_pauses_with_notice(hermes_home, no_judge):
    from hermes_cli.goals import GoalManager, plan_compression_exhaustion_recovery

    mgr = _goal("rec-second")
    _prompt, _notice, state = plan_compression_exhaustion_recovery(mgr, None)

    prompt, notice, next_state = plan_compression_exhaustion_recovery(mgr, state)

    assert prompt is None
    assert notice == PAUSE_NOTICE
    assert next_state is None
    reloaded = GoalManager("rec-second").state
    assert reloaded.status == "paused"
    assert reloaded.paused_reason == "context compression exhausted twice consecutively"
    assert reloaded.turns_used == 0
    assert no_judge == []


def test_cleared_state_after_success_gives_a_fresh_retry(hermes_home, no_judge):
    """The host clears the state after a successful turn (None): the next exhaustion retries."""
    from hermes_cli.goals import plan_compression_exhaustion_recovery

    mgr = _goal("rec-reset")
    plan_compression_exhaustion_recovery(mgr, None)

    prompt, notice, state = plan_compression_exhaustion_recovery(mgr, None)

    assert prompt == mgr.next_continuation_prompt()
    assert notice == RETRY_NOTICE
    assert state["attempts"] == 1
    assert mgr.state.status == "active"


def test_state_from_a_different_goal_does_not_count(hermes_home, no_judge):
    from hermes_cli.goals import plan_compression_exhaustion_recovery

    mgr = _goal("rec-other", "the new goal")
    stale = {"goal_created_at": float(mgr.state.created_at) - 10.0, "goal": "the new goal", "attempts": 1}
    renamed = {"goal_created_at": float(mgr.state.created_at), "goal": "the old goal", "attempts": 1}

    for previous in (stale, renamed, {"attempts": "garbage"}, "not-a-dict"):
        prompt, notice, state = plan_compression_exhaustion_recovery(mgr, previous)
        assert prompt is not None, previous
        assert notice == RETRY_NOTICE
        assert state["attempts"] == 1
    assert mgr.state.status == "active"


# ── TUI host: delegates to the shared bound, behaviour unchanged ─────


@pytest.fixture
def tui_server(hermes_home, monkeypatch):
    with patch.dict("sys.modules", {"hermes_cli.env_loader": MagicMock(), "hermes_cli.banner": MagicMock()}):
        mod = importlib.import_module("tui_gateway.server")
    monkeypatch.setattr(mod, "_hermes_home", hermes_home)
    monkeypatch.setattr(mod, "_cfg_cache", None)
    monkeypatch.setattr(mod, "_cfg_sig", None)
    monkeypatch.setattr(mod, "_cfg_path", None)
    yield mod
    mod._sessions.clear()


_EXHAUSTED = {"failed": True, "partial": True, "compression_exhausted": True, "completed": False,
              "final_response": "Context length exceeded."}


def test_tui_delegates_to_shared_bound(tui_server, monkeypatch):
    from hermes_cli import goals

    _goal("tui-delegate")
    calls = []

    def _shared(mgr, state):
        calls.append(state)
        return "shared-prompt", "shared-notice", {"marker": True}

    monkeypatch.setattr(goals, "plan_compression_exhaustion_recovery", _shared)
    session = {"session_key": "tui-delegate"}

    prompt, notice = tui_server._plan_goal_compression_recovery(
        session, _EXHAUSTED, status="error", raw="Context length exceeded.")

    assert (prompt, notice) == ("shared-prompt", "shared-notice")
    assert calls == [None]
    assert session[tui_server._GOAL_COMPRESSION_RECOVERY_ATTEMPTS] == {"marker": True}


def test_tui_exhausted_success_exhausted_retries_again(tui_server, no_judge):
    mgr = _goal("tui-reset")
    session = {"session_key": "tui-reset"}
    key = tui_server._GOAL_COMPRESSION_RECOVERY_ATTEMPTS

    first = tui_server._plan_goal_compression_recovery(session, _EXHAUSTED, status="error", raw="x")
    assert first == (mgr.next_continuation_prompt(), RETRY_NOTICE)
    assert session[key]["attempts"] == 1

    tui_server._plan_goal_compression_recovery(
        session, {"final_response": "real progress"}, status="complete", raw="real progress")
    assert key not in session

    again = tui_server._plan_goal_compression_recovery(session, _EXHAUSTED, status="error", raw="x")
    assert again == (mgr.next_continuation_prompt(), RETRY_NOTICE)

    paused = tui_server._plan_goal_compression_recovery(session, _EXHAUSTED, status="error", raw="x")
    assert paused == (None, PAUSE_NOTICE)
    assert key not in session
    from hermes_cli.goals import GoalManager

    assert GoalManager("tui-reset").state.status == "paused"
