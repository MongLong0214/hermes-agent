"""The executor records the outcome of every tool call on the call's result row (``_tool_outcome``).

The relayed-owner declaration (gateway/owner_relay.py) may cite only a call that succeeded, and it reads that
from here, where the executor produced the result: never from the result's text. Real ``AIAgent`` and the
real sequential executor; only the dispatch is stubbed.
"""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.tool_executor import append_peer_refusals, execute_tool_calls_segmented
from tests.agent.test_tool_call_incremental_persistence import _make_agent, _mock_tool_call
from tools.relay_resolve_tool import reset_reply_only, set_reply_only


@pytest.fixture(autouse=True)
def _no_titles(monkeypatch):
    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *args, **kwargs: None)


@pytest.mark.parametrize("tool, result, outcome", [
    ("terminal", '{"output":"operation failed", "exit_code":1, "error":null}', "failed"),
    ("web_search", '{"success":false,"message":"operation failed"}', "failed"),
    ("web_search", '{"error": "no such thing"}', "failed"),
    ("terminal", '{"output":"fine", "exit_code":0}', "ok"),
    ("web_search", '{"results": []}', "ok"),
])
def test_the_executor_records_each_calls_outcome_on_its_result_row(tool, result, outcome):
    agent = _make_agent()
    call = _mock_tool_call(name=tool, call_id="call-1")
    messages: list = []
    with (
        patch("model_tools.handle_function_call", return_value=result),
        patch("agent.tool_executor.maybe_persist_tool_result", side_effect=lambda **kwargs: kwargs["content"]),
    ):
        execute_tool_calls_segmented(agent, SimpleNamespace(tool_calls=[call]), messages, "task-1",
                                     segments=[("sequential", [call])])
    [row] = [m for m in messages if m.get("role") == "tool"]
    assert row["tool_call_id"] == "call-1" and row["_tool_outcome"] == outcome


def test_a_call_the_executor_refused_is_recorded_refused():
    rows: list = []
    call = SimpleNamespace(id="call-9", function=SimpleNamespace(name="terminal", arguments="{}"))
    token = set_reply_only(True)
    try:
        append_peer_refusals(SimpleNamespace(), [call], rows)
    finally:
        reset_reply_only(token)
    append_peer_refusals(SimpleNamespace(), [call], rows)  # an ordinary peer refusal
    assert [row["_tool_outcome"] for row in rows] == ["refused", "refused"]
    assert "reply-only" in rows[0]["content"] and "reply-only" not in rows[1]["content"]


def test_make_tool_result_message_carries_the_outcome_as_an_internal_key():
    from agent.tool_dispatch_helpers import make_tool_result_message

    row = make_tool_result_message("terminal", "x", "call-1", outcome="failed")
    assert row["_tool_outcome"] == "failed" and "_tool_outcome" not in make_tool_result_message(
        "terminal", "x", "call-1")
