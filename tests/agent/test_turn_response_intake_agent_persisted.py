"""L4-2/R67-1 ROUND1-ESCAPE sibling — incomplete-scratchpad exhaustion path.

``normalize_model_response`` ends the turn directly on <REASONING_SCRATCHPAD> exhaustion
(opened but never closed, 2 retries burned): it calls ``agent._persist_session()`` — which
already records the real flush outcome on ``agent._last_persist_succeeded``
(session_persistence.py, L4-2 base fix) — immediately before building the partial result.
Round 1 fixed the sibling early-truncation paths in ``agent/turn_truncation.py`` by passing
``agent=agent`` into ``partial_result()`` so the dict carries an honest ``agent_persisted``.
This call site was missed: it called ``partial_result()`` without ``agent``, so the returned
dict carried no ``agent_persisted`` key at all and the gateway fell back to "a session DB
exists" -> True even when the flush that just ran had failed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_response_intake import normalize_model_response


def _make_agent():
    agent = MagicMock()
    agent.api_mode = "chat_completions"
    agent.quiet_mode = True
    agent.tool_progress_callback = None
    agent._incomplete_scratchpad_retries = 2  # the next increment (3) exceeds the <= 2 retry cap
    agent._get_messages_up_to_last_assistant.return_value = [
        {"role": "user", "content": "write an essay"}
    ]
    return agent


def _make_response():
    assistant_message = SimpleNamespace(
        content="<REASONING_SCRATCHPAD>thinking forever", finish_reason="length", tool_calls=None,
    )
    transport = MagicMock()
    transport.normalize_response.return_value = assistant_message
    return SimpleNamespace(), transport, assistant_message


def test_scratchpad_exhaustion_reports_agent_persisted_false_on_flush_failure():
    agent = _make_agent()
    response, transport, _assistant_message = _make_response()
    agent._get_transport.return_value = transport

    def _failing_persist(messages, conversation_history):
        agent._last_persist_succeeded = False  # what the real flush sets on a write exception

    agent._persist_session = MagicMock(side_effect=_failing_persist)

    verdict = normalize_model_response(
        agent, response=response, messages=[{"role": "user", "content": "write an essay"}],
        api_messages=[], conversation_history=None, api_call_count=1, api_duration=0.1,
        api_start_time=0.0, api_request_id="req-1", effective_task_id="task-1", turn_id="turn-1",
    )

    assert verdict.action == "return"
    agent._persist_session.assert_called_once()
    assert verdict.result["agent_persisted"] is False


def test_scratchpad_exhaustion_reports_agent_persisted_true_on_ordinary_success():
    """Guardrail: nothing sets ``_last_persist_succeeded`` on an ordinary successful flush (it
    simply doesn't exist) — the default must stay True, matching the pre-L4-2 contract."""
    agent = _make_agent()
    response, transport, _assistant_message = _make_response()
    agent._get_transport.return_value = transport
    del agent._last_persist_succeeded  # MagicMock auto-creates attrs; ensure it truly is absent

    verdict = normalize_model_response(
        agent, response=response, messages=[{"role": "user", "content": "write an essay"}],
        api_messages=[], conversation_history=None, api_call_count=1, api_duration=0.1,
        api_start_time=0.0, api_request_id="req-1", effective_task_id="task-1", turn_id="turn-1",
    )

    assert verdict.result["agent_persisted"] is True
