"""L4-2/R67-1 — early-truncation path. ``recover_from_truncation`` / ``continue_codex_incomplete``
end the turn directly (``conversation_loop`` returns ``verdict.result``/the dict unchanged,
bypassing ``agent/turn_finalizer.py::finalize_turn`` entirely — see
``agent/conversation_loop.py``'s ``return _something.result`` early exits). Both call
``agent._persist_session()`` right before returning, and that call already records the real
outcome on ``agent._last_persist_succeeded`` (session_persistence.py, L4-2 base fix). Before this
fix, ``partial_result()`` never read that attribute, so the returned dict carried no
``agent_persisted`` key at all and the gateway fell back to "a session DB exists" -> True even
when the flush had just failed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.turn_truncation import _Trunc, continue_codex_incomplete


def test_continue_codex_incomplete_exhaustion_reports_agent_persisted_false_on_flush_failure():
    """The ``n >= 3`` exhaustion branch (#90393 ladder) persists then returns a terminal partial
    result without ever reaching finalize_turn."""
    agent = MagicMock()
    agent.max_tokens = 2000
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent._codex_incomplete_retries = 2  # the next call exhausts the cap (n becomes 3)
    agent._codex_reasoning_only_streak = 0
    agent._build_assistant_message.side_effect = lambda msg, fr: {
        "role": "assistant", "content": msg.content or "", "finish_reason": fr,
        "codex_reasoning_items": msg.codex_reasoning_items,
    }
    agent._interim_assistant_visible_text.return_value = ""

    def _failing_persist(messages, conversation_history):
        agent._last_persist_succeeded = False  # what the real flush sets on a write exception

    agent._persist_session = MagicMock(side_effect=_failing_persist)

    message = SimpleNamespace(content="partial answer", tool_calls=None, reasoning=None,
                               codex_reasoning_items=None)
    messages = [{"role": "user", "content": "write an essay"}]

    result = continue_codex_incomplete(
        agent, message, "incomplete", messages=messages,
        conversation_history=None, api_call_count=3, response=None,
    )

    assert result["final_response"] == "Codex response remained incomplete after 3 continuation attempts"
    agent._persist_session.assert_called_once()
    assert result["agent_persisted"] is False


def test_continue_codex_incomplete_exhaustion_reports_agent_persisted_true_on_ordinary_success():
    """Guardrail: nothing sets ``_last_persist_succeeded`` on an ordinary successful flush (it
    simply doesn't exist, same as every FakeAgent in test_turn_finalizer_final_response_persistence
    .py) — the default must stay True, matching the pre-L4-2 contract."""
    agent = MagicMock()
    agent.max_tokens = 2000
    agent.quiet_mode = True
    agent.log_prefix = ""
    agent._codex_incomplete_retries = 2
    agent._codex_reasoning_only_streak = 0
    agent._build_assistant_message.side_effect = lambda msg, fr: {
        "role": "assistant", "content": msg.content or "", "finish_reason": fr,
        "codex_reasoning_items": msg.codex_reasoning_items,
    }
    agent._interim_assistant_visible_text.return_value = ""
    del agent._last_persist_succeeded  # MagicMock auto-creates attrs; ensure it truly is absent

    message = SimpleNamespace(content="partial answer", tool_calls=None, reasoning=None,
                               codex_reasoning_items=None)
    messages = [{"role": "user", "content": "write an essay"}]

    result = continue_codex_incomplete(
        agent, message, "incomplete", messages=messages,
        conversation_history=None, api_call_count=3, response=None,
    )

    assert result["agent_persisted"] is True


def test_end_turn_reports_agent_persisted_false_when_flush_failed():
    """``_Trunc.end_turn`` (the shared terminal path for the abort/rollback/first-message-truncated
    branches of ``recover_from_truncation``) is the sibling bug in the same file: same
    persist-then-return-partial_result shape, same missing flag."""
    agent = MagicMock()

    def _failing_persist(messages, conversation_history):
        agent._last_persist_succeeded = False

    agent._persist_session = MagicMock(side_effect=_failing_persist)

    st = _Trunc(
        agent=agent, response=None, finish_reason="length",
        conversation_history=[], api_call_count=1,
        effective_task_id="t", current_turn_user_idx=0,
        messages=[{"role": "user", "content": "hi"}],
        length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )

    verdict = st.end_turn("final text", "error text")

    assert verdict.action == "return"
    agent._persist_session.assert_called_once()
    assert verdict.result["agent_persisted"] is False


def test_end_turn_reports_agent_persisted_true_when_nothing_set_the_attribute():
    """Guardrail for the same default as above, on the ``end_turn`` sibling."""
    agent = MagicMock()
    del agent._last_persist_succeeded

    st = _Trunc(
        agent=agent, response=None, finish_reason="length",
        conversation_history=[], api_call_count=1,
        effective_task_id="t", current_turn_user_idx=0,
        messages=[{"role": "user", "content": "hi"}],
        length_continue_retries=0, truncated_response_parts=[],
        truncated_tool_call_retries=0, retry_count=0, compression_attempts=0,
    )

    verdict = st.end_turn("final text", "error text")

    assert verdict.result["agent_persisted"] is True
