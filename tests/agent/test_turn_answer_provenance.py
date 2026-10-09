"""``model_answer``: whether a turn's final text is the model's own answer (ESCAPE-SUPP-18).

The /acp receipt is certified only by a model answer, so the flag is decided where the text is
produced, before the explainer or a fallback string can stand in for it. Notices with positive
call counts (empty-response explainer, failed tool-budget summary) must not carry it; genuine
answers (plain, recovered stream text, a model-written budget summary) must.
"""

from unittest.mock import patch

import pytest

from tests.agent.test_turn_completion_explainer import _make_agent, _mock_response
from tests.agent.test_turn_finalizer_iteration_limit_exit import _finalize, _LimitAgent


def _run(agent, **patches):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("do something")


def test_a_plain_answer_carries_provenance():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [_mock_response(content="The answer is 42.")]
    result = _run(agent)
    assert result["final_response"].startswith("The answer is 42.")
    assert result["model_answer"] is True


def test_the_empty_response_explainer_is_not_an_answer():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop") for _ in range(8)
    ]
    result = _run(agent)
    assert "No reply:" in result["final_response"] and result["api_calls"] > 0
    assert result["model_answer"] is False


def test_recovered_stream_text_is_an_answer():
    agent = _make_agent()
    recovered = "I inspected the gateway and the turn stopped after the stream timed out."

    def _fake_api_call(_api_kwargs):
        agent._current_streamed_assistant_text = recovered
        return _mock_response(content=None, finish_reason="stop")

    with patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call):
        result = _run(agent)
    assert result["turn_exit_reason"] == "partial_stream_recovery"
    assert result["model_answer"] is True


class _SummaryAgent(_LimitAgent):
    def __init__(self, summary, answered):
        super().__init__()
        self._summary, self._answered = summary, answered

    def _emit_diagnostic_status(self, *_args, **_kwargs):
        pass

    def _handle_max_iterations(self, messages, api_call_count):
        self._iteration_summary_answered = self._answered
        return self._summary


@pytest.mark.parametrize("summary,answered,expected", [
    ("Here is what I finished and what remains.", True, True),
    ("I reached the iteration limit and couldn't generate a summary.", False, False),
])
def test_a_budget_summary_is_an_answer_only_when_the_model_wrote_it(summary, answered, expected):
    agent = _SummaryAgent(summary, answered)
    result = _finalize(agent, final_response=None, exit_reason="budget_exhausted")
    assert result["turn_exit_reason"].startswith("max_iterations_reached")
    assert result["model_answer"] is expected


def test_a_stale_summary_flag_does_not_leak_into_the_next_budget_exit():
    agent = _SummaryAgent("I reached the iteration limit and couldn't generate a summary.", False)
    agent._iteration_summary_answered = True  # left over from an earlier turn
    agent._handle_max_iterations = lambda messages, n: agent._summary  # sets nothing this time
    result = _finalize(agent, final_response=None, exit_reason="budget_exhausted")
    assert result["model_answer"] is False


def test_a_preserved_verification_answer_is_an_answer():
    agent = _LimitAgent()
    result = _finalize(agent, final_response=None, exit_reason="budget_exhausted",
                       pending_verification_response="Composed answer withheld by the verifier.")
    assert result["model_answer"] is True


@pytest.mark.parametrize("outcome,expected", [("text", True), ("empty", False), ("raise", False)])
def test_handle_max_iterations_reports_whether_the_model_wrote_the_summary(outcome, expected):
    from agent import chat_completion_helpers as helpers

    agent = _make_agent()

    def build(_agent, _messages, _request_id):
        def attempt(_retry):
            if outcome == "raise":
                raise RuntimeError("summary call failed")
            return "Model-written summary." if outcome == "text" else ""
        return attempt

    with patch.dict(helpers._SUMMARY_ATTEMPT_BUILDERS, {agent.api_mode: build}):
        text = helpers.handle_max_iterations(agent, [{"role": "user", "content": "task"}], 10)
    assert bool(text.strip())
    assert agent._iteration_summary_answered is expected


def _rewriting_hook(rewrite):
    def transform(agent, response, **_kwargs):
        return rewrite(response), True, response
    return transform


@pytest.mark.parametrize("model_text,rewrite,expected", [
    # Twelfth supplementary review: an output hook turning the empty sentinel into prose.
    ("(empty)", lambda r: "⚠️ No reply: the model produced nothing.", False),
    # A silence marker stays silence even when later decoration defeats the gateway's exact match.
    ("[SILENT]", lambda r: r, False),
    ("[SILENT]", lambda r: r + "\n\n📝 Files changed: a.py", False),
    # Control: a hook reshaping a real answer leaves it an answer.
    ("The answer is 42.", lambda r: r.upper(), True),
])
def test_provenance_is_judged_on_the_untransformed_model_text(model_text, rewrite, expected):
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [_mock_response(content=model_text)]
    with patch("agent.turn_finalizer.apply_llm_output_transform", _rewriting_hook(rewrite)):
        result = _run(agent)
    assert result["turn_exit_reason"].startswith("text_response")
    assert result["model_answer"] is expected
