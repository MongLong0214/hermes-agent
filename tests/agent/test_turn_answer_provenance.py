"""Answer provenance for /acp receipts (ESCAPE-SUPP-18/19/20, SUPP13-R1).

Producers record two texts at the one output-hook seam, before any footer, explainer or media
decoration: ``answer_origin`` (the text the model produced, handed to the hook) and ``answer_body``
(the text the hook returned), plus ``answer_disposition`` (what the hook declared it did to the
answer), and only for exits that produce the model's own text. The gateway's certification boundary
(``GatewayTurnMixin._acp_is_model_answer``) judges them and the delivered reply with one rule. These
tests drive the real finalizer and conversation loop and judge their results at that boundary.
"""

from unittest.mock import patch

import pytest

from gateway.run_turn import GatewayTurnMixin
from tests.agent.test_turn_completion_explainer import _make_agent, _mock_response
from tests.agent.test_turn_finalizer_iteration_limit_exit import _finalize, _LimitAgent

NON_ANSWERS = ["(empty)", "[SILENT]", "NO_REPLY", "<|eos|>", "   "]
NOTICE = "⚠️ No reply: the model produced nothing. Send `continue` to try again."
HOOK_NOTICE = "No reply: the model produced no usable answer."
MARKERS = ["(empty)", "[SILENT]", "NO_REPLY", "<|eos|>"]
MISSING_MEDIA = "\nMEDIA:/nonexistent/acp-probe/missing.png"
FOOTER = "\n\n📝 Files changed: a.py (patch failed)"


def _certifies(result):
    return GatewayTurnMixin._acp_is_model_answer(result)


def _preserved(text):
    return {"text": text, "answer": "preserved"}


def _hook(rewrite):
    """A transform_llm_output plugin returning ``rewrite(text)`` (None leaves the text unchanged); a
    rewrite may return a plain string or a ``{"text", "answer"}`` declaration."""
    def invoke(name, logger, **kwargs):
        if name != "transform_llm_output":
            return []
        out = rewrite(kwargs["response_text"])
        return [] if out is None else [out]
    return patch("agent.turn_finalizer._invoke_hook_safely", side_effect=invoke)


def _decorate(suffix):
    """Footer decoration, added after the hook seam exactly where the real mutation footer is."""
    return patch("agent.turn_finalizer._append_file_mutation_footer",
                 side_effect=lambda agent, text, logger: f"{text}{suffix}" if text else text)


def _run(agent):
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation("do something")


# --- the conversation loop (finish_text_response seam) ---------------------------------------


def test_a_plain_answer_certifies():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [_mock_response(content="The answer is 42.")]
    result = _run(agent)
    assert result["answer_origin"] == result["answer_body"] == "The answer is 42."
    assert result["answer_disposition"] == "unchanged"
    assert _certifies(result)


def test_the_empty_response_explainer_does_not_certify():
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="", finish_reason="stop") for _ in range(8)
    ]
    result = _run(agent)
    assert "No reply:" in result["final_response"] and result["api_calls"] > 0
    assert result["answer_origin"] is None and not _certifies(result)


def test_recovered_stream_text_certifies():
    agent = _make_agent()
    recovered = "I inspected the gateway and the turn stopped after the stream timed out."

    def _fake_api_call(_api_kwargs):
        agent._current_streamed_assistant_text = recovered
        return _mock_response(content=None, finish_reason="stop")

    with patch.object(agent, "_interruptible_api_call", side_effect=_fake_api_call):
        result = _run(agent)
    assert result["turn_exit_reason"] == "partial_stream_recovery"
    assert _certifies(result)


@pytest.mark.parametrize("model_text,rewrite,expected", [
    ("(empty)", lambda r: NOTICE, False),                 # a hook cannot create an answer
    ("[SILENT]", lambda r: None, False),
    ("NO_REPLY", lambda r: NOTICE, False),
    ("<|eos|>", lambda r: None, False),
    ("The answer is 42.", lambda r: "(empty)", False),    # nor keep one it replaced (SUPP13-R1)
    ("The answer is 42.", lambda r: "   ", False),
    ("The answer is 42.", lambda r: _preserved(r.upper()), True),  # control: a declared reformat
    ("The answer is 42.", lambda r: r.upper(), False),    # an undeclared rewrite proves nothing
    ("The answer is 42.", lambda r: HOOK_NOTICE, False),  # SUPP13-R1 ①: a hook-written notice
])
def test_a_text_response_is_judged_at_both_sides_of_the_hook(model_text, rewrite, expected):
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [_mock_response(content=model_text)]
    with _hook(rewrite):
        result = _run(agent)
    assert result["turn_exit_reason"].startswith("text_response")
    assert _certifies(result) is expected


# --- every finalizer producer (its _persist_step seam) ----------------------------------------


class _ProducerAgent(_LimitAgent):
    def __init__(self, *, summary=None, answered=False, budget_remaining=1):
        super().__init__(budget_remaining=budget_remaining)
        self._summary, self._answered = summary, answered

    def _emit_diagnostic_status(self, *_args, **_kwargs):
        pass

    def _handle_max_iterations(self, messages, api_call_count):
        self._iteration_summary_answered = self._answered
        return self._summary


def _produce(producer, text, *, answered=True):
    """Finalize a turn whose model text ``text`` came from ``producer``."""
    if producer == "budget-summary":
        agent = _ProducerAgent(summary=text, answered=answered, budget_remaining=0)
        return _finalize(agent, final_response=None, exit_reason="budget_exhausted")
    if producer == "verification-candidate":
        return _finalize(_ProducerAgent(budget_remaining=0), final_response=None,
                         exit_reason="budget_exhausted", pending_verification_response=text)
    exit_reason = {"stream-recovery": "partial_stream_recovery",
                   "housekeeping": "fallback_prior_turn_content",
                   "text": "text_response(finish_reason=stop)"}[producer]
    return _finalize(_ProducerAgent(), final_response=text, exit_reason=exit_reason, api_call_count=2)


PRODUCERS = ["text", "stream-recovery", "housekeeping", "budget-summary", "verification-candidate"]


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("decorated", [False, True])
def test_each_producer_certifies_a_genuine_answer(producer, decorated):
    with _hook(lambda r: None), _decorate(FOOTER if decorated else ""):
        assert _certifies(_produce(producer, "The answer is 42."))


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("model_text", NON_ANSWERS)
@pytest.mark.parametrize("rewrite", [lambda r: None, lambda r: NOTICE, lambda r: _preserved(NOTICE)],
                         ids=["unchanged", "hook-notice", "hook-declared-notice"])
@pytest.mark.parametrize("decorated", [False, True])
def test_no_producer_certifies_a_non_answer_however_it_is_shaped(producer, model_text, rewrite, decorated):
    with _hook(rewrite), _decorate(FOOTER if decorated else ""):
        assert not _certifies(_produce(producer, model_text))


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("replacement", ["(empty)", "   ", "[SILENT]", "<|eos|>"])
def test_no_producer_certifies_a_genuine_answer_a_hook_replaced(producer, replacement):
    with _hook(lambda r: replacement), _decorate(FOOTER):
        assert not _certifies(_produce(producer, "The answer is 42."))


def test_a_budget_summary_the_model_did_not_write_does_not_certify():
    with _hook(lambda r: None):
        result = _produce("budget-summary", "I reached the iteration limit and couldn't generate a summary.",
                          answered=False)
    assert result["answer_origin"] is None and not _certifies(result)


def test_a_stale_summary_flag_does_not_leak_into_the_next_budget_exit():
    agent = _ProducerAgent(summary="I reached the iteration limit and couldn't generate a summary.",
                           budget_remaining=0)
    agent._iteration_summary_answered = True  # left over from an earlier turn
    agent._handle_max_iterations = lambda messages, n: agent._summary  # sets nothing this time
    with _hook(lambda r: None):
        result = _finalize(agent, final_response=None, exit_reason="budget_exhausted")
    assert not _certifies(result)


def test_a_seam_record_from_another_turn_is_not_provenance():
    agent = _ProducerAgent()
    agent._llm_output_seam = ("an-earlier-turn", "The answer is 42.", "The answer is 42.", "unchanged")
    agent._llm_output_transform = ("turn", False, None)  # this turn's hook already fired, on nothing
    result = _finalize(agent, final_response="⚠️ notice", exit_reason="text_response(finish_reason=stop)")
    assert result["answer_origin"] is None and not _certifies(result)


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


# --- hook provenance (SUPP13-R1) ---------------------------------------------------------------


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("rewrite,disposition", [
    (lambda r: HOOK_NOTICE, "undeclared"),                                    # ① plain string
    (lambda r: {"text": HOOK_NOTICE}, "undeclared"),                          # no declaration
    (lambda r: {"text": HOOK_NOTICE, "answer": "kept"}, "undeclared"),        # invalid declaration
    (lambda r: {"text": HOOK_NOTICE, "answer": "suppressed"}, "suppressed"),
    (lambda r: {"text": "The answer is 42 (reformatted).", "answer": "suppressed"}, "suppressed"),
], ids=["plain-notice", "mapping-undeclared", "mapping-invalid", "suppressed-notice", "suppressed-answerlike"])
@pytest.mark.parametrize("decorated", [False, True])
def test_no_producer_certifies_an_answer_the_hook_did_not_declare_preserved(producer, rewrite, disposition,
                                                                          decorated):
    with _hook(rewrite), _decorate(FOOTER if decorated else ""):
        result = _produce(producer, "The answer is 42.")
    assert result["answer_disposition"] == disposition
    assert result["answer_origin"] == "The answer is 42." and not _certifies(result)


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("model_text", ["42", '{"answer":42}', "The answer is 42."])
@pytest.mark.parametrize("decorated", [False, True])
def test_each_producer_certifies_a_declared_preserving_rewrite(producer, model_text, decorated):
    with _hook(lambda r: _preserved("The answer is 42.")), _decorate(FOOTER if decorated else ""):
        result = _produce(producer, model_text)
    assert result["answer_disposition"] == "preserved"
    assert result["answer_body"] == "The answer is 42." and _certifies(result)


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("marker", MARKERS)
@pytest.mark.parametrize("declare", [_preserved, lambda t: t], ids=["preserved", "plain"])
@pytest.mark.parametrize("decorated", [False, True])
def test_no_producer_certifies_a_marker_left_once_attachments_are_extracted(producer, marker, declare, decorated):
    """SUPP13-R1 ②: what the adapter sends of ``marker + MEDIA:<missing>`` is the marker alone, so a
    declared "preserved" cannot carry it, with or without the footer behind it."""
    with _hook(lambda r: declare(marker + MISSING_MEDIA)), _decorate(FOOTER if decorated else ""):
        assert not _certifies(_produce(producer, "The answer is 42."))


@pytest.mark.parametrize("producer", PRODUCERS)
@pytest.mark.parametrize("hooked", [False, True])
def test_a_genuine_answer_with_a_real_attachment_certifies(producer, hooked, tmp_path):
    chart = tmp_path / "chart.png"
    chart.write_bytes(b"\x89PNG")
    answer = f"The answer is 42.\nMEDIA:{chart}"
    with _hook((lambda r: _preserved(r)) if hooked else (lambda r: None)), _decorate(FOOTER):
        assert _certifies(_produce(producer, answer))


def test_the_first_declared_text_wins_and_an_empty_one_passes_the_turn_on():
    """The mapping's text is consumed exactly as a plain string: an empty one leaves the response
    for the next plugin, and the winner's declaration is the turn's."""
    from agent.turn_finalizer import apply_llm_output_transform

    agent = _ProducerAgent()
    results = [{"text": "", "answer": "suppressed"}, _preserved("The answer is 42."), "ignored"]
    with patch("agent.turn_finalizer._invoke_hook_safely", return_value=results):
        text, transformed, pre = apply_llm_output_transform(agent, "42", turn_id="t1", logger=None)
    assert (text, transformed, pre) == ("The answer is 42.", True, "42")
    assert agent._llm_output_seam == ("t1", "42", "The answer is 42.", "preserved")


@pytest.mark.parametrize("rewrite,expected", [
    (lambda r: HOOK_NOTICE, False),
    (lambda r: {"text": HOOK_NOTICE, "answer": "suppressed"}, False),
    (lambda r: _preserved("The answer is 42."), True),
])
def test_the_conversation_loop_carries_the_hooks_declaration(rewrite, expected):
    agent = _make_agent()
    agent.client.chat.completions.create.side_effect = [_mock_response(content="42")]
    with _hook(rewrite):
        result = _run(agent)
    assert result["turn_exit_reason"].startswith("text_response")
    assert _certifies(result) is expected
