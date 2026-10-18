"""#92 review round 3 (H92-03): a turn the selector accepts as well formed is this event's answer only when
it opened with this claim's own row and carries no other peer row — the provenance facts
(``turn_provenance``) that also decide, with the H92-02 Limit on top, whether a refused answer is kept. Otherwise it takes the existing explicit-failure path: never HTTP 200,
never recorded as the answer, never posted; the failure notice is kept.

The reviewer's witness (``h92-r3/review_evidence/test_suffix_normal_control.py``) is copied below byte for
byte under its banner; only its imports and fixture aliases are merged into this header. Ours follow: each
foreign-row position on the normal path, the binding that cannot be confirmed, and normal-path controls —
including the opening row built by the agent's own staging and persist-override code.
"""
import json
from types import SimpleNamespace

import pytest
import test_shim_d2_refused_answer_kept as o4

gw = o4.gw
shim = o4.shim
_pin_the_module_home = o4._pin_the_module_home


# ════════════════ reviewer: test_suffix_normal_control.py (verbatim) ════════════════

@pytest.mark.parametrize("place", ["opening", "final", "tool"])
def test_foreign_peer_in_an_otherwise_normal_suffix_is_refused(gw, shim, monkeypatch, place):
    gw.actor.shape = "plain"
    run = gw.actor.run_conversation

    def foreign(*args, **kwargs):
        result = run(*args, **kwargs)
        meta = o4.new_peer_metadata(binding="canonical", author_id=o4.SIGNER, channel_id=o4.CHANNEL,
                                   event_id="f" * 64, receipt="canonical-receipt:v2:" + "e" * 64)
        if place == "opening":
            row = result["messages"][gw.actor._persist_user_message_idx]
            row["content"] = "another event's request"
        elif place == "final":
            row = result["messages"][-1]
        else:
            result["messages"].insert(-1, {"role": "assistant", "tool_calls": [{"id": "call-1"}]})
            row = {"role": "tool", "tool_call_id": "call-1", "content": "foreign data"}
            result["messages"].insert(-1, row)
        row.update(display_kind="canonical_peer", display_metadata={"canonical_peer": meta})
        return result

    monkeypatch.setattr(gw.actor, "run_conversation", foreign)
    o4._drive(shim)
    o4._attributed_and_submitted(gw, shim)
    print("NORMAL_FOREIGN_SUFFIX", json.dumps({"codes": o4._codes(gw), "kept": o4._held(gw),
          "bound": gw.bound, "posts": shim.posts()}))
    assert o4._codes(gw) == [(409, "canonical_event_failed")]
    assert o4._held(gw) == []


def test_normal_same_event_control(gw, shim):
    gw.actor.shape = "plain"
    o4._drive(shim)
    o4._attributed_and_submitted(gw, shim)
    assert o4._codes(gw) == [(200, "text")]
    assert o4._held(gw) == [o4.FINAL]
    assert [post["stdin"] for post in shim.posts()] == [f"@cto-repo-factory {o4.FINAL}"]


# ════════════════ ours ════════════════


def _another_claim(event_id="f" * 64, receipt="canonical-receipt:v2:" + "e" * 64):
    return o4.new_peer_metadata(binding="canonical", author_id=o4.SIGNER, channel_id=o4.CHANNEL,
                                event_id=event_id, receipt=receipt)


def _agent_row(text, kwargs, overridden):
    """The opening row exactly as the agent builds it (``agent.turn_context._stage_turn_user_message``) and,
    once the turn is finalized, as its persist override leaves it (``_apply_persist_user_message_override``)."""
    from agent.session_persistence import SessionPersistenceMixin
    from agent.turn_context import _stage_turn_user_message

    row, _ = _stage_turn_user_message(SimpleNamespace(_pending_cli_user_message=None), text,
                                      kwargs["persist_user_message"], None, None,
                                      kwargs["persist_user_display_kind"], kwargs["persist_user_display_metadata"])
    if overridden:
        SessionPersistenceMixin._apply_persist_user_message_override(SimpleNamespace(
            _persist_user_message_idx=0, _persist_user_message_override=kwargs["persist_user_message"],
            _persist_user_message_timestamp=None, _persist_user_message_platform_id=None), [row])
    return row


def _rows(gw, monkeypatch, edit):
    run = gw.actor.run_conversation

    def edited(text, **kwargs):
        result = run(text, **kwargs)
        edit(result["messages"], gw.actor._persist_user_message_idx, text, kwargs)
        return result

    monkeypatch.setattr(gw.actor, "run_conversation", edited)


def _the_normal_path_refuses_it(gw, shim):
    o4._attributed_and_submitted(gw, shim)
    assert gw.refusals == [] and len(gw.runs) == 1, "reach: the selector accepted the turn's shape"
    assert o4._codes(gw) == [(409, "canonical_event_failed")], "accepted as this event's answer"
    assert o4._held(gw) == [] and gw.bound == [], "recorded or kept as an answer"
    [post] = shim.posts()  # the existing failure notice, never the answer
    assert "canonical_event_failed" in post["stdin"] and o4.FINAL not in post["stdin"]


def _marked(row, meta):
    row.update(display_kind="canonical_peer", display_metadata={"canonical_peer": meta})


def _opening_of_another_claim_of_this_event(rows, at, text, kwargs):
    row = _agent_row(text, kwargs, overridden=True)
    other = dict(kwargs["persist_user_display_metadata"]["canonical_peer"], nonce="0" * 32)
    _marked(row, other)
    rows[at] = row


def _opening_unmarked_body_only(rows, at, text, kwargs):
    rows[at] = {"role": "user", "content": kwargs["persist_user_message"]}  # no nonce: cannot be confirmed


def _tool_call_row_of_another_event(rows, at, text, kwargs):
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]}
    _marked(call, _another_claim())
    rows[-1:-1] = [call, {"role": "tool", "tool_call_id": "call-1", "content": "data"}]


def _final_peer_kind_only(rows, at, text, kwargs):
    rows[-1]["display_kind"] = "canonical_peer"


def _final_of_another_receipt_of_this_event(rows, at, text, kwargs):
    _marked(rows[-1], _another_claim(event_id=o4.EVENT))


@pytest.mark.parametrize("edit", [_opening_of_another_claim_of_this_event, _opening_unmarked_body_only,
                                  _tool_call_row_of_another_event, _final_peer_kind_only,
                                  _final_of_another_receipt_of_this_event], ids=lambda f: f.__name__.strip("_"))
def test_h92_03_a_well_formed_turn_whose_rows_are_not_this_claims_is_refused(gw, shim, monkeypatch, edit):
    gw.actor.shape = "plain"
    _rows(gw, monkeypatch, edit)
    o4._drive(shim)
    _the_normal_path_refuses_it(gw, shim)


@pytest.mark.parametrize("overridden", [False, True], ids=["as_sent", "after_persist_override"])
def test_h92_03_control_the_agents_own_opening_row_is_answered_once(gw, shim, monkeypatch, overridden):
    def agent_row(rows, at, text, kwargs):
        rows[at] = _agent_row(text, kwargs, overridden)

    gw.actor.shape = "plain"
    _rows(gw, monkeypatch, agent_row)
    o4._drive(shim)
    o4._attributed_and_submitted(gw, shim)
    assert o4._codes(gw) == [(200, "text")] and o4._held(gw) == [o4.FINAL]
    assert [post["stdin"] for post in shim.posts()] == [f"@cto-repo-factory {o4.FINAL}"]


def test_h92_03_control_a_tool_round_trip_of_this_turn_is_answered_once(gw, shim, monkeypatch):
    def tool_round_trip(rows, at, text, kwargs):
        rows[-1:-1] = [{"role": "assistant", "content": "", "tool_calls": [{"id": "call-1"}]},
                       {"role": "tool", "tool_call_id": "call-1", "content": "data"}]

    gw.actor.shape = "plain"
    _rows(gw, monkeypatch, tool_round_trip)
    o4._drive(shim)
    o4._attributed_and_submitted(gw, shim)
    assert gw.refusals == [] and o4._codes(gw) == [(200, "text")] and o4._held(gw) == [o4.FINAL]
    assert [post["stdin"] for post in shim.posts()] == [f"@cto-repo-factory {o4.FINAL}"]


# ── the split: interior rows on the normal path keep their 20462c50 outcome ─────────────────────────
#
# The normal path refuses only what the shared provenance facts refuse (an opening row not this claim's, or
# another peer row); the refused-answer keep adds the H92-02 Limit (no user row inside the turn). Each turn
# below is measured at 20462c50 and must keep that outcome; with a foreign peer row added it is refused.
# The selector itself (unchanged) already refuses any user row after the opening row, so (a) and (c)-in-turn
# were 409 at 20462c50 too: "matching 20462c50" is that outcome, not 200. A summary the real assembly leaves
# inside the turn is a standalone row the selector refuses; the refused path then keeps the answer (409
# canonical_answer_unrecorded), as at 20462c50.

NOTICE = "[ASYNC DELEGATION COMPLETE — deleg-1]\nThe delegated task finished."  # the format of tools/process_registry_notifications.py


def _call(i):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"call-{i}"}]}


def _result(i):
    return {"role": "tool", "tool_call_id": f"call-{i}", "content": f"data {i}"}


def _compacted(messages, start, end, tail_start):
    """Real compaction assembly (``ContextCompressor._assemble_compressed``: head + summary or merged carrier
    + tail, alternation-safe placement); only the summarizer LLM's text is supplied."""
    from unittest.mock import patch

    from agent.context_compressor import ContextCompressor

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(model="test/model", threshold_percent=0.85, protect_first_n=start,
                                       protect_last_n=2, quiet_mode=True)
    return compressor._assemble_compressed(messages, start, end,
                                           SimpleNamespace(tail_start=tail_start, summary_indices=set()),
                                           "SUMMARY OF THE COMPACTED TURNS")


def _turn(shape, foreign):
    """(run_conversation, where the boundary row is) for a same-event turn of *shape*."""
    from agent.turn_context import reanchor_current_turn_user_idx

    def run(actor, text, kwargs):
        opening = _agent_row(text, kwargs, overridden=False)
        final = {"role": "assistant", "content": o4.FINAL, "finish_reason": "stop"}
        if shape == "a_nudge":  # turn_truncation.py appends the nudge as a bare user row
            rows = [opening, {"role": "assistant", "content": "", "finish_reason": "incomplete"},
                    {"role": "user", "content": o4._CODEX_INCOMPLETE_NUDGE}, final]
        elif shape == "b_compaction_before_opening":  # the turn's row is in the protected tail
            rows = _compacted([{"role": "user", "content": "owner one"}, {"role": "assistant", "content": "a1"},
                               {"role": "user", "content": "owner two"}, {"role": "assistant", "content": "a2"},
                               opening, _call(1), _result(1)], 2, 4, 4) + [final]
        elif shape == "b_compaction_inside_turn":  # the turn's row is the protected head; its middle compacts
            rows = _compacted([opening, _call(1), _result(1), _call(2), _result(2)], 1, 3, 3) + [final]
        elif shape == "c_notice_in_history":  # where MCP reload / delegation notices land: before the turn
            rows = [{"role": "user", "content": NOTICE}, {"role": "assistant", "content": "noted"}, opening, final]
        else:  # c_steered_into_the_turn: a busy-session steer appends a real user row mid-turn
            rows = [opening, _call(1), _result(1), {"role": "user", "content": NOTICE}, final]
        if foreign:
            _marked(rows[-1], _another_claim())
        actor.runs.append(text)
        actor._persist_user_message_idx = reanchor_current_turn_user_idx(rows, text)
        return {"completed": True, "session_id": actor.session_id, "final_response": o4.FINAL, "messages": rows}

    return run


SHAPES = ["a_nudge", "b_compaction_before_opening", "b_compaction_inside_turn", "c_notice_in_history",
          "c_steered_into_the_turn"]
AT_20462C50 = {  # (HTTP outcome, answer held) measured at 20462c50: r3fix/split-measurements-20462c50.txt
    "a_nudge": ([(409, "canonical_event_failed")], []),
    "b_compaction_before_opening": ([(200, "text")], [o4.FINAL]),
    "b_compaction_inside_turn": ([(409, "canonical_answer_unrecorded")], [o4.FINAL]),
    "c_notice_in_history": ([(200, "text")], [o4.FINAL]),
    "c_steered_into_the_turn": ([(409, "canonical_event_failed")], [])}


def _drive_turn(gw, shim, monkeypatch, shape, foreign):
    run = _turn(shape, foreign)
    monkeypatch.setattr(gw.actor, "run_conversation", lambda text, **kwargs: run(gw.actor, text, kwargs))
    o4._drive(shim)
    o4._attributed_and_submitted(gw, shim)
    outcome = {"codes": o4._codes(gw), "held": o4._held(gw), "posts": [p["stdin"][:60] for p in shim.posts()],
               "selector_refused": bool(gw.refusals), "bound": gw.bound}
    print("SPLIT_MEASUREMENT", shape, "foreign" if foreign else "same-event", json.dumps(outcome))
    return outcome


@pytest.mark.parametrize("shape", SHAPES)
def test_split_a_same_event_turn_keeps_its_20462c50_outcome(gw, shim, monkeypatch, shape):
    out = _drive_turn(gw, shim, monkeypatch, shape, foreign=False)
    codes, held = AT_20462C50[shape]
    assert (out["codes"], out["held"]) == (codes, held), "the outcome differs from 20462c50's"
    if codes == [(200, "text")]:
        assert [p["stdin"] for p in shim.posts()] == [f"@cto-repo-factory {o4.FINAL}"]
    else:  # refused by the selector; delivered never, the sender told once
        assert out["selector_refused"] and len(shim.posts()) == 1 and o4.FINAL not in shim.posts()[0]["stdin"]


@pytest.mark.parametrize("shape", SHAPES)
def test_split_the_same_turn_with_a_foreign_peer_row_is_refused_on_either_path(gw, shim, monkeypatch, shape):
    out = _drive_turn(gw, shim, monkeypatch, shape, foreign=True)
    assert out["codes"] == [(409, "canonical_event_failed")], "a turn with another event's peer row answered"
    assert out["held"] == [] and len(shim.posts()) == 1 and o4.FINAL not in shim.posts()[0]["stdin"]
