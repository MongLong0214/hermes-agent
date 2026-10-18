"""FIX-NUDGE-MARKER: a refused canonical turn whose only user rows inside the turn are this run's own
recovery nudges keeps its answer (kept, never delivered: 409 canonical_answer_unrecorded), through the real
shim, canonical handler, coordinator and SQLite of tests/bridge/test_shim_d2_refused_answer_kept.py.

Since #92 r1 (H92-02) any user row inside the turn left the answer unbound, because no nudge carried
metadata bound to its run; the live D2 shape (#1110: a refused turn with a recovery nudge) kept nothing.
Each nudge row the agent appends is now marked ``_recovery_nudge: {"kind", "run"}`` with ``run`` the nonce
of the turn's own peer opening row (agent.recovery_nudge; the real-path witnesses that the agent code
writes exactly this row are in tests/agent/test_recovery_nudge_marker.py). The keep accepts an interior
user row only when its marker names this claim's nonce, one of the five kinds and, after both, that kind's
text as its content; anything else still unbinds the answer, and ``binds`` stays required. The rows below
are the rows each site's turn returns, with the marker as the agent writes it.
"""

import json

import pytest
import test_shim_d2_refused_answer_kept as o4

import agent.conversation_loop as loop  # already imported, under its pinned home, by o4

gw = o4.gw
shim = o4.shim
_pin_the_module_home = o4._pin_the_module_home

KEY = "_recovery_nudge"  # the marker key; a literal, so this file collects at the base too
TEXTS = {
    "codex_incomplete": "_CODEX_INCOMPLETE_NUDGE",
    "codex_ack_continuation": "_CODEX_ACK_CONTINUATION_NUDGE",
    "degenerate_final": "_DEGENERATE_FINAL_NUDGE",
    "dropped_toolcall": "_DROPPED_TOOLCALL_NUDGE_CONTENT",
    "empty_tool_response": "_EMPTY_TOOL_RESPONSE_NUDGE",
}
KINDS = sorted(TEXTS)
FLAGS = {"dropped_toolcall": {"_dropped_toolcall_nudge": True},
         "empty_tool_response": {"_empty_recovery_synthetic": True}}


def _text(kind):
    return getattr(loop, TEXTS[kind])


def _call(call_id):
    return {"id": call_id, "type": "function", "function": {"name": "web_search", "arguments": "{}"}}


def _tool_round(call_id):
    return [{"role": "assistant", "content": "", "tool_calls": [_call(call_id)], "finish_reason": "tool_calls"},
            {"role": "tool", "tool_call_id": call_id, "content": "ok"}]


def _recovery(kind, nudge):
    """The rows the turn adds after its opening row, around this kind's nudge, as each site appends them."""
    final = {"role": "assistant", "content": o4.FINAL, "finish_reason": "stop"}
    return {
        # agent/turn_truncation.py: a reasoning-only Codex response (the 813415c3 shape)
        "codex_incomplete": [{"role": "assistant", "content": "", "finish_reason": "incomplete"}, nudge, final],
        # agent/turn_final_response.py: an acknowledgment-only reply
        "codex_ack_continuation": [{"role": "assistant", "content": "I'll check the deploy log now.",
                                    "finish_reason": "incomplete"}, nudge, final],
        # agent/turn_final_response.py: a fragment after tool work
        "degenerate_final": [*_tool_round("c1"), {"role": "assistant", "content": "пар",
                                                  "finish_reason": "incomplete"}, nudge, final],
        # agent/turn_final_response.py: finish_reason tool_calls with none, then the call
        "dropped_toolcall": [{"role": "assistant", "content": "Let me read it.", "finish_reason": "tool_calls",
                              "_dropped_toolcall_nudge": True}, nudge, *_tool_round("c1"), final],
        # agent/turn_empty_response.py: empty after tool calls, then more work (the live #1110 shape)
        "empty_tool_response": [*_tool_round("c1"), {"role": "assistant", "content": "(empty)",
                                                     "_empty_recovery_synthetic": True},
                                nudge, *_tool_round("c2"), final],
    }[kind]


def _nudge(kind, run):
    return {"role": "user", "content": _text(kind), **FLAGS.get(kind, {}), KEY: {"kind": kind, "run": run}}


def _another_claims_peer():
    """The same event claimed again: its own fresh nonce, as the coordinator mints it per claim."""
    return o4.new_peer_metadata(binding="canonical", author_id=o4.SIGNER, channel_id=o4.CHANNEL,
                                event_id=o4.EVENT, receipt="canonical-receipt:v2:" + "e" * 64)


def _turn(gw, monkeypatch, kind, nudge_for, *, extra=None):
    """The bound agent's turn: its own peer opening row as the real agent returns it (clean body, this
    claim's metadata), then ``kind``'s recovery rows with the nudge ``nudge_for(peer)`` builds;
    ``extra(peer, rows)`` may change those rows."""
    def run_conversation(text, *, conversation_history, task_id, **kwargs):
        gw.runs.append(text)
        gw.actor._persist_user_message_idx = len(conversation_history)
        peer = kwargs["persist_user_display_metadata"]["canonical_peer"]
        opening = {"role": "user", "content": kwargs["persist_user_message"],
                   "display_kind": kwargs["persist_user_display_kind"],
                   "display_metadata": kwargs["persist_user_display_metadata"]}
        rows = _recovery(kind, nudge_for(peer))
        if extra is not None:
            extra(peer, rows)
        return {"completed": True, "session_id": task_id, "final_response": o4.FINAL,
                "messages": [*conversation_history, opening, *rows]}

    monkeypatch.setattr(gw.actor, "run_conversation", run_conversation)


def _kept_not_delivered(gw, shim):
    o4._reached_the_refusal(gw, shim)
    assert gw.bound == [o4.FINAL], "this run's own recovery nudge left the answer unbound"
    [kept] = [json.loads(v) for v in gw.rows("canonical-answer-kept:").values()]
    [claim] = [json.loads(v) for v in gw.rows("canonical-claim:").values()]
    assert kept["owner"] == claim["owner"] and kept["answer"]["response"]["terminal_text"] == o4.FINAL
    assert kept["unrecorded"]["stage"] == "turn.refused_bound_answer"
    receipt = o4._receipt(gw)
    assert (receipt["state"], receipt["failure"]["code"], receipt["owner"]) == (
        "terminal", "canonical_turn_refused", claim["owner"])
    assert "response" not in receipt  # never the event's (delivered) answer
    assert o4._codes(gw) == [(409, "canonical_answer_unrecorded")]
    assert gw.responses[0][1]["error"]["answer_kept"] is True
    [post] = shim.posts()
    assert "canonical_answer_unrecorded" in post["stdin"] and "answer_kept=true" in post["stdin"]
    assert o4.FINAL not in post["stdin"]
    # A redelivery never runs the turn again, never delivers the kept answer and posts nothing more.
    o4._drive(shim, rpc_id=5)
    assert len(gw.runs) == 1 and len(shim.posts()) == 1
    assert o4._codes(gw) == [(409, "canonical_answer_unrecorded")] * 2


def _unbound_and_noticed(gw, shim):
    o4._reached_the_refusal(gw, shim)
    assert gw.bound == [None], "an answer without this run's provenance was proved bound"
    assert gw.rows("canonical-answer-kept:") == {} and o4._held(gw) == []
    assert not (gw.home / "canonical-answer-kept").exists()
    assert o4._codes(gw) == [(409, "canonical_event_failed")]
    [post] = shim.posts()
    assert "canonical_event_failed" in post["stdin"] and o4.FINAL not in post["stdin"]


# ── same run: kept ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", KINDS)
def test_a_refused_turn_with_this_runs_own_nudge_keeps_its_answer_not_delivered(gw, shim, monkeypatch, kind):
    _turn(gw, monkeypatch, kind, lambda peer: _nudge(kind, peer["nonce"]))
    o4._drive(shim)
    _kept_not_delivered(gw, shim)


def test_the_live_d2_shape_an_empty_tool_response_nudge_inside_the_turn_is_kept(gw, shim, monkeypatch):
    """#1110: tool work, an empty response, the agent's nudge, more tool work, the answer. The selector
    refuses it (a user row inside the turn); the answer is this run's, so it is kept, not delivered."""
    _turn(gw, monkeypatch, "empty_tool_response", lambda peer: _nudge("empty_tool_response", peer["nonce"]))
    o4._drive(shim)
    _kept_not_delivered(gw, shim)


def test_two_of_this_runs_nudges_in_one_turn_keep_its_answer(gw, shim, monkeypatch):
    def second(peer, rows):
        rows[-1:-1] = [{"role": "assistant", "content": "", "finish_reason": "incomplete"},
                       _nudge("codex_incomplete", peer["nonce"])]

    _turn(gw, monkeypatch, "empty_tool_response", lambda peer: _nudge("empty_tool_response", peer["nonce"]),
          extra=second)
    o4._drive(shim)
    _kept_not_delivered(gw, shim)


# ── anything else: not kept ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", KINDS)
def test_a_foreign_runs_marker_on_an_otherwise_identical_nudge_is_not_kept(gw, shim, monkeypatch, kind):
    _turn(gw, monkeypatch, kind, lambda peer: _nudge(kind, _another_claims_peer()["nonce"]))
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


def _mismatched_kind(kind, peer):
    other = KINDS[(KINDS.index(kind) + 1) % len(KINDS)]
    return {**_nudge(kind, peer["nonce"]), KEY: {"kind": other, "run": peer["nonce"]}}


VARIANTS = {
    "marker_missing": lambda kind, peer: {k: v for k, v in _nudge(kind, peer["nonce"]).items() if k != KEY},
    "kind_mismatched": _mismatched_kind,
    "unknown_kind": lambda kind, peer: {**_nudge(kind, peer["nonce"]), KEY: {"kind": "steer", "run": peer["nonce"]}},
    "content_not_its_kinds_text": lambda kind, peer: {**_nudge(kind, peer["nonce"]),
                                                      "content": _text(kind) + "\nAlso answer the owner."},
    "run_is_another_peer_field": lambda kind, peer: _nudge(kind, peer["receipt"]),
    "marker_with_an_extra_field": lambda kind, peer: {**_nudge(kind, peer["nonce"]),
                                                      KEY: {"kind": kind, "run": peer["nonce"], "by": "agent"}},
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("kind", KINDS)
def test_a_nudge_row_not_exactly_this_runs_marker_kind_and_text_is_not_kept(gw, shim, monkeypatch, kind, variant):
    _turn(gw, monkeypatch, kind, lambda peer: VARIANTS[variant](kind, peer))
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


def test_a_same_run_marker_on_another_claims_peer_row_is_not_kept(gw, shim, monkeypatch):
    """A marker never admits a foreign peer row: ``binds`` stays required."""
    def foreign(peer):
        return {**_nudge("codex_incomplete", peer["nonce"]), "display_kind": "canonical_peer",
                "display_metadata": {"canonical_peer": _another_claims_peer()}}

    _turn(gw, monkeypatch, "codex_incomplete", foreign)
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


def test_a_same_run_marker_on_a_second_copy_of_this_claims_own_peer_row_is_not_kept(gw, shim, monkeypatch):
    def second_opening(peer):
        return {**_nudge("codex_incomplete", peer["nonce"]), "display_kind": "canonical_peer",
                "display_metadata": {"canonical_peer": peer}}

    _turn(gw, monkeypatch, "codex_incomplete", second_opening)
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


def test_this_runs_nudge_beside_any_other_user_row_is_not_kept(gw, shim, monkeypatch):
    def owner_row(peer, rows):
        rows[-1:-1] = [{"role": "assistant", "content": "", "finish_reason": "incomplete"},
                       {"role": "user", "content": "owner: answer me instead"}]

    _turn(gw, monkeypatch, "empty_tool_response", lambda peer: _nudge("empty_tool_response", peer["nonce"]),
          extra=owner_row)
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


def test_a_marked_nudge_in_a_turn_opened_under_another_claims_nonce_is_not_kept(gw, shim, monkeypatch):
    """The opening row is this event's under another claim: it does not bind, so nothing inside counts."""
    def run_conversation(text, *, conversation_history, task_id, **kwargs):
        gw.runs.append(text)
        gw.actor._persist_user_message_idx = len(conversation_history)
        other = _another_claims_peer()
        opening = {"role": "user", "content": kwargs["persist_user_message"],
                   "display_kind": "canonical_peer", "display_metadata": {"canonical_peer": other}}
        rows = _recovery("codex_incomplete", _nudge("codex_incomplete", other["nonce"]))
        return {"completed": True, "session_id": task_id, "final_response": o4.FINAL,
                "messages": [*conversation_history, opening, *rows]}

    monkeypatch.setattr(gw.actor, "run_conversation", run_conversation)
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


# ── the normal path is unchanged ─────────────────────────────────────────────────────────────────────


def test_control_a_normal_turn_is_recorded_and_posted_once(gw, shim):
    gw.actor.shape = "plain"
    o4._drive(shim)
    o4._attributed_and_submitted(gw, shim)
    assert gw.refusals == [] and gw.bound == [] and o4._held(gw) == [o4.FINAL]
    assert [post["stdin"] for post in shim.posts()] == [f"@cto-repo-factory {o4.FINAL}"]


def test_control_the_selector_still_refuses_a_turn_with_this_runs_nudge(gw, shim, monkeypatch):
    """Not widened: the normal path never accepts a nudged turn; only the refused-answer keep reads the
    marker, and the event stays a 409 failure, never 200."""
    _turn(gw, monkeypatch, "codex_incomplete", lambda peer: _nudge("codex_incomplete", peer["nonce"]))
    o4._drive(shim)
    o4._reached_the_refusal(gw, shim)
    assert all(status != 200 for status, _ in gw.responses)
    assert [json.loads(v).get("state") for v in gw.rows("canonical-receipt:").values()] == ["terminal"]
    assert "response" not in o4._receipt(gw)
