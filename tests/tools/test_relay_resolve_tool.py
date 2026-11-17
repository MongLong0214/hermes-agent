"""``relay_resolve``: the structured declaration a relayed owner turn closes its instruction with.

Reply recovery is not re-execution (CEO, 2026-11-17): the declaration is a tool call, not a word in an
answer, and a recovery turn can make no other. These cover the handler's own gate (the relayed owner turn
and the event it was bound to), what the tool is not (a tool every turn sees), and the reply-only refusal the
executor applies to every other call. What a declaration must cite to count is judged by the gateway, from
the turn's own rows (tests/gateway/test_canonical_owner_relay.py).
"""
import json
from types import SimpleNamespace

import pytest

from tools.relay_resolve_tool import (
    OUTCOMES, TOOL_NAME, bind_scope, evidence_items, is_reply_only_turn, reset_reply_only, reset_scope,
    set_reply_only,
)
from tools.registry import registry


def _call(args):
    result = registry.dispatch(TOOL_NAME, args)
    return json.loads(result) if isinstance(result, str) else result


@pytest.fixture
def scope():
    token = bind_scope("event-1")
    yield "event-1"
    reset_scope(token)


GOOD = {"event_id": "event-1", "outcome": "done", "evidence": [{"tool_call_id": "call-1"}]}


def test_the_tool_declares_one_event_resolved_from_inside_its_relayed_owner_turn(scope):
    assert _call(GOOD) == {"declared": True, "event_id": "event-1", "outcome": "done"}
    assert set(OUTCOMES) == {"done", "answered", "declined", "deferred_with_reason"}


def test_the_tool_refuses_anywhere_but_a_relayed_owner_turn():
    """No turn but a relayed owner turn binds a scope, so a call from any other (an ordinary turn, a peer's,
    a cron job) declares nothing."""
    assert "error" in _call(GOOD)


def test_a_resolve_for_another_event_is_refused(scope):
    refused = _call({**GOOD, "event_id": "event-2"})
    assert "error" in refused and "declared" not in refused


@pytest.mark.parametrize("change", [
    {"outcome": "finished"}, {"outcome": ""}, {"outcome": None},
    {"evidence": []}, {"evidence": "see above"}, {"evidence": None}, {"evidence": [{}]},
    {"evidence": [{"note": "x"}]}, {"evidence": [{"tool_call_id": ""}]}, {"evidence": [{"answer": "   "}]},
    {"evidence": [{"tool_call_id": "c", "answer": "both"}]}, {"evidence": ["call-1"]},
    {"evidence": [{"tool_call_id": 7}]},
])
def test_a_declaration_without_a_known_outcome_and_well_formed_evidence_is_refused(scope, change):
    refused = _call({**GOOD, **change})
    assert "error" in refused and "declared" not in refused


def test_evidence_is_a_closed_list_of_tool_results_and_explicit_answers():
    assert evidence_items([{"tool_call_id": " call-1 "}, {"answer": " stopped "}]) == [
        ("tool", "call-1"), ("answer", "stopped")]
    assert evidence_items([]) is None and evidence_items([{"tool_call_id": "a"}] * 65) is None


def test_the_tool_is_in_no_turns_tool_selection():
    """It is registered so a relayed owner turn can dispatch it, and available to nobody's tool definitions."""
    import model_tools

    names = {tool["function"]["name"] for tool in model_tools.get_tool_definitions(quiet_mode=True)}
    assert TOOL_NAME not in names and registry.get_entry(TOOL_NAME) is not None


def _tool_call(name):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments="{}"), id=f"call-{name}")


def test_a_reply_only_turn_runs_no_tool_but_the_declaration():
    """The recovery turn replies; it cannot repeat the work. Every other call is refused before anything runs,
    whatever the tool and whichever loop would have run it."""
    from agent.tool_executor import peer_batch_refused

    agent = SimpleNamespace(_turn_principal=None)
    assert not is_reply_only_turn()
    assert peer_batch_refused(agent, [_tool_call("terminal")]) is False  # an ordinary owner turn: untouched
    token = set_reply_only(True)
    try:
        assert is_reply_only_turn()
        assert peer_batch_refused(agent, [_tool_call("terminal")]) is True
        assert peer_batch_refused(agent, [_tool_call("memory"), _tool_call(TOOL_NAME)]) is True
        assert peer_batch_refused(agent, [_tool_call(TOOL_NAME)]) is False
    finally:
        reset_reply_only(token)
    assert not is_reply_only_turn()


# ---- the peer turn of a post a bound CTO declared owed -------------------------------------------------

def _peer(resolve_event):
    """A bound canonical peer turn: its agent, and the grant its verified ingress issued (``resolve_event``: the
    event of an owed acp-kind the bound CTO declared, the only event the declaration may name; else None)."""
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant

    agent = SimpleNamespace(_turn_principal="peer")
    grant = PeerToolGrant("acp-canonical-ceo", agent, lambda: None, resolve_event=resolve_event)
    return agent, grant, bind_grant(grant)


def _resolve_call(event_id, call_id="call-r"):
    args = {"event_id": event_id, "outcome": "answered", "evidence": [{"answer": "머지하세요"}]}
    return SimpleNamespace(function=SimpleNamespace(name=TOOL_NAME, arguments=json.dumps(args)), id=call_id), args


def test_a_peer_turn_may_declare_only_the_event_of_the_owed_kind_its_grant_names():
    """Review: ACK-closing questions is rejected. The peer policy lets ``relay_resolve`` through for a peer turn
    whose grant names an event (the bound CTO declared it owed) and for that event alone; with no event on the
    grant (an untagged post, a report) or for any other event, or beside any other tool, the batch is refused."""
    from agent.canonical_peer_tools import reset_grant
    from agent.tool_executor import peer_batch_refused

    for resolve_event, event_id, refused in (("event-1", "event-1", False), ("event-1", "event-2", True),
                                             (None, "event-1", True)):
        agent, grant, token = _peer(resolve_event)
        try:
            call, _ = _resolve_call(event_id)
            assert peer_batch_refused(agent, [call]) is refused, (resolve_event, event_id)
        finally:
            reset_grant(token)
    agent, grant, token = _peer("event-1")
    try:
        call, _ = _resolve_call("event-1")
        assert peer_batch_refused(agent, [call, _tool_call("terminal")]) is True  # beside any other tool
        assert peer_batch_refused(agent, [_tool_call("terminal")]) is True  # no other tool is opened by it
    finally:
        reset_grant(token)


def test_the_peer_turns_declaration_runs_only_as_checked_and_only_in_its_scope():
    from agent.canonical_peer_tools import dispatch_handler, reset_grant
    from agent.tool_executor import peer_batch_refused

    agent, grant, token = _peer("event-1")
    try:
        call, args = _resolve_call("event-1")
        assert peer_batch_refused(agent, [call]) is False
        handler = dispatch_handler(TOOL_NAME, args, "")
        assert callable(handler)
        scope = bind_scope("event-1")
        try:
            assert json.loads(handler(args))["declared"] is True
            assert "error" in json.loads(handler({**args, "evidence": [{"answer": "changed after the check"}]}))
        finally:
            reset_scope(scope)
        assert "error" in json.loads(handler(args))  # no relayed owner or owed-kind scope bound: nothing declared
    finally:
        reset_grant(token)
    assert "error" in json.loads(handler(args))  # outside the turn's grant
