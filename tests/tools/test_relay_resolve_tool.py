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


GOOD = {"event_id": "event-1", "outcome": "done", "evidence": [{"tool_call_id": "call-1"}],
        "public_reply": "The project is stopped."}


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
    {"evidence": []}, {"evidence": "see above"}, {"evidence": [{}]},   # (done needs a tool result to cite)
    {"evidence": [{"note": "x"}]}, {"evidence": [{"tool_call_id": ""}]}, {"evidence": [{"answer": "   "}]},
    {"evidence": [{"answer": "Do not merge; it loses messages."}]},    # evidence prose is no answer
    {"evidence": [{"tool_call_id": "c", "answer": "both"}]}, {"evidence": ["call-1"]},
    {"evidence": [{"tool_call_id": 7}]},
    {"public_reply": None}, {"public_reply": ""}, {"public_reply": "   "}, {"public_reply": 7},
    {"public_reply": "x" * 2_001},
    {"public_reply": "See /Users/someone/.hermes/export.json"}, {"public_reply": "call_a1b2c3d4e5f6 passed"},
])
def test_a_declaration_without_a_known_outcome_and_well_formed_evidence_is_refused(scope, change):
    refused = _call({**GOOD, **change})
    assert "error" in refused and "declared" not in refused


def test_evidence_is_a_closed_list_of_tool_results_and_explicit_answers():
    assert evidence_items([{"tool_call_id": " call-1 "}, {"tool_call_id": "call-2"}]) == [
        ("tool", "call-1"), ("tool", "call-2")]
    assert evidence_items([]) == [] and evidence_items(None) == []  # optional: only ``done`` needs a call to cite
    assert evidence_items([{"tool_call_id": "a"}] * 65) is None
    assert evidence_items([{"answer": "prose"}]) is None  # an answer is the public_reply, never evidence


@pytest.mark.parametrize("outcome", ["answered", "declined", "deferred_with_reason"])
def test_an_answer_declined_or_deferred_needs_only_its_public_reply(scope, outcome):
    ok = _call({"event_id": "event-1", "outcome": outcome, "public_reply": "Do not merge; it loses messages."})
    assert ok["declared"] is True
    assert "error" in _call({"event_id": "event-1", "outcome": "done", "public_reply": "All done."})  # done cites work


def test_the_tool_has_a_required_public_reply_and_says_what_is_published():
    from tools.relay_resolve_tool import SCHEMA

    assert "public_reply" in SCHEMA["parameters"]["required"]
    assert SCHEMA["parameters"]["properties"]["public_reply"]["type"] == "string"
    assert "<outcome>: <public_reply>" in SCHEMA["description"] and "internal" in SCHEMA["description"]


@pytest.mark.parametrize("text", [
    "Saved to /Users/someone/.hermes/private/customer-export.json", "Saved to /tmp/x/y", "at /etc/hosts",
    "Saved to ~/private/out.txt", "Saved to C:\\Users\\someone\\out.txt", "(/Users/someone/a)",
    # the review's forms: a drive with a forward slash, a UNC share, a single-segment root, a file: URI
    "Saved at C:/Users/someone/private/export.json", "Saved at d:/x", "Saved at \\\\server\\private\\customer-export.json",
    "Saved at \\\\host\\share", "Saved at \\\\", "Saved at /private-report.json", "see /opt/x", "see /etc/hosts",
    "Saved at /export.json", "Saved at file:///Users/someone/private/export.json", "see file:/tmp/x",
    "see FILE:///etc/hosts", "see file:relative.txt", "the file: it is ready",
    # one strict rule: every slash outside an http(s) URL is refused, and so is every backslash: no /word exception
    "Saved to /root", "Saved to /srv", "Look in /etc", "see /tmp.", "see /Users", "/stop", "Use /stop.", "/status",
    "Run /status and /help", "/help me", "/STOP!", "Use /stop, then /status.", "Use /stop (or /help).", "Try /stop?",
    "/중지 로 취소", "/abcdefghijklmnop", "Open HTTPS://Example.com/A/b. Then /stop", "see (https://a.b/c) and /stop;",
    "Users\\someone\\x", "a\\b", "see https://a.b/x\\y", "see https://a.b/c\\",
    # a URL is exactly its own characters: closing markup and quotes end it, and whatever follows is judged
    "See <https://example.com>/Users/someone/private/export.json",
    "See [report](https://example.com)/Users/someone/private/export.json",
    "See (https://example.com)/Users/someone/private/export.json", 'See "https://example.com"/Users/someone/x',
    "See 'https://example.com'/Users/someone/x", "See `https://example.com`/Users/someone/x",
    "See [x](https://example.com/a)/Users/someone/x", "See <https://example.com/a/b>/etc",
    'See https://example.com"file:private.txt', "See <https://example.com>file:private.txt",
    "See (https://example.com)file:private.txt", "See https://example.com'file:private.txt",
    "See https://example.com/a>/Users/x", "See https://example.com/a)/Users/x", "See https://example.com/a]/x",
    "see https://user:pw@example.com/x", "see https://[::1]/x", "see https://exa mple.com/x /y",
    "see https://example.com:port/x", "see https:///x", "see https://", "see https://_x/y",
    "Output:/Users/someone/private/export.json", "Output:/x", "Saved to /Customer Exports/private.json",
    "Saved to /customer_export", "Users/someone/private/export.json", "see ./run.sh", "see ../x", "see a/b/c.txt",
    "Handled and/or deferred", "Handled 1/2.", "Version 2.1/2.2 shipped", "TCP/IP", "a/b",
    "see (/stop)", "see ('/ok')", "see x/stop", "/stop/", "/stop/x", "/stop.json", "/stop-all", "/stop_all", "/stop2",
    "/abcdefghijklmnopq", "see /stop:/x", "see /Users/someone/x?u=https://a.b", "Saved to https:///Users/someone/x",
    "see http://a.b/c and /Users/someone/x", "see ahttps://a.b/c", "see https://a.b/c /Users/someone/x",
    "see call_a1b2c3d4e5f6", "see call_ABCdef0123456789abcdefgh", "see fc_0123456789abcdef01234567",
    "see toolu_0123456789abcdef", "see acp_call_3",
    "session 20261117_064935_ab12cd", "session 20261117_064935_ab12cd34",
    "session 7b622a35-9409-4df4-bfe7-1e140ed238ba",
    "receipt canonical-receipt:v2:" + "ab" * 32, "claim canonical-claim:v1:" + "ab" * 32,
    "kept canonical-answer-kept:v1:" + "ab" * 32, "ledger owner-relay-record:v1:session:record",
    "owner " + "cd" * 16, "event " + "ef" * 32, "digest sha256:" + "ab" * 32,
])
def test_public_reply_problem_refuses_a_path_or_an_id_this_codebase_issues(text):
    from tools.relay_resolve_tool import public_reply_problem

    assert public_reply_problem(text) is not None


@pytest.mark.parametrize("text", [
    "The project is stopped; see https://example.com/runs/42 for the report.",
    "call me back after the freeze", "call_back is a word",
    "프로젝트를 중지했고 보고서는 스레드에 있습니다.", "Fixed in 3 of 20260101 files", "ticket ab-12 closed",
    "http://example.com/a/b is up", "meet at 5:30", "profile c: drive is unused",
    "a path-like word like private-report is fine", "Say stop to cancel.",
    "Open HTTPS://Example.com/A/b. Then stop", "see (https://example.com/a/b) and https://example.com/c;",
    "https://example.com", "https://example.com/runs/42?x=1&y=2#top, then done",
    # plain URLs in markup or quotes still pass
    "See <https://example.com/runs/42>.", "See <https://example.com>", "See (https://example.com/a/b).",
    "See [the report](https://example.com/runs/42?x=1#top)", 'See "https://example.com/a/b"',
    "See 'https://example.com/a/b'", "See `https://example.com/a/b`", "See https://example.com:8080/a/b?c=d&e=f#g",
    "See <https://example.com/a>, <https://example.org/b>; and [c](http://example.net/c).",
    "See https://sub.example-site.co.kr/%ED%95%9C/a_b~c@d:e!$&*+,;=?q#f",
])
def test_public_reply_problem_lets_prose_through(text):
    from tools.relay_resolve_tool import public_reply_problem

    assert public_reply_problem(text) is None


def test_public_reply_problem_refuses_the_ids_of_the_turn_it_is_given():
    from tools.relay_resolve_tool import public_reply_problem

    assert public_reply_problem("Result of call-1 was ok", known_ids=["call-1"]) is not None
    assert public_reply_problem("Result of call-10 was ok", known_ids=["call-1"]) is None  # a whole id, not a prefix
    assert public_reply_problem("It is done", known_ids=["call-1", "xy"]) is None


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
    args = {"event_id": event_id, "outcome": "answered", "evidence": [],
            "public_reply": "머지하세요."}
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
            assert "error" in json.loads(handler({**args, "public_reply": "changed after the check"}))
        finally:
            reset_scope(scope)
        assert "error" in json.loads(handler(args))  # no relayed owner or owed-kind scope bound: nothing declared
    finally:
        reset_grant(token)
    assert "error" in json.loads(handler(args))  # outside the turn's grant
