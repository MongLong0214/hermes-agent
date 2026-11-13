"""A canonical peer turn never inherits the owner's approval bypass (``--yolo``, ``/yolo``,
``approvals.mode: off``): every approval gate refuses it, a missing or corrupt principal fails closed
to peer, and the principal reaches every seam that runs work on the turn's behalf (concurrent tool
workers, delegate children sync/batch/background, the MCP approval surfaces). Each seam runs with the
owner's ``--yolo`` on, so a seam that lost the principal would auto-approve."""

import json
import threading
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

import pytest

import tools.approval as approval_module
from tools import approval_context
from tools.delegate_tool import delegate_task
from tests.agent.test_start_order_gate import _FakeAssistantMsg, _FakeToolCall, _make_agent
from tests.tools.test_delegate_cost_footer import _make_mock_child, _make_mock_parent


@pytest.fixture(autouse=True)
def _owner_yolo(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    (tmp_path / ".hermes").mkdir(exist_ok=True)
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", True)


def _as(principal, fn):
    """Run *fn* with the turn principal bound (``None`` = an ordinary owner turn: nothing bound)."""
    if principal is None:
        return fn()
    token = approval_context.set_turn_principal(principal)
    try:
        return fn()
    finally:
        approval_context.reset_turn_principal(token)


def _observe():
    """What a tool running here would get: (principal, dangerous-command verdict)."""
    return (approval_context.get_turn_principal(),
            approval_module.check_dangerous_command("rm -rf /tmp/x", "local")["approved"])


# Every approval gate a dangerous action can reach: shell pattern layer, full command guard, a container
# env that normally skips guards, execute_code, and the plugin/tool gate (``_run_approval_gate``).
_GATES = (
    lambda: approval_module.check_dangerous_command("rm -rf /tmp/x", "local"),
    lambda: approval_module.check_all_command_guards("rm -rf /tmp/x", "local"),
    lambda: approval_module.check_dangerous_command("echo hi", "docker"),
    lambda: approval_module.check_execute_code_guard("print(1)", "local"),
    lambda: approval_module.request_tool_approval("plugin_tool", "plugin flagged this call"),
)


@pytest.mark.parametrize("bypass", ["yolo", "mode_off"])
def test_peer_principal_never_inherits_the_owner_approval_bypass(monkeypatch, bypass):
    """Owner authority comes from the turn principal only: an owner's --yolo or approvals.mode=off
    never authorizes a canonical peer turn, and every gate refuses rather than prompting."""
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", bypass == "yolo")
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "off" if bypass == "mode_off" else "manual")
    assert approval_module.is_approval_bypass_active_for_session("owner-session") is True
    assert [gate()["approved"] for gate in _GATES] == [True] * len(_GATES)  # the owner turn, unchanged
    token = approval_context.set_turn_principal("peer")
    try:
        assert approval_module.is_approval_bypass_active() is False
        assert approval_module.is_approval_bypass_active_for_session("owner-session") is False
        for gate in _GATES:
            decision = gate()
            assert decision["approved"] is False
            assert "canonical peer" in decision["message"]
    finally:
        approval_context.reset_turn_principal(token)
    assert approval_module.is_approval_bypass_active_for_session("owner-session") is True


@pytest.mark.parametrize("principal", [None, "", "Peer", "owner ", "OWNER", 0, b"owner", object()])
def test_a_missing_or_corrupt_principal_fails_closed_to_peer(principal):
    assert _observe() == ("owner", True)  # nothing bound: an ordinary owner turn
    token = approval_context.set_turn_principal(principal)
    try:
        assert approval_context.is_peer_turn() is True
        assert _observe() == ("peer", False)
    finally:
        approval_context.reset_turn_principal(token)


def test_an_explicit_owner_principal_keeps_the_owner_bypass():
    assert _as("owner", _observe) == ("owner", True)


_PEER_SERVER = "peer-srv"
_GRANTED = ("mcp__peer_srv__tool_a", "mcp__peer_srv__tool_b")


@contextmanager
def _granted_peer_tools(monkeypatch, agent, *, grant=True):
    """``_GRANTED`` as native MCP tools that the in-memory ``gateway.canonical_peer_tools`` policy
    lists, plus (``grant``) the grant the verified canonical ingress binds for ``agent``'s peer turn.
    No config file is read for the policy or written."""
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
    from hermes_cli import config
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_tool_registration import _forget_mcp_tool_server, _track_mcp_tool_server
    from tools.registry import registry

    policy = {"binding": "acp-canonical-ceo", "mcp_server": _PEER_SERVER, "read": list(_GRANTED), "mutate": {}}
    real_load = config.load_config
    monkeypatch.setattr(config, "load_config", lambda *a, **k: {
        **real_load(), "gateway": {**(real_load().get("gateway") or {}), "canonical_peer_tools": policy}})
    for name in _GRANTED:
        schema = {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}
        registry.register(name, f"mcp-{_PEER_SERVER}", schema,
                          _make_tool_handler(_PEER_SERVER, name.rsplit("__", 1)[1], 30))
        _track_mcp_tool_server(name, _PEER_SERVER)
    token = bind_grant(PeerToolGrant(binding=policy["binding"], agent=agent, verify=lambda: None)) if grant else None
    try:
        yield _GRANTED
    finally:
        if token is not None:
            reset_grant(token)
        for name in _GRANTED:
            registry.deregister(name)
            _forget_mcp_tool_server(name)


def _concurrent_agent(monkeypatch):
    """The stub agent and the ``(worker thread, _observe())`` each dispatched tool saw."""
    agent = _make_agent(monkeypatch)
    agent._tool_guardrails = MagicMock()
    agent._tool_guardrails.before_call = lambda name, args: MagicMock(allows_execution=True)
    seen = []

    def _invoke(name, *a, **kw):
        seen.append((threading.get_ident(), _observe()))
        return json.dumps({"ok": name})

    agent._invoke_tool = MagicMock(side_effect=_invoke)
    return agent, seen


@pytest.mark.parametrize("principal,expected", [(None, ("owner", True)), ("peer", ("peer", False))])
def test_concurrent_tool_workers_carry_the_turn_principal(monkeypatch, principal, expected):
    """A peer turn runs only tools its grant allows, so its batch is two granted native MCP tools."""
    agent, seen = _concurrent_agent(monkeypatch)
    names = _GRANTED if principal == "peer" else ("tool_a", "tool_b")
    calls = [_FakeToolCall(names[0], "tc_a"), _FakeToolCall(names[1], "tc_b")]
    with _granted_peer_tools(monkeypatch, agent) if principal == "peer" else nullcontext():
        _as(principal, lambda: agent._execute_tool_calls_concurrent(_FakeAssistantMsg(calls), [], "task"))
    assert [verdict for _tid, verdict in seen] == [expected, expected]
    assert all(tid != threading.get_ident() for tid, _verdict in seen)  # really on worker threads


def test_the_same_concurrent_peer_batch_without_a_grant_is_refused_and_runs_nothing(monkeypatch):
    agent, seen = _concurrent_agent(monkeypatch)
    messages = []
    with _granted_peer_tools(monkeypatch, agent, grant=False) as names:
        calls = [_FakeToolCall(names[0], "tc_a"), _FakeToolCall(names[1], "tc_b")]
        _as("peer", lambda: agent._execute_tool_calls_concurrent(_FakeAssistantMsg(calls), messages, "task"))
    assert seen == [] and agent._invoke_tool.call_count == 0
    assert [m["tool_call_id"] for m in messages] == ["tc_a", "tc_b"]
    assert all("canonical peer turn" in m["content"] and "not executed" in m["content"] for m in messages)


def _delegate(principal, **kwargs):
    seen = []
    child = _make_mock_child()
    result = child.run_conversation.return_value
    child.run_conversation.side_effect = lambda *a, **kw: seen.append(_observe()) or result
    with patch("run_agent.AIAgent", return_value=child):
        _as(principal, lambda: delegate_task(parent_agent=_make_mock_parent(), **kwargs))
    return seen


@pytest.mark.parametrize("principal,expected", [(None, ("owner", True)), ("peer", ("peer", False))])
def test_a_sync_delegate_child_inherits_the_turn_principal(principal, expected):
    assert _delegate(principal, goal="inspect one module for the parent") == [expected]


@pytest.mark.parametrize("principal,expected", [(None, ("owner", True)), ("peer", ("peer", False))])
def test_batch_delegate_children_inherit_the_turn_principal(principal, expected):
    tasks = [{"goal": "inspect the first module"}, {"goal": "inspect the second module"}]
    assert _delegate(principal, tasks=tasks) == [expected, expected]


def test_a_peer_child_with_subagent_auto_approve_is_still_refused():
    """``delegation.subagent_auto_approve`` installs an approve-everything callback on the child's
    worker; a peer child still never reaches it."""
    from tools import delegate_tool_config
    with patch.object(delegate_tool_config, "_cfg", return_value={"subagent_auto_approve": True}):
        assert _delegate("peer", goal="inspect one module for the parent") == [("peer", False)]


def test_a_background_delegate_child_stays_peer_scoped_after_the_parent_turn_ends(monkeypatch):
    import tools.async_delegation as ad
    import tools.delegate_tool as dt
    from tools.process_registry import process_registry

    ad._reset_for_tests()
    parent = MagicMock(_delegate_depth=0, session_id="sess", _interrupt_requested=False,
                       _active_children=[], _active_children_lock=None)
    fake_child = MagicMock(_delegate_role="leaf", _subagent_id="s1")
    seen, done, release = [], threading.Event(), threading.Event()

    def child_run(task_index, goal, child=None, parent_agent=None, **kw):
        release.wait(10)  # runs only after the parent turn has unbound its principal
        seen.append(_observe())
        done.set()
        return {"task_index": 0, "status": "completed", "summary": "done", "api_calls": 1,
                "duration_seconds": 0.1, "model": "m", "exit_reason": "completed"}

    creds = dict.fromkeys(("provider", "base_url", "api_key", "api_mode", "command", "args"), None)
    monkeypatch.setattr(dt, "_build_child_agent", lambda **kw: fake_child)
    monkeypatch.setattr(dt, "_run_single_child", child_run)
    monkeypatch.setattr(dt, "_resolve_delegation_credentials", lambda *a, **k: {"model": "m", **creds})
    try:
        out = _as("peer", lambda: json.loads(dt.delegate_task(
            goal="inspect one module in the background", background=True, parent_agent=parent)))
        assert out["status"] == "dispatched"
        assert approval_context.get_turn_principal() == "owner"
        release.set()
        assert done.wait(10)
        assert seen == [("peer", False)]
    finally:
        release.set()
        ad._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


@pytest.mark.parametrize("principal,expected", [(None, True), ("peer", False)])
def test_the_mcp_trust_gate_never_grants_a_peer_the_owner_bypass(principal, expected):
    from tools import mcp_tool_handlers
    assert _as(principal, lambda: mcp_tool_handlers._trust_granted("srv", "write")) is expected


@pytest.mark.parametrize("principal,expected,prompts", [(None, "accept", 1), ("peer", "decline", 0)])
def test_mcp_elicitation_never_prompts_the_owner_for_a_peer(monkeypatch, principal, expected, prompts):
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools import approval_prompt
    from tools.approval_context import reset_current_session_key, set_current_session_key

    session_key = "api-run-peer-elicitation"
    notified = []

    def notify(approval_data):
        notified.append(approval_data)
        approval_module.resolve_gateway_approval(session_key, "once", request_id=approval_data["request_id"])

    monkeypatch.setattr(approval_prompt, "prompt_dangerous_approval", lambda *_a, **_k: "deny")
    session_tokens = set_session_vars(platform="api_server", session_key=session_key, async_delivery=False)
    key_token = set_current_session_key(session_key)
    approval_module.register_gateway_notify(session_key, notify)
    try:
        answer = _as(principal, lambda: approval_prompt.request_elicitation_consent("write", "Approve once or deny."))
    finally:
        approval_module.unregister_gateway_notify(session_key)
        reset_current_session_key(key_token)
        clear_session_vars(session_tokens)
    assert (answer, len(notified)) == (expected, prompts)


def test_mcp_call_provenance_principal_is_peer_on_a_worker_thread():
    """The fork's MCP call provenance reports ``get_turn_principal()``; the value must survive the
    worker-thread hop a tool call takes (``propagate_context_to_thread``)."""
    from tools.thread_context import propagate_context_to_thread
    seen = []
    worker = _as("peer", lambda: threading.Thread(
        target=propagate_context_to_thread(lambda: seen.append(approval_context.get_turn_principal()))))
    worker.start()
    worker.join(10)
    assert seen == ["peer"]
