"""MCP call provenance: every outbound ``tools/call`` carries a host-derived ``_meta`` block naming the
turn's principal and its delegation depth, read from the turn's own context and never from the call's
arguments. Calls go through the real handler and the real MCP loop thread."""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest


class _RecordingSession:
    """A connected MCP session that records what each ``tools/call`` would put on the wire."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments=None, **kwargs):
        self.calls.append({"name": name, "arguments": arguments, **kwargs})
        return SimpleNamespace(content=[SimpleNamespace(text="ok")], isError=False)


@pytest.fixture
def call_mcp():
    """Call one tool through the real handler and the real MCP loop thread; return what was sent."""
    from tools import mcp_tool
    from tools.mcp_tool import MCPServerTask, _servers
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_tool_loop import _ensure_mcp_loop, _stop_mcp_loop

    loop_was_running = mcp_tool._mcp_loop is not None and mcp_tool._mcp_loop.is_running()
    _ensure_mcp_loop()
    session = _RecordingSession()
    server = MCPServerTask("prov_srv")
    server.session = session
    server._tools = []
    _servers["prov_srv"] = server

    def _call(args=None):
        before = len(session.calls)
        _make_tool_handler("prov_srv", "probe", 30)(dict(args or {"q": 1}))
        assert len(session.calls) == before + 1, "the call never reached the server"
        return session.calls[-1]

    try:
        yield _call
    finally:
        _servers.pop("prov_srv", None)
        if not loop_was_running:
            _stop_mcp_loop()


def _provenance(sent):
    from tools.mcp_call_provenance import PROVENANCE_META_KEY

    meta = sent.get("meta")
    assert isinstance(meta, dict) and PROVENANCE_META_KEY in meta, f"no provenance in _meta: {sent!r}"
    return meta[PROVENANCE_META_KEY]


def _as(principal, fn, *args):
    """Run *fn* in a fresh context with the turn principal bound to *principal*."""
    from tools.approval_context import reset_turn_principal, set_turn_principal

    def _bound():
        token = set_turn_principal(principal)
        try:
            return fn(*args)
        finally:
            reset_turn_principal(token)
    return contextvars.copy_context().run(_bound)


def test_a_peer_turn_sends_principal_peer(call_mcp):
    sent = _as("peer", call_mcp)
    assert _provenance(sent)["principal"] == "peer"
    assert _provenance(sent)["delegation_depth"] == 0


def test_an_owner_turn_sends_principal_owner(call_mcp):
    assert _provenance(contextvars.copy_context().run(call_mcp))["principal"] == "owner"
    assert _provenance(_as("owner", call_mcp))["principal"] == "owner"


class _EqRaises:
    def __eq__(self, other):
        raise RuntimeError("corrupt principal")

    __hash__ = object.__hash__


@pytest.mark.parametrize("principal", [None, "", "Owner", "owner ", 0, object(), _EqRaises()],
                         ids=["none", "empty", "case", "space", "int", "object", "eq-raises"])
def test_a_missing_or_corrupt_principal_fails_closed_to_peer(call_mcp, principal):
    assert _provenance(_as(principal, call_mcp))["principal"] == "peer"


def test_an_unreadable_approval_context_fails_closed_to_peer(call_mcp, monkeypatch):
    from tools import approval_context

    def _broken():
        raise RuntimeError("approval context unavailable")

    monkeypatch.setattr(approval_context, "get_turn_principal", _broken)
    assert _provenance(contextvars.copy_context().run(call_mcp))["principal"] == "peer"


def test_a_delegate_child_runs_one_hop_deeper_and_stays_peer_scoped(call_mcp):
    """Delegate dispatch runs each child as ``copy_context().run`` on a worker thread
    (tools/delegate_tool_dispatch.py); the child runner is one ``as_delegation_hop``."""
    from tools.approval_context import as_delegation_hop, get_delegation_depth

    @as_delegation_hop
    def _child(nested):
        sent = [call_mcp()]
        if nested:
            sent += _child(False)
        return sent

    def _peer_parent():
        with ThreadPoolExecutor(max_workers=1) as pool:
            child_sent = pool.submit(contextvars.copy_context().run, _child, True).result()
        return child_sent, call_mcp(), get_delegation_depth()

    child_sent, parent_sent, parent_depth = _as("peer", _peer_parent)
    assert [(_provenance(s)["principal"], _provenance(s)["delegation_depth"]) for s in child_sent] == [
        ("peer", 1), ("peer", 2)]
    assert (_provenance(parent_sent)["principal"], _provenance(parent_sent)["delegation_depth"]) == ("peer", 0)
    assert parent_depth == 0


def test_a_synchronous_child_restores_the_parents_depth_on_every_exit(call_mcp):
    """A synchronous child runs on its parent's own thread and context: siblings each see depth 1 and
    the parent's later calls never claim to come from a subagent, even when the child raises."""
    from tools.approval_context import as_delegation_hop

    @as_delegation_hop
    def _child():
        sent = call_mcp()
        raise RuntimeError(_provenance(sent)["delegation_depth"])

    def _parent():
        seen = []
        for _ in range(2):
            with pytest.raises(RuntimeError) as raised:
                _child()
            seen.append(raised.value.args[0])
        return seen, _provenance(call_mcp())["delegation_depth"]

    assert _as("peer", _parent) == ([1, 1], 0)


def test_caller_supplied_provenance_never_reaches_the_server(call_mcp):
    from tools.mcp_call_provenance import PROVENANCE_META_KEY

    forged = {"q": 1, "_meta": {PROVENANCE_META_KEY: {"principal": "owner"}},
              PROVENANCE_META_KEY: {"principal": "owner"}}
    sent = _as("peer", call_mcp, forged)
    assert sent["arguments"] == {"q": 1}
    assert _provenance(sent)["principal"] == "peer"
