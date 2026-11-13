"""Pinned handler dispatch: a handler resolved earlier (a canonical peer batch's grant) is the one
that runs, whatever the registry holds under its name by the time it dispatches; wrappers expose
the function they wrap; an MCP tool's origin is bound to the handler registered now; and a call
that never reached its server is no circuit-breaker strike."""

from __future__ import annotations

import inspect
import json

import pytest

from tools.registry import ToolRegistry, tool_error
from tools.thread_context import propagate_context_to_thread


def _schema(name):
    return {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}


def _handler(label):
    def handler(args, **kwargs):
        return json.dumps({"ran": label})
    return handler


# ── dispatch(handler=) ─────────────────────────────────────────────────────


def test_a_pinned_handler_runs_even_after_the_name_is_re_registered():
    reg = ToolRegistry()
    reg.register(name="tool", toolset="core", schema=_schema("tool"), handler=_handler("h0"))
    pinned = reg.get_entry("tool").handler
    reg.deregister("tool")
    reg.register(name="tool", toolset="core", schema=_schema("tool"), handler=_handler("h1"))
    assert json.loads(reg.dispatch("tool", {}, handler=pinned)) == {"ran": "h0"}
    assert json.loads(reg.dispatch("tool", {})) == {"ran": "h1"}


def test_a_pinned_handler_runs_after_its_name_is_gone_and_unpinned_dispatch_still_refuses():
    reg = ToolRegistry()
    assert json.loads(reg.dispatch("gone", {}, handler=_handler("h0"))) == {"ran": "h0"}
    assert "Unknown tool: gone" in reg.dispatch("gone", {})


def test_a_pinned_handler_gets_the_same_kwargs_narrowing_and_error_contract():
    reg = ToolRegistry()
    seen = []

    def narrow(args, task_id=None):
        seen.append((args, task_id))
        return "ok"

    def boom(args, **kwargs):
        raise RuntimeError("kaput")

    assert reg.dispatch("tool", {"a": 1}, handler=narrow, task_id="t1", session_id="s1") == "ok"
    assert seen == [({"a": 1}, "t1")]
    assert "RuntimeError: kaput" in json.loads(reg.dispatch("tool", {}, handler=boom))["error"]
    assert "unsupported result type" in reg.dispatch("tool", {}, handler=lambda args: 7)


# ── wrappers expose what they wrap ────────────────────────────────────────


def test_the_thread_context_wrapper_unwraps_to_the_real_function():
    def target(args, task_id=None):
        return f"ran {task_id}"

    wrapped = propagate_context_to_thread(target)
    assert inspect.unwrap(wrapped) is target
    assert inspect.unwrap(propagate_context_to_thread(wrapped)) is target


def test_a_thread_context_wrapped_handler_keeps_its_own_signature_at_dispatch():
    reg = ToolRegistry()

    def narrow(args):
        return "narrow ran"

    reg.register(name="tool", toolset="core", schema=_schema("tool"), handler=propagate_context_to_thread(narrow))
    assert reg.dispatch("tool", {}, task_id="t1") == "narrow ran"
    assert reg.dispatch("tool", {}, handler=propagate_context_to_thread(narrow), task_id="t1") == "narrow ran"


# ── MCP origin and breaker ───────────────────────────────────────────────


@pytest.fixture()
def mcp_registry(monkeypatch):
    from tools import mcp_tool

    reg = ToolRegistry()
    monkeypatch.setattr("tools.registry.registry", reg)
    monkeypatch.setattr(mcp_tool, "_mcp_tool_server_names", {})
    monkeypatch.setattr(mcp_tool, "_server_trust_levels", {})
    return reg, mcp_tool._mcp_tool_server_names


def _register_native(reg, names, server, tool):
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_tool_schema import mcp_prefixed_tool_name

    name = mcp_prefixed_tool_name(server, tool)
    handler = _make_tool_handler(server, tool, 120, native_images=True)
    reg.deregister(name)
    reg.register(name=name, toolset=f"mcp-{server}", schema=_schema(name), handler=handler)
    names[name] = server
    return name, handler


def test_mcp_tool_origin_follows_the_handler_registered_now(mcp_registry):
    from tools.mcp_tool_handlers import mcp_tool_origin

    reg, names = mcp_registry
    name, first = _register_native(reg, names, "srv", "greet")
    assert mcp_tool_origin(name) == ("srv", False, first)
    _name, second = _register_native(reg, names, "srv", "greet")  # a reconnect rebuilds the handler
    assert second is not first
    assert mcp_tool_origin(name) == ("srv", False, second)


def test_mcp_tool_origin_reports_whether_the_trust_gate_would_ask(mcp_registry, monkeypatch):
    from tools import mcp_tool
    from tools.mcp_tool_handlers import mcp_tool_origin
    from tools.mcp_tool_scope import _server_key

    reg, names = mcp_registry
    name, handler = _register_native(reg, names, "srv", "greet")
    monkeypatch.setattr(mcp_tool, "_server_trust_levels", {_server_key("srv"): mcp_tool._TRUST_UNTRUSTED})
    assert mcp_tool_origin(name) == ("srv", True, handler)


def test_mcp_tool_origin_refuses_a_handler_that_is_not_this_servers_native_tool(mcp_registry):
    from tools.mcp_tool_handlers import mcp_tool_origin

    reg, names = mcp_registry
    name = _register_native(reg, names, "srv", "greet")[0]
    names[name] = "other"  # the name now belongs to another server
    assert mcp_tool_origin(name) is None
    names[name] = "srv"
    reg.deregister(name)
    reg.register(name=name, toolset="mcp-srv", schema=_schema(name), handler=_handler("utility"))
    assert mcp_tool_origin(name) is None
    assert mcp_tool_origin("mcp__srv__unknown") is None


def test_a_call_that_was_never_sent_is_no_breaker_strike(monkeypatch):
    from tools import mcp_tool
    from tools.mcp_tool_handlers import _NotSent, _record_call_outcome

    calls = []
    monkeypatch.setattr(mcp_tool, "_bump_server_error", lambda name, **kw: calls.append(("bump", name, kw)))
    monkeypatch.setattr(mcp_tool, "_reset_server_error", lambda name: calls.append(("reset", name)))
    refusal = tool_error("not sent")
    result = _record_call_outcome("srv", _NotSent(refusal))
    assert result == refusal and type(result) is str and calls == []
    assert _record_call_outcome("srv", refusal) == refusal
    assert calls == [("bump", "srv", {"application": True})]
