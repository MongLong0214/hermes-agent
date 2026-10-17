"""A canonical peer turn under the CEO binding runs exactly the configured ACP CEO tools.

Every refusal below is checked next to a call the same turn DOES run, so a green refusal never
just means "the peer path still runs nothing". Refused means zero executions: no tool handler is
reached and every call in the batch gets the peer refusal result."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

SERVER = "acp-ceo"
BINDING = "acp-canonical-ceo"
RUN = "run_7717a3af19ca405090a4e879"
DISPATCHERS = ("execute_tool_calls_sequential", "execute_tool_calls_concurrent", "execute_tool_calls_segmented")


def ceo(tool: str) -> str:
    return f"mcp__acp_ceo__{tool}"


# Native MCP tools per server. ``owner_decision_submit`` is a real acp_ceo tool the policy never
# lists; ``ghost`` is listed but never registered; the other server's ``x`` is listed by mistake
# and must still be refused.
SERVER_TOOLS = {SERVER: ["run_get", "escalation_resolve", "ceo_decision_submit", "run_create", "owner_decision_submit"],
                "other": ["x"]}
REGISTERED = [ceo(t) for t in SERVER_TOOLS[SERVER]] + ["mcp__other__x"]
OWNER_META_KEYS = {"session_id", "session_key", "platform", "chat_id", "cron", "parent_chat_id", "principal",
                   "delegation_depth", "lineage_root_digest"}


def _policy(**changes) -> dict:
    policy = {
        "binding": BINDING, "mcp_server": SERVER,
        "read": [ceo("run_get"), ceo("ghost"), "mcp__other__x"],
        "mutate": {
            ceo("escalation_resolve"): {"runId": [RUN]},
            ceo("ceo_decision_submit"): {"runId": [RUN], "decision": ["FINAL_REVISE", "CONFIRM"]},
            ceo("run_create"): {"projectId": ["proj_fresh"], "kind": ["STANDARD_WORK"], "idempotencyKey": ["key-1"]},
        },
    }
    policy.update(changes)
    return policy


def _write_config(policy: dict) -> None:
    from hermes_cli.config import get_config_path

    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name("config.yaml.staged")
    staged.write_text(yaml.safe_dump({"gateway": {"canonical_peer_tools": policy}}), encoding="utf-8")
    os.replace(staged, path)


def digest(entry: dict) -> str:
    """The documented peer_grant format, computed independently of the code under test."""
    canonical = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _defs(*names: str) -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool",
                                              "parameters": {"type": "object", "properties": {}}}}
            for n in names]


class Sent(list):
    """``(tool, arguments)`` of every call that reached the MCP wire; ``metas`` its provenance."""

    def __init__(self):
        super().__init__()
        self.metas: list = []


@pytest.fixture
def ceo_tools(monkeypatch):
    """The tools as REAL MCP tool handlers (``_make_tool_handler``) over a fake transport."""
    import model_tools
    from tools.mcp_call_provenance import PROVENANCE_META_KEY
    from tools.mcp_tool import MCPServerTask
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_tool_registration import _forget_mcp_tool_server, _record_tool_trust_metadata, _track_mcp_tool_server
    from tools.registry import registry

    sent = Sent()

    def session(server):
        def call_tool(tool, arguments=None, meta=None):
            name = f"mcp__{server.replace('-', '_')}__{tool}"
            sent.append((name, dict(arguments)))
            sent.metas.append(meta[PROVENANCE_META_KEY])
            return SimpleNamespace(content=[SimpleNamespace(text=f"ok:{name}")], isError=False)
        return SimpleNamespace(call_tool=call_tool)

    for server, tools in SERVER_TOOLS.items():
        task = MCPServerTask(server)
        task.session, task._tools = session(server), []
        monkeypatch.setitem(_core._servers, server, task)
        _record_tool_trust_metadata(server, {}, [SimpleNamespace(name=t, annotations=None) for t in tools])
        for tool in tools:
            name = f"mcp__{server.replace('-', '_')}__{tool}"
            registry.register(name, f"mcp-{server}", _defs(name)[0]["function"], _make_tool_handler(server, tool, 30))
            _track_mcp_tool_server(name, server)
    monkeypatch.setattr("tools.mcp_tool_loop._run_on_mcp_loop",
                        lambda call, timeout=30: asyncio.run(call() if callable(call) else call))
    # The tool_call unwrap scopes the bridge to the session's own (pre-assembly) tool defs.
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda *a, **k: _defs(*REGISTERED))
    _write_config(_policy())
    try:
        yield sent
    finally:
        for name in REGISTERED:
            registry.deregister(name)
            _forget_mcp_tool_server(name)
        for server in SERVER_TOOLS:
            _core._server_trust_levels.pop(server, None)
            _core._tool_read_only_hints.pop(server, None)


@pytest.fixture
def agent(ceo_tools):
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions",
              return_value=_defs(*REGISTERED, "terminal", "write_file", "computer_use")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.load_config_readonly", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        built = AIAgent(api_key="test-key-1234567890", base_url="https://openrouter.ai/api/v1", quiet_mode=True,
                        skip_context_files=True, skip_memory=True, platform="telegram")
    built.client = MagicMock()
    built.compression_enabled = False
    built.save_trajectories = False
    return built


@contextmanager
def peer_turn(agent, *, binding=BINDING, verify=lambda: None, grant=True):
    """What the verified canonical ingress binds for one peer turn."""
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
    from tools.approval_context import reset_turn_principal, set_turn_principal

    principal = set_turn_principal("peer")
    agent._turn_principal = "peer"
    token = bind_grant(PeerToolGrant(binding=binding, agent=agent, verify=verify)) if grant else None
    try:
        yield
    finally:
        if token is not None:
            reset_grant(token)
        agent._turn_principal = None
        reset_turn_principal(principal)


def call(name: str, args: dict | None = None, call_id: str = "c1"):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


def run(agent, calls, dispatcher="execute_tool_calls_sequential") -> list:
    import agent.tool_executor as executor

    messages: list = []
    getattr(executor, dispatcher)(agent, SimpleNamespace(tool_calls=list(calls)), messages, "task")
    return messages


def real_result(message, tool) -> bool:
    # MCP results reach the model fenced as untrusted content; the server's own text is inside.
    return f"ok:{tool}" in message["content"] and "not executed" not in message["content"]


def refused(messages, calls) -> bool:
    return (len(messages) == len(calls)
            and all("canonical peer turn" in m["content"] and "not executed" in m["content"] for m in messages))


@pytest.mark.parametrize("dispatcher", DISPATCHERS)
def test_ceo_peer_turn_runs_a_listed_read_tool_and_returns_its_real_result(agent, ceo_tools, dispatcher):
    with peer_turn(agent):
        messages = run(agent, [call(ceo("run_get"), {"runId": "anything"})], dispatcher)
    assert ceo_tools == [(ceo("run_get"), {"runId": "anything"})]
    assert real_result(messages[0], ceo("run_get"))


@pytest.mark.parametrize("name", ["terminal", "write_file", "computer_use", "mcp__other__x",
                                  ceo("owner_decision_submit"), ceo("ghost")])
def test_unlisted_unregistered_and_foreign_tools_are_refused_next_to_a_listed_one(agent, ceo_tools, name):
    with peer_turn(agent):
        assert not refused(run(agent, [call(ceo("run_get"))]), [1])
        messages = run(agent, [call(name, {"command": "id", "runId": RUN})])
    assert refused(messages, [1]) and ceo_tools == [(ceo("run_get"), {})]


@pytest.mark.parametrize("dispatcher", DISPATCHERS)
def test_a_batch_with_any_disallowed_call_runs_nothing(agent, ceo_tools, dispatcher, monkeypatch):
    import model_tools

    dispatched = []
    real = model_tools.handle_function_call
    monkeypatch.setattr(model_tools, "handle_function_call",
                        lambda name, *a, **k: (dispatched.append(name), real(name, *a, **k))[1])
    batch = [call(ceo("run_get"), {}, "c1"), call("terminal", {"command": "id"}, "c2")]
    with peer_turn(agent):
        messages = run(agent, batch, dispatcher)
        assert refused(messages, batch) and dispatched == [] and ceo_tools == []
        run(agent, batch[:1], dispatcher)
    assert ceo_tools == [(ceo("run_get"), {})]


@pytest.mark.parametrize("tool, args", [
    (ceo("escalation_resolve"), {"runId": RUN, "resolution": "free text"}),
    (ceo("ceo_decision_submit"), {"runId": RUN, "decision": "CONFIRM"}),
    (ceo("ceo_decision_submit"), {"runId": RUN, "decision": "FINAL_REVISE"}),
    (ceo("run_create"), {"projectId": "proj_fresh", "kind": "STANDARD_WORK", "idempotencyKey": "key-1"}),
])
def test_a_listed_mutate_tool_runs_only_with_every_constrained_argument_in_scope(agent, ceo_tools, tool, args):
    out_of_scope = [{**args, key: "other"} for key in args if key != "resolution"]
    missing = [{k: v for k, v in args.items() if k != key} for key in args if key != "resolution"]
    with peer_turn(agent):
        for bad in out_of_scope + missing:
            assert refused(run(agent, [call(tool, bad)]), [1]), bad
        assert ceo_tools == []
        run(agent, [call(tool, args)])
    assert ceo_tools == [(tool, args)]


def test_owner_decision_required_and_non_string_values_are_out_of_scope(agent, ceo_tools):
    tool = ceo("ceo_decision_submit")
    with peer_turn(agent):
        for args in ({"runId": RUN, "decision": "OWNER_DECISION_REQUIRED"}, {"runId": [RUN], "decision": "CONFIRM"},
                     {"runId": RUN, "decision": None}):
            assert refused(run(agent, [call(tool, args)]), [1]), args
        assert ceo_tools == []
        run(agent, [call(tool, {"runId": RUN, "decision": "CONFIRM"})])
    assert len(ceo_tools) == 1


def test_mutate_tool_runs_only_while_listed_and_a_config_edit_applies_without_restart(agent, ceo_tools):
    tool, args = ceo("escalation_resolve"), {"runId": RUN}
    unlisted = _policy()
    del unlisted["mutate"][tool]
    with peer_turn(agent):
        _write_config(unlisted)
        assert refused(run(agent, [call(tool, args)]), [1]) and ceo_tools == []
        _write_config(_policy())
        run(agent, [call(tool, args)])
        assert ceo_tools == [(tool, args)]
        # A name in ``read`` never widens a ``mutate`` entry's argument scope.
        _write_config(_policy(read=[ceo("run_get"), tool]))
        assert refused(run(agent, [call(tool, {"runId": "other"})]), [1])
        _write_config(_policy(read=[], mutate={}))
        assert refused(run(agent, [call(ceo("run_get"))]), [1])
    assert ceo_tools == [(tool, args)]


def _stale_fence() -> None:
    raise ValueError("canonical_binding_stale")


@pytest.mark.parametrize("grant", [{"binding": "acp-canonical-cto"}, {"verify": _stale_fence}, {"grant": False}])
def test_a_grant_for_another_binding_a_broken_fence_or_no_grant_runs_nothing(agent, ceo_tools, grant):
    with peer_turn(agent, **grant):
        assert refused(run(agent, [call(ceo("run_get"))]), [1])
    with peer_turn(agent):
        run(agent, [call(ceo("run_get"))])
    assert ceo_tools == [(ceo("run_get"), {})]


def test_a_grant_never_authorizes_a_different_agent(agent, ceo_tools):
    other = SimpleNamespace(_turn_principal="peer", _interrupt_requested=False, log_prefix="")
    with peer_turn(agent):
        assert refused(run(other, [call(ceo("run_get"))]), [1])
        run(agent, [call(ceo("run_get"))])
    assert ceo_tools == [(ceo("run_get"), {})]


def test_the_wrapper_runs_only_listed_in_scope_inner_calls_and_nested_batches_are_all_or_nothing(agent, ceo_tools):
    def wrapped(*entries, call_id="w1"):
        return call("tool_call", {"calls": [{"name": n, "arguments": a} for n, a in entries]}, call_id)

    escalation = (ceo("escalation_resolve"), {"runId": RUN})
    with peer_turn(agent):
        for bad in (wrapped(("terminal", {"command": "id"})), wrapped((ceo("owner_decision_submit"), {})),
                    wrapped((ceo("escalation_resolve"), {"runId": "other"})),
                    wrapped(escalation, ("terminal", {"command": "id"})),
                    call("tool_call", {"name": "tool_call", "arguments": {"name": ceo("run_get")}})):
            assert refused(run(agent, [bad]), [1]), bad
        mixed = [call(ceo("run_get"), {}, "c1"), wrapped(("mcp__other__x", {}), call_id="w2")]
        assert refused(run(agent, mixed), mixed)
        assert ceo_tools == []
        messages = run(agent, [wrapped(escalation)])
    assert ceo_tools == [escalation] and real_result(messages[0], ceo("escalation_resolve"))


def test_discovery_meta_tools_see_only_the_allowed_tools(agent, ceo_tools):
    with peer_turn(agent):
        assert refused(run(agent, [call("tool_describe", {"names": [ceo("run_get"), "mcp__other__x"]})]), [1])
        described = json.loads(run(agent, [call("tool_describe", {"names": [ceo("run_get")]})])[0]["content"])
        found = json.loads(run(agent, [call("tool_search", {"queries": ["mcp tool"], "limit": 25})])[0]["content"])
    assert set(described["tools"]) == {ceo("run_get")}
    listed = set(found["tools"])
    assert ceo("run_get") in listed and listed <= set(_policy()["read"]) | set(_policy()["mutate"])


def test_a_peer_turn_never_prompts_for_an_approval(agent, ceo_tools, monkeypatch):
    import tools.approval_prompt as approval_prompt
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_scope import _server_key

    asked = []
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent", lambda *a, **k: asked.append(a) or "accept")
    monkeypatch.setitem(_core._server_trust_levels, _server_key(SERVER), _core._TRUST_UNTRUSTED)
    with peer_turn(agent):
        # An untrusted server's write-capable tool would need the owner's approval: refused, not asked.
        assert refused(run(agent, [call(ceo("run_get"))]), [1])
        monkeypatch.setitem(_core._server_trust_levels, _server_key(SERVER), _core._TRUST_FULL)
        run(agent, [call(ceo("run_get"))])
    assert asked == [] and ceo_tools == [(ceo("run_get"), {})]


def test_elicitation_consent_declines_in_a_peer_turn_without_reaching_the_owner(monkeypatch):
    from tools import approval_context, approval_gateway_wait, approval_prompt
    from tools import approval as approval_mod
    from tools.approval_context import reset_turn_principal, set_turn_principal

    reached = []
    monkeypatch.setattr(approval_context, "_is_gateway_approval_context", lambda: True)
    monkeypatch.setattr(approval_mod, "_gateway_notify_cb", lambda key: (lambda *a, **k: None))
    monkeypatch.setattr(approval_gateway_wait, "_await_gateway_decision",
                        lambda *a, **k: reached.append(a) or {"approved": False, "outcome": "denied"})
    approval_prompt.request_elicitation_consent("run it?", "owner question")
    assert len(reached) == 1, "an owner turn still reaches the owner"
    token = set_turn_principal("peer")
    try:
        assert approval_prompt.request_elicitation_consent("run it?", "peer question") == "decline"
    finally:
        reset_turn_principal(token)
    assert len(reached) == 1


def test_owner_turns_ignore_the_peer_policy_and_send_unchanged_provenance(agent, ceo_tools):
    from tools.mcp_call_provenance import build_call_provenance

    _write_config(_policy(read=[], mutate={}))
    run(agent, [call("mcp__other__x", {"any": "thing"})])
    assert ceo_tools == [("mcp__other__x", {"any": "thing"})]
    assert ceo_tools.metas == [build_call_provenance()] and set(ceo_tools.metas[0]) == OWNER_META_KEYS
    assert ceo_tools.metas[0]["principal"] == "owner"


def test_an_allowed_peer_call_carries_principal_peer_computed_depth_and_its_entry_digest(agent, ceo_tools):
    from tools.approval_context import bind_delegation_depth, reset_delegation_depth

    escalation = {"runId": RUN, "note": "free"}
    with peer_turn(agent):
        run(agent, [call(ceo("run_get"), {"runId": "r"})])
        depth = bind_delegation_depth(2)
        try:
            run(agent, [call(ceo("escalation_resolve"), escalation)])
        finally:
            reset_delegation_depth(depth)
    assert ceo_tools == [(ceo("run_get"), {"runId": "r"}), (ceo("escalation_resolve"), escalation)]
    read_meta, mutate_meta = ceo_tools.metas
    assert read_meta["principal"] == mutate_meta["principal"] == "peer"
    assert (read_meta["delegation_depth"], mutate_meta["delegation_depth"]) == (0, 2)
    assert read_meta["peer_grant"] == digest({"read": ceo("run_get")})
    assert mutate_meta["peer_grant"] == digest({"mutate": {ceo("escalation_resolve"): {"runId": [RUN]}}})
    assert set(read_meta) == OWNER_META_KEYS | {"peer_grant"}


def _rewrite_after_check(monkeypatch, how: str, rewrite):
    """Change the call between the peer check and the MCP send, the way a plugin hook or a wrapper
    re-interpretation would."""
    if how == "hook":
        monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks",
                            lambda name, args, **kw: (None, rewrite(name, dict(args))))
    else:
        import tools.tool_search as tool_search

        real = tool_search.resolve_underlying_call

        def reinterpret(args):
            name, inner, error = real(args)
            return name, rewrite(name, dict(inner)), error
        monkeypatch.setattr(tool_search, "resolve_underlying_call", reinterpret)


@pytest.mark.parametrize("how", ["hook", "wrapper"])
def test_arguments_changed_after_the_check_are_never_sent(agent, ceo_tools, monkeypatch, how):
    tool = ceo("escalation_resolve")
    batch = [call("tool_call", {"name": tool, "arguments": {"runId": RUN}}) if how == "wrapper"
             else call(tool, {"runId": RUN})]
    with peer_turn(agent):
        run(agent, batch)
        assert ceo_tools == [(tool, {"runId": RUN})]
        for rewrite in (lambda n, a: {**a, "runId": "run_other"}, lambda n, a: {**a, "extra": "added"},
                        lambda n, a: {k: v for k, v in a.items() if k != "runId"}):
            _rewrite_after_check(monkeypatch, how, rewrite)
            messages = run(agent, batch)
            assert "NOT run" in messages[0]["content"], messages[0]["content"]
    assert ceo_tools == [(tool, {"runId": RUN})]


def test_a_policy_swapped_on_disk_between_check_and_send_lets_no_other_call_through(agent, ceo_tools, monkeypatch):
    tool = ceo("escalation_resolve")
    widened = _policy()
    widened["mutate"][tool] = {"runId": [RUN, "run_other"]}

    def swap_policy_then(rewrite):
        def hook(name, args, **_kw):
            _write_config(widened)
            return None, rewrite(dict(args))
        monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", hook)

    with peer_turn(agent):
        swap_policy_then(lambda a: {**a, "runId": "run_other"})
        assert "NOT run" in run(agent, [call(tool, {"runId": RUN})])[0]["content"]
        assert ceo_tools == []
        _write_config(_policy())
        swap_policy_then(lambda a: a)
        run(agent, [call(tool, {"runId": RUN})])
    # The call that was checked went out, under the entry its check read, not the swapped one.
    assert ceo_tools == [(tool, {"runId": RUN})]
    assert ceo_tools.metas[0]["peer_grant"] == digest({"mutate": {tool: {"runId": [RUN]}}})


def test_a_pin_lives_only_until_the_next_check(agent, ceo_tools, monkeypatch):
    from tools.registry import registry

    monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", lambda name, args, **kw: ("blocked", None))
    with peer_turn(agent):
        run(agent, [call(ceo("run_get"))])  # checked and pinned, then blocked before its send
        run(agent, [call("terminal", {"command": "id"})])  # a refused batch leaves no pin behind
        unchecked = registry.get_entry(ceo("run_get")).handler({})
    assert "NOT run" in unchecked and ceo_tools == []
