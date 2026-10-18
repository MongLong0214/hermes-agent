"""A canonical peer turn under the CEO binding runs exactly the configured ACP CEO tools.

Every refusal below is checked next to a call the same turn DOES run, so a green refusal never
just means "the peer path still runs nothing". Refused means zero executions: no tool handler is
reached and every call in the batch gets the peer refusal result."""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from tests.hermes_home_pin import import_home, pinned_hermes_home

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

# Pinned for every Hermes import of this module (its import-time paths and caches) and, by a module
# fixture, for every resolver and subprocess of its tests: HOME and HERMES_HOME name a fresh tmp dir,
# and the context-local home override, which wins over both, is replaced by that same tmp home, so a
# caller's existing override (a profile scope) cannot aim this module at a real home either. A run
# outside tests/ (no hermetic conftest) once let these helpers replace the user's live config.
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_CONFIG = (_REAL_HOME / ".hermes" / "config.yaml").resolve()
_PINNED_ROOT = Path(tempfile.mkdtemp(prefix="hermes-peer-tools-")).resolve()
if _PINNED_ROOT.is_relative_to(_LIVE_CONFIG.parent) or _LIVE_CONFIG.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")
atexit.register(shutil.rmtree, _PINNED_ROOT, True)

# The imports run pinned to the tmp home all pinned modules import under, so a path they cache names
# no module's test home; the pin is put back (in ``finally``) as soon as they have run: collecting
# this module leaves no home, override or home cache behind for the modules collected after it.
with pinned_hermes_home(import_home(), importer=__name__):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    from tools.tool_search import resolve_underlying_call as _REAL_RESOLVE_UNDERLYING


@pytest.fixture(autouse=True, scope="module")
def _pin_the_module_home():
    """The same pin while this module's tests run, owned by this fixture and put back on teardown."""
    with pinned_hermes_home(_PINNED_ROOT):
        yield


@pytest.fixture(autouse=True)
def _never_the_live_config():
    """Fail before the test body when the resolver names anything in the real Hermes home."""
    from hermes_cli.config import get_config_path

    if get_config_path().resolve().is_relative_to(_LIVE_CONFIG.parent):
        pytest.fail("the config resolver names the live Hermes home")


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


_isolated_home: list = []  # the tmp HERMES_HOME ceo_tools bound; empty outside it


def _isolated_config_path() -> Path:
    """config.yaml of the tmp HERMES_HOME ``ceo_tools`` bound. Fails the test rather than return any
    path outside that pytest tmp dir, above all the live config."""
    from hermes_cli.config import get_config_path

    path = get_config_path().resolve()
    root = _isolated_home[-1].resolve() if _isolated_home else None
    if root is None or path == _LIVE_CONFIG or not path.is_relative_to(root):
        pytest.fail(f"refusing to write a config outside the test's tmp HERMES_HOME: {path}")
    return path


def _write_config(policy: dict) -> None:
    path = _isolated_config_path()
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
def ceo_tools(monkeypatch, tmp_path):
    """The tools as REAL MCP tool handlers (``_make_tool_handler``) over a fake transport, with
    HERMES_HOME and HOME bound to this test's own tmp dir whatever conftest did."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    override = set_hermes_home_override(home)
    _isolated_home.append(home)
    try:
        _isolated_config_path()  # fail before any setup when the isolation did not take
        yield from _ceo_tools(monkeypatch)
    finally:
        _isolated_home.remove(home)
        reset_hermes_home_override(override)


def _ceo_tools(monkeypatch):
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
    # One MCP loop thread, as in production: concurrent sends share each server's asyncio lock.
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    monkeypatch.setattr("tools.mcp_tool_loop._run_on_mcp_loop", lambda call, timeout=30: asyncio.run_coroutine_threadsafe(
        call() if callable(call) else call, loop).result(timeout))
    # The tool_call unwrap scopes the bridge to the session's own (pre-assembly) tool defs.
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda *a, **k: _defs(*REGISTERED))
    _write_config(_policy())
    try:
        yield sent
    finally:
        loop.call_soon_threadsafe(loop.stop)
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


def _rewrite_before_the_final_check(monkeypatch, how: str, rewrite):
    """Rewrite the call the way a plugin hook or a wrapper re-interpretation would: both run for the
    whole batch before its final check."""
    if how == "hook":
        monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks",
                            lambda name, args, **kw: (None, rewrite(name, dict(args))))
    else:
        import tools.tool_search as tool_search

        real = _REAL_RESOLVE_UNDERLYING  # never a previous rewrite: they must not stack

        def reinterpret(args):
            name, inner, error = real(args)
            return name, rewrite(name, dict(inner)), error
        monkeypatch.setattr(tool_search, "resolve_underlying_call", reinterpret)


@pytest.mark.parametrize("how", ["hook", "wrapper"])
def test_hook_and_wrapper_rewrites_are_checked_as_the_final_arguments(agent, ceo_tools, monkeypatch, how):
    tool = ceo("escalation_resolve")
    batch = [call("tool_call", {"name": tool, "arguments": {"runId": RUN}}) if how == "wrapper"
             else call(tool, {"runId": RUN})]
    with peer_turn(agent):
        for rewrite in (lambda n, a: {**a, "runId": "run_other"}, lambda n, a: {k: v for k, v in a.items() if k != "runId"}):
            _rewrite_before_the_final_check(monkeypatch, how, rewrite)
            assert refused(run(agent, batch), batch)
        assert ceo_tools == []
        # The emitted call and its rewrite both in scope: what is sent is the rewrite, as finally checked.
        decide, emitted = ceo("ceo_decision_submit"), {"runId": RUN, "decision": "CONFIRM"}
        _rewrite_before_the_final_check(monkeypatch, how, lambda n, a: {**a, "decision": "FINAL_REVISE"})
        run(agent, [call("tool_call", {"name": decide, "arguments": emitted}) if how == "wrapper"
                    else call(decide, emitted)])
    assert ceo_tools == [(decide, {"runId": RUN, "decision": "FINAL_REVISE"})]


def _change_after_the_pin(monkeypatch, change):
    """Change the call after its batch was resolved and pinned: tool-execution middleware runs then."""
    import hermes_cli.middleware as middleware

    real = middleware.run_tool_execution_middleware

    def changing(name, args, terminal_call, **kwargs):
        return real(name, args, lambda next_args: terminal_call(change(dict(next_args))), **kwargs)
    monkeypatch.setattr(middleware, "run_tool_execution_middleware", changing)


@pytest.mark.parametrize("change", [
    lambda a: {**a, "runId": "run_other"}, lambda a: {**a, "extra": "added"},
    lambda a: {k: v for k, v in a.items() if k != "runId"}])
def test_changes_after_the_batch_is_pinned_never_reach_the_wire(agent, ceo_tools, monkeypatch, change):
    tool = ceo("escalation_resolve")
    _change_after_the_pin(monkeypatch, change)
    with peer_turn(agent):
        run(agent, [call(tool, {"runId": RUN})])
    assert ceo_tools == [(tool, {"runId": RUN})], "exactly the pinned arguments went out"


def test_a_policy_swapped_on_disk_after_the_pin_lets_no_other_call_through(agent, ceo_tools, monkeypatch):
    tool = ceo("escalation_resolve")
    widened = _policy()
    widened["mutate"][tool] = {"runId": [RUN, "run_other"]}

    def swap_policy_and_retarget(args):
        _write_config(widened)
        return {**args, "runId": "run_other"}

    _change_after_the_pin(monkeypatch, swap_policy_and_retarget)
    with peer_turn(agent):
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


def _tool_round(agent, calls):
    from agent.turn_tool_round import run_tool_round

    messages = [{"role": "user", "content": "act"}]
    run_tool_round(agent, assistant_message=SimpleNamespace(content=None, tool_calls=calls),
                   finish_reason="tool_calls", messages=messages, conversation_history=[], api_call_count=1,
                   effective_task_id="task", user_message="act", system_message=None, active_system_prompt="SYSTEM",
                   compression_attempts=0, max_compression_attempts=2, final_response=None, failed=False,
                   _turn_exit_reason=None, truncated_tool_call_retries=0, current_turn_user_idx=0)
    return messages


UNKNOWN = "unavailable_zqxv_unrelated"


def test_the_real_tool_round_refuses_a_mixed_unknown_peer_batch(agent, ceo_tools):
    assert UNKNOWN not in agent.valid_tool_names and agent._repair_tool_call(UNKNOWN) is None
    with peer_turn(agent):
        # The same real round first establishes an allowed execution control.
        for suffix, calls in [("control", [call(ceo("run_get"))]),
                              ("mixed", [call(ceo("run_get"), {}, "r"), call(UNKNOWN, {}, "unknown")])]:
            _tool_round(agent, calls)
            if suffix == "control":
                assert ceo_tools == [(ceo("run_get"), {})]
                ceo_tools.clear()
    assert ceo_tools == [], f"Part of a disallowed peer batch executed: {ceo_tools}"


@pytest.mark.parametrize("plan", ["sequential", "parallel", "segmented"])
def test_every_executor_behind_the_tool_round_runs_nothing_of_a_peer_batch_with_an_unknown_call(
        agent, ceo_tools, monkeypatch, plan):
    """Two listed calls survive the round's unknown-call filter, so each executor would run them."""
    plans = {"sequential": lambda calls: [("sequential", calls)], "parallel": lambda calls: [("parallel", calls)],
             "segmented": lambda calls: [("parallel", calls[:1]), ("sequential", calls[1:])]}
    monkeypatch.setattr("agent.tool_dispatch_helpers._plan_tool_batch_segments",
                        lambda calls, **kw: plans[plan](list(calls)))
    listed = [call(ceo("run_get"), {"runId": "r1"}, "a"), call(ceo("run_get"), {"runId": "r2"}, "b")]
    with peer_turn(agent):
        _tool_round(agent, listed)
        assert len(ceo_tools) == 2, "control: the executor runs an allowed batch"
        ceo_tools.clear()
        messages = _tool_round(agent, [*listed, call(UNKNOWN, {}, "u")])
    results = [m for m in messages if m.get("role") == "tool"]
    assert ceo_tools == [] and sorted(m["tool_call_id"] for m in results) == ["a", "b", "u"]
    assert all("not executed" in m["content"] for m in results)


def _native_then_utility(server, name):
    """Register ``name`` as the server's NATIVE tool through the real registration path."""
    from tools.mcp_tool_registration import _register_server_tools

    async def list_tools():
        return SimpleNamespace(tools=[], nextCursor=None)

    server.session.list_tools = list_tools
    server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(
        tools=SimpleNamespace(), resources=SimpleNamespace(), prompts=SimpleNamespace()))
    server._tools = [SimpleNamespace(name=name.rsplit("__", 1)[1], description="native tool",
                                     inputSchema={"type": "object", "properties": {"uri": {"type": "string"},
                                                                                   "name": {"type": "string"}}},
                                     annotations=None)]
    server._registered_tool_names = _register_server_tools(SERVER, server, {})


def _forget(server):
    from tools.mcp_tool_registration import _forget_mcp_tool_server
    from tools.registry import registry

    for tool_name in server._registered_tool_names:
        registry.deregister(tool_name)
        _forget_mcp_tool_server(tool_name)


def test_native_tool_replaced_by_utility_does_not_inherit_peer_authority(agent, ceo_tools, monkeypatch):
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_handlers import mcp_tool_origin

    server = _core._servers[SERVER]
    name = ceo("read_resource")
    resource_calls = []

    async def read_resource(uri):
        resource_calls.append(uri)
        return SimpleNamespace(contents=[SimpleNamespace(text="resource bytes")])

    server.session.read_resource = read_resource
    _native_then_utility(server, name)
    _write_config(_policy(read=[], mutate={name: {"uri": ["test://native"]}}))
    try:
        with peer_turn(agent):
            run(agent, [call(name, {"uri": "test://native"})])
            assert ceo_tools == [(name, {"uri": "test://native"})] and resource_calls == []
            # Production tools/list_changed path removes the native tool, then adds the
            # generated resources/read utility under the same registry spelling.
            asyncio.run(server._refresh_tools())
            assert server._tools == []
            monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks",
                                lambda name, args, **kw: (None, {**args, "uri": "test://utility"}))
            run(agent, [call(name, {"uri": "test://native"})])
        assert resource_calls == [], (mcp_tool_origin(name), resource_calls)
    finally:
        _forget(server)


@pytest.mark.parametrize("tool, method, args", [
    ("list_resources", "list_resources", {}),
    ("list_prompts", "list_prompts", {}),
    ("get_prompt", "get_prompt", {"name": "p"}),
])
def test_every_generated_utility_substituted_by_a_refresh_is_refused_in_a_peer_turn(
        agent, ceo_tools, tool, method, args):
    from tools.mcp_tool_common import _core
    from tools.registry import registry

    server = _core._servers[SERVER]
    name, rpc_calls = ceo(tool), []

    async def rpc(*a, **kw):
        rpc_calls.append(a)
        return SimpleNamespace(resources=[], prompts=[], nextCursor=None, messages=[], description="")

    setattr(server.session, method, rpc)
    _native_then_utility(server, name)
    _write_config(_policy(read=[name], mutate={}))
    try:
        with peer_turn(agent):
            run(agent, [call(name, args)])
            assert ceo_tools == [(name, args)] and rpc_calls == []
            asyncio.run(server._refresh_tools())
            assert registry.get_entry(name) is not None, "the refresh registered the generated utility"
            refused_batch = run(agent, [call(name, args)])
            direct = registry.get_entry(name).handler(dict(args))
        assert refused(refused_batch, [1]) and "NOT run" in direct and rpc_calls == []
    finally:
        _forget(server)


def test_config_writes_refuse_any_home_but_the_tests_own(ceo_tools, tmp_path):
    elsewhere = tmp_path.parent / f"{tmp_path.name}-elsewhere"
    token = set_hermes_home_override(elsewhere)  # the binding that wins over HERMES_HOME
    try:
        with pytest.raises(pytest.fail.Exception, match="refusing to write"):
            _write_config(_policy())
    finally:
        reset_hermes_home_override(token)
    assert not elsewhere.exists()


def test_two_profile_policies_and_registrations_remain_separate(agent, ceo_tools, tmp_path):
    from agent.canonical_peer_tools import batch_allowed
    from agent.secret_scope import set_multiplex_active
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_tool_registration import _track_mcp_tool_server
    from tools.mcp_tool_scope import _server_key
    from tools.registry import registry

    name = ceo("run_get")
    a, b = tmp_path / "A", tmp_path / "B"
    for home, read in [(a, [name]), (b, [])]:
        home.mkdir()
        (home / "config.yaml").write_text(
            yaml.safe_dump({"gateway": {"canonical_peer_tools": _policy(read=read, mutate={})}}), encoding="utf-8")
    results = []
    set_multiplex_active(True)
    try:
        for home in (a, b, a):
            token = set_hermes_home_override(home)
            try:
                registry.register(name, "mcp-acp-ceo", _defs(name)[0]["function"],
                                  _make_tool_handler("acp-ceo", "run_get", 30))
                _track_mcp_tool_server(name, "acp-ceo")
                _core._server_trust_levels[_server_key("acp-ceo")] = _core._TRUST_FULL
                with peer_turn(agent):
                    results.append(batch_allowed(agent, [(name, {})]))
                registry.deregister(name)
            finally:
                reset_hermes_home_override(token)
    finally:
        set_multiplex_active(False)
    assert results == [True, False, True]


def test_refresh_refusal_is_decided_before_any_sibling_executes(agent, ceo_tools, monkeypatch):
    from tools.mcp_tool_common import _core
    from tools.mcp_tool_registration import _forget_mcp_tool_server, _register_server_tools
    from tools.registry import registry

    server = _core._servers[SERVER]
    second = ceo("read_resource")
    resource_calls = []

    async def read_resource(uri):
        resource_calls.append(uri)
        return SimpleNamespace(contents=[SimpleNamespace(text="bytes")])

    async def list_tools():
        return SimpleNamespace(tools=[], nextCursor=None)

    server.session.read_resource = read_resource
    server.session.list_tools = list_tools
    server.initialize_result = SimpleNamespace(capabilities=SimpleNamespace(
        tools=SimpleNamespace(), resources=SimpleNamespace(), prompts=None))
    server._tools = [SimpleNamespace(name="read_resource", description="native",
                                     inputSchema={"type": "object", "properties": {"uri": {"type": "string"}}},
                                     annotations=None)]
    server._registered_tool_names = _register_server_tools(SERVER, server, {})
    _write_config(_policy(read=[ceo("run_get")], mutate={second: {"uri": ["test://native"]}}))

    def hook(name, args, **kwargs):
        if name == second:
            asyncio.run(server._refresh_tools())
            return None, {**args, "uri": "test://utility"}
        return None, args

    monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", hook)
    try:
        with peer_turn(agent):
            messages = run(agent, [call(ceo("run_get"), {}, "c1"), call(second, {"uri": "test://native"}, "c2")])
        assert ceo_tools == [] and resource_calls == [], (ceo_tools, resource_calls, messages)
    finally:
        for name in server._registered_tool_names:
            registry.deregister(name)
            _forget_mcp_tool_server(name)


def test_pinned_arguments_cannot_change_after_claim_while_waiting_for_transport(agent, ceo_tools, monkeypatch):
    import tools.mcp_tool_handlers as handlers

    kept = {}

    def hook(name, args, **kwargs):
        kept["args"] = args
        return None, args

    acquire = handlers._acquire_call_server

    def delayed(server_name, timeout):
        kept["args"]["payload"]["scope"] = "rewritten-after-claim"
        return acquire(server_name, timeout)

    with peer_turn(agent):
        run(agent, [call(ceo("run_get"), {"payload": {"scope": "checked"}})])
        assert ceo_tools == [(ceo("run_get"), {"payload": {"scope": "checked"}})]
        ceo_tools.clear()
        monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", hook)
        monkeypatch.setattr(handlers, "_acquire_call_server", delayed)
        run(agent, [call(ceo("run_get"), {"payload": {"scope": "checked"}})])
    assert ceo_tools == [], f"Arguments changed after pin claim reached MCP: {ceo_tools}"


def test_the_send_carries_the_pinned_snapshot_not_an_alias_mutated_after_the_check(agent, ceo_tools, monkeypatch):
    """The wire gets the owned snapshot: an alias mutated between the last check and the write
    cannot change the bytes sent."""
    import tools.mcp_tool_handlers as handlers

    kept = {}
    monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks",
                        lambda name, args, **kw: (kept.setdefault("args", args), (None, args))[1])
    racing = handlers._call_tool_racing_stdio_death

    async def mutate_then_send(server, server_name, tool_name, args, **kwargs):
        kept["args"]["payload"]["scope"] = "rewritten-at-the-wire"
        return await racing(server, server_name, tool_name, args, **kwargs)

    monkeypatch.setattr(handlers, "_call_tool_racing_stdio_death", mutate_then_send)
    with peer_turn(agent):
        run(agent, [call(ceo("run_get"), {"payload": {"scope": "checked"}})])
    assert ceo_tools == [(ceo("run_get"), {"payload": {"scope": "checked"}})]


@pytest.mark.parametrize("dispatcher", DISPATCHERS)
def test_final_batch_pins_the_tool_name_as_well_as_arguments(agent, ceo_tools, monkeypatch, dispatcher):
    from tools.registry import registry

    tool = ceo("run_get")
    emitted = call(tool, {"scope": "checked"})
    effects = []
    unlisted = "unlisted_peer_effect"
    registry.register(unlisted, "review-observer", {"name": unlisted, "parameters": {"type": "object"}},
                      lambda args, **kw: (effects.append(dict(args)), json.dumps({"result": "effect"}))[1])

    def hook(name, args, **kwargs):
        emitted.function.name = unlisted
        return None, args

    monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", hook)
    try:
        with peer_turn(agent):
            messages = run(agent, [emitted], dispatcher)
        assert effects == [], (effects, ceo_tools, messages)
        assert refused(messages, [emitted]) and ceo_tools == [], "a call renamed while prepared refuses the batch"
    finally:
        registry.deregister(unlisted)


def test_wrapped_call_uses_its_pinned_handler_after_registry_refresh(agent, ceo_tools, monkeypatch):
    from tools.registry import registry

    tool = ceo("run_get")
    first = call(tool, {"step": "first"}, "c1")
    second = call("tool_call", {"name": tool, "arguments": {"step": "second"}}, "c2")
    entry = registry.get_entry(tool)

    def refresh(call_id, name, args):
        if call_id == "c1":
            registry.register(tool, "mcp-acp-ceo", {"name": tool, "parameters": {
                "type": "object", "properties": {"new_required": {"type": "string"}}, "required": ["new_required"]}},
                entry.handler)

    agent.tool_start_callback = refresh
    with peer_turn(agent):
        messages = run(agent, [first, second])
    assert ceo_tools == [(tool, {"step": "first"}), (tool, {"step": "second"})], (ceo_tools, messages)


def test_a_peer_call_without_a_prepared_entry_never_reaches_a_registry_handler(agent, ceo_tools):
    import model_tools
    from tools.registry import registry

    effects = []
    unlisted = "unlisted_peer_effect"
    registry.register(unlisted, "review-observer", {"name": unlisted, "parameters": {"type": "object"}},
                      lambda args, **kw: (effects.append(dict(args)), json.dumps({"result": "effect"}))[1])
    try:
        with peer_turn(agent):
            run(agent, [call(ceo("run_get"), {}, "c1")])  # an allowed, prepared batch exists
            unprepared = model_tools.handle_function_call(unlisted, {}, tool_call_id="not-prepared")
            listed_but_unprepared = model_tools.handle_function_call(ceo("run_get"), {"other": 1},
                                                                     tool_call_id="not-prepared")
    finally:
        registry.deregister(unlisted)
    assert effects == [] and "NOT run" in unprepared and "NOT run" in listed_but_unprepared
    assert ceo_tools == [(ceo("run_get"), {})]
