"""Whole-batch admission for canonical peer turns: a peer batch runs only as a whole, decided on the
batch the model emitted, by the tool round and by every executor behind it.

Every refusal below is checked next to a batch the same turn DOES run. The tools are REAL MCP tool
handlers (``_make_tool_handler``) over a fake transport that records each call reaching the wire;
refused means no call reached it and every call of the batch got the peer refusal result."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

import pytest

try:
    import pwd  # the account database names the real home even when HOME is redirected
except ImportError:  # not POSIX
    pwd = None

# Pinned BEFORE any dependent Hermes import (see tests/agent/test_canonical_peer_tools.py): HOME,
# HERMES_HOME and the context-local home override name a fresh tmp dir for this whole module.
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_CONFIG = (_REAL_HOME / ".hermes" / "config.yaml").resolve()
_PINNED_ROOT = Path(tempfile.mkdtemp(prefix="hermes-peer-batch-")).resolve()
if _PINNED_ROOT.is_relative_to(_LIVE_CONFIG.parent) or _LIVE_CONFIG.is_relative_to(_PINNED_ROOT):
    raise RuntimeError(f"the pinned test home {_PINNED_ROOT} overlaps the real home")
atexit.register(shutil.rmtree, _PINNED_ROOT, True)
_ENV_BEFORE = {key: os.environ.get(key) for key in ("HOME", "HERMES_HOME")}
os.environ["HOME"] = str(_PINNED_ROOT)
os.environ["HERMES_HOME"] = str(_PINNED_ROOT / ".hermes")

from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_MODULE_PIN = set_hermes_home_override(_PINNED_ROOT / ".hermes")


@pytest.fixture(autouse=True, scope="module")
def _release_the_module_pin():
    yield
    reset_hermes_home_override(_MODULE_PIN)
    for key, value in _ENV_BEFORE.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _never_the_live_config():
    from hermes_cli.config import get_config_path

    if get_config_path().resolve().is_relative_to(_LIVE_CONFIG.parent):
        pytest.fail("the config resolver names the live Hermes home")


import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import hermes_yaml as yaml

SERVER = "acp-ceo"
BINDING = "acp-canonical-ceo"
RUN = "run_0000000000000000000000ab"
DISPATCHERS = ("execute_tool_calls_sequential", "execute_tool_calls_concurrent", "execute_tool_calls_segmented")
UNKNOWN = "unavailable_zqxv_unrelated"


def ceo(tool: str) -> str:
    return f"mcp__acp_ceo__{tool}"


SERVER_TOOLS = [ceo(t) for t in ("run_get", "escalation_resolve", "owner_decision_submit")]
POLICY = {"binding": BINDING, "mcp_server": SERVER, "read": [ceo("run_get")],
          "mutate": {ceo("escalation_resolve"): {"runId": [RUN, "run_second"]}}}


def _defs(*names: str) -> list[dict]:
    return [{"type": "function", "function": {"name": n, "description": f"{n} tool", "parameters": {
        "type": "object", "properties": {"runId": {"type": "string"}}}}} for n in names]


@pytest.fixture
def ceo_tools(monkeypatch, tmp_path):
    """The tools as REAL MCP tool handlers over a fake transport, the policy in this test's own tmp
    HERMES_HOME. Yields the ``(tool, arguments)`` that reached the wire."""
    import model_tools
    from hermes_cli.config import get_config_path
    from tools import mcp_tool_handlers, mcp_tool_loop
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.mcp_tool_registration import _forget_mcp_tool_server, _track_mcp_tool_server
    from tools.registry import registry

    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    override = set_hermes_home_override(home)
    sent: list = []

    def server_for(server_name):
        def call_tool(tool, arguments=None, **_):
            sent.append((ceo(tool), dict(arguments)))
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"ok:{ceo(tool)}")], isError=False)
        return SimpleNamespace(session=SimpleNamespace(call_tool=call_tool), _rpc_lock=asyncio.Lock())

    try:
        config = get_config_path().resolve()
        if config == _LIVE_CONFIG or not config.is_relative_to(home.resolve()):
            pytest.fail(f"refusing to write a config outside the test's tmp HERMES_HOME: {config}")
        config.write_text(yaml.safe_dump({"gateway": {"canonical_peer_tools": POLICY}}), encoding="utf-8")
        monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda name, timeout: (server_for(name), None))
        monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda call, timeout=30: asyncio.run(call()))
        for name in SERVER_TOOLS:
            registry.register(name, f"mcp-{SERVER}", _defs(name)[0]["function"],
                              _make_tool_handler(SERVER, name.rsplit("__", 1)[1], 30))
            _track_mcp_tool_server(name, SERVER)
        monkeypatch.setattr(model_tools, "get_tool_definitions", lambda *a, **k: _defs(*SERVER_TOOLS))
        yield sent
    finally:
        for name in SERVER_TOOLS:
            registry.deregister(name)
            _forget_mcp_tool_server(name)
        reset_hermes_home_override(override)


@pytest.fixture
def agent(ceo_tools):
    from run_agent import AIAgent

    with (
        patch("model_tools.get_tool_definitions", return_value=_defs(*SERVER_TOOLS, "terminal", "write_file")),
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
    built._flush_messages_to_session_db = lambda *a, **k: True
    return built


@contextmanager
def peer_turn(agent):
    """What the verified canonical ingress binds for one peer turn."""
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
    from tools.approval_context import reset_turn_principal, set_turn_principal

    principal = set_turn_principal("peer")
    agent._turn_principal = "peer"
    token = bind_grant(PeerToolGrant(binding=BINDING, agent=agent, verify=lambda: None))
    try:
        yield
    finally:
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


def tool_round(agent, calls) -> list:
    from agent.turn_tool_round import run_tool_round

    messages = [{"role": "user", "content": "act"}]
    run_tool_round(agent, assistant_message=SimpleNamespace(content=None, tool_calls=list(calls)),
                   finish_reason="tool_calls", messages=messages, conversation_history=[], api_call_count=1,
                   effective_task_id="task", user_message="act", system_message=None, active_system_prompt="SYSTEM",
                   compression_attempts=0, max_compression_attempts=2, final_response=None, failed=False,
                   _turn_exit_reason=None, truncated_tool_call_retries=0, current_turn_user_idx=0)
    return [m for m in messages if m.get("role") == "tool"]


def refused(results, ids) -> bool:
    return (sorted(m["tool_call_id"] for m in results) == sorted(ids)
            and all("canonical peer turn" in m["content"] and "not executed" in m["content"] for m in results))


# ── every executor (counterexample 1) ────────────────────────────────────


@pytest.mark.parametrize("dispatcher", DISPATCHERS)
def test_every_executor_refuses_a_batch_with_one_ungranted_call_whole(agent, ceo_tools, dispatcher, monkeypatch):
    import model_tools

    dispatched: list = []
    real = model_tools.handle_function_call
    monkeypatch.setattr(model_tools, "handle_function_call",
                        lambda name, *a, **k: (dispatched.append(name), real(name, *a, **k))[1])
    batch = [call(ceo("run_get"), {"runId": "r1"}, "a"), call(ceo("owner_decision_submit"), {}, "b")]
    with peer_turn(agent):
        assert refused(run(agent, batch, dispatcher), ["a", "b"])
        assert dispatched == [] and ceo_tools == []
        results = run(agent, batch[:1], dispatcher)  # the granted call alone runs
    assert ceo_tools == [(ceo("run_get"), {"runId": "r1"})] and "ok:" in results[0]["content"]


@pytest.mark.parametrize("dispatcher", DISPATCHERS)
def test_every_executor_runs_an_allowed_batch_only_through_its_prepared_calls(agent, ceo_tools, dispatcher):
    batch = [call(ceo("run_get"), {"runId": "r1"}, "a"), call(ceo("escalation_resolve"), {"runId": RUN}, "b")]
    with peer_turn(agent):
        results = run(agent, batch, dispatcher)
    assert sorted(ceo_tools) == sorted([(ceo("run_get"), {"runId": "r1"}), (ceo("escalation_resolve"), {"runId": RUN})])
    assert all("ok:" in m["content"] for m in results)


@pytest.mark.parametrize("plan", ["sequential", "parallel", "segmented"])
def test_the_round_refuses_a_peer_batch_with_an_unknown_call_before_any_executor_sees_it(
        agent, ceo_tools, monkeypatch, plan):
    """Two granted calls survive the round's unknown-call filter, so each executor would run them."""
    plans = {"sequential": lambda calls: [("sequential", calls)], "parallel": lambda calls: [("parallel", calls)],
             "segmented": lambda calls: [("parallel", calls[:1]), ("sequential", calls[1:])]}
    monkeypatch.setattr("agent.tool_dispatch_helpers._plan_tool_batch_segments",
                        lambda calls, **kw: plans[plan](list(calls)))
    granted = [call(ceo("run_get"), {"runId": "r1"}, "a"), call(ceo("run_get"), {"runId": "r2"}, "b")]
    assert UNKNOWN not in agent.valid_tool_names and agent._repair_tool_call(UNKNOWN) is None
    with peer_turn(agent):
        tool_round(agent, granted)
        assert len(ceo_tools) == 2, "control: the round runs an allowed batch"
        ceo_tools.clear()
        results = tool_round(agent, [*granted, call(UNKNOWN, {}, "u")])
    assert ceo_tools == [] and refused(results, ["a", "b", "u"])


# ── the emitted batch decides, before dedupe or cap (counterexamples 1 and 2) ──


def _spy_pruning(agent, monkeypatch) -> list:
    seen: list = []
    dedupe, cap = agent._deduplicate_tool_calls, agent._cap_delegate_task_calls
    monkeypatch.setattr(agent, "_deduplicate_tool_calls", lambda calls: (seen.append("dedupe"), dedupe(calls))[1])
    monkeypatch.setattr(agent, "_cap_delegate_task_calls", lambda calls: (seen.append("cap"), cap(calls))[1])
    return seen


def test_the_round_decides_before_dedupe_and_cap_and_refuses_every_emitted_call(agent, ceo_tools, monkeypatch):
    seen = _spy_pruning(agent, monkeypatch)
    duplicate = [call(ceo("run_get"), {"runId": "r1"}, "a"), call(ceo("run_get"), {"runId": "r1"}, "b")]
    with peer_turn(agent):
        results = tool_round(agent, [*duplicate, call("terminal", {"command": "id"}, "t")])
        assert seen == [] and ceo_tools == [] and refused(results, ["a", "b", "t"])
        tool_round(agent, duplicate)  # control: an allowed batch is pruned as before and runs
    assert seen == ["cap", "dedupe"] and ceo_tools == [(ceo("run_get"), {"runId": "r1"})]


def test_a_name_the_round_repaired_is_not_the_emitted_batch(agent, ceo_tools, monkeypatch):
    typo = ceo("run_gett")
    monkeypatch.setattr(agent, "_repair_tool_call", lambda name: ceo("run_get") if name == typo else None)
    with peer_turn(agent):
        assert refused(tool_round(agent, [call(typo, {"runId": "r1"}, "a")]), ["a"])
    assert ceo_tools == []
    tool_round(agent, [call(typo, {"runId": "r1"}, "a")])  # an owner turn still runs the repaired call
    assert ceo_tools == [(ceo("run_get"), {"runId": "r1"})]


def test_hook_rewrites_are_prepared_once_and_checked_as_the_final_arguments(agent, ceo_tools, monkeypatch):
    rewrites: list = []

    def hook(name, args, **kw):
        rewrites.append(name)
        return None, {**args, "runId": target}

    monkeypatch.setattr("hermes_cli.plugins._dispatch_pre_tool_call_hooks", hook)
    with peer_turn(agent):
        target = "run_other"  # the hook moves an in-scope call out of scope: refused before any send
        assert refused(run(agent, [call(ceo("escalation_resolve"), {"runId": RUN})]), ["c1"])
        target, rewrites[:] = "run_second", []  # emitted and rewrite both in scope: the rewrite is sent
        run(agent, [call(ceo("escalation_resolve"), {"runId": RUN})])
    assert ceo_tools == [(ceo("escalation_resolve"), {"runId": "run_second"})]
    assert rewrites == [ceo("escalation_resolve")], "the hooks ran once, when the batch was prepared"


def test_request_middleware_runs_once_for_a_prepared_peer_call(agent, ceo_tools, monkeypatch):
    from hermes_cli import middleware

    seen: list = []
    real = middleware.apply_tool_request_middleware
    monkeypatch.setattr(middleware, "apply_tool_request_middleware",
                        lambda name, args, **kw: (seen.append(name), real(name, args, **kw))[1])
    run(agent, [call(ceo("run_get"), {"runId": "r1"})])  # control: an owner call passes it once, at dispatch
    assert seen == [ceo("run_get")]
    seen.clear()
    with peer_turn(agent):
        run(agent, [call(ceo("run_get"), {"runId": "r2"})])
    assert seen == [ceo("run_get")], "a peer call's request middleware ran once, when its batch was prepared"
    assert ceo_tools == [(ceo("run_get"), {"runId": "r1"}), (ceo("run_get"), {"runId": "r2"})]


def test_a_peer_call_without_a_prepared_entry_is_blocked_at_parse(agent, ceo_tools):
    from agent.tool_executor import _parse_tool_call

    with peer_turn(agent):
        parsed = _parse_tool_call(agent, call(ceo("run_get")))
    assert parsed.scope_block is not None and "canonical peer turn" in parsed.scope_block
    assert ceo("run_get") in parsed.scope_block and "{name}" not in parsed.scope_block
    assert _parse_tool_call(agent, call(ceo("run_get"))).scope_block is None  # owner turn


# ── codex app-server (counterexample 3) ──────────────────────────────────


@pytest.mark.parametrize("principal", ["agent", "context"])
def test_a_codex_app_server_turn_under_a_peer_principal_is_refused_before_the_request(monkeypatch, principal):
    from agent import codex_runtime
    from tools.approval_context import reset_turn_principal, set_turn_principal

    reached: list = []

    def reach(*a, **k):
        reached.append("session")
        raise LookupError("stop: the request path was reached")

    monkeypatch.setattr(codex_runtime, "_ensure_codex_session", reach)
    stub = SimpleNamespace(_turn_principal="peer" if principal == "agent" else None,
                           compression_checkpoint_required=False)
    token = set_turn_principal("peer") if principal == "context" else None
    try:
        with pytest.raises(PermissionError, match="canonical_peer_runtime_refused"):
            codex_runtime.run_codex_app_server_turn(stub, user_message="hi", original_user_message="hi",
                                                    messages=[], effective_task_id="t")
    finally:
        if token is not None:
            reset_turn_principal(token)
    assert reached == []
    stub._turn_principal = None
    with pytest.raises(LookupError):  # an owner turn goes on to the request path
        codex_runtime.run_codex_app_server_turn(stub, user_message="hi", original_user_message="hi",
                                                messages=[], effective_task_id="t")
    assert reached == ["session"]


# ── owner batches (counterexample 4) ─────────────────────────────────────


def test_owner_batches_ignore_the_peer_policy_in_the_round_and_every_executor(agent, ceo_tools, monkeypatch):
    seen = _spy_pruning(agent, monkeypatch)
    results = tool_round(agent, [call(ceo("run_get"), {}, "a"), call(ceo("owner_decision_submit"), {}, "b"),
                                 call(UNKNOWN, {}, "u")])
    assert seen == ["cap", "dedupe"]
    assert ceo_tools == [(ceo("run_get"), {}), (ceo("owner_decision_submit"), {})]
    assert not any("canonical peer turn" in m["content"] for m in results)
    for dispatcher in DISPATCHERS:
        ceo_tools.clear()
        run(agent, [call(ceo("owner_decision_submit"), {"runId": "x"}, "o")], dispatcher)
        assert ceo_tools == [(ceo("owner_decision_submit"), {"runId": "x"})], dispatcher
