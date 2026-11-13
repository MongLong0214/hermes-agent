"""The peer tool grant: a canonical peer turn discovers and dispatches only what its grant allows.

Every refusal below is checked next to a call the same turn DOES run, so a green refusal never
just means "the peer path runs nothing". The tools are REAL MCP tool handlers
(``_make_tool_handler``) over a fake transport that records each call reaching the wire."""

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

# Pinned BEFORE any dependent Hermes import, for every resolver, import-time path, cache and
# subprocess of this module: HOME and HERMES_HOME name a fresh tmp dir, and the context-local home
# override, which wins over both, is replaced by that same tmp home, so a caller's existing override
# (a profile scope) cannot aim this module at a real home either. A run outside tests/ (no hermetic
# conftest) once let peer tool test helpers replace the user's live config.
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_CONFIG = (_REAL_HOME / ".hermes" / "config.yaml").resolve()
_PINNED_ROOT = Path(tempfile.mkdtemp(prefix="hermes-peer-tools-")).resolve()
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
    """Fail before the test body when the resolver names anything in the real Hermes home."""
    from hermes_cli.config import get_config_path

    if get_config_path().resolve().is_relative_to(_LIVE_CONFIG.parent):
        pytest.fail("the config resolver names the live Hermes home")


import asyncio
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import hermes_yaml as yaml

SERVER = "acp-ceo"
BINDING = "acp-canonical-ceo"
RUN = "run_0000000000000000000000ab"


def ceo(tool: str) -> str:
    return f"mcp__acp_ceo__{tool}"


# Native MCP tools per server. ``owner_decision_submit`` is a real tool the policy never lists;
# ``ghost`` is listed but never registered; the other server's ``x`` is listed by mistake and must
# still be refused.
SERVER_TOOLS = {SERVER: ["run_get", "escalation_resolve", "run_create", "owner_decision_submit"], "other": ["x"]}
REGISTERED = [ceo(t) for t in SERVER_TOOLS[SERVER]] + ["mcp__other__x"]
ALLOWED = {ceo("run_get"), ceo("escalation_resolve"), ceo("run_create")}


def _policy(**changes) -> dict:
    policy = {
        "binding": BINDING, "mcp_server": SERVER,
        "read": [ceo("run_get"), ceo("ghost"), "mcp__other__x"],
        "mutate": {
            ceo("escalation_resolve"): {"runId": [RUN]},
            ceo("run_create"): {"projectId": ["proj_fresh"], "kind": ["STANDARD_WORK"]},
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


def _schema(name: str, run_id_type: str = "string") -> dict:
    return {"name": name, "description": f"{name} tool for acp ceo runs",
            "parameters": {"type": "object", "properties": {"runId": {"type": run_id_type}}}}


def _defs(*names: str) -> list[dict]:
    return [{"type": "function", "function": _schema(n)} for n in names]


def _register(name: str, server: str, tool: str, run_id_type: str = "string"):
    from tools.mcp_tool_handlers import _make_tool_handler
    from tools.registry import registry

    handler = _make_tool_handler(server, tool, 30)
    registry.deregister(name)
    registry.register(name, f"mcp-{server}", _schema(name, run_id_type), handler)
    return handler


@pytest.fixture
def ceo_tools(monkeypatch, tmp_path):
    """The tools as REAL MCP tool handlers over a fake transport, with HERMES_HOME and HOME bound to
    this test's own tmp dir whatever conftest did. Yields the ``(tool, arguments)`` sent."""
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
    from tools import mcp_tool_handlers, mcp_tool_loop
    from tools.mcp_tool_registration import _forget_mcp_tool_server, _track_mcp_tool_server
    from tools.registry import registry

    sent: list = []

    def server_for(server_name):
        def call_tool(tool, arguments=None, **_):
            name = f"mcp__{server_name.replace('-', '_')}__{tool}"
            sent.append((name, dict(arguments)))
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"ok:{name}")], isError=False)
        return SimpleNamespace(session=SimpleNamespace(call_tool=call_tool), _rpc_lock=asyncio.Lock())

    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda name, timeout: (server_for(name), None))
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda call, timeout=30: asyncio.run(call()))
    for server, tools in SERVER_TOOLS.items():
        for tool in tools:
            name = f"mcp__{server.replace('-', '_')}__{tool}"
            _register(name, server, tool)
            _track_mcp_tool_server(name, server)
    # The bridge scopes discovery to the session's own (pre-assembly) tool defs.
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda *a, **k: _defs(*REGISTERED))
    _write_config(_policy())
    try:
        yield sent
    finally:
        for name in REGISTERED:
            registry.deregister(name)
            _forget_mcp_tool_server(name)


AGENT = SimpleNamespace(name="the agent the grant was issued to")


@contextmanager
def peer_turn(*, agent=AGENT, binding=BINDING, verify=lambda: None, grant=True):
    """What the verified canonical ingress binds for one peer turn."""
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
    from tools.approval_context import reset_turn_principal, set_turn_principal

    principal = set_turn_principal("peer")
    bound = PeerToolGrant(binding=binding, agent=agent, verify=verify) if grant else None
    token = bind_grant(bound) if grant else None
    try:
        yield bound
    finally:
        if token is not None:
            reset_grant(token)
        reset_turn_principal(principal)


def call(name: str, args: dict | None = None, call_id: str = "c1") -> str:
    import model_tools

    return model_tools.handle_function_call(name, dict(args or {}), task_id="t", tool_call_id=call_id)


def tool_call(name: str, args: dict | None = None, call_id: str = "c1"):
    return SimpleNamespace(id=call_id, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args or {})))


def ran(result: str, name: str) -> bool:
    return f"ok:{name}" in result


def refused(result: str) -> bool:
    return "was NOT run: a canonical peer turn dispatches only the calls its batch prepared" in result


# ── discovery (counterexamples 1 and 5) ──────────────────────────────────


def _search(query: str = "acp ceo runs tool") -> dict:
    return json.loads(call("tool_search", {"queries": [query], "limit": 20}))


def _describe(*names: str) -> dict:
    return json.loads(call("tool_describe", {"names": list(names)}))


def test_peer_discovery_lists_only_the_granted_tools(ceo_tools):
    with peer_turn():
        searched, described = _search(), _describe(ceo("run_get"), ceo("owner_decision_submit"), "mcp__other__x")
    assert set(searched["tools"]) == ALLOWED and searched["total_available"] == len(ALLOWED)
    assert set(described["tools"]) == {ceo("run_get")}
    assert ceo("owner_decision_submit") not in json.dumps(described["tools"])


@pytest.mark.parametrize("scope", [{"grant": False}, {"binding": "acp-canonical-cto"}, {"verify": "stale"}])
def test_a_peer_turn_without_an_active_grant_discovers_nothing(ceo_tools, scope):
    from agent.canonical_peer_tools import scope_discovery_defs

    with peer_turn(**({"verify": _stale_fence} if scope == {"verify": "stale"} else scope)):
        assert _search()["total_available"] == 0
        assert scope_discovery_defs(_defs(*REGISTERED)) == []


def test_owner_turns_ignore_the_peer_policy_and_discover_and_run_unchanged(ceo_tools):
    from agent.canonical_peer_tools import dispatch_handler, prepared_call, scope_discovery_defs

    defs = _defs(*REGISTERED, "terminal")
    assert scope_discovery_defs(defs) == defs
    assert _search()["total_available"] == len(REGISTERED)
    assert set(_describe(*REGISTERED)["tools"]) == set(REGISTERED)
    assert dispatch_handler(ceo("owner_decision_submit"), {}, "c1") is None and prepared_call("c1") is None
    assert ran(call(ceo("owner_decision_submit"), {"runId": "anything"}), ceo("owner_decision_submit"))
    assert ceo_tools == [(ceo("owner_decision_submit"), {"runId": "anything"})]


# ── dispatch (counterexample 2) ──────────────────────────────────────────


def test_an_ungranted_call_resolves_to_refused_and_its_handler_never_runs(ceo_tools):
    from agent.canonical_peer_tools import REFUSED, batch_allowed, dispatch_handler

    with peer_turn():
        assert batch_allowed(AGENT, [(ceo("run_get"), {"runId": RUN})])
        assert dispatch_handler(ceo("owner_decision_submit"), {}, "c2") is REFUSED
        assert refused(call(ceo("owner_decision_submit"), {}, call_id="c2"))
        assert ran(call(ceo("run_get"), {"runId": RUN}), ceo("run_get"))  # the granted call next to it runs
        assert refused(call(ceo("run_get"), {"runId": "run_other"}))  # pinned for exactly its arguments
    assert ceo_tools == [(ceo("run_get"), {"runId": RUN})]


@pytest.mark.parametrize("batch", [
    [(ceo("run_get"), {}), (ceo("owner_decision_submit"), {})],  # one unlisted call
    [(ceo("run_get"), {}), ("mcp__other__x", {})],  # listed, but another server's tool
    [(ceo("run_get"), {}), (ceo("ghost"), {})],  # listed, never registered
    [(ceo("run_get"), {}), (ceo("escalation_resolve"), {"runId": "run_other"})],  # out-of-scope value
    [(ceo("run_get"), {}), (ceo("escalation_resolve"), {})],  # constrained argument missing
    [(ceo("run_get"), {}), (ceo("run_create"), {"projectId": "proj_fresh", "kind": 7})],  # not a string
    [(ceo("run_get"), {}), (ceo("run_get"), None)],  # arguments that did not parse
])
def test_a_batch_with_any_refused_call_pins_nothing_and_runs_nothing(ceo_tools, batch):
    from agent.canonical_peer_tools import batch_allowed

    with peer_turn() as grant:
        assert batch_allowed(AGENT, [batch[0]])  # the same first call alone is allowed
        assert not batch_allowed(AGENT, batch)
        assert grant.pins == {} and grant.prepared == {}
        assert refused(call(*batch[0]))
    assert ceo_tools == []


def test_mutate_wins_over_read_and_runs_only_with_every_constrained_argument_in_scope(ceo_tools):
    from agent.canonical_peer_tools import batch_allowed

    _write_config(_policy(read=[ceo("run_get"), ceo("escalation_resolve")]))
    with peer_turn():
        assert not batch_allowed(AGENT, [(ceo("escalation_resolve"), {"runId": "run_other"})])
        args = {"projectId": "proj_fresh", "kind": "STANDARD_WORK", "title": "free text"}
        assert batch_allowed(AGENT, [(ceo("escalation_resolve"), {"runId": RUN}), (ceo("run_create"), args)])
        assert ran(call(ceo("escalation_resolve"), {"runId": RUN}), ceo("escalation_resolve"))
        assert ran(call(ceo("run_create"), args, call_id="c2"), ceo("run_create"))
    assert ceo_tools == [(ceo("escalation_resolve"), {"runId": RUN}), (ceo("run_create"), args)]


def _stale_fence() -> None:
    raise RuntimeError("the binding moved to another session")


@pytest.mark.parametrize("scope", [{"binding": "acp-canonical-cto"}, {"verify": _stale_fence},
                                   {"agent": SimpleNamespace()}])
def test_a_grant_for_another_binding_or_agent_or_with_a_broken_fence_allows_nothing(ceo_tools, scope):
    from agent.canonical_peer_tools import batch_allowed

    with peer_turn(**scope):
        assert not batch_allowed(AGENT, [(ceo("run_get"), {})])
        assert refused(call(ceo("run_get"), {}))
    with peer_turn():  # the same call under the right grant runs
        assert batch_allowed(AGENT, [(ceo("run_get"), {})]) and ran(call(ceo("run_get"), {}), ceo("run_get"))


def test_a_write_tool_of_an_untrusted_server_is_never_a_peer_tool(ceo_tools, monkeypatch):
    from agent.canonical_peer_tools import batch_allowed
    from tools import mcp_tool_handlers

    monkeypatch.setattr(mcp_tool_handlers, "_trust_gate_asks", lambda server, tool: tool == "escalation_resolve")
    with peer_turn():
        assert batch_allowed(AGENT, [(ceo("run_get"), {})])
        assert not batch_allowed(AGENT, [(ceo("escalation_resolve"), {"runId": RUN})])


def test_the_tool_call_wrapper_is_allowed_only_when_every_inner_call_is(ceo_tools):
    from agent.canonical_peer_tools import batch_allowed

    inner = {"name": ceo("run_get"), "arguments": {"runId": RUN}}
    with peer_turn() as grant:
        assert batch_allowed(AGENT, [("tool_call", inner)])
        assert list(grant.pins) == [(ceo("run_get"), json.dumps({"runId": RUN}, separators=(",", ":")))]
        assert not batch_allowed(AGENT, [("tool_call", {"name": ceo("owner_decision_submit"), "arguments": {}})])
        assert batch_allowed(AGENT, [("tool_describe", {"names": [ceo("run_get")]})])
        assert not batch_allowed(AGENT, [("tool_describe", {"names": [ceo("run_get"), ceo("ghost")]})])


def test_a_config_edit_applies_to_the_next_batch_without_a_restart(ceo_tools):
    from agent.canonical_peer_tools import batch_allowed

    with peer_turn():
        assert not batch_allowed(AGENT, [(ceo("owner_decision_submit"), {})])
        _write_config(_policy(read=[ceo("run_get"), ceo("owner_decision_submit")]))
        assert batch_allowed(AGENT, [(ceo("owner_decision_submit"), {})])
        _write_config(_policy(read=[]))
        assert not batch_allowed(AGENT, [(ceo("run_get"), {})])


# ── a prepared call is never re-coerced (counterexample 3) ───────────────


def test_a_prepared_call_runs_with_exactly_the_admitted_arguments(ceo_tools):
    from agent.canonical_peer_tools import batch_allowed, prepared_call

    admitted = {"runId": "123"}
    with peer_turn():
        emitted = tool_call(ceo("run_get"), admitted)
        assert batch_allowed(AGENT, [(ceo("run_get"), admitted)], emitted=[("c1", emitted)])
        assert prepared_call("c1", emitted).args == admitted
        # A refresh re-registers the tool with an integer runId: coercing again would send 123.
        _register(ceo("run_get"), SERVER, "run_get", run_id_type="integer")
        assert ran(call(ceo("run_get"), admitted), ceo("run_get"))
        assert prepared_call("c1") is None  # consumed by its dispatch
    assert ceo_tools == [(ceo("run_get"), {"runId": "123"})]
    ceo_tools.clear()
    assert ran(call(ceo("run_get"), admitted), ceo("run_get"))  # an owner call is still coerced
    assert ceo_tools == [(ceo("run_get"), {"runId": 123})]


def test_a_prepared_entry_runs_only_its_own_name(ceo_tools):
    from agent.canonical_peer_tools import batch_allowed

    with peer_turn():
        assert batch_allowed(AGENT, [(ceo("run_get"), {})], emitted=[("c1", tool_call(ceo("run_get")))])
        assert refused(call(ceo("owner_decision_submit"), {}))  # same call id, another tool
    assert ceo_tools == []


# ── the grant digest (counterexample 4) ──────────────────────────────────


def _pin_digests(grant) -> list:
    return [pin.digest for pins in grant.pins.values() for pin in pins]


def test_the_grant_digest_is_the_applied_entry_and_changes_when_the_entry_changes(ceo_tools):
    from agent.canonical_peer_tools import batch_allowed, peer_grant_digest

    first = {"mutate": {ceo("escalation_resolve"): {"runId": [RUN]}}}
    second = {"mutate": {ceo("escalation_resolve"): {"runId": [RUN, "run_second"]}}}
    assert peer_grant_digest(first) == digest(first) != peer_grant_digest(second) == digest(second)
    with peer_turn() as grant:
        assert batch_allowed(AGENT, [(ceo("escalation_resolve"), {"runId": RUN}), (ceo("run_get"), {})])
        assert _pin_digests(grant) == [digest(first), digest({"read": ceo("run_get")})]
        _write_config(_policy(mutate={ceo("escalation_resolve"): {"runId": [RUN, "run_second"]}}))
        assert batch_allowed(AGENT, [(ceo("escalation_resolve"), {"runId": RUN})])
        assert _pin_digests(grant) == [digest(second)]
