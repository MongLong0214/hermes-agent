"""The MCP handler side of a canonical peer turn: it sends only a call its admitted batch pinned.

Each pinned call is claimed once, by exactly the handler the batch resolved, and sent as the
pinned snapshot while its arguments still equal the checked ones. Anything else is refused and
nothing reaches the wire. Owner turns are unchanged. The tools are REAL MCP tool handlers
(``_make_tool_handler``) over a fake transport that records every call reaching it."""

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

# Pinned BEFORE any dependent Hermes import (see tests/agent/test_canonical_peer_tools.py): a run
# outside tests/ once let peer tool test helpers replace the user's live config.
_REAL_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir) if pwd is not None else Path.home()
_LIVE_CONFIG = (_REAL_HOME / ".hermes" / "config.yaml").resolve()
_PINNED_ROOT = Path(tempfile.mkdtemp(prefix="hermes-peer-mcp-pin-")).resolve()
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


import asyncio
import json
from contextlib import contextmanager
from types import SimpleNamespace

import hermes_yaml as yaml

SERVER = "acp-ceo"
BINDING = "acp-canonical-ceo"
RUN = "run_0000000000000000000000ab"
AGENT = SimpleNamespace(name="the agent the grant was issued to")


def ceo(tool: str) -> str:
    return f"mcp__acp_ceo__{tool}"


TOOLS = ["run_get", "escalation_resolve"]
POLICY = {"binding": BINDING, "mcp_server": SERVER, "read": [ceo("run_get")],
          "mutate": {ceo("escalation_resolve"): {"runId": [RUN]}}}


@pytest.fixture
def wire(monkeypatch, tmp_path):
    """Real handlers over a fake transport in this test's own tmp HERMES_HOME; yields the
    ``(tool, arguments object)`` pairs that reached the wire (``read_resource`` included)."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path))
    override = set_hermes_home_override(home)
    try:
        from hermes_cli.config import get_config_path

        config = get_config_path().resolve()
        if config == _LIVE_CONFIG or not config.is_relative_to(home.resolve()):
            pytest.fail(f"refusing to write a config outside the test's tmp HERMES_HOME: {config}")
        config.write_text(yaml.safe_dump({"gateway": {"canonical_peer_tools": POLICY}}), encoding="utf-8")
        yield from _wire(monkeypatch)
    finally:
        reset_hermes_home_override(override)


def _wire(monkeypatch):
    from tools import mcp_tool_discovery, mcp_tool_handlers, mcp_tool_loop
    from tools.mcp_tool_registration import _forget_mcp_tool_server, _track_mcp_tool_server
    from tools.registry import registry

    sent: list = []

    def call_tool(tool, arguments=None, **_):
        sent.append((ceo(tool), arguments))
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=f"ok:{ceo(tool)}")], isError=False)

    async def read_resource(uri):
        sent.append(("read_resource", uri))
        return SimpleNamespace(contents=[SimpleNamespace(text="resource body")])

    server = SimpleNamespace(session=SimpleNamespace(call_tool=call_tool, read_resource=read_resource),
                             _rpc_lock=asyncio.Lock())
    monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", lambda name, timeout: (server, None))
    monkeypatch.setattr(mcp_tool_discovery, "_get_connected_server_for_call", lambda name: server)
    monkeypatch.setattr(mcp_tool_loop, "_run_on_mcp_loop", lambda call, timeout=30: asyncio.run(call()))
    for tool in TOOLS:
        registry.deregister(ceo(tool))
        registry.register(ceo(tool), f"mcp-{SERVER}", {"name": ceo(tool), "description": tool, "parameters": {
            "type": "object", "properties": {"runId": {"type": "string"}}}},
            mcp_tool_handlers._make_tool_handler(SERVER, tool, 30))
        _track_mcp_tool_server(ceo(tool), SERVER)
    try:
        yield sent
    finally:
        for tool in TOOLS:
            registry.deregister(ceo(tool))
            _forget_mcp_tool_server(ceo(tool))


@contextmanager
def peer_turn(*, verify=lambda: None, grant=True):
    """What the verified canonical ingress binds for one peer turn."""
    from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
    from tools.approval_context import reset_turn_principal, set_turn_principal

    principal = set_turn_principal("peer")
    bound = PeerToolGrant(binding=BINDING, agent=AGENT, verify=verify) if grant else None
    token = bind_grant(bound) if grant else None
    try:
        yield bound
    finally:
        if token is not None:
            reset_grant(token)
        reset_turn_principal(principal)


def admit(*calls: tuple) -> bool:
    from agent.canonical_peer_tools import batch_allowed

    return batch_allowed(AGENT, list(calls))


def direct(name: str, args: dict, handler=None) -> str:
    """Call the MCP handler itself, as a caller outside the executors' pin lookup would."""
    from tools.registry import registry

    return (handler or registry.get_entry(name).handler)(args)


def ran(result: str, name: str) -> bool:
    return f"ok:{name}" in result


def not_sent(result: str) -> bool:
    return "was NOT run: in a canonical peer turn only the exact call" in result


def _middleware(monkeypatch, rewrite) -> None:
    """A plugin's tool execution middleware that hands the next layer ``rewrite(args)``."""
    from hermes_cli.middleware import TOOL_EXECUTION_MIDDLEWARE
    from hermes_cli.plugins import _delivery_manager

    def middleware(args, next_call):
        return next_call(rewrite(args))

    monkeypatch.setitem(_delivery_manager()._middleware, TOOL_EXECUTION_MIDDLEWARE, [middleware])


def _dispatch_prepared(name: str, args: dict, call_id: str = "c1") -> str:
    """One prepared peer call through the real dispatcher (``model_tools.handle_function_call``)."""
    import model_tools
    from agent.canonical_peer_tools import batch_allowed

    emitted = SimpleNamespace(id=call_id, type="function",
                              function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    assert batch_allowed(AGENT, [(name, args)], emitted=[(call_id, emitted)])
    return model_tools.handle_function_call(name, dict(args), task_id="t", tool_call_id=call_id)


# ── a pinned call is sent once ───────────────────────────────────────────


def test_a_pinned_call_is_sent_once_and_a_second_send_of_the_same_pin_is_refused(wire):
    with peer_turn() as grant:
        assert admit((ceo("run_get"), {"runId": RUN}))
        snapshot = grant.pins[(ceo("run_get"), json.dumps({"runId": RUN}, separators=(",", ":")))][0].snapshot
        caller_args = {"runId": RUN}
        assert ran(direct(ceo("run_get"), caller_args), ceo("run_get"))
        assert not_sent(direct(ceo("run_get"), {"runId": RUN}))
        assert not any(grant.pins.values())
        assert [a["tool"] for a in grant.attempted] == [ceo("run_get")]
    assert wire == [(ceo("run_get"), {"runId": RUN})]
    assert wire[0][1] is snapshot and wire[0][1] is not caller_args  # the owned snapshot, never the caller's dict


def test_a_call_pinned_twice_by_its_batch_is_sent_exactly_twice(wire):
    with peer_turn():
        assert admit((ceo("run_get"), {"runId": RUN}), (ceo("run_get"), {"runId": RUN}))
        assert ran(direct(ceo("run_get"), {"runId": RUN}), ceo("run_get"))
        assert ran(direct(ceo("run_get"), {"runId": RUN}), ceo("run_get"))
        assert not_sent(direct(ceo("run_get"), {"runId": RUN}))
    assert wire == [(ceo("run_get"), {"runId": RUN})] * 2


# ── arguments changed after the pin ──────────────────────────────────────


REWRITES = {
    "a pinned value": lambda a: {**a, "runId": "run_other"},
    "a model or hook supplied _meta": lambda a: {**a, "_meta": {"progressToken": "forged", "principal": "owner"}},
    "an extra argument": lambda a: {**a, "dryRun": False},
    "in place": lambda a: (a.update(runId="run_other"), a)[1],
}


@pytest.mark.parametrize("tool", ["run_get", "escalation_resolve"])
@pytest.mark.parametrize("rewrite", list(REWRITES))
def test_arguments_a_middleware_changes_after_the_pin_are_refused_and_nothing_is_sent(wire, monkeypatch, tool, rewrite):
    with peer_turn():
        assert ran(_dispatch_prepared(ceo(tool), {"runId": RUN}), ceo(tool))  # the same call, unchanged, runs
        _middleware(monkeypatch, REWRITES[rewrite])
        assert not_sent(_dispatch_prepared(ceo(tool), {"runId": RUN}, call_id="c2"))
    assert wire == [(ceo(tool), {"runId": RUN})]


@pytest.mark.parametrize("change", ["an alias the caller kept", "the fence"])
def test_a_change_between_the_claim_and_the_send_is_refused_and_its_charge_refunded(wire, monkeypatch, change):
    from tools import mcp_tool_handlers

    caller_args, fence = {"runId": RUN}, {"held": True}
    refunds: list = []
    charge = SimpleNamespace(refund=lambda: refunds.append(1))
    monkeypatch.setattr(mcp_tool_handlers, "_trust_gate_check",
                        lambda server, tool, booked=None, args=None: booked.append(charge))
    acquire = mcp_tool_handlers._acquire_call_server

    def acquire_then_change(name, timeout):  # runs after the claim, before the send
        if change == "the fence":
            fence["held"] = False
        else:
            caller_args["runId"] = "run_other"
        return acquire(name, timeout)

    def verify():
        if not fence["held"]:
            raise RuntimeError("the binding moved to another session")

    with peer_turn(verify=verify):
        assert admit((ceo("run_get"), {"runId": RUN}))
        assert ran(direct(ceo("run_get"), {"runId": RUN}), ceo("run_get"))  # the same call, unchanged, runs
        assert admit((ceo("run_get"), {"runId": RUN}))
        monkeypatch.setattr(mcp_tool_handlers, "_acquire_call_server", acquire_then_change)
        assert not_sent(direct(ceo("run_get"), caller_args))
    assert wire == [(ceo("run_get"), {"runId": RUN})]
    assert refunds == [1]  # booked for the refused call only


# ── an unpinned call ─────────────────────────────────────────────────────


def _other_handler():
    from tools.mcp_tool_handlers import _make_tool_handler

    return _make_tool_handler(SERVER, "run_get", 30)  # the same tool, but not the handler the batch resolved


UNPINNED = {
    "no batch admitted": lambda: direct(ceo("run_get"), {"runId": RUN}),
    "other arguments": lambda: direct(ceo("run_get"), {"runId": "run_other"}),
    "another tool": lambda: direct(ceo("escalation_resolve"), {"runId": RUN}),
    "another handler": lambda: direct(ceo("run_get"), {"runId": RUN}, handler=_other_handler()),
}


@pytest.mark.parametrize("case", list(UNPINNED))
def test_an_unpinned_mcp_call_in_a_peer_turn_is_refused_and_nothing_is_sent(wire, case):
    with peer_turn():
        if case != "no batch admitted":
            assert admit((ceo("run_get"), {"runId": RUN}))
        assert not_sent(UNPINNED[case]())
        if case != "no batch admitted":
            assert ran(direct(ceo("run_get"), {"runId": RUN}), ceo("run_get"))  # its pinned neighbour runs
    assert wire == ([] if case == "no batch admitted" else [(ceo("run_get"), {"runId": RUN})])


def test_a_peer_turn_without_a_grant_sends_no_mcp_call(wire):
    with peer_turn(grant=False):
        assert not_sent(direct(ceo("run_get"), {"runId": RUN}))
    assert wire == []


def test_an_mcp_utility_is_never_sent_in_a_peer_turn(wire):
    from tools.mcp_tool_handlers import _make_read_resource_handler

    read = _make_read_resource_handler(SERVER, 30)
    with peer_turn():
        assert admit((ceo("run_get"), {"runId": RUN}))
        assert not_sent(read({"uri": "acp://runs"}))
    assert json.loads(read({"uri": "acp://runs"})) == {"result": "resource body"}  # owner turn: unchanged
    assert wire == [("read_resource", "acp://runs")]


# ── owner turns are unchanged (control) ──────────────────────────────────


def test_owner_turn_mcp_calls_are_sent_as_given_every_time(wire, monkeypatch):
    import model_tools

    args = {"runId": "anything", "_meta": {"progressToken": 7}}
    assert ran(direct(ceo("escalation_resolve"), args), ceo("escalation_resolve"))
    assert ran(direct(ceo("escalation_resolve"), args), ceo("escalation_resolve"))
    _middleware(monkeypatch, REWRITES["a pinned value"])
    assert ran(model_tools.handle_function_call(ceo("run_get"), {"runId": RUN}, task_id="t", tool_call_id="c9"),
               ceo("run_get"))
    assert wire == [(ceo("escalation_resolve"), args), (ceo("escalation_resolve"), args),
                    (ceo("run_get"), {"runId": "run_other"})]
    assert wire[0][1] is args  # an owner call sends the caller's own arguments
