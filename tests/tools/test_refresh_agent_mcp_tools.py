"""Tests for the shared MCP agent-tool refresh helper and discovery-wait bound.

``refresh_agent_mcp_tools`` is the single rebuild path used by the TUI
``reload.mcp`` RPC, the gateway reload, and the late-binding refresh thread —
so a slow MCP server that connects after the agent's one-time tool snapshot is
picked up everywhere identically.  These assert the *contracts* those callers
rely on (name-based diff, in-place mutation, agent-scoped filtering) rather than
freezing any particular tool list.
"""

import json
import threading
import types

import pytest

from tools import mcp_tool
from tools import mcp_tool_agent as _mcp_agent


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _agent(tool_names, *, enabled=None, disabled=None):
    a = types.SimpleNamespace()
    a.tools = [_tool(n) for n in tool_names]
    a.valid_tool_names = set(tool_names)
    a.enabled_toolsets = enabled
    a.disabled_toolsets = disabled
    return a


def test_refresh_adds_late_landing_tools(monkeypatch):
    """A server that registers after build → its tools land in the snapshot."""
    agent = _agent(["read_file", "terminal"])

    new_defs = [_tool(n) for n in ("read_file", "terminal", "mcp_granola_get_account_info")]
    monkeypatch.setattr(mcp_tool, "get_tool_definitions", lambda **kw: new_defs, raising=False)
    # get_tool_definitions is imported inside the helper from model_tools, so patch there too.
    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: new_defs)

    added = _mcp_agent.refresh_agent_mcp_tools(agent)

    assert added == {"mcp_granola_get_account_info"}
    assert "mcp_granola_get_account_info" in agent.valid_tool_names
    assert len(agent.tools) == 3

    side = _agent(["read_file", "terminal"])
    side.side_agent = True
    monkeypatch.setattr(model_tools, "get_tool_definitions",
                        lambda **kw: new_defs + [_tool("manage_connections")])

    _mcp_agent.refresh_agent_mcp_tools(side)

    assert "manage_connections" not in side.valid_tool_names
    assert "manage_connections" not in [t["function"]["name"] for t in side.tools]


def test_refresh_preserves_memory_provider_and_context_engine_tools(monkeypatch):
    """B1 regression: a rebuild must NOT drop post-build-injected tools.

    get_tool_definitions() returns only the registry-derived tools. agent_init
    appends memory-provider tools (mem0/honcho/…) and context-engine tools
    (lcm_*) directly onto agent.tools AFTER that. A naive
    `agent.tools = get_tool_definitions()` would silently delete them on every
    refresh. The helper must re-inject them.
    """
    # Agent already carries: a built-in, a memory-provider tool, a context tool.
    agent = _agent(["read_file", "memory_search", "lcm_grep"])

    # Provider exposes its schemas; context compressor exposes lcm_*.
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "memory_search", "description": "", "parameters": {}}
        ]
    )
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [
            {"name": "lcm_grep", "description": "", "parameters": {}}
        ]
    )
    agent._context_engine_tool_names = {"lcm_grep"}

    import model_tools
    # The registry now ALSO has a newly-connected MCP tool, but does NOT contain
    # the memory/context tools (they're never in get_tool_definitions output).
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_server_tool")],
    )

    added = _mcp_agent.refresh_agent_mcp_tools(agent)

    # The new MCP tool landed AND the injected families survived.
    assert "mcp_new_server_tool" in agent.valid_tool_names
    assert "memory_search" in agent.valid_tool_names   # not clobbered
    assert "lcm_grep" in agent.valid_tool_names         # not clobbered
    assert added == {"mcp_new_server_tool"}


def test_refresh_does_not_reinject_disabled_memory_provider_tools(monkeypatch):
    """A refresh removes stale provider tools when memory becomes disabled."""
    agent = _agent(
        ["read_file", "memory_search"],
        enabled=["all"],
        disabled=["memory"],
    )
    agent._memory_manager = types.SimpleNamespace(
        get_all_tool_schemas=lambda: [
            {"name": "memory_search", "description": "", "parameters": {}}
        ]
    )

    import model_tools
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **kw: [_tool("read_file")],
    )

    _mcp_agent.refresh_agent_mcp_tools(agent)

    assert "memory_search" not in agent.valid_tool_names
    assert all(t["function"]["name"] != "memory_search" for t in agent.tools)


def test_refresh_respects_context_engine_toolset_gate(monkeypatch):
    """#5544: context-engine tools must NOT be re-injected on a restricted
    toolset. A platform with enabled_toolsets that excludes context_engine
    must not get lcm_* leaked back in by a refresh."""
    agent = _agent(["read_file"], enabled=["coding"])  # context_engine NOT enabled
    agent.context_compressor = types.SimpleNamespace(
        get_tool_schemas=lambda: [{"name": "lcm_grep", "description": "", "parameters": {}}]
    )
    agent._context_engine_tool_names = set()

    import model_tools
    monkeypatch.setattr(
        model_tools, "get_tool_definitions",
        lambda **kw: [_tool("read_file"), _tool("mcp_new_tool")],
    )

    _mcp_agent.refresh_agent_mcp_tools(agent)

    assert "mcp_new_tool" in agent.valid_tool_names  # MCP tool still lands
    assert "lcm_grep" not in agent.valid_tool_names   # gated out (#5544)




def test_refresh_is_thread_safe_under_concurrent_calls(monkeypatch):
    """Concurrent refreshes keep tools / valid_tool_names coherent.

    The registry alternates between two DIFFERENT tool sets every call, so the
    write path (publish) runs repeatedly rather than short-circuiting on the
    no-change early return — this actually exercises the lock. The invariant:
    a reader of ``valid_tool_names`` must always match ``agent.tools``, and the
    final published pair must be one of the two valid sets (never a mix).
    """
    agent = _agent(["a"])

    import itertools
    set_a = [_tool("a"), _tool("b")]
    set_b = [_tool("a"), _tool("c")]
    flip = itertools.cycle([set_a, set_b])
    flip_lock = threading.Lock()

    def _gtd(**kw):
        with flip_lock:
            return list(next(flip))

    import model_tools
    monkeypatch.setattr(model_tools, "get_tool_definitions", _gtd)

    errors = []

    def _worker():
        try:
            for _ in range(50):
                _mcp_agent.refresh_agent_mcp_tools(agent)
                # Coherence invariant: the name set must match the tool list
                # at every observation, never a torn cross-attribute state.
                names = {t["function"]["name"] for t in agent.tools}
                assert agent.valid_tool_names == names
                assert names in ({"a", "b"}, {"a", "c"})
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert agent.valid_tool_names in ({"a", "b"}, {"a", "c"})


# ── discovery-wait bound (mcp_discovery_timeout config) ──────────────────────




def test_wait_returns_instantly_when_no_discovery_thread(monkeypatch):
    """The common case (no MCP / discovery done) pays ~0s regardless of bound."""
    import time
    from hermes_cli import mcp_startup

    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", {})
    import hermes_cli.config as cfg
    monkeypatch.setattr(cfg, "load_config", lambda: {"mcp_discovery_timeout": 999.0})

    t0 = time.time()
    mcp_startup.wait_for_mcp_discovery()
    assert time.time() - t0 < 0.2  # never blocks on the bound when nothing's pending


# ---------------------------------------------------------------------------
# preserve_prefix: the tool array is a cached request prefix (#100336)
# ---------------------------------------------------------------------------


def _registered(monkeypatch, names):
    """Make the registry report exactly *names* as still registered."""
    from tools import registry as registry_mod

    entries = [types.SimpleNamespace(name=n) for n in names]
    monkeypatch.setattr(
        registry_mod.registry, "get_all_entries", lambda: entries, raising=False
    )


def _serve(monkeypatch, defs):
    import model_tools

    monkeypatch.setattr(model_tools, "get_tool_definitions", lambda **kw: list(defs))


def test_preserve_prefix_carries_a_flapping_tool_forward(monkeypatch):
    """A check_fn flip must not shrink a live session's tool prefix.

    ``browser_navigate``'s availability probe fails this turn (headless box,
    expired credential, docker blip) so ``get_tool_definitions`` omits it. The
    tool is still *registered* — only its probe flapped — so the snapshot must
    keep it, byte-for-byte, instead of forking the cached prefix.
    """
    agent = _agent(["read_file", "browser_navigate", "terminal"])
    before = list(agent.tools)

    _serve(monkeypatch, [_tool("read_file"), _tool("terminal")])
    _registered(monkeypatch, ["read_file", "browser_navigate", "terminal"])

    added = _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert added == set()
    assert agent.tools == before
    assert "browser_navigate" in agent.valid_tool_names


def test_preserve_prefix_appends_late_arrivals_at_the_tail(monkeypatch):
    """``get_definitions`` sorts by name, so a late tool can splice in at 0.

    Under ``preserve_prefix`` the live order is authoritative and the new tool
    extends the array, leaving every earlier byte where the provider cached it.
    """
    agent = _agent(["read_file", "terminal"])

    # Sorted order would put the new tool first.
    _serve(monkeypatch, [_tool("aaa_mcp_late"), _tool("read_file"), _tool("terminal")])
    _registered(monkeypatch, ["aaa_mcp_late", "read_file", "terminal"])

    added = _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert added == {"aaa_mcp_late"}
    assert [t["function"]["name"] for t in agent.tools] == [
        "read_file", "terminal", "aaa_mcp_late",
    ]


def test_preserve_prefix_keeps_sent_tool_schemas_byte_identical(monkeypatch):
    """A between-turns refresh must not rewrite a tool this session already sent.

    The chat template prints every tool schema before the conversation. A local
    server reuses the previous prompt only when the next one starts with those
    exact tokens, so a new description on a tool that was already sent makes the
    follow-up read the whole prompt again. New tools may append at the tail.
    """
    agent = _agent(["read_file"])
    agent.tools[0]["function"]["description"] = "Read a file. schema v1"
    sent = json.dumps(agent.tools)

    fresh = _tool("read_file")
    fresh["function"]["description"] = "Read a file. schema v2"
    fresh["function"]["parameters"] = {
        "type": "object", "properties": {"path": {"type": "string"}},
    }
    _serve(monkeypatch, [fresh, _tool("mcp_late")])
    _registered(monkeypatch, ["read_file", "mcp_late"])

    _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert json.dumps(agent.tools[:1]) == sent
    assert [t["function"]["name"] for t in agent.tools] == ["read_file", "mcp_late"]


def test_explicit_reload_replaces_a_same_name_schema(monkeypatch):
    """``/reload-mcp`` takes the fresh schema when the tool name stays the same.

    Between-turns refresh freezes the bytes already sent. Reload is the other
    path (``preserve_prefix`` off): the user already accepted a cache break, so
    a new description on ``read_file`` has to be what the next request sends.
    """
    agent = _agent(["read_file"])
    agent.tools[0]["function"]["description"] = "Read a file. schema v1"

    fresh = _tool("read_file")
    fresh["function"]["description"] = "Read a file. schema v2"
    _serve(monkeypatch, [fresh])
    _registered(monkeypatch, ["read_file"])

    _mcp_agent.refresh_agent_mcp_tools(agent)

    assert agent.tools[0]["function"]["description"] == "Read a file. schema v2"


def test_preserve_prefix_keeps_the_bridge_tools_byte_identical(monkeypatch):
    """``tool_search``'s description is derived from the session at build time: the
    deferred-tool count, the embedded listing, and whether ``manage_connections`` was
    present. Every one of those inputs can move between turns (a late MCP server, a
    ``check_fn`` flap on a portal blip), and a moved byte in the tool array re-prefills
    the whole cached history. The refresh must leave the bridge entries exactly as
    built; a search still reads the live catalog at dispatch."""
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES

    built = _tool("tool_search")
    built["function"]["description"] = "Search 21 additional tools. connectors__ hint present."
    agent = _agent(["read_file", "manage_connections"])
    agent.tools.append(built)
    agent.valid_tool_names.add("tool_search")
    before = json.dumps(agent.tools, sort_keys=True)

    fresh_bridge = _tool("tool_search")
    fresh_bridge["function"]["description"] = "Search 33 additional tools."
    # manage_connections flapped out (portal blip); a late server grew the count.
    _serve(monkeypatch, [_tool("read_file"), fresh_bridge, _tool("mcp_late_tool")])
    _registered(monkeypatch, ["read_file", "manage_connections", "mcp_late_tool", *BRIDGE_TOOL_NAMES])

    added = _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert added == {"mcp_late_tool"}
    assert json.dumps(agent.tools[:3], sort_keys=True) == before
    assert [t["function"]["name"] for t in agent.tools][-1] == "mcp_late_tool"


def test_preserve_prefix_keeps_a_deactivated_bridge_tool_in_place(monkeypatch):
    """The bridge tools are synthesized by ``assemble_tool_defs``, never registered. When the
    deferred set shrinks under the activation threshold (an MCP server drops), the fresh
    snapshot carries no ``tool_search`` at all; the sent one must still hold its slot, or
    every tool after it moves and the prefix re-prefills. Search reads the live catalog."""
    built = _tool("tool_search")
    built["function"]["description"] = "Search 21 additional tools."
    agent = _agent(["read_file"])
    agent.tools.append(built)
    agent.tools.append(_tool("terminal"))
    agent.valid_tool_names.update({"tool_search", "terminal"})
    before = json.dumps(agent.tools, sort_keys=True)

    _serve(monkeypatch, [_tool("read_file"), _tool("terminal")])
    _registered(monkeypatch, ["read_file", "terminal"])

    _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert json.dumps(agent.tools, sort_keys=True) == before


# ---------------------------------------------------------------------------
# tools[] freeze: eviction rebuild + the /reload-mcp re-probe hatch
# ---------------------------------------------------------------------------


def test_eviction_rebuild_restores_the_sessions_saved_tool_order(monkeypatch):
    """A fresh AIAgent for an EXISTING session must keep the saved tools[] pin.

    Gateway agent-cache eviction rebuilds the agent; ``agent_init`` re-probes
    every ``check_fn`` and ``browser_exec``'s flips false. The persisted
    name list stands in for the missing predecessor: the tool is carried
    forward from the registry schema, byte-for-byte in its old slot.
    """
    from tools import registry as registry_mod

    saved = ["read_file", "browser_exec", "terminal"]
    entries = {n: types.SimpleNamespace(name=n, schema=_tool(n)["function"]) for n in saved}
    monkeypatch.setattr(registry_mod.registry, "get_all_entries", lambda: list(entries.values()), raising=False)
    monkeypatch.setattr(registry_mod.registry, "get_entry", lambda name, **kw: entries.get(name), raising=False)

    rebuilt = _agent(["read_file", "terminal"])  # probe flipped: browser_exec gone
    changed = _mcp_agent.restore_agent_tool_prefix(rebuilt, saved)

    assert changed is True
    assert [t["function"]["name"] for t in rebuilt.tools] == saved
    assert rebuilt.valid_tool_names == set(saved)


def test_resume_on_another_surface_restores_the_pinned_tool_bytes(monkeypatch, tmp_path):
    """One durable session hops gateway -> ``-q --resume``: the new process derives different
    bytes for the SAME tools (tool_search's per-surface deferred catalog, per-surface dynamic
    PARAMETERS like delegate_task's, the one-shot footprint pruning skill_manage). tools[] heads
    every request, so a pin written by the same code hands back exactly what the session sent;
    one written by other code (``hermes update``) takes the current definitions instead."""
    from hermes_state import SessionDB
    from tools import registry as registry_mod

    def _described(name, description, **params):
        tool = _tool(name)
        tool["function"]["description"] = description
        tool["function"]["parameters"] = {"type": "object", "properties": params}
        return tool

    sent = _agent([])
    sent.tools = [_tool("read_file"), _described("delegate_task", "delegate", group={"type": "string"}),
                  _described("skill_manage", "lands in /home/u/.hermes/skills"),
                  _described("tool_search", "Search 6 additional tools.")]
    static = {"skill_manage": _described("skill_manage", "lands in the profile's skills dir")["function"]}
    monkeypatch.setattr(registry_mod.registry, "get_all_entries",
                        lambda: [types.SimpleNamespace(name=n) for n in ("read_file", "delegate_task", "skill_manage")],
                        raising=False)
    monkeypatch.setattr(registry_mod.registry, "get_entry",
                        lambda name, **kw: types.SimpleNamespace(name=name, schema=static[name]), raising=False)
    this_surface = [_tool("read_file"), _described("delegate_task", "delegate"),  # drops `group` here
                    _described("tool_search", "Search 5 additional tools.")]
    with SessionDB(db_path=tmp_path / "state.db") as db:
        sent._session_db = db
        for sid in ("s1", "s2"):
            db.create_session(sid, source="tui")
            sent.session_id = sid
            _mcp_agent.persist_agent_tool_names(sent)
        # Stored once, like the system prompt: a ~50KB array per session row would bloat state.db.
        stored = db._conn.execute("SELECT COUNT(*) FROM system_prompts").fetchone()[0]

        resumed = _agent([])
        resumed.tools, resumed._session_db, resumed.session_id = list(this_surface), db, "s1"
        _mcp_agent.restore_agent_tool_prefix(resumed, json.loads(db.get_session("s1")["tool_names"]))
        repinned = db.get_session("s1")["tool_names"]

        # The pin came from other code: every tool built here takes this build's definition.
        monkeypatch.setattr(_mcp_agent, "tool_pin_version", lambda: "sha-after-hermes-update")
        updated = _agent([])
        updated.tools, updated._session_db, updated.session_id = list(this_surface), db, "s2"
        _mcp_agent.restore_agent_tool_prefix(updated, json.loads(db.get_session("s2")["tool_names"]))
        upgraded_pin = json.loads(db.get_session("s2")["tool_names"])

    assert json.dumps(resumed.tools) == json.dumps(sent.tools)
    assert resumed.valid_tool_names == {"read_file", "delegate_task", "skill_manage", "tool_search"}
    assert stored == 1
    assert json.loads(repinned)["tools"] == sent.tools  # unchanged pin, no rewrite per hop
    assert updated.tools == [*this_surface[:2], {"type": "function", "function": {**static["skill_manage"]}},
                             this_surface[2]]
    assert upgraded_pin == {"version": "sha-after-hermes-update", "tools": updated.tools}


def test_a_pin_never_re_adds_a_tool_this_sessions_config_excludes(monkeypatch):
    """A pin from a surface where ``terminal`` was allowed must not hand it back where config
    disables it, nor ``browser_exec`` (host Python) once ``terminal`` is gone. A client-surface
    tool (``focus_pane``) is still carried: no config choice removed it here."""
    import model_tools

    monkeypatch.setattr(_mcp_agent, "persist_agent_tool_names", lambda agent: None)
    pin = {"version": _mcp_agent.tool_pin_version(),
           "tools": [_tool(n) for n in ("read_file", "terminal", "browser_exec", "focus_pane")]}
    agent = _agent(["read_file"], enabled=["hermes-cli"], disabled=["terminal"])

    _mcp_agent.restore_agent_tool_prefix(agent, pin)

    assert [t["function"]["name"] for t in agent.tools] == ["read_file", "focus_pane"]
    assert agent.valid_tool_names == {"read_file", "focus_pane"}


def test_reprobe_tool_availability_drops_cached_check_fn_verdicts(monkeypatch):
    """/reload-mcp is the explicit hatch: a cached False must be re-probed."""
    from tools import registry as registry_mod
    import model_tools

    verdict = {"ok": False}

    def probe():
        return verdict["ok"]

    monkeypatch.setattr(registry_mod, "check_fn_cache_scope", lambda: "test-scope")
    assert registry_mod._check_fn_cached(probe) is False
    verdict["ok"] = True
    assert registry_mod._check_fn_cached(probe) is False  # TTL cache replays stale verdict
    with model_tools._tool_defs_cache_lock:
        model_tools._tool_defs_cache[("sentinel",)] = []

    _mcp_agent.reprobe_tool_availability()

    assert registry_mod._check_fn_cached(probe) is True
    assert ("sentinel",) not in model_tools._tool_defs_cache


# ---------------------------------------------------------------------------
# Bot Mode dynamic capability: every snapshot rebuild re-runs its auth gate
# ---------------------------------------------------------------------------


class _BotModeDB:
    def __init__(self, home, title):
        self.db_path = str(home / "state.db")
        self._title = title

    def get_session_title(self, _session_id):
        return self._title


@pytest.fixture
def managed_bot_home(tmp_path):
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "researcher"
    profile.mkdir(parents=True)
    (profile / "profile.yaml").write_text(
        "ui_meta:\n  hermes-bots:\n    shape: cloud\n",
        encoding="utf-8",
    )
    return home


def _bot_mode_agent(home, *, title="Bot Chat"):
    agent = _agent(["read_file"])
    agent._session_db = _BotModeDB(home, title)
    agent.session_id = "session-1"
    agent._session_title_hint = None
    agent._bot_mode_protocol = True
    return agent


def _message_agent_schema_count(agent):
    return sum(
        t.get("function", {}).get("name") == "message_agent"
        for t in agent.tools
        if isinstance(t, dict)
    )


def _assert_tool_snapshot_coherent(agent):
    names = {t["function"]["name"] for t in agent.tools}
    assert agent.valid_tool_names == names


@pytest.mark.parametrize("rebuild", ["compaction", "reload", "between_turns", "resume"])
def test_authorized_message_agent_survives_every_snapshot_rebuild(
    managed_bot_home, monkeypatch, rebuild
):
    """Compaction, live refreshes and eviction/resume all preserve the guarded tool."""
    from tools.bot_mode_dm import ensure_message_agent_tool
    from tools import registry as registry_mod

    agent = _bot_mode_agent(managed_bot_home)
    _serve(monkeypatch, [_tool("read_file")])
    entry = types.SimpleNamespace(name="read_file", schema=_tool("read_file")["function"])
    monkeypatch.setattr(registry_mod.registry, "get_all_entries", lambda: [entry], raising=False)
    monkeypatch.setattr(
        registry_mod.registry,
        "get_entry",
        lambda name, **_kw: entry if name == "read_file" else None,
        raising=False,
    )

    assert ensure_message_agent_tool(agent) is True

    def rebuild_snapshot():
        if rebuild == "resume":
            agent.tools = [_tool("read_file")]
            agent.valid_tool_names = {"read_file"}
            _mcp_agent.restore_agent_tool_prefix(agent, ["read_file", "message_agent"])
        else:
            _mcp_agent.refresh_agent_mcp_tools(
                agent,
                content_aware=rebuild == "compaction",
                preserve_prefix=rebuild == "between_turns",
            )

    for _ in range(2):
        rebuild_snapshot()
        assert _message_agent_schema_count(agent) == 1
        assert "message_agent" in agent.valid_tool_names
        _assert_tool_snapshot_coherent(agent)


@pytest.mark.parametrize(
    ("title", "managed"),
    [("Ordinary chat", True), ("Bot Chat", False)],
)
def test_snapshot_rebuild_never_grants_message_agent_to_unauthorized_sessions(
    tmp_path, managed_bot_home, monkeypatch, title, managed
):
    """Ordinary and unmanaged chats remain fail-closed across repeated rebuilds."""
    home = managed_bot_home if managed else tmp_path / "unmanaged"
    home.mkdir(exist_ok=True)
    agent = _bot_mode_agent(home, title=title)
    # Even a stale/leaked dynamic capability is scrubbed unless the live gate re-authorizes it.
    agent.tools.append(_tool("message_agent"))
    agent.valid_tool_names.add("message_agent")
    _serve(monkeypatch, [_tool("read_file")])

    for _ in range(2):
        _mcp_agent.refresh_agent_mcp_tools(agent, content_aware=True)
        assert _message_agent_schema_count(agent) == 0
        assert "message_agent" not in agent.valid_tool_names
        _assert_tool_snapshot_coherent(agent)


# ---------------------------------------------------------------------------
# Deferred listing growth: a late MCP server must reach tool_search's listing
# ---------------------------------------------------------------------------


@pytest.fixture
def listing_servers(monkeypatch):
    """Group ``mcp__<server>__<tool>`` names by server, as the registry does for MCP toolsets, and
    let a test declare servers that are configured but unavailable."""
    from tools import tool_search_catalog as catalog

    monkeypatch.setattr(catalog, "_classify_source", lambda name: ("mcp", "mcp-" + name.split("__")[1]))
    hidden = []
    monkeypatch.setattr(catalog, "hidden_declared_sources", lambda: list(hidden))
    return hidden


def _render_bridge(hidden, servers, *, unavailable=(), max_tokens=4000):
    """The ``tool_search`` def the real assembler renders for ``{server: tool_count}``."""
    from tools import tool_search_catalog as catalog
    from tools.tool_search import bridge_tool_schemas

    hidden[:] = [{"name": n, "tool_count": 50, "unavailable": "probe failed"} for n in unavailable]
    deferred = [{"type": "function", "function": {
        "name": f"mcp__{server}__tool{i:02d}", "description": "Does a, b, and c: then stops. More text."}}
        for server, count in servers.items() for i in range(count)]
    listing, form = catalog.build_catalog_listing_with_form(deferred, max_tokens=max_tokens)
    return bridge_tool_schemas(len(deferred), listing=listing, listing_form=form)[0], form


def _connect_server(monkeypatch, fresh_bridge):
    """A server registering its tools moves the registry generation, so tools[] is not frozen."""
    from tools import registry as registry_mod
    from tools.tool_search_catalog import BRIDGE_TOOL_NAMES

    monkeypatch.setattr(registry_mod.registry, "_generation", registry_mod.registry._generation + 1)
    _registered(monkeypatch, ["read_file", *BRIDGE_TOOL_NAMES])
    _serve(monkeypatch, [_tool("read_file"), fresh_bridge])


def _bridge_agent(built):
    agent = _agent(["read_file"])
    agent.tools.append(built)
    agent.valid_tool_names.add("tool_search")
    return agent


@pytest.mark.parametrize("late", [{"acp": 1}, {"acp": 3}], ids=["one-tool-server", "multi-tool-server"])
def test_a_server_connected_after_turn_start_reaches_the_bridge_listing(monkeypatch, listing_servers, late):
    """The listing is the only place a deferred tool is named to the model. A one-tool server
    renders as a bare line without its group heading, so it must count as growth too."""
    built, _ = _render_bridge(listing_servers, {"commitlore": 2})
    grown, _ = _render_bridge(listing_servers, {"commitlore": 2, **late})
    agent = _bridge_agent(built)
    _connect_server(monkeypatch, grown)

    _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert agent.tools == [_tool("read_file"), grown]
    assert "mcp__acp__tool00" in agent.tools[1]["function"]["description"]


def test_an_unchanged_or_shrunk_listing_keeps_the_sent_bridge_bytes(monkeypatch, listing_servers):
    built, _ = _render_bridge(listing_servers, {"commitlore": 2, "acp": 1})
    agent = _bridge_agent(built)
    for fresh_servers in ({"commitlore": 2, "acp": 1}, {"commitlore": 1, "acp": 1}, {"commitlore": 2}):
        fresh, _ = _render_bridge(listing_servers, fresh_servers)
        # A recount alone (the deferred-count sentence) moves the description, not the listing.
        fresh["function"]["description"] = fresh["function"]["description"].replace("Search ", "Search  ", 1)
        _connect_server(monkeypatch, fresh)
        assert _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True) == set()
        assert agent.tools[1] is built


def test_a_listing_grown_only_by_a_check_fn_flip_stays_frozen(monkeypatch, listing_servers):
    """Registry generation and selection unchanged: only a check_fn flip can grow the listing,
    and tools[] stays frozen for that (search reads the live catalog at dispatch)."""
    built, _ = _render_bridge(listing_servers, {"commitlore": 2})
    grown, _ = _render_bridge(listing_servers, {"commitlore": 2, "acp": 1})
    agent = _bridge_agent(built)
    _connect_server(monkeypatch, built)
    _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)  # records generation + selection
    _serve(monkeypatch, [_tool("read_file"), grown])

    _mcp_agent.refresh_agent_mcp_tools(agent, preserve_prefix=True)

    assert agent.tools[1] is built


def test_only_a_newly_advertised_tool_counts_as_listing_growth(listing_servers):
    """Availability wording and the listing budget change the bridge text without adding a tool:
    a server turning unavailable, a summarized server whose names fit once another server left,
    a server collapsed to a summary, or a listing that fits only now (the kept bridge named
    nothing and sent the model to search). A new server, summarized, listed or headless, and a
    new tool of a listed server are growth."""
    grew = _mcp_agent._bridge_listing_grew
    bridge = lambda *a, **k: _render_bridge(listing_servers, *a, **k)
    listed, _ = bridge({"a": 2})
    assert not grew(listed, bridge({}, unavailable=["a"])[0])
    summarized, form = bridge({"x": 50, "y": 16}, max_tokens=100)
    expanded, expanded_form = bridge({"y": 16}, unavailable=["x"], max_tokens=100)
    assert (form, expanded_form) == ("groups", "names")
    assert not grew(summarized, expanded)
    full, full_form = bridge({"x": 3, "y": 16})
    collapsed, collapsed_form = bridge({"x": 3, "y": 16}, max_tokens=100)
    assert (full_form, collapsed_form) == ("full", "mixed")
    assert not grew(full, collapsed)

    servers = {f"s{i:02d}": 2 for i in range(16)}
    unlisted, unlisted_form = bridge(servers, max_tokens=200)
    fits_now, fits_form = bridge({"s00": 2}, unavailable=sorted(servers)[1:], max_tokens=200)
    assert unlisted_form == "none" and fits_form != "none"
    assert not grew(unlisted, fits_now)

    assert grew(listed, bridge({"a": 2, "acp": 1})[0])
    assert grew(listed, bridge({"a": 2, "acp": 2})[0])
    assert grew(listed, bridge({"a": 3})[0])
    assert grew(summarized, bridge({"x": 50, "y": 16, "acp": 16}, max_tokens=100)[0])


def test_the_listing_parse_round_trips_every_rendered_form(listing_servers):
    """``_listing_groups`` inverts ``build_catalog_listing_with_form`` for every form it renders:
    each available group's names (headless one-tool lines keyed ``""``), ``"summary"`` for a
    collapsed group and ``"unavailable"`` for a declared server that is down."""
    from tools import tool_search_catalog as catalog

    seen = set()
    # A small group's names are cheaper than its summary line, so only a listing of groups whose
    # names all outgrow their summaries collapses all the way to "groups".
    for servers, max_tokens in [(servers, max_tokens) for servers in (
            {"solo": 1, "pair": 2, "mid": 6, "big": 40}, {"wide": 5, "mid": 6, "big": 40})
            for max_tokens in range(60, 1600, 4)]:
        bridge, form = _render_bridge(listing_servers, servers, unavailable=["down"], max_tokens=max_tokens)
        if form == "none":
            continue
        seen.add(form)
        parsed = catalog._listing_groups(bridge["function"]["description"])
        expected = {"": set(), "down": "unavailable"}
        for server, count in servers.items():
            names = frozenset(f"mcp__{server}__tool{i:02d}" for i in range(count))
            if f"{server} ({count} tools — names not listed;" in bridge["function"]["description"]:
                expected[server] = "summary"
            elif count == 1:
                expected[""] |= names
            else:
                expected[server] = names
        expected[""] = frozenset(expected[""])
        assert parsed == expected, (form, max_tokens)
    assert seen == {"full", "names", "mixed", "groups"}
