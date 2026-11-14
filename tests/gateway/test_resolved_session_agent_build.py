"""``GatewayRunner._build_agent_for_resolved_session``: the agent an ordinary turn on a resolved session
would run on, built and cached for that exact session through ``TurnRunner.resolve_turn_agent``.

A canonical event that finds no agent cached (a clean restart) builds one this way. The build is for
the bound session (its id and key), creates or routes nothing, and is refused when this process could
not run that session's turns (proxy mode, no origin, an unresolvable model runtime).
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Actor:
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.session_id = kwargs["session_id"]
        self._session_db = kwargs.get("session_db")

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass


class _Factory:
    """Stands in for ``run_agent.AIAgent`` on the ordinary build path; records every construction."""

    def __init__(self):
        self.built = []

    def __call__(self, **kwargs):
        actor = _Actor(kwargs)
        self.built.append(actor)
        return actor


@pytest.fixture
def gw(tmp_path, monkeypatch):
    import run_agent

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("GATEWAY_PROXY_URL", raising=False)
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    store = runner.session_store
    bound = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="bound", chat_type="dm", user_id="user"))
    other = store.get_or_create_session(
        SessionSource(platform=Platform.TELEGRAM, chat_id="other", chat_type="dm", user_id="user"))
    with runner._agent_cache_lock:
        runner._agent_cache.clear()
    factory = _Factory()
    monkeypatch.setattr(run_agent, "AIAgent", factory)
    runtime_calls = []

    def runtime(**kwargs):
        runtime_calls.append(kwargs)
        return "test-model", {"provider": "openrouter", "api_key": "test-key", "base_url": "http://127.0.0.1:9"}

    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", runtime)
    try:
        yield runner, store, bound, other, factory, runtime_calls
    finally:
        store.close_all_db_handles()


def _rows(store):
    db = store._db
    with db._lock:
        sessions = db._conn.execute("SELECT id, ended_at FROM sessions ORDER BY id").fetchall()
        routes = db._conn.execute("SELECT * FROM gateway_routing ORDER BY 1").fetchall()
    return sessions, routes


def test_the_agent_builder_resolves_the_bound_session(gw):
    runner, store, bound, other, factory, runtime_calls = gw
    before = _rows(store)
    asyncio.run(runner._build_agent_for_resolved_session(bound))
    [actor] = factory.built
    assert actor.kwargs["session_id"] == bound.session_id != other.session_id
    assert actor.kwargs["gateway_session_key"] == bound.session_key
    assert [c["session_key"] for c in runtime_calls] == [bound.session_key]
    with runner._agent_cache_lock:
        assert set(runner._agent_cache) == {bound.session_key}
        cached = runner._agent_cache[bound.session_key]
    assert cached[0] is actor and cached[3] == bound.session_id
    assert _rows(store) == before  # no session, route or binding created or touched


def test_a_second_build_reuses_the_cached_agent(gw):
    runner, _store, bound, _other, factory, _calls = gw
    asyncio.run(runner._build_agent_for_resolved_session(bound))
    asyncio.run(runner._build_agent_for_resolved_session(bound))
    assert len(factory.built) == 1


@pytest.mark.parametrize("case", ["no_origin", "proxy", "runtime_unresolvable"])
def test_a_session_this_process_cannot_run_is_refused_and_nothing_is_built(gw, monkeypatch, case):
    import dataclasses

    runner, _store, bound, _other, factory, _calls = gw
    entry = bound
    if case == "no_origin":
        entry = dataclasses.replace(bound, origin=None)
    elif case == "proxy":
        monkeypatch.setattr(runner, "_get_proxy_url", lambda: "http://127.0.0.1:9")
    else:
        def unresolvable(**_kwargs):
            raise RuntimeError("no credentials")
        monkeypatch.setattr(runner, "_resolve_session_agent_runtime", unresolvable)
    with pytest.raises(ValueError, match="^canonical_runtime_refused$"):
        asyncio.run(runner._build_agent_for_resolved_session(entry))
    assert factory.built == []
    with runner._agent_cache_lock:
        assert runner._agent_cache == {}


def test_resolve_turn_agent_reports_an_unresolvable_runtime_with_the_original_error(gw, monkeypatch):
    from gateway.run_turn_runner import TurnRuntimeUnavailable

    runner, _store, bound, _other, factory, _calls = gw
    original = RuntimeError("no credentials")

    def unresolvable(**_kwargs):
        raise original
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", unresolvable)
    import run_agent

    async def turn_context():  # built on a running loop, as every turn's is
        return runner._run_agent_build_turn_context(
            runner._run_agent_display_settings(bound.origin), run_agent.AIAgent, message="",
            source=bound.origin, session_key=bound.session_key, run_generation=None, context_prompt="",
            history=[], session_id=bound.session_id, channel_prompt=None,
        )
    _ctx, turn_runner, _ = asyncio.run(turn_context())
    with pytest.raises(TurnRuntimeUnavailable) as raised:
        turn_runner.resolve_turn_agent()
    assert raised.value.error is original and factory.built == []
