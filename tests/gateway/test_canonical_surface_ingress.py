"""Behavioral API contract for the canonical existing-only ingress."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_state import SessionDB


_API_KEY = "canonical-surface-test-key"
_ROUTE = "/v1/canonical-surface/events"
_IDENTITY_ROUTE = "/v1/canonical-surface/identity"


class _ExistingCachedAgent:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.compression_in_place = True
        self.calls: list[tuple[str, list[dict[str, object]], str]] = []

    def run_conversation(
        self,
        text: str,
        *,
        conversation_history: list[dict[str, object]],
        task_id: str,
    ) -> dict[str, object]:
        self.calls.append((text, conversation_history, task_id))
        self._persist_user_message_idx = len(conversation_history)
        return {
            "completed": True,
            "failed": False,
            "interrupted": False,
            "partial": False,
            "session_id": task_id,
            "final_response": "request-owned terminal",
            "messages": [
                *conversation_history,
                {"role": "user", "content": text},
                {"role": "assistant", "content": "request-owned terminal"},
            ],
        }


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app["gateway_runner"] = adapter.gateway_runner
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


def test_authenticated_request_reuses_exact_existing_row_cache_and_lease(tmp_path, monkeypatch):
    async def exercise() -> None:
        hermes_home = tmp_path / "home"
        hermes_home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        config = GatewayConfig(sessions_dir=hermes_home / "sessions")
        runner = GatewayRunner(config)
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat-42",
            chat_type="dm",
            user_id="user-42",
        )
        entry = runner.session_store.get_or_create_session(source)
        db = runner.session_store._db
        assert db is not None
        before_ids = {row[0] for row in db._conn.execute("SELECT id FROM sessions")}
        binding = SimpleNamespace(
            name="canonical",
            session_key=entry.session_key,
            session_id=entry.session_id,
            telegram_chat_id=str(source.chat_id),
            telegram_chat_type=str(source.chat_type),
            telegram_user_id=str(source.user_id),
            telegram_thread_id=None,
            allowed_author_ids=("author-7",),
            allowed_channel_ids=("channel-9",),
        )
        runner.config.canonical_surface_bindings = {"canonical": binding}
        runner.session_store.config = runner.config
        agent = _ExistingCachedAgent(entry.session_id)
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)

        construction_attempts: list[str] = []
        import run_agent

        def forbidden_constructor(*args, **kwargs):
            construction_attempts.append("AIAgent")
            raise AssertionError("canonical ingress must not construct an agent")

        monkeypatch.setattr(run_agent, "AIAgent", forbidden_constructor)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _API_KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_app(adapter)))
        await client.start_server()
        payload = {
            "binding": "canonical",
            "event_id": "event-11",
            "author_id": "author-7",
            "channel_id": "channel-9",
            "text": "use exactly this cached turn",
        }
        try:
            response = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json=payload,
            )
            assert response.status == 200
            assert await response.json() == {
                "event_id": "event-11",
                "text": "request-owned terminal",
            }

            replay = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json=payload,
            )
            assert replay.status == 200
            assert await replay.json() == {
                "event_id": "event-11", "text": "request-owned terminal"
            }
            assert len(agent.calls) == 1

            malformed_auth = await client.post(_ROUTE, json=payload)
            assert malformed_auth.status == 401
            injected_target = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={**payload, "target": "not-authority"},
            )
            assert injected_target.status == 400
            rejected_principal = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={**payload, "author_id": "other-author"},
            )
            assert rejected_principal.status == 403
            original_tip = db.get_compression_tip
            monkeypatch.setattr(db, "get_compression_tip", lambda session_id: "stale-tip")
            stale_tip = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json=payload,
            )
            assert stale_tip.status == 409
            monkeypatch.setattr(db, "get_compression_tip", original_tip)
            after_ids = {row[0] for row in db._conn.execute("SELECT id FROM sessions")}
        finally:
            await client.close()
            adapter._response_store.close()
            runner.session_store.close_all_db_handles()

        assert agent.calls == [(payload["text"], [], entry.session_id)]
        assert construction_attempts == []
        assert after_ids == before_ids
        assert runner.session_store.lookup_by_session_key(entry.session_key) is entry

    asyncio.run(exercise())


@pytest.fixture
def ingress(tmp_path, monkeypatch):
    """Real resolver/runner/SQLite, but no listener or external model."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    binding = SimpleNamespace(
        name="canonical", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    agent = _ExistingCachedAgent(entry.session_id)
    runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _API_KEY}))
    adapter.gateway_runner = runner
    payload = dict(binding=binding.name, event_id="event", author_id="author", channel_id="channel", text="hello")

    async def send(**changes):
        async def read():
            return json.dumps({**payload, **changes}).encode()
        request = SimpleNamespace(headers={"Authorization": f"Bearer {_API_KEY}"}, read=read)
        response = await adapter._handle_canonical_surface_event(request)
        return response.status, json.loads(response.text)

    import run_agent
    real_agent_class = run_agent.AIAgent
    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail("new actor forbidden"))
    yield SimpleNamespace(runner=runner, agent=agent, adapter=adapter, send=send, binding=binding, entry=entry,
                          real_agent_class=real_agent_class)
    adapter._response_store.close()
    runner.session_store.close_all_db_handles()


def test_concurrent_delivery_is_busy_then_replays_terminal(ingress, monkeypatch):
    async def exercise():
        entered, release = asyncio.Event(), asyncio.Event()
        real_run = ingress.runner.run_bound_existing_turn

        async def held(*args, **kwargs):
            entered.set()
            await release.wait()
            return await real_run(*args, **kwargs)

        monkeypatch.setattr(ingress.runner, "run_bound_existing_turn", held)
        first = asyncio.create_task(ingress.send())
        try:
            await entered.wait()
            status, body = await ingress.send()
            assert (status, body["error"]["code"]) == (409, "canonical_turn_busy")
        finally:
            release.set()
            original = await first
        assert original[0] == 200
        assert await ingress.send() == original
        assert len(ingress.agent.calls) == 1

    asyncio.run(exercise())


def test_active_method_entry_reserves_route_and_cached_actor(ingress, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)

    async def exercise():
        first = asyncio.create_task(ingress.send())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            store = ingress.runner.session_store
            key = ingress.entry.session_key
            with pytest.raises(ValueError, match="canonical_turn_busy"):
                store.reset_session(key)
            with pytest.raises(ValueError, match="canonical_turn_busy"):
                store.switch_session(key, "another-session")
            ingress.runner._evict_cached_agent(key)
            assert store.lookup_by_session_key(key) is ingress.entry
            assert ingress.runner._agent_cache[key][0] is ingress.agent
        finally:
            release.set()
        assert (await first)[0] == 200
        assert store.reset_session(key).session_id != ingress.entry.session_id

    asyncio.run(exercise())


@pytest.mark.parametrize("invalidation", ["cross_process", "dead_session"])
def test_paused_canonical_actor_blocks_turn_runner_invalidation(ingress, monkeypatch, invalidation):
    """The ordinary turn's real cache writer must not evict the reserved actor."""
    import gateway.run as gateway_run
    import run_agent

    runner, entry = ingress.runner, ingress.entry
    entered, release = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", lambda **kw: ("test-model", {}))
    monkeypatch.setattr(runner, "_resolve_turn_agent_config", lambda *a: {"model": "test-model", "runtime": {}})
    monkeypatch.setattr(runner, "_agent_config_signature", lambda *a, **kw: "exact")
    monkeypatch.setattr(run_agent, "AIAgent", lambda **kw: pytest.fail("reserved key constructed a new actor"))
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

    async def exercise():
        first = asyncio.create_task(ingress.send())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert runner.session_store.canonical_entry_reserved(entry.session_key)
            if invalidation == "cross_process":
                await runner._session_db.append_message(entry.session_id, role="user", content="foreign")
                target_id = entry.session_id
            else:
                target_id = "different-session"
                monkeypatch.setattr(runner.session_store, "_is_session_ended_in_db", lambda sid: True)

            with pytest.raises(ValueError, match="canonical_turn_busy"):
                await runner._run_agent(
                    message="wrong target", context_prompt="", history=[],
                    source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"),
                    session_id=target_id, session_key=entry.session_key,
                )
            assert runner._agent_cache[entry.session_key][0] is ingress.agent
            assert ingress.agent.calls == []
        finally:
            release.set()
            await first

    asyncio.run(exercise())


def test_turn_runner_construction_cannot_replace_newly_reserved_canonical_actor(ingress, monkeypatch):
    """Reservation acquired during construction still owns the original cache entry."""
    import gateway.run as gateway_run
    import run_agent

    runner, entry = ingress.runner, ingress.entry
    constructing, finish_construction = threading.Event(), threading.Event()
    canonical_entered, release_canonical = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation
    constructed = []

    def held(*args, **kwargs):
        canonical_entered.set()
        assert release_canonical.wait(10)
        return original(*args, **kwargs)

    def construct(**kw):
        constructing.set()
        assert finish_construction.wait(10)
        actor = _ExistingCachedAgent(kw["session_id"])
        constructed.append(actor)
        return actor

    monkeypatch.setattr(ingress.agent, "run_conversation", held)
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", lambda **kw: ("test-model", {}))
    monkeypatch.setattr(runner, "_resolve_turn_agent_config", lambda *a: {"model": "test-model", "runtime": {}})
    monkeypatch.setattr(runner, "_agent_config_signature", lambda *a, **kw: "different-signature")
    monkeypatch.setattr(run_agent, "AIAgent", construct)
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {})

    async def exercise():
        writer = asyncio.create_task(runner._run_agent(
            message="ordinary turn", context_prompt="", history=[],
            source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user"),
            session_id=entry.session_id, session_key=entry.session_key,
        ))
        first = None
        try:
            assert await asyncio.to_thread(constructing.wait, 10)
            first = asyncio.create_task(ingress.send())
            assert await asyncio.to_thread(canonical_entered.wait, 10)
            assert runner.session_store.canonical_entry_reserved(entry.session_key)
            finish_construction.set()
            with pytest.raises(ValueError, match="canonical_turn_busy"):
                await writer
            assert len(constructed) == 1
            assert runner._agent_cache[entry.session_key][0] is ingress.agent
            assert ingress.agent.calls == []
        finally:
            finish_construction.set()
            release_canonical.set()
            if first is not None:
                await first
            if not writer.done():
                await writer

    asyncio.run(exercise())


@pytest.mark.parametrize("writer", ["cap", "idle", "pressure"])
def test_paused_canonical_actor_survives_cache_eviction_writers(ingress, monkeypatch, writer):
    import time
    from gateway.agent_cache_pressure import AgentCacheBounds

    entered, release = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)
    runner = ingress.runner
    key = ingress.entry.session_key
    other = _ExistingCachedAgent("other")
    cleaned = []
    monkeypatch.setattr(runner, "_commit_then_release_soft", lambda *args: cleaned.append(args))
    monkeypatch.setattr(runner, "_release_evicted_agent_soft", lambda *args: cleaned.append(args))

    async def exercise():
        first = asyncio.create_task(ingress.send())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert runner.session_store.canonical_entry_reserved(key)
            if writer == "cap":
                monkeypatch.setattr(runner, "_agent_cache_cap", lambda: 1)
                with runner._agent_cache_lock:
                    runner._agent_cache["other"] = (other, "sig")
                    runner._enforce_agent_cache_cap()
            elif writer == "idle":
                ingress.agent._last_activity_ts = time.time() - 1000
                monkeypatch.setattr(runner, "_agent_cache_idle_ttl", lambda: 1)
                assert runner._sweep_idle_cached_agents() == 0
            else:
                import gateway.agent_cache_pressure as pressure
                monkeypatch.setattr(runner, "_agent_cache_bounds", lambda: AgentCacheBounds(
                    memory_high_mb=1, protect_recent=0,
                ))
                monkeypatch.setattr(pressure, "read_anon_rss_mb", lambda: 100)
                monkeypatch.setattr(pressure, "transcript_persistence_caught_up", lambda agent: True)
                assert runner._sweep_agent_cache_under_pressure() == 0
            assert runner._agent_cache[key][0] is ingress.agent
            assert not cleaned
        finally:
            release.set()
        assert (await first)[0] == 200

    asyncio.run(exercise())


def test_idle_sweep_keeps_reserved_actor_but_evicts_unrelated_key(ingress, monkeypatch):
    import time

    runner = ingress.runner
    key = ingress.entry.session_key
    other = _ExistingCachedAgent("other")
    ingress.agent._last_activity_ts = time.time() - 1000
    setattr(other, "_last_activity_ts", time.time() - 1000)
    monkeypatch.setattr(runner, "_agent_cache_idle_ttl", lambda: 1)
    monkeypatch.setattr(runner, "_release_evicted_agent_soft", lambda agent: None)
    with runner._agent_cache_lock:
        runner._agent_cache["other"] = (other, "sig")
    token = runner.session_store.reserve_canonical_entry(ingress.entry)
    try:
        assert runner._sweep_idle_cached_agents() == 1
        assert runner._agent_cache[key][0] is ingress.agent
        assert "other" not in runner._agent_cache
    finally:
        runner.session_store.release_canonical_entry(ingress.entry, token)


def test_paused_canonical_actor_blocks_cache_rebaseline(ingress, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)

    async def exercise():
        first = asyncio.create_task(ingress.send())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            key, sid = ingress.entry.session_key, ingress.entry.session_id
            before = ingress.runner._agent_cache[key]
            await ingress.runner._session_db.append_message(sid, role="user", content="foreign")
            await ingress.runner._refresh_agent_cache_message_count(key, sid)
            assert ingress.runner._agent_cache[key] is before
        finally:
            release.set()
        assert (await first)[0] == 200

    asyncio.run(exercise())


def test_paused_canonical_actor_blocks_heal_new_and_compression(ingress, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)

    async def exercise():
        first = asyncio.create_task(ingress.send())
        store = ingress.runner.session_store
        entry = ingress.entry
        sid = entry.session_id
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            with pytest.raises(ValueError, match="canonical_turn_busy"):
                store.get_or_create_session(entry.origin, force_new=True)
            monkeypatch.setattr(store, "_compression_tip_for_session_id", lambda _sid: "compressed-child")
            with pytest.raises(ValueError, match="canonical_turn_busy"):
                store.get_or_create_session(entry.origin)
            with pytest.raises(ValueError, match="canonical_turn_busy"):
                store.advance_compression_session(entry.session_key, sid, "compressed-child")
            assert store.lookup_by_session_key(entry.session_key) is entry
            assert entry.session_id == sid
        finally:
            release.set()
        assert (await first)[0] == 200

    asyncio.run(exercise())


def test_paused_canonical_actor_blocks_direct_runner_repoint(ingress, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)

    async def exercise():
        first = asyncio.create_task(ingress.send())
        store = ingress.runner.session_store
        entry = ingress.entry
        sid = entry.session_id
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            with pytest.raises(ValueError, match="canonical_turn_busy"):
                store.repoint_session_entry(entry, sid, "runner-child")
            assert store.lookup_by_session_key(entry.session_key) is entry
            assert entry.session_id == sid
        finally:
            release.set()
        assert (await first)[0] == 200
        assert store.repoint_session_entry(entry, sid, "runner-child")
        assert entry.session_id == "runner-child"

    asyncio.run(exercise())


def test_paused_canonical_actor_defers_transcript_retry_without_losing_backlog(ingress, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)
    store = ingress.runner.session_store
    db = store._db
    assert db is not None
    parent = ingress.entry.session_id
    child = f"{parent}_compressed"
    old = {"role": "assistant", "content": "pending old"}
    new = {"role": "user", "content": "pending new"}

    async def exercise():
        first = asyncio.create_task(ingress.send())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            db.end_session(parent, "compression")
            db.create_session(child, source="telegram", parent_session_id=parent)
            db.replace_messages(child, [{"role": "user", "content": "summary"}])
            await asyncio.to_thread(store.append_to_transcript, parent, old)
            assert ingress.entry.session_id == parent
            assert store._transcript_reroutes.get(parent) is None
            assert [m["content"] for m in store._dirty_transcripts[parent]] == [old["content"]]
            assert [m["content"] for m in db.get_messages_as_conversation(child)] == ["summary"]
        finally:
            release.set()
        assert (await first) == (200, {"event_id": "event", "text": "request-owned terminal"})
        assert ingress.agent.calls[0][2] == parent
        assert ingress.entry.session_id == parent

    asyncio.run(exercise())
    store.append_to_transcript(parent, new)
    assert ingress.entry.session_id == child
    assert [m["content"] for m in db.get_messages_as_conversation(child)] == [
        "summary", old["content"], new["content"],
    ]
    assert parent not in store._dirty_transcripts


def test_recovery_race_does_not_reopen_old_row_before_reserved_publication(ingress, monkeypatch):
    store = ingress.runner.session_store
    db = store._db
    assert db is not None
    source = ingress.entry.origin
    key = ingress.entry.session_key
    old_id = ingress.entry.session_id
    db.end_session(old_id, "agent_close")
    with store._lock:
        store._entries.pop(key)

    queried, release = threading.Event(), threading.Event()
    original_query = store._query_recoverable_session

    def paused_query(**kwargs):
        recovered = original_query(**kwargs)
        assert recovered is not None and recovered.session_id == old_id
        queried.set()
        assert release.wait(10)
        return recovered

    monkeypatch.setattr(store, "_query_recoverable_session", paused_query)

    async def exercise():
        recovery = asyncio.create_task(asyncio.to_thread(store.get_or_create_session, source))
        token = None
        current = None
        try:
            assert await asyncio.to_thread(queried.wait, 10)
            # Simulate another route publisher winning while the DB query was
            # lock-free. Public same-key get_or_create calls are singleflight.
            current = replace(ingress.entry, session_id=f"{old_id}_new")
            with store._lock:
                store._entries[key] = current
            token = store.reserve_canonical_entry(current)
            release.set()
            assert await recovery is current
            assert store.lookup_by_session_key(key) is current
            assert db.get_session(old_id)["ended_at"] is not None
        finally:
            release.set()
            if token is not None and current is not None:
                store.release_canonical_entry(current, token)
            await recovery

    asyncio.run(exercise())


def test_transcript_retry_racing_reservation_accounts_for_written_message_once(ingress, monkeypatch):
    store = ingress.runner.session_store
    db = store._db
    assert db is not None
    entry = ingress.entry
    parent = entry.session_id
    child = f"{parent}_compressed"
    db.end_session(parent, "compression")
    db.create_session(child, source="telegram", parent_session_id=parent)
    db.replace_messages(child, [{"role": "user", "content": "summary"}])
    store._dirty_transcripts[parent] = [{"role": "assistant", "content": "older"}]
    entered, release = threading.Event(), threading.Event()
    original_append = store._append_transcript_message

    def paused_append(session_id, message):
        if session_id == child:
            entered.set()
            assert release.wait(10)
        return original_append(session_id, message)

    monkeypatch.setattr(store, "_append_transcript_message", paused_append)

    async def exercise():
        append = asyncio.create_task(asyncio.to_thread(
            store.append_to_transcript, parent, {"role": "user", "content": "next"},
        ))
        token = None
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            token = store.reserve_canonical_entry(entry)
            release.set()
            await append
            assert entry.session_id == parent
            assert store._transcript_reroutes.get(parent) is None
            assert [m["content"] for m in store._dirty_transcripts[parent]] == ["next"]
        finally:
            release.set()
            if token is not None:
                store.release_canonical_entry(entry, token)
            await append

    asyncio.run(exercise())
    store.append_to_transcript(parent, {"role": "assistant", "content": "last"})
    assert [m["content"] for m in db.get_messages_as_conversation(child)] == [
        "summary", "older", "next", "last",
    ]
    assert entry.session_id == child


def test_cancelled_turn_keeps_outward_callbacks_quarantined_until_worker_exits(ingress, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    observations = []
    outward = lambda *_args: None
    ingress.agent.callback = outward
    original_run = ingress.agent.run_conversation

    def held(*args, **kwargs):
        observations.append(ingress.agent.callback)
        entered.set()
        assert release.wait(10)
        observations.append(ingress.agent.callback)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)
    store = ingress.runner.session_store
    leases = ingress.runner._turn_leases
    original_reservation_release = store.release_canonical_entry
    original_lease_release = leases.release
    released = []

    async def exercise():
        finished = asyncio.Event()

        def release_reservation(*args):
            released.append("reservation")
            return original_reservation_release(*args)

        def release_lease(*args):
            released.append("lease")
            original_lease_release(*args)
            finished.set()

        monkeypatch.setattr(store, "release_canonical_entry", release_reservation)
        monkeypatch.setattr(leases, "release", release_lease)
        first = asyncio.create_task(ingress.send())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert observations == [None]
            assert ingress.agent.callback is None
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert ingress.agent.callback is None
            assert store.canonical_entry_reserved(ingress.entry.session_key)
            assert leases._leases[ingress.entry.session_id].lock.locked()
            assert released == []
        finally:
            release.set()
        await asyncio.wait_for(finished.wait(), 10)
        assert observations == [None, None]
        assert ingress.agent.callback is outward
        assert not store.canonical_entry_reserved(ingress.entry.session_key)
        assert not leases._leases[ingress.entry.session_id].lock.locked()
        assert released == ["reservation", "lease"]

    asyncio.run(exercise())


def test_shutdown_cache_sweep_preserves_paused_canonical_worker_and_cleans_idle(ingress, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original_run = ingress.agent.run_conversation

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(ingress.agent, "run_conversation", held)
    runner = ingress.runner
    key = ingress.entry.session_key
    idle = _ExistingCachedAgent("idle")
    with runner._agent_cache_lock:
        runner._agent_cache["idle"] = (idle, "sig")
    cleaned = []

    async def exercise():
        swept = asyncio.Event()
        continue_shutdown = asyncio.Event()

        async def cleanup(agent, *, context):
            assert runner._agent_cache_lock.acquire(blocking=False)
            runner._agent_cache_lock.release()
            assert runner.session_store._lock.acquire(blocking=False)
            runner.session_store._lock.release()
            cleaned.append((agent, context))
            swept.set()
            await continue_shutdown.wait()

        monkeypatch.setattr(runner, "_cleanup_agent_resources_off_loop", cleanup)
        # stop() must exercise its real cache sweep, not remove PID files or
        # close process-global clients owned by other tests.
        monkeypatch.setattr("gateway.status.remove_pid_file", lambda: None)
        monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda: None)
        monkeypatch.setattr("agent.auxiliary_client.shutdown_cached_clients", lambda: None)
        first = asyncio.create_task(ingress.send())
        stop = None
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert runner.session_store.canonical_entry_reserved(key)
            stop = asyncio.create_task(runner.stop())
            await asyncio.wait_for(swept.wait(), 10)
            assert cleaned == [(idle, "shutdown idle-cache")]
            assert runner._agent_cache[key][0] is ingress.agent
            assert "idle" not in runner._agent_cache
            assert runner.session_store.canonical_entry_reserved(key)
        finally:
            release.set()
            continue_shutdown.set()
            if stop is not None:
                await asyncio.wait_for(stop, 10)

        assert not runner.session_store.canonical_entry_reserved(key)
        assert runner._agent_cache[key][0] is ingress.agent
        assert cleaned == [(idle, "shutdown idle-cache")]

    asyncio.run(exercise())


def test_missing_agent_receipt_replays_after_cache_is_restored(ingress):
    async def exercise():
        cached = ingress.runner._agent_cache.pop(ingress.entry.session_key)
        original = await ingress.send()
        assert (original[0], original[1]["error"]["code"]) == (409, "canonical_agent_missing")
        ingress.runner._agent_cache[ingress.entry.session_key] = cached
        assert await ingress.send() == original
        assert ingress.agent.calls == []
    asyncio.run(exercise())


def test_reserved_turn_accepts_existing_direct_agent_cache(ingress):
    ingress.runner._agent_cache[ingress.entry.session_key] = ingress.agent

    async def exercise():
        assert await ingress.send() == (200, {"event_id": "event", "text": "request-owned terminal"})
        assert ingress.agent.calls == [("hello", [], ingress.entry.session_id)]
        assert not ingress.runner.session_store.canonical_entry_reserved(ingress.entry.session_key)
        assert not ingress.runner._turn_leases._leases[ingress.entry.session_id].lock.locked()

    asyncio.run(exercise())


def test_execution_exception_is_explicit_uncertainty(ingress, monkeypatch):
    def interrupted(*args, **kwargs):
        ingress.agent.calls.append("started")
        raise RuntimeError("private execution detail")
    monkeypatch.setattr(ingress.agent, "run_conversation", interrupted)

    async def exercise():
        original = await ingress.send()
        assert (original[0], original[1]["error"]["code"]) == (409, "canonical_event_uncertain")
        assert await ingress.send() == original
        assert ingress.agent.calls == ["started"]
        assert "private execution detail" not in str(original)
    asyncio.run(exercise())


def test_terminal_receipt_survives_reopened_db_without_cached_actor(ingress):
    from hermes_state import SessionDB

    async def exercise():
        original = await ingress.send()
        assert original[0] == 200
        old = ingress.runner.session_store._db
        path = old.db_path
        old.close()
        ingress.runner.session_store._db = SessionDB(path)
        ingress.runner._agent_cache.clear()
        ingress.adapter._canonical_active_events.clear()
        assert await ingress.send() == original
        assert len(ingress.agent.calls) == 1
        status, body = await ingress.send(text="changed payload")
        assert (status, body["error"]["code"]) == (409, "canonical_event_conflict")
    asyncio.run(exercise())


def test_cancelled_claim_survives_reopen_without_automatic_turn(ingress, monkeypatch):
    from hermes_state import SessionDB

    async def exercise():
        entered = asyncio.Event()
        async def interrupted(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()
        monkeypatch.setattr(ingress.runner, "run_bound_existing_turn", interrupted)
        first = asyncio.create_task(ingress.send())
        await entered.wait()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        old = ingress.runner.session_store._db
        path = old.db_path
        old.close()
        ingress.runner.session_store._db = SessionDB(path)
        status, body = await ingress.send()
        assert (status, body["error"]["code"]) == (409, "canonical_event_uncertain")
        assert ingress.agent.calls == []
    asyncio.run(exercise())


def test_terminal_commit_failure_never_reexecutes(ingress, monkeypatch):
    from gateway.canonical_surface import CanonicalEventReceipt

    def unavailable(*args):
        raise OSError("isolated terminal commit failure")
    monkeypatch.setattr(CanonicalEventReceipt, "complete", unavailable)
    async def exercise():
        first = await ingress.send()
        assert (first[0], first[1]["error"]["code"]) == (409, "canonical_event_uncertain")
        assert await ingress.send() == first
        assert len(ingress.agent.calls) == 1
    asyncio.run(exercise())


def test_two_connections_claim_once_and_binding_identity_scopes_event(ingress):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from gateway.canonical_surface import CanonicalEventReceipt, CanonicalIngressEvent, CanonicalTurnResult
    from hermes_state import SessionDB

    db = ingress.runner.session_store._db
    other = SessionDB(db.db_path)
    event = CanonicalIngressEvent("canonical", "event", "author", "channel", "hello")
    receipts = [CanonicalEventReceipt(handle, ingress.binding, event) for handle in (db, other)]
    barrier = Barrier(2)
    def claim(receipt):
        barrier.wait()
        try:
            assert receipt.claim() is None
            return "claimed"
        except ValueError as exc:
            return str(exc)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(claim, receipts))
        assert sorted(outcomes) == ["canonical_event_uncertain", "claimed"]
        winner = receipts[outcomes.index("claimed")]
        loser = receipts[outcomes.index("canonical_event_uncertain")]
        with pytest.raises(ValueError, match="canonical_event_uncertain"):
            loser.complete(CanonicalTurnResult("canonical", "wrong-owner"))
        winner.complete(CanonicalTurnResult("canonical", "terminal"))
        assert loser.claim() == CanonicalTurnResult("canonical", "terminal")
        # An ACL edit is not a new server identity and must not reset dedupe.
        edited_acl = SimpleNamespace(**{**vars(ingress.binding), "allowed_author_ids": ("author", "new")})
        assert CanonicalEventReceipt(other, edited_acl, event).claim().terminal_text == "terminal"
        # A different configured destination is a distinct binding identity.
        rebound = SimpleNamespace(**{**vars(ingress.binding), "telegram_chat_id": "different"})
        assert CanonicalEventReceipt(other, rebound, event).claim() is None
        assert ingress.agent.calls == []
    finally:
        other.close()


def test_refusal_paths_never_reach_the_cached_actor(ingress):
    """Every refusal happens before the cached actor runs.

    Unknown binding, a principal outside the binding's allow-lists, an
    oversize or heartbeat-shaped payload (no author/channel), a missing or
    wrong bearer, and a replayed event_id whose payload changed are all
    refused; only the one exact admitted event runs, exactly once.
    """

    async def raw(body: bytes, *, bearer: str | None = _API_KEY):
        async def read():
            return body
        headers = {"Authorization": f"Bearer {bearer}"} if bearer is not None else {}
        # The 401 audit log reads method/path/transport off the request.
        request = SimpleNamespace(
            headers=headers, read=read, method="POST", path_qs=_ROUTE, transport=None
        )
        response = await ingress.adapter._handle_canonical_surface_event(request)
        return response.status, json.loads(response.text)

    def code(result):
        return result[0], result[1]["error"]["code"]

    async def exercise():
        assert code(await ingress.send(binding="not-configured")) == (404, "canonical_binding_unknown")
        assert code(await ingress.send(author_id="stranger")) == (403, "canonical_principal_rejected")
        assert code(await ingress.send(channel_id="elsewhere")) == (403, "canonical_principal_rejected")
        assert code(await ingress.send(text="x" * 16_385)) == (400, "canonical_invalid_request")
        heartbeat_shaped = json.dumps(
            {"binding": "canonical", "event_id": "tick", "text": "[System: Heartbeat]"}
        ).encode()
        assert code(await raw(heartbeat_shaped)) == (400, "canonical_invalid_request")
        admitted = json.dumps(
            dict(binding="canonical", event_id="event", author_id="author", channel_id="channel", text="hello")
        ).encode()
        assert code(await raw(admitted, bearer=None)) == (401, "gateway_auth_failed")
        assert code(await raw(admitted, bearer="not-the-key")) == (401, "gateway_auth_failed")
        assert ingress.agent.calls == []

        first = await ingress.send()
        assert first == (200, {"event_id": "event", "text": "request-owned terminal"})
        assert code(await ingress.send(text="changed payload")) == (409, "canonical_event_conflict")
        assert code(await ingress.send(author_id="stranger")) == (403, "canonical_principal_rejected")
        assert await ingress.send() == first
        assert ingress.agent.calls == [("hello", [], ingress.entry.session_id)]

    asyncio.run(exercise())


def test_identity_proves_live_head_from_read_only_store_and_rejects_targets(ingress, monkeypatch):
    db = ingress.runner.session_store._db
    assert db is not None
    db.create_session("root", source="telegram")
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("root", ingress.entry.session_id)))
    ingress.binding.session_id = "root"
    before_sessions = db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    before_meta = db._conn.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0]
    from hermes_state import SessionDB
    original_init = SessionDB.__init__
    opened = []

    def track_open(self, *args, **kwargs):
        opened.append(kwargs.get("read_only"))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(SessionDB, "__init__", track_open)

    async def exercise():
        client = TestClient(TestServer(_app(ingress.adapter)))
        await client.start_server()
        try:
            async def get(suffix="", key=_API_KEY):
                headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
                response = await client.get("/v1/canonical-surface/identity" + suffix, headers=headers)
                return response.status, await response.json()

            assert (await get(key=None))[0] == 401
            assert (await get(key="wrong"))[0] == 401
            assert (await get("?session_id=root"))[0] == 400
            status, proof = await get()
            assert status == 200
            if sys.platform == "darwin":
                import psutil
                token = f"darwin-tv:{psutil.Process(os.getpid()).create_time():.6f}"
            else:
                token = "linux-clk:" + Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()[19]
            assert proof == {
                "session_id": ingress.entry.session_id,
                "lineage_root_digest": "sha256:" + hashlib.sha256(
                    b"hermes.target-bind:lineage-root\0root").hexdigest(),
                "process_pid": os.getpid(), "process_started_at": token,
            }
            assert opened and all(opened)
            db.create_session("sibling", parent_session_id="root", source="telegram")
            ingress.binding.session_id = "sibling"
            assert (await get())[0] == 409
            ingress.binding.session_id = "absent"
            assert (await get())[0] == 409
            ingress.runner.config.canonical_surface_bindings = {
                "one": ingress.binding, "two": ingress.binding}
            assert (await get())[0] == 409
        finally:
            await client.close()

    asyncio.run(exercise())
    assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == before_sessions + 1
    assert db._conn.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0] == before_meta
    assert ingress.agent.calls == []



def test_ancestor_pin_runs_two_distinct_events_on_same_cached_head(ingress):
    db = ingress.runner.session_store._db
    assert db is not None
    db.create_session("root", source="telegram")
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("root", ingress.entry.session_id)))
    ingress.binding.session_id = "root"
    before = db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    async def exercise():
        assert (await ingress.send())[0] == 200
        assert (await ingress.send(event_id="next", text="second"))[0] == 200

    asyncio.run(exercise())
    assert [call[0] for call in ingress.agent.calls] == ["hello", "second"]
    assert all(call[2] == ingress.entry.session_id for call in ingress.agent.calls)
    assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == before


@pytest.mark.parametrize("transition", ["after_admission", "after_lease", "after_transcript_load"])
def test_rebound_live_sibling_refuses_old_cached_actor(ingress, monkeypatch, transition):
    db = ingress.runner.session_store._db
    assert db is not None
    db.create_session("root", source="telegram")
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("root", ingress.entry.session_id)))
    ingress.binding.session_id = "root"
    before = db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def rebind():
        db.create_session("sibling", parent_session_id="root", source="telegram")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?", (ingress.entry.session_id,)))
        store = ingress.runner.session_store
        with store._lock:
            store._entries[ingress.entry.session_key] = replace(ingress.entry, session_id="sibling")

    if transition == "after_admission":
        real_run = ingress.runner.run_bound_existing_turn

        async def run_after_rebind(*args, **kwargs):
            rebind()
            return await real_run(*args, **kwargs)

        monkeypatch.setattr(ingress.runner, "run_bound_existing_turn", run_after_rebind)
    elif transition == "after_lease":
        real_acquire = ingress.runner._turn_leases.acquire

        async def acquire_then_rebind(*args, **kwargs):
            lease = await real_acquire(*args, **kwargs)
            rebind()
            return lease

        monkeypatch.setattr(ingress.runner._turn_leases, "acquire", acquire_then_rebind)
    else:
        real_load = ingress.runner.async_session_store.load_transcript

        async def load_then_rebind(session_id):
            history = await real_load(session_id)
            rebind()
            return history

        monkeypatch.setattr(ingress.runner.async_session_store, "load_transcript", load_then_rebind)

    status, body = asyncio.run(ingress.send())
    assert status == 409, body
    assert body["error"]["code"] == "canonical_binding_stale"
    assert ingress.agent.calls == []
    assert db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == before + 1
    assert db.get_session(ingress.entry.session_id)["ended_at"] is not None
    assert db.get_session("sibling")["ended_at"] is None


@pytest.mark.parametrize("change", ["evicted", "replaced", "session_id_changed"])
def test_cache_change_during_transcript_load_refuses_old_actor(ingress, monkeypatch, change):
    real_load = ingress.runner.async_session_store.load_transcript

    async def load_then_change_cache(session_id):
        history = await real_load(session_id)
        with ingress.runner._agent_cache_lock:
            if change == "evicted":
                ingress.runner._agent_cache.pop(ingress.entry.session_key)
            elif change == "replaced":
                replacement = _ExistingCachedAgent(ingress.entry.session_id)
                ingress.runner._agent_cache[ingress.entry.session_key] = (
                    replacement, "exact", 0, ingress.entry.session_id
                )
            else:
                ingress.runner._agent_cache[ingress.entry.session_key] = (
                    ingress.agent, "exact", 0, "different-session"
                )
        return history

    monkeypatch.setattr(ingress.runner.async_session_store, "load_transcript", load_then_change_cache)
    status, body = asyncio.run(ingress.send())
    assert status == 409, body
    assert body["error"]["code"] == "canonical_binding_stale"
    assert ingress.agent.calls == []


def test_identity_never_reads_writer_session_row_or_flushes_queued_tokens(ingress, monkeypatch):
    db = ingress.runner.session_store._db
    assert db is not None
    def forbidden(*args, **kwargs):
        pytest.fail("identity must only read a separately opened read-only DB")
    monkeypatch.setattr(db, "get_session", forbidden)
    monkeypatch.setattr(db, "flush_token_counts", forbidden)
    request = SimpleNamespace(headers={"Authorization": f"Bearer {_API_KEY}"}, query={},
                              method="GET", path_qs="/v1/canonical-surface/identity", raw_path="/v1/canonical-surface/identity", transport=None)
    response = asyncio.run(ingress.adapter._handle_canonical_surface_identity(request))
    assert response.status == 200
    assert json.loads(response.text)["session_id"] == ingress.entry.session_id


def test_method_entry_rebind_rejects_before_real_agent_side_effects(ingress, monkeypatch):
    """The rebind happens after the wrapper's check, at the real method boundary."""
    db = ingress.runner.session_store._db
    assert db is not None
    db.create_session("root", source="telegram")
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?", ("root", ingress.entry.session_id)))
    ingress.binding.session_id = "root"

    from agent import background_review

    def forbidden_review(_agent):
        pytest.fail("stale canonical turn cancelled background review")

    monkeypatch.setattr(background_review, "cancel_background_review_for_live_turn", forbidden_review)

    def rebind_at_method_entry(text, *, conversation_history, task_id):
        db.create_session("sibling", parent_session_id="root", source="telegram")
        db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = datetime('now') WHERE id = ?", (ingress.entry.session_id,)))
        store = ingress.runner.session_store
        with store._lock:
            store._entries[ingress.entry.session_key] = replace(ingress.entry, session_id="sibling")
        return ingress.real_agent_class.run_conversation(
            ingress.agent, text, conversation_history=conversation_history, task_id=task_id)

    monkeypatch.setattr(ingress.agent, "run_conversation", rebind_at_method_entry)
    status, body = asyncio.run(ingress.send())
    assert status == 409, body
    assert body["error"]["code"] == "canonical_binding_stale"
    assert ingress.agent.calls == []


@pytest.mark.parametrize("change", ["evicted", "replaced", "session_id_changed"])
def test_method_entry_cache_change_rejects_before_real_agent_side_effects(ingress, monkeypatch, change):
    """Change the cache after the wrapper check but before the real method preflight."""
    from agent import background_review

    monkeypatch.setattr(
        background_review, "cancel_background_review_for_live_turn",
        lambda _agent: pytest.fail("stale cached actor reached background review"),
    )

    def change_cache_at_method_entry(text, *, conversation_history, task_id):
        with ingress.runner._agent_cache_lock:
            if change == "evicted":
                ingress.runner._agent_cache.pop(ingress.entry.session_key)
            elif change == "replaced":
                ingress.runner._agent_cache[ingress.entry.session_key] = (
                    _ExistingCachedAgent(ingress.entry.session_id), "exact", 0, ingress.entry.session_id
                )
            else:
                ingress.runner._agent_cache[ingress.entry.session_key] = (
                    ingress.agent, "exact", 0, "different-session"
                )
        return ingress.real_agent_class.run_conversation(
            ingress.agent, text, conversation_history=conversation_history, task_id=task_id
        )

    monkeypatch.setattr(ingress.agent, "run_conversation", change_cache_at_method_entry)
    status, body = asyncio.run(ingress.send())
    assert status == 409, body
    assert body["error"]["code"] == "canonical_binding_stale"
    assert ingress.agent.calls == []


def test_real_agent_without_canonical_turn_has_no_method_entry_preflight(monkeypatch):
    """An unrelated turn never inherits the gateway canonical guard."""
    import run_agent
    from agent import background_review

    class ReachedReview(Exception):
        pass

    def reached(_agent):
        raise ReachedReview

    monkeypatch.setattr(background_review, "cancel_background_review_for_live_turn", reached)
    with pytest.raises(ReachedReview):
        run_agent.AIAgent.run_conversation(SimpleNamespace(), "unrelated")


def test_method_entry_guard_is_consumed_before_nested_actor_work(monkeypatch):
    import run_agent
    from agent import background_review
    from gateway.canonical_surface import canonical_method_entry_preflight

    class ReachedReview(Exception):
        pass

    actor = SimpleNamespace()
    calls = []

    def guard(agent, task_id):
        calls.append((agent, task_id))

    def reached(_agent):
        assert canonical_method_entry_preflight.get() is None
        raise ReachedReview

    monkeypatch.setattr(background_review, "cancel_background_review_for_live_turn", reached)
    token = canonical_method_entry_preflight.set(guard)
    try:
        with pytest.raises(ReachedReview):
            run_agent.AIAgent.run_conversation(actor, "hello", task_id="bound")
        assert calls == [(actor, "bound")]
    finally:
        canonical_method_entry_preflight.reset(token)


def test_identity_proves_ancestor_of_live_head_without_writer_access(ingress, monkeypatch):
    db = ingress.runner.session_store._db
    assert db is not None
    db.create_session("lineage-root", source="telegram")
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("lineage-root", ingress.entry.session_id),
    ))
    ingress.binding.session_id = "lineage-root"
    before = (
        db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        db._conn.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0],
    )
    opened = []
    original_init = SessionDB.__init__

    def track_open(self, *args, **kwargs):
        opened.append(kwargs.get("read_only"))
        return original_init(self, *args, **kwargs)

    monkeypatch.setattr(SessionDB, "__init__", track_open)
    monkeypatch.setattr(db, "get_session", lambda *a: pytest.fail("writer read forbidden"))

    async def exercise():
        client = TestClient(TestServer(_app(ingress.adapter)))
        await client.start_server()
        try:
            response = await client.get(_IDENTITY_ROUTE, headers={"Authorization": f"Bearer {_API_KEY}"})
            assert response.status == 200
            proof = await response.json()
            if sys.platform == "darwin":
                import psutil
                started = f"darwin-tv:{psutil.Process(os.getpid()).create_time():.6f}"
            else:
                started = "linux-clk:" + Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()[19]
            assert proof == {
                "session_id": ingress.entry.session_id,
                "lineage_root_digest": SessionDB._target_bind_lineage_root_digest("lineage-root"),
                "process_pid": os.getpid(), "process_started_at": started,
            }
        finally:
            await client.close()

    asyncio.run(exercise())
    assert opened and all(opened)
    assert before == (
        db._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        db._conn.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0],
    )
    assert ingress.agent.calls == []



def test_identity_rejects_in_place_live_head_drift_during_read_only_proof(ingress, monkeypatch):
    store = ingress.runner.session_store
    db = store._db
    assert db is not None
    original_head = ingress.entry.session_id
    db.create_session("other-live-head", source="telegram")
    original_tip = SessionDB.get_compression_tip
    proof_reads = []

    def drift_during_read(self, session_id):
        if self is not db:
            assert self.read_only
            assert session_id == original_head
            with store._lock:
                assert store._entries[ingress.binding.session_key] is ingress.entry
                ingress.entry.session_id = "other-live-head"
            proof_reads.append(session_id)
        return original_tip(self, session_id)

    monkeypatch.setattr(SessionDB, "get_compression_tip", drift_during_read)

    async def exercise():
        client = TestClient(TestServer(_app(ingress.adapter)))
        await client.start_server()
        try:
            response = await client.get(_IDENTITY_ROUTE, headers={"Authorization": f"Bearer {_API_KEY}"})
            assert response.status == 409
            assert await response.json() == {
                "error": {"code": "canonical_binding_stale", "message": "Canonical request rejected."}
            }
        finally:
            await client.close()

    asyncio.run(exercise())
    assert proof_reads == [original_head]
    assert store.lookup_by_session_key_existing(ingress.binding.session_key) is ingress.entry
    assert ingress.entry.session_id == "other-live-head"
    assert ingress.agent.calls == []



def test_identity_rejects_any_query_string_even_empty(ingress):
    async def exercise():
        server = TestServer(_app(ingress.adapter))
        await server.start_server()
        try:
            for suffix in ("?", "?session_id=ignored"):
                reader, writer = await asyncio.open_connection(server.host, server.port)
                try:
                    writer.write((
                        f"GET {_IDENTITY_ROUTE}{suffix} HTTP/1.1\r\n"
                        f"Host: {server.host}\r\n"
                        f"Authorization: Bearer {_API_KEY}\r\n"
                        "Connection: close\r\n\r\n"
                    ).encode("ascii"))
                    await writer.drain()
                    response = await reader.read()
                    status_line, _, body = response.partition(b"\r\n\r\n")
                    assert status_line.split(b"\r\n", 1)[0] == b"HTTP/1.1 400 Bad Request"
                    assert json.loads(body)["error"]["code"] == "canonical_invalid_request"
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            await server.close()

    asyncio.run(exercise())
