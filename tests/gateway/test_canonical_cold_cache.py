"""Cold canonical ingress restores an existing actor, never a session row."""
from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.fixture
def cold(tmp_path, monkeypatch):
    import run_agent
    import gateway.run as gateway_run

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(run_agent, "_hermes_home", tmp_path)
    config_data = {
        "sessions_dir": str(tmp_path / "sessions"),
        "model": {"default": "test/model"},
        "agent": {"disabled_toolsets": ["all"]},
        "gateway": {"platforms": {"telegram": {"skip_context_files": True}}},
        "compression": {"enabled": False, "in_place": True},
        "session_title": {"enabled": False},
    }
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: copy.deepcopy(config_data))
    initial = GatewayRunner(GatewayConfig.from_dict(config_data))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = initial.session_store.get_or_create_session(source)
    db = initial.session_store._db
    db.set_session_title(entry.session_id, "Existing canonical conversation")
    db.update_system_prompt(entry.session_id, "Persisted canonical system prefix.")
    db.append_message(entry.session_id, role="user", content="remember this prefix")
    db.append_message(entry.session_id, role="assistant", content="remembered")
    config_data["canonical_surface_bindings"] = {"bound": {
        "session_key": entry.session_key, "session_id": entry.session_id,
        "telegram": {"chat_id": "chat", "chat_type": "dm", "user_id": "user", "thread_id": None},
        "buzz": {"author_ids": ["author"], "channel_ids": ["channel"]},
    }}
    initial.session_store.close_all_db_handles()
    runner = GatewayRunner(GatewayConfig.from_dict(config_data))
    # Gateway startup loads the persisted routing index, not an AIAgent.
    assert runner.session_store.lookup_by_session_key(entry.session_key).session_id == entry.session_id
    db = runner.session_store._db
    assert runner._agent_cache == {}
    before_ids = {r[0] for r in db._conn.execute("SELECT id FROM sessions")}
    provider_calls = []
    constructors = []
    client = MagicMock()

    def complete(**kwargs):
        provider_calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="restored terminal", tool_calls=None), finish_reason="stop",
        )], model="test/model", usage=None)

    client.chat.completions.create.side_effect = complete
    monkeypatch.setattr(run_agent, "OpenAI", lambda **kw: client)
    monkeypatch.setattr(run_agent, "get_tool_definitions", lambda *a, **kw: [])
    monkeypatch.setattr(run_agent, "check_toolset_requirements", lambda *a, **kw: {})
    monkeypatch.setattr("agent.model_metadata.fetch_model_metadata", lambda *a, **kw: {})
    monkeypatch.setattr(runner, "_resolve_session_agent_runtime", lambda **kw: (
        "test/model", {"api_key": "test-only-key", "base_url": "https://provider.invalid/v1", "provider": "openai"},
    ))
    real_agent = run_agent.AIAgent

    class ConstructedAgent(real_agent):
        def __init__(self, **kwargs):
            constructors.append(kwargs.copy())
            super().__init__(**kwargs, skip_memory=True, skip_background_review=True)
            self.client = client
            self.compression_enabled = False
            self.compression_in_place = True
            self.save_trajectories = False

    errors = []
    restore = runner._restore_bound_existing_agent

    def checked_restore(*args):
        try:
            return restore(*args)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            raise

    monkeypatch.setattr(runner, "_restore_bound_existing_agent", checked_restore)
    monkeypatch.setattr(run_agent, "AIAgent", ConstructedAgent)
    monkeypatch.setattr(db, "create_session", lambda *a, **kw: pytest.fail("new session primitive forbidden"))
    monkeypatch.setattr(runner.session_store, "get_or_create_session", lambda *a, **kw: pytest.fail("new routing session forbidden"))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter.gateway_runner = runner
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    payload = dict(binding="bound", event_id="cold", author_id="author", channel_id="channel", text="continue the same session")
    state = SimpleNamespace(errors=errors, runner=runner, db=db, entry=entry, app=app, payload=payload,
                            constructors=constructors, provider_calls=provider_calls, before_ids=before_ids)
    yield state
    adapter._response_store.close()
    runner.session_store.close_all_db_handles()


async def telegram_handoff(cold, monkeypatch, *, external=False):
    """Enter the real Telegram cache guard with the cold actor's signature."""
    import gateway.run as gateway_run
    import run_agent
    from gateway.run import TurnRunner
    from gateway.turn_context import TurnContext

    runner = cold.runner
    cached = runner._agent_cache[cold.entry.session_key]
    kwargs = cold.constructors[0]
    ctx = TurnContext(
        source=cold.entry.origin, message="continue on Telegram",
        history=cold.db.get_messages_as_conversation(cold.entry.session_id),
        context_prompt=kwargs["ephemeral_system_prompt"],
        session_id=cold.entry.session_id, session_key=cold.entry.session_key,
        user_config=gateway_run._load_gateway_config(),
        enabled_toolsets=kwargs["enabled_toolsets"],
        disabled_toolsets=kwargs["disabled_toolsets"], AIAgent=run_agent.AIAgent,
        resolve_display_setting=lambda *_: False, _run_still_current=lambda: True,
        _hooks_ref=runner.hooks,
    )
    signature = runner._agent_config_signature

    def verify_signature(*args, **kw):
        actual = signature(*args, **kw)
        assert actual == cached[1], "handoff configuration drift"
        return actual

    class ExternalInvalidation(Exception):
        pass

    def rebuild(*args, **kw):
        if external:
            raise ExternalInvalidation
        pytest.fail(f"Telegram rebuilt cold actor: cache={cached[2]}, "
                    f"DB={cold.db.get_session(cold.entry.session_id)['message_count']}")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "_agent_config_signature", verify_signature)
        patch.setattr(runner, "_construct_gateway_agent", rebuild)
        patch.setattr(runner, "_release_evicted_agent_soft", lambda actor: None)
        if external:
            with pytest.raises(ExternalInvalidation):
                await asyncio.to_thread(TurnRunner(runner, ctx).run_sync)
            assert cold.entry.session_key not in runner._agent_cache
        else:
            await asyncio.to_thread(TurnRunner(runner, ctx).run_sync)
            assert runner._agent_cache[cold.entry.session_key][0] is cached[0]


@pytest.mark.parametrize("external_at", [None, "before", "during", "after"])
def test_cold_to_telegram_owns_only_its_persisted_count(cold, monkeypatch, external_at):
    def external_write():
        cold.db.append_message(cold.entry.session_id, role="user", content="external writer")

    construct = cold.runner._construct_gateway_agent

    def observed(*args, **kwargs):
        actor = construct(*args, **kwargs)
        run = actor.run_conversation

        def run_observed(*a, **kw):
            if external_at == "before":
                external_write()
            result = run(*a, **kw)
            if external_at == "during":
                external_write()
            return result

        actor.run_conversation = run_observed
        return actor

    monkeypatch.setattr(cold.runner, "_construct_gateway_agent", observed)

    async def exercise():
        async with TestClient(TestServer(cold.app)) as client:
            assert (await post(client, cold.payload))[0] == 200, cold.errors
            if external_at is None:
                # Also preserve a canonical warm turn before ordinary Telegram.
                assert (await post(client, {**cold.payload, "event_id": "warm"}))[0] == 200
        if external_at == "after":
            external_write()
        await telegram_handoff(cold, monkeypatch, external=external_at is not None)
        if external_at is None:
            assert len(cold.constructors) == 1
            assert len(cold.provider_calls) == 3
            for before, after in zip(cold.provider_calls, cold.provider_calls[1:]):
                assert after["messages"][:len(before["messages"])] == before["messages"]
                assert after.get("tools") == before.get("tools")

    asyncio.run(exercise())


@pytest.mark.parametrize("external", [False, True])
def test_cancelled_late_persistence_reconciles_before_lease_release(cold, monkeypatch, external):
    import threading
    from gateway.canonical_surface import CanonicalIngressEvent, request_local_reply_sink

    async def exercise():
        runner = cold.runner
        loop = asyncio.get_running_loop()
        entered, interrupted = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        construct = runner._construct_gateway_agent
        release_lease = runner._turn_leases.release
        counts_at_release = []

        def observed(*args, **kwargs):
            actor = construct(*args, **kwargs)
            flush = actor._flush_messages_to_session_db
            interrupt = actor.interrupt
            blocked = False

            def late_flush(messages, *a, **kw):
                nonlocal blocked
                if not blocked and any(m.get("content") == "restored terminal" for m in messages):
                    blocked = True
                    loop.call_soon_threadsafe(entered.set)
                    assert release.wait(10)
                    if external:
                        cold.db.append_message(cold.entry.session_id, role="user", content="external during cancel")
                return flush(messages, *a, **kw)

            def stop(**kw):
                interrupt(**kw)
                interrupted.set()

            actor._flush_messages_to_session_db = late_flush
            actor.interrupt = stop
            return actor

        def observe_release(token):
            counts_at_release.append((
                runner._agent_cache[cold.entry.session_key][2],
                cold.db.get_session(cold.entry.session_id)["message_count"],
            ))
            return release_lease(token)

        monkeypatch.setattr(runner, "_construct_gateway_agent", observed)
        monkeypatch.setattr(runner._turn_leases, "release", observe_release)
        task = asyncio.create_task(runner.run_bound_existing_turn(
            runner.config.canonical_surface_bindings["bound"], CanonicalIngressEvent(**cold.payload),
            cold.entry, reply_sink=request_local_reply_sink(lambda result: None),
        ))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            task.cancel()
            await asyncio.wait_for(interrupted.wait(), 10)
            assert not task.done()
            assert counts_at_release == []
            assert runner._turn_leases._leases[cold.entry.session_id].holder is not None
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        assert task.cancelled()
        assert counts_at_release == ([(2, 5)] if external else [(4, 4)])
        assert cold.db.get_messages_as_conversation(cold.entry.session_id)[-1]["content"] == "restored terminal"
        await telegram_handoff(cold, monkeypatch, external=external)
        if not external:
            assert len(cold.constructors) == 1
            before, after = cold.provider_calls
            assert after["messages"][:len(before["messages"])] == before["messages"]
            assert after.get("tools") == before.get("tools")

    asyncio.run(exercise())


async def post(client, payload):
    response = await client.post("/v1/canonical-surface/events", headers={"Authorization": "Bearer test-key"}, json=payload)
    return response.status, await response.json()


def test_cold_http_restores_existing_id_and_replays_without_new_rows(cold):
    async def exercise():
        async with TestClient(TestServer(cold.app)) as client:
            result = await post(client, cold.payload)
            assert result == (200, {"event_id": "cold", "text": "restored terminal"}), cold.errors
            assert await post(client, cold.payload) == result
        assert len(cold.constructors) == 1
        assert cold.constructors[0]["session_id"] == cold.entry.session_id
        assert len(cold.provider_calls) == 1
        messages = cold.provider_calls[0]["messages"]
        assert any(m.get("content") == "remember this prefix" for m in messages)
        assert any(m.get("content") == "remembered" for m in messages)
        assert {r[0] for r in cold.db._conn.execute("SELECT id FROM sessions")} == cold.before_ids
        assert cold.runner._agent_cache[cold.entry.session_key][0].session_id == cold.entry.session_id
        stored = cold.db.get_messages_as_conversation(cold.entry.session_id)
        assert [m["role"] for m in stored] == ["user", "assistant", "user", "assistant"]
    asyncio.run(exercise())



def test_waiting_cold_ingress_does_not_invalidate_running_turn(cold, monkeypatch):
    async def exercise():
        runner = cold.runner
        generation = runner._begin_session_run_generation(cold.entry.session_key)
        lease = await runner._turn_leases.acquire(cold.entry.session_id, owner_key="telegram", generation=generation)
        waiting = asyncio.Event()
        real_acquire = runner._turn_leases.acquire

        async def observe(*args, **kwargs):
            waiting.set()
            return await real_acquire(*args, **kwargs)

        monkeypatch.setattr(runner._turn_leases, "acquire", observe)
        async with TestClient(TestServer(cold.app)) as client:
            task = asyncio.create_task(post(client, cold.payload))
            try:
                await asyncio.wait_for(waiting.wait(), 10)
                assert runner._is_session_run_current(cold.entry.session_key, generation)
                assert cold.constructors == []
                assert cold.provider_calls == []
            finally:
                runner._turn_leases.release(lease)
                result = await task
            assert result[0] == 200
    asyncio.run(exercise())


def test_cancelled_constructor_keeps_lease_until_cleanup(cold, monkeypatch):
    import threading
    from gateway.canonical_surface import CanonicalIngressEvent, request_local_reply_sink

    async def exercise():
        runner = cold.runner
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        construct = runner._construct_gateway_agent
        disposed = []
        real_release = runner._release_evicted_agent_soft

        def blocked(*args, **kwargs):
            agent = construct(*args, **kwargs)
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10)
            return agent

        def dispose(agent):
            disposed.append(agent)
            real_release(agent)

        monkeypatch.setattr(runner, "_construct_gateway_agent", blocked)
        monkeypatch.setattr(runner, "_release_evicted_agent_soft", dispose)
        event = CanonicalIngressEvent(**cold.payload)
        binding = runner.config.canonical_surface_bindings["bound"]
        task = asyncio.create_task(runner.run_bound_existing_turn(
            binding, event, cold.entry, reply_sink=request_local_reply_sink(lambda result: None),
        ))
        contender = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            task.cancel()
            checkpoint = asyncio.Event()
            loop.call_soon(checkpoint.set)
            await checkpoint.wait()
            assert not task.done(), "constructor outlived released turn ownership"
            assert runner._turn_leases._leases[cold.entry.session_id].holder is not None
            contender = asyncio.create_task(runner._turn_leases.acquire(
                cold.entry.session_id, owner_key="next", generation=0,
            ))
            task.cancel()
            checkpoint.clear()
            loop.call_soon(checkpoint.set)
            await checkpoint.wait()
            assert not contender.done()
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if contender:
                runner._turn_leases.release(await contender)
        assert task.cancelled()
        assert len(disposed) == 1
        assert runner._agent_cache == {}
        assert cold.provider_calls == []
        assert len(cold.db.get_messages_as_conversation(cold.entry.session_id)) == 2
    asyncio.run(exercise())


def test_concurrent_cold_then_warm_preserves_actor_and_prompt_prefix(cold, monkeypatch):
    import threading

    async def exercise():
        runner = cold.runner
        loop = asyncio.get_running_loop()
        entered, waiting = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        construct = runner._construct_gateway_agent
        acquire = runner._turn_leases.acquire
        attempts = 0

        def blocked(*args, **kwargs):
            agent = construct(*args, **kwargs)
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(10)
            return agent

        async def observe(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                waiting.set()
            return await acquire(*args, **kwargs)

        monkeypatch.setattr(runner, "_construct_gateway_agent", blocked)
        monkeypatch.setattr(runner._turn_leases, "acquire", observe)
        async with TestClient(TestServer(cold.app)) as client:
            first = asyncio.create_task(post(client, cold.payload))
            second = None
            try:
                await asyncio.wait_for(entered.wait(), 10)
                second = asyncio.create_task(post(client, {**cold.payload, "event_id": "second", "text": "second turn"}))
                await asyncio.wait_for(waiting.wait(), 10)
                assert len(cold.constructors) == 1
                assert not second.done()
                assert cold.provider_calls == []
            finally:
                release.set()
                results = await asyncio.gather(*([first, second] if second else [first]))
            assert [r[0] for r in results] == [200, 200], cold.errors
            actor = runner._agent_cache[cold.entry.session_key][0]
            # A warm turn must not resolve new configuration or reconstruct.
            monkeypatch.setattr(runner, "_restore_bound_existing_agent", lambda *a: pytest.fail("warm restore"))
            third = await post(client, {**cold.payload, "event_id": "third", "text": "third turn"})
            assert third[0] == 200
            assert runner._agent_cache[cold.entry.session_key][0] is actor
        assert len(cold.constructors) == 1
        assert len(cold.provider_calls) == 3
        for before, after in zip(cold.provider_calls, cold.provider_calls[1:]):
            assert after["messages"][:len(before["messages"])] == before["messages"]
            assert after.get("tools") == before.get("tools")
        assert {r[0] for r in cold.db._conn.execute("SELECT id FROM sessions")} == cold.before_ids
    asyncio.run(exercise())


@pytest.mark.parametrize("changes", [
    {"binding": "unknown"}, {"author_id": "wrong"}, {"channel_id": "wrong"},
    {"session_id": "caller-target"},
])
def test_cold_denial_has_no_constructor_provider_or_transcript_mutation(cold, changes):
    async def exercise():
        async with TestClient(TestServer(cold.app)) as client:
            status, _ = await post(client, {**cold.payload, **changes})
            assert status in {400, 403, 404, 409}
        assert cold.constructors == []
        assert cold.provider_calls == []
        assert cold.runner._agent_cache == {}
        assert len(cold.db.get_messages_as_conversation(cold.entry.session_id)) == 2
        assert {r[0] for r in cold.db._conn.execute("SELECT id FROM sessions")} == cold.before_ids
    asyncio.run(exercise())


def test_binding_stales_during_constructor_disposes_without_publish(cold, monkeypatch):
    construct = cold.runner._construct_gateway_agent
    disposed = []
    release = cold.runner._release_evicted_agent_soft

    def stale(*args, **kwargs):
        agent = construct(*args, **kwargs)
        cold.db._execute_write(lambda conn: conn.execute(
            "UPDATE sessions SET ended_at = 1 WHERE id = ?", (cold.entry.session_id,),
        ))
        return agent

    def dispose(agent):
        disposed.append(agent)
        release(agent)

    monkeypatch.setattr(cold.runner, "_construct_gateway_agent", stale)
    monkeypatch.setattr(cold.runner, "_release_evicted_agent_soft", dispose)

    async def exercise():
        async with TestClient(TestServer(cold.app)) as client:
            status, body = await post(client, cold.payload)
            assert status == 409
            assert "canonical_binding_stale" in str(body)
        assert len(disposed) == 1
        assert cold.provider_calls == []
        assert cold.runner._agent_cache == {}
        assert len(cold.db.get_messages_as_conversation(cold.entry.session_id)) == 2
    asyncio.run(exercise())


def test_cancelled_running_actor_retains_lease_callbacks_and_eviction_protection(cold, monkeypatch):
    import threading
    from gateway.canonical_surface import CanonicalIngressEvent, request_local_reply_sink

    async def exercise():
        runner = cold.runner
        loop = asyncio.get_running_loop()
        entered, interrupted = asyncio.Event(), asyncio.Event()
        release = threading.Event()
        construct = runner._construct_gateway_agent
        actors = []
        callback = lambda *a: pytest.fail("outward callback")

        def observed(*args, **kwargs):
            actor = construct(*args, **kwargs)
            actors.append(actor)
            actor.step_callback = callback
            run = actor.run_conversation
            interrupt = actor.interrupt

            def blocked(*a, **kw):
                loop.call_soon_threadsafe(entered.set)
                assert release.wait(10)
                assert actor.step_callback is None
                return run(*a, **kw)

            def stop(**kwargs):
                interrupt(**kwargs)
                interrupted.set()

            actor.run_conversation = blocked
            actor.interrupt = stop
            return actor

        monkeypatch.setattr(runner, "_construct_gateway_agent", observed)
        # Zero cap deliberately challenges the just-published actor too.
        monkeypatch.setattr(runner, "_agent_cache_cap", lambda: 0)
        event = CanonicalIngressEvent(**cold.payload)
        task = asyncio.create_task(runner.run_bound_existing_turn(
            runner.config.canonical_surface_bindings["bound"], event, cold.entry,
            reply_sink=request_local_reply_sink(lambda result: None),
        ))
        contender = None
        try:
            await asyncio.wait_for(entered.wait(), 10)
            actor = actors[0]
            assert runner._agent_cache[cold.entry.session_key][0] is actor
            with runner._agent_cache_lock:
                runner._enforce_agent_cache_cap()
            assert runner._agent_cache[cold.entry.session_key][0] is actor
            task.cancel()
            await asyncio.wait_for(interrupted.wait(), 10)
            contender = asyncio.create_task(runner._turn_leases.acquire(
                cold.entry.session_id, owner_key="next", generation=0,
            ))
            task.cancel()
            checkpoint = asyncio.Event()
            loop.call_soon(checkpoint.set)
            await checkpoint.wait()
            assert not task.done() and not contender.done()
            assert actor.step_callback is None
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            if contender:
                runner._turn_leases.release(await contender)
        assert task.cancelled()
        assert actors[0].step_callback is callback
        assert runner._canonical_active_agents == {}
        assert cold.provider_calls == []
    asyncio.run(exercise())


def test_cancelled_provider_worker_cannot_outlive_turn_lease(cold, monkeypatch):
    import threading
    import agent.chat_completion_helpers as helpers
    from gateway.canonical_surface import CanonicalIngressEvent, request_local_reply_sink

    async def exercise():
        runner = cold.runner
        loop = asyncio.get_running_loop()
        entered, detached = asyncio.Event(), asyncio.Event()
        release, finished = threading.Event(), threading.Event()
        construct = runner._construct_gateway_agent
        release_lease = runner._turn_leases.release
        released_while_provider_live = []

        def observed(*args, **kwargs):
            actor = construct(*args, **kwargs)
            complete = actor.client.chat.completions.create.side_effect

            def blocked(**kw):
                loop.call_soon_threadsafe(entered.set)
                try:
                    assert release.wait(10)
                    return complete(**kw)
                finally:
                    finished.set()

            actor.client.chat.completions.create.side_effect = blocked
            run = actor.run_conversation

            def run_observed(*a, **kw):
                try:
                    return run(*a, **kw)
                finally:
                    loop.call_soon_threadsafe(detached.set)

            actor.run_conversation = run_observed
            return actor

        def lease_release(token):
            released_while_provider_live.append(not finished.is_set())
            return release_lease(token)

        monkeypatch.setattr(runner, "_construct_gateway_agent", observed)
        monkeypatch.setattr(runner._turn_leases, "release", lease_release)
        # The real worker remains blocked after socket abort, as an SDK can.
        monkeypatch.setattr(helpers, "_join_worker_for_relay_teardown", lambda *a, **kw: None)
        task = asyncio.create_task(runner.run_bound_existing_turn(
            runner.config.canonical_surface_bindings["bound"], CanonicalIngressEvent(**cold.payload),
            cold.entry, reply_sink=request_local_reply_sink(lambda result: None),
        ))
        try:
            await asyncio.wait_for(entered.wait(), 10)
            task.cancel()
            await asyncio.wait_for(detached.wait(), 10)
            checkpoint = asyncio.Event()
            loop.call_soon(checkpoint.set)
            await checkpoint.wait()
            assert not task.done()
            assert runner._turn_leases._leases[cold.entry.session_id].holder is not None
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(finished.wait, 10)
        assert task.cancelled()
        assert released_while_provider_live == [False]
    asyncio.run(exercise())
