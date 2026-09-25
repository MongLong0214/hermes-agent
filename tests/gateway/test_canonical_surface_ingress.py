"""Authenticated canonical-surface ingress (``POST /v1/canonical-surface/events``).

Every refusal code is answered before a receipt is claimed, and one event id runs its turn at most
once: a replay answers from the durable receipt, a changed payload conflicts, and past the claim the
first answer is the one a replay gives — the recorded terminal, or uncertainty that never runs again.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

from gateway.canonical_surface import CanonicalSurfaceBinding
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.run import _INTERRUPT_REASON_RESET, _INTERRUPT_REASON_STOP, GatewayRunner
from gateway.session import SessionSource


_ROUTE = "/v1/canonical-surface/events"
_TERMINAL = "request-owned terminal"


class _CachedActor:
    """Stands in for the bound cached AIAgent: deterministic, no model call."""

    compression_in_place = True

    def __init__(self, session_id, db):
        self.session_id = session_id
        self._session_db = db
        self.calls = []
        self.fail_with = None
        self.during_turn = None

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass

    def run_conversation(self, text, *, conversation_history, task_id):
        self.calls.append(text)
        if self.during_turn is not None:
            self.during_turn()
        if self.fail_with is not None:
            raise self.fail_with
        self._persist_user_message_idx = len(conversation_history)
        return {"completed": True, "session_id": task_id, "final_response": _TERMINAL,
                "messages": [*conversation_history, {"role": "user", "content": text},
                             {"role": "assistant", "content": _TERMINAL}]}


@pytest.fixture
def ingress(tmp_path, monkeypatch):
    """Real runner, resolver, coordinator and SQLite under a temp home; no model."""
    home, user_home = tmp_path / "hermes-home", tmp_path / "user"
    home.mkdir()
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    actor = _CachedActor(entry.session_id, db)
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)
    import run_agent
    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail("canonical ingress built an actor"))
    key = secrets.token_hex(32)

    def receipts():
        # Its own read-only connection, so the durable rows stay readable across a closed handle.
        with contextlib.closing(sqlite3.connect(db.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            rows = conn.execute(
                "SELECT value FROM state_meta WHERE key LIKE 'canonical-receipt:%'").fetchall()
        return [json.loads(row[0]) for row in rows]

    runners = [runner]
    yield SimpleNamespace(runner=runner, actor=actor, key=key, receipts=receipts, entry=entry,
                          source=source, binding=binding, home=home, runners=runners)
    for each in runners:
        each.session_store.close_all_db_handles()


@pytest.fixture
def send(ingress):
    """POST one event to the registered route handler, as the authenticated caller by default."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": ingress.key}))
    adapter.gateway_runner = ingress.runner
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == ("POST", _ROUTE)]

    async def post(body=None, *, bearer=ingress.key, profile=None, **changes):
        raw = body if body is not None else json.dumps(_payload(**changes)).encode()

        async def read():
            return raw
        headers = {"Authorization": f"Bearer {bearer}"} if bearer is not None else {}
        # The 401 audit log reads method/path/transport off the request.
        request = SimpleNamespace(headers=headers, read=read, method="POST", path_qs=_ROUTE, transport=None)
        token = _api_request_profile.set(profile)
        try:
            response = await handler(request)
        finally:
            _api_request_profile.reset(token)
        return response.status, json.loads(response.text)

    post.adapter = adapter
    yield post
    adapter._response_store.close()


def _payload(**changes):
    return {"binding": "canonical", "event_id": "event", "author_id": "author",
            "channel_id": "channel", "text": "hello", **changes}


def _code(result):
    return result[0], result[1]["error"]["code"]


def _rebuild_evicted_actor(ingress):
    """/new and /stop evict the cached actor; the next ordinary turn rebuilds it on the same DB."""
    with ingress.runner._agent_cache_lock:
        assert ingress.entry.session_key not in ingress.runner._agent_cache
        ingress.runner._agent_cache[ingress.entry.session_key] = (
            ingress.actor, "exact", 0, ingress.entry.session_id)


def test_refusal_paths_never_reach_the_cached_actor(ingress, send, monkeypatch):
    async def exercise():
        assert _code(await send(binding="not-configured")) == (404, "canonical_binding_unknown")
        assert _code(await send(author_id="stranger")) == (403, "canonical_principal_rejected")
        assert _code(await send(channel_id="elsewhere")) == (403, "canonical_principal_rejected")
        assert _code(await send(text="x" * 16_385)) == (400, "canonical_invalid_request")
        heartbeat = json.dumps({"binding": "canonical", "event_id": "tick", "text": "[System: Heartbeat]"})
        assert _code(await send(heartbeat.encode())) == (400, "canonical_invalid_request")
        assert _code(await send(bearer=None)) == (401, "gateway_auth_failed")
        assert _code(await send(bearer="not-the-key")) == (401, "gateway_auth_failed")
        # Another served profile's own key does not reach the launch profile's bindings.
        other_key = secrets.token_hex(32)
        monkeypatch.setenv("API_SERVER_KEY", other_key)
        assert _code(await send(bearer=other_key, profile="other")) == (404, "canonical_binding_unknown")
        # A normal turn holding the session's slot refuses before any claim, so its code is exact.
        turn = ingress.runner._session_state(ingress.entry.session_key).turn
        turn.agent = object()
        assert _code(await send()) == (409, "canonical_turn_busy")
        turn.agent = None
        assert ingress.actor.calls == [] and ingress.receipts() == []

        # Nothing was stored for a refusal, so the same event runs once the slot is free.
        first = await send()
        assert first == (200, {"event_id": "event", "text": _TERMINAL})
        assert _code(await send(text="changed payload")) == (409, "canonical_event_conflict")
        assert _code(await send(author_id="stranger")) == (403, "canonical_principal_rejected")
        assert await send() == first
        assert ingress.actor.calls == ["hello"]

        # A turn that fails after its claim leaves the receipt pending: reported as uncertainty
        # with no detail, and a replay never runs it again.
        ingress.actor.fail_with = RuntimeError("private execution detail")
        failed = await send(event_id="event-2")
        assert _code(failed) == (409, "canonical_event_uncertain")
        assert "private execution detail" not in json.dumps(failed[1])
        ingress.actor.fail_with = None
        assert await send(event_id="event-2") == failed
        assert ingress.actor.calls == ["hello", "hello"]

    asyncio.run(exercise())


@pytest.mark.parametrize("mid_turn", ["new_command", "actor_replaced"])
def test_head_check_failing_after_the_claim_is_uncertain_on_first_answer_and_replay(ingress, send, mid_turn):
    """Regression: a claimed turn whose head check failed answered its refusal code (409
    canonical_turn_interrupted after /new, canonical_agent_replaced after an actor swap), which reads
    as "not run"; a client resending under a new event id then ran the turn a second time. The same
    codes are exact before a claim, so only the coordinator can tell the two apart."""
    replacement = _CachedActor(ingress.entry.session_id, ingress.actor._session_db)

    async def exercise():
        loop = asyncio.get_running_loop()

        def new_command():
            # The real /new interrupt, on the loop, while the worker is still inside the actor.
            asyncio.run_coroutine_threadsafe(ingress.runner._interrupt_and_clear_session(
                ingress.entry.session_key, ingress.source,
                interrupt_reason=_INTERRUPT_REASON_RESET, invalidation_reason="new_command",
            ), loop).result(timeout=5)

        def actor_replaced():
            with ingress.runner._agent_cache_lock:
                ingress.runner._agent_cache[ingress.entry.session_key] = (
                    replacement, "exact", 0, ingress.entry.session_id)

        ingress.actor.during_turn = {"new_command": new_command, "actor_replaced": actor_replaced}[mid_turn]
        first = await send()
        ingress.actor.during_turn = None
        if mid_turn == "new_command":
            _rebuild_evicted_actor(ingress)
        assert _code(first) == (409, "canonical_event_uncertain")
        assert await send() == first
        assert ingress.actor.calls == ["hello"] and replacement.calls == []
        assert [receipt["state"] for receipt in ingress.receipts()] == ["pending"]

    asyncio.run(exercise())


def test_terminal_recorded_before_a_stop_is_the_first_answer_and_the_replay(ingress, send, monkeypatch):
    runner, session_key = ingress.runner, ingress.entry.session_key
    db = ingress.actor._session_db
    record_terminal = db.compare_and_set_meta

    def record_then_stop(*args, **kwargs):
        recorded = record_terminal(*args, **kwargs)
        # /stop's interrupt and slot release, landing just after the terminal became durable.
        generation = runner._interrupt_running_turn(
            session_key, interrupt_reason=_INTERRUPT_REASON_STOP, invalidation_reason="stop_command")
        runner._drop_turn_slot(session_key, run_generation=generation)
        return recorded

    monkeypatch.setattr(db, "compare_and_set_meta", record_then_stop)

    async def exercise():
        first = await send()
        _rebuild_evicted_actor(ingress)
        assert first == (200, {"event_id": "event", "text": _TERMINAL})
        assert await send() == first
        assert ingress.actor.calls == ["hello"]
        assert [receipt["state"] for receipt in ingress.receipts()] == ["terminal"]

    asyncio.run(exercise())


@pytest.mark.parametrize("actor_gone", ["stop_evicts", "reopened_db", "restart"])
def test_replay_without_the_cached_actor_answers_from_the_recorded_receipt(ingress, send, actor_gone):
    """Regression: a replay that found no cached actor answered 409 canonical_binding_stale, which
    reads as "not run"; a client resending under a new event id then ran the event a second time
    once an ordinary turn rebuilt the actor. A recorded event answers from its receipt instead."""
    runner, session_key = ingress.runner, ingress.entry.session_key

    async def lose_the_actor():
        if actor_gone == "stop_evicts":
            await runner._interrupt_and_clear_session(
                session_key, ingress.source, interrupt_reason=_INTERRUPT_REASON_STOP,
                invalidation_reason="stop_command")
        elif actor_gone == "reopened_db":
            from hermes_state import SessionDB
            store = runner.session_store
            path = store._db.db_path
            store._db.close()
            store._db = SessionDB(path)
            runner._agent_cache.clear()
        else:
            runner.session_store.close_all_db_handles()
            restarted = GatewayRunner(GatewayConfig(sessions_dir=ingress.home / "sessions"))
            restarted.config.canonical_surface_bindings = {ingress.binding.name: ingress.binding}
            ingress.runners.append(restarted)
            send.adapter.gateway_runner = restarted
        current = send.adapter.gateway_runner
        with current._agent_cache_lock:
            assert session_key not in current._agent_cache

    async def exercise():
        first = await send()
        assert first == (200, {"event_id": "event", "text": _TERMINAL})
        ingress.actor.fail_with = RuntimeError("private execution detail")
        uncertain = await send(event_id="event-2")
        assert _code(uncertain) == (409, "canonical_event_uncertain")
        ingress.actor.fail_with = None

        await lose_the_actor()
        assert await send() == first
        assert await send(event_id="event-2") == uncertain
        assert _code(await send(text="changed payload")) == (409, "canonical_event_conflict")
        # Only an event nothing was recorded for is refused, and the refusal claims nothing.
        assert _code(await send(event_id="never-seen")) == (409, "canonical_binding_stale")
        assert ingress.actor.calls == ["hello", "hello"]
        assert sorted(receipt["state"] for receipt in ingress.receipts()) == ["pending", "terminal"]

    asyncio.run(exercise())


def test_loopback_http_ingress_claims_once_and_replays_the_durable_terminal(ingress):
    """Real listener on 127.0.0.1, real HTTP client, real coordinator and SQLite; stub actor."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "host": "127.0.0.1", "port": 0, "key": ingress.key}))
    adapter.gateway_runner = ingress.runner

    async def exercise():
        assert await adapter.connect()
        try:
            port = adapter._site._server.sockets[0].getsockname()[1]
            url = f"http://127.0.0.1:{port}{_ROUTE}"
            auth = {"Authorization": f"Bearer {ingress.key}"}
            async with aiohttp.ClientSession() as client:
                async def post(payload, headers):
                    async with client.post(url, json=payload, headers=headers) as response:
                        return response.status, await response.json()

                for headers in ({}, {"Authorization": "Bearer wrong"}):
                    status, body = await post(_payload(), headers)
                    assert (status, body["error"]["code"]) == (401, "gateway_auth_failed")
                assert ingress.actor.calls == [] and ingress.receipts() == []

                accepted = await post(_payload(), auth)
                assert accepted == (200, {"event_id": "event", "text": _TERMINAL})
                [receipt] = ingress.receipts()
                assert receipt["state"] == "terminal"
                assert receipt["response"]["terminal_text"] == _TERMINAL

                assert await post(_payload(), auth) == accepted
                status, body = await post(_payload(text="changed payload"), auth)
                assert (status, body["error"]["code"]) == (409, "canonical_event_conflict")
                assert ingress.actor.calls == ["hello"]
                assert ingress.receipts() == [receipt]
        finally:
            await adapter.disconnect()

    asyncio.run(exercise())
