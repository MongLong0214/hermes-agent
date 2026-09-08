"""Behavioral API contract for the canonical existing-only ingress."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_API_KEY = "canonical-surface-test-key"
_ROUTE = "/v1/canonical-surface/events"


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
    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail("new actor forbidden"))
    yield SimpleNamespace(runner=runner, agent=agent, adapter=adapter, send=send, binding=binding, entry=entry)
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


def test_missing_agent_receipt_replays_after_cache_is_restored(ingress):
    async def exercise():
        cached = ingress.runner._agent_cache.pop(ingress.entry.session_key)
        original = await ingress.send()
        assert (original[0], original[1]["error"]["code"]) == (409, "canonical_agent_missing")
        ingress.runner._agent_cache[ingress.entry.session_key] = cached
        assert await ingress.send() == original
        assert ingress.agent.calls == []
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
