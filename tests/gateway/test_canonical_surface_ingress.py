"""Behavioral API contract for the canonical existing-only ingress."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
