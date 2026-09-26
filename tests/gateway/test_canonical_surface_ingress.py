"""Behavioral API contract for the canonical existing-only ingress."""

from __future__ import annotations

import asyncio
import json
import os
import sys
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
    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail("new actor forbidden"))
    yield SimpleNamespace(runner=runner, agent=agent, adapter=adapter, send=send, binding=binding, entry=entry)
    adapter._response_store.close()
    runner.session_store.close_all_db_handles()


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
