"""Behavioral contract for existing-only canonical surface ingress."""

from __future__ import annotations

import asyncio
from copy import deepcopy
import json
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    MAX_REQUEST_BYTES,
    body_limit_middleware,
    security_headers_middleware,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_API_KEY = "sk-canonical-surface-test-key-00000001"
_ROUTE = "/v1/canonical-surface/events"


def _binding_payload(entry, source: SessionSource) -> dict[str, Any]:
    return {
        "session_key": entry.session_key,
        "session_id": entry.session_id,
        "telegram": {
            "chat_id": source.chat_id,
            "chat_type": source.chat_type,
            "user_id": source.user_id,
            "thread_id": source.thread_id,
        },
        "buzz": {
            "author_ids": ["buzz-author-7"],
            "channel_ids": ["buzz-channel-9"],
        },
    }


def _app_for(adapter: APIServerAdapter) -> web.Application:
    middlewares = [
        middleware
        for middleware in (body_limit_middleware, security_headers_middleware)
        if middleware is not None
    ]
    app = web.Application(middlewares=middlewares)
    app["api_server_adapter"] = adapter
    app["gateway_runner"] = adapter.gateway_runner
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


class _CachedFakeAgent:
    def __init__(self, session_id: str, db) -> None:
        self.session_id = session_id
        self._db = db
        self.calls: list[dict[str, Any]] = []

    def run_conversation(
        self,
        message: str,
        *,
        conversation_history: list[dict[str, Any]],
        task_id: str,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "self": self,
                "message": message,
                "history": deepcopy(conversation_history),
                "task_id": task_id,
            }
        )
        self._persist_user_message_idx = len(conversation_history)
        self._db.append_message(task_id, role="user", content=message)
        self._db.append_message(task_id, role="assistant", content="terminal reply")
        return {
            "completed": True,
            "failed": False,
            "interrupted": False,
            "partial": False,
            "final_response": "terminal reply",
            "messages": [
                *conversation_history,
                {"role": "user", "content": message},
                {"role": "assistant", "content": "terminal reply"},
            ],
            "session_id": task_id,
        }


class _NoContactRunner:
    def __init__(self) -> None:
        object.__setattr__(self, "contacted", [])

    def __getattribute__(self, name: str):
        if name in {"contacted", "__class__"}:
            return object.__getattribute__(self, name)
        contacted = object.__getattribute__(self, "contacted")
        contacted.append(name)
        raise AssertionError(f"runner contacted before target rejection: {name}")


async def _scenario_authenticated_request_reuses_exact_existing_row_cache_and_lease(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    initial_config = GatewayConfig(sessions_dir=hermes_home / "sessions")
    runner = GatewayRunner(initial_config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="tg-chat-42",
        chat_type="dm",
        user_id="tg-user-42",
        user_name="Canonical User",
    )
    entry = runner.session_store.get_or_create_session(source)
    session_key = entry.session_key
    session_id = entry.session_id
    db = runner.session_store._db
    assert db is not None

    configured = GatewayConfig.from_dict(
        {
            "sessions_dir": str(hermes_home / "sessions"),
            "canonical_surface_bindings": {
                "ceo": _binding_payload(entry, source),
            },
        }
    )
    runner.config = configured
    runner.session_store.config = configured

    fake_agent = _CachedFakeAgent(session_id, db)
    with runner._agent_cache_lock:
        runner._agent_cache[session_key] = (fake_agent, "seeded", 0, session_id)

    order: list[tuple[str, str]] = []
    original_acquire = runner._turn_leases.acquire

    async def recording_acquire(lease_session_id, *args, **kwargs):
        order.append(("lease", lease_session_id))
        return await original_acquire(lease_session_id, *args, **kwargs)

    monkeypatch.setattr(runner._turn_leases, "acquire", recording_acquire)
    original_load = runner.async_session_store.load_transcript

    async def recording_load(load_session_id):
        order.append(("history", load_session_id))
        return await original_load(load_session_id)

    monkeypatch.setattr(runner.async_session_store, "load_transcript", recording_load)

    pre_ids = {
        row[0]
        for row in db._conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    }
    pre_row = deepcopy(db.get_session(session_id))
    pre_entry_dict = deepcopy(entry.to_dict())
    original_entry = entry

    forbidden_calls: list[str] = []

    def forbid(name):
        def _forbidden(*args, **kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"forbidden path contacted: {name}")

        return _forbidden

    for method_name in (
        "get_or_create_session",
        "_get_or_create_session_impl",
        "reset_session",
        "switch_session",
        "_recover_session_from_db",
    ):
        monkeypatch.setattr(
            runner.session_store,
            method_name,
            forbid(f"SessionStore.{method_name}"),
        )
    monkeypatch.setattr(db, "create_session", forbid("SessionDB.create_session"))

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", forbid("AIAgent"))

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": _API_KEY})
    )
    adapter.gateway_runner = runner
    monkeypatch.setattr(adapter, "send", forbid("APIServerAdapter.send"))
    client = TestClient(TestServer(_app_for(adapter)))
    await client.start_server()
    try:
        response = await client.post(
            _ROUTE,
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "binding": "ceo",
                "event_id": "buzz-event-123",
                "author_id": "buzz-author-7",
                "channel_id": "buzz-channel-9",
                "text": "perform exactly one canonical turn",
            },
        )
        body = await response.json()
    finally:
        await client.close()
        adapter._response_store.close()

    assert response.status == 200
    assert body == {"event_id": "buzz-event-123", "text": "terminal reply"}
    assert order[:2] == [("lease", session_id), ("history", session_id)]
    assert fake_agent.calls == [
        {
            "self": fake_agent,
            "message": "perform exactly one canonical turn",
            "history": [],
            "task_id": session_id,
        }
    ]

    post_ids = {
        row[0]
        for row in db._conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    }
    assert post_ids == pre_ids
    assert runner.session_store._entries[session_key] is original_entry
    assert original_entry.to_dict() == pre_entry_dict
    assert db.get_session(session_id)["id"] == pre_row["id"]
    assert db.get_session(session_id)["session_key"] == pre_row["session_key"]

    messages = db.get_messages_as_conversation(session_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [message["content"] for message in messages] == [
        "perform exactly one canonical turn",
        "terminal reply",
    ]
    assert forbidden_calls == []

    runner.session_store.close_all_db_handles()


def test_authenticated_request_reuses_exact_existing_row_cache_and_lease(
    tmp_path, monkeypatch
):
    asyncio.run(
        _scenario_authenticated_request_reuses_exact_existing_row_cache_and_lease(
            tmp_path, monkeypatch
        )
    )


async def _scenario_caller_target_injection_is_rejected_before_contact(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    runner = _NoContactRunner()
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": _API_KEY})
    )
    adapter.gateway_runner = runner
    client = TestClient(TestServer(_app_for(adapter)))
    await client.start_server()
    try:
        response = await client.post(
            _ROUTE,
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "binding": "ceo",
                "event_id": "buzz-event-injected",
                "author_id": "buzz-author-7",
                "channel_id": "buzz-channel-9",
                "text": "must not run",
                "user_id": "caller-controlled-telegram-target",
            },
        )
        status = response.status
        raw_body = await response.text()
    finally:
        await client.close()
        adapter._response_store.close()

    assert status == 400
    assert json.loads(raw_body) == {
        "error": {
            "code": "canonical_invalid_request",
            "message": "Canonical request rejected.",
        }
    }
    assert runner.contacted == []


def test_caller_target_injection_is_rejected_before_contact(tmp_path, monkeypatch):
    asyncio.run(
        _scenario_caller_target_injection_is_rejected_before_contact(
            tmp_path, monkeypatch
        )
    )


async def _scenario_compression_tip_mismatch_fails_closed_without_constructor_fallback(
    tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    initial_config = GatewayConfig(sessions_dir=hermes_home / "sessions")
    runner = GatewayRunner(initial_config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="tg-chat-stale-tip",
        chat_type="dm",
        user_id="tg-user-stale-tip",
    )
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    assert db is not None

    configured = GatewayConfig.from_dict(
        {
            "sessions_dir": str(hermes_home / "sessions"),
            "canonical_surface_bindings": {
                "ceo": _binding_payload(entry, source),
            },
        }
    )
    runner.config = configured
    runner.session_store.config = configured

    fake_agent = _CachedFakeAgent(entry.session_id, db)
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (
            fake_agent,
            "seeded",
            0,
            entry.session_id,
        )

    monkeypatch.setattr(db, "get_compression_tip", lambda _session_id: "new-tip")
    pre_messages = db.get_messages_as_conversation(entry.session_id)

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": _API_KEY})
    )
    adapter.gateway_runner = runner
    client = TestClient(TestServer(_app_for(adapter)))
    await client.start_server()
    try:
        response = await client.post(
            _ROUTE,
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "binding": "ceo",
                "event_id": "buzz-event-stale-tip",
                "author_id": "buzz-author-7",
                "channel_id": "buzz-channel-9",
                "text": "must not run",
            },
        )
        body = await response.json()
    finally:
        await client.close()
        adapter._response_store.close()

    assert response.status == 409
    assert body == {
        "error": {
            "code": "canonical_binding_stale",
            "message": "Canonical request rejected.",
        }
    }
    assert fake_agent.calls == []
    assert db.get_messages_as_conversation(entry.session_id) == pre_messages
    runner.session_store.close_all_db_handles()


def test_compression_tip_mismatch_fails_closed_without_constructor_fallback(
    tmp_path, monkeypatch
):
    asyncio.run(
        _scenario_compression_tip_mismatch_fails_closed_without_constructor_fallback(
            tmp_path, monkeypatch
        )
    )


async def _scenario_no_constructor_fallback_for_missing_or_stale_existing_state(
    case, expected_status, expected_code, tmp_path, monkeypatch
):
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    initial_config = GatewayConfig(sessions_dir=hermes_home / "sessions")
    runner = GatewayRunner(initial_config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=f"tg-chat-{case}",
        chat_type="dm",
        user_id=f"tg-user-{case}",
    )
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    assert db is not None

    binding_payload = _binding_payload(entry, source)
    if case == "generation_mismatch":
        binding_payload["session_id"] = "different-session-generation"
    configured = GatewayConfig.from_dict(
        {
            "sessions_dir": str(hermes_home / "sessions"),
            "canonical_surface_bindings": {"ceo": binding_payload},
        }
    )
    runner.config = configured
    runner.session_store.config = configured

    fake_agent = _CachedFakeAgent(entry.session_id, db)
    if case != "cache_missing":
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (
                fake_agent,
                "seeded",
                0,
                entry.session_id,
            )

    original_get_session = db.get_session
    if case == "row_missing":
        monkeypatch.setattr(db, "get_session", lambda _session_id: None)
    elif case == "row_ended":
        ended_row = deepcopy(original_get_session(entry.session_id))
        ended_row["ended_at"] = "2026-08-24T00:00:00+00:00"
        ended_row["end_reason"] = "test-ended"
        monkeypatch.setattr(db, "get_session", lambda _session_id: ended_row)
    elif case == "reset_required":
        monkeypatch.setattr(
            runner.session_store,
            "_should_reset",
            lambda _entry, _source: "idle",
        )

    forbidden_calls: list[str] = []

    def forbid(name):
        def _forbidden(*args, **kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"forbidden fallback contacted: {name}")

        return _forbidden

    for method_name in (
        "get_or_create_session",
        "_get_or_create_session_impl",
        "reset_session",
        "switch_session",
        "_recover_session_from_db",
    ):
        monkeypatch.setattr(
            runner.session_store,
            method_name,
            forbid(f"SessionStore.{method_name}"),
        )
    monkeypatch.setattr(db, "create_session", forbid("SessionDB.create_session"))

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", forbid("AIAgent"))
    pre_ids = {
        row[0]
        for row in db._conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    }
    pre_messages = db.get_messages_as_conversation(entry.session_id)

    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": _API_KEY})
    )
    adapter.gateway_runner = runner
    client = TestClient(TestServer(_app_for(adapter)))
    await client.start_server()
    try:
        response = await client.post(
            _ROUTE,
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "binding": "missing" if case == "binding_missing" else "ceo",
                "event_id": f"buzz-event-{case}",
                "author_id": "buzz-author-7",
                "channel_id": "buzz-channel-9",
                "text": "must not run",
            },
        )
        body = await response.json()
    finally:
        await client.close()
        adapter._response_store.close()

    assert response.status == expected_status
    assert body == {
        "error": {
            "code": expected_code,
            "message": "Canonical request rejected.",
        }
    }
    post_ids = {
        row[0]
        for row in db._conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()
    }
    assert post_ids == pre_ids
    assert db.get_messages_as_conversation(entry.session_id) == pre_messages
    assert fake_agent.calls == []
    assert forbidden_calls == []
    runner.session_store.close_all_db_handles()


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_code"),
    [
        ("binding_missing", 404, "canonical_binding_unknown"),
        ("row_missing", 409, "canonical_binding_stale"),
        ("row_ended", 409, "canonical_binding_stale"),
        ("reset_required", 409, "canonical_binding_stale"),
        ("generation_mismatch", 409, "canonical_binding_stale"),
        ("cache_missing", 409, "canonical_agent_missing"),
    ],
)
def test_no_constructor_fallback_for_missing_or_stale_existing_state(
    case, expected_status, expected_code, tmp_path, monkeypatch
):
    asyncio.run(
        _scenario_no_constructor_fallback_for_missing_or_stale_existing_state(
            case,
            expected_status,
            expected_code,
            tmp_path,
            monkeypatch,
        )
    )


async def _scenario_authentication_and_bounds_reject_before_downstream_contact(
    case, expected_status, expected_code, caplog
):
    raw_marker = "/Users/ceo/private/secret.txt?token=sk-live-never-log"
    payload = {
        "binding": "ceo",
        "event_id": "buzz-event-bounds",
        "author_id": "buzz-author-7",
        "channel_id": "buzz-channel-9",
        "text": raw_marker,
    }
    headers = {"Content-Type": "application/json"}
    if case != "no_auth":
        token = raw_marker if case == "bad_auth" else _API_KEY
        headers["Authorization"] = f"Bearer {token}"

    if case == "malformed_json":
        raw = b'{"text":"' + raw_marker.encode()
    elif case == "non_object_json":
        raw = json.dumps([raw_marker]).encode()
    elif case == "oversize_body":
        raw = raw_marker.encode() + b"x" * MAX_REQUEST_BYTES
    elif case == "duplicate_field":
        raw = (
            '{"binding":"ceo","event_id":"first","event_id":"'
            + raw_marker
            + '","author_id":"buzz-author-7","channel_id":"buzz-channel-9",'
            '"text":"must not run"}'
        ).encode()
    elif case == "empty_field":
        payload["text"] = "   "
        payload["event_id"] = raw_marker
        raw = json.dumps(payload).encode()
    elif case == "overbound_field":
        payload["event_id"] = raw_marker + "x" * 257
        raw = json.dumps(payload).encode()
    else:
        raw = json.dumps(payload).encode()

    runner = _NoContactRunner()
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": _API_KEY})
    )
    adapter.gateway_runner = runner
    client = TestClient(TestServer(_app_for(adapter)))
    await client.start_server()
    try:
        response = await client.post(_ROUTE, headers=headers, data=raw)
        response_text = await response.text()
    finally:
        await client.close()
        adapter._response_store.close()

    assert response.status == expected_status
    assert json.loads(response_text)["error"]["code"] == expected_code
    assert runner.contacted == []
    assert raw_marker not in response_text
    assert raw_marker not in caplog.text


@pytest.mark.parametrize(
    ("case", "expected_status", "expected_code"),
    [
        ("no_auth", 401, "gateway_auth_failed"),
        ("bad_auth", 401, "gateway_auth_failed"),
        ("malformed_json", 400, "canonical_invalid_request"),
        ("non_object_json", 400, "canonical_invalid_request"),
        ("oversize_body", 413, "body_too_large"),
        ("duplicate_field", 400, "canonical_invalid_request"),
        ("empty_field", 400, "canonical_invalid_request"),
        ("overbound_field", 400, "canonical_invalid_request"),
    ],
)
def test_authentication_and_bounds_reject_before_downstream_contact(
    case, expected_status, expected_code, caplog
):
    asyncio.run(
        _scenario_authentication_and_bounds_reject_before_downstream_contact(
            case, expected_status, expected_code, caplog
        )
    )
