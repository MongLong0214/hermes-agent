"""Strict request-local canonical reply-sink behavioral contracts."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import fields, FrozenInstanceError
import json
from types import SimpleNamespace
import threading
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.canonical_surface import CanonicalTurnResult, request_local_reply_sink
from gateway.config import PlatformConfig
from gateway.platforms import base as platform_base
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner


_API_KEY = "canonical-test-key"
_CANONICAL_ROUTE = "/v1/canonical-surface/events"


def _api_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post(_CANONICAL_ROUTE, adapter._handle_canonical_surface_event)
    return app


class _APIResultRunner:
    def __init__(self) -> None:
        binding = SimpleNamespace(name="ceo")
        self.config = SimpleNamespace(canonical_surface_bindings={"ceo": binding})
        self.session_store = object()
        self.sinks: list[Any] = []

    async def run_bound_existing_turn(
        self, _binding, _event, _entry, *, reply_sink=None
    ) -> CanonicalTurnResult:
        self.sinks.append(reply_sink)
        return CanonicalTurnResult(
            binding_name="ceo",
            terminal_text="same request terminal",
        )


async def _scenario_c2_r3_api_reads_back_only_same_request_terminal(monkeypatch):
    from gateway.canonical_surface import ExistingCanonicalBindingResolver

    monkeypatch.setattr(
        ExistingCanonicalBindingResolver,
        "resolve",
        lambda _self, _binding, _event: SimpleNamespace(),
    )
    runner = _APIResultRunner()
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _API_KEY}))
    adapter.gateway_runner = runner
    client = TestClient(TestServer(_api_app(adapter)))
    await client.start_server()
    try:
        response = await client.post(
            _CANONICAL_ROUTE,
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "binding": "ceo",
                "event_id": "api-event-one",
                "author_id": "buzz-author",
                "channel_id": "buzz-channel",
                "text": "return to this request",
            },
        )
        status = response.status
        raw_body = await response.text()
    finally:
        await client.close()
        adapter._response_store.close()

    assert status == 200
    assert json.loads(raw_body) == {
        "event_id": "api-event-one",
        "text": "same request terminal",
    }
    assert len(runner.sinks) == 1
    assert runner.sinks[0] is not None

    result = CanonicalTurnResult("ceo", "same request terminal")
    assert {field.name for field in fields(result)} == {"binding_name", "terminal_text"}
    with pytest.raises(FrozenInstanceError):
        result.terminal_text = "mutated"  # type: ignore[misc]


def test_c2_r3_api_reads_back_only_same_request_terminal(monkeypatch):
    asyncio.run(_scenario_c2_r3_api_reads_back_only_same_request_terminal(monkeypatch))


class _TurnLeases:
    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.released: list[object] = []

    async def acquire(self, session_id: str, **_kwargs: Any) -> object:
        self.acquired.append(session_id)
        return object()

    def release(self, token: object) -> None:
        self.released.append(token)


class _TranscriptStore:
    def __init__(self, history: list[dict[str, Any]]) -> None:
        self.history = history
        self.load_calls = 0
        self._store = self

    async def load_transcript(self, _session_id: str) -> list[dict[str, Any]]:
        self.load_calls += 1
        return deepcopy(self.history)


class _ResultAgent:
    def __init__(self, session_id: str, new_messages: list[dict[str, Any]], final: str):
        self.session_id = session_id
        self._new_messages = new_messages
        self._final = final

    def run_conversation(
        self,
        message: str,
        *,
        conversation_history: list[dict[str, Any]],
        task_id: str,
    ) -> dict[str, Any]:
        self._persist_user_message_idx = len(conversation_history)
        return {
            "completed": True,
            "failed": False,
            "interrupted": False,
            "partial": False,
            "final_response": self._final,
            "messages": [
                *deepcopy(conversation_history),
                {"role": "user", "content": message},
                *deepcopy(self._new_messages),
            ],
            "session_id": task_id,
        }


def _trusted_sink():
    async def _discard(_result: CanonicalTurnResult) -> None:
        return None

    return request_local_reply_sink(_discard)


def _runner_with_agent(agent: Any, history: list[dict[str, Any]] | None = None):
    runner = object.__new__(GatewayRunner)
    runner._agent_cache_lock = threading.Lock()
    runner._agent_cache = {
        "binding-key": (agent, "seeded", 0, agent.session_id),
    }
    leases = _TurnLeases()
    store = _TranscriptStore(history or [])
    runner._turn_leases = leases
    runner.session_store = store
    runner._async_session_store = store
    runner._begin_session_run_generation = lambda session_key: 1
    runner._init_cached_agent_for_turn = lambda agent, interrupt_depth: None
    return runner, leases


class _ForbiddenCacheGate:
    def __init__(self) -> None:
        self.touched = 0

    def __enter__(self):
        self.touched += 1
        return self

    def __exit__(self, *_args: Any) -> None:
        return None


def test_c2_r11_missing_sink_refuses_before_cache_lease_or_turn():
    runner = object.__new__(GatewayRunner)
    cache_gate = _ForbiddenCacheGate()
    runner._agent_cache_lock = cache_gate
    runner._agent_cache = {}
    leases = _TurnLeases()
    runner._turn_leases = leases
    binding = SimpleNamespace(
        name="ceo", session_key="binding-key", session_id="existing-session"
    )
    entry = SimpleNamespace(
        session_key="binding-key", session_id="existing-session"
    )
    event = SimpleNamespace(text="must not run")

    with pytest.raises(ValueError, match="^canonical_reply_sink_missing$"):
        asyncio.run(runner.run_bound_existing_turn(binding, event, entry))

    assert cache_gate.touched == 0
    assert leases.acquired == []


@pytest.mark.parametrize(
    ("new_messages", "final_response"),
    [
        (
            [
                {"role": "assistant", "content": "first candidate"},
                {"role": "assistant", "content": "second candidate"},
            ],
            "second candidate",
        ),
        ([{"role": "assistant", "content": "structured terminal"}], "other text"),
    ],
    ids=["ambiguous", "final-response-mismatch"],
)
def test_c2_r7_terminal_ambiguity_and_mismatch_fail_closed(
    new_messages, final_response
):
    session_id = "existing-session"
    runner, leases = _runner_with_agent(
        _ResultAgent(session_id, new_messages, final_response),
        history=[
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ],
    )
    binding = SimpleNamespace(
        name="ceo", session_key="binding-key", session_id=session_id
    )
    entry = SimpleNamespace(session_key="binding-key", session_id=session_id)
    event = SimpleNamespace(text="current request")

    with pytest.raises(ValueError, match="^canonical_turn_refused$"):
        asyncio.run(
            runner.run_bound_existing_turn(
                binding,
                event,
                entry,
                reply_sink=_trusted_sink(),
            )
        )

    assert leases.acquired == [session_id]
    assert len(leases.released) == 1


class _RouteAdapter:
    def __init__(self, expected_calls: int = 1) -> None:
        self.sent: list[dict[str, Any]] = []
        self._expected_calls = expected_calls
        self._all_entered = asyncio.Event()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        if len(self.sent) >= self._expected_calls:
            self._all_entered.set()
        await self._all_entered.wait()
        return SimpleNamespace(success=True)


def _event_sink_or_inert(adapter, event):
    factory = getattr(platform_base, "request_local_reply_sink_for_event", None)
    if factory is not None:
        return factory(adapter, event)

    async def _inert(_result: CanonicalTurnResult) -> None:
        return None

    return request_local_reply_sink(_inert)


def _telegram_event(route: str):
    return SimpleNamespace(
        source=SimpleNamespace(
            platform="telegram",
            chat_id=f"chat-{route}",
            chat_type="dm",
            thread_id=f"thread-{route}",
        ),
        message_id=f"message-{route}",
        reply_to_message_id=None,
        raw_message={},
    )


async def _scenario_c2_r1_r5_telegram_origin_is_request_local_under_concurrency():
    adapter = _RouteAdapter(expected_calls=2)
    sink_a = _event_sink_or_inert(adapter, _telegram_event("A"))
    sink_b = _event_sink_or_inert(adapter, _telegram_event("B"))

    await asyncio.gather(
        sink_a.publish(CanonicalTurnResult("ceo", "terminal A")),
        sink_b.publish(CanonicalTurnResult("ceo", "terminal B")),
    )

    assert sorted(adapter.sent, key=lambda call: call["chat_id"]) == [
        {
            "chat_id": "chat-A",
            "content": "terminal A",
            "reply_to": "message-A",
            "metadata": {
                "direct_messages_topic_id": "thread-A",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "message-A",
                "thread_id": "thread-A",
            },
        },
        {
            "chat_id": "chat-B",
            "content": "terminal B",
            "reply_to": "message-B",
            "metadata": {
                "direct_messages_topic_id": "thread-B",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "message-B",
                "thread_id": "thread-B",
            },
        },
    ]


def test_c2_r1_r5_telegram_origin_is_request_local_under_concurrency():
    asyncio.run(_scenario_c2_r1_r5_telegram_origin_is_request_local_under_concurrency())


def test_c2_r2_buzz_origin_keeps_channel_thread_and_event_anchor():
    async def _scenario():
        adapter = _RouteAdapter()
        event = SimpleNamespace(
            source=SimpleNamespace(
                platform="buzz",
                chat_id="buzz-channel-A",
                chat_type="channel",
                thread_id="buzz-thread-A",
            ),
            message_id="buzz-event-A",
            reply_to_message_id=None,
            raw_message={},
        )
        sink = _event_sink_or_inert(adapter, event)

        await sink.publish(CanonicalTurnResult("ceo", "buzz terminal A"))

        assert adapter.sent == [
            {
                "chat_id": "buzz-channel-A",
                "content": "buzz terminal A",
                "reply_to": "buzz-event-A",
                "metadata": {"thread_id": "buzz-thread-A"},
            }
        ]

    asyncio.run(_scenario())


def test_c2_r10_publisher_exception_is_typed_and_never_retried():
    class _ExplodingAdapter:
        def __init__(self) -> None:
            self.calls = 0

        async def send(self, **_kwargs):
            self.calls += 1
            raise RuntimeError("private transport detail")

    async def _scenario():
        adapter = _ExplodingAdapter()
        sink = _event_sink_or_inert(adapter, _telegram_event("failure"))

        with pytest.raises(ValueError, match="^canonical_reply_publish_failed$"):
            await sink.publish(CanonicalTurnResult("ceo", "must not retry"))

        assert adapter.calls == 1

    asyncio.run(_scenario())


def test_c2_r6_same_request_sink_is_one_shot_under_concurrency():
    async def _scenario():
        adapter = _RouteAdapter()
        sink = _event_sink_or_inert(adapter, _telegram_event("one-shot"))
        result = CanonicalTurnResult("ceo", "single terminal")

        outcomes = await asyncio.gather(
            sink.publish(result),
            sink.publish(result),
            return_exceptions=True,
        )

        assert len(adapter.sent) == 1
        assert sum(outcome is None for outcome in outcomes) == 1
        errors = [outcome for outcome in outcomes if isinstance(outcome, ValueError)]
        assert len(errors) == 1
        assert str(errors[0]) == "canonical_reply_already_published"

    asyncio.run(_scenario())


def test_c2_r4_route_and_target_payload_injection_stops_before_actor():
    class _ForbiddenIngressRunner:
        def __init__(self) -> None:
            self.touched = 0

        @property
        def config(self):
            self.touched += 1
            raise AssertionError("actor authority must not be consulted")

    async def _scenario():
        runner = _ForbiddenIngressRunner()
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _API_KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_api_app(adapter)))
        await client.start_server()
        base_payload = {
            "binding": "ceo",
            "event_id": "injection-event",
            "author_id": "buzz-author",
            "channel_id": "buzz-channel",
            "text": "must be rejected",
        }
        forbidden = (
            "routing_key",
            "session_id",
            "chat_id",
            "thread_id",
            "reply_to",
            "target",
            "target_id",
            "destination",
            "destination_id",
        )
        try:
            for field in forbidden:
                response = await client.post(
                    _CANONICAL_ROUTE,
                    headers={"Authorization": f"Bearer {_API_KEY}"},
                    json={**base_payload, field: "attacker-route"},
                )
                assert response.status == 400
                assert (await response.json())["error"]["code"] == "canonical_invalid_request"
            duplicate_channel = (
                b'{"binding":"ceo","event_id":"injection-event",'
                b'"author_id":"buzz-author","channel_id":"buzz-channel",'
                b'"channel_id":"attacker-route","text":"must be rejected"}'
            )
            response = await client.post(
                _CANONICAL_ROUTE,
                headers={
                    "Authorization": f"Bearer {_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=duplicate_channel,
            )
            assert response.status == 400
            assert (await response.json())["error"]["code"] == "canonical_invalid_request"
        finally:
            await client.close()
            adapter._response_store.close()

        assert runner.touched == 0

    asyncio.run(_scenario())


class _ScriptedResultAgent:
    def __init__(self, session_id: str, payload: dict[str, Any]) -> None:
        self.session_id = session_id
        self._payload = payload

    def run_conversation(
        self,
        message: str,
        *,
        conversation_history: list[dict[str, Any]],
        task_id: str,
    ) -> dict[str, Any]:
        payload = deepcopy(self._payload)
        self._persist_user_message_idx = len(conversation_history)
        payload.setdefault("completed", True)
        payload.setdefault("session_id", task_id)
        payload.setdefault(
            "messages",
            [
                *deepcopy(conversation_history),
                {"role": "user", "content": message},
                *deepcopy(payload.pop("new_messages", [])),
            ],
        )
        return payload


@pytest.mark.parametrize(
    "payload",
    [
        {"final_response": "", "new_messages": []},
        {
            "final_response": "   ",
            "new_messages": [{"role": "assistant", "content": "   "}],
        },
        {
            "failed": True,
            "final_response": "failed terminal",
            "new_messages": [{"role": "assistant", "content": "failed terminal"}],
        },
        {
            "partial": True,
            "final_response": "partial terminal",
            "new_messages": [{"role": "assistant", "content": "partial terminal"}],
        },
        {
            "cancelled": True,
            "final_response": "cancelled terminal",
            "new_messages": [{"role": "assistant", "content": "cancelled terminal"}],
        },
    ],
    ids=["missing", "blank", "failed", "partial", "cancelled"],
)
def test_c2_r8_no_valid_terminal_refuses_without_publishing(payload):
    session_id = "existing-session"
    runner, leases = _runner_with_agent(_ScriptedResultAgent(session_id, payload))
    binding = SimpleNamespace(
        name="ceo", session_key="binding-key", session_id=session_id
    )
    entry = SimpleNamespace(session_key="binding-key", session_id=session_id)
    event = SimpleNamespace(text="current request")
    published: list[CanonicalTurnResult] = []

    async def _capture(result: CanonicalTurnResult) -> None:
        published.append(result)

    with pytest.raises(ValueError, match="^canonical_turn_refused$"):
        asyncio.run(
            runner.run_bound_existing_turn(
                binding,
                event,
                entry,
                reply_sink=request_local_reply_sink(_capture),
            )
        )

    assert published == []
    assert leases.acquired == [session_id]
    assert len(leases.released) == 1


@pytest.mark.parametrize(
    "payload",
    [
        {
            "already_sent": True,
            "final_response": "terminal",
            "new_messages": [{"role": "assistant", "content": "terminal"}],
        },
        {
            "progress": True,
            "final_response": "terminal",
            "new_messages": [{"role": "assistant", "content": "terminal"}],
        },
        {
            "preview": True,
            "final_response": "terminal",
            "new_messages": [{"role": "assistant", "content": "terminal"}],
        },
        {
            "stream": True,
            "final_response": "terminal",
            "new_messages": [{"role": "assistant", "content": "terminal"}],
        },
        {
            "final_response": "terminal",
            "new_messages": [
                {"role": "assistant", "content": "progress update"},
                {"role": "assistant", "content": "terminal"},
            ],
        },
    ],
    ids=["already-sent", "progress", "preview", "stream", "interim-assistant"],
)
def test_c2_r9_progress_preview_stream_and_already_sent_never_publish(payload):
    session_id = "existing-session"
    runner, leases = _runner_with_agent(_ScriptedResultAgent(session_id, payload))
    binding = SimpleNamespace(
        name="ceo", session_key="binding-key", session_id=session_id
    )
    entry = SimpleNamespace(session_key="binding-key", session_id=session_id)
    event = SimpleNamespace(text="current request")

    with pytest.raises(ValueError, match="^canonical_turn_refused$"):
        asyncio.run(
            runner.run_bound_existing_turn(
                binding,
                event,
                entry,
                reply_sink=_trusted_sink(),
            )
        )

    assert leases.acquired == [session_id]
    assert len(leases.released) == 1


async def _exercise_endpoint_terminal_tail(
    monkeypatch,
    *,
    new_messages: list[dict[str, Any]],
    final_response: str,
    event_id: str,
):
    from gateway import canonical_surface

    session_id = "existing-session"
    agent = _ResultAgent(session_id, new_messages, final_response)
    cached_publications: list[str] = []
    stale_clarify = lambda *_args, **_kwargs: cached_publications.append("clarify")
    stale_title = lambda *_args, **_kwargs: cached_publications.append("title")
    agent.clarify_callback = stale_clarify
    agent._on_session_title = stale_title

    runner, leases = _runner_with_agent(
        agent,
        history=[
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ],
    )
    binding = SimpleNamespace(
        name="ceo", session_key="binding-key", session_id=session_id
    )
    runner.config = SimpleNamespace(canonical_surface_bindings={"ceo": binding})
    monkeypatch.setattr(
        canonical_surface.ExistingCanonicalBindingResolver,
        "resolve",
        lambda _self, _binding, _event: SimpleNamespace(
            session_key="binding-key",
            session_id=session_id,
        ),
    )

    constructor_calls: list[str] = []

    def _forbidden_builder(*_args, **_kwargs):
        constructor_calls.append("build-agent")
        raise AssertionError("canonical tail refusal must not construct a fallback agent")

    runner._build_agent = _forbidden_builder
    session_side_effects: list[str] = []

    def _forbidden_session_side_effect(name: str):
        def _forbidden(*_args, **_kwargs):
            session_side_effects.append(name)
            raise AssertionError(f"canonical tail path attempted session side effect: {name}")

        return _forbidden

    for method_name in (
        "get_or_create_session",
        "reset_session",
        "switch_session",
        "_recover_session_from_db",
    ):
        setattr(
            runner.session_store,
            method_name,
            _forbidden_session_side_effect(method_name),
        )

    originating_publications: list[CanonicalTurnResult] = []
    real_sink_factory = canonical_surface.request_local_reply_sink

    def _recording_sink_factory(publisher):
        async def _recording_publish(result):
            originating_publications.append(result)
            await publisher(result)

        return real_sink_factory(_recording_publish)

    monkeypatch.setattr(
        canonical_surface,
        "request_local_reply_sink",
        _recording_sink_factory,
    )

    fallback_publications: list[dict[str, Any]] = []

    async def _record_fallback_publication(*args, **kwargs):
        fallback_publications.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(success=True)

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _API_KEY}))
    adapter.gateway_runner = runner
    adapter.send = _record_fallback_publication
    client = TestClient(TestServer(_api_app(adapter)))
    await client.start_server()
    try:
        response = await client.post(
            _CANONICAL_ROUTE,
            headers={"Authorization": f"Bearer {_API_KEY}"},
            json={
                "binding": "ceo",
                "event_id": event_id,
                "author_id": "buzz-author",
                "channel_id": "buzz-channel",
                "text": "exercise terminal tail",
            },
        )
        status = response.status
        payload = await response.json()
    finally:
        await client.close()
        adapter._response_store.close()

    return {
        "status": status,
        "payload": payload,
        "originating_publications": originating_publications,
        "cached_publications": cached_publications,
        "fallback_publications": fallback_publications,
        "constructor_calls": constructor_calls,
        "session_side_effects": session_side_effects,
        "session_id": agent.session_id,
        "callbacks_restored": (
            agent.clarify_callback is stale_clarify
            and agent._on_session_title is stale_title
        ),
        "leases": leases,
    }


@pytest.mark.parametrize(
    "trailing_role",
    [
        pytest.param("progress", id="progress"),
        pytest.param("error", id="error"),
        pytest.param("fallback", id="fallback"),
        pytest.param("system", id="system"),
        pytest.param("unknown", id="unknown"),
        pytest.param("future_delivery_notice_v9", id="unknown-scalar-spelling"),
    ],
)
def test_c2_trailing_nonsemantic_role_refuses_at_endpoint_without_publication(
    trailing_role, monkeypatch
):
    outcome = asyncio.run(
        _exercise_endpoint_terminal_tail(
            monkeypatch,
            new_messages=[
                {"role": "assistant", "content": "candidate terminal"},
                {"role": trailing_role, "content": "unexpected trailing route"},
            ],
            final_response="candidate terminal",
            event_id=f"trailing-{trailing_role}",
        )
    )

    assert (
        outcome["status"],
        outcome["payload"],
        len(outcome["originating_publications"]),
    ) == (
        409,
        {
            "error": {
                "code": "canonical_turn_refused",
                "message": "Canonical request rejected.",
            }
        },
        0,
    )
    assert outcome["cached_publications"] == []
    assert outcome["fallback_publications"] == []
    assert outcome["constructor_calls"] == []
    assert outcome["session_side_effects"] == []
    assert outcome["session_id"] == "existing-session"
    assert outcome["callbacks_restored"] is True
    assert outcome["leases"].acquired == ["existing-session"]
    assert len(outcome["leases"].released) == 1


def test_c2_tool_call_before_final_physical_assistant_publishes_exactly_once(monkeypatch):
    outcome = asyncio.run(
        _exercise_endpoint_terminal_tail(
            monkeypatch,
            new_messages=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-status",
                            "type": "function",
                            "function": {"name": "status", "arguments": "{}"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-status",
                    "name": "status",
                    "content": "ready",
                },
                {"role": "assistant", "content": "tool-backed terminal"},
            ],
            final_response="tool-backed terminal",
            event_id="tool-sequence-control",
        )
    )

    assert outcome["status"] == 200
    assert outcome["payload"] == {
        "event_id": "tool-sequence-control",
        "text": "tool-backed terminal",
    }
    assert outcome["originating_publications"] == [
        CanonicalTurnResult("ceo", "tool-backed terminal")
    ]
    assert outcome["cached_publications"] == []
    assert outcome["fallback_publications"] == []
    assert outcome["constructor_calls"] == []
    assert outcome["session_side_effects"] == []
    assert outcome["session_id"] == "existing-session"
    assert outcome["callbacks_restored"] is True
    assert outcome["leases"].acquired == ["existing-session"]
    assert len(outcome["leases"].released) == 1


def test_c2_r12_existing_only_order_releases_before_same_request_publish(monkeypatch):
    from gateway import canonical_surface

    order: list[str] = []
    session_id = "existing-session"

    class _OrderLock:
        def __enter__(self):
            order.append("cache")
            return self

        def __exit__(self, *_args):
            return None

    class _OrderedLeases:
        async def acquire(self, _session_id: str, **_kwargs):
            order.append("lease")
            return object()

        def release(self, _token):
            order.append("release")

    class _OrderedAgent(_ResultAgent):
        def run_conversation(self, *args, **kwargs):
            order.append("turn")
            return super().run_conversation(*args, **kwargs)

    async def _scenario():
        agent = _OrderedAgent(
            session_id,
            [{"role": "assistant", "content": "ordered terminal"}],
            "ordered terminal",
        )
        runner, _leases = _runner_with_agent(agent)
        runner._agent_cache_lock = _OrderLock()
        runner._turn_leases = _OrderedLeases()
        binding = SimpleNamespace(
            name="ceo", session_key="binding-key", session_id=session_id
        )
        runner.config = SimpleNamespace(canonical_surface_bindings={"ceo": binding})
        runner._build_agent = lambda *_args, **_kwargs: pytest.fail(
            "new agent construction is forbidden"
        )

        monkeypatch.setattr(
            canonical_surface.ExistingCanonicalBindingResolver,
            "resolve",
            lambda _self, _binding, _event: (
                order.append("resolve")
                or SimpleNamespace(
                    session_key="binding-key",
                    session_id=session_id,
                )
            ),
        )
        real_sink_factory = canonical_surface.request_local_reply_sink

        def _recording_sink_factory(publisher):
            async def _recording_publish(result):
                order.append("publish")
                await publisher(result)

            return real_sink_factory(_recording_publish)

        monkeypatch.setattr(
            canonical_surface,
            "request_local_reply_sink",
            _recording_sink_factory,
        )

        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _API_KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_api_app(adapter)))
        await client.start_server()
        try:
            response = await client.post(
                _CANONICAL_ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={
                    "binding": "ceo",
                    "event_id": "ordered-event",
                    "author_id": "buzz-author",
                    "channel_id": "buzz-channel",
                    "text": "ordered request",
                },
            )
            assert response.status == 200
            assert await response.json() == {
                "event_id": "ordered-event",
                "text": "ordered terminal",
            }
        finally:
            await client.close()
            adapter._response_store.close()

    asyncio.run(_scenario())
    assert order == ["resolve", "cache", "lease", "turn", "release", "publish"]


class _StaleDeliveryCallbackAgent(_ResultAgent):
    def __init__(self, session_id: str, exit_kind: str) -> None:
        super().__init__(
            session_id,
            [{"role": "assistant", "content": "canonical terminal"}],
            "canonical terminal",
        )
        self.exit_kind = exit_kind
        self.wrong_route_calls: list[str] = []
        self.persistent_authority = object()
        self._session_db = object()
        self._session_db_created = True
        self.platform = "telegram"
        self.model = "gateway-model"
        self.provider = "gateway-provider"

        def _stale_delivery(label: str) -> str:
            self.wrong_route_calls.append(label)
            return "wrong-route response"

        self.clarify_callback = lambda *_args, **_kwargs: _stale_delivery("clarify")
        self.future_delivery_callback = lambda *_args, **_kwargs: _stale_delivery(
            "future"
        )
        self._on_session_title = lambda title, source: _stale_delivery(
            f"title:{source}:{title}"
        )

    def run_conversation(self, message, *args, **kwargs):
        from agent.turn_context import _maybe_title_session_at_turn_start

        _maybe_title_session_at_turn_start(
            self,
            [{"role": "user", "content": message}],
        )
        if self.clarify_callback is not None:
            self.clarify_callback("wrong-route question", ["yes"])
        if self.future_delivery_callback is not None:
            self.future_delivery_callback("wrong-route future delivery")
        if self.exit_kind == "agent-exception":
            raise RuntimeError("agent turn exploded")
        result = super().run_conversation(message, *args, **kwargs)
        self._persist_user_message_idx = len(kwargs["conversation_history"])
        if self.exit_kind == "selector-refusal":
            result["final_response"] = "mismatched terminal"
        return result


def _run_stale_callback_turn(agent: _StaleDeliveryCallbackAgent, monkeypatch):
    def _complete_title_from_turn(
        _session_db,
        _session_id,
        user_message,
        *,
        title_callback=None,
        **_kwargs,
    ):
        if title_callback is not None:
            title_callback(f"B-derived title: {user_message}", "llm")

    monkeypatch.setattr(
        "agent.title_generator.maybe_auto_title",
        _complete_title_from_turn,
    )
    runner, leases = _runner_with_agent(agent)
    binding = SimpleNamespace(
        name="ceo", session_key="binding-key", session_id=agent.session_id
    )
    entry = SimpleNamespace(session_key="binding-key", session_id=agent.session_id)
    event = SimpleNamespace(text="canonical request")
    return (
        lambda: asyncio.run(
            runner.run_bound_existing_turn(
                binding,
                event,
                entry,
                reply_sink=_trusted_sink(),
            )
        ),
        leases,
    )


def test_c2_b1_canonical_quarantines_all_cached_delivery_callbacks(monkeypatch):
    agent = _StaleDeliveryCallbackAgent("existing-session", "success")
    stale_clarify = agent.clarify_callback
    stale_future = agent.future_delivery_callback
    stale_title = agent._on_session_title
    authority = agent.persistent_authority
    run_turn, leases = _run_stale_callback_turn(agent, monkeypatch)

    result = run_turn()

    assert result == CanonicalTurnResult("ceo", "canonical terminal")
    assert agent.wrong_route_calls == []
    assert agent.clarify_callback is stale_clarify
    assert agent.future_delivery_callback is stale_future
    assert agent._on_session_title is stale_title
    assert agent.persistent_authority is authority
    assert leases.acquired == [agent.session_id]
    assert len(leases.released) == 1


@pytest.mark.parametrize(
    ("exit_kind", "expected_error", "expected_message"),
    [
        ("selector-refusal", ValueError, "^canonical_turn_refused$"),
        ("agent-exception", RuntimeError, "^agent turn exploded$"),
    ],
)
def test_c2_b1_canonical_restores_cached_callbacks_on_exceptional_exit(
    exit_kind, expected_error, expected_message, monkeypatch
):
    agent = _StaleDeliveryCallbackAgent("existing-session", exit_kind)
    stale_clarify = agent.clarify_callback
    stale_future = agent.future_delivery_callback
    stale_title = agent._on_session_title
    authority = agent.persistent_authority
    run_turn, leases = _run_stale_callback_turn(agent, monkeypatch)

    with pytest.raises(expected_error, match=expected_message):
        run_turn()

    assert agent.wrong_route_calls == []
    assert agent.clarify_callback is stale_clarify
    assert agent.future_delivery_callback is stale_future
    assert agent._on_session_title is stale_title
    assert agent.persistent_authority is authority
    assert leases.acquired == [agent.session_id]
    assert len(leases.released) == 1


class _LegacyRotationCapableAgent(_ResultAgent):
    def __init__(self, session_id: str) -> None:
        super().__init__(
            session_id,
            [{"role": "assistant", "content": "must not publish"}],
            "must not publish",
        )
        self.compression_in_place = False
        self.run_calls = 0
        self.created_sessions: list[str] = []
        self.callback_assignments: list[tuple[str, Any]] = []
        self.clarify_callback = lambda *_args, **_kwargs: None
        self._on_session_title = lambda *_args, **_kwargs: None
        self._observe_callback_assignments = True

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            getattr(self, "_observe_callback_assignments", False)
            and name in {"clarify_callback", "_on_session_title"}
        ):
            self.callback_assignments.append((name, value))
        object.__setattr__(self, name, value)

    def run_conversation(self, *args, **kwargs):
        self.run_calls += 1
        child_session_id = "legacy-compression-child"
        self.created_sessions.append(child_session_id)
        self.session_id = child_session_id
        result = super().run_conversation(*args, **kwargs)
        result["session_id"] = child_session_id
        return result


def test_c2_b2_legacy_rotation_capability_refuses_before_turn_mutation(monkeypatch):
    from gateway import canonical_surface

    async def _scenario():
        session_id = "existing-session"
        agent = _LegacyRotationCapableAgent(session_id)
        stale_clarify = agent.clarify_callback
        stale_title = agent._on_session_title
        runner, leases = _runner_with_agent(agent, _long_history())
        binding = SimpleNamespace(
            name="ceo", session_key="binding-key", session_id=session_id
        )
        runner.config = SimpleNamespace(canonical_surface_bindings={"ceo": binding})
        monkeypatch.setattr(
            canonical_surface.ExistingCanonicalBindingResolver,
            "resolve",
            lambda _self, _binding, _event: SimpleNamespace(
                session_key="binding-key",
                session_id=session_id,
            ),
        )

        built_agents: list[object] = []

        def _forbidden_builder(*_args, **_kwargs):
            built_agents.append(object())
            raise AssertionError("canonical rotation refusal must not build an agent")

        runner._build_agent = _forbidden_builder
        published: list[CanonicalTurnResult] = []
        real_sink_factory = canonical_surface.request_local_reply_sink

        def _recording_sink_factory(publisher):
            async def _recording_publish(result):
                published.append(result)
                await publisher(result)

            return real_sink_factory(_recording_publish)

        monkeypatch.setattr(
            canonical_surface,
            "request_local_reply_sink",
            _recording_sink_factory,
        )

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": _API_KEY})
        )
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_api_app(adapter)))
        await client.start_server()
        try:
            response = await client.post(
                _CANONICAL_ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={
                    "binding": "ceo",
                    "event_id": "legacy-rotation-event",
                    "author_id": "buzz-author",
                    "channel_id": "buzz-channel",
                    "text": "must refuse before rotation",
                },
            )
            status = response.status
            payload = await response.json()
        finally:
            await client.close()
            adapter._response_store.close()

        assert status == 409
        assert payload == {
            "error": {
                "code": "canonical_turn_refused",
                "message": "Canonical request rejected.",
            }
        }
        assert runner.async_session_store.load_calls == 0
        assert agent.run_calls == 0
        assert agent.created_sessions == []
        assert built_agents == []
        assert published == []
        assert agent.session_id == session_id
        assert agent.clarify_callback is stale_clarify
        assert agent._on_session_title is stale_title
        assert agent.callback_assignments == []
        assert leases.acquired == [session_id]
        assert len(leases.released) == 1

    asyncio.run(_scenario())


class _UnexpectedCanonicalExceptionAgent(_ResultAgent):
    def __init__(self, session_id: str, raw_marker: str) -> None:
        super().__init__(
            session_id,
            [{"role": "assistant", "content": "must not publish"}],
            "must not publish",
        )
        self.compression_in_place = True
        self.raw_marker = raw_marker
        self.run_calls = 0

    def run_conversation(self, *_args, **_kwargs):
        self.run_calls += 1
        raise RuntimeError(self.raw_marker)


def test_c2_b3_public_http_boundary_sanitizes_unexpected_exception(
    monkeypatch, caplog
):
    from gateway import canonical_surface

    async def _scenario():
        raw_marker = (
            "/Users/private/canonical.db|request-body-secret|runtime-provider-detail"
        )
        session_id = "existing-session"
        agent = _UnexpectedCanonicalExceptionAgent(session_id, raw_marker)
        runner, leases = _runner_with_agent(agent)
        binding = SimpleNamespace(
            name="ceo", session_key="binding-key", session_id=session_id
        )
        runner.config = SimpleNamespace(canonical_surface_bindings={"ceo": binding})
        monkeypatch.setattr(
            canonical_surface.ExistingCanonicalBindingResolver,
            "resolve",
            lambda _self, _binding, _event: SimpleNamespace(
                session_key="binding-key",
                session_id=session_id,
            ),
        )

        built_agents: list[object] = []

        def _forbidden_builder(*_args, **_kwargs):
            built_agents.append(object())
            raise AssertionError("unexpected exception must not trigger fallback")

        runner._build_agent = _forbidden_builder
        published: list[CanonicalTurnResult] = []
        real_sink_factory = canonical_surface.request_local_reply_sink

        def _recording_sink_factory(publisher):
            async def _recording_publish(result):
                published.append(result)
                await publisher(result)

            return real_sink_factory(_recording_publish)

        monkeypatch.setattr(
            canonical_surface,
            "request_local_reply_sink",
            _recording_sink_factory,
        )

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": _API_KEY})
        )
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_api_app(adapter)))
        await client.start_server()
        try:
            response = await client.post(
                _CANONICAL_ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={
                    "binding": "ceo",
                    "event_id": "unexpected-exception-event",
                    "author_id": "buzz-author",
                    "channel_id": "buzz-channel",
                    "text": "request-body-secret",
                },
            )
            status = response.status
            content_type = response.content_type
            raw_body = await response.text()
        finally:
            await client.close()
            adapter._response_store.close()

        assert status == 500
        assert content_type == "application/json"
        assert json.loads(raw_body) == {
            "error": {
                "code": "canonical_internal_error",
                "message": "Canonical request failed.",
            }
        }
        assert raw_marker not in raw_body
        assert "request-body-secret" not in raw_body
        assert raw_marker not in caplog.text
        assert published == []
        assert agent.run_calls == 1
        assert built_agents == []
        assert leases.acquired == [session_id]
        assert len(leases.released) == 1

    asyncio.run(_scenario())


class _RowReducingCompressionAgent:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.run_calls = 0

    def run_conversation(
        self,
        message: str,
        *,
        conversation_history: list[dict[str, Any]],
        task_id: str,
    ) -> dict[str, Any]:
        assert len(conversation_history) >= 12
        self.run_calls += 1
        rebuilt = [
            {"role": "assistant", "content": "compressed prior-turn summary"},
            {
                "role": "user",
                "content": f"API-only enriched: {message}",
            },
            {"role": "assistant", "content": "compressed canonical terminal"},
        ]
        self._persist_user_message_idx = 1
        return {
            "completed": True,
            "failed": False,
            "interrupted": False,
            "partial": False,
            "final_response": "compressed canonical terminal",
            "messages": rebuilt,
            "session_id": task_id,
        }


def _long_history() -> list[dict[str, Any]]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"old row {index}",
        }
        for index in range(12)
    ]


def test_c2_b2_compression_reanchored_boundary_publishes_one_terminal(monkeypatch):
    from gateway import canonical_surface

    async def _scenario():
        session_id = "existing-session"
        agent = _RowReducingCompressionAgent(session_id)
        runner, leases = _runner_with_agent(agent, _long_history())
        binding = SimpleNamespace(
            name="ceo", session_key="binding-key", session_id=session_id
        )
        runner.config = SimpleNamespace(canonical_surface_bindings={"ceo": binding})
        monkeypatch.setattr(
            canonical_surface.ExistingCanonicalBindingResolver,
            "resolve",
            lambda _self, _binding, _event: SimpleNamespace(
                session_key="binding-key",
                session_id=session_id,
            ),
        )

        published: list[CanonicalTurnResult] = []
        real_sink_factory = canonical_surface.request_local_reply_sink

        def _recording_sink_factory(publisher):
            async def _recording_publish(result):
                published.append(result)
                await publisher(result)

            return real_sink_factory(_recording_publish)

        monkeypatch.setattr(
            canonical_surface,
            "request_local_reply_sink",
            _recording_sink_factory,
        )

        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"key": _API_KEY})
        )
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_api_app(adapter)))
        await client.start_server()
        try:
            response = await client.post(
                _CANONICAL_ROUTE,
                headers={"Authorization": f"Bearer {_API_KEY}"},
                json={
                    "binding": "ceo",
                    "event_id": "compression-event",
                    "author_id": "buzz-author",
                    "channel_id": "buzz-channel",
                    "text": "canonical request",
                },
            )
            status = response.status
            payload = await response.json()
        finally:
            await client.close()
            adapter._response_store.close()

        assert status == 200
        assert payload == {
            "event_id": "compression-event",
            "text": "compressed canonical terminal",
        }
        assert published == [
            CanonicalTurnResult("ceo", "compressed canonical terminal")
        ]
        assert agent.run_calls == 1
        assert leases.acquired == [session_id]
        assert len(leases.released) == 1

    asyncio.run(_scenario())


class _CorruptCompressionAnchorAgent(_ResultAgent):
    def __init__(self, session_id: str, anchor_kind: str) -> None:
        super().__init__(
            session_id,
            [{"role": "assistant", "content": "canonical terminal"}],
            "canonical terminal",
        )
        self.anchor_kind = anchor_kind

    def run_conversation(self, *args, **kwargs):
        result = super().run_conversation(*args, **kwargs)
        anchors: dict[str, Any] = {
            "none": None,
            "bool": True,
            "string": "2",
            "negative": -1,
            "out-of-range": len(result["messages"]),
            "non-user": 1,
        }
        if self.anchor_kind == "missing":
            del self._persist_user_message_idx
        else:
            self._persist_user_message_idx = anchors[self.anchor_kind]
        return result


@pytest.mark.parametrize(
    "anchor_kind",
    [
        "missing",
        "none",
        "bool",
        "string",
        "negative",
        "out-of-range",
        "non-user",
    ],
)
def test_c2_b2_corrupt_or_missing_compression_anchor_fails_closed(anchor_kind):
    session_id = "existing-session"
    agent = _CorruptCompressionAnchorAgent(session_id, anchor_kind)
    runner, leases = _runner_with_agent(
        agent,
        history=[
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
        ],
    )
    binding = SimpleNamespace(
        name="ceo", session_key="binding-key", session_id=session_id
    )
    entry = SimpleNamespace(session_key="binding-key", session_id=session_id)

    with pytest.raises(ValueError, match="^canonical_turn_refused$"):
        asyncio.run(
            runner.run_bound_existing_turn(
                binding,
                SimpleNamespace(text="canonical request"),
                entry,
                reply_sink=_trusted_sink(),
            )
        )

    assert leases.acquired == [session_id]
    assert len(leases.released) == 1
