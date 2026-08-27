"""Request-local reply and terminal grammar contracts for canonical ingress."""

from __future__ import annotations

import asyncio
import threading

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.canonical_surface import CanonicalTurnResult
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


_KEY = "canonical-reply-sink-test-key"
_ROUTE = "/v1/canonical-surface/events"


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


def test_c2_r3_api_reads_back_only_same_request_terminal(tmp_path, monkeypatch):
    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="reply-chat",
            chat_type="dm",
            user_id="reply-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        raw_binding = {
            "session_key": entry.session_key,
            "session_id": entry.session_id,
            "telegram": {
                "chat_id": source.chat_id,
                "chat_type": source.chat_type,
                "user_id": source.user_id,
                "thread_id": None,
            },
            "buzz": {"author_ids": ["author"], "channel_ids": ["channel"]},
        }
        runner.config = GatewayConfig.from_dict(
            {"sessions_dir": str(home / "sessions"), "canonical_surface_bindings": {"bound": raw_binding}}
        )
        runner.session_store.config = runner.config
        calls: list[tuple[str, object]] = []
        gates = {"first": asyncio.Event(), "second": asyncio.Event()}

        async def request_local_turn(binding, event, existing, *, reply_sink):
            calls.append((event.event_id, reply_sink))
            if event.event_id in gates:
                await gates[event.event_id].wait()
            return CanonicalTurnResult(
                binding_name=binding.name,
                terminal_text=f"terminal-for-{event.event_id}",
            )

        monkeypatch.setattr(runner, "run_bound_existing_turn", request_local_turn)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_app(adapter)))
        await client.start_server()
        payload = {
            "binding": "bound",
            "event_id": "single",
            "author_id": "author",
            "channel_id": "channel",
            "text": "a caller cannot choose reply delivery",
        }
        try:
            single = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_KEY}"},
                json=payload,
            )
            assert single.status == 200
            assert await single.json() == {"event_id": "single", "text": "terminal-for-single"}
            assert len(calls) == 1

            first_payload = {**payload, "event_id": "first"}
            second_payload = {**payload, "event_id": "second"}
            first_task = asyncio.create_task(
                client.post(_ROUTE, headers={"Authorization": f"Bearer {_KEY}"}, json=first_payload)
            )
            second_task = asyncio.create_task(
                client.post(_ROUTE, headers={"Authorization": f"Bearer {_KEY}"}, json=second_payload)
            )
            while len(calls) != 3:
                await asyncio.sleep(0)
            gates["second"].set()
            second = await second_task
            gates["first"].set()
            first = await first_task
            assert await second.json() == {"event_id": "second", "text": "terminal-for-second"}
            assert await first.json() == {"event_id": "first", "text": "terminal-for-first"}
            assert calls[1][1] is not calls[2][1]

            injected = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_KEY}"},
                json={**payload, "reply_to": "attacker-selected-target"},
            )
            assert injected.status == 400
            assert len(calls) == 3
        finally:
            await client.close()
            adapter._response_store.close()
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_c2_tool_turn_grammar_refuses_ambiguous_or_malformed_current_turn(tmp_path, monkeypatch):
    """A tool-call carrier is not a completed current-turn terminal."""

    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="grammar-chat",
            chat_type="dm",
            user_id="grammar-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        binding = type(
            "Binding",
            (),
            {
                "name": "bound",
                "session_key": entry.session_key,
                "session_id": entry.session_id,
                "telegram_chat_id": str(source.chat_id),
                "telegram_chat_type": str(source.chat_type),
                "telegram_user_id": str(source.user_id),
                "telegram_thread_id": None,
                "allowed_author_ids": ("author",),
                "allowed_channel_ids": ("channel",),
            },
        )()
        runner.config.canonical_surface_bindings = {"bound": binding}
        runner.session_store.config = runner.config

        class AmbiguousToolAgent:
            session_id = entry.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0

            def run_conversation(self, text, *, conversation_history, task_id):
                self.calls += 1
                self._persist_user_message_idx = len(conversation_history)
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "must-not-publish",
                    "messages": [
                        *conversation_history,
                        {"role": "user", "content": text},
                        {
                            "role": "assistant",
                            "content": "must-not-publish",
                            "tool_calls": [{"id": "call-1", "name": "lookup"}],
                        },
                    ],
                }

        agent = AmbiguousToolAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_app(adapter)))
        await client.start_server()
        try:
            response = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_KEY}"},
                json={
                    "binding": "bound",
                    "event_id": "ambiguous-tool-terminal",
                    "author_id": "author",
                    "channel_id": "channel",
                    "text": "current tool turn",
                },
            )
            assert response.status == 409
            assert await response.json() == {
                "error": {
                    "code": "canonical_turn_refused",
                    "message": "Canonical request rejected.",
                }
            }
            assert agent.calls == 1
        finally:
            await client.close()
            adapter._response_store.close()
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_c2_tool_turn_grammar_valid_controls_publish_exactly_once(tmp_path, monkeypatch):
    """A row-reducing in-place compression must not lose the current turn."""

    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="tool-chat",
            chat_type="dm",
            user_id="tool-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        binding = type(
            "Binding",
            (),
            {
                "name": "bound",
                "session_key": entry.session_key,
                "session_id": entry.session_id,
                "telegram_chat_id": str(source.chat_id),
                "telegram_chat_type": str(source.chat_type),
                "telegram_user_id": str(source.user_id),
                "telegram_thread_id": None,
                "allowed_author_ids": ("author",),
                "allowed_channel_ids": ("channel",),
            },
        )()
        runner.config.canonical_surface_bindings = {"bound": binding}
        runner.session_store.config = runner.config

        class CompressingToolAgent:
            session_id = entry.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0
                self.rows_removed = 0

            def _compress_rows(self, rows):
                self.rows_removed = len(rows) - 1
                reduced = [{"role": "system", "content": "summary"}, rows[-1]]
                self._persist_user_message_idx = len(reduced) - 1
                return reduced

            def run_conversation(self, text, *, conversation_history, task_id):
                self.calls += 1
                uncompressed = [
                    *conversation_history,
                    {"role": "user", "content": "prior question"},
                    {"role": "assistant", "content": "prior answer"},
                    {"role": "user", "content": text},
                ]
                self._persist_user_message_idx = len(uncompressed) - 1
                compressed = self._compress_rows(uncompressed)
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "tool-chain terminal",
                    "messages": [
                        *compressed,
                        {
                            "role": "assistant",
                            "tool_calls": [{"id": "call-1", "name": "lookup"}],
                        },
                        {"role": "tool", "tool_call_id": "call-1", "content": "lookup result"},
                        {"role": "assistant", "content": "tool-chain terminal"},
                    ],
                }

        agent = CompressingToolAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_app(adapter)))
        await client.start_server()
        try:
            response = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_KEY}"},
                json={
                    "binding": "bound",
                    "event_id": "tool-chain",
                    "author_id": "author",
                    "channel_id": "channel",
                    "text": "compress then use a tool",
                },
            )
            assert response.status == 200
            assert await response.json() == {"event_id": "tool-chain", "text": "tool-chain terminal"}
            assert agent.calls == 1
            assert agent.rows_removed == 2
        finally:
            await client.close()
            adapter._response_store.close()
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_c2_b1_canonical_quarantines_all_cached_delivery_callbacks(tmp_path, monkeypatch):
    """Canonical ingress must not invoke a prior route's delivery hooks."""

    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="callback-chat",
            chat_type="dm",
            user_id="callback-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        binding = type(
            "Binding",
            (),
            {
                "name": "bound",
                "session_key": entry.session_key,
                "session_id": entry.session_id,
                "telegram_chat_id": str(source.chat_id),
                "telegram_chat_type": str(source.chat_type),
                "telegram_user_id": str(source.user_id),
                "telegram_thread_id": None,
                "allowed_author_ids": ("author",),
                "allowed_channel_ids": ("channel",),
            },
        )()
        runner.config.canonical_surface_bindings = {"bound": binding}
        runner.session_store.config = runner.config
        wrong_route: list[str] = []
        hook_names = (
            "callback",
            "_on_session_title",
            "_title_failure_callback",
            "stream_delta_callback",
            "tool_progress_callback",
            "tool_start_callback",
            "tool_complete_callback",
            "interim_assistant_callback",
            "status_callback",
            "notice_callback",
            "clarify_callback",
            "background_review_callback",
            "event_callback",
            "reaction_callback",
            "step_callback",
        )

        class CachedCallbackAgent:
            session_id = entry.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0
                self.worker: threading.Thread | None = None
                for name in hook_names:
                    setattr(self, name, lambda *args, _name=name, **kwargs: wrong_route.append(_name))

            def run_conversation(self, text, *, conversation_history, task_id):
                self.calls += 1
                for name in hook_names:
                    callback = getattr(self, name)
                    if callback is not None:
                        callback("current turn")
                delayed_title = self._on_session_title
                self.worker = threading.Thread(
                    target=lambda: delayed_title is not None and delayed_title("late", "canonical"),
                )
                self.worker.start()
                self._persist_user_message_idx = len(conversation_history)
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "quarantined terminal",
                    "messages": [
                        *conversation_history,
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "quarantined terminal"},
                    ],
                }

        agent = CachedCallbackAgent()
        original_callbacks = {name: getattr(agent, name) for name in hook_names}
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_app(adapter)))
        await client.start_server()
        try:
            response = await client.post(
                _ROUTE,
                headers={"Authorization": f"Bearer {_KEY}"},
                json={
                    "binding": "bound",
                    "event_id": "callback-quarantine",
                    "author_id": "author",
                    "channel_id": "channel",
                    "text": "do not contact the previous route",
                },
            )
            assert response.status == 200
            assert await response.json() == {
                "event_id": "callback-quarantine",
                "text": "quarantined terminal",
            }
            assert agent.worker is not None
            agent.worker.join(timeout=1)
            assert not agent.worker.is_alive()
            assert wrong_route == []
            assert agent.calls == 1
            assert {name: getattr(agent, name) for name in hook_names} == original_callbacks
            for name, original in original_callbacks.items():
                assert getattr(agent, name) is original
        finally:
            await client.close()
            adapter._response_store.close()
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_c2_b1_rotation_refusal_and_internal_exception_are_safe(tmp_path, monkeypatch):
    """Refusal is pre-mutation and admitted failures have one public-safe shape."""

    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="safe-chat",
            chat_type="dm",
            user_id="safe-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        binding = type(
            "Binding",
            (),
            {
                "name": "bound",
                "session_key": entry.session_key,
                "session_id": entry.session_id,
                "telegram_chat_id": str(source.chat_id),
                "telegram_chat_type": str(source.chat_type),
                "telegram_user_id": str(source.user_id),
                "telegram_thread_id": None,
                "allowed_author_ids": ("author",),
                "allowed_channel_ids": ("channel",),
            },
        )()
        runner.config.canonical_surface_bindings = {"bound": binding}
        runner.session_store.config = runner.config
        original_load = runner.async_session_store.load_transcript
        loads = 0
        callbacks: list[str] = []

        class ControlledAgent:
            session_id = entry.session_id

            def __init__(self, *, in_place: bool, fail: bool) -> None:
                self.compression_in_place = in_place
                self.fail = fail
                self.calls = 0
                self._on_session_title = lambda *args: callbacks.append("title")

            def run_conversation(self, text, *, conversation_history, task_id):
                self.calls += 1
                if self.fail:
                    raise RuntimeError("provider=/private/path request-body=do-not-leak")
                self._persist_user_message_idx = len(conversation_history)
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "safe terminal",
                    "messages": [
                        *conversation_history,
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "safe terminal"},
                    ],
                }

        unsafe = ControlledAgent(in_place=False, fail=False)
        unsafe_title = unsafe._on_session_title
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (unsafe, "exact", 0, entry.session_id)

        async def forbidden_load(session_id):
            nonlocal loads
            loads += 1
            raise AssertionError("rotation refusal must precede transcript loading")

        monkeypatch.setattr(runner.async_session_store, "load_transcript", forbidden_load)
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))
        adapter.gateway_runner = runner
        client = TestClient(TestServer(_app(adapter)))
        await client.start_server()
        payload = {
            "binding": "bound",
            "event_id": "safe-boundary",
            "author_id": "author",
            "channel_id": "channel",
            "text": "one safe request",
        }
        try:
            refused = await client.post(_ROUTE, headers={"Authorization": f"Bearer {_KEY}"}, json=payload)
            assert refused.status == 409
            assert loads == 0
            assert unsafe.calls == 0
            assert unsafe._on_session_title is unsafe_title

            monkeypatch.setattr(runner.async_session_store, "load_transcript", original_load)
            failing = ControlledAgent(in_place=True, fail=True)
            failing_title = failing._on_session_title
            with runner._agent_cache_lock:
                runner._agent_cache[entry.session_key] = (failing, "exact", 0, entry.session_id)
            response = await client.post(_ROUTE, headers={"Authorization": f"Bearer {_KEY}"}, json=payload)
            body = await response.text()
            assert response.status == 500
            assert response.headers["Content-Type"].startswith("application/json")
            assert body == '{"error": {"code": "canonical_internal_error", "message": "Canonical request failed."}}'
            assert "private" not in body
            assert "request-body" not in body
            assert failing.calls == 1
            assert failing._on_session_title is failing_title
            assert callbacks == []
        finally:
            await client.close()
            adapter._response_store.close()
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())
