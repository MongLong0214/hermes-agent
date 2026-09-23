"""Focused contracts for canonical request-local replies and existing actors."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from gateway.canonical_surface import (
    CanonicalTurnResult,
    request_local_reply_sink,
    require_request_local_reply_sink,
)
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def test_request_local_reply_sink_is_opaque_single_use_and_normalizes_failure():
    async def exercise() -> None:
        published: list[CanonicalTurnResult] = []

        async def collect(result: CanonicalTurnResult) -> None:
            published.append(result)

        sink = request_local_reply_sink(collect)
        assert require_request_local_reply_sink(sink) is sink
        result = CanonicalTurnResult("bound", "terminal")
        await sink.publish(result)
        assert published == [result]
        with pytest.raises(ValueError, match="^canonical_reply_already_published$"):
            await sink.publish(result)

        class Impostor:
            async def publish(self, result):
                pass

        with pytest.raises(ValueError, match="^canonical_reply_sink_missing$"):
            require_request_local_reply_sink(Impostor())

        async def fail(_result: CanonicalTurnResult) -> None:
            raise RuntimeError("do not expose publisher details")

        with pytest.raises(ValueError, match="^canonical_reply_publish_failed$"):
            await request_local_reply_sink(fail).publish(result)

    asyncio.run(exercise())


def test_existing_cached_actor_turn_returns_only_current_terminal(tmp_path, monkeypatch):
    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="canonical-chat",
            chat_type="dm",
            user_id="canonical-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        binding = SimpleNamespace(
            name="bound",
            session_key=entry.session_key,
            session_id=entry.session_id,
            telegram_chat_id=source.chat_id,
            telegram_chat_type=source.chat_type,
            telegram_user_id=source.user_id,
            telegram_thread_id=source.thread_id,
        )
        event = SimpleNamespace(text="canonical turn")
        prior_route_calls: list[str] = []

        class ExistingAgent:
            session_id = entry.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0
                self.callback = lambda *_args, **_kwargs: prior_route_calls.append("callback")

            def run_conversation(self, text, *, conversation_history, task_id):
                self.calls += 1
                self.callback("must stay request-local") if self.callback else None
                self._persist_user_message_idx = len(conversation_history)
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "canonical terminal",
                    "messages": [
                        *conversation_history,
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "canonical terminal"},
                    ],
                }

        agent = ExistingAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
        published: list[CanonicalTurnResult] = []

        async def publish(result: CanonicalTurnResult) -> None:
            published.append(result)

        sink = request_local_reply_sink(publish)
        try:
            with pytest.raises(ValueError, match="^canonical_reply_sink_missing$"):
                await runner.run_bound_existing_turn(binding, event, entry)
            assert agent.calls == 0

            result = await runner.run_bound_existing_turn(
                binding, event, entry, reply_sink=sink
            )
            assert result == CanonicalTurnResult("bound", "canonical terminal")
            assert published == []
            await sink.publish(result)
            assert published == [result]
            assert agent.calls == 1
            assert prior_route_calls == []
            assert agent.callback is not None

            with runner._agent_cache_lock:
                runner._agent_cache.pop(entry.session_key)
            with pytest.raises(ValueError, match="^canonical_agent_missing$"):
                await runner.run_bound_existing_turn(binding, event, entry, reply_sink=sink)
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_existing_cached_actor_accepts_current_same_key_rotation_but_rejects_previous_entry(tmp_path, monkeypatch):
    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        key = "agent:main:telegram:dm:canonical-chat"
        origin = SimpleNamespace(
            chat_id="canonical-chat",
            chat_type="dm",
            user_id="canonical-user",
            thread_id=None,
        )
        previous = SimpleNamespace(session_key=key, session_id="previous-session", origin=origin)
        current = SimpleNamespace(session_key=key, session_id="current-session", origin=origin)
        binding = SimpleNamespace(
            name="bound",
            session_key=key,
            session_id=previous.session_id,
            telegram_chat_id=origin.chat_id,
            telegram_chat_type=origin.chat_type,
            telegram_user_id=origin.user_id,
            telegram_thread_id=origin.thread_id,
        )
        event = SimpleNamespace(text="rotated canonical turn")
        existing_lookups: list[str] = []

        def lookup_existing(session_key: str):
            existing_lookups.append(session_key)
            return current

        monkeypatch.setattr(runner.session_store, "lookup_by_session_key_existing", lookup_existing)

        class CurrentAgent:
            session_id = current.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0

            def run_conversation(self, text, *, conversation_history, task_id):
                self.calls += 1
                self._persist_user_message_idx = len(conversation_history)
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "rotated terminal",
                    "messages": [
                        *conversation_history,
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "rotated terminal"},
                    ],
                }

        agent = CurrentAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[key] = (agent, "exact", 0, current.session_id)
        try:
            result = await runner.run_bound_existing_turn(
                binding,
                event,
                current,
                reply_sink=request_local_reply_sink(lambda _result: asyncio.sleep(0)),
            )
            assert result == CanonicalTurnResult("bound", "rotated terminal")
            assert agent.calls == 1

            with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                await runner.run_bound_existing_turn(
                    binding,
                    event,
                    previous,
                    reply_sink=request_local_reply_sink(lambda _result: asyncio.sleep(0)),
                )
            assert agent.calls == 1
            assert existing_lookups == [key, key, key, key]
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_post_lease_head_replacement_refuses_before_transcript_or_actor(tmp_path, monkeypatch):
    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="canonical-chat",
            chat_type="dm",
            user_id="canonical-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        replacement = SimpleNamespace(
            session_key=entry.session_key,
            session_id=entry.session_id,
            origin=entry.origin,
        )
        binding = SimpleNamespace(
            name="bound",
            session_key=entry.session_key,
            session_id=entry.session_id,
            telegram_chat_id=source.chat_id,
            telegram_chat_type=source.chat_type,
            telegram_user_id=source.user_id,
            telegram_thread_id=source.thread_id,
        )
        current_head: list[object] = [entry]
        existing_lookups: list[str] = []

        def lookup_existing(session_key: str):
            existing_lookups.append(session_key)
            return current_head[0]

        def unexpected(*_args, **_kwargs):
            raise AssertionError("canonical turn must not load, recover, or create a session")

        monkeypatch.setattr(runner.session_store, "lookup_by_session_key_existing", lookup_existing)
        monkeypatch.setattr(runner.session_store, "lookup_by_session_key", unexpected)
        monkeypatch.setattr(runner.session_store, "get_or_create_session", unexpected)
        monkeypatch.setattr(runner.session_store, "_route_recover", unexpected, raising=False)
        real_leases = runner._turn_leases

        class LeaseThatRotatesHead:
            async def acquire(self, *args, **kwargs):
                current_head[0] = replacement
                return await real_leases.acquire(*args, **kwargs)

            def release(self, lease):
                real_leases.release(lease)

        runner._turn_leases = LeaseThatRotatesHead()

        class ExplodingAgent:
            session_id = entry.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0

            def run_conversation(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("canonical actor must not run after a stale post-lease head")

        agent = ExplodingAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
        try:
            with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                await runner.run_bound_existing_turn(
                    binding,
                    SimpleNamespace(text="must not run"),
                    entry,
                    reply_sink=request_local_reply_sink(lambda _result: asyncio.sleep(0)),
                )
            assert current_head[0] is replacement
            assert existing_lookups == [entry.session_key]
            assert agent.calls == 0
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


def test_transcript_load_head_replacement_refuses_before_actor(tmp_path, monkeypatch):
    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="canonical-chat",
            chat_type="dm",
            user_id="canonical-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        replacement = SimpleNamespace(
            session_key=entry.session_key,
            session_id=entry.session_id,
            origin=entry.origin,
        )
        binding = SimpleNamespace(
            name="bound",
            session_key=entry.session_key,
            session_id=entry.session_id,
            telegram_chat_id=source.chat_id,
            telegram_chat_type=source.chat_type,
            telegram_user_id=source.user_id,
            telegram_thread_id=source.thread_id,
        )
        current_head: list[object] = [entry]
        existing_lookups: list[str] = []

        def lookup_existing(session_key: str):
            existing_lookups.append(session_key)
            return current_head[0]

        def unexpected(*_args, **_kwargs):
            raise AssertionError("canonical turn must not load, recover, or create a session")

        monkeypatch.setattr(runner.session_store, "lookup_by_session_key_existing", lookup_existing)
        monkeypatch.setattr(runner.session_store, "lookup_by_session_key", unexpected)
        monkeypatch.setattr(runner.session_store, "get_or_create_session", unexpected)
        monkeypatch.setattr(runner.session_store, "_route_recover", unexpected, raising=False)

        class TranscriptThatRotatesHead:
            _store = runner.session_store

            async def load_transcript(self, session_id: str):
                assert session_id == entry.session_id
                current_head[0] = replacement
                return []

        monkeypatch.setattr(runner, "_async_session_store", TranscriptThatRotatesHead())

        class ExplodingAgent:
            session_id = entry.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0

            def run_conversation(self, *_args, **_kwargs):
                self.calls += 1
                raise AssertionError("canonical actor must not run after a transcript-time head change")

        agent = ExplodingAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
        try:
            with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                await runner.run_bound_existing_turn(
                    binding,
                    SimpleNamespace(text="must not run"),
                    entry,
                    reply_sink=request_local_reply_sink(lambda _result: asyncio.sleep(0)),
                )
            assert current_head[0] is replacement
            assert existing_lookups == [entry.session_key, entry.session_key]
            assert agent.calls == 0
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


@pytest.mark.parametrize("swap_current_head", [False, True])
def test_actor_completion_requires_unchanged_existing_current_head(
    tmp_path, monkeypatch, swap_current_head
):
    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="canonical-chat",
            chat_type="dm",
            user_id="canonical-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        replacement = SimpleNamespace(
            session_key=entry.session_key,
            session_id="replacement-session",
            origin=entry.origin,
        )
        binding = SimpleNamespace(
            name="bound",
            session_key=entry.session_key,
            session_id=entry.session_id,
            telegram_chat_id=source.chat_id,
            telegram_chat_type=source.chat_type,
            telegram_user_id=source.user_id,
            telegram_thread_id=source.thread_id,
        )
        current_head: list[object] = [entry]

        def lookup_existing(session_key: str):
            assert session_key == entry.session_key
            return current_head[0]

        monkeypatch.setattr(
            runner.session_store, "lookup_by_session_key_existing", lookup_existing
        )

        class SwappingAgent:
            session_id = entry.session_id
            compression_in_place = True

            def __init__(self) -> None:
                self.calls = 0

            def run_conversation(self, text, *, conversation_history, task_id):
                self.calls += 1
                self._persist_user_message_idx = len(conversation_history)
                if swap_current_head:
                    current_head[0] = replacement
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "canonical terminal",
                    "messages": [
                        *conversation_history,
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": "canonical terminal"},
                    ],
                }

        agent = SwappingAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
        published: list[CanonicalTurnResult] = []

        async def publish(result: CanonicalTurnResult) -> None:
            published.append(result)

        try:
            if swap_current_head:
                with pytest.raises(ValueError, match="^canonical_binding_stale$"):
                    await runner.run_bound_existing_turn(
                        binding,
                        SimpleNamespace(text="canonical turn"),
                        entry,
                        reply_sink=request_local_reply_sink(publish),
                    )
                assert current_head[0] is replacement
            else:
                result = await runner.run_bound_existing_turn(
                    binding,
                    SimpleNamespace(text="canonical turn"),
                    entry,
                    reply_sink=request_local_reply_sink(publish),
                )
                assert result == CanonicalTurnResult("bound", "canonical terminal")
                assert current_head[0] is entry
            assert agent.calls == 1
            assert published == []
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())


@pytest.mark.parametrize("terminal_flag", [None, "failed", "partial", "interrupted"])
def test_selector_refuses_each_explicit_unsuccessful_terminal_flag(terminal_flag):
    result = {
        "completed": True,
        "session_id": "canonical-session",
        "final_response": "canonical terminal",
        "messages": [
            {"role": "user", "content": "canonical turn"},
            {"role": "assistant", "content": "canonical terminal"},
        ],
    }
    if terminal_flag is not None:
        result[terminal_flag] = True

    if terminal_flag is None:
        assert GatewayRunner._select_canonical_turn_result(
            result,
            binding_name="bound",
            history_boundary=0,
            expected_session_id="canonical-session",
        ) == CanonicalTurnResult("bound", "canonical terminal")
    else:
        with pytest.raises(ValueError, match="^canonical_turn_refused$"):
            GatewayRunner._select_canonical_turn_result(
                result,
                binding_name="bound",
                history_boundary=0,
                expected_session_id="canonical-session",
            )


@pytest.mark.parametrize("outcome", ["malformed", "raises"])
def test_callback_quarantine_restores_every_request_callback_on_failure(tmp_path, outcome):
    async def exercise() -> None:
        home = tmp_path / "home"
        home.mkdir()
        runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="canonical-chat",
            chat_type="dm",
            user_id="canonical-user",
        )
        entry = runner.session_store.get_or_create_session(source)
        binding = SimpleNamespace(
            name="bound",
            session_key=entry.session_key,
            session_id=entry.session_id,
            telegram_chat_id=source.chat_id,
            telegram_chat_type=source.chat_type,
            telegram_user_id=source.user_id,
            telegram_thread_id=source.thread_id,
        )
        callback_names = (
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
        outward = {name: [] for name in callback_names}
        delivered = []

        class FailingAgent:
            compression_in_place = True

            def __init__(self) -> None:
                self.session_id = entry.session_id
                for name in callback_names:
                    setattr(self, name, lambda *args, _name=name: outward[_name].append(args))

            def run_conversation(self, text, *, conversation_history, task_id):
                for name in callback_names:
                    callback = getattr(self, name)
                    if callback is not None:
                        callback("during-run")
                if outcome == "raises":
                    raise RuntimeError("actor failure")
                self._persist_user_message_idx = len(conversation_history)
                return {
                    "completed": True,
                    "session_id": task_id,
                    "final_response": "",
                    "messages": [*conversation_history, {"role": "user", "content": text}],
                }

        agent = FailingAgent()
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)

        async def publish(result):
            delivered.append(result)

        expected_error = RuntimeError if outcome == "raises" else ValueError
        try:
            with pytest.raises(expected_error):
                await runner.run_bound_existing_turn(
                    binding,
                    SimpleNamespace(text="failed canonical turn"),
                    entry,
                    reply_sink=request_local_reply_sink(publish),
                )
            assert outward == {name: [] for name in callback_names}
            assert delivered == []
            for name in callback_names:
                getattr(agent, name)("after-run")
            assert outward == {name: [("after-run",)] for name in callback_names}
        finally:
            runner.session_store.close_all_db_handles()

    asyncio.run(exercise())
