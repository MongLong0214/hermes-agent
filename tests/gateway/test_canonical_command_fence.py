"""A /new command cannot dismantle a canonical worker's existing route."""

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.base import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class PausedAgent:
    def __init__(self, session_id, entered, release):
        self.session_id = session_id
        self.compression_in_place = True
        self.entered = entered
        self.release = release

    def run_conversation(self, text, *, conversation_history, task_id):
        self.entered.set()
        assert self.release.wait(10)
        self._persist_user_message_idx = len(conversation_history)
        return dict(completed=True, failed=False, interrupted=False, partial=False,
                    session_id=task_id, final_response="done",
                    messages=[*conversation_history, {"role": "user", "content": text},
                              {"role": "assistant", "content": "done"}])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    runner.config.canonical_surface_bindings = {
        "bound": SimpleNamespace(name="bound", session_key=entry.session_key, session_id=entry.session_id,
                                 telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
                                 telegram_thread_id=None, allowed_author_ids=("author",),
                                 allowed_channel_ids=("channel",))
    }
    entered, release = threading.Event(), threading.Event()
    agent = PausedAgent(entry.session_id, entered, release)
    runner._agent_cache[entry.session_key] = (agent, "exact", 0, entry.session_id)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter.gateway_runner = runner

    async def send():
        async def read():
            return json.dumps(dict(binding="bound", event_id="event", author_id="author",
                                   channel_id="channel", text="hello")).encode()
        response = await adapter._handle_canonical_surface_event(
            SimpleNamespace(headers={"Authorization": "Bearer test-key"}, read=read))
        return response.status, json.loads(response.text)

    yield SimpleNamespace(runner=runner, source=source, entry=entry, agent=agent,
                          entered=entered, release=release, send=send)
    release.set()
    adapter._response_store.close()
    runner.session_store.close_all_db_handles()


@pytest.mark.parametrize("caller", ["busy", "direct", "confirmed"])
def test_new_keeps_paused_canonical_worker_and_state_intact_then_succeeds(setup, caller):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        key = entry.session_key
        event = MessageEvent(text="/new", source=source, message_id="new-event")
        state = runner._session_state(key)
        db = runner.session_store._db
        before_ids = {r[0] for r in db._conn.execute("SELECT id FROM sessions")}
        cached = runner._agent_cache[key]
        first = asyncio.create_task(setup.send())
        try:
            assert await asyncio.to_thread(setup.entered.wait, 10)
            assert runner.session_store.canonical_entry_reserved(key)
            state.persistent.pending_command_text = "keep queue"
            generation = state.persistent.run_generation

            async def invoke():
                if caller == "busy":
                    return await runner._busy_new_command(event, key, source)
                if caller == "confirmed":
                    async def confirmed_callback():
                        return await runner._handle_reset_command(event)
                    return await confirmed_callback()
                return await runner._handle_reset_command(event)

            refused = await invoke()
            assert "busy" in str(refused).lower() or "running" in str(refused).lower()
            assert runner.session_store._entries[key] is entry
            assert runner._agent_cache[key] is cached
            assert state.persistent.run_generation == generation
            assert state.persistent.pending_command_text == "keep queue"
            assert runner._pending_messages[key] == "keep queue"
            assert {r[0] for r in db._conn.execute("SELECT id FROM sessions")} == before_ids
        finally:
            setup.release.set()
        assert (await asyncio.wait_for(first, 10))[0] == 200
        assert not runner.session_store.canonical_entry_reserved(key)
        result = await invoke()
        assert "Session is busy — wait for the current response before `/new`." not in str(result)
        assert runner.session_store._entries[key].session_id != entry.session_id
    asyncio.run(exercise())


def test_branch_refuses_paused_canonical_worker_before_creating_child_then_succeeds(setup):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        key = entry.session_key
        db = runner.session_store._db
        db.append_message(entry.session_id, role="user", content="parent history")
        event = MessageEvent(text="/branch alternative", source=source, message_id="branch-event")
        state = runner._session_state(key)
        state.persistent.pending_command_text = "keep queue"
        before_ids = {r[0] for r in db._conn.execute("SELECT id FROM sessions")}
        cached = runner._agent_cache[key]
        first = asyncio.create_task(setup.send())
        try:
            assert await asyncio.to_thread(setup.entered.wait, 10)
            assert runner.session_store.canonical_entry_reserved(key)
            generation = state.persistent.run_generation
            refused = await asyncio.wait_for(runner._handle_branch_command(event), 10)
            assert "busy" in str(refused).lower() or "running" in str(refused).lower()
            assert {r[0] for r in db._conn.execute("SELECT id FROM sessions")} == before_ids
            assert runner.session_store._entries[key] is entry
            assert runner._agent_cache[key] is cached
            assert state.persistent.run_generation == generation
            assert state.persistent.pending_command_text == "keep queue"
            assert runner._pending_messages[key] == "keep queue"
        finally:
            setup.release.set()
        assert (await asyncio.wait_for(first, 10))[0] == 200
        assert not runner.session_store.canonical_entry_reserved(key)
        result = await asyncio.wait_for(runner._handle_branch_command(event), 10)
        assert "busy" not in str(result).lower()
        child = runner.session_store._entries[key]
        assert child.session_id != entry.session_id
        assert db.get_session(child.session_id)["parent_session_id"] == entry.session_id
        assert child.session_id in {r[0] for r in db._conn.execute("SELECT id FROM sessions")}
    asyncio.run(exercise())


@pytest.mark.parametrize("command", ["/resume", "/sessions"])
def test_resume_refuses_paused_canonical_worker_before_releasing_state_then_succeeds(setup, command):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        key = entry.session_key
        db = runner.session_store._db
        target_id = "prior_session_for_resume"
        db.create_session(target_id, "telegram", session_key=key, user_id="user", chat_id="chat")
        db.set_session_title(target_id, "Prior work")
        event = MessageEvent(text=f"{command} {target_id}", source=source, message_id="resume-event")
        state = runner._session_state(key)
        state.persistent.pending_command_text = "keep queue"
        cached = runner._agent_cache[key]
        first = asyncio.create_task(setup.send())

        async def invoke():
            if command == "/sessions":
                return await runner._handle_sessions_command(event)
            return await runner._handle_resume_command(event)

        try:
            assert await asyncio.to_thread(setup.entered.wait, 10)
            assert runner.session_store.canonical_entry_reserved(key)
            # The canonical ingress reserves the route; seed its legacy running
            # slot to prove /resume cannot clear it before a rejected switch.
            runner._running_agents[key] = setup.agent
            running = runner._running_agents[key]
            generation = state.persistent.run_generation
            rows = list(db._conn.execute("SELECT * FROM sessions ORDER BY id"))
            refused = await asyncio.wait_for(invoke(), 10)
            assert "busy" in str(refused).lower() or "running" in str(refused).lower()
            assert runner._running_agents[key] is running
            assert runner.session_store.canonical_entry_reserved(key)
            assert runner.session_store._entries[key] is entry
            assert runner._agent_cache[key] is cached
            assert state.persistent.run_generation == generation
            assert state.persistent.pending_command_text == "keep queue"
            assert runner._pending_messages[key] == "keep queue"
            assert list(db._conn.execute("SELECT * FROM sessions ORDER BY id")) == rows
        finally:
            setup.release.set()
        assert (await asyncio.wait_for(first, 10))[0] == 200
        assert not runner.session_store.canonical_entry_reserved(key)
        result = await asyncio.wait_for(invoke(), 10)
        assert "busy" not in str(result).lower()
        assert runner.session_store._entries[key].session_id == target_id

    asyncio.run(exercise())


@pytest.mark.parametrize("command", ["/resume", "/sessions"])
def test_cancelled_resume_holds_claim_through_switch_and_route_cleanup(setup, monkeypatch, command):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        key = entry.session_key
        db = runner.session_store._db
        target_id = "prior_session_for_cancelled_resume"
        db.create_session(target_id, "telegram", session_key=key, user_id="user", chat_id="chat")
        db.set_session_title(target_id, "Prior work")
        event = MessageEvent(text=f"{command} {target_id}", source=source, message_id="cancel-resume")
        state = runner._session_state(key)
        state.persistent.pending_command_text = "old queue"
        cached = runner._agent_cache[key]
        generation = state.persistent.run_generation
        before_rows = list(db._conn.execute("SELECT * FROM sessions ORDER BY id"))
        entered, release = threading.Event(), threading.Event()
        cleanup_entered, finish_cleanup = asyncio.Event(), asyncio.Event()
        original_switch = runner.session_store.switch_session
        original_get_title = runner._session_db.get_session_title

        def paused_switch(*args, **kwargs):
            entered.set()
            assert release.wait(10)
            return original_switch(*args, **kwargs)

        async def paused_title(*args, **kwargs):
            cleanup_entered.set()
            await finish_cleanup.wait()
            return await original_get_title(*args, **kwargs)

        monkeypatch.setattr(runner.session_store, "switch_session", paused_switch)
        monkeypatch.setattr(runner._session_db, "get_session_title", paused_title)

        async def invoke():
            if command == "/sessions":
                return await runner._handle_sessions_command(event)
            return await runner._handle_resume_command(event)

        task = asyncio.create_task(invoke())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runner.session_store._command_claims.get(key)
            with pytest.raises(ValueError, match="canonical_binding_stale"):
                runner.session_store.reserve_canonical_entry(entry)
            assert runner.session_store._entries[key] is entry
            assert runner._agent_cache[key] is cached
            assert state.persistent.run_generation == generation
            assert list(db._conn.execute("SELECT * FROM sessions ORDER BY id")) == before_rows
            release.set()
            await asyncio.wait_for(cleanup_entered.wait(), 10)
            # The route has switched, but reply metadata/cleanup is still in flight.
            assert runner.session_store._command_claims.get(key)
            with pytest.raises(ValueError, match="canonical_binding_stale"):
                runner.session_store.reserve_canonical_entry(runner.session_store._entries[key])
        finally:
            release.set()
            finish_cleanup.set()

        for _ in range(200):
            if not runner.session_store._command_claims.get(key):
                break
            await asyncio.sleep(0.01)
        assert not runner.session_store._command_claims.get(key)
        switched = runner.session_store._entries[key]
        assert switched.session_id == target_id
        assert key not in runner._agent_cache
        assert db.get_session(entry.session_id)["end_reason"] == "session_switch"
        reservation = runner.session_store.reserve_canonical_entry(switched)
        runner.session_store.release_canonical_entry(switched, reservation)

    asyncio.run(exercise())


def test_claimed_switch_requires_exact_current_route_and_token(setup):
    store, entry = setup.runner.session_store, setup.entry
    token = store.claim_session_command(entry, entry.session_id)
    try:
        with pytest.raises(ValueError, match="canonical_turn_busy"):
            store.switch_session(entry.session_key, "other", command_claim=object())
        with pytest.raises(ValueError, match="canonical_turn_busy"):
            store.switch_session(entry.session_key, "other")
        assert store._entries[entry.session_key] is entry
        assert entry.session_id == setup.agent.session_id
        entry.session_id = "stale"
        with pytest.raises(ValueError, match="canonical_turn_busy"):
            store.switch_session(entry.session_key, "other", command_claim=token)
    finally:
        store.release_session_command(entry, token)


def test_compress_refuses_paused_canonical_turn_without_writes_but_preview_works(setup, monkeypatch):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        db = runner.session_store._db
        key = entry.session_key
        for role, content in [("user", "first"), ("assistant", "reply"),
                              ("user", "second"), ("assistant", "reply")]:
            db.append_message(entry.session_id, role=role, content=content)
        event = MessageEvent(text="/compress", source=source, message_id="compress-event")
        preview = MessageEvent(text="/compress --preview", source=source, message_id="preview-event")
        state = runner._session_state(key)
        state.persistent.pending_command_text = "keep queue"
        cached = runner._agent_cache[key]
        first = asyncio.create_task(setup.send())
        try:
            assert await asyncio.to_thread(setup.entered.wait, 10)
            before = list(db._conn.execute("SELECT * FROM sessions ORDER BY id"))
            messages = list(db._conn.execute("SELECT * FROM messages ORDER BY id"))
            generation = state.persistent.run_generation
            result = await asyncio.wait_for(runner._handle_compress_command(event), 10)
            assert "busy" in result.lower()
            report = await asyncio.wait_for(runner._handle_compress_command(preview), 10)
            assert "busy" not in report.lower() and "🗜️" in report
            assert list(db._conn.execute("SELECT * FROM sessions ORDER BY id")) == before
            assert list(db._conn.execute("SELECT * FROM messages ORDER BY id")) == messages
            assert runner.session_store._entries[key] is entry
            assert runner._agent_cache[key] is cached
            assert state.persistent.run_generation == generation
            assert state.persistent.pending_command_text == "keep queue"
            assert runner._pending_messages[key] == "keep queue"
        finally:
            setup.release.set()
        assert (await asyncio.wait_for(first, 10))[0] == 200
        assert not runner.session_store.canonical_entry_reserved(key)
    asyncio.run(exercise())


def test_compression_route_publication_requires_exact_claim_and_entry(setup):
    store, entry = setup.runner.session_store, setup.entry
    parent = entry.session_id
    token = store.claim_session_command(entry, parent)
    try:
        with pytest.raises(ValueError, match="canonical_turn_busy"):
            store.advance_compression_session(entry.session_key, parent, "child")
        with pytest.raises(ValueError, match="canonical_turn_busy"):
            store.advance_compression_session(
                entry.session_key, parent, "child", command_claim=object(),
                expected_entry=entry,
            )
        assert entry.session_id == parent
        assert store.advance_compression_session(
            entry.session_key, parent, "child", command_claim=token,
            expected_entry=entry,
        ) is entry
        assert entry.session_id == "child"
        with pytest.raises(ValueError, match="canonical_turn_busy"):
            store.claim_session_command(entry, entry.session_id)
    finally:
        store.release_session_command(entry, token)


def test_cancelled_compress_keeps_claim_until_executor_and_cleanup_finish(setup, monkeypatch):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        db = runner.session_store._db
        for role in ("user", "assistant", "user", "assistant"):
            db.append_message(entry.session_id, role=role, content="history")
        entered, release = threading.Event(), threading.Event()
        agent = MagicMock()
        agent.session_id = entry.session_id
        agent._cached_system_prompt = ""
        agent.tools = None
        agent.context_compressor.has_content_to_compress.return_value = True
        agent._last_compaction_in_place = True
        agent._compression_skipped_due_to_lock = False

        def compress(*_args, **_kwargs):
            entered.set()
            assert release.wait(10)
            return ([{"role": "assistant", "content": "summary"}], "")

        agent._compress_context.side_effect = compress
        monkeypatch.setattr("run_agent.AIAgent", lambda **_kwargs: agent)
        monkeypatch.setattr("gateway.run._seed_hygiene_system_prompt", lambda *_args: None)
        monkeypatch.setattr(runner, "_resolve_session_agent_runtime",
                            lambda **_kwargs: ("test-model", {"api_key": "test-key"}))
        event = MessageEvent(text="/compress", source=source, message_id="compress-event")
        task = asyncio.create_task(runner._handle_compress_command(event))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert runner.session_store._command_claims.get(entry.session_key)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert runner.session_store._command_claims.get(entry.session_key)
            with pytest.raises(ValueError, match="canonical_binding_stale"):
                runner.session_store.reserve_canonical_entry(entry)
            assert "busy" in (await runner._handle_compress_command(event)).lower()
        finally:
            release.set()
        for _ in range(100):
            if not runner.session_store._command_claims.get(entry.session_key):
                break
            await asyncio.sleep(0.01)
        assert not runner.session_store._command_claims.get(entry.session_key)
        agent._compress_context.assert_called_once()
    asyncio.run(exercise())


def test_new_cleanup_timeout_detaches_old_agent_before_releasing_claim(setup, monkeypatch):
    async def exercise():
        import gateway.slash_commands as commands

        runner, entry, source = setup.runner, setup.entry, setup.source
        entered, release = threading.Event(), threading.Event()
        monkeypatch.setattr(commands, "_RESET_CLEANUP_TIMEOUT_S", 0.02)

        def paused_cleanup(agent):
            assert agent is setup.agent
            entered.set()
            assert release.wait(10)

        monkeypatch.setattr(runner, "_cleanup_agent_resources", paused_cleanup)
        event = MessageEvent(text="/new", source=source, message_id="new-timeout")
        task = asyncio.create_task(runner._handle_reset_command(event))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            await asyncio.wait_for(task, 5)
            child = runner.session_store._entries[entry.session_key]
            assert child.session_id != entry.session_id
            assert not runner.session_store._command_claims.get(entry.session_key)
            reservation = runner.session_store.reserve_canonical_entry(child)
            try:
                release.set()
                await asyncio.sleep(0.05)
                assert runner.session_store._entries[entry.session_key] is child
                assert runner._agent_cache.get(entry.session_key) is None
            finally:
                runner.session_store.release_canonical_entry(child, reservation)
        finally:
            release.set()

    asyncio.run(exercise())


@pytest.mark.parametrize("caller", ["direct", "busy"])
def test_cancelled_new_keeps_claim_during_old_agent_cleanup(setup, monkeypatch, caller):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        entered, release = threading.Event(), threading.Event()

        def paused_cleanup(agent):
            assert agent is setup.agent
            entered.set()
            assert release.wait(10)

        monkeypatch.setattr(runner, "_cleanup_agent_resources", paused_cleanup)
        event = MessageEvent(text="/new", source=source, message_id="new-cancel")

        async def invoke():
            if caller == "busy":
                return await runner._busy_new_command(event, entry.session_key, source)
            return await runner._handle_reset_command(event)

        task = asyncio.create_task(invoke())
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(ValueError, match="canonical_binding_stale"):
                runner.session_store.reserve_canonical_entry(entry)
            assert "busy" in str(await invoke()).lower()
        finally:
            release.set()
        for _ in range(200):
            if not runner.session_store._command_claims.get(entry.session_key):
                break
            await asyncio.sleep(0.01)
        assert not runner.session_store._command_claims.get(entry.session_key)
        assert runner.session_store._entries[entry.session_key].session_id != entry.session_id

    asyncio.run(exercise())


def test_cancelled_branch_keeps_claim_through_parent_read_and_child_seed(setup, monkeypatch):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        db = runner.session_store._db
        db.append_message(entry.session_id, role="user", content="parent history")
        entered, release = threading.Event(), threading.Event()
        original = runner.session_store.load_transcript

        def paused_read(session_id):
            entered.set()
            assert release.wait(10)
            return original(session_id)

        monkeypatch.setattr(runner.session_store, "load_transcript", paused_read)
        event = MessageEvent(text="/branch alternative", source=source, message_id="branch-cancel")
        task = asyncio.create_task(runner._handle_branch_command(event))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            with pytest.raises(ValueError, match="canonical_binding_stale"):
                runner.session_store.reserve_canonical_entry(entry)
            assert "busy" in (await runner._handle_branch_command(event)).lower()
        finally:
            release.set()
        for _ in range(200):
            if not runner.session_store._command_claims.get(entry.session_key):
                break
            await asyncio.sleep(0.01)
        assert not runner.session_store._command_claims.get(entry.session_key)
        child = runner.session_store._entries[entry.session_key]
        assert child.session_id != entry.session_id
        assert db.get_session(child.session_id)["parent_session_id"] == entry.session_id

    asyncio.run(exercise())


@pytest.mark.parametrize("command", ["new", "branch", "resume", "compress"])
def test_cancelled_request_during_claim_acquisition_retains_worker_ownership(setup, monkeypatch, command):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        entered, release = threading.Event(), threading.Event()
        working, finish = asyncio.Event(), asyncio.Event()
        original_claim = runner.session_store.claim_session_command

        def paused_claim(*args):
            entered.set()
            assert release.wait(10)
            return original_claim(*args)

        async def paused_work(*args, **kwargs):
            working.set()
            await finish.wait()
            return "done"

        monkeypatch.setattr(runner.session_store, "claim_session_command", paused_claim)
        handler = {
            "new": ("_handle_reset_command", "_execute_reset_command", "/new"),
            "branch": ("_handle_branch_command", "_execute_branch_command", "/branch test"),
            "resume": ("_handle_resume_command", "_execute_resume_switch", "/resume prior"),
            "compress": ("_handle_compress_command", "_execute_compress_command", "/compress"),
        }[command]
        monkeypatch.setattr(runner, handler[1], paused_work)
        if command == "resume":
            db = runner.session_store._db
            db.create_session("prior", "telegram", session_key=entry.session_key,
                              user_id="user", chat_id="chat")
            db.set_session_title("prior", "Prior")
        task = asyncio.create_task(getattr(runner, handler[0])(
            MessageEvent(text=handler[2], source=source, message_id="claim-cancel")))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
            await asyncio.wait_for(working.wait(), 10)
            with pytest.raises(ValueError, match="canonical_binding_stale"):
                runner.session_store.reserve_canonical_entry(entry)
        finally:
            release.set()
            finish.set()
        for _ in range(200):
            if not runner.session_store._command_claims.get(entry.session_key):
                break
            await asyncio.sleep(0.01)
        assert not runner.session_store._command_claims.get(entry.session_key)
        token = runner.session_store.reserve_canonical_entry(entry)
        runner.session_store.release_canonical_entry(entry, token)

    asyncio.run(exercise())


@pytest.mark.parametrize("command", ["new", "busy_new", "branch", "resume", "compress"])
def test_cancelled_claimed_worker_during_threaded_claim_releases_before_exit(setup, monkeypatch, command):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        entered, release = threading.Event(), threading.Event()
        original_claim = runner.session_store.claim_session_command
        original_create_task = asyncio.create_task
        worker = []
        worker_names = {
            "new": "claimed_reset", "busy_new": "claimed_busy_new",
            "branch": "claimed_branch", "resume": "claimed_switch",
            "compress": "claimed_compression",
        }

        def paused_claim(*args):
            entered.set()
            assert release.wait(10)
            return original_claim(*args)

        def capture_worker(coro, *args, **kwargs):
            task = original_create_task(coro, *args, **kwargs)
            if coro.cr_code.co_name == worker_names[command]:
                worker.append(task)
            return task

        if command == "resume":
            db = runner.session_store._db
            db.create_session("prior", "telegram", session_key=entry.session_key,
                              user_id="user", chat_id="chat")
            db.set_session_title("prior", "Prior")
        monkeypatch.setattr(runner.session_store, "claim_session_command", paused_claim)
        monkeypatch.setattr(asyncio, "create_task", capture_worker)
        handler = {
            "new": (runner._handle_reset_command, "/new"),
            "busy_new": (None, "/new"),
            "branch": (runner._handle_branch_command, "/branch test"),
            "resume": (runner._handle_resume_command, "/resume prior"),
            "compress": (runner._handle_compress_command, "/compress"),
        }[command]
        event = MessageEvent(text=handler[1], source=source, message_id="worker-cancel")
        requester = original_create_task(
            runner._busy_new_command(event, entry.session_key, source)
            if command == "busy_new" else handler[0](event)
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert len(worker) == 1
            worker[0].cancel()
            worker[0].cancel()
            await asyncio.sleep(0)
            assert not worker[0].done(), "claimed worker exited before its claim thread"
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(requester, 10)
        assert worker[0].done()
        assert not runner.session_store._command_claims.get(entry.session_key)
        token = runner.session_store.reserve_canonical_entry(entry)
        runner.session_store.release_canonical_entry(entry, token)

    asyncio.run(exercise())


@pytest.mark.parametrize("stage", ["execution", "release"])
def test_cancelled_branch_worker_drains_threaded_stage_before_releasing_claim(setup, monkeypatch, stage):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        entered, finish = threading.Event(), threading.Event()
        original_create_task = asyncio.create_task
        original_release = runner.session_store.release_session_command
        worker = []

        def capture_worker(coro, *args, **kwargs):
            task = original_create_task(coro, *args, **kwargs)
            if coro.cr_code.co_name == "claimed_branch":
                worker.append(task)
            return task

        def paused_thread():
            entered.set()
            assert finish.wait(10)

        async def threaded_execution(*args):
            await asyncio.to_thread(paused_thread)
            return "done"

        def threaded_release(*args):
            paused_thread()
            return original_release(*args)

        monkeypatch.setattr(asyncio, "create_task", capture_worker)
        monkeypatch.setattr(runner, "_execute_branch_command",
                            threaded_execution if stage == "execution" else lambda *args: asyncio.sleep(0))
        if stage == "release":
            monkeypatch.setattr(runner.session_store, "release_session_command", threaded_release)
        event = MessageEvent(text="/branch test", source=source, message_id="thread-stage")
        requester = original_create_task(runner._handle_branch_command(event))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            assert len(worker) == 1
            worker[0].cancel()
            await asyncio.sleep(0)
            worker[0].cancel()
            await asyncio.sleep(0)
            assert not worker[0].done()
            with pytest.raises(ValueError, match="canonical_binding_stale"):
                runner.session_store.reserve_canonical_entry(entry)
        finally:
            finish.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(requester, 10)
        assert not runner.session_store._command_claims.get(entry.session_key)
        token = runner.session_store.reserve_canonical_entry(entry)
        runner.session_store.release_canonical_entry(entry, token)

    asyncio.run(exercise())


@pytest.mark.parametrize("command", ["new", "busy_new", "branch", "resume", "compress"])
def test_command_refuses_repointed_route_during_claim_acquisition(setup, monkeypatch, command):
    async def exercise():
        runner, entry, source = setup.runner, setup.entry, setup.source
        store, db = runner.session_store, runner.session_store._db
        original_id = entry.session_id
        successor_id = "successor_for_claim_race"
        db.create_session(successor_id, "telegram", session_key=entry.session_key,
                          user_id="user", chat_id="chat")
        if command == "resume":
            db.create_session("prior", "telegram", session_key=entry.session_key,
                              user_id="user", chat_id="chat")
            db.set_session_title("prior", "Prior")
        original_create_task = asyncio.create_task
        scheduled = []
        worker_names = {
            "new": "claimed_reset", "busy_new": "claimed_busy_new",
            "branch": "claimed_branch", "resume": "claimed_switch",
            "compress": "claimed_compression",
        }

        def repoint_at_worker_scheduling(coro, *args, **kwargs):
            if coro.cr_code.co_name == worker_names[command]:
                assert not scheduled
                scheduled.append(coro.cr_code.co_name)
                assert store.repoint_session_entry(entry, original_id, successor_id)
            return original_create_task(coro, *args, **kwargs)

        async def forbidden_work(*args, **kwargs):
            pytest.fail(f"/{command} executed against a repointed route")

        monkeypatch.setattr(asyncio, "create_task", repoint_at_worker_scheduling)
        handler = {
            "new": (runner._handle_reset_command, "_execute_reset_command", "/new"),
            "busy_new": (None, "_busy_new_command_claimed", "/new"),
            "branch": (runner._handle_branch_command, "_execute_branch_command", "/branch test"),
            "resume": (runner._handle_resume_command, "_execute_resume_switch", "/resume prior"),
            "compress": (runner._handle_compress_command, "_execute_compress_command", "/compress"),
        }[command]
        monkeypatch.setattr(runner, handler[1], forbidden_work)
        event = MessageEvent(text=handler[2], source=source, message_id="repoint-race")
        task = asyncio.create_task(
            runner._busy_new_command(event, entry.session_key, source)
            if command == "busy_new" else handler[0](event)
        )
        result = await asyncio.wait_for(task, 10)
        assert scheduled == [worker_names[command]]
        assert "busy" in str(result).lower()
        assert store._entries[entry.session_key] is entry
        assert entry.session_id == successor_id
        assert not store._command_claims.get(entry.session_key)
        assert db.get_session(successor_id)["end_reason"] is None

    asyncio.run(exercise())
