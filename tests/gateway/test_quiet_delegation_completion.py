"""``delegation.completion_delivery: quiet``: a background delegation's result on a push chat is recorded
in the session the chat's next message reads, and the chat gets a plain notice; no agent turn starts."""

import asyncio
import queue
import threading
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_state import AsyncSessionDB, SessionDB


class AdmittingHandler(AsyncMock):
    async def _execute_mock_call(self, event, *args, **kwargs):
        result = await super()._execute_mock_call(event, *args, **kwargs)
        event._gateway_accepted = True
        return result


SESSION_KEY = "agent:main:telegram:dm:12345:678"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import gateway.run as gateway_run
    import tools.process_registry as pr_module

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(pr_module, "CHECKPOINT_PATH", tmp_path / "processes.json")
    registry = pr_module.ProcessRegistry()
    registry.completion_queue = queue.Queue()
    monkeypatch.setattr(pr_module, "process_registry", registry)
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("sess_parent", source="telegram")
    yield SimpleNamespace(path=tmp_path, db=db, registry=registry)
    db.close()


def _config(home, mode):
    (home.path / "config.yaml").write_text(f"delegation:\n  completion_delivery: {mode}\n", encoding="utf-8")


def _runner(home):
    runner = object.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(
        handle_message=AdmittingHandler(), send=AsyncMock(return_value=SimpleNamespace(success=True)), config=None)}
    entry = SimpleNamespace(session_key=SESSION_KEY, session_id="sess_parent")
    runner.session_store = SimpleNamespace(_ensure_loaded=lambda: None, _entries={},
                                           get_or_create_session=lambda source, **kwargs: entry)
    runner._session_source_cache = {}
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    runner._session_db = AsyncSessionDB(home.db)
    runner._peek_session_state = lambda session_key: None
    runner._thread_metadata_for_source = lambda source: None
    # The notice must use the transport the completion resolved, even when the generic lookup finds none.
    runner._delivery_adapter_for = lambda source: None
    return runner


def _event(delegation_id="deleg_quiet", **extra):
    return {"type": "async_delegation", "delegation_id": delegation_id, "session_key": SESSION_KEY,
            "parent_session_id": "sess_parent", "goal": "Investigate flaky test", "status": "completed",
            "summary": "Found it", "api_calls": 1, "duration_seconds": 12.0, "dispatched_at": 1000.0,
            "completed_at": 1012.0, **extra}


def _persist_pending(event):
    from tools import async_delegation

    async_delegation._persist_dispatch({"delegation_id": event["delegation_id"], "session_key": event["session_key"],
                                        "origin_ui_session_id": "", "parent_session_id": event["parent_session_id"],
                                        "dispatched_at": event["dispatched_at"]})
    async_delegation._persist_completion(event, {"status": "completed", "summary": event["summary"]})


def _durable(delegation_id):
    from tools import async_delegation

    return async_delegation.get_durable_delegation(delegation_id)


def _deliver(runner, event):
    from gateway.run import _format_gateway_process_notification

    return asyncio.run(runner._deliver_completion_notification(_format_gateway_process_notification(event), event))


def test_quiet_records_the_result_for_the_next_turn_and_sends_a_plain_notice(home):
    _config(home, "quiet")
    runner = _runner(home)
    event = _event()
    _persist_pending(event)

    assert _deliver(runner, event) is True

    runner.adapters[Platform.TELEGRAM].handle_message.assert_not_awaited()
    rows = home.db.get_messages("sess_parent")
    assert len(rows) == 1 and rows[0]["role"] == "user" and rows[0]["display_kind"] == "async_delegation_complete"
    history = home.db.get_messages_as_conversation("sess_parent")
    assert any(m["role"] == "user" and "Found it" in m["content"] for m in history), "the next turn reads the result"
    adapter = runner.adapters[Platform.TELEGRAM]
    adapter.send.assert_awaited_once()
    notice = adapter.send.await_args.args[1]
    assert "Investigate flaky test" in notice and "Found it" not in notice
    assert _durable("deleg_quiet")["delivery_state"] == "delivered"


def test_quiet_defers_while_a_turn_holds_the_session_then_records_once(home):
    _config(home, "quiet")
    runner = _runner(home)
    event = _event()
    _persist_pending(event)
    assert home.db.acquire_session_turn_lease("sess_parent", "live-turn", wait_seconds=0)
    try:
        assert _deliver(runner, dict(event)) is False
        assert home.db.get_messages("sess_parent") == []
        runner.adapters[Platform.TELEGRAM].send.assert_not_awaited()
        row = _durable("deleg_quiet")
        assert row["delivery_state"] == "pending" and row["delivery_attempts"] == 0, "deferred, no attempt spent"
    finally:
        home.db.release_session_turn_lease("sess_parent", "live-turn")

    assert _deliver(runner, dict(event)) is True
    assert len(home.db.get_messages("sess_parent")) == 1
    runner.adapters[Platform.TELEGRAM].send.assert_awaited_once()
    runner.adapters[Platform.TELEGRAM].handle_message.assert_not_awaited()


def test_wake_stays_the_default(home):
    runner = _runner(home)
    assert runner._load_delegation_completion_delivery() == "wake"
    event = _event()
    _persist_pending(event)

    assert _deliver(runner, event) is True
    runner.adapters[Platform.TELEGRAM].handle_message.assert_awaited_once()
    assert home.db.get_messages("sess_parent") == []
    runner.adapters[Platform.TELEGRAM].send.assert_not_awaited()


def test_quiet_without_a_spawning_session_keeps_the_wake_path(home):
    _config(home, "quiet")
    runner = _runner(home)
    event = _event(parent_session_id="")

    assert _deliver(runner, event) is True
    runner.adapters[Platform.TELEGRAM].handle_message.assert_awaited_once()
    assert home.db.get_messages("sess_parent") == []


def test_unknown_completion_delivery_value_falls_back_to_wake(home):
    _config(home, "silent")
    assert _runner(home)._load_delegation_completion_delivery() == "wake"


def test_quiet_group_records_every_unit_under_its_own_id_on_replay(home):
    """A replay after unit A's row was written but before its claim was acknowledged, now grouped with
    unit B: B's result must be recorded too, not acknowledged under A's existing row."""
    _config(home, "quiet")
    runner = _runner(home)
    evt_a = _event("deleg_a", summary="Result A")
    evt_b = _event("deleg_b", summary="Result B", goal="Second task")
    for evt in (evt_a, evt_b):
        _persist_pending(evt)
    home.db.append_delegation_delivery("sess_parent", "earlier attempt for A", {"delegation_id": "deleg_a"})

    assert asyncio.run(runner._deliver_async_delegation_group([dict(evt_a), dict(evt_b)])) is True

    rows = home.db.get_messages("sess_parent")
    assert len(rows) == 2, "A keeps its one row and B gets its own"
    assert any("Result B" in row["content"] for row in rows), "B's result is recorded"
    assert _durable("deleg_a")["delivery_state"] == "delivered"
    assert _durable("deleg_b")["delivery_state"] == "delivered"
    runner.adapters[Platform.TELEGRAM].handle_message.assert_not_awaited()


@pytest.mark.parametrize("failure", ["unsuccessful", "raises"])
def test_quiet_unsent_notice_keeps_the_claim_and_resends_only_the_notice(home, failure):
    _config(home, "quiet")
    runner = _runner(home)
    adapter = runner.adapters[Platform.TELEGRAM]
    first = SimpleNamespace(success=False) if failure == "unsuccessful" else RuntimeError("telegram down")
    adapter.send = AsyncMock(side_effect=[first, SimpleNamespace(success=True)])
    event = _event()
    _persist_pending(event)

    assert _deliver(runner, dict(event)) is False
    assert len(home.db.get_messages("sess_parent")) == 1, "the result is recorded once"
    assert _durable("deleg_quiet")["delivery_state"] == "pending", "the claim stays retryable"

    assert _deliver(runner, dict(event)) is True
    assert len(home.db.get_messages("sess_parent")) == 1, "the replay re-finds the row instead of writing another"
    assert adapter.send.await_count == 2
    assert _durable("deleg_quiet")["delivery_state"] == "delivered"


def test_mode_change_during_a_group_does_not_record_one_unit_under_another(home, monkeypatch):
    """The group decides wake once; a switch to quiet before the delivery must not record the coalesced
    text under the primary's existing row and acknowledge the sibling without its result."""
    runner = _runner(home)
    reads = iter(["wake"])
    monkeypatch.setattr(runner, "_load_delegation_completion_delivery", lambda: next(reads, "quiet"))
    evt_a = _event("deleg_a", summary="Result A")
    evt_b = _event("deleg_b", summary="Result B", goal="Second task")
    for evt in (evt_a, evt_b):
        _persist_pending(evt)
    home.db.append_delegation_delivery("sess_parent", "earlier attempt for A", {"delegation_id": "deleg_a"})

    assert asyncio.run(runner._deliver_async_delegation_group([dict(evt_a), dict(evt_b)])) is True

    handler = runner.adapters[Platform.TELEGRAM].handle_message
    handler.assert_awaited_once()
    assert "Result B" in handler.await_args.args[0].text, "B's result is delivered in the one wake turn"
    assert len(home.db.get_messages("sess_parent")) == 1, "nothing was recorded quietly"


def test_quiet_group_without_spawning_sessions_stays_one_wake_turn(home):
    _config(home, "quiet")
    runner = _runner(home)
    events = [_event("deleg_x", parent_session_id="", summary="Result X"),
              _event("deleg_y", parent_session_id="", summary="Result Y", goal="Other task")]
    for evt in events:
        _persist_pending(evt)

    assert asyncio.run(runner._deliver_async_delegation_group([dict(e) for e in events])) is True

    runner.adapters[Platform.TELEGRAM].handle_message.assert_awaited_once()
    assert _durable("deleg_x")["delivery_state"] == "delivered"
    assert _durable("deleg_y")["delivery_state"] == "delivered"


def test_quiet_notice_primes_a_cold_relay_before_sending(home):
    """A relay's per-chat routing is cold right after a restart; the notice must leave with the chat's
    logical platform and DM user, as the wake path primes them."""
    from gateway.relay.adapter import RelayAdapter
    from gateway.session import SessionSource

    _config(home, "quiet")
    runner = _runner(home)
    relay = object.__new__(RelayAdapter)
    relay._scope_by_chat, relay._dm_user_by_chat, relay._platform_by_chat = {}, {}, {}
    relay._chat_type_by_chat, relay._last_inbound_ts_by_chat = {}, {}
    relay.handle_message = AdmittingHandler()
    seen = {}

    async def send(chat_id, content, metadata=None, **kwargs):
        seen.update(platform=relay._platform_by_chat.get(str(chat_id)), scope=relay._with_scope(str(chat_id), None))
        return SimpleNamespace(success=True)

    relay.send = send
    runner.adapters = {Platform.TELEGRAM: relay}
    runner.session_store._entries[SESSION_KEY] = SimpleNamespace(origin=SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", thread_id="678", user_id="u1"))
    event = _event()
    _persist_pending(event)

    assert _deliver(runner, event) is True
    assert seen, "the notice was sent"
    assert str(getattr(seen["platform"], "value", seen["platform"])) == "telegram"
    assert seen["scope"].get("user_id") == "u1"
    relay.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_quiet_notice_does_not_seal_an_open_foreground_stream(home):
    """A quiet notice can be delivered while a Relay foreground stream is open; it must not be
    absorbed as that stream's final response (QD-04)."""
    from gateway.session import SessionSource
    from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
    from tests.gateway.relay.test_relay_live_cards import _connected_adapter

    _config(home, "quiet")
    runner = _runner(home)
    relay, _stub = _connected_adapter()
    relay.handle_message = AdmittingHandler()
    runner.adapters = {Platform.TELEGRAM: relay}
    runner.session_store._entries[SESSION_KEY] = SimpleNamespace(
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="C1", chat_type="dm", user_id="u1"))
    event = _event()
    _persist_pending(event)

    cfg = StreamConsumerConfig(transport="auto", chat_type="dm", edit_interval=0.01, buffer_threshold=1, cursor="")
    consumer = GatewayStreamConsumer(relay, "C1", cfg, metadata={"thread_ts": "1700.8", "message_id": "1700.801"})
    task = asyncio.create_task(consumer.run())
    consumer.on_delta("partial answer on screen")
    await asyncio.sleep(0.08)
    try:
        assert relay._open_draft_by_chat, "the foreground stream is open"
        delivered = await runner._deliver_completion_notification(
            _format_delivery_text(event), event)
        assert delivered is True
        assert relay._open_draft_by_chat, "the notice must not have sealed the open stream"
    finally:
        consumer.finish(final_text="final answer")
        await asyncio.sleep(0.08)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    assert len(home.db.get_messages("sess_parent")) == 1


def _format_delivery_text(event):
    from gateway.run import _format_gateway_process_notification

    return _format_gateway_process_notification(event)
