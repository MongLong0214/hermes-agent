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
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(handle_message=AdmittingHandler())}
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
    runner._deliver_platform_notice = AsyncMock()
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
    runner._deliver_platform_notice.assert_awaited_once()
    notice = runner._deliver_platform_notice.await_args.args[1]
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
        runner._deliver_platform_notice.assert_not_awaited()
        row = _durable("deleg_quiet")
        assert row["delivery_state"] == "pending" and row["delivery_attempts"] == 0, "deferred, no attempt spent"
    finally:
        home.db.release_session_turn_lease("sess_parent", "live-turn")

    assert _deliver(runner, dict(event)) is True
    assert len(home.db.get_messages("sess_parent")) == 1
    runner._deliver_platform_notice.assert_awaited_once()
    runner.adapters[Platform.TELEGRAM].handle_message.assert_not_awaited()


def test_wake_stays_the_default(home):
    runner = _runner(home)
    assert runner._load_delegation_completion_delivery() == "wake"
    event = _event()
    _persist_pending(event)

    assert _deliver(runner, event) is True
    runner.adapters[Platform.TELEGRAM].handle_message.assert_awaited_once()
    assert home.db.get_messages("sess_parent") == []
    runner._deliver_platform_notice.assert_not_awaited()


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
