"""An uncertain live cron send must not be recorded as delivered.

When the live adapter's send had begun but did not confirm within the timeout, the scheduler
correctly refused a standalone resend (it could duplicate the message), but then reported the
target as delivered: the continuation sessions were seeded as for a confirmed send and nothing was
written to ``last_delivery_unverified``. These tests drive a real gateway loop whose send stalls
past a shortened confirmation timeout, through the real job store. (``run_one_job`` hands its
send to the durable delivery queue, whose drain calls ``_deliver_result``; the queue's own
uncertain handling is covered separately.)
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from agent.monitoring.cron_health import project_execution_event
from cron import jobs, scheduler, scheduler_delivery
from gateway.config import GatewayConfig, Platform, PlatformConfig


@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(started.set)
        loop.run_forever()

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(5)
    yield loop
    for task in asyncio.all_tasks(loop) if not loop.is_closed() else ():
        loop.call_soon_threadsafe(task.cancel)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


@pytest.fixture
def stalled_live_send(tmp_path, monkeypatch, loop_thread):
    """A real HERMES_HOME, a Telegram target, and a live adapter whose send starts and then
    never confirms within the (shortened) timeout. Records every send and every seed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("cron: {wrap_response: false}\n")
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_SECS", 0.2)
    calls = SimpleNamespace(live=[], standalone=[], seeded=[])
    live_started = threading.Event()

    async def live_send(chat_id, content, metadata=None):
        calls.live.append(content)
        live_started.set()
        await asyncio.sleep(30)  # accepted by the transport, confirmation never arrives in time
        return SimpleNamespace(success=True, message_id="m-live", raw_response={})

    async def standalone_send(*args, **kwargs):
        calls.standalone.append(args)
        return {"success": True, "message_id": "m-standalone"}

    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone_send)
    monkeypatch.setattr(
        scheduler_delivery, "_seed_live_delivery_sessions",
        lambda t, message_id: calls.seeded.append((t.where, message_id)))
    calls.adapters = {Platform.TELEGRAM: SimpleNamespace(send=live_send)}
    calls.loop = loop_thread
    calls.live_started = live_started
    return calls


def _assert_uncertain_not_delivered(job_id, calls):
    assert calls.live_started.is_set(), "the live send never began; this would not be the race"
    assert calls.standalone == [], "an uncertain send must never be resent"
    assert calls.seeded == [], "an unconfirmed send must not seed continuation sessions"
    saved = jobs.get_job(job_id)
    assert saved["last_delivery_unverified"] and any(
        "telegram:123" in str(entry) for entry in saved["last_delivery_unverified"]), saved


def test_deliver_result_marks_the_target_uncertain(stalled_live_send):
    calls = stalled_live_send
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    assert scheduler._deliver_result(job, "the brief", adapters=calls.adapters, loop=calls.loop) is None

    assert job.get("_delivery_uncertain") == ["telegram:123"]
    assert len(calls.live) == 1
    _assert_uncertain_not_delivered(job["id"], calls)


def test_cron_health_counts_an_uncertain_delivery_as_known():
    record = {"status": "completed", "source": "builtin", "job_id": "job-1"}

    event = project_execution_event(record, delivery_outcome="uncertain")

    assert event.delivery_outcome == "uncertain"


# The uncertain text send carried a MEDIA attachment. The attachment is skipped (the loop is
# contended) and that skip is reported as a partial-delivery error; the uncertainty must survive
# alongside it rather than be dropped.


@pytest.fixture
def media_brief(tmp_path, monkeypatch):
    media = tmp_path / "banner.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(
        "gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
        staticmethod(lambda files, *args, **kwargs: list(files)))
    return f"the brief\nMEDIA:{media}"


def test_uncertain_send_with_skipped_media_stays_uncertain(stalled_live_send, media_brief):
    calls = stalled_live_send
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    error = scheduler._deliver_result(job, media_brief, adapters=calls.adapters, loop=calls.loop)

    assert error and "media attachment" in error, \
        "the skipped attachment must still be reported alongside the uncertainty"
    assert job.get("_delivery_uncertain") == ["telegram:123"]
    _assert_uncertain_not_delivered(job["id"], calls)


def test_mixed_targets_uncertain_and_failed_keeps_both(stalled_live_send, monkeypatch):
    """One target's send began and never confirmed, another failed outright on both lanes: the
    failure is recorded, and the first target is still uncertain (its message may have landed)."""
    calls = stalled_live_send
    stalled = calls.adapters[Platform.TELEGRAM].send

    async def live_send(chat_id, content, metadata=None):
        if str(chat_id) == "456":
            raise RuntimeError("chat not found")
        return await stalled(chat_id, content, metadata)

    async def standalone_fails(*args, **kwargs):
        calls.standalone.append(args)
        return {"error": "chat not found"}

    calls.adapters[Platform.TELEGRAM] = SimpleNamespace(send=live_send)
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone_fails)
    job = jobs.create_job(
        prompt="fixture only", schedule="every 1h", deliver="telegram:123,telegram:456")

    error = scheduler._deliver_result(job, "the brief", adapters=calls.adapters, loop=calls.loop)

    assert error and "456" in error, "the failed target must still be reported"
    assert calls.standalone and all("456" in str(args) for args in calls.standalone), \
        "123 must never be resent"
    assert job.get("_delivery_uncertain") == ["telegram:123"]
