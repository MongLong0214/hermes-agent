"""Regression for sol-audit R68-2: an uncertain live cron send was recorded as delivered.

When the live adapter's send had begun but did not confirm within the timeout, the scheduler
correctly refused a standalone resend (it could duplicate the message), but then reported the
target as delivered: ``_delivery_accepted`` was set, the continuation sessions were seeded as for
a confirmed send, nothing was written to ``last_delivery_unverified``, and the execution ledger
row said ``delivered``. These tests drive a real gateway loop whose send stalls past a shortened
confirmation timeout, through the real job store, execution ledger and durable delivery queue.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from cron import delivery_queue, executions, jobs, scheduler, scheduler_delivery
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
    monkeypatch.setattr(scheduler_delivery, "_LIVE_SEND_CONFIRM_TIMEOUT_S", 0.2)
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
    row = executions.latest_execution(job_id)
    assert row["delivery_outcome"] == "uncertain", row


def test_in_flight_timeout_is_recorded_uncertain_not_delivered(stalled_live_send, monkeypatch):
    calls = stalled_live_send
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    assert scheduler.run_one_job(job, adapters=calls.adapters, loop=calls.loop) is True

    assert job.get("_delivery_accepted") is None
    _assert_uncertain_not_delivered(job["id"], calls)


def test_deliver_result_marks_the_target_uncertain_not_accepted(stalled_live_send):
    calls = stalled_live_send
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    assert scheduler._deliver_result(job, "the brief", adapters=calls.adapters, loop=calls.loop) is None

    assert not job.get("_delivery_accepted")
    assert job.get("_delivery_uncertain") == ["telegram:123"]
    assert calls.standalone == [] and calls.seeded == []


def test_queued_worker_delivery_records_the_uncertain_outcome(stalled_live_send, monkeypatch):
    """Durable-queue path: the gateway draining a restart-safe worker's row hits the same
    stalled send. The row must end ``unknown`` (never replayed), and the worker's own job and
    execution records must say uncertain, not delivered."""
    calls = stalled_live_send
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")
    execution = executions.create_execution(job["id"], source="fixture")
    job["execution_id"] = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
    original_wait = delivery_queue.enqueue_and_wait

    def drain_before_wait(execution_id, queued_job, content, *, for_failure=False):
        delivery_queue.enqueue(execution_id, queued_job, content, for_failure=for_failure)
        assert scheduler.drain_delivery_queue(calls.adapters, calls.loop) == 1
        return original_wait(execution_id, queued_job, content, for_failure=for_failure)

    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", drain_before_wait)

    scheduler.run_one_job(job)

    assert delivery_queue.get_status(execution["id"])["status"] == "unknown"
    assert len(calls.live) == 1
    _assert_uncertain_not_delivered(job["id"], calls)


# Round-2 R68-2: the uncertain text send carried a MEDIA attachment. The attachment is skipped (the
# loop is contended) and that skip is reported as a partial-delivery error; the error then
# overrode the uncertainty, so both the ledger and the queue row said ``failed``. The uncertainty
# must survive, and the partial error must be kept alongside it rather than dropped.


@pytest.fixture
def media_brief(tmp_path, monkeypatch):
    media = tmp_path / "banner.png"
    media.write_bytes(b"\x89PNG\r\n\x1a\n")
    monkeypatch.setattr(
        "gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
        staticmethod(lambda files, *args, **kwargs: list(files)))
    return f"the brief\nMEDIA:{media}"


def test_uncertain_send_with_skipped_media_stays_uncertain(stalled_live_send, media_brief, monkeypatch):
    calls = stalled_live_send
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", media_brief, None))
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    assert scheduler.run_one_job(job, adapters=calls.adapters, loop=calls.loop) is True

    _assert_uncertain_not_delivered(job["id"], calls)
    assert jobs.get_job(job["id"])["last_delivery_error"], \
        "the skipped attachment must still be reported alongside the uncertainty"


def test_mixed_targets_uncertain_and_failed_stays_uncertain(stalled_live_send, monkeypatch):
    """One target's send began and never confirmed, another failed outright on both lanes: the
    failure is recorded, but the run is still uncertain (one message may have landed)."""
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
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(
        prompt="fixture only", schedule="every 1h", deliver="telegram:123,telegram:456")

    assert scheduler.run_one_job(job, adapters=calls.adapters, loop=calls.loop) is True

    assert all("456" in str(args) for args in calls.standalone), "123 must never be resent"
    assert job.get("_delivery_uncertain") == ["telegram:123"]
    assert jobs.get_job(job["id"])["last_delivery_error"], "the failed target must still be reported"
    assert executions.latest_execution(job["id"])["delivery_outcome"] == "uncertain"


def test_queued_uncertain_send_with_skipped_media_stays_unknown(
        stalled_live_send, media_brief, monkeypatch):
    calls = stalled_live_send
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", media_brief, None))
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")
    execution = executions.create_execution(job["id"], source="fixture")
    job["execution_id"] = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
    original_wait = delivery_queue.enqueue_and_wait

    def drain_before_wait(execution_id, queued_job, content, *, for_failure=False):
        delivery_queue.enqueue(execution_id, queued_job, content, for_failure=for_failure)
        assert scheduler.drain_delivery_queue(calls.adapters, calls.loop) == 1
        return original_wait(execution_id, queued_job, content, for_failure=for_failure)

    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", drain_before_wait)

    scheduler.run_one_job(job)

    row = delivery_queue.get_status(execution["id"])
    assert row["status"] == "unknown", row
    assert "media attachment" in (row["error"] or ""), row
    assert len(calls.live) == 1
    _assert_uncertain_not_delivered(job["id"], calls)
    assert jobs.get_job(job["id"])["last_delivery_error"], "the skipped attachment was dropped"
