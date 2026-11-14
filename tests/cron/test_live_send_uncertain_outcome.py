"""An uncertain live cron send must not be recorded as delivered.

When the live adapter's send had begun but did not confirm within the timeout, the scheduler
correctly refused a standalone resend (it could duplicate the message), but then reported the
target as delivered: the continuation sessions were seeded as for a confirmed send and nothing was
written to ``last_delivery_unverified``. These tests drive a real gateway loop whose send stalls
past a shortened confirmation timeout, through the real job store, execution ledger and durable
delivery queue. ``run_one_job`` hands an agent run's send to the queue, whose gateway drain calls
``_deliver_result``; a script run and a crash notice send directly.
"""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from agent.monitoring.cron_health import project_execution_event
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


def _run_then_drain(calls, monkeypatch, job, *, drain_timing="after_run"):
    """Run an agent job (it only queues its send), and let the gateway drain the row through the
    stalled live adapter: after the run, or before the run's own bookkeeping lands."""
    if drain_timing == "before_bookkeeping":
        real_enqueue = delivery_queue.enqueue

        def enqueue_then_drain(*args, **kwargs):
            queued = real_enqueue(*args, **kwargs)
            assert scheduler.drain_delivery_queue(calls.adapters, calls.loop) == 1
            return queued

        monkeypatch.setattr(delivery_queue, "enqueue", enqueue_then_drain)
    scheduler.run_one_job(job)
    if drain_timing == "after_run":
        assert calls.live == [], "the run only queues; the gateway drain sends"
        assert scheduler.drain_delivery_queue(calls.adapters, calls.loop) == 1
    execution = executions.latest_execution(job["id"])
    return execution, delivery_queue.get_status(execution["id"])


@pytest.mark.parametrize("drain_timing", ["after_run", "before_bookkeeping"])
def test_queued_send_that_began_unconfirmed_is_recorded_uncertain(
        stalled_live_send, monkeypatch, drain_timing):
    """The drained row ends ``unknown`` and is never replayed; the run is neither delivered nor
    failed: ledger ``uncertain``, job ``ok`` with no delivery error, the target unverified."""
    calls = stalled_live_send
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    execution, queued = _run_then_drain(calls, monkeypatch, job, drain_timing=drain_timing)

    assert queued["status"] == "unknown", queued
    assert queued["error"] == delivery_queue.UNCERTAIN_SEND_ERROR, queued
    assert len(calls.live) == 1
    _assert_uncertain_not_delivered(job["id"], calls)
    assert execution["delivery_outcome"] == "uncertain", execution
    saved = jobs.get_job(job["id"])
    assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None), saved
    assert scheduler.drain_delivery_queue(calls.adapters, calls.loop) == 0
    assert len(calls.live) == 1, "an uncertain queued send is never replayed"


def test_queued_uncertain_send_with_skipped_media_stays_unknown(
        stalled_live_send, media_brief, monkeypatch):
    calls = stalled_live_send
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", media_brief, None))
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    execution, queued = _run_then_drain(calls, monkeypatch, job)

    assert queued["status"] == "unknown", queued
    assert queued["error"].startswith(delivery_queue.UNCERTAIN_SEND_ERROR + "; "), queued
    assert "media attachment" in queued["error"], "the skipped attachment was dropped"
    assert len(calls.live) == 1
    _assert_uncertain_not_delivered(job["id"], calls)
    assert execution["delivery_outcome"] == "uncertain", execution
    saved = jobs.get_job(job["id"])
    assert saved["last_status"] == "delivery_failed", saved
    assert saved["last_delivery_error"] == delivery_queue.UNCERTAIN_SEND_ERROR, saved


def _one_target_refuses(calls, monkeypatch, provider_text):
    stalled = calls.adapters[Platform.TELEGRAM].send

    async def live_send(chat_id, content, metadata=None):
        if str(chat_id) == "456":
            raise RuntimeError(provider_text)
        return await stalled(chat_id, content, metadata)

    async def standalone_fails(*args, **kwargs):
        calls.standalone.append(args)
        return {"error": provider_text}

    calls.adapters[Platform.TELEGRAM] = SimpleNamespace(send=live_send)
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone_fails)


PROVIDER_TEXT = 'Bad Request: chat not found {"ok":false,"error_code":400}'


def test_queued_mixed_targets_keep_provider_text_off_the_job_record(stalled_live_send, monkeypatch):
    """One target began and never confirmed, another was refused with a provider body: the run is
    still uncertain, the refusal stays on the private queue row, and the job record says only the
    fixed uncertain text."""
    calls = stalled_live_send
    _one_target_refuses(calls, monkeypatch, PROVIDER_TEXT)
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(
        prompt="fixture only", schedule="every 1h", deliver="telegram:123,telegram:456")

    execution, queued = _run_then_drain(calls, monkeypatch, job)

    assert calls.standalone and all("456" in str(args) for args in calls.standalone), \
        "123 must never be resent"
    assert queued["status"] == "unknown" and "chat not found" in queued["error"], queued
    assert execution["delivery_outcome"] == "uncertain", execution
    saved = jobs.get_job(job["id"])
    assert saved["last_status"] == "delivery_failed", saved
    assert saved["last_delivery_error"] == delivery_queue.UNCERTAIN_SEND_ERROR, saved


def test_direct_send_that_began_unconfirmed_is_recorded_uncertain(stalled_live_send, monkeypatch):
    """A script run sends directly through the live adapter: same outcome on the ledger, and the
    other target's provider text does not reach the job record."""
    calls = stalled_live_send
    _one_target_refuses(calls, monkeypatch, PROVIDER_TEXT)
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(
        prompt="fixture only", schedule="every 1h", deliver="telegram:123,telegram:456")

    assert scheduler.run_one_job(
        dict(job, no_agent=True), adapters=calls.adapters, loop=calls.loop) is True

    assert delivery_queue.get_status(executions.latest_execution(job["id"])["id"]) is None
    assert calls.standalone and all("456" in str(args) for args in calls.standalone), \
        "123 must never be resent"
    assert calls.seeded == [], "an unconfirmed send must not seed continuation sessions"
    assert executions.latest_execution(job["id"])["delivery_outcome"] == "uncertain"
    saved = jobs.get_job(job["id"])
    assert saved["last_delivery_error"] == delivery_queue.UNCERTAIN_SEND_ERROR, saved


def test_crash_notice_that_began_unconfirmed_is_recorded_uncertain(stalled_live_send, monkeypatch):
    calls = stalled_live_send

    def crash(job, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(scheduler, "run_job", crash)
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")

    assert scheduler.run_one_job(job, adapters=calls.adapters, loop=calls.loop) is False

    _assert_uncertain_not_delivered(job["id"], calls)
    assert executions.latest_execution(job["id"])["delivery_outcome"] == "uncertain"
    assert jobs.get_job(job["id"])["last_delivery_error"] is None


# A restart-safe worker has no live adapters: a script run's send and a crash notice ride the
# durable queue, and the worker waits for the gateway's drain (enqueue_and_wait). The drained row
# says the send began and never confirmed; the worker must book that as uncertain too.


def _in_restart_safe_worker(calls, monkeypatch, job):
    """Make this process *job*'s restart-safe worker; its queued send is drained by the gateway
    (the stalled live adapter) before the worker waits on the row."""
    execution = executions.create_execution(job["id"], source="fixture")
    job["execution_id"] = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
    real_wait = delivery_queue.enqueue_and_wait

    def drain_then_wait(execution_id, queued_job, content, *, for_failure=False):
        delivery_queue.enqueue(execution_id, queued_job, content, for_failure=for_failure)
        assert scheduler.drain_delivery_queue(calls.adapters, calls.loop) == 1
        return real_wait(execution_id, queued_job, content, for_failure=for_failure)

    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", drain_then_wait)
    return execution["id"]


@pytest.mark.parametrize("mixed", [False, True])
def test_worker_script_send_that_began_unconfirmed_is_recorded_uncertain(
        stalled_live_send, monkeypatch, mixed):
    calls = stalled_live_send
    deliver = "telegram:123"
    if mixed:
        _one_target_refuses(calls, monkeypatch, PROVIDER_TEXT)
        deliver += ",telegram:456"
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver=deliver)
    execution_id = _in_restart_safe_worker(calls, monkeypatch, job)

    assert scheduler.run_one_job(dict(job, no_agent=True)) is True

    assert delivery_queue.get_status(execution_id)["status"] == "unknown"
    assert len(calls.live) == 1 and calls.seeded == []
    assert all("456" in str(args) for args in calls.standalone), "123 must never be resent"
    assert executions.get_execution(execution_id)["delivery_outcome"] == "uncertain"
    saved = jobs.get_job(job["id"])
    assert "chat not found" not in json.dumps(saved), "the queue row's text reached jobs.json"
    if mixed:
        assert (saved["last_status"], saved["last_delivery_error"]) == (
            "delivery_failed", delivery_queue.UNCERTAIN_SEND_ERROR), saved
    else:
        assert (saved["last_status"], saved["last_delivery_error"]) == ("ok", None), saved


def test_worker_crash_notice_that_began_unconfirmed_is_recorded_uncertain(
        stalled_live_send, monkeypatch):
    calls = stalled_live_send

    def crash(job, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(scheduler, "run_job", crash)
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")
    execution_id = _in_restart_safe_worker(calls, monkeypatch, job)

    assert scheduler.run_one_job(job) is False

    _assert_uncertain_not_delivered(job["id"], calls)
    assert executions.get_execution(execution_id)["delivery_outcome"] == "uncertain"
    assert jobs.get_job(job["id"])["last_delivery_error"] is None


def test_worker_wait_timeout_mid_send_is_uncertain_with_fixed_text(stalled_live_send, monkeypatch):
    """The worker gave up while a gateway was mid-send (the row is fenced ``unknown`` and never
    resent): uncertain, and the unverified entry the worker writes is fixed text, not the row's."""
    calls = stalled_live_send
    job = jobs.create_job(prompt="fixture only", schedule="every 1h", deliver="telegram:123")
    execution = executions.create_execution(job["id"], source="fixture")
    job["execution_id"] = execution["id"]
    monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", execution["id"])
    real_wait = delivery_queue.enqueue_and_wait

    def claimed_then_timed_out(execution_id, queued_job, content, *, for_failure=False):
        delivery_queue.enqueue(execution_id, queued_job, content, for_failure=for_failure)
        assert delivery_queue.claim_next()["execution_id"] == execution_id  # a gateway began it
        delivery_queue._ACTIVE_DELIVERIES.discard(execution_id)
        return real_wait(execution_id, queued_job, content, for_failure=for_failure, timeout=0)

    monkeypatch.setattr(delivery_queue, "enqueue_and_wait", claimed_then_timed_out)

    assert scheduler._deliver_result(job, "the brief") is None

    assert delivery_queue.get_status(execution["id"])["status"] == "unknown"
    assert job["_delivery_uncertain"] == [{"queue": execution["id"]}]
    assert jobs.get_job(job["id"])["last_delivery_unverified"] == [
        f"queued delivery {execution['id']} (send began, not confirmed; not resent)"]


def test_interrupted_run_notice_drained_later_records_only_the_fixed_text(
        stalled_live_send, monkeypatch):
    """Shutdown booked the run without its execution id; the drain that settles the queued notice
    later still writes the job's delivery error, and only the fixed uncertain text."""
    calls = stalled_live_send
    _one_target_refuses(calls, monkeypatch, PROVIDER_TEXT)
    monkeypatch.setattr(
        scheduler, "run_job", lambda job, **kwargs: (True, "output", "the brief", None))
    job = jobs.create_job(
        prompt="fixture only", schedule="every 1h", deliver="telegram:123,telegram:456")
    before = jobs.get_job(job["id"])
    scheduler._interrupted_job_ids.add(scheduler._inflight_key(job["id"]))

    execution, queued = _run_then_drain(calls, monkeypatch, job)

    assert queued["status"] == "unknown" and "chat not found" in queued["error"], queued
    assert execution["delivery_outcome"] == "uncertain", execution
    saved = jobs.get_job(job["id"])
    assert saved["last_delivery_error"] == delivery_queue.UNCERTAIN_SEND_ERROR, saved
    assert "chat not found" not in json.dumps(saved), saved
    assert saved["repeat"] == before["repeat"] and saved["next_run_at"] == before["next_run_at"]
