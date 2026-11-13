"""Regression tests for #82232.

When the shutdown interrupts an in-flight cron job, the job's own worker
thread tries to deliver an "interrupted" notice — and loses, because
``_bounded_adapter_teardown`` has already closed the transport by the time it
gets there. The notice is dropped, and ``_consume_interrupted_flag`` discards
the resulting ``delivery_error`` with it, so the run's only trace is a line in
jobs.json.

The gateway now sends that notice itself in the post-interrupt phase, while
adapters are still connected — the same window
``_notify_active_sessions_of_shutdown`` uses for chat sessions, which never
saw cron work because cron runs outside ``_running_agents`` (#60432).
"""

from unittest.mock import patch

import pytest

from gateway.config import Platform
from tests.gateway.restart_test_helpers import make_restart_runner
from tools import browser_session_lifecycle as bt_lifecycle

@pytest.fixture(autouse=True)
def _reset_cron_running_set():
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._interrupted_job_ids.clear()

def _telegram_job(job_id="be62d36a9914", name="daily-digest", chat_id="123456"):
    return {
        "id": job_id,
        "name": name,
        "deliver": f"telegram:{chat_id}",
    }

def _telegram_target(chat_id="123456"):
    return {"platform": "telegram", "chat_id": chat_id, "thread_id": None}

def _bind_notifier(runner):
    from gateway.run import GatewayRunner

    runner._notify_interrupted_cron_jobs = (
        GatewayRunner._notify_interrupted_cron_jobs.__get__(runner, GatewayRunner)
    )
    runner._thread_metadata_for_target = (
        GatewayRunner._thread_metadata_for_target.__get__(runner, GatewayRunner)
    )
    return runner

class TestNotifyInterruptedCronJobs:
    @pytest.mark.asyncio
    async def test_owner_is_told_the_run_was_killed(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 1
        assert len(adapter.sent) == 1
        body = adapter.sent[0]
        assert "daily-digest" in body
        assert adapter.sent_calls[0][0] == "123456"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("setting", [None, False, True])
    async def test_interrupt_notice_is_a_suppressible_diagnostic(self, tmp_path, monkeypatch, setting):
        import json
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
        cfg = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
        (tmp_path / "config.yaml").write_text(json.dumps(cfg))
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()
        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])
        expected = 0 if setting is True else 1
        assert sent == expected
        assert len(adapter.sent) == expected

    @pytest.mark.asyncio
    async def test_local_only_job_stays_silent(self):
        """deliver=local, and deliver=origin with no resolvable origin
        (#43014), resolve to zero targets and must not fall back to a home
        channel."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = {"id": "j1", "name": "local-job", "deliver": "local"}

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets", return_value=[]):
            sent = await runner._notify_interrupted_cron_jobs(["j1"])

        assert sent == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_respects_platform_gateway_restart_notification_false(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        runner.config.platforms[Platform.TELEGRAM].gateway_restart_notification = False
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_failure_deliver_local_suppresses_interrupt_notice(self):
        """Interrupted notices are failure-category engine status (NS-788):
        a job with failure_deliver='local' opted out of failure pings, and
        the shutdown notice must honor that. Real target resolution — no
        _resolve_delivery_targets patch — so the failure_deliver override is
        actually exercised."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = dict(_telegram_job(), failure_deliver="local")

        with patch("cron.jobs.get_job", return_value=job):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_failure_deliver_unset_notice_reaches_deliver_target(self):
        """Control for the suppress test: same job without failure_deliver,
        same real resolution path — the notice goes to the deliver target."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 1
        assert adapter.sent_calls[0][0] == "123456"

    @pytest.mark.asyncio
    async def test_empty_job_list_is_a_noop(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)

        assert await runner._notify_interrupted_cron_jobs([]) == 0
        assert adapter.sent == []

    @pytest.mark.asyncio
    async def test_a_raising_adapter_cannot_block_shutdown(self):
        """Best-effort by construction: a wedged adapter must not propagate."""
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        async def _boom(*_a, **_kw):
            raise RuntimeError("transport already closed")

        adapter.send = _boom

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 0

    @pytest.mark.asyncio
    async def test_duplicate_targets_send_once_per_job(self):
        runner, adapter = make_restart_runner()
        _bind_notifier(runner)
        job = _telegram_job()

        with patch("cron.jobs.get_job", return_value=job), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target(), _telegram_target()]):
            sent = await runner._notify_interrupted_cron_jobs([job["id"]])

        assert sent == 1
        assert len(adapter.sent) == 1

class TestShutdownDeliversNoticeBeforeDisconnect:
    @pytest.mark.asyncio
    async def test_notice_is_sent_while_the_adapter_is_still_connected(self, monkeypatch):
        """The whole point is ordering: a notice sent after teardown is lost,
        which is the bug."""
        import cron.scheduler as sched
        import tools.process_registry as _pr
        import tools.terminal_tool as _tt

        runner, adapter = make_restart_runner()
        runner._restart_drain_timeout = 0.01  # force the interrupt path
        runner._cron_drain_timeout = 0.01  # don't wait out the 30s cron drain budget
        sched._running_job_ids.add("be62d36a9914")

        monkeypatch.setattr(_pr.process_registry, "kill_all", lambda task_id=None: 1)
        monkeypatch.setattr(_tt, "cleanup_all_environments", lambda: None)
        monkeypatch.setattr(bt_lifecycle, "cleanup_all_browsers", lambda: None)

        events: list[str] = []
        real_send = adapter.send

        async def _tracking_send(chat_id, content, reply_to=None, metadata=None):
            if "was cut short" in content:
                events.append("cron_notice")
            return await real_send(chat_id, content, reply_to=reply_to, metadata=metadata)

        async def _tracking_disconnect():
            events.append("disconnect")

        adapter.send = _tracking_send
        adapter.disconnect = _tracking_disconnect

        with patch("gateway.status.remove_pid_file"), \
             patch("gateway.status.publish_runtime_status"), \
             patch("cron.scheduler.mark_job_run"), \
             patch("cron.jobs.get_job", return_value=_telegram_job()), \
             patch("cron.scheduler._resolve_delivery_targets",
                   return_value=[_telegram_target()]):
            await runner.stop()

        assert "cron_notice" in events, "interrupted-cron notice was never sent"
        assert "disconnect" in events
        assert events.index("cron_notice") < events.index("disconnect"), (
            f"notice sent after adapter teardown — it would be lost: {events}"
        )


class TestDeliveryErrorIsRecordedWhenTheNoticeCannotBeSent:
    """``_consume_interrupted_flag`` skips ``finish_completed_run``: shutdown already wrote this
    run's ``last_status``, and a second ``mark_job_run`` would skip a fire or retire the job early.
    The interrupted tail still records what happened to the notice."""

    def _job(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from cron.executions import create_execution
        from cron.jobs import create_job, get_job

        job = create_job(prompt="x", schedule="every 5m", name="interrupted")
        return get_job(job["id"]), create_execution(job["id"], source="direct")

    def test_interrupted_run_records_delivery_error_without_mark_job_run(self, tmp_path, monkeypatch):
        from cron.jobs import get_job
        from cron.scheduler_bookkeeping import finish_interrupted_run

        before, execution = self._job(tmp_path, monkeypatch)

        finish_interrupted_run(before, execution["id"], "telegram: 502 Bad Gateway")

        after = get_job(before["id"])
        # Raw until the closed public copy lands (E3-L2).
        assert after["last_delivery_error"] == "telegram: 502 Bad Gateway"
        assert after["repeat"] == before["repeat"] and after["next_run_at"] == before["next_run_at"]

    def test_interrupted_uncertain_send_records_no_provider_text(self, tmp_path, monkeypatch):
        from cron import delivery_queue
        from cron.executions import get_execution
        from cron.jobs import get_job
        from cron.scheduler_bookkeeping import finish_interrupted_run

        before, execution = self._job(tmp_path, monkeypatch)
        job = dict(before, _delivery_uncertain=["telegram:123"])

        finish_interrupted_run(job, execution["id"], 'telegram:456: Bad Request: chat not found {"ok":false}')

        after = get_job(before["id"])
        assert "chat not found" not in str(after["last_delivery_error"]), after
        assert after["last_delivery_error"] == delivery_queue.UNCERTAIN_SEND_ERROR
        assert get_execution(execution["id"])["delivery_outcome"] == "uncertain"
        assert after["repeat"] == before["repeat"] and after["next_run_at"] == before["next_run_at"]


class TestInterruptedRunKeepsItsQueuedNoticeOutcome:
    """An interrupted agent run hands its failure notice to the delivery queue like any run, but
    only ``finish_completed_run`` settled a queued send, so the notice's outcome was lost."""

    def _interrupted_run(self, tmp_path, monkeypatch, send_result, *, drain_first):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from cron import delivery_queue, executions, jobs, scheduler
        from gateway.config import GatewayConfig, PlatformConfig

        config = GatewayConfig()
        config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)

        async def send(platform, pconfig, chat_id, text, **kwargs):
            return send_result

        monkeypatch.setattr("tools.send_message_tool._send_to_platform", send)
        monkeypatch.setattr(scheduler, "run_job", lambda job, **kw: (True, "raw", "the result", None))
        if drain_first:
            real_enqueue = delivery_queue.enqueue

            def enqueue_then_drain(*args, **kwargs):
                queued = real_enqueue(*args, **kwargs)
                assert scheduler.drain_delivery_queue({}, None) == 1
                return queued

            monkeypatch.setattr(delivery_queue, "enqueue", enqueue_then_drain)
        marks = []
        real_mark = scheduler.mark_job_run
        monkeypatch.setattr(
            scheduler, "mark_job_run", lambda *a, **k: marks.append(a) or real_mark(*a, **k))
        job = jobs.create_job(prompt="p", schedule="every 1h", deliver="telegram:fixture")
        before = jobs.get_job(job["id"])
        scheduler._interrupted_job_ids.add(scheduler._inflight_key(job["id"]))

        assert scheduler.run_one_job(job) is True

        assert marks == [], "shutdown already advanced this run"
        after = jobs.get_job(job["id"])
        assert after["repeat"] == before["repeat"] and after["next_run_at"] == before["next_run_at"]
        return after, executions.latest_execution(job["id"])

    def test_refused_notice_drained_before_the_tail_is_recorded(self, tmp_path, monkeypatch):
        after, execution = self._interrupted_run(
            tmp_path, monkeypatch, {"success": False, "error": "transport refused"}, drain_first=True)

        assert "transport refused" in (after["last_delivery_error"] or ""), after
        assert execution["delivery_outcome"] == "failed", execution

    def test_notice_still_queued_settles_the_ledger_when_drained(self, tmp_path, monkeypatch):
        from cron import executions, scheduler

        _after, execution = self._interrupted_run(
            tmp_path, monkeypatch, {"success": True, "message_id": "r"}, drain_first=False)

        assert execution["delivery_outcome"] == "queued", execution
        assert scheduler.drain_delivery_queue({}, None) == 1
        assert executions.get_execution(execution["id"])["delivery_outcome"] == "delivered"
