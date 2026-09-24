"""Legacy snapshot drift skips alert once per drift episode, not once per tick (#44585 + #73506).

Field report: a fleet-wide config change moved the global default provider and every unpinned
snapshot-era cron started alerting on every tick -- 40 jobs x N ticks of identical "Skipped to
prevent unintended spend" spam. The drift guard correctly fails closed; the alert reuses the
#73506 alert-once shape (persisted per-job bit, cleared when the condition heals), as
pre-dispatch preflight already does for blocked_config.

Contract:
- First drifted tick delivers ONE loud, actionable alert; later drifted ticks deliver nothing.
- The bit is consumed only once a delivery target accepted the alert: a failed, local-only,
  unresolved or suppressed delivery, or a worker that lost its fire claim, leaves it armed.
- Any successful run, or a tick where the drift healed, re-arms it for the next episode.
- The alert travels out-of-band; failure text that merely quotes a marker is an ordinary failure.
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.jobs as cron_jobs
import cron.scheduler as sched
import cron.scheduler_delivery as delivery


def _job(**overrides):
    job = {
        "id": "drift-once-test",
        "name": "drift once test",
        "prompt": "hello",
        "enabled": True,
        "state": "scheduled",
        "schedule": {"kind": "interval", "minutes": 5, "display": "every 5m"},
        "deliver": "local",
        "model": None,
        "provider": None,
        "provider_snapshot": "openrouter",
        "base_url": None,
    }
    job.update(overrides)
    return job


def _tick(
    job, tmp_path, current_provider, deliveries, *, deliver_side_effect=None,
    before_delivery_fence=None,
):
    """Run one run_one_job tick with the provider resolution pinned; every send is accepted."""
    fake_db = MagicMock()

    def fake_deliver(job, content, adapters=None, loop=None, **kwargs):
        if deliver_side_effect is not None:
            deliver_side_effect(job, content)
        deliveries.append(content)
        job["_delivery_accepted"] = True
        return None

    real_fence = sched.fire_claim_fence
    fence_entries = 0

    @contextmanager
    def fenced_delivery(job_id, *, expected_owner):
        nonlocal fence_entries
        fence_entries += 1
        # _save_compose_deliver fences output first and delivery second.  Transfer
        # exactly after the pre-delivery ownership validation, before the actual
        # delivery fence decides whether this worker may send.
        if fence_entries == 2 and before_delivery_fence is not None:
            before_delivery_fence(job_id, expected_owner)
        with real_fence(job_id, expected_owner=expected_owner) as owns:
            yield owns

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={
                   "api_key": "test-key",
                   "base_url": "https://example.invalid/v1",
                   "provider": current_provider,
                   "api_mode": "chat_completions",
               }), \
         patch.object(sched, "fire_claim_fence", fenced_delivery), \
         patch.object(sched, "_deliver_result", side_effect=fake_deliver), \
         patch("run_agent.AIAgent") as mock_agent_cls:
        mock_agent = MagicMock()
        mock_agent.run_conversation.return_value = {"final_response": "ok"}
        mock_agent_cls.return_value = mock_agent
        ok = sched.run_one_job(job)
    return ok, mock_agent_cls.called


def _replace_fire_claim(job_id, owner):
    def replace(jobs, _i, job):
        job["fire_claim"] = {"by": owner, "at": "2026-01-01T00:00:01+00:00"}
        cron_jobs.save_jobs(jobs)

    cron_jobs._with_job(job_id, replace)
    return cron_jobs.load_jobs()[0]


class TestDriftAlertOnce:
    def test_two_drifted_ticks_alert_exactly_once(self, tmp_path):
        job = _job()
        deliveries = []
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            for _ in range(2):
                fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
                ok, agent_called = _tick(fresh, tmp_path, "nous", deliveries)
                assert agent_called is False, "drifted tick must not spend"

            stored = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            assert stored.get("drift_alerted") is True

        assert len(deliveries) == 1, f"expected 1 alert, got {len(deliveries)}: {deliveries}"
        blob = deliveries[0].lower()
        assert "drift" in blob
        assert "pin" in blob
        assert "host running hermes" in blob
        # The single alert must carry the complete supported remediation
        # command — the generic summarizer's 180-char truncation must not eat it.
        assert "hermes cron edit drift-once-test" in deliveries[0]
        assert "cronjob action=update" not in deliveries[0]
        assert "[drift_skip" not in deliveries[0]

    def test_stale_scheduled_owner_cannot_consume_replacement_drift_alert(self, tmp_path):
        """A stale scheduled worker must not silence its replacement's drift alert.

        The old worker validates its lease, then loses it before the drift guard
        persists the alert bit.  Its stale write must compare the persisted owner
        atomically, leaving the replacement to send the one alert.
        """
        stale_owner = "stale-owner"
        replacement_owner = "replacement-owner"
        stale = _job(fire_claim={"by": stale_owner, "at": "2026-01-01T00:00:00+00:00"})
        deliveries = []

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([stale])

            replaced = False

            def replace_after_validation(job_id, *, expected_owner):
                nonlocal replaced
                if replaced:
                    return False
                # Let the stale worker validate first, then deterministically model
                # the TTL recovery handing the persisted claim to a replacement.
                assert cron_jobs.heartbeat_fire_claim(
                    job_id, expected_owner=expected_owner)
                current = cron_jobs.load_jobs()[0]
                current["fire_claim"] = {
                    "by": replacement_owner, "at": "2026-01-01T00:00:01+00:00"}
                cron_jobs.save_jobs([current])
                replaced = True
                return True

            with patch.object(sched, "heartbeat_fire_claim", side_effect=replace_after_validation):
                ok, agent_called = _tick(stale, tmp_path, "nous", deliveries)
            assert ok is True
            assert agent_called is False
            assert not cron_jobs.load_jobs()[0].get("drift_alerted")

            replacement = cron_jobs.load_jobs()[0]
            ok, agent_called = _tick(replacement, tmp_path, "nous", deliveries)

        assert ok is True
        assert agent_called is False
        assert len(deliveries) == 1
        assert "drift" in deliveries[0].lower()

    def test_claim_transfer_before_scheduled_delivery_leaves_alert_for_replacement(self, tmp_path):
        """A loses the claim after CAS; B owns the one alert rather than both workers sending none."""
        owner_a, owner_b = "drift-owner-a", "drift-owner-b"
        job = _job(fire_claim={"by": owner_a, "at": "2026-01-01T00:00:00+00:00"})
        deliveries = []

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            ok, agent_called = _tick(
                job, tmp_path, "nous", deliveries,
                before_delivery_fence=lambda job_id, owner: (
                    owner == owner_a and _replace_fire_claim(job_id, owner_b)),
            )
            assert ok is True
            assert agent_called is False

            replacement = cron_jobs.load_jobs()[0]
            ok, agent_called = _tick(replacement, tmp_path, "nous", deliveries)

        assert ok is True
        assert agent_called is False
        assert len(deliveries) == 1
        assert "drift" in deliveries[0].lower()

    def test_successful_scheduled_drift_delivery_suppresses_replacement(self, tmp_path):
        """Once A actually sends, B must observe the committed alert bit and stay quiet."""
        owner_a, owner_b = "drift-owner-a", "drift-owner-b"
        job = _job(fire_claim={"by": owner_a, "at": "2026-01-01T00:00:00+00:00"})
        deliveries = []

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            ok, agent_called = _tick(job, tmp_path, "nous", deliveries)
            assert ok is True
            assert agent_called is False
            replacement = _replace_fire_claim(job["id"], owner_b)
            ok, agent_called = _tick(replacement, tmp_path, "nous", deliveries)

        assert ok is True
        assert agent_called is False
        assert len(deliveries) == 1

    def test_failed_scheduled_drift_delivery_does_not_suppress_replacement(self, tmp_path):
        """A delivery exception does not consume the bit; B can deliver the alert once."""
        owner_a, owner_b = "drift-owner-a", "drift-owner-b"
        job = _job(fire_claim={"by": owner_a, "at": "2026-01-01T00:00:00+00:00"})
        deliveries = []

        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            ok, agent_called = _tick(
                job, tmp_path, "nous", deliveries,
                deliver_side_effect=lambda *_: (_ for _ in ()).throw(RuntimeError("delivery failed")),
            )
            assert ok is True
            assert agent_called is False
            replacement = _replace_fire_claim(job["id"], owner_b)
            ok, agent_called = _tick(replacement, tmp_path, "nous", deliveries)

        assert ok is True
        assert agent_called is False
        assert len(deliveries) == 1
        assert "drift" in deliveries[0].lower()

    def test_healed_drift_clears_bit_and_redrift_realerts(self, tmp_path):
        job = _job()
        deliveries = []
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            # Tick 1: drifted -> one alert, bit set.
            fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            _tick(fresh, tmp_path, "nous", deliveries)
            assert len(deliveries) == 1

            # Tick 2: drift healed (resolution matches snapshot) -> runs, bit cleared.
            fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            ok, agent_called = _tick(fresh, tmp_path, "openrouter", deliveries)
            assert agent_called is True
            stored = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            assert not stored.get("drift_alerted")

            # Tick 3: drifts again -> re-alerts (not swallowed).
            fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            _tick(fresh, tmp_path, "nous", deliveries)

        drift_alerts = [d for d in deliveries if "drift" in d.lower()]
        assert len(drift_alerts) == 2, f"expected re-alert after heal: {deliveries}"

    def test_non_drift_failures_untouched_by_the_bit(self, tmp_path):
        """A job with the drift bit set whose run fails for another reason
        still alerts — only the drift branch consults the bit."""
        job = _job(provider_snapshot=None, drift_alerted=True)
        deliveries = []

        def fake_deliver(jb, content, adapters=None, loop=None, **kwargs):
            deliveries.append(content)
            jb["_delivery_accepted"] = True
            return None

        fake_db = MagicMock()
        with cron_jobs.use_cron_store(tmp_path):
            cron_jobs.save_jobs([job])
            fresh = [j for j in cron_jobs.load_jobs() if j["id"] == job["id"]][0]
            with patch("cron.scheduler._hermes_home", tmp_path), \
                 patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
                 patch("hermes_cli.env_loader.load_hermes_dotenv"), \
                 patch("hermes_cli.env_loader.reset_secret_source_cache"), \
                 patch("hermes_state_registry.acquire", return_value=fake_db), \
                 patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
                 patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                       return_value={
                           "api_key": "test-key",
                           "base_url": "https://example.invalid/v1",
                           "provider": "openrouter",
                           "api_mode": "chat_completions",
                       }), \
                 patch.object(sched, "_deliver_result", side_effect=fake_deliver), \
                 patch("run_agent.AIAgent") as mock_agent_cls:
                mock_agent = MagicMock()
                mock_agent.run_conversation.side_effect = RuntimeError("boom unrelated")
                mock_agent_cls.return_value = mock_agent
                sched.run_one_job(fresh)

        assert len(deliveries) == 1, "non-drift failure must still deliver"
        assert "failed:" in deliveries[0] and "drifted" not in deliveries[0], (
            "the run's failure notice, not the drift alert")


_ALERT_TEXT = "Nothing was charged"


class _Telegram:
    """Standalone Telegram sender stand-in: chats in ``failing`` answer like a deleted chat."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.sent = []

    async def send(self, platform, pconfig, chat_id, text, **kwargs):
        if chat_id in self.failing:
            return {"error": "Bad Request: chat not found"}
        self.sent.append((chat_id, text))
        return {"success": True, "message_id": f"m{len(self.sent)}"}

    def alerts(self):
        return [chat for chat, text in self.sent if _ALERT_TEXT in text]


@pytest.fixture
def telegram_home(tmp_path, monkeypatch):
    """Real store + real delivery path under a temp HERMES_HOME; only the wire send is faked."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: main-model\n  provider: nous\ncron:\n  preflight: false\n")
    config = GatewayConfig()
    config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
    monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
    telegram = _Telegram()
    monkeypatch.setattr("tools.send_message_tool._send_to_platform", telegram.send)
    return telegram


def _stored_job(schedule="every 1h", deliver="telegram:good", **fields):
    job = cron_jobs.create_job(prompt="fixture", schedule=schedule, deliver=deliver)

    def stamp(jobs, _i, record):
        record.update(fields)
        cron_jobs.save_jobs(jobs)

    cron_jobs._with_job(job["id"], stamp)
    return job["id"]


def _drain_right_after_enqueue(monkeypatch):
    """The live gateway drains the durable queue while the restart-safe worker waits on it."""
    import cron.delivery_queue as queue

    real_enqueue = queue.enqueue

    def enqueue_then_drain(*args, **kwargs):
        row = real_enqueue(*args, **kwargs)
        with patch.dict(os.environ):
            os.environ.pop("_HERMES_CRON_EXTERNAL_WORKER", None)
            sched.drain_delivery_queue(None, None)
        return row

    monkeypatch.setattr(queue, "enqueue", enqueue_then_drain)


def _real_tick(job_id, *, owner=None, response="ok", external=False):
    """One run_one_job over the stored record; returns whether an agent was built. ``external``
    runs it as the restart-safe worker that owns this execution (delivery via the durable queue)."""
    if owner is not None:
        _replace_fire_claim(job_id, owner)
    job = cron_jobs.get_job(job_id)
    worker_env = {}
    if external:
        from cron.executions import create_execution
        job["execution_id"] = create_execution(job_id, source="direct")["id"]
        worker_env["_HERMES_CRON_EXTERNAL_WORKER"] = job["execution_id"]
    with patch.dict(os.environ, worker_env), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=MagicMock()), \
         patch("tools.mcp_tool_discovery.discover_mcp_tools", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               side_effect=lambda **kw: {
                   "api_key": "test-key", "base_url": "https://example.invalid/v1",
                   "provider": kw.get("requested") or "nous", "api_mode": "chat_completions"}), \
         patch("run_agent.AIAgent") as agent_cls:
        agent_cls.return_value.run_conversation.return_value = {"final_response": response}
        assert sched.run_one_job(job) is True
    return agent_cls.called


class TestDriftAlertEpisode:
    def test_one_failing_target_does_not_repeat_the_alert_to_the_healthy_one(self, telegram_home):
        telegram_home.failing.add("bad")
        job_id = _stored_job(deliver="telegram:good,telegram:bad", provider_snapshot="openrouter")

        for tick in range(3):
            assert _real_tick(job_id, owner=f"owner-{tick}") is False

        assert telegram_home.alerts() == ["good"]

    def test_durable_queue_delivery_keeps_one_alert_per_episode(self, telegram_home, monkeypatch):
        """Restart-safe workers hand the send to the gateway; its per-target outcome must reach
        the alert-once commit instead of one joined error that reads as "nothing was sent"."""
        _drain_right_after_enqueue(monkeypatch)
        telegram_home.failing.add("bad")
        job_id = _stored_job(deliver="telegram:good,telegram:bad", provider_snapshot="openrouter")

        for tick in range(3):
            assert _real_tick(job_id, owner=f"owner-{tick}", external=True) is False

        assert telegram_home.alerts() == ["good"]

    def test_queued_alert_that_no_target_accepted_is_sent_after_heal(
        self, telegram_home, monkeypatch,
    ):
        """No gateway within the wait budget leaves the alert queued, not sent: when the drain then
        reaches no target, the episode's one alert must still go out once the target heals."""
        import cron.delivery_queue as queue
        monkeypatch.setattr(queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0)
        job_id = _stored_job(provider_snapshot="openrouter")
        assert _real_tick(job_id, owner="owner-0", external=True) is False
        telegram_home.failing.add("good")
        sched.drain_delivery_queue(None, None)
        telegram_home.failing.clear()
        _drain_right_after_enqueue(monkeypatch)

        for tick in range(1, 4):
            assert _real_tick(job_id, owner=f"owner-{tick}", external=True) is False

        assert telegram_home.alerts() == ["good"]

    def test_queued_alert_with_unknown_send_outcome_is_rearmed(self, telegram_home, monkeypatch):
        """A gateway that dies mid-send leaves the row ``unknown``: nothing attests a target took
        the alert, so it counts as not sent."""
        import cron.delivery_queue as queue
        monkeypatch.setattr(queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0)
        job_id = _stored_job(provider_snapshot="openrouter")
        assert _real_tick(job_id, owner="owner-0", external=True) is False
        claimed = queue.claim_next()
        queue._ACTIVE_DELIVERIES.discard(claimed["execution_id"])  # its gateway is gone
        assert queue.recover_abandoned() == 1
        _drain_right_after_enqueue(monkeypatch)

        for tick in range(1, 3):
            assert _real_tick(job_id, owner=f"owner-{tick}", external=True) is False

        assert telegram_home.alerts() == ["good"]

    def test_alert_waiting_in_the_queue_is_not_resent_each_tick(self, telegram_home, monkeypatch):
        import cron.delivery_queue as queue
        monkeypatch.setattr(queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0)
        job_id = _stored_job(provider_snapshot="openrouter")

        for tick in range(3):
            assert _real_tick(job_id, owner=f"owner-{tick}", external=True) is False
        sched.drain_delivery_queue(None, None)

        assert telegram_home.alerts() == ["good"]

    def test_open_bot_chat_receipt_is_settled_through_its_lane(
        self, telegram_home, tmp_path, monkeypatch,
    ):
        """A queued Bot Chat receipt is admission, not delivery: while it is open the alert waits
        (a re-send is a new, paid turn), a terminal failure re-arms it once, settled consumes it."""
        from tools import bot_live_delivery as mailbox
        owner = dict(profile_home=str(tmp_path.resolve()), session_id="bot", lease_id="lease",
                     live_session_id="live")
        monkeypatch.setattr(mailbox, "find_canonical_live_owner", lambda home: owner)
        job_id = _stored_job(deliver="bot-chat", provider_snapshot="openrouter")

        def turns_after(ticks, finish=None):
            if finish is not None:
                claimed = mailbox.claim_pending_delivery(tmp_path, owner)
                mailbox.complete_delivery(tmp_path, claimed["delivery_id"], status=finish)
            for tick in ticks:
                assert _real_tick(job_id, owner=f"owner-{tick}") is False
            return len(list((tmp_path / "runtime/bot_live_delivery").glob("*.json")))

        assert turns_after(range(2)) == 1
        assert turns_after(range(2, 4), finish="failed") == 2
        assert turns_after(range(4, 6), finish="settled") == 2

    def test_healed_drift_drops_an_alert_still_parked_in_the_queue(
        self, telegram_home, monkeypatch,
    ):
        import cron.delivery_queue as queue
        monkeypatch.setattr(queue, "DEFAULT_DELIVERY_WAIT_TIMEOUT_SECONDS", 0)
        job_id = _stored_job(provider_snapshot="openrouter")
        assert _real_tick(job_id, owner="owner-0", external=True) is False
        cron_jobs.update_job(job_id, {"provider": "nous"})

        # The healed run fails, so only the heal (not a healthy run) can drop the parked alert.
        assert _real_tick(job_id, owner="owner-1", response="[CRON_FAILURE]\nboom") is True

        stored = cron_jobs.get_job(job_id)
        assert not stored.get("drift_alerted") and not stored.get("drift_alert_parked")

    def test_successful_run_rearms_the_alert_for_the_next_drift_episode(self, telegram_home):
        job_id = _stored_job(provider_snapshot="openrouter")
        assert _real_tick(job_id) is False
        cron_jobs.update_job(job_id, {"provider": "nous", "model": "main-model"})
        assert _real_tick(job_id) is True
        cron_jobs.update_job(job_id, {"pinned": False})
        assert _real_tick(job_id) is False

        assert telegram_home.alerts() == ["good", "good"]

    def test_claimless_run_keeps_the_alert_until_a_target_accepts_it(self, telegram_home):
        telegram_home.failing.add("good")
        job_id = _stored_job(provider_snapshot="openrouter")
        assert _real_tick(job_id) is False
        telegram_home.failing.clear()
        assert _real_tick(job_id) is False

        assert telegram_home.alerts() == ["good"]

    @pytest.mark.parametrize("fields, extra_yaml", [
        ({"failure_deliver": "local"}, ""),
        ({"failure_deliver": "origin"}, ""),
        ({}, "display:\n  suppress_warning_notifications: true\n"),
    ], ids=["local-only", "unresolved-origin", "warning-suppressed"])
    def test_alert_that_reached_no_target_stays_armed(
        self, telegram_home, tmp_path, fields, extra_yaml,
    ):
        config = tmp_path / "config.yaml"
        config.write_text(config.read_text() + extra_yaml)
        job_id = _stored_job(provider_snapshot="openrouter", **fields)

        assert _real_tick(job_id, owner="owner-0") is False

        assert telegram_home.sent == []
        assert not cron_jobs.get_job(job_id).get("drift_alerted")

    def test_agent_failure_quoting_drift_markers_is_an_ordinary_failure(self, telegram_home):
        evidence = ("upstream log said [legacy_provider_drift:silent] and "
                    "[legacy_provider_drift] Provider drifted from 'a' to 'b'")
        job_id = _stored_job()

        assert _real_tick(job_id, response=f"[CRON_FAILURE]\n{evidence}") is True

        assert len(telegram_home.sent) == 1
        assert evidence in telegram_home.sent[0][1]
        assert telegram_home.alerts() == []

    def test_one_shot_alert_never_points_at_editing_the_consumed_job(self, telegram_home):
        job_id = _stored_job(schedule="2030-01-01T09:00:00Z", provider_snapshot="openrouter")
        assert _real_tick(job_id) is False

        [(_chat, text)] = telegram_home.sent
        assert _ALERT_TEXT in text
        assert "hermes cron edit" not in text
        assert "hermes cron create" in text


def test_drift_alert_state_is_written_only_by_the_fire_claim_owner(telegram_home):
    job_id = _stored_job(provider_snapshot="openrouter")
    _replace_fire_claim(job_id, "owner-a")

    assert cron_jobs.set_drift_alert(job_id, True, expected_fire_owner="owner-b") is False
    assert not cron_jobs.get_job(job_id).get("drift_alerted")
    assert cron_jobs.set_drift_alert(job_id, True, expected_fire_owner="owner-a") is True
    assert cron_jobs.get_job(job_id).get("drift_alerted") is True


def test_unprepared_targets_do_not_count_as_accepted():
    """A target list alone cannot attest that a send was attempted."""
    job = {"id": "no-prepared-target", "name": "no-prepared-target", "deliver": "telegram:123"}
    target = {"platform": "telegram", "chat_id": "123"}

    with patch.object(delivery, "_resolve_delivery_targets", return_value=[target]), \
         patch.object(delivery._sched, "load_config", return_value={"cron": {"wrap_response": False}}), \
         patch("gateway.config.load_gateway_config", return_value={}), \
         patch("gateway.media_policy.apply_media_policy_env"), \
         patch.object(delivery, "_prepare_target_delivery", return_value=None) as prepare, \
         patch.object(delivery, "_deliver_via_live_adapter") as live_send, \
         patch.object(delivery, "_deliver_standalone") as standalone_send:
        delivery._deliver_result(job, "payload")

    assert prepare.call_count == 1
    live_send.assert_not_called()
    standalone_send.assert_not_called()
    assert not job.get("_delivery_accepted")
