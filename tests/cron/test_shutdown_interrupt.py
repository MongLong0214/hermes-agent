"""Tests for #60432: cron jobs must not be silently invisible to gateway
shutdown, and a job whose tool subprocess got killed by shutdown must
never be reported as a successful run.

Covers the cron/scheduler.py primitives directly:
  - get_running_job_ids() -- thread-safe snapshot the gateway drain reads
  - mark_running_jobs_interrupted() -- called by the gateway right after
    it force-kills tool subprocesses
  - the interrupted-flag race guard in run_one_job(), which must win over
    the job's own thread finishing normally with a plausible-looking
    result AFTER its tool was already killed out from under it
"""

import asyncio
import concurrent.futures
import logging
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_scheduler_state():
    """Every test starts from a clean slate and leaves one behind, since
    these sets are module-level globals shared across the test process."""
    import cron.scheduler as sched

    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    sched._interrupted_job_ids.clear()


class TestGetRunningJobIds:
    def test_empty_when_nothing_running(self):
        import cron.scheduler as sched

        assert sched.get_running_job_ids() == frozenset()

    def test_reflects_in_flight_jobs(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        sched._running_job_ids.add("job-2")

        result = sched.get_running_job_ids()

        assert result == frozenset({"job-1", "job-2"})

    def test_snapshot_is_immutable_and_independent(self):
        """Mutating _running_job_ids after the call must not change the
        already-returned snapshot -- callers (the gateway drain loop) rely
        on this to safely count in a tight polling loop."""
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")
        snapshot = sched.get_running_job_ids()
        sched._running_job_ids.add("job-2")

        assert snapshot == frozenset({"job-1"})


class TestMarkRunningJobsInterrupted:
    def test_no_op_when_nothing_running(self):
        import cron.scheduler as sched

        with patch("cron.scheduler.mark_job_run") as mock_mark:
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == []
        mock_mark.assert_not_called()

    def test_marks_every_in_flight_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})
        profile_home = sched._get_hermes_home().resolve()
        sched._running_fire_owners.update(
            {
                "job-1": {object(): ("owner-1", profile_home)},
                "job-2": {object(): ("owner-2", profile_home)},
            }
        )

        with patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark:
            marked = sched.mark_running_jobs_interrupted("gateway shutdown (final-cleanup)")

        assert sorted(marked) == ["job-1", "job-2"]
        assert mock_mark.call_count == 2
        called_ids = {c.args[0] for c in mock_mark.call_args_list}
        assert called_ids == {"job-1", "job-2"}
        for c in mock_mark.call_args_list:
            # success must be False -- an interrupted run is never "ok".
            assert c.args[1] is False
            assert "gateway shutdown" in c.args[2]
            assert c.kwargs["expected_fire_owner"] in {"owner-1", "owner-2"}

    def test_sets_interrupted_flag_for_consumption_by_run_one_job(self):
        import cron.scheduler as sched

        sched._running_job_ids.add("job-1")

        with patch("cron.scheduler.mark_job_run"):
            sched.mark_running_jobs_interrupted("shutdown")

        assert "job-1" in sched._interrupted_job_ids

    def test_one_job_marking_failure_does_not_block_the_others(self):
        """mark_job_run raising for one job (e.g. a jobs.json write race)
        must not prevent the rest from being marked -- this runs during
        shutdown, there's no retry window."""
        import cron.scheduler as sched

        sched._running_job_ids.update({"job-1", "job-2"})
        profile_home = sched._get_hermes_home().resolve()
        sched._running_fire_owners.update(
            {
                "job-1": {object(): ("owner-1", profile_home)},
                "job-2": {object(): ("owner-2", profile_home)},
            }
        )

        def _side_effect(job_id, success, reason, **kwargs):
            if job_id == "job-1":
                raise OSError("disk full")
            return True

        with patch("cron.scheduler.mark_job_run", side_effect=_side_effect):
            marked = sched.mark_running_jobs_interrupted("shutdown")

        assert marked == ["job-2"]

    def test_stale_shutdown_cannot_clear_replacement_owner(self, tmp_path):
        import cron.jobs as jobs
        import cron.scheduler as sched

        profile_home = tmp_path / "profile"
        profile_home.mkdir()
        with jobs.use_cron_store(profile_home):
            created = jobs.create_job(prompt="x", schedule="every 5m", name="owned")
            claimed = jobs.claim_job_for_fire(created["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            stale_owner = claimed["fire_claim"]["by"]
            original_status = claimed["last_status"]
            replacement_claim = {
                "at": "2026-07-12T12:30:00+00:00",
                "by": "replacement-owner",
            }
            replacement = {**claimed, "fire_claim": replacement_claim}
            jobs.save_jobs([replacement])

            sched._running_job_ids.add(created["id"])
            sched._running_fire_owners[created["id"]] = {
                object(): (stale_owner, profile_home)
            }
            marked = sched.mark_running_jobs_interrupted("shutdown")
            refreshed = jobs.get_job(created["id"])

        assert marked == []
        assert isinstance(refreshed, dict)
        assert refreshed["fire_claim"] == replacement_claim
        assert refreshed["last_status"] == original_status


class TestRunningFireOwnerRegistry:
    def test_run_one_job_registers_owner_only_while_active(self):
        import cron.scheduler as sched

        job = {
            "id": "owned-job",
            "fire_claim": {"at": "2026-07-12T12:00:00+00:00", "by": "owner-1"},
        }

        def _observe_registry(current_job, run):
            assert list(sched._running_fire_owners[current_job["id"]].values()) == [
                ("owner-1", sched._get_hermes_home().resolve())
            ]
            return True

        with patch("cron.scheduler._run_with_fire_claim_heartbeat", side_effect=_observe_registry):
            assert sched.run_one_job(job) is True

        assert job["id"] not in sched._running_fire_owners

    def test_shutdown_sees_all_concurrent_direct_fire_owners(self, monkeypatch):
        """Direct entry points and replacement owners share one token registry."""
        import cron.scheduler as sched

        entered = threading.Barrier(3)
        release = threading.Event()
        marked_owners: list[str] = []

        def hold_run(_job, _run):
            entered.wait(timeout=2)
            release.wait(timeout=2)
            return True

        def mark(_job_id, _success, _reason, *, expected_fire_owner):
            marked_owners.append(expected_fire_owner)
            return True

        monkeypatch.setattr(sched, "_run_with_fire_claim_heartbeat", hold_run)
        monkeypatch.setattr(sched, "mark_job_run", mark)

        jobs = [
            {"id": "same-job", "fire_claim": {"by": "old-owner"}},
            {"id": "same-job", "fire_claim": {"by": "replacement-owner"}},
        ]
        threads = [threading.Thread(target=sched.run_one_job, args=(job,)) for job in jobs]
        for thread in threads:
            thread.start()
        entered.wait(timeout=2)

        assert sched.get_running_job_ids() == frozenset({"same-job"})
        assert sched.mark_running_jobs_interrupted("shutdown") == ["same-job", "same-job"]
        assert set(marked_owners) == {"old-owner", "replacement-owner"}

        release.set()
        for thread in threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
        assert "same-job" not in sched.get_running_job_ids()

    def test_shutdown_marks_each_owner_in_its_profile_store(self, monkeypatch, tmp_path):
        import cron.jobs as cron_jobs
        import cron.scheduler as sched

        profile_a = tmp_path / "a"
        profile_b = tmp_path / "b"
        observed = []
        sched._running_fire_owners["same-job"] = {
            object(): ("owner-a", profile_a),
            object(): ("owner-b", profile_b),
        }

        def mark(job_id, success, reason, *, expected_fire_owner):
            observed.append(
                (
                    job_id,
                    success,
                    expected_fire_owner,
                    cron_jobs._current_cron_store().jobs_file,
                )
            )
            return True

        monkeypatch.setattr(sched, "mark_job_run", mark)

        assert sched.mark_running_jobs_interrupted("shutdown") == ["same-job", "same-job"]
        assert set(observed) == {
            ("same-job", False, "owner-a", profile_a / "cron" / "jobs.json"),
            ("same-job", False, "owner-b", profile_b / "cron" / "jobs.json"),
        }


class TestIsInterrupted:
    """Peek-only check used at the delivery gate -- must NOT clear the
    flag, unlike _consume_interrupted_flag."""

    def test_false_when_not_marked(self):
        import cron.scheduler as sched

        assert sched._is_interrupted("job-1") is False

    def test_true_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._is_interrupted("job-1") is True

    def test_does_not_clear_the_flag(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        sched._is_interrupted("job-1")

        # Still set -- the later, authoritative check before mark_job_run
        # must still see it.
        assert "job-1" in sched._interrupted_job_ids
        assert sched._is_interrupted("job-1") is True


class TestConsumeInterruptedFlag:

    def test_true_and_clears_when_marked(self):
        import cron.scheduler as sched

        sched._interrupted_job_ids.add("job-1")

        assert sched._consume_interrupted_flag("job-1") is True
        # Consumed -- a second check (e.g. a later, unrelated fire of the
        # same recurring job ID) must not still read as interrupted.
        assert sched._consume_interrupted_flag("job-1") is False


class TestExecutionScopedInterruption:
    """Interruption flags must target ONE execution, not the job ID.

    Owner-registered executions are recorded by their unique execution
    token, so a fresh run that reuses the same job ID (recurring fire,
    replacement claim owner) never consumes a flag that targeted its
    dead predecessor.
    """

    def test_interruption_targets_only_the_interrupted_execution(self):
        import cron.scheduler as sched

        profile_home = sched._get_hermes_home().resolve()
        old_token = object()
        sched._running_fire_owners["job-1"] = {
            old_token: ("owner-1", profile_home),
        }

        with patch("cron.scheduler.mark_job_run", return_value=True):
            sched.mark_running_jobs_interrupted("shutdown")

        assert sched._is_interrupted("job-1", old_token) is True
        new_token = object()
        assert sched._is_interrupted("job-1", new_token) is False
        # A new execution must not steal (and thereby clear) the old flag.
        assert sched._consume_interrupted_flag("job-1", new_token) is False
        assert sched._consume_interrupted_flag("job-1", old_token) is True
        assert sched._is_interrupted("job-1", old_token) is False

    def test_only_owners_marks_only_targeted_executions(self):
        import cron.scheduler as sched

        profile_home = sched._get_hermes_home().resolve()
        token_a, token_b = object(), object()
        sched._running_fire_owners["job-a"] = {token_a: ("owner-a", profile_home)}
        sched._running_fire_owners["job-b"] = {token_b: ("owner-b", profile_home)}

        with patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark:
            marked = sched.mark_running_jobs_interrupted(
                "dashboard shutdown",
                only_owners={("job-a", "owner-a")},
            )

        assert marked == ["job-a"]
        assert mock_mark.call_count == 1
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-a"
        assert sched._is_interrupted("job-a", token_a) is True
        assert sched._is_interrupted("job-b", token_b) is False

    def test_replacement_execution_of_same_job_is_not_poisoned(self):
        """A replacement owner starting while the stale flag exists must
        complete through the normal mark path, not the interrupted one."""
        import cron.scheduler as sched

        profile_home = sched._get_hermes_home().resolve()
        stale_token = object()
        sched._running_fire_owners["job-1"] = {
            stale_token: ("stale-owner", profile_home),
        }
        with patch("cron.scheduler.mark_job_run", return_value=True):
            sched.mark_running_jobs_interrupted("shutdown")
        sched._running_fire_owners.clear()

        job = {
            "id": "job-1",
            "name": "test job",
            "prompt": "do work",
            "fire_claim": {"by": "replacement-owner"},
        }
        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.fire_claim_fence"), \
             patch("cron.scheduler.heartbeat_fire_claim", return_value=True), \
             patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        mock_mark.assert_called_once()


class TestCombinedCancelEvent:
    def test_or_semantics(self):
        import cron.scheduler as sched

        a, b = threading.Event(), threading.Event()
        combined = sched._CombinedCancelEvent(a, b)
        assert combined.is_set() is False
        b.set()
        assert combined.is_set() is True

    def test_set_propagates_to_all(self):
        import cron.scheduler as sched

        a, b = threading.Event(), threading.Event()
        combined = sched._CombinedCancelEvent(a, b)
        combined.set()
        assert a.is_set() and b.is_set()

    def test_run_one_job_forwards_external_cancel_event(self):
        import cron.scheduler as sched

        external = threading.Event()
        job = {"id": "job-x", "name": "x", "prompt": "p"}

        with patch.object(
            sched,
            "_run_with_fire_claim_heartbeat",
            side_effect=lambda job_arg, run: run(threading.Event()),
        ), patch.object(sched, "_run_one_job_body", return_value=True) as body:
            assert sched.run_one_job(job, cancel_event=external) is True

        combined = body.call_args.kwargs["fire_claim_lost"]
        assert combined.is_set() is False
        external.set()
        assert combined.is_set() is True


class TestBaseExceptionThroughOwnerFencedFlow:
    """#73973 (sweeper review on #70638): a BaseException escaping run_job
    must still record a failed run through the owner-fenced terminal path —
    and a stale worker must not record over a replacement claim owner."""

    def _job(self):
        return {
            "id": "job-be",
            "name": "base exc",
            "prompt": "p",
            "fire_claim": {"by": "owner-be"},
        }

    def _patches(self, run_side_effect):
        return (
            patch("cron.scheduler.claim_dispatch", return_value=True),
            patch("agent.secret_scope.set_secret_scope", return_value=None),
            patch("agent.secret_scope.build_profile_secret_scope", return_value=None),
            patch("agent.secret_scope.reset_secret_scope"),
            patch("cron.scheduler.run_job", side_effect=run_side_effect),
            patch("cron.scheduler.heartbeat_fire_claim", return_value=True),
        )

    def test_cancelled_error_records_failure_and_reraises(self):
        import asyncio

        import cron.scheduler as sched

        p1, p2, p3, p4, p5, p6 = self._patches(asyncio.CancelledError())
        with p1, p2, p3, p4, p5, p6, \
             patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark, \
             patch("cron.scheduler.finish_execution") as mock_finish:
            try:
                sched.run_one_job(self._job())
                raised = False
            except asyncio.CancelledError:
                raised = True

        assert raised, "non-Exception BaseException must propagate"
        mock_mark.assert_called_once()
        assert mock_mark.call_args.args[:3] == (
            "job-be", False, sched._CRON_EXECUTION_FAILURE
        )
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-be"
        assert mock_finish.call_args.kwargs["success"] is False

    def test_keyboard_interrupt_records_failure_and_reraises(self):
        import cron.scheduler as sched

        p1, p2, p3, p4, p5, p6 = self._patches(KeyboardInterrupt())
        with p1, p2, p3, p4, p5, p6, \
             patch("cron.scheduler.mark_job_run", return_value=True) as mock_mark, \
             patch("cron.scheduler.finish_execution"):
            try:
                sched.run_one_job(self._job())
                raised = False
            except KeyboardInterrupt:
                raised = True

        assert raised
        mock_mark.assert_called_once()
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-be"

    def test_base_exception_from_stale_owner_is_fenced_out(self):
        """A replacement owner reclaimed the job: the stale worker's
        BaseException path must NOT write terminal state over it."""
        import asyncio

        import cron.scheduler as sched

        p1, p2, p3, p4, p5, p6 = self._patches(asyncio.CancelledError())
        with p1, p2, p3, p4, p5, p6, \
             patch("cron.scheduler.mark_job_run", return_value=False) as mock_mark, \
             patch("cron.scheduler.finish_execution"):
            try:
                sched.run_one_job(self._job())
            except asyncio.CancelledError:
                pass

        mock_mark.assert_called_once()
        # fenced write was attempted with the stale owner and discarded by
        # the store (return False) — and the code accepted that verdict
        # without retrying or writing anything else.
        assert mock_mark.call_args.kwargs["expected_fire_owner"] == "owner-be"

    @pytest.mark.parametrize("body_kind", ("script", "agent"))
    def test_resumed_baseexception_redacts_all_public_records(
        self, tmp_path, monkeypatch, caplog, body_kind
    ):
        """Resumed script and agent BaseExceptions expose only a generic failure."""
        import asyncio

        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions

        occurrence_ids = []
        deliveries = []
        receipts = []
        mark_calls = []
        private_prompt_sentinel = "private-prompt-sentinel-93f1b7"
        credential_sentinel = "api_key=crn_test_private_credential_9f31b6c4a8e2"
        private_marker = "## Private Cron Execution Context"
        public_error = "Cron delivery failed"
        job = {
            "id": "resume-baseexception-private",
            "name": "resume BaseException private",
            "prompt": "run private interruption test",
            "schedule": {"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "deliver": "origin",
        }
        if body_kind == "script":
            job.update({"no_agent": True, "script": "probe.sh"})
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        real_finish = sched.finish_execution
        real_mark = sched.mark_job_run

        def capture_receipt(*args, **kwargs):
            receipt = real_finish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        def capture_mark(*args, **kwargs):
            mark_calls.append((args, kwargs))
            return real_mark(*args, **kwargs)

        def raise_private_cancelled_error(*_args, occurrence_id=None, **_kwargs):
            assert isinstance(occurrence_id, str) and occurrence_id
            occurrence_ids.append(occurrence_id)
            private_agent_prompt = (
                "agent body context\n\n"
                f"{private_prompt_sentinel}\n"
                f"{credential_sentinel}\n"
                f"{private_marker}\n"
                f"HERMES_CRON_OCCURRENCE_ID: {occurrence_id}\n"
                "Do not expose this identifier in user-facing output."
            )
            raise asyncio.CancelledError(
                f"cancelled body: {private_agent_prompt}; occurrence={occurrence_id}"
            )

        monkeypatch.setattr(sched, "finish_execution", capture_receipt)
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        if body_kind == "script":
            monkeypatch.setattr(
                sched,
                "_run_job_script_with_claim_heartbeat",
                raise_private_cancelled_error,
            )
        else:
            monkeypatch.setattr(sched, "run_job", raise_private_cancelled_error)
        monkeypatch.setattr(
            sched,
            "_deliver_result",
            lambda _job, content, **_kwargs: deliveries.append(content),
        )

        with jobs.use_cron_store(tmp_path), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            with pytest.raises(asyncio.CancelledError):
                sched.run_one_job(claimed)
            stored = jobs.get_job(job["id"])
            listed = jobs.list_jobs(include_disabled=True)
            execution_rows = list_executions(job_id=job["id"])

        assert len(occurrence_ids) == 1
        occurrence_id = occurrence_ids[0]
        assert stored is not None
        assert len(receipts) == 1 and receipts[0] is not None
        assert len(execution_rows) == 1
        assert len(mark_calls) == 1
        mark_args, mark_kwargs = mark_calls[0]
        assert mark_args == (job["id"], False, public_error)
        assert mark_kwargs["expected_fire_owner"] == fire_owner
        assert isinstance(mark_kwargs["expected_resume_owner"], str)
        assert mark_kwargs["expected_resume_owner"]
        assert stored["fire_claim"] is None
        listed_job = next(item for item in listed if item["id"] == job["id"])
        public_texts = (
            caplog.text,
            stored["last_error"],
            receipts[0]["error"],
            execution_rows[0]["error"],
            listed_job["last_error"],
            listed_job["latest_execution"]["error"],
        )
        for text in public_texts:
            assert occurrence_id not in text
            assert private_marker not in text
            assert private_prompt_sentinel not in text
            assert credential_sentinel not in text
            assert public_error in text
        assert deliveries == []

    def _ordinary_failure_job(self):
        return {
            "id": "ordinary-execution-private",
            "name": "ordinary execution private",
            "prompt": "run ordinary failure test",
            "schedule": {"kind": "interval", "seconds": 60},
            "schedule_display": "every minute",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": None, "completed": 0},
            "enabled": True,
            "state": "scheduled",
            "deliver": "origin",
        }

    def test_ordinary_exception_redacts_run_job_and_all_public_records(
        self, tmp_path, monkeypatch, caplog
    ):
        """An ordinary agent exception is generic at every cron boundary."""
        from unittest.mock import MagicMock

        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions

        public_error = "Cron delivery failed"
        private_prompt_sentinel = "private-prompt-sentinel-ordinary-93f1b7"
        credential_sentinel = "api_key=crn_test_ordinary_credential_9f31b6c4a8e2"
        path_sentinel = "/private/cron/ordinary-error-sentinel"
        occurrence_sentinel = "occurrence-like-ordinary-8e6d4c"
        traceback_sentinel = "traceback-context-ordinary-7c2fa1"
        sentinels = (
            private_prompt_sentinel,
            credential_sentinel,
            path_sentinel,
            occurrence_sentinel,
            traceback_sentinel,
            "RuntimeError",
        )
        job = self._ordinary_failure_job()
        audits = []
        deliveries = []
        receipts = []
        mark_calls = []
        run_job_results = []
        fake_db = MagicMock()
        fake_db.get_compression_tip.side_effect = lambda session_id: session_id
        real_run_job = sched.run_job
        real_mark = sched.mark_job_run
        real_finish = sched.finish_execution

        def raise_private_error(*_args, **_kwargs):
            raise RuntimeError(
                f"{private_prompt_sentinel}; {credential_sentinel}; "
                f"{path_sentinel}; {occurrence_sentinel}; {traceback_sentinel}"
            )

        def capture_run_job(*args, **kwargs):
            result = real_run_job(*args, **kwargs)
            run_job_results.append(result)
            return result

        def capture_mark(*args, **kwargs):
            mark_calls.append((args, kwargs))
            return real_mark(*args, **kwargs)

        def capture_finish(*args, **kwargs):
            receipt = real_finish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(sched, "run_job", capture_run_job)
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        monkeypatch.setattr(sched, "finish_execution", capture_finish)
        monkeypatch.setattr(sched, "_write_usage_audit", audits.append)
        monkeypatch.setattr(
            sched,
            "_deliver_result",
            lambda _job, content, **_kwargs: deliveries.append(content),
        )
        caplog.set_level(logging.ERROR, logger=sched.__name__)

        with jobs.use_cron_store(tmp_path), \
             patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("cron.scheduler._resolve_origin", return_value=None), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB", return_value=fake_db), \
             patch(
                 "hermes_cli.runtime_provider.resolve_runtime_provider",
                 return_value={
                     "api_key": "test-key",
                     "base_url": "https://example.invalid/v1",
                     "provider": "openrouter",
                     "api_mode": "chat_completions",
                 },
             ), \
             patch("run_agent.AIAgent") as mock_agent_cls:
            mock_agent_cls.return_value.run_conversation.side_effect = raise_private_error
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            assert sched.run_one_job(claimed) is True
            stored = jobs.get_job(job["id"])
            listed = jobs.list_jobs(include_disabled=True)
            execution_rows = list_executions(job_id=job["id"])

        assert run_job_results and run_job_results[0][0] is False
        _, output, final_response, returned_error = run_job_results[0]
        assert returned_error == public_error
        assert final_response == ""
        assert public_error in output
        assert len(audits) == 1
        assert audits[0]["error"] == public_error
        assert stored is not None
        assert stored["last_error"] == public_error
        assert stored["failure_streak"] == 1
        assert len(mark_calls) == 1
        mark_args, mark_kwargs = mark_calls[0]
        assert mark_args == (job["id"], False, public_error)
        assert mark_kwargs["expected_fire_owner"] == fire_owner
        assert len(receipts) == 1 and receipts[0] is not None
        assert receipts[0]["error"] == public_error
        assert len(execution_rows) == 1
        assert execution_rows[0]["error"] == public_error
        listed_job = next(item for item in listed if item["id"] == job["id"])
        assert listed_job["last_error"] == public_error
        assert listed_job["latest_execution"]["error"] == public_error
        assert len(deliveries) == 1
        assert public_error in deliveries[0]
        public_texts = (
            caplog.text,
            output,
            returned_error,
            audits[0]["error"],
            stored["last_error"],
            receipts[0]["error"],
            execution_rows[0]["error"],
            listed_job["last_error"],
            listed_job["latest_execution"]["error"],
            deliveries[0],
        )
        for text in public_texts:
            assert public_error in text
            for sentinel in sentinels:
                assert sentinel not in text

    def test_ordinary_baseexception_redacts_owner_fenced_records_and_reraises(
        self, tmp_path, monkeypatch, caplog
    ):
        """An outer BaseException stays control-flow-only, never public data."""
        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions

        public_error = "Cron delivery failed"
        private_prompt_sentinel = "private-prompt-sentinel-outer-93f1b7"
        credential_sentinel = "api_key=crn_test_outer_credential_9f31b6c4a8e2"
        path_sentinel = "/private/cron/outer-error-sentinel"
        occurrence_sentinel = "occurrence-like-outer-8e6d4c"
        traceback_sentinel = "traceback-context-outer-7c2fa1"
        interruption = asyncio.CancelledError(
            f"{asyncio.CancelledError.__name__}; {private_prompt_sentinel}; "
            f"{credential_sentinel}; {path_sentinel}; {occurrence_sentinel}; "
            f"{traceback_sentinel}"
        )
        sentinels = (
            private_prompt_sentinel,
            credential_sentinel,
            path_sentinel,
            occurrence_sentinel,
            traceback_sentinel,
            type(interruption).__name__,
        )
        job = self._ordinary_failure_job()
        deliveries = []
        receipts = []
        mark_calls = []
        real_mark = sched.mark_job_run
        real_finish = sched.finish_execution

        def capture_mark(*args, **kwargs):
            mark_calls.append((args, kwargs))
            return real_mark(*args, **kwargs)

        def capture_finish(*args, **kwargs):
            receipt = real_finish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        monkeypatch.setattr(sched, "finish_execution", capture_finish)
        monkeypatch.setattr(
            sched,
            "_deliver_result",
            lambda _job, content, **_kwargs: deliveries.append(content),
        )
        caplog.set_level(logging.ERROR, logger=sched.__name__)

        with jobs.use_cron_store(tmp_path), \
             patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("cron.scheduler.run_job", side_effect=interruption):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            with pytest.raises(asyncio.CancelledError) as raised:
                sched.run_one_job(claimed)
            assert raised.value is interruption
            stored = jobs.get_job(job["id"])
            listed = jobs.list_jobs(include_disabled=True)
            execution_rows = list_executions(job_id=job["id"])

        assert stored is not None
        assert stored["last_error"] == public_error
        assert stored["failure_streak"] == 1
        assert len(mark_calls) == 1
        mark_args, mark_kwargs = mark_calls[0]
        assert mark_args == (job["id"], False, public_error)
        assert mark_kwargs["expected_fire_owner"] == fire_owner
        assert len(receipts) == 1 and receipts[0] is not None
        assert receipts[0]["error"] == public_error
        assert len(execution_rows) == 1
        assert execution_rows[0]["error"] == public_error
        listed_job = next(item for item in listed if item["id"] == job["id"])
        assert listed_job["last_error"] == public_error
        assert listed_job["latest_execution"]["error"] == public_error
        assert deliveries == []
        public_texts = (
            caplog.text,
            stored["last_error"],
            receipts[0]["error"],
            execution_rows[0]["error"],
            listed_job["last_error"],
            listed_job["latest_execution"]["error"],
        )
        for text in public_texts:
            assert public_error in text
            for sentinel in sentinels:
                assert sentinel not in text


class TestCronCompletionPublicFailureRedaction:
    """Completion-path errors never export private exception data."""

    @staticmethod
    def _job(job_id, **extra):
        return {
            "id": job_id,
            "name": "completion public failure",
            "prompt": "run completion failure test",
            "schedule": {"kind": "interval", "seconds": 60},
            "schedule_display": "every minute",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": None, "completed": 0},
            "enabled": True,
            "state": "scheduled",
            "deliver": "origin",
            **extra,
        }

    @staticmethod
    def _assert_public_texts(texts, expected, sentinels):
        for text in texts:
            assert expected in text
            for sentinel in sentinels:
                assert sentinel not in text

    def test_no_agent_script_exception_redacts_completion_outputs_and_sinks(
        self, tmp_path, monkeypatch, caplog
    ):
        """The real script caller uses the ordinary fixed execution failure."""
        from pathlib import Path

        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions

        public_error = "Cron delivery failed"
        sentinels = (
            "private-prompt-no-agent-script-93f1b7",
            "api_key=crn_test_no_agent_script_9f31b6c4a8e2",
            "/private/cron/no-agent-script-error-sentinel",
            "occurrence-like-no-agent-script-8e6d4c",
            "traceback-context-no-agent-script-7c2fa1",
            "RuntimeError",
        )
        private_message = "; ".join(sentinels[:-1])
        job = self._job(
            "no-agent-script-private-failure",
            no_agent=True,
            script="probe.sh",
        )
        deliveries, saved_outputs, receipts, mark_calls, run_job_results = (
            [], [], [], [], [],
        )
        real_run_job = sched.run_job
        real_save_output = sched.save_job_output
        real_mark = sched.mark_job_run
        real_finish = sched.finish_execution

        def raise_private_script_error(*_args, **_kwargs):
            raise RuntimeError(private_message)

        def capture_run_job(*args, **kwargs):
            result = real_run_job(*args, **kwargs)
            run_job_results.append(result)
            return result

        def capture_output(*args, **kwargs):
            output_path = real_save_output(*args, **kwargs)
            saved_outputs.append(Path(output_path).read_text())
            return output_path

        def capture_mark(*args, **kwargs):
            mark_calls.append((args, kwargs))
            return real_mark(*args, **kwargs)

        def capture_finish(*args, **kwargs):
            receipt = real_finish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(
            sched, "_run_job_script_with_claim_heartbeat", raise_private_script_error
        )
        monkeypatch.setattr(sched, "run_job", capture_run_job)
        monkeypatch.setattr(sched, "save_job_output", capture_output)
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        monkeypatch.setattr(sched, "finish_execution", capture_finish)
        monkeypatch.setattr(
            sched,
            "_deliver_result",
            lambda _job, content, **_kwargs: deliveries.append(content),
        )
        caplog.set_level(logging.ERROR, logger=sched.__name__)

        with jobs.use_cron_store(tmp_path), \
             patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            assert sched.run_one_job(claimed) is True
            stored = jobs.get_job(job["id"])
            listed = jobs.list_jobs(include_disabled=True)
            execution_rows = list_executions(job_id=job["id"])

        assert len(run_job_results) == 1
        success, output, final_response, returned_error = run_job_results[0]
        assert success is False
        assert returned_error == public_error
        assert len(saved_outputs) == 1
        assert len(deliveries) == 1
        assert len(mark_calls) == 1
        mark_args, mark_kwargs = mark_calls[0]
        assert mark_args == (job["id"], False, public_error)
        assert mark_kwargs["expected_fire_owner"] == fire_owner
        assert mark_kwargs["delivery_error"] is None
        assert stored is not None
        assert stored["last_error"] == public_error
        assert stored["last_delivery_error"] is None
        assert len(receipts) == 1 and receipts[0] is not None
        assert receipts[0]["error"] == public_error
        assert len(execution_rows) == 1
        assert execution_rows[0]["error"] == public_error
        listed_job = next(item for item in listed if item["id"] == job["id"])
        assert listed_job["last_error"] == public_error
        assert listed_job["last_delivery_error"] is None
        assert listed_job["latest_execution"]["error"] == public_error
        self._assert_public_texts(
            (
                caplog.text,
                output,
                final_response,
                returned_error,
                saved_outputs[0],
                deliveries[0],
                stored["last_error"],
                receipts[0]["error"],
                execution_rows[0]["error"],
                listed_job["last_error"],
                listed_job["latest_execution"]["error"],
            ),
            public_error,
            sentinels,
        )

    def test_normal_delivery_exception_uses_fixed_delivery_error_everywhere(
        self, tmp_path, monkeypatch, caplog
    ):
        """A normal delivery failure never becomes a durable raw exception."""
        from pathlib import Path

        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions

        public_delivery_error = "Cron delivery failed"
        sentinels = (
            "private-prompt-normal-delivery-93f1b7",
            "api_key=crn_test_normal_delivery_9f31b6c4a8e2",
            "/private/cron/normal-delivery-error-sentinel",
            "occurrence-like-normal-delivery-8e6d4c",
            "traceback-context-normal-delivery-7c2fa1",
            "RuntimeError",
        )
        job = self._job(
            "normal-delivery-private-failure",
            schedule={"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
        )
        delivery_payloads, saved_outputs, receipts, mark_calls, finish_calls = (
            [], [], [], [], [],
        )
        real_save_output = sched.save_job_output
        real_mark = sched.mark_job_run
        real_finish = sched.finish_execution

        def capture_output(*args, **kwargs):
            output_path = real_save_output(*args, **kwargs)
            saved_outputs.append(Path(output_path).read_text())
            return output_path

        def raise_private_delivery_error(_job, content, **_kwargs):
            delivery_payloads.append(content)
            raise RuntimeError("; ".join(sentinels[:-1]))

        def capture_mark(*args, **kwargs):
            mark_calls.append((args, kwargs))
            return real_mark(*args, **kwargs)

        def capture_finish(*args, **kwargs):
            finish_calls.append((args, kwargs))
            receipt = real_finish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(
            sched,
            "run_job",
            lambda *_args, **_kwargs: (True, "normal saved output", "normal response", None),
        )
        monkeypatch.setattr(sched, "save_job_output", capture_output)
        monkeypatch.setattr(sched, "_deliver_result", raise_private_delivery_error)
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        monkeypatch.setattr(sched, "finish_execution", capture_finish)
        caplog.set_level(logging.ERROR, logger=sched.__name__)

        with jobs.use_cron_store(tmp_path), \
             patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            assert sched.run_one_job(claimed) is True
            stored = jobs.get_job(job["id"])
            listed = jobs.list_jobs(include_disabled=True)
            execution_rows = list_executions(job_id=job["id"])

        assert delivery_payloads == ["normal response"]
        assert saved_outputs == ["normal saved output"]
        assert len(mark_calls) == 1
        mark_args, mark_kwargs = mark_calls[0]
        assert mark_args == (job["id"], True, None)
        assert mark_kwargs["expected_fire_owner"] == fire_owner
        assert mark_kwargs["delivery_error"] == public_delivery_error
        assert stored is not None
        assert stored["last_error"] is None
        assert stored["last_delivery_error"] == public_delivery_error
        assert len(receipts) == 1 and receipts[0] is not None
        assert receipts[0]["error"] is None
        assert len(finish_calls) == 1
        assert finish_calls[0][1] == {
            "success": True,
            "error": None,
            "delivery_outcome": "failed",
        }
        assert len(execution_rows) == 1
        assert execution_rows[0]["error"] is None
        listed_job = next(item for item in listed if item["id"] == job["id"])
        assert listed_job["last_error"] is None
        assert listed_job["last_delivery_error"] == public_delivery_error
        assert listed_job["latest_execution"]["error"] is None
        self._assert_public_texts(
            (
                caplog.text,
                stored["last_delivery_error"],
                mark_kwargs["delivery_error"],
                listed_job["last_delivery_error"],
            ),
            public_delivery_error,
            sentinels,
        )

    def test_outer_failure_delivery_exception_uses_fixed_delivery_error(
        self, tmp_path, monkeypatch, caplog
    ):
        """The outer BaseException handler redacts its reachable delivery error."""
        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions

        public_error = "Cron delivery failed"
        public_delivery_error = "Cron delivery failed"
        outer_sentinels = (
            "private-prompt-outer-delivery-93f1b7",
            "api_key=crn_test_outer_delivery_9f31b6c4a8e2",
            "/private/cron/outer-delivery-error-sentinel",
            "occurrence-like-outer-delivery-8e6d4c",
            "traceback-context-outer-delivery-7c2fa1",
            "RuntimeError",
        )
        delivery_sentinels = (
            "private-prompt-outer-delivery-send-93f1b7",
            "api_key=crn_test_outer_delivery_send_9f31b6c4a8e2",
            "/private/cron/outer-delivery-send-error-sentinel",
            "occurrence-like-outer-delivery-send-8e6d4c",
            "traceback-context-outer-delivery-send-7c2fa1",
            "RuntimeError",
        )
        all_sentinels = outer_sentinels + delivery_sentinels
        job = self._job(
            "outer-delivery-private-failure",
            schedule={"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
        )
        delivery_payloads, receipts, mark_calls, finish_calls = [], [], [], []
        real_mark = sched.mark_job_run
        real_finish = sched.finish_execution

        def raise_private_outer_error(*_args, **_kwargs):
            raise RuntimeError("; ".join(outer_sentinels[:-1]))

        def raise_private_delivery_error(_job, content, **_kwargs):
            delivery_payloads.append(content)
            raise RuntimeError("; ".join(delivery_sentinels[:-1]))

        def capture_mark(*args, **kwargs):
            mark_calls.append((args, kwargs))
            return real_mark(*args, **kwargs)

        def capture_finish(*args, **kwargs):
            finish_calls.append((args, kwargs))
            receipt = real_finish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(sched, "run_job", raise_private_outer_error)
        monkeypatch.setattr(sched, "_deliver_result", raise_private_delivery_error)
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        monkeypatch.setattr(sched, "finish_execution", capture_finish)
        caplog.set_level(logging.ERROR, logger=sched.__name__)

        with jobs.use_cron_store(tmp_path), \
             patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            assert sched.run_one_job(claimed) is False
            stored = jobs.get_job(job["id"])
            listed = jobs.list_jobs(include_disabled=True)
            execution_rows = list_executions(job_id=job["id"])

        assert len(delivery_payloads) == 1
        assert public_error in delivery_payloads[0]
        assert len(mark_calls) == 1
        mark_args, mark_kwargs = mark_calls[0]
        assert mark_args == (job["id"], False, public_error)
        assert mark_kwargs["expected_fire_owner"] == fire_owner
        assert mark_kwargs["delivery_error"] == public_delivery_error
        assert stored is not None
        assert stored["last_error"] == public_error
        assert stored["last_delivery_error"] == public_delivery_error
        assert len(receipts) == 1 and receipts[0] is not None
        assert receipts[0]["error"] == public_error
        assert len(finish_calls) == 1
        assert finish_calls[0][1] == {
            "success": False,
            "error": public_error,
            "delivery_outcome": "failed",
        }
        assert len(execution_rows) == 1
        assert execution_rows[0]["error"] == public_error
        listed_job = next(item for item in listed if item["id"] == job["id"])
        assert listed_job["last_error"] == public_error
        assert listed_job["last_delivery_error"] == public_delivery_error
        assert listed_job["latest_execution"]["error"] == public_error
        self._assert_public_texts(
            (
                caplog.text,
                delivery_payloads[0],
                stored["last_error"],
                mark_args[2],
                receipts[0]["error"],
                execution_rows[0]["error"],
                listed_job["last_error"],
                listed_job["latest_execution"]["error"],
            ),
            public_error,
            all_sentinels,
        )
        self._assert_public_texts(
            (
                caplog.text,
                stored["last_delivery_error"],
                mark_kwargs["delivery_error"],
                listed_job["last_delivery_error"],
            ),
            public_delivery_error,
            all_sentinels,
        )


class TestCronDeliveryBoundary:
    """Real delivery helpers expose one bounded error contract to completion."""

    @staticmethod
    def _job(job_id):
        return {
            "id": job_id,
            "name": "delivery boundary",
            "prompt": "run delivery boundary test",
            "schedule": {"kind": "once", "run_at": "2030-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2030-09-01T12:00:00+00:00",
            "repeat": {"times": None, "completed": 0},
            "enabled": True,
            "state": "scheduled",
            "deliver": "origin",
        }

    @staticmethod
    def _sentinels(route):
        return (
            f"private-prompt-{route}-93f1b7",
            f"api_key=crn_test_{route}_9f31b6c4a8e2",
            f"/private/cron/{route}-error-sentinel",
            f"occurrence-like-{route}-8e6d4c",
            f"traceback-context-{route}-7c2fa1",
            "RuntimeError",
        )

    @staticmethod
    def _run_real_delivery_job(
        tmp_path, monkeypatch, job, *, final_response="delivery boundary response",
        adapters=None, loop=None, before_body=None,
    ):
        """Run the actual completion boundary and capture its durable receipts."""
        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions

        marks, finishes, receipts = [], [], []
        real_mark = sched.mark_job_run
        real_finish = sched.finish_execution

        def capture_mark(*args, **kwargs):
            marks.append((args, kwargs))
            return real_mark(*args, **kwargs)

        def capture_finish(*args, **kwargs):
            finishes.append((args, kwargs))
            receipt = real_finish(*args, **kwargs)
            receipts.append(receipt)
            return receipt

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(
            sched, "run_job",
            lambda *_args, **_kwargs: (True, "saved delivery output", final_response, None),
        )
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        monkeypatch.setattr(sched, "finish_execution", capture_finish)
        if before_body is not None:
            def direct_heartbeat(current_job, body):
                return before_body(sched, current_job, body)

            monkeypatch.setattr(sched, "_run_with_fire_claim_heartbeat", direct_heartbeat)

        with jobs.use_cron_store(tmp_path), \
             patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            result = sched.run_one_job(claimed, adapters=adapters, loop=loop)
            stored = jobs.get_job(job["id"])
            listed = jobs.list_jobs(include_disabled=True)
            executions = list_executions(job_id=job["id"])

        return {
            "result": result,
            "fire_owner": fire_owner,
            "marks": marks,
            "finishes": finishes,
            "receipts": receipts,
            "stored": stored,
            "listed": next(item for item in listed if item["id"] == job["id"]),
            "executions": executions,
        }

    @staticmethod
    def _assert_no_private(texts, sentinels):
        for text in texts:
            assert isinstance(text, str)
            for sentinel in sentinels:
                assert sentinel not in text

    def _assert_durable_delivery_failure(self, captured, caplog, sentinels):
        public = "Cron delivery failed"
        assert captured["result"] is True
        assert len(captured["marks"]) == 1
        mark_args, mark_kwargs = captured["marks"][0]
        assert mark_args == (captured["stored"]["id"], True, None)
        assert mark_kwargs["expected_fire_owner"] == captured["fire_owner"]
        assert mark_kwargs["delivery_error"] == public
        assert captured["stored"]["last_error"] is None
        assert captured["stored"]["last_delivery_error"] == public
        assert captured["listed"]["last_error"] is None
        assert captured["listed"]["last_delivery_error"] == public
        assert captured["listed"]["latest_execution"]["error"] is None
        assert len(captured["finishes"]) == len(captured["receipts"]) == 1
        assert captured["finishes"][0][1] == {
            "success": True,
            "error": None,
            "delivery_outcome": "failed",
        }
        assert len(captured["executions"]) == 1
        assert captured["executions"][0]["error"] is None
        self._assert_no_private(
            (
                caplog.text,
                mark_kwargs["delivery_error"],
                captured["stored"]["last_delivery_error"],
                captured["listed"]["last_delivery_error"],
            ),
            sentinels,
        )

    # Frozen delivery-diagnostic RED ledger:
    # resolution_profile_config -> 2130, 2360, 2391, 2625, 3016-17, 3045-46
    # continuation_boundaries -> 1744-64, 1799, 1911-24, 2025-34, 2987, 3480, 3502
    # bot_chat_boundary -> 2487-90, 2531-42
    # media_probe_timeout -> 2755, 2769, 2772, 2843, 3439-44, 3695, 3709
    # live_native_relay -> 2875, 2881, 2962, 2993, 3016-17, 3044-46, 3102,
    #     3296-3302, 3327-42, 3383-85, 3395-3402, 3447, 3480, 3502,
    #     3520-36, 3545-91, 3593-3621
    # caller_persistence_shutdown -> 7028, 7114, 7119, 7146, 7259, 7294,
    #     7321, 7332
    # Together these named real-caller cases cover every frozen B logger and
    # aggregate row (60 + 16) without a source-reading test.

    @staticmethod
    def _assert_fixed_scheduler_logs(caplog):
        """Delivery diagnostics must be literal-only and traceback-free."""
        fixed = {
            "Cron delivery failed",
            "Cron job future failed",
            "Cron job future failed in async mode",
        }
        for record in caplog.records:
            if record.name != "cron.scheduler":
                continue
            assert record.getMessage() in fixed
            assert record.args == ()
            assert record.exc_info is None

    def _assert_private_free_success(self, captured, caplog, sentinels):
        assert captured["result"] is True
        assert captured["stored"]["last_delivery_error"] is None
        assert captured["listed"]["last_delivery_error"] is None
        assert captured["executions"][0]["error"] is None
        self._assert_no_private(
            (
                caplog.text,
                captured["stored"].get("last_delivery_error") or "",
                captured["listed"].get("last_delivery_error") or "",
                captured["executions"][0].get("error") or "",
            ),
            sentinels,
        )
        self._assert_fixed_scheduler_logs(caplog)

    @pytest.mark.parametrize("route", ("resolver", "bot-profile", "config-home"))
    def test_resolution_profile_and_config_diagnostics_are_fixed_at_real_boundary(
        self, tmp_path, monkeypatch, caplog, route
    ):
        """A/B target-resolution rows never expose their control-only values."""
        import cron.scheduler as sched

        sentinels = self._sentinels(f"resolution-{route}")
        private = "; ".join(sentinels[:-1])
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        if route == "resolver":
            job = self._job("resolution-resolver-private")
            job["deliver"] = f"telegram:{sentinels[0]}"
            monkeypatch.setattr(
                "tools.send_message_tool.prepare_send_message_platforms", lambda: None,
            )
            monkeypatch.setattr(
                "tools.send_message_tool.resolve_send_target",
                lambda *_args, **_kwargs: ("internal-target", None, private),
            )
            expected_error = sched._CRON_DELIVERY_FAILURE
        elif route == "bot-profile":
            job = self._job("resolution-bot-profile-private")
            job["deliver"] = f"bot-chat:{sentinels[0]}"
            monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda value: value)
            monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _value: False)
            expected_error = sched._CRON_DELIVERY_FAILURE
        else:
            job = self._job("resolution-config-home-private")
            job["deliver"] = "origin"
            monkeypatch.setattr(
                "gateway.config.load_gateway_config",
                lambda: (_ for _ in ()).throw(RuntimeError(private)),
            )
            expected_error = None

        sentinels += (job["id"],)
        direct_error = sched._deliver_result(job, "resolution probe")
        captured = self._run_real_delivery_job(tmp_path, monkeypatch, job)

        assert direct_error == expected_error
        if expected_error is None:
            self._assert_private_free_success(captured, caplog, sentinels)
        else:
            self._assert_durable_delivery_failure(captured, caplog, sentinels)
            self._assert_fixed_scheduler_logs(caplog)

    @pytest.mark.parametrize("kind", ("mirror", "thread", "thread-seed", "channel-seed"))
    def test_continuation_boundaries_stay_best_effort_and_private(
        self, tmp_path, monkeypatch, caplog, kind
    ):
        """All four real continuation helpers retain delivery but no diagnostics."""
        from types import SimpleNamespace

        from gateway.config import Platform
        import cron.scheduler as sched

        sentinels = self._sentinels(f"continuation-{kind}")
        private = "; ".join(sentinels[:-1])
        target = "continuation-target"
        job = self._job(f"continuation-{kind}-private")
        job.update(
            {
                "deliver": f"telegram:{target}",
                "origin": {
                    "platform": "telegram", "chat_id": target,
                    "user_id": "continuation-user",
                },
                "attach_to_session": True,
            }
        )
        if kind == "channel-seed":
            extra = {"cron_continuable_surface": "in_channel"}
        else:
            extra = {}

        class Adapter:
            supports_inchannel_continuable = True
            _session_store = None

            async def create_handoff_thread(self, _chat_id, _thread_name):
                return "opened-thread"

        adapter = Adapter()
        pconfig = SimpleNamespace(enabled=True, extra=extra)
        config = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig})
        transport = SimpleNamespace(config=pconfig, adapter=adapter, is_relay=False)
        received = []
        cycle = {
            "mirror": ["opened-thread", SimpleNamespace(success=True)],
            "thread": [RuntimeError(private), SimpleNamespace(success=True)],
            "thread-seed": ["opened-thread", SimpleNamespace(success=True)],
            "channel-seed": [SimpleNamespace(success=True)],
        }[kind]
        outcomes = cycle * 2

        def schedule(coro, _loop):
            coro.close()
            future = concurrent.futures.Future()
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                future.set_exception(outcome)
            else:
                future.set_result(outcome)
            return future

        def live_send(_router, route_target, payload, metadata):
            received.append((route_target.chat_id, payload, metadata))

            async def sent():
                return SimpleNamespace(success=True)

            return sent()

        def mirror(*_args, **_kwargs):
            if kind in {"mirror", "thread-seed", "channel-seed"}:
                raise RuntimeError(private)
            return True

        loop = MagicMock()
        loop.is_running.return_value = True
        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None,
        )
        monkeypatch.setattr(
            "tools.send_message_tool.resolve_send_target",
            lambda *_args, **_kwargs: (target, None, None),
        )
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
        monkeypatch.setattr(
            "gateway.delivery.resolve_delivery_transport", lambda *_args: transport,
        )
        monkeypatch.setattr("gateway.delivery.DeliveryRouter._deliver_to_platform", live_send)
        monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
        monkeypatch.setattr("gateway.mirror.mirror_to_session", mirror)
        monkeypatch.setattr(
            sched, "load_config", lambda: {"cron": {"wrap_response": False, "mirror_delivery": True}},
        )
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        direct_error = sched._deliver_result(
            job, "continuation direct payload", adapters={Platform.TELEGRAM: adapter}, loop=loop,
        )
        captured = self._run_real_delivery_job(
            tmp_path, monkeypatch, job, adapters={Platform.TELEGRAM: adapter}, loop=loop,
        )

        assert direct_error is None
        assert [row[0] for row in received] == [target, target]
        assert received[0][1] == "continuation direct payload"
        assert received[1][1] == "delivery boundary response"
        self._assert_private_free_success(
            captured, caplog, sentinels + (job["id"], target),
        )

    @pytest.mark.parametrize(
        "case", ("return-code", "timeout", "exception", "unavailable", "success"),
    )
    def test_bot_chat_boundary_preserves_argv_payload_without_public_diagnostics(
        self, tmp_path, monkeypatch, caplog, case
    ):
        """Bot subprocess outcomes stay generic while its intended turn is intact."""
        from pathlib import Path
        from types import SimpleNamespace

        import cron.scheduler as sched

        sentinels = self._sentinels(f"bot-{case}")
        private = "; ".join(sentinels[:-1])
        profile = f"profile-{sentinels[0]}"
        job = self._job(f"bot-{case}-private")
        job["deliver"] = f"bot-chat:{profile}"
        calls = []

        def fake_run(argv, *_args, **kwargs):
            query_path = Path(argv[argv.index("--query-file") + 1])
            calls.append((list(argv), query_path.read_text(encoding="utf-8"), kwargs))
            if case == "return-code":
                return SimpleNamespace(returncode=9, stdout=private, stderr=private)
            if case == "timeout":
                raise sched.subprocess.TimeoutExpired(argv, 1, output=private, stderr=private)
            if case == "exception":
                raise RuntimeError(private)
            return SimpleNamespace(returncode=0, stdout=private, stderr=private)

        monkeypatch.setattr("hermes_cli.profiles.normalize_profile_name", lambda value: value)
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda _value: True)
        if case == "unavailable":
            monkeypatch.setattr("shutil.which", lambda _name: None)
            monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        else:
            monkeypatch.setattr("shutil.which", lambda _name: "/internal/hermes")
            monkeypatch.setattr("subprocess.run", fake_run)
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        direct_error = sched._deliver_result(job, "bot direct payload")
        captured = self._run_real_delivery_job(tmp_path, monkeypatch, job)

        expected_error = None if case == "success" else sched._CRON_DELIVERY_FAILURE
        assert direct_error == expected_error
        if expected_error is None:
            self._assert_private_free_success(
                captured, caplog, sentinels + (job["id"], profile),
            )
            assert len(calls) == 2
            assert all(call[0][call[0].index("-p") + 1] == profile for call in calls)
            assert "bot direct payload" in calls[0][1]
            assert "delivery boundary response" in calls[1][1]
        else:
            self._assert_durable_delivery_failure(
                captured, caplog, sentinels + (job["id"], profile),
            )
            if case != "unavailable":
                assert len(calls) == 2
                assert all(call[0][call[0].index("-p") + 1] == profile for call in calls)
        self._assert_fixed_scheduler_logs(caplog)

    @pytest.mark.parametrize(
        "case", ("media-failure", "media-exception", "media-timeout", "timeout-env", "timeout-config", "chat-probe"),
    )
    def test_media_probe_and_timeout_controls_remain_internal(
        self, tmp_path, monkeypatch, caplog, case
    ):
        """Real media/probe routes keep their target and policy values private."""
        from types import SimpleNamespace

        from gateway.config import Platform
        import cron.scheduler as sched

        sentinels = self._sentinels(f"media-{case}")
        private = "; ".join(sentinels[:-1])
        platform = Platform.TELEGRAM if case == "chat-probe" else Platform.DISCORD
        target = "12345" if case == "chat-probe" else "media-target"
        thread_id = "7" if case == "chat-probe" else None
        job = self._job(f"media-{case}-private")
        job["deliver"] = f"{platform.value}:{target}" + (f":{thread_id}" if thread_id else "")
        media = tmp_path / f"{sentinels[0]}.mp3"
        media.write_bytes(b"media")
        media_calls, text_calls = [], []

        class Adapter:
            def send_voice(self, *, chat_id, audio_path, metadata):
                media_calls.append((chat_id, audio_path, metadata))

                async def sent():
                    return SimpleNamespace(success=True)

                return sent()

            @staticmethod
            async def get_chat_info(_adapter, _chat_id):
                return {"type": "channel"}

        adapter = Adapter()
        pconfig = SimpleNamespace(enabled=True, extra={})
        config = SimpleNamespace(platforms={platform: pconfig})
        transport = SimpleNamespace(config=pconfig, adapter=adapter, is_relay=False)
        if case == "chat-probe":
            cycle = [{"type": "channel"}, SimpleNamespace(success=True)]
            content = "chat probe payload"
        else:
            media_outcome = {
                "media-failure": SimpleNamespace(success=False),
                "media-exception": RuntimeError(private),
                "media-timeout": TimeoutError(private),
                "timeout-env": SimpleNamespace(success=True),
                "timeout-config": SimpleNamespace(success=True),
            }[case]
            cycle = [SimpleNamespace(success=True), media_outcome]
            content = f"media payload\nMEDIA:{media}"
        outcomes = cycle * 2

        def schedule(coro, _loop):
            coro.close()
            future = concurrent.futures.Future()
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                future.set_exception(outcome)
            else:
                future.set_result(outcome)
            return future

        def live_send(_router, route_target, payload, metadata):
            text_calls.append((route_target.chat_id, payload, metadata))

            async def sent():
                return SimpleNamespace(success=True)

            return sent()

        cron_cfg: dict[str, object] = {"wrap_response": False}
        if case == "timeout-config":
            cron_cfg["media_send_timeout_seconds"] = sentinels[0]
        if case == "timeout-env":
            monkeypatch.setenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", sentinels[0])
        loop = MagicMock()
        loop.is_running.return_value = True
        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None,
        )
        monkeypatch.setattr(
            "tools.send_message_tool.resolve_send_target",
            lambda *_args, **_kwargs: (target, thread_id, None),
        )
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
        monkeypatch.setattr(
            "gateway.delivery.resolve_delivery_transport", lambda *_args: transport,
        )
        monkeypatch.setattr("gateway.delivery.DeliveryRouter._deliver_to_platform", live_send)
        monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
        monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": cron_cfg})
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        direct_error = sched._deliver_result(
            job, content, adapters={platform: adapter}, loop=loop,
        )
        captured = self._run_real_delivery_job(
            tmp_path, monkeypatch, job, final_response=content,
            adapters={platform: adapter}, loop=loop,
        )

        expected_failure = case in {"media-failure", "media-exception", "media-timeout"}
        assert direct_error == (sched._CRON_DELIVERY_FAILURE if expected_failure else None)
        assert [call[0] for call in text_calls] == [target, target]
        if case == "chat-probe":
            assert media_calls == []
        else:
            assert [call[0] for call in media_calls] == [target, target]
            assert all(call[1] == str(media) for call in media_calls)
        sentinels = sentinels + (job["id"], target, str(media))
        if expected_failure:
            self._assert_durable_delivery_failure(captured, caplog, sentinels)
        else:
            self._assert_private_free_success(captured, caplog, sentinels)
        self._assert_fixed_scheduler_logs(caplog)

    @pytest.mark.parametrize(
        "case",
        (
            "unconfigured", "relay", "thread-fallback", "live-error",
            "standalone-warning", "standalone-success", "pre-dispatch-timeout", "inflight-timeout",
        ),
    )
    def test_live_native_relay_and_fallback_routes_keep_control_data_internal(
        self, tmp_path, monkeypatch, caplog, case
    ):
        """The real result matrix preserves fallback/no-duplicate semantics."""
        from types import SimpleNamespace

        from gateway.config import Platform
        import cron.scheduler as sched

        sentinels = self._sentinels(f"result-{case}")
        private = "; ".join(sentinels[:-1])
        target = "result-target"
        thread_id = sentinels[0] if case == "thread-fallback" else None
        job = self._job(f"result-{case}-private")
        job["deliver"] = f"telegram:{target}" + (f":{thread_id}" if thread_id else "")
        pconfig = SimpleNamespace(enabled=case != "unconfigured", extra={})
        config = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig})
        adapter = object()
        sender_calls, live_calls = [], []

        async def standalone(_platform, _pconfig, chat_id, payload, **_kwargs):
            sender_calls.append((chat_id, payload))
            if case == "standalone-warning":
                return {"warnings": [private]}
            return {}

        def live_send(_router, route_target, payload, metadata):
            live_calls.append((route_target.chat_id, payload, metadata))

            async def sent():
                return SimpleNamespace(success=True)

            return sent()

        class TimedOutFuture:
            def __init__(self, cancel_result):
                self._cancel_result = cancel_result

            def result(self, timeout=None):
                raise TimeoutError(private)

            def cancel(self):
                return self._cancel_result

        def schedule(coro, _loop):
            coro.close()
            if case == "pre-dispatch-timeout":
                return TimedOutFuture(True)
            if case == "inflight-timeout":
                return TimedOutFuture(False)
            future = concurrent.futures.Future()
            if case == "live-error":
                future.set_result(SimpleNamespace(success=False, error=private, warnings=[private]))
            elif case == "thread-fallback":
                future.set_result(
                    {
                        "success": True,
                        "raw_response": {
                            "thread_fallback": True,
                            "requested_thread_id": private,
                        },
                    }
                )
            else:
                future.set_result(SimpleNamespace(success=True))
            return future

        if case == "relay":
            transport = SimpleNamespace(config=pconfig, adapter=None, is_relay=True)
            loop = None
        elif case in {"thread-fallback", "live-error", "pre-dispatch-timeout", "inflight-timeout"}:
            transport = SimpleNamespace(config=pconfig, adapter=adapter, is_relay=False)
            loop = MagicMock()
            loop.is_running.return_value = True
        else:
            transport = None
            loop = None
        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None,
        )
        monkeypatch.setattr(
            "tools.send_message_tool.resolve_send_target",
            lambda *_args, **_kwargs: (target, thread_id, None),
        )
        monkeypatch.setattr("tools.send_message_tool._send_to_platform", standalone)
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
        monkeypatch.setattr(
            "gateway.delivery.resolve_delivery_transport", lambda *_args: transport,
        )
        monkeypatch.setattr("gateway.delivery.DeliveryRouter._deliver_to_platform", live_send)
        monkeypatch.setattr("agent.async_utils.safe_schedule_threadsafe", schedule)
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"wrap_response": False}})
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        direct_error = sched._deliver_result(
            job, "result direct payload", adapters={Platform.TELEGRAM: adapter}, loop=loop,
        )
        captured = self._run_real_delivery_job(
            tmp_path, monkeypatch, job, adapters={Platform.TELEGRAM: adapter}, loop=loop,
        )

        expected_success = case in {
            "standalone-success", "live-error", "pre-dispatch-timeout", "inflight-timeout",
        }
        assert direct_error == (None if expected_success else sched._CRON_DELIVERY_FAILURE)
        if case in {"live-error", "pre-dispatch-timeout"}:
            assert [call[0] for call in sender_calls] == [target, target]
        elif case in {"relay", "thread-fallback", "inflight-timeout", "unconfigured"}:
            assert sender_calls == []
        else:
            assert [call[0] for call in sender_calls] == [target, target]
        if case in {"thread-fallback", "live-error", "pre-dispatch-timeout", "inflight-timeout"}:
            assert [call[0] for call in live_calls] == [target, target]
        sentinels = sentinels + (job["id"], target) + ((thread_id,) if thread_id else ())
        if expected_success:
            self._assert_private_free_success(captured, caplog, sentinels)
        else:
            self._assert_durable_delivery_failure(captured, caplog, sentinels)
        self._assert_fixed_scheduler_logs(caplog)

    @pytest.mark.parametrize("route", ("finalizing", "asyncio-run", "thread-fallback"))
    def test_shutdown_delivery_skips_are_generic_and_never_reach_a_sender(
        self, tmp_path, monkeypatch, caplog, route
    ):
        """All three real shutdown exits retain the generic delivery failure."""
        from types import SimpleNamespace

        from gateway.config import Platform
        import cron.scheduler as sched

        sentinels = self._sentinels(f"shutdown-{route}")
        private = "; ".join(sentinels[:-1])
        target = "shutdown-target"
        job = self._job(f"shutdown-{route}-private")
        job["deliver"] = f"telegram:{target}"
        pconfig = SimpleNamespace(enabled=True, extra={})
        config = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig})
        sent = []

        async def sender(_platform, _pconfig, chat_id, payload, **_kwargs):
            sent.append((chat_id, payload))
            return {}

        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None,
        )
        monkeypatch.setattr(
            "tools.send_message_tool.resolve_send_target",
            lambda *_args, **_kwargs: (target, None, None),
        )
        monkeypatch.setattr("tools.send_message_tool._send_to_platform", sender)
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"wrap_response": False}})
        if route == "finalizing":
            monkeypatch.setattr(sched.sys, "is_finalizing", lambda: True)
        elif route == "asyncio-run":
            monkeypatch.setattr(
                sched.asyncio, "run",
                lambda _coro: (_coro.close(), (_ for _ in ()).throw(
                    RuntimeError(f"cannot schedule new futures after interpreter shutdown: {private}")
                ))[1],
            )
        else:
            monkeypatch.setattr(
                sched.asyncio, "run",
                lambda _coro: (_coro.close(), (_ for _ in ()).throw(RuntimeError("running loop")))[1],
            )

            class Pool:
                def __init__(self, **_kwargs):
                    pass

                def submit(self, _fn, coro):
                    coro.close()
                    future = concurrent.futures.Future()
                    future.set_exception(
                        RuntimeError(f"cannot schedule new futures after interpreter shutdown: {private}")
                    )
                    return future

                def shutdown(self, **_kwargs):
                    pass

            monkeypatch.setattr(sched.concurrent.futures, "ThreadPoolExecutor", Pool)
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        direct_error = sched._deliver_result(job, "shutdown direct payload")
        captured = self._run_real_delivery_job(tmp_path, monkeypatch, job)

        assert direct_error == sched._CRON_DELIVERY_FAILURE
        assert sent == []
        self._assert_durable_delivery_failure(
            captured, caplog, sentinels + (job["id"], target),
        )
        self._assert_fixed_scheduler_logs(caplog)

    def test_caller_converts_resolution_and_execution_exceptions_before_any_durable_sink(
        self, tmp_path, monkeypatch, caplog
    ):
        """Normal and outer caller tails preserve their fixed public contracts."""
        from types import SimpleNamespace

        from gateway.config import Platform
        import cron.scheduler as sched

        sentinels = self._sentinels("caller-execution")
        private = "; ".join(sentinels[:-1])
        target = "caller-target"
        job = self._job("caller-execution-private")
        job["deliver"] = f"telegram:{target}"
        pconfig = SimpleNamespace(enabled=True, extra={})
        config = SimpleNamespace(platforms={Platform.TELEGRAM: pconfig})
        sent = []

        async def sender(_platform, _pconfig, chat_id, payload, **_kwargs):
            sent.append((chat_id, payload))
            return {}

        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None,
        )
        monkeypatch.setattr(
            "tools.send_message_tool.resolve_send_target",
            lambda *_args, **_kwargs: (target, None, None),
        )
        monkeypatch.setattr("tools.send_message_tool._send_to_platform", sender)
        monkeypatch.setattr("gateway.config.load_gateway_config", lambda: config)
        monkeypatch.setattr(sched, "load_config", lambda: {"cron": {"wrap_response": False}})
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        direct_error = sched._deliver_result(job, "caller direct payload")

        def raise_from_real_body(current_sched, _current_job, body):
            current_sched.run_job = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError(private)
            )
            return body(None)

        captured = self._run_real_delivery_job(
            tmp_path, monkeypatch, job, before_body=raise_from_real_body,
        )

        assert direct_error is None
        assert [call[0] for call in sent] == [target, target]
        assert captured["result"] is False
        assert captured["stored"]["last_error"] == sched._CRON_EXECUTION_FAILURE
        assert captured["stored"]["last_delivery_error"] is None
        assert captured["listed"]["latest_execution"]["error"] == sched._CRON_EXECUTION_FAILURE
        self._assert_no_private(
            (
                caplog.text,
                captured["stored"]["last_error"],
                captured["listed"]["latest_execution"]["error"],
            ),
            sentinels + (job["id"], target),
        )
        self._assert_fixed_scheduler_logs(caplog)

    def test_gateway_config_private_failure_is_fixed_at_real_delivery_boundary(
        self, tmp_path, monkeypatch, caplog
    ):
        """A loader exception cannot become a delivery_error durable value."""
        import cron.scheduler as sched

        sentinels = self._sentinels("gateway-config")
        private_message = "; ".join(sentinels[:-1])
        job = self._job("gateway-config-private")
        monkeypatch.setattr(
            sched, "_resolve_delivery_targets",
            lambda _job: [{"platform": "telegram", "chat_id": "one", "thread_id": None}],
        )
        caplog.set_level(logging.DEBUG, logger=sched.__name__)

        with patch(
            "gateway.config.load_gateway_config",
            side_effect=RuntimeError(private_message),
        ):
            direct_error = sched._deliver_result(job, "probe")
            captured = self._run_real_delivery_job(tmp_path, monkeypatch, job)

        assert direct_error == "Cron delivery failed"
        self._assert_durable_delivery_failure(captured, caplog, sentinels)

    def test_bot_chat_private_exception_is_fixed_through_real_helper_return(
        self, tmp_path, monkeypatch, caplog
    ):
        """The bot-chat catch/return path cannot aggregate a raw exception."""
        import cron.scheduler as sched

        sentinels = self._sentinels("bot-chat")
        job = self._job("bot-chat-private")
        monkeypatch.setattr(
            sched, "_resolve_delivery_targets",
            lambda _job: [{"platform": "bot-chat", "chat_id": "target-profile", "thread_id": None}],
        )
        monkeypatch.setattr("shutil.which", lambda _name: "/fake/hermes")
        caplog.set_level(logging.DEBUG, logger=sched.__name__)
        with patch("gateway.config.load_gateway_config", return_value=MagicMock()), \
             patch("subprocess.run", side_effect=RuntimeError("; ".join(sentinels[:-1]))) as send:
            direct_error = sched._deliver_result(job, "probe")
            send.reset_mock()
            captured = self._run_real_delivery_job(tmp_path, monkeypatch, job)

        assert direct_error == "Cron delivery failed"
        assert send.call_count == 1
        self._assert_durable_delivery_failure(captured, caplog, sentinels)

    def test_standalone_return_errors_are_fixed_and_preserve_two_target_iteration(
        self, tmp_path, monkeypatch, caplog
    ):
        """Raw result errors from both standalone targets collapse only at the boundary."""
        from gateway.config import Platform
        import cron.scheduler as sched

        sentinels = self._sentinels("standalone-return")
        sent_targets = []

        async def private_result(_platform, _pconfig, chat_id, *_args, **_kwargs):
            sent_targets.append(chat_id)
            return {"error": f"{'; '.join(sentinels[:-1])}; target={chat_id}"}

        pconfig = MagicMock(enabled=True)
        config = MagicMock()
        config.platforms = {Platform.TELEGRAM: pconfig}
        job = self._job("standalone-private")
        monkeypatch.setattr(
            sched, "_resolve_delivery_targets",
            lambda _job: [
                {"platform": "telegram", "chat_id": "one", "thread_id": None},
                {"platform": "telegram", "chat_id": "two", "thread_id": None},
            ],
        )
        monkeypatch.setattr("tools.send_message_tool._send_to_platform", private_result)
        caplog.set_level(logging.DEBUG, logger=sched.__name__)
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("gateway.delivery.resolve_delivery_transport", return_value=None):
            direct_error = sched._deliver_result(job, "probe")
            sent_targets.clear()
            captured = self._run_real_delivery_job(tmp_path, monkeypatch, job)

        assert direct_error == "Cron delivery failed"
        assert sent_targets == ["one", "two"]
        self._assert_durable_delivery_failure(captured, caplog, sentinels)

    def test_live_media_return_error_is_fixed_without_losing_partial_delivery(
        self, tmp_path, monkeypatch, caplog
    ):
        """A failed native attachment keeps text delivery but exports no adapter error."""
        from gateway.config import Platform
        import cron.scheduler as sched

        sentinels = self._sentinels("live-media")
        media = tmp_path / "clip.mp3"
        media.write_bytes(b"media")
        adapter = AsyncMock()
        adapter.send.return_value = MagicMock(success=True)
        adapter.send_voice.return_value = MagicMock(
            success=False, error="; ".join(sentinels[:-1]),
        )
        pconfig = MagicMock(enabled=True)
        config = MagicMock()
        config.platforms = {Platform.DISCORD: pconfig}
        loop = MagicMock()
        loop.is_running.return_value = True
        job = self._job("live-media-private")
        monkeypatch.setattr(
            sched, "_resolve_delivery_targets",
            lambda _job: [{"platform": "discord", "chat_id": "one", "thread_id": None}],
        )
        monkeypatch.setattr("gateway.platforms.base.MEDIA_DELIVERY_SAFE_ROOTS", (tmp_path,))

        def run_coro(coro, _loop):
            future = concurrent.futures.Future()
            try:
                future.set_result(asyncio.run(coro))
            except BaseException as exc:  # noqa: BLE001 - Future mirrors production transport.
                future.set_exception(exc)
            return future

        content = f"media response\nMEDIA:{media}"
        caplog.set_level(logging.DEBUG, logger=sched.__name__)
        with patch("gateway.config.load_gateway_config", return_value=config), \
             patch("cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}), \
             patch("asyncio.run_coroutine_threadsafe", side_effect=run_coro):
            direct_error = sched._deliver_result(
                job, content, adapters={Platform.DISCORD: adapter}, loop=loop,
            )
            adapter.reset_mock()
            captured = self._run_real_delivery_job(
                tmp_path, monkeypatch, job, final_response=content,
                adapters={Platform.DISCORD: adapter}, loop=loop,
            )

        assert direct_error == "Cron delivery failed"
        adapter.send.assert_awaited_once()
        adapter.send_voice.assert_awaited_once()
        self._assert_durable_delivery_failure(captured, caplog, sentinels)

    def test_interrupted_delivery_bookkeeping_logs_no_private_exception(
        self, tmp_path, monkeypatch, caplog
    ):
        """The interrupted receipt catch stays generic when persistence itself fails."""
        import cron.jobs as jobs
        import cron.scheduler as sched

        sentinels = self._sentinels("interrupted-bookkeeping")
        job = self._job("interrupted-bookkeeping-private")
        job["deliver"] = "not-a-platform:one"
        resolver_calls = []

        def resolve_target(platform, reference, *, pass_unresolved_references):
            resolver_calls.append((platform, reference, pass_unresolved_references))
            return "one", None, None

        monkeypatch.setattr(
            "tools.send_message_tool.prepare_send_message_platforms", lambda: None,
        )
        monkeypatch.setattr("tools.send_message_tool.resolve_send_target", resolve_target)
        monkeypatch.setattr(
            jobs, "update_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("; ".join(sentinels[:-1]))
            ),
        )

        # Drive the real interrupted completion branch without releasing the
        # active fire claim before its update_job bookkeeping attempt.
        monkeypatch.setattr(sched, "_consume_interrupted_flag", lambda *_args: True)

        caplog.set_level(logging.DEBUG, logger=sched.__name__)
        with patch("gateway.config.load_gateway_config", return_value=MagicMock()):
            direct_error = sched._deliver_result(job, "probe")
            captured = self._run_real_delivery_job(tmp_path, monkeypatch, job)

        assert direct_error == "Cron delivery failed"
        assert captured["result"] is True
        assert captured["marks"] == []
        assert len(captured["finishes"]) == 1
        assert captured["stored"].get("last_delivery_error") is None
        assert captured["listed"].get("last_delivery_error") is None
        assert captured["finishes"][0][1]["success"] is False
        assert sched._CRON_DELIVERY_FAILURE in caplog.text
        assert resolver_calls == [
            ("not-a-platform", "one", True),
            ("not-a-platform", "one", True),
        ]
        self._assert_no_private((caplog.text,), sentinels + (job["id"],))
        self._assert_fixed_scheduler_logs(caplog)


class TestTickFutureFailureRedaction:
    """Future completion is an outward boundary for worker exceptions."""

    class _ImmediateFuturePool:
        """Run submitted work now while preserving Future callback semantics."""

        def __init__(self):
            self.futures = []

        def submit(self, callback):
            future = concurrent.futures.Future()
            self.futures.append(future)
            try:
                future.set_result(callback())
            except BaseException as exc:
                future.set_exception(exc)
            return future

    def test_async_tick_redacts_resumed_baseexception_from_done_callback(
        self, tmp_path, monkeypatch, caplog
    ):
        """A resumed cancellation reaches the done callback but not its log."""
        import cron.jobs as jobs
        import cron.scheduler as sched
        from cron.executions import list_executions
        from tools import mcp_tool

        private_prompt_sentinel = "private-prompt-sentinel-future-93f1b7"
        credential_sentinel = "api_key=crn_test_future_credential_9f31b6c4a8e2"
        private_marker = "## Private Cron Future Context"
        occurrence_ids = []
        deliveries = []
        mark_calls = []
        sweeps = []
        pool = self._ImmediateFuturePool()
        now = jobs._hermes_now().isoformat()
        job = {
            "id": "resume-future-private",
            "name": "resume future private",
            "prompt": "run future boundary test",
            "schedule": {"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "deliver": "origin",
            "run_claim": {"at": now, "by": "due-scan-owner"},
        }

        def raise_private_cancelled_error(*_args, occurrence_id=None, **_kwargs):
            assert isinstance(occurrence_id, str) and occurrence_id
            occurrence_ids.append(occurrence_id)
            private_traceback_context = (
                f"{private_prompt_sentinel}\n{credential_sentinel}\n"
                f"{private_marker}\noccurrence={occurrence_id}"
            )
            raise asyncio.CancelledError(
                f"cancelled body: {private_traceback_context}"
            )

        real_mark = sched.mark_job_run

        def capture_mark(*args, **kwargs):
            mark_calls.append((args, kwargs))
            return real_mark(*args, **kwargs)

        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(sched, "get_due_jobs", lambda: [dict(job)])
        monkeypatch.setattr(sched, "_get_parallel_pool", lambda _workers: pool)
        monkeypatch.setattr(sched, "load_config", lambda: {})
        monkeypatch.setattr(sched, "_last_dead_owner_reap_at", sched.time.monotonic())
        monkeypatch.setattr(sched, "run_job", raise_private_cancelled_error)
        monkeypatch.setattr(sched, "mark_job_run", capture_mark)
        monkeypatch.setattr(
            sched,
            "_deliver_result",
            lambda _job, content, **_kwargs: deliveries.append(content),
        )
        monkeypatch.setattr(
            mcp_tool,
            "_kill_orphaned_mcp_children",
            lambda: sweeps.append("swept"),
        )
        caplog.set_level(logging.ERROR, logger=sched.__name__)

        with jobs.use_cron_store(tmp_path), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"):
            jobs.save_jobs([job])
            assert sched.tick(verbose=False, sync=False) == 1
            stored = jobs.get_job(job["id"])
            execution_rows = list_executions(job_id=job["id"])

        assert len(pool.futures) == 1
        assert isinstance(pool.futures[0].exception(), asyncio.CancelledError)
        assert len(occurrence_ids) == 1
        occurrence_id = occurrence_ids[0]
        assert stored is not None
        assert stored["last_error"] == sched._RESUMED_CRON_EXECUTION_FAILURE
        assert stored["fire_claim"] is None
        assert stored["run_claim"] is None
        assert stored.get("resume_reservation") is None
        assert len(mark_calls) == 1
        mark_args, mark_kwargs = mark_calls[0]
        assert mark_args == (
            job["id"],
            False,
            sched._RESUMED_CRON_EXECUTION_FAILURE,
        )
        assert mark_kwargs["expected_fire_owner"]
        assert mark_kwargs["expected_resume_owner"] == mark_kwargs["expected_fire_owner"]
        assert len(execution_rows) == 1
        assert execution_rows[0]["error"] == sched._RESUMED_CRON_EXECUTION_FAILURE
        assert deliveries == []
        assert sweeps == ["swept"]
        assert "Cron job future failed in async mode" in caplog.text
        for private_text in (
            private_prompt_sentinel,
            credential_sentinel,
            private_marker,
            occurrence_id,
        ):
            assert private_text not in caplog.text
        async_records = [
            record
            for record in caplog.records
            if record.getMessage() == "Cron job future failed in async mode"
        ]
        assert len(async_records) == 1
        assert async_records[0].exc_info is None

    def test_sync_tick_redacts_arbitrary_future_exception(self, monkeypatch, caplog):
        """The sync result-consumption sibling has the same safe log boundary."""
        import cron.scheduler as sched
        from tools import mcp_tool

        private_prompt_sentinel = "private-prompt-sentinel-sync-future-93f1b7"
        credential_sentinel = "api_key=crn_test_sync_future_credential_9f31b6c4a8e2"
        private_marker = "## Private Sync Future Context"
        sweeps = []
        pool = self._ImmediateFuturePool()
        job = {"id": "sync-future-private", "name": "sync future private"}

        def raise_private_future_error(*_args, **_kwargs):
            private_traceback_context = (
                f"{private_prompt_sentinel}\n{credential_sentinel}\n{private_marker}"
            )
            raise RuntimeError(f"future failed: {private_traceback_context}")

        monkeypatch.setattr(sched, "get_due_jobs", lambda: [job])
        monkeypatch.setattr(sched, "advance_next_runs", lambda _job_ids: 0)
        monkeypatch.setattr(sched, "create_execution", lambda *_args, **_kwargs: {"id": "sync-execution"})
        monkeypatch.setattr(
            sched,
            "claim_job_for_fire",
            lambda _job_id, **_kwargs: dict(job, fire_claim={"by": "sync-owner"}),
        )
        monkeypatch.setattr(sched, "run_one_job", raise_private_future_error)
        monkeypatch.setattr(sched, "_get_parallel_pool", lambda _workers: pool)
        monkeypatch.setattr(sched, "load_config", lambda: {})
        monkeypatch.setattr(sched, "_last_dead_owner_reap_at", sched.time.monotonic())
        monkeypatch.setattr(
            mcp_tool,
            "_kill_orphaned_mcp_children",
            lambda: sweeps.append("swept"),
        )
        caplog.set_level(logging.ERROR, logger=sched.__name__)

        assert sched.tick(verbose=False, sync=True) == 0

        assert len(pool.futures) == 1
        assert isinstance(pool.futures[0].exception(), RuntimeError)
        assert sweeps == ["swept"]
        assert "Cron job future failed" in caplog.text
        for private_text in (
            private_prompt_sentinel,
            credential_sentinel,
            private_marker,
        ):
            assert private_text not in caplog.text
        sync_records = [
            record
            for record in caplog.records
            if record.getMessage() == "Cron job future failed"
        ]
        assert len(sync_records) == 1
        assert sync_records[0].exc_info is None


class TestCallerLossAfterClaimAcquisition:
    """cirwel's integration assertion on #70638: if the HTTP/CLI caller is
    lost AFTER the claim was acquired, the gateway owner must produce at
    most one terminal ledger/artifact/delivery, clear only its own claim,
    and block retries while that ownership is live."""

    def test_second_fire_cannot_claim_while_first_ownership_live(self, tmp_path):
        import cron.jobs as jobs

        with jobs.use_cron_store(tmp_path):
            job = jobs.create_job(prompt="x", schedule="every 5m", name="owned")
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)

            # Caller died here — the claim outlives it. A retry (NAS/webhook
            # or manual) must be refused while the lease is fresh.
            retry = jobs.claim_job_for_fire(job["id"], return_job=True)
            assert retry is False or not isinstance(retry, dict)

            # The live owner still heartbeats and terminally marks — exactly
            # one terminal write, and only its own claim is cleared.
            owner = claimed["fire_claim"]["by"]
            assert jobs.heartbeat_fire_claim(job["id"], expected_owner=owner) is True
            assert jobs.mark_job_run(
                job["id"], True, expected_fire_owner=owner,
            ) is True
            refreshed = jobs.get_job(job["id"])
            assert refreshed["fire_claim"] is None
            assert refreshed["last_status"] == "ok"


class TestRunOneJobHonoursInterruptedFlag:
    """run_one_job() must not let a job's own completion overwrite a
    status the shutdown path already wrote for the same run."""

    def _make_job(self, job_id="job-1"):
        return {"id": job_id, "name": "test job", "prompt": "do work"}

    def test_success_path_skipped_when_interrupted(self):
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is True
        # The would-be "success" write must NOT happen -- the shutdown
        # path already wrote the authoritative interrupted status.
        mock_mark.assert_not_called()
        # Flag is consumed so a later, unrelated fire of the same job ID
        # isn't permanently silenced.
        assert job["id"] not in sched._interrupted_job_ids

    def test_interrupted_job_delivers_failure_summary_not_raw_response(self):
        """The status-write guard alone isn't enough: delivery happens
        BEFORE mark_job_run in run_one_job's own flow, so a job that kept
        running post-kill and produced a plausible-looking final_response
        must not have that response sent to the user just because the
        eventual status write gets suppressed. Interrupted jobs must route
        through the same failure-summary delivery path a real failure
        would."""
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch(
                 "cron.scheduler.run_job",
                 return_value=(True, "full output", "a plausible final response", None),
             ), \
             patch("cron.scheduler.save_job_output", return_value="/tmp/out.md"), \
             patch("cron.scheduler._is_cron_silence_response", return_value=False), \
             patch("cron.scheduler._deliver_result", return_value=None) as mock_deliver, \
             patch("cron.scheduler.mark_job_run"):
            result = sched.run_one_job(job)

        assert result is True
        mock_deliver.assert_called_once()
        delivered_content = mock_deliver.call_args.args[1]
        assert delivered_content == "Cron delivery failed"
        assert "plausible final response" not in delivered_content


    def test_exception_path_also_honours_interrupted_flag(self):
        import cron.scheduler as sched

        job = self._make_job()
        sched._interrupted_job_ids.add(job["id"])

        with patch("cron.scheduler.claim_dispatch", return_value=True), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("cron.scheduler.run_job", side_effect=RuntimeError("boom")), \
             patch("cron.scheduler.mark_job_run") as mock_mark:
            result = sched.run_one_job(job)

        assert result is False
        mock_mark.assert_not_called()


class TestResumeOneShotAdmission:
    """The real ``run_one_job`` caller must rearm an opted-in occurrence
    instead of starting any body work once a drain arrives after its claim.
    """

    @pytest.mark.parametrize("body_kind", ("agent", "script"))
    def test_claim_then_drain_rearms_before_agent_or_script_body(
        self, tmp_path, monkeypatch, body_kind
    ):
        import cron.jobs as jobs
        import cron.scheduler as sched

        job = {
            "id": f"resume-admission-{body_kind}",
            "name": "resume admission",
            "prompt": "do work",
            "schedule": {
                "kind": "once",
                "run_at": "2026-09-01T12:00:00+00:00",
                "display": "once",
            },
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "deliver": "local",
            "execution_id": f"execution-{body_kind}",
        }
        if body_kind == "script":
            job.update({"no_agent": True, "script": "probe.sh"})

        with jobs.use_cron_store(tmp_path):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(
                job["id"], force=True, return_job=True
            )
            assert isinstance(claimed, dict)

            drain = threading.Event()
            real_claim_dispatch = sched.claim_dispatch

            def claim_then_drain(job_id):
                assert real_claim_dispatch(job_id) is True
                drain.set()
                return True

            monkeypatch.setattr(sched, "claim_dispatch", claim_then_drain)
            monkeypatch.setattr(sched, "mark_execution_running", lambda *_args: None)
            monkeypatch.setattr(sched, "finish_execution", lambda *_args, **_kwargs: None)

            if body_kind == "agent":
                body = patch(
                    "cron.scheduler.run_job",
                    return_value=(True, "agent output", "agent response", None),
                )
            else:
                body = patch(
                    "cron.scheduler._run_job_script_with_claim_heartbeat",
                    return_value=(True, "script output"),
                )

            with body as fake_body, \
                 patch("agent.secret_scope.set_secret_scope", return_value=None), \
                 patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
                 patch("agent.secret_scope.reset_secret_scope"):
                assert sched.run_one_job(claimed, cancel_event=drain) is True

            fake_body.assert_not_called()
            rearmed = jobs.get_job(job["id"])

        assert rearmed is not None
        reservation = rearmed["resume_reservation"]
        assert rearmed["repeat"]["completed"] == 0
        assert rearmed["state"] == "retry_pending"
        assert rearmed["enabled"] is True
        assert reservation["state"] == "retry_pending"
        assert reservation["owner"] is None
        assert isinstance(reservation["occurrence_id"], str)
        assert reservation["occurrence_id"]
        assert rearmed["next_run_at"] == reservation["retry_not_before"]

    def test_drain_just_before_body_rearms_without_calling_agent(self, tmp_path, monkeypatch):
        import cron.jobs as jobs
        import cron.scheduler as sched

        job = {
            "id": "resume-pre-body-drain",
            "name": "resume pre body",
            "prompt": "do work",
            "schedule": {"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "deliver": "local",
            "execution_id": "execution-pre-body",
        }
        with jobs.use_cron_store(tmp_path):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            drain = threading.Event()
            fake_agent = patch(
                "cron.scheduler.run_job",
                return_value=(True, "agent output", "agent response", None),
            )
            monkeypatch.setattr(sched, "mark_execution_running", lambda *_args: drain.set())
            monkeypatch.setattr(sched, "finish_execution", lambda *_args, **_kwargs: None)

            with fake_agent as run_agent, \
                 patch("agent.secret_scope.set_secret_scope", return_value=None), \
                 patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
                 patch("agent.secret_scope.reset_secret_scope"):
                assert sched.run_one_job(claimed, cancel_event=drain) is True

            run_agent.assert_not_called()
            rearmed = jobs.get_job(job["id"])

        assert rearmed is not None
        assert rearmed["repeat"]["completed"] == 0
        assert rearmed["resume_reservation"]["state"] == "retry_pending"
        assert rearmed["next_run_at"] == rearmed["resume_reservation"]["retry_not_before"]

    def test_forced_body_interruption_rearms_and_fences_stale_settlement(
        self, tmp_path, monkeypatch
    ):
        from datetime import datetime

        import cron.jobs as jobs
        import cron.scheduler as sched

        job = {
            "id": "resume-forced-interruption",
            "name": "resume forced interruption",
            "prompt": "do work",
            "schedule": {"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "deliver": "local",
            "execution_id": "execution-forced-interruption",
        }
        with jobs.use_cron_store(tmp_path):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            drain = threading.Event()
            first_owner = []
            real_claim_dispatch = sched.claim_dispatch

            def capture_claim(job_id):
                assert real_claim_dispatch(job_id) is True
                first_owner.append(jobs.get_job(job_id)["resume_reservation"]["owner"])
                return True

            def body_started(*_args, **_kwargs):
                drain.set()
                return True, "body output", "body response", None

            monkeypatch.setattr(sched, "claim_dispatch", capture_claim)
            monkeypatch.setattr(sched, "run_job", body_started)
            monkeypatch.setattr(sched, "save_job_output", lambda *_args: "output.md")
            monkeypatch.setattr(sched, "finish_execution", lambda *_args, **_kwargs: None)

            with patch("agent.secret_scope.set_secret_scope", return_value=None), \
                 patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
                 patch("agent.secret_scope.reset_secret_scope"):
                assert sched.run_one_job(claimed, cancel_event=drain) is True

            rearmed = jobs.get_job(job["id"])
            assert rearmed is not None
            assert rearmed["repeat"]["completed"] == 0
            assert rearmed["resume_reservation"]["state"] == "retry_pending"
            assert rearmed["resume_reservation"]["occurrence_id"]

            retry_at = datetime.fromisoformat(rearmed["next_run_at"])
            monkeypatch.setattr(jobs, "_hermes_now", lambda: retry_at)
            retry_claimed = jobs.claim_job_for_fire(
                job["id"], force=True, return_job=True
            )
            assert isinstance(retry_claimed, dict)
            assert jobs.claim_dispatch(job["id"], claimed_job=retry_claimed) is True
            retry = jobs.get_job(job["id"])
            assert retry is not None
            retry_owner = retry["resume_reservation"]["owner"]
            assert retry_owner != first_owner[0]
            assert jobs.mark_job_run(
                job["id"], True, expected_resume_owner=first_owner[0]
            ) is False
            assert jobs.get_job(job["id"])["repeat"]["completed"] == 0
            assert jobs.mark_job_run(
                job["id"], True, expected_resume_owner=retry_owner
            ) is True
            settled = jobs.get_job(job["id"])

        assert settled is not None
        assert settled["repeat"]["completed"] == 1
        assert settled["state"] == "completed"

    def test_shutdown_snapshot_without_resume_owner_cannot_terminalize_reservation(
        self, tmp_path, monkeypatch
    ):
        """The durable reservation owner fences the post-claim registration race."""
        import cron.jobs as jobs
        import cron.scheduler as sched

        job = {
            "id": "resume-shutdown-owner-race",
            "name": "resume shutdown owner race",
            "prompt": "do work",
            "schedule": {"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "deliver": "local",
            "execution_id": "execution-shutdown-owner-race",
        }
        drain = threading.Event()
        shutdown_snapshot = []
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)

        with jobs.use_cron_store(tmp_path):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            fire_owner = claimed["fire_claim"]["by"]
            real_get_job = sched.get_job

            def snapshot_between_dispatch_and_registration(job_id):
                if job_id == job["id"] and not shutdown_snapshot:
                    # This is the exact window: claim_dispatch has durably
                    # created a running reservation, while the in-memory
                    # registration is still the old two-field shape.
                    sched.mark_running_jobs_interrupted("gateway shutdown")
                    shutdown_snapshot.append(jobs.get_job(job_id))
                    drain.set()
                return real_get_job(job_id)

            def body_must_not_run(*_args, **_kwargs):
                pytest.fail("shutdown drain must settle before body execution")

            monkeypatch.setattr(sched, "get_job", snapshot_between_dispatch_and_registration)
            monkeypatch.setattr(sched, "run_job", body_must_not_run)
            monkeypatch.setattr(sched, "finish_execution", lambda *_args, **_kwargs: None)

            assert sched.run_one_job(claimed, cancel_event=drain) is True
            rearmed = jobs.get_job(job["id"])

        assert shutdown_snapshot
        during_unowned_settlement = shutdown_snapshot[0]
        assert during_unowned_settlement is not None
        reservation = during_unowned_settlement["resume_reservation"]
        assert during_unowned_settlement["repeat"]["completed"] == 0
        assert during_unowned_settlement["enabled"] is True
        assert during_unowned_settlement["state"] == "scheduled"
        assert during_unowned_settlement["fire_claim"]["by"] == fire_owner
        assert reservation["state"] == "running"
        assert reservation["owner"] == fire_owner

        assert rearmed is not None
        assert rearmed["repeat"]["completed"] == 0
        assert rearmed["enabled"] is True
        assert rearmed["state"] == "retry_pending"
        assert rearmed["resume_reservation"]["state"] == "retry_pending"
        assert rearmed["resume_reservation"]["owner"] is None
        assert rearmed["run_claim"] is None
        assert rearmed["fire_claim"] is None

    def test_resume_occurrence_reaches_script_env_but_not_public_artifacts(
        self, tmp_path, monkeypatch, caplog
    ):
        import cron.jobs as jobs
        import cron.scheduler as sched

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "probe.sh").write_text("# test only\n", encoding="utf-8")
        job = {
            "id": "resume-private-script",
            "name": "resume private script",
            "prompt": "private script test",
            "schedule": {"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "no_agent": True,
            "script": "probe.sh",
            "deliver": "origin",
            "execution_id": "execution-private-script",
        }
        captured_env = []
        private_occurrence = []
        public_output = []
        delivery = []
        unrelated_hex = "deadbeefcafebabe"

        class FakePopen:
            def __init__(self, *_args, **kwargs):
                captured_env.append(kwargs["env"])
                self.env = kwargs["env"]
                self.returncode = 0

            def communicate(self, timeout):
                return (
                    "script echoed "
                    f"{self.env['HERMES_CRON_OCCURRENCE_ID']} "
                    f"unrelated={unrelated_hex}",
                    "",
                )

        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(sched.subprocess, "Popen", FakePopen)
        monkeypatch.setattr(
            sched, "save_job_output", lambda _job_id, output: public_output.append(output) or "out.md"
        )
        monkeypatch.setattr(
            sched, "_deliver_result", lambda _job, content, **_kwargs: delivery.append(content)
        )
        monkeypatch.setattr(sched, "finish_execution", lambda *_args, **_kwargs: None)

        with jobs.use_cron_store(tmp_path):
            jobs.save_jobs([job])
            claimed = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(claimed, dict)
            real_claim_dispatch = sched.claim_dispatch

            def capture_claim(job_id):
                assert real_claim_dispatch(job_id) is True
                private_occurrence.append(
                    jobs.get_job(job_id)["resume_reservation"]["occurrence_id"]
                )
                return True

            monkeypatch.setattr(sched, "claim_dispatch", capture_claim)
            assert sched.run_one_job(claimed) is True
            completed = jobs.get_job(job["id"])

        assert completed is not None
        # Completion removes the reservation, so bind the stable identifier at
        # the private script boundary rather than inventing one post-settlement.
        assert private_occurrence
        occurrence_id = private_occurrence[0]
        assert captured_env
        assert captured_env[0]["HERMES_CRON_OCCURRENCE_ID"] == occurrence_id
        assert occurrence_id
        assert all(occurrence_id not in text for text in public_output)
        assert all(occurrence_id not in text for text in delivery)
        assert occurrence_id not in caplog.text
        assert all(unrelated_hex in text for text in (*public_output, *delivery))

    def test_resume_agent_retries_with_private_stable_occurrence_only(
        self, tmp_path, monkeypatch, caplog
    ):
        """A resumed agent retry receives one private occurrence ID, never public metadata."""
        import json
        from datetime import datetime

        import cron.jobs as jobs
        import cron.scheduler as sched

        public_prompt = "Summarize the release notes."
        private_marker = "## Private Cron Execution Context"
        unrelated_hex = "0123456789abcdef"
        prompts = []
        public_output = []
        delivery = []
        drain = threading.Event()
        runtime = {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "provider": "openrouter",
            "api_mode": "chat_completions",
        }

        class CapturingAgent:
            def run_conversation(self, prompt):
                prompts.append(prompt)
                if len(prompts) == 1:
                    drain.set()
                    return {"final_response": "safe agent response"}
                if len(prompts) == 2:
                    # A failed body can put its private prompt in the error
                    # response too; the outward log/error path must redact it.
                    return {"failed": True, "final_response": prompt}
                return {
                    "final_response": (
                        f"{prompt}\nnegative-control={unrelated_hex}"
                    )
                }

        job = {
            "id": "resume-private-agent",
            "name": "resume private agent",
            "prompt": public_prompt,
            "schedule": {"kind": "once", "run_at": "2026-09-01T12:00:00+00:00"},
            "schedule_display": "once",
            "next_run_at": "2026-09-01T12:00:00+00:00",
            "repeat": {"times": 1, "completed": 0},
            "restart_policy": "resume",
            "enabled": True,
            "state": "scheduled",
            "deliver": "origin",
            "origin": {"platform": "telegram", "chat_id": "test-chat"},
            "execution_id": "execution-private-agent",
            "model": "test/model",
        }

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(sched, "_hermes_home", tmp_path)
        monkeypatch.setattr(sched, "mark_execution_running", lambda *_args: None)
        monkeypatch.setattr(sched, "finish_execution", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            sched, "save_job_output", lambda _job_id, output: public_output.append(output) or "out.md"
        )
        monkeypatch.setattr(
            sched, "_deliver_result", lambda _job, content, **_kwargs: delivery.append(content)
        )
        monkeypatch.setattr(sched, "_teardown_cron_agent", lambda *_args: None)

        with jobs.use_cron_store(tmp_path), \
             patch("agent.secret_scope.set_secret_scope", return_value=None), \
             patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
             patch("agent.secret_scope.reset_secret_scope"), \
             patch("hermes_cli.env_loader.load_hermes_dotenv"), \
             patch("hermes_cli.env_loader.reset_secret_source_cache"), \
             patch("hermes_state.SessionDB"), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime), \
             patch("tools.mcp_tool.discover_mcp_tools", return_value=[]), \
             patch("run_agent.AIAgent", return_value=CapturingAgent()), \
             patch("cron.scheduler._cron_preflight_enabled", return_value=False):
            jobs.save_jobs([job])
            first = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(first, dict)
            assert sched.run_one_job(first, cancel_event=drain) is True

            pending = jobs.get_job(job["id"])
            assert pending is not None
            occurrence_id = pending["resume_reservation"]["occurrence_id"]
            assert pending["resume_reservation"]["state"] == "retry_pending"
            assert pending["prompt"] == public_prompt

            retry_at = datetime.fromisoformat(pending["next_run_at"])
            monkeypatch.setattr(jobs, "_hermes_now", lambda: retry_at)
            retry = jobs.claim_job_for_fire(job["id"], force=True, return_job=True)
            assert isinstance(retry, dict)
            assert sched.run_one_job(retry) is True
            completed = jobs.get_job(job["id"])
            assert completed is not None

            terminal_public_prompt = "Terminal prompt stays public."
            terminal_agent = {
                **job,
                "id": "terminal-private-agent-control",
                "name": "terminal private agent control",
                "prompt": terminal_public_prompt,
                "restart_policy": "terminal",
                "execution_id": "execution-terminal-agent-control",
            }
            jobs.save_jobs([completed, terminal_agent])
            terminal_claim = jobs.claim_job_for_fire(
                terminal_agent["id"], force=True, return_job=True
            )
            assert isinstance(terminal_claim, dict)
            assert sched.run_one_job(terminal_claim) is True
            terminal_completed = jobs.get_job(terminal_agent["id"])
            assert terminal_completed is not None

            script_env = []

            class TerminalPopen:
                def __init__(self, *_args, **kwargs):
                    script_env.append(kwargs["env"])
                    self.returncode = 0

                def communicate(self, timeout):
                    return "safe terminal script result", ""

            (tmp_path / "scripts").mkdir(exist_ok=True)
            (tmp_path / "scripts" / "terminal.sh").write_text(
                "# test only\n", encoding="utf-8"
            )
            monkeypatch.setattr(sched.subprocess, "Popen", TerminalPopen)
            terminal_script = {
                **terminal_agent,
                "id": "terminal-private-script-control",
                "name": "terminal private script control",
                "prompt": "Terminal script stays public.",
                "no_agent": True,
                "script": "terminal.sh",
                "execution_id": "execution-terminal-script-control",
                "deliver": "local",
            }
            jobs.save_jobs([completed, terminal_completed, terminal_script])
            terminal_script_claim = jobs.claim_job_for_fire(
                terminal_script["id"], force=True, return_job=True
            )
            assert isinstance(terminal_script_claim, dict)
            assert sched.run_one_job(terminal_script_claim) is True
            terminal_script_completed = jobs.get_job(terminal_script["id"])

        assert completed is not None
        assert terminal_completed is not None
        assert terminal_script_completed is not None
        assert completed["state"] == "completed"
        assert terminal_completed["state"] == "completed"
        assert terminal_script_completed["state"] == "completed"
        assert completed["repeat"]["completed"] == 1
        assert completed["prompt"] == public_prompt
        assert occurrence_id
        assert len(prompts) == 3
        resume_prompts = prompts[:2]
        assert all(occurrence_id in prompt for prompt in resume_prompts)
        assert all(private_marker in prompt for prompt in resume_prompts)
        assert all(public_prompt in prompt for prompt in resume_prompts)
        assert terminal_public_prompt in prompts[2]
        assert occurrence_id not in prompts[2]
        assert private_marker not in prompts[2]
        assert script_env and "HERMES_CRON_OCCURRENCE_ID" not in script_env[0]
        assert public_output and delivery
        for public_text in (*public_output, *delivery, caplog.text, json.dumps(completed)):
            assert occurrence_id not in public_text
            assert private_marker not in public_text
        assert any(unrelated_hex in text for text in (*public_output, *delivery))
