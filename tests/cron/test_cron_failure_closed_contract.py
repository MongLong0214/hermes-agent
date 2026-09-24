"""Closed cron failure contract: a failed run's raw error stays in the host's private stores.

A cron failure string carries provider response bodies, filesystem paths, secret-like tokens and
script stderr tails. The chat notice and the job record (``last_error`` /
``last_delivery_error``, which the cronjob tool, the dashboard and ``/api/jobs`` serve) must carry
none of it: they are closed labels. The raw text remains where the operator diagnoses from, the
run's executions row (``hermes cron runs``).

These drive the real store under a temp HERMES_HOME and the real ``run_one_job`` body; only the
agent run and the platform send are stubbed.
"""

import json
import time

import pytest

_PATH = "/Users/alice/private/leakcanary-7f3a/app.py"
_TOKEN = "sk-leakcanary0123456789abcdefghijklmnop"
_STDERR = "Traceback (most recent call last): KeyError: 'leakcanary-stderr-91c2'"
_CANARY = "leakcanary"

_RAW_ERRORS = {
    "unclassified": f"RuntimeError: open('{_PATH}') failed with {_TOKEN}; stderr: {_STDERR}",
    "provider": f"HTTP 401: invalid api key {_TOKEN} (config {_PATH}) {_STDERR}",
    "script-timeout": f"Script timed out after 30s: {_PATH}",
    "blocked-config": (
        f"[blocked_config] provider credential missing: rejected {_TOKEN} "
        f"[profile 'default', HERMES_HOME {_PATH}]. Set the provider API key."),
}


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_running_state():
    import cron.scheduler as sched

    for registry in (sched._running_job_ids, sched._running_fire_owners, sched._interrupted_job_ids):
        registry.clear()
    yield
    for registry in (sched._running_job_ids, sched._running_fire_owners, sched._interrupted_job_ids):
        registry.clear()


def _claimed_job():
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="closed-contract")
    assert claim_job_for_fire(job["id"]) is True
    return get_job(job["id"])


# The pre-run check returns its verdict and never raises, so it has no "raised" row.
_RUN_CASES = [(shape, mode) for shape in sorted(_RAW_ERRORS) for mode in ("returned", "raised")
              if not (shape == "blocked-config" and mode == "raised")]


@pytest.mark.parametrize("shape, mode", _RUN_CASES)
def test_failed_run_raw_error_reaches_only_the_executions_row(temp_home, monkeypatch, shape, mode):
    import cron.scheduler as sched
    from cron.executions import get_execution
    from cron.jobs import get_job

    raw = _RAW_ERRORS[shape]
    job = _claimed_job()
    delivered = []

    def fake_run_job(job, **kwargs):
        if mode == "raised":
            raise RuntimeError(raw)
        return False, f"# output\n\n{raw}\n", "", raw

    def failing_send(job, content, **kwargs):
        delivered.append(content)
        return f"live adapter send failed: 502 from {_PATH}; stderr: {_STDERR}"

    monkeypatch.setattr(sched, "run_job", fake_run_job)
    monkeypatch.setattr(sched, "_deliver_result", failing_send)

    sched.run_one_job(job)

    record = get_job(job["id"])
    assert delivered, "the failure notice must have been sent"
    assert record["last_status"] in {"error", "blocked_config"}
    assert record["last_error"], "the record must still say the run failed"
    assert record["last_delivery_error"], "the record must still say delivery failed"
    for surface, text in (
        ("chat notice", delivered[0]),
        ("last_error", record["last_error"]),
        ("last_delivery_error", record["last_delivery_error"]),
    ):
        assert _CANARY not in text, f"{surface} leaked raw failure text: {text!r}"
        assert _TOKEN not in text and _PATH not in text, f"{surface}: {text!r}"
        # A home outside the user's is named, not printed: the chat need not be the operator.
        assert str(temp_home) not in text, f"{surface} carries the host path: {text!r}"
    # Closed, not deleted: the operator's run history keeps the full text.
    assert _PATH in get_execution(job["execution_id"])["error"]


def _fire_forward_failure(job_id, monkeypatch):
    from cron.jobs import get_job, note_fire_forward_failure
    from tools.cronjob_tools import _format_job

    assert note_fire_forward_failure(job_id, f"forward to {_PATH} failed; stderr: {_STDERR}")
    record = get_job(job_id)
    assert record["last_fire_error"]["at"], "the miss must still be stamped"
    return [json.dumps(record["last_fire_error"]), json.dumps(_format_job(record)["last_fire_error"])]


def _manual_claim_failure(job_id, monkeypatch):
    import tools.cronjob_tools as cronjob_tools
    from cron.jobs import get_job

    def failing_claim(*args, **kwargs):
        raise RuntimeError(_RAW_ERRORS["unclassified"])

    monkeypatch.setattr(cronjob_tools, "claim_job_for_fire", failing_claim)
    claimed, result = cronjob_tools._claim_for_manual_run(job_id, "manual run")
    assert claimed is None and result["success"] is False and result["error"]
    return [result["error"], get_job(job_id)["last_error"]]


def _background_run_completion(job_id, monkeypatch):
    from cron.jobs import get_job, mark_job_run, save_job_output
    from tools.cronjob_tools import _manual_run_completion

    raw = _RAW_ERRORS["unclassified"]
    save_job_output(job_id, f"# closed-contract (FAILED)\n\n## Error\n\n{raw}\n")
    mark_job_run(job_id, False, raw)
    result = {"claimed": True, "success": False, "error": get_job(job_id)["last_error"]}
    block = _manual_run_completion(result, job_id, "closed-contract", "telegram", time.time())
    return [block["summary"], str(block["error"])]


@pytest.mark.parametrize("surface", [_fire_forward_failure, _manual_claim_failure,
                                     _background_run_completion], ids=lambda f: f.__name__)
def test_failures_outside_a_scheduled_run_are_stored_and_relayed_closed(
        temp_home, monkeypatch, surface):
    """The other writers of a job's failure fields, and the manual-run result the calling agent
    relays to chat, carry the same closed text as a scheduled run's failure."""
    from cron.jobs import create_job

    job = create_job(prompt="x", schedule="every 5m", name="closed-contract")
    for text in surface(job["id"], monkeypatch):
        assert _CANARY not in text and _TOKEN not in text, f"raw failure text: {text!r}"
        assert str(temp_home) not in text, f"host path: {text!r}"


def test_listing_serves_a_record_written_before_closure_closed(temp_home):
    """A job whose record still holds raw text (written by an older build) is listed with the
    same closed labels a fresh failure gets."""
    from cron.jobs import create_job, get_job, update_job
    from tools.cronjob_tools import _format_job

    job = create_job(prompt="x", schedule="every 5m", name="legacy-record")
    raw_error, raw_delivery = _RAW_ERRORS["unclassified"], f"stderr: {_STDERR} at {_PATH}"
    raw_fire = {"at": "2026-09-24T00:00:00+00:00", "detail": f"POST {_PATH} failed: {_STDERR}"}
    update_job(job["id"], {"last_status": "error", "last_error": raw_error,
                           "last_delivery_error": raw_delivery, "last_fire_error": raw_fire})

    listed = _format_job(get_job(job["id"]))

    assert _CANARY not in json.dumps(
        [listed["last_error"], listed["last_delivery_error"], listed["last_fire_error"]])
    from cron.jobs_public_status import public_delivery_error, public_fire_error, public_run_error

    assert listed["last_error"] == public_run_error(raw_error)
    assert listed["last_delivery_error"] == public_delivery_error(raw_delivery)
    assert listed["last_fire_error"] == public_fire_error(raw_fire)
    assert listed["last_fire_error"]["at"] == raw_fire["at"], "the miss keeps its time"
    assert public_run_error(listed["last_error"]) == listed["last_error"], "labels are fixed points"
