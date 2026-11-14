"""Closed cron failure labels in the job store (cron/jobs_public_status.py).

``jobs.json`` is served to chat tools, ``cron list``, the dashboard and ``/api/jobs``, so the
failure fields there are fixed labels: a provider body with a key or a URL, a host path or a
stderr tail never reaches it. The raw text stays in the run output and the executions ledger.
"""

import json

import pytest

_TOKEN = "sk-leakcanary0123456789abcdefghijklmnop"
_URL = "https://user:hunter2@api.leakcanary.example/v1/chat?key=leakcanary-q"
_PATH = "/Users/alice/private/leakcanary-7f3a/app.py"
_PROVIDER_ERROR = f"HTTP 401: invalid api key {_TOKEN} from {_URL} (config {_PATH})"
_DELIVERY_ERROR = f"telegram send failed: 400 Bad Request at {_URL}; stderr {_PATH}"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _assert_no_raw(text):
    for private in ("leakcanary", _TOKEN, "hunter2", _PATH):
        assert private not in text, f"raw failure text leaked: {text!r}"


def test_failed_run_stores_closed_labels_not_the_provider_text(home):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="x", schedule="every 5m", name="closed-store")
    assert mark_job_run(job["id"], False, _PROVIDER_ERROR, delivery_error=_DELIVERY_ERROR)

    _assert_no_raw((home / "cron" / "jobs.json").read_text(encoding="utf-8"))
    record = get_job(job["id"])
    assert record["last_status"] == "error"
    assert record["last_error"] and record["last_error"].endswith("Details: `hermes cron runs`.")
    assert record["last_delivery_error"], "the record must still say delivery failed"
    assert record["last_failure"]["detail"] == record["last_error"]


def test_delivery_failure_on_a_good_run_is_closed_and_the_reason_logged(home, caplog):
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="x", schedule="every 5m", name="closed-delivery")
    with caplog.at_level("WARNING", logger="cron.jobs"):
        assert mark_job_run(job["id"], True, delivery_error=_DELIVERY_ERROR)

    record = get_job(job["id"])
    assert (record["last_status"], record["last_error"]) == ("delivery_failed", None)
    _assert_no_raw(json.dumps(record))
    assert _DELIVERY_ERROR in caplog.text, "the adapter's reason survives only in the log"


def test_uncertain_send_text_stays_as_written(home):
    from cron.delivery_queue import UNCERTAIN_SEND_ERROR
    from cron.jobs import create_job, get_job, mark_job_run

    job = create_job(prompt="x", schedule="every 5m", name="uncertain")
    assert mark_job_run(job["id"], True, delivery_error=UNCERTAIN_SEND_ERROR)
    assert get_job(job["id"])["last_delivery_error"] == UNCERTAIN_SEND_ERROR


def test_listing_projects_a_record_written_raw_before_closure(home):
    """A record an older build wrote raw is served through the same closed labels."""
    from cron.jobs import create_job, get_job, update_job
    from tools.cronjob_tools import _format_job

    job = create_job(prompt="x", schedule="every 5m", name="legacy-record")
    raw_fire = {"at": "2026-09-24T00:00:00+00:00", "detail": f"POST {_PATH} failed: {_URL}"}
    update_job(job["id"], {"last_status": "error", "last_error": _PROVIDER_ERROR,
                           "last_delivery_error": _DELIVERY_ERROR, "last_fire_error": raw_fire})

    listed = _format_job(get_job(job["id"]))

    _assert_no_raw(json.dumps([listed["last_error"], listed["last_delivery_error"],
                               listed["last_fire_error"]]))
    assert listed["last_error"].endswith("Details: `hermes cron runs`.")
    assert listed["last_delivery_error"] and listed["last_fire_error"]["at"] == raw_fire["at"]


# ---- the label table ----------------------------------------------------------------------


@pytest.mark.parametrize("text, no_agent, kind", [
    ("Script timed out after 30s: /x/429.py", False, "script_timeout"),
    ("terminal: idle for 600s (limit 600s); HTTP 503 timed out", False, "inactivity"),
    ("Interrupted by gateway shutdown before terminal completion.", False, "shutdown"),
    ("[blocked_config] provider credential missing: no key at /h", False, "missing_credential"),
    ("[blocked_config:silent] delivery platform 'x' is off", False, "delivery_unconfigured"),
    ("[blocked_config] something else entirely", False, "blocked_config"),
    ("ImportError: cannot import name 'x' from 'agent.y'", False, "import_error"),
    ("ImportError: cannot import name 'x' from 'agent.y'", True, "script_failed"),
    ("Restart-safe cron worker exited with status 9 after adopting this execution; worker "
     "stderr: HTTP 429 at /x", False, "outcome_unknown"),
    ("RuntimeError: open('/x') failed", False, "unknown"),
])
def test_failure_kind_matches_scheduler_shapes_before_the_provider(text, no_agent, kind):
    from cron.jobs_public_status import failure_kind

    assert failure_kind(text, no_agent=no_agent) == kind


def test_provider_errors_get_the_shared_gloss():
    from cron.jobs_public_status import failure_kind, provider_reason

    assert provider_reason(failure_kind(_PROVIDER_ERROR)) is not None


def test_labels_are_fixed_points_and_blank_stays_none():
    from cron.jobs_public_status import (
        DELIVERY_FAILED, public_delivery_error, public_fire_error, public_run_error)

    label = public_run_error(_PROVIDER_ERROR)
    assert public_run_error(label) == label
    assert public_run_error(label, agent_declared=True) == label
    assert public_run_error("  ") is None and public_delivery_error("") is None
    assert public_delivery_error(DELIVERY_FAILED) == DELIVERY_FAILED
    assert public_fire_error({"at": 3, "detail": ""}) is None
    assert public_fire_error({"at": 3, "detail": "x"})["at"] is None


def test_projection_keeps_only_listed_fields():
    from cron.jobs_public_status import DASHBOARD_CRON_JOB_FIELDS, project_cron_job

    record = {"id": "a", "origin": {"chat_id": "1"}, "fire_claim": {"by": "h:1"},
              "last_error": _PROVIDER_ERROR, "skills": ["s"]}
    projected = project_cron_job(record, DASHBOARD_CRON_JOB_FIELDS)
    assert set(projected) == {"id", "last_error", "skills"}
    projected["skills"].append("t")
    assert record["skills"] == ["s"], "the projection is a copy"
