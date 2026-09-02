"""Characterization + unit tests for the `run_one_job` shared helper (Phase 4A).

`tick`'s per-job body (`_process_job`) is the execute → save → deliver → mark
sequence that fires ONE due job. Phase 4A extracts it into a module-level
`run_one_job(job, *, adapters=None, loop=None, verbose=False)` so the external
Chronos provider's `fire_due` can reuse the IDENTICAL body — no duplicated
correctness.

The first test characterizes the sequence as driven through `tick()` (proving
the extraction didn't change `tick`'s behavior); the rest unit-test the
extracted helper directly.
"""
import logging

import pytest

import cron.scheduler as s


def _patch_pipeline(monkeypatch, *, success=True, output="out", final="final response",
                    error=None, silent_marker_in=None):
    """Patch the job pipeline primitives and record the call order."""
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        calls.append(("run_job", job["id"]))
        fr = final if silent_marker_in is None else silent_marker_in
        return (success, output, fr, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, **_kw):
        calls.append(("mark", jid, ok))

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    return calls


def test_tick_process_job_sequence(monkeypatch):
    """Characterization: a single due job driven through tick() runs the
    sequence run_job → save → deliver → mark, in that order."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "claim_job_for_fire", lambda _job_id, **_kwargs: True)

    s.tick(verbose=False, sync=True)

    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j1", True)


def test_tick_skips_job_when_durable_fire_claim_is_lost(monkeypatch):
    """A manual/external fire that wins the shared CAS must exclude ticker."""
    calls = _patch_pipeline(monkeypatch)
    monkeypatch.setattr(s, "get_due_jobs", lambda: [{"id": "j1", "name": "t"}])
    monkeypatch.setattr(s, "claim_job_for_fire", lambda _job_id: False)

    assert s.tick(verbose=False, sync=True) == 0
    assert calls == []


def test_run_one_job_success_sequence(monkeypatch):
    """The extracted helper runs the same execute→save→deliver→mark sequence
    for a successful job."""
    calls = _patch_pipeline(monkeypatch)

    ok = s.run_one_job({"id": "j2", "name": "t"})

    assert ok is True
    assert [c[0] for c in calls] == ["run_job", "save", "deliver", "mark"]
    assert calls[-1] == ("mark", "j2", True)


def test_run_one_job_exception_delivers_failure_alert(monkeypatch):
    """An exception escaping the run body must not become a silent error row."""
    delivered = []
    marked = []
    finished = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j3"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            RuntimeError("Gemini HTTP 503 (UNAVAILABLE)")
        ),
    )
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append((job["id"], content)) or None,
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)),
    )
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda *args, **kwargs: finished.append((args, kwargs)),
    )

    ok = s.run_one_job({"id": "j3", "name": "morning", "deliver": "telegram"})

    assert ok is False
    assert delivered == [("j3", s._CRON_EXECUTION_FAILURE)]
    assert marked == [
        (("j3", False, s._CRON_EXECUTION_FAILURE), {"delivery_error": None})
    ]
    assert finished == [
        (
            ("exec-j3",),
            {
                "success": False,
                "error": s._CRON_EXECUTION_FAILURE,
                "delivery_outcome": "delivered",
            },
        )
    ]


def test_run_one_job_exception_records_failure_alert_delivery_error(monkeypatch):
    """A failed fallback alert must populate last_delivery_error."""
    marked = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j4"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("provider failed")),
    )
    monkeypatch.setattr(s, "_deliver_result", lambda *_a, **_kw: "send failed: 502")
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)),
    )
    monkeypatch.setattr(s, "finish_execution", lambda *_a, **_kw: None)

    assert s.run_one_job({"id": "j4", "deliver": "telegram"}) is False
    assert marked == [
        (("j4", False, s._CRON_EXECUTION_FAILURE),
        {"delivery_error": s._CRON_DELIVERY_FAILURE})
    ]


def _patch_escaped_failure(monkeypatch, delivered, *, exec_id, err):
    """Make run_job raise, and capture what the escape handler delivers."""
    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": exec_id})
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError(err)),
    )
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append(content) or None,
    )
    monkeypatch.setattr(s, "mark_job_run", lambda *_a, **_kw: None)
    monkeypatch.setattr(s, "finish_execution", lambda *_a, **_kw: None)
    # Deterministic threshold: default 3, independent of the host config.
    monkeypatch.setattr(s, "load_config", lambda: {})


def test_escaped_failure_delivery_hides_streak_and_exception_detail(monkeypatch):
    """The public alert is neutral even when the internal streak advances."""
    delivered = []
    _patch_escaped_failure(
        monkeypatch, delivered, exec_id="exec-j5", err="cannot import name X"
    )

    ok = s.run_one_job(
        {
            "id": "j5",
            "name": "scout",
            "deliver": "telegram",
            "schedule": {"kind": "interval"},
            "failure_streak": 2,  # + this run = 3 = default threshold
        }
    )

    assert ok is False
    assert delivered == [s._CRON_EXECUTION_FAILURE]


def test_escaped_failure_delivery_is_neutral_below_the_threshold(monkeypatch):
    """A first escaped failure uses the same public-neutral contract."""
    delivered = []
    _patch_escaped_failure(
        monkeypatch, delivered, exec_id="exec-j6", err="provider failed"
    )

    ok = s.run_one_job(
        {
            "id": "j6",
            "name": "scout",
            "deliver": "telegram",
            "schedule": {"kind": "interval"},
            "failure_streak": 0,
        }
    )

    assert ok is False
    assert delivered == [s._CRON_EXECUTION_FAILURE]


def test_run_one_job_exception_after_delivery_does_not_redeliver(monkeypatch):
    """Once delivery has been attempted, the outer handler must not send again."""
    delivered = []
    mark_calls = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j5"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (True, "out", "final response", None),
    )
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append((job["id"], content)) or None,
    )

    def fake_mark(*args, **kwargs):
        mark_calls.append((args, kwargs))
        if len(mark_calls) == 1:
            raise RuntimeError("bookkeeping boom")

    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    monkeypatch.setattr(s, "finish_execution", lambda *_a, **_kw: None)

    ok = s.run_one_job({"id": "j5", "name": "once", "deliver": "telegram"})

    assert ok is False
    assert delivered == [("j5", "final response")]
    assert mark_calls[0] == (("j5", True, None), {"delivery_error": None})
    assert mark_calls[1] == (
        ("j5", False, s._CRON_EXECUTION_FAILURE),
        {"delivery_error": None},
    )


def test_run_one_job_keyboard_interrupt_skips_delivery_and_reraises(monkeypatch):
    """Hard interrupts must not attempt failure delivery; they re-raise."""
    delivered = []
    marked = []
    finished = []

    monkeypatch.setattr(
        s, "create_execution", lambda *_a, **_kw: {"id": "exec-j6"}
    )
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda _execution_id: None)
    monkeypatch.setattr(
        s,
        "run_job",
        lambda *_a, **_kw: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda job, content, **_kw: delivered.append((job["id"], content)) or None,
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda *args, **kwargs: marked.append((args, kwargs)),
    )
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda *args, **kwargs: finished.append((args, kwargs)),
    )

    with pytest.raises(KeyboardInterrupt):
        s.run_one_job({"id": "j6", "name": "interrupt", "deliver": "telegram"})

    assert delivered == []
    assert marked == [(("j6", False, s._CRON_EXECUTION_FAILURE), {})]
    assert finished == [
        (
            ("exec-j6",),
            {
                "success": False,
                "error": s._CRON_EXECUTION_FAILURE,
                "delivery_outcome": "suppressed",
            },
        )
    ]


def test_run_one_job_installs_secret_scope_under_multiplex(monkeypatch, tmp_path):
    """Regression: under profile isolation (multiplex active), run_one_job must
    execute run_job inside a profile secret scope so credential reads
    (resolve_runtime_provider -> get_secret) don't fail-close with
    UnscopedSecretError, and must tear the scope down afterward.

    Behavior contract: a scope is present during run_job and absent after,
    regardless of the concrete secret values.
    """
    from agent import secret_scope as ss

    # Point cron's home resolution at a profile whose .env carries a secret.
    (tmp_path / ".env").write_text("OPENROUTER_BASE_URL=https://openrouter.ai/api/v1\n")
    monkeypatch.setattr(s, "_get_hermes_home", lambda: tmp_path)

    scope_during_run = {}

    def fake_run_job(job, *, defer_agent_teardown=None, **kw):
        # This is where resolve_runtime_provider() would read a secret. Prove a
        # scope is installed and the profile's secret resolves without raising.
        scope_during_run["scope"] = ss.current_secret_scope()
        scope_during_run["base_url"] = ss.get_secret("OPENROUTER_BASE_URL")
        return (True, "out", "final", None)

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    monkeypatch.setattr(s, "mark_job_run", lambda *a, **k: None)

    ss.set_multiplex_active(True)
    try:
        ok = s.run_one_job({"id": "j7", "name": "t"})
    finally:
        ss.set_multiplex_active(False)

    assert ok is True
    # Scope was installed during run_job and the profile secret resolved.
    assert scope_during_run["scope"] is not None
    assert scope_during_run["base_url"] == "https://openrouter.ai/api/v1"
    # And it was torn down after run_one_job returned (no leak).
    assert ss.current_secret_scope() is None


def _run_real_failure_through_all_public_sinks(monkeypatch, job):
    """Capture the real run_job result before every durable/public sink."""
    observed = {
        "returned": [],
        "saved": [],
        "delivered": [],
        "marked": [],
        "finished": [],
    }
    real_run_job = s.run_job

    def capture_run_job(*args, **kwargs):
        result = real_run_job(*args, **kwargs)
        observed["returned"].append(result)
        return result

    monkeypatch.setattr(s, "create_execution", lambda *_a, **_kw: {"id": "exec-public"})
    monkeypatch.setattr(s, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(s, "mark_execution_running", lambda *_a, **_kw: None)
    monkeypatch.setattr(s, "run_job", capture_run_job)
    monkeypatch.setattr(
        s,
        "save_job_output",
        lambda job_id, output: observed["saved"].append((job_id, output)) or "/tmp/output.md",
    )
    monkeypatch.setattr(
        s,
        "_deliver_result",
        lambda current_job, content, **_kw: observed["delivered"].append(
            (current_job["id"], content)
        )
        or None,
    )
    monkeypatch.setattr(
        s,
        "mark_job_run",
        lambda *args, **kwargs: observed["marked"].append((args, kwargs)),
    )
    monkeypatch.setattr(
        s,
        "finish_execution",
        lambda *args, **kwargs: observed["finished"].append((args, kwargs)),
    )

    assert s.run_one_job(job) is True
    return observed


def _assert_only_fixed_public_failure(observed, caplog, private_text, job_id):
    """Every returned, saved, delivered, ledger, and log failure is neutral."""
    public = s._CRON_DELIVERY_FAILURE
    assert observed["returned"] == [(False, public, public, public)]
    assert observed["saved"] == [(job_id, public)]
    assert observed["delivered"] == [(job_id, public)]
    assert observed["marked"] == [
        ((job_id, False, public), {"delivery_error": None})
    ]
    assert observed["finished"] == [
        (
            ("exec-public",),
            {
                "success": False,
                "error": public,
                "delivery_outcome": "delivered",
            },
        )
    ]
    assert private_text not in caplog.text
    for record in caplog.records:
        if record.name == s.__name__ and record.levelno >= logging.ERROR:
            assert record.getMessage() == public
            assert record.args == ()
            assert record.exc_info is None


def test_no_agent_script_failure_is_fixed_before_return_and_durable_sinks(
    monkeypatch, tmp_path, caplog
):
    """A nonzero script's private stderr/path never reaches a public failure sink."""
    private_path_and_stderr = "/private/no-agent-script/DO-NOT-DISCLOSE"
    home = tmp_path / "home"
    scripts = home / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "failure.py").write_text(
        "import sys\n"
        f"sys.stderr.write({private_path_and_stderr!r})\n"
        "sys.exit(7)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(s, "_get_hermes_home", lambda: home)
    caplog.set_level(logging.ERROR, logger=s.__name__)

    job = {
        "id": "public-script-failure",
        "name": "public script failure",
        "no_agent": True,
        "script": "failure.py",
        "deliver": "telegram",
    }
    observed = _run_real_failure_through_all_public_sinks(monkeypatch, job)

    _assert_only_fixed_public_failure(
        observed, caplog, private_path_and_stderr, job["id"]
    )


def test_monitor_source_failure_is_fixed_before_return_and_durable_sinks(
    monkeypatch, tmp_path, caplog
):
    """A monitor exception string cannot reach any public/durable failure sink."""
    from cron.monitor import MonitorOutcome
    import cron.monitor as monitor

    private_exception = "PRIVATE_MONITOR_EXCEPTION_DO_NOT_DISCLOSE"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(s, "_get_hermes_home", lambda: home)
    monkeypatch.setattr(
        monitor,
        "check_monitor",
        lambda _job: MonitorOutcome(ok=False, error=private_exception),
    )
    caplog.set_level(logging.ERROR, logger=s.__name__)

    job = {
        "id": "public-monitor-failure",
        "name": "public monitor failure",
        "monitor_script": "source.py",
        "deliver": "telegram",
    }
    observed = _run_real_failure_through_all_public_sinks(monkeypatch, job)

    _assert_only_fixed_public_failure(observed, caplog, private_exception, job["id"])
