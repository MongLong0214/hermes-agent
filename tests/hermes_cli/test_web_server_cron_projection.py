"""The dashboard serves a cron job as the closed projection shared with ``/api/jobs``."""

import json

import pytest

import hermes_cli.web_routers.cron as _rt_cron
import hermes_cli.web_server_cron as _web_server_cron

_CANARY = "leakcanary"
_PROFILE_ANNOTATIONS = {"profile", "profile_name", "hermes_home", "is_default_profile",
                        "scheduler_heartbeat_age_s"}


@pytest.fixture()
def default_profile(tmp_path, monkeypatch):
    from hermes_cli import profiles

    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: home / "profiles")
    return home


@pytest.mark.asyncio
async def test_dashboard_jobs_carry_only_the_shared_projection(default_profile):
    job = _web_server_cron._call_cron_for_profile(
        "default", "create_job", prompt="x", schedule="every 1h", name="projection",
        model="edit-me-model")
    raw = {
        "last_status": "error",
        "last_error": f"RuntimeError: /Users/alice/{_CANARY}/app.py failed",
        "last_delivery_error": f"telegram 400: chat {_CANARY}-4242 not found",
        "last_fire_error": {"at": "2026-09-24T00:00:00+00:00", "detail": f"{_CANARY} refused"},
        "origin": {"platform": "telegram", "chat_id": f"{_CANARY}-chat", "user_id": _CANARY},
        "fire_claim": {"pid": 4242, "host": f"{_CANARY}-host"},
    }
    _web_server_cron._call_cron_for_profile("default", "update_job", job["id"], raw)
    stored = _web_server_cron._call_cron_for_profile("default", "get_job", job["id"])

    listed = await _rt_cron.list_cron_jobs(profile="default")
    fetched = await _rt_cron.get_cron_job(job["id"])

    for served in (*listed, fetched):
        assert _CANARY not in json.dumps(served), served
    from cron.jobs_public_status import DASHBOARD_CRON_JOB_FIELDS, closed_status_fields

    for served in (*listed, fetched):
        assert set(served) <= set(DASHBOARD_CRON_JOB_FIELDS) | _PROFILE_ANNOTATIONS
        for field, value in closed_status_fields(stored).items():
            assert served[field] == value
        # Projection, not a blank: what the dashboard edits still round-trips.
        assert served["model"] == stored["model"] and served["prompt"] == stored["prompt"]
