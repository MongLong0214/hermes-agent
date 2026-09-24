"""``/api/jobs`` serves a closed projection of the cron record, never the private record itself.

The record holds fire/run claims (owner pid and host), the creating chat's origin ids, the workdir
and raw failure text; a bearer token for the management API must not read any of it. Driven
through the real aiohttp handlers against the real store under a temp HERMES_HOME.
"""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

_CANARY = "leakcanary"
_PRIVATE_FIELDS = {
    "origin": {"platform": "telegram", "chat_id": f"{_CANARY}-chat-4412", "user_id": f"{_CANARY}-u"},
    "fire_claim": {"by": f"{_CANARY}-host:4242", "at": "2026-09-24T00:00:00+00:00"},
    "run_claim": {"by": f"{_CANARY}-host:4242", "at": "2026-09-24T00:00:00+00:00"},
    "last_error": f"RuntimeError: /Users/alice/{_CANARY}/app.py sk-{_CANARY}0123456789abcdefghij",
    "last_delivery_error": f"stderr: Traceback KeyError '{_CANARY}-stderr' at /Users/alice/{_CANARY}",
    "last_fire_error": {"at": "2026-09-24T00:00:00+00:00", "detail": f"POST /Users/alice/{_CANARY} failed"},
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from cron.jobs import create_job, update_job

    workdir = tmp_path / f"{_CANARY}-workdir"
    workdir.mkdir()
    job = create_job(prompt="report", schedule="every 5m", name="projection")
    update_job(job["id"], {**_PRIVATE_FIELDS, "workdir": str(workdir), "last_status": "error"})
    return job["id"]


def _app() -> web.Application:
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={}))
    app = web.Application(middlewares=[cors_middleware])
    app.router.add_get("/api/jobs", adapter._handle_list_jobs)
    app.router.add_get("/api/jobs/{job_id}", adapter._handle_get_job)
    app.router.add_post("/api/jobs/{job_id}/pause", adapter._handle_pause_job)
    return app


@pytest.mark.asyncio
async def test_jobs_routes_serve_only_allow_listed_fields_with_closed_status(store):
    async with TestClient(TestServer(_app())) as cli:
        bodies = []
        for method, path in (("get", "/api/jobs"), ("get", f"/api/jobs/{store}"),
                             ("post", f"/api/jobs/{store}/pause")):
            resp = await getattr(cli, method)(path)
            assert resp.status == 200, (path, await resp.text())
            bodies.append(await resp.json())

    served = [bodies[0]["jobs"][0], bodies[1]["job"], bodies[2]["job"]]
    assert all(job["id"] == store for job in served)
    for job in served:
        assert _CANARY not in json.dumps(job), f"private record content crossed the API: {job}"
    from cron.jobs_public_status import PUBLIC_CRON_JOB_FIELDS

    for job in served:
        assert set(job) <= set(PUBLIC_CRON_JOB_FIELDS)
        assert not set(job) & {"origin", "fire_claim", "run_claim", "workdir", "last_error"}
        # Status is closed, not dropped: a failed delivery/fire still reads as failed.
        assert job["last_delivery_error"] and job["last_fire_error"]["at"]
