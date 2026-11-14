"""Refusals at authenticated canonical-surface ingress (``POST /v1/canonical-surface/events``).

Each is answered before any receipt is claimed, so nothing ran under the event's id: a request
without the API server key, a binding whose session no longer exists (the surface binds existing
sessions only and never creates one), and text over the cap.

The cap case is the regression for agent-control-plane #1111: on 2026-10-16 a 17,475-char thread
reply was answered 400 ``canonical_invalid_request`` and the gateway recorded nothing, so the refusal
was visible only in the bridge's own log. The answer the bridge reads stays the same (it tells the
sender from it); the gateway's log names the refused event by its correlation and sizes, never by
its text, and the text is never truncated into a turn.

Real runner, session store, resolver, coordinator and SQLite under a tmp home; the cached actor has
no model.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import re
import secrets
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from gateway.canonical_surface import _MAX_TEXT_CHARS, CanonicalSurfaceBinding
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.run import GatewayRunner
from gateway.session import SessionSource

_ROUTE = "/v1/canonical-surface/events"
_IDENTITY_ROUTE = "/v1/canonical-surface/identity"
_TERMINAL = "request-owned terminal"


class _CachedActor:
    """The bound cached agent, with no model: records the exact text of every turn it ran."""

    compression_in_place = True

    def __init__(self, session_id, db):
        self.session_id = session_id
        self._session_db = db
        self.calls = []

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        self.calls.append(kwargs.get("persist_user_message", text))
        self._persist_user_message_idx = len(conversation_history)
        return {"completed": True, "session_id": task_id, "final_response": _TERMINAL,
                "messages": [*conversation_history, {"role": "user", "content": text},
                             {"role": "assistant", "content": _TERMINAL}]}


@pytest.fixture
def ingress(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    # The home override wins over HERMES_HOME, and a module collected in the same process may leave
    # one set (tests/gateway/test_canonical_peer_tool_grant.py pins its own at import), so pin this one.
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    override = set_hermes_home_override(home)
    try:
        yield from _ingress_under(home, monkeypatch)
    finally:
        reset_hermes_home_override(override)


def _ingress_under(home, monkeypatch):
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    actor = _CachedActor(entry.session_id, db)
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)
    import run_agent
    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail("canonical ingress built an agent"))

    def rows(sql):
        with contextlib.closing(sqlite3.connect(db.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            return conn.execute(sql).fetchall()

    key = secrets.token_hex(32)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": key}))
    adapter.gateway_runner = runner
    try:
        yield SimpleNamespace(
            runner=runner, actor=actor, key=key, entry=entry, binding=binding, adapter=adapter,
            receipts=lambda: rows("SELECT key FROM state_meta WHERE key LIKE 'canonical-receipt:%'"),
            sessions=lambda: rows("SELECT id, ended_at FROM sessions ORDER BY id"))
    finally:
        adapter._response_store.close()
        runner.session_store.close_all_db_handles()


def _handler(adapter, method, route):
    [handler] = [h for m, path, h in adapter._http_route_table() if (m, path) == (method, route)]
    return handler


def _send(ingress, *, bearer="key", profile=None, **changes):
    """POST one event through the registered route handler; ``bearer="key"`` is the real key."""
    handler = _handler(ingress.adapter, "POST", _ROUTE)
    raw = json.dumps({"binding": "canonical", "event_id": "evt-1", "author_id": "author",
                      "channel_id": "channel", "text": "hello", **changes}).encode()
    bearer = ingress.key if bearer == "key" else bearer

    async def read():
        return raw

    async def go():
        headers = {"Authorization": f"Bearer {bearer}"} if bearer is not None else {}
        request = SimpleNamespace(headers=headers, read=read, method="POST", path_qs=_ROUTE, transport=None)
        token = _api_request_profile.set(profile)
        try:
            response = await handler(request)
        finally:
            _api_request_profile.reset(token)
        return response.status, json.loads(response.text)
    return asyncio.run(go())


def _code(result):
    return result[0], result[1]["error"]["code"]


def test_a_request_without_the_api_server_key_is_401_and_claims_no_receipt(ingress):
    for bearer in (None, "not-the-key"):
        assert _code(_send(ingress, bearer=bearer)) == (401, "gateway_auth_failed")
    assert ingress.actor.calls == [] and ingress.receipts() == []

    # Control: the same event with the key runs once and claims its receipt.
    assert _send(ingress) == (200, {"event_id": "evt-1", "text": _TERMINAL})
    assert ingress.actor.calls == ["hello"] and len(ingress.receipts()) == 1


@pytest.mark.parametrize("case", ["no_such_session", "ended"])
def test_a_binding_whose_session_does_not_exist_is_refused_and_nothing_is_created(ingress, case):
    binding = ingress.binding
    if case == "no_such_session":
        binding = dataclasses.replace(binding, session_key=binding.session_key + ":ghost",
                                      session_id="no-such-session")
        ingress.runner.config.canonical_surface_bindings = {binding.name: binding}
    else:
        ingress.runner.session_store._db.end_session(ingress.entry.session_id, "test_ended")
    before = ingress.sessions()

    assert _code(_send(ingress)) == (409, "canonical_binding_stale")
    assert ingress.sessions() == before
    assert ingress.actor.calls == [] and ingress.receipts() == []


def test_an_unknown_binding_or_another_profile_is_404_and_claims_no_receipt(ingress, monkeypatch):
    assert _code(_send(ingress, binding="not-configured")) == (404, "canonical_binding_unknown")
    # Another served profile's own key does not reach the launch profile's bindings.
    other_key = secrets.token_hex(32)
    monkeypatch.setenv("API_SERVER_KEY", other_key)
    assert _code(_send(ingress, bearer=other_key, profile="other")) == (404, "canonical_binding_unknown")
    assert ingress.actor.calls == [] and ingress.receipts() == []


def test_without_a_runner_the_routes_are_unavailable_and_without_a_key_identity_is_refused(ingress):
    ingress.adapter.gateway_runner = None
    assert _code(_send(ingress)) == (503, "canonical_unavailable")
    assert _code(_get_identity(ingress, ingress.key)) == (503, "canonical_unavailable")
    keyless = APIServerAdapter(PlatformConfig(enabled=True))
    keyless.gateway_runner = ingress.runner
    try:
        assert _get_identity(SimpleNamespace(adapter=keyless), None)[0] == 401
    finally:
        keyless._response_store.close()
    assert ingress.actor.calls == [] and ingress.receipts() == []


@pytest.mark.parametrize("outcome, expected", [
    (ValueError("canonical_turn_busy"), (409, "canonical_turn_busy")),
    (ValueError("canonical_principal_rejected"), (403, "canonical_principal_rejected")),
    (ValueError("canonical_receipt_conflict"), (409, "canonical_event_conflict")),
    (ValueError("canonical_roster_unavailable"), (503, "canonical_roster_unavailable")),
    (ValueError("canonical_actor_finish_stale"), (409, "canonical_event_uncertain")),
    (RuntimeError("private detail"), (409, "canonical_event_uncertain")),
    (("pending", None, None), (409, "canonical_event_uncertain")),
    (("terminal", None, {"side_effects": "none"}), (409, "canonical_event_failed")),
    (("unrecorded", None, {"side_effects": "none", "answer_kept": True}), (409, "canonical_answer_unrecorded")),
])
def test_each_coordinator_outcome_is_answered_without_detail_and_never_invites_a_rerun(
        ingress, monkeypatch, outcome, expected):
    from gateway.canonical_surface import CanonicalReceiptCoordinator, CanonicalReceiptResult

    async def submit(self, binding, event):
        if isinstance(outcome, Exception):
            raise outcome
        return CanonicalReceiptResult(*outcome)

    monkeypatch.setattr(CanonicalReceiptCoordinator, "submit", submit)
    status, body = _send(ingress)
    assert (status, body["error"]["code"]) == expected
    assert "private detail" not in json.dumps(body)
    if expected[1] in {"canonical_event_failed", "canonical_answer_unrecorded"}:
        assert body["error"]["side_effects"] == "none"
    if expected[1] == "canonical_answer_unrecorded":
        assert (body["error"]["delivery"], body["error"]["answer_kept"]) == ("not_delivered", True)


def _cap_records(caplog):
    return [r.getMessage() for r in caplog.records if "text_over_limit" in r.getMessage()]


def test_an_event_over_the_cap_is_refused_whole_and_logged_by_correlation_never_text(ingress, caplog):
    caplog.set_level(logging.INFO)
    text = "Q" * (_MAX_TEXT_CHARS + 1)

    assert _send(ingress, text=text) == (400, {"error": {
        "code": "canonical_invalid_request", "message": "Canonical request rejected."}})
    [message] = _cap_records(caplog)
    for part in ("binding='canonical'", "event_id='evt-1'", f"chars={len(text)}", f"limit={_MAX_TEXT_CHARS}"):
        assert part in message, message
    assert "Q" * 16 not in message
    # Refused whole: no turn ran on a truncated body and nothing was claimed.
    assert ingress.actor.calls == [] and ingress.receipts() == []


def test_control_an_event_at_the_cap_runs_its_whole_text_and_is_not_logged(ingress, caplog):
    caplog.set_level(logging.INFO)
    text = "Q" * _MAX_TEXT_CHARS

    assert _send(ingress, text=text) == (200, {"event_id": "evt-1", "text": _TERMINAL})
    assert ingress.actor.calls == [text]
    assert _cap_records(caplog) == []


def _get_identity(ingress, key, profile=None, path=_IDENTITY_ROUTE):
    handler = _handler(ingress.adapter, "GET", _IDENTITY_ROUTE)

    async def go():
        token = _api_request_profile.set(profile)
        try:
            response = await handler(SimpleNamespace(
                headers={"Authorization": f"Bearer {key}"} if key else {},
                method="GET", path_qs=path, raw_path=path, transport=None, match_info={}))
            return response.status, json.loads(response.text)
        finally:
            _api_request_profile.reset(token)
    return asyncio.run(go())


def test_identity_get_is_authenticated_and_reports_the_live_head_without_writes(ingress, monkeypatch):
    from hermes_state_target_bind import _lineage_root_digest

    def get(key, profile=None, path=_IDENTITY_ROUTE):
        return _get_identity(ingress, key, profile, path)

    assert get(None)[0] == 401 and get("wrong")[0] == 401
    assert get(ingress.key, profile="other")[0] == 404
    assert _code(get(ingress.key, path=_IDENTITY_ROUTE + "?binding=canonical")) == (400, "canonical_invalid_request")
    status, identity = get(ingress.key)
    assert status == 200
    assert identity["session_id"] == ingress.entry.session_id
    assert identity["lineage_root_digest"] == _lineage_root_digest(ingress.entry.session_id)
    assert identity["process_pid"] > 0
    token_pattern = r"darwin-tv:\d+\.\d{6}" if sys.platform == "darwin" else r"linux-clk:\d+"
    assert re.fullmatch(token_pattern, identity["process_started_at"])
    assert set(identity) == {"session_id", "lineage_root_digest", "process_pid", "process_started_at"}
    assert get(ingress.key)[1] == identity

    ingress.runner.session_store._db.end_session(ingress.entry.session_id, "test_ended")
    assert _code(get(ingress.key)) == (409, "canonical_binding_stale")
    ingress.runner.session_store._db.reopen_session(ingress.entry.session_id)
    # Identity answers for the sole binding only: with two, it names none.
    second = dataclasses.replace(ingress.binding, name="second")
    ingress.runner.config.canonical_surface_bindings = {"canonical": ingress.binding, "second": second}
    assert _code(get(ingress.key)) == (404, "canonical_binding_unknown")
    ingress.runner.config.canonical_surface_bindings = {"canonical": ingress.binding}

    from gateway.canonical_surface import ExistingCanonicalBindingResolver
    with monkeypatch.context() as patched:
        def roster_unavailable(self, binding, event):
            raise ValueError("canonical_roster_unavailable")
        patched.setattr(ExistingCanonicalBindingResolver, "resolve", roster_unavailable)
        assert _code(get(ingress.key)) == (503, "canonical_roster_unavailable")
    with monkeypatch.context() as patched:
        # The store resolved, then its state.db is gone by the time the lineage is read.
        real, calls = ExistingCanonicalBindingResolver._existing_db_path, []

        def vanishing(self, session_key):
            calls.append(session_key)
            return real(self, session_key) if len(calls) == 1 else None
        patched.setattr(ExistingCanonicalBindingResolver, "_existing_db_path", vanishing)
        assert _code(get(ingress.key)) == (409, "canonical_binding_stale")
    monkeypatch.setattr("gateway.platforms.api_server_canonical._process_started_at", lambda pid: None)
    assert _code(get(ingress.key)) == (503, "canonical_unavailable")
    assert ingress.actor.calls == [] and ingress.receipts() == []


def test_identity_get_default_profile_http_mirror(ingress):
    """The registered mirror accepts only the validated default profile and the exact path."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    async def exercise():
        adapter = ingress.adapter
        app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
        handler = _handler(adapter, "GET", _IDENTITY_ROUTE)
        app.router.add_route("GET", _IDENTITY_ROUTE, handler)
        app.router.add_route("GET", f"/p/{{profile}}{_IDENTITY_ROUTE}", handler)
        headers = {"Authorization": f"Bearer {ingress.key}"}
        async with TestClient(TestServer(app)) as client:
            bare = await client.get(_IDENTITY_ROUTE, headers=headers)
            assert bare.status == 200
            mirror = await client.get("/p/default" + _IDENTITY_ROUTE, headers=headers)
            assert (mirror.status, await mirror.json()) == (200, await bare.json())
            query = await client.get("/p/default" + _IDENTITY_ROUTE + "?binding=canonical", headers=headers)
            assert (query.status, (await query.json())["error"]["code"]) == (400, "canonical_invalid_request")
            denied = await client.get("/p/default" + _IDENTITY_ROUTE, headers={"Authorization": "Bearer wrong"})
            assert denied.status == 401
            foreign = await client.get("/p/other" + _IDENTITY_ROUTE, headers=headers)
            assert foreign.status == 404

    asyncio.run(exercise())
    assert ingress.actor.calls == [] and ingress.receipts() == []
