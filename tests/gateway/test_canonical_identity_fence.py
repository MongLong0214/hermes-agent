"""The bound turn runs only the exact actor, head and session it was admitted on, and runs it as a peer.

Every refusal below is raised before the cached actor's turn starts. The runner, its session store and
its SessionDB are real (under a tmp home); the cached actor is a stand-in with no model.
"""

import asyncio
import dataclasses
import json
import sqlite3
from types import SimpleNamespace

import pytest

from agent.canonical_peer import new_peer_metadata
from gateway.canonical_surface import (
    CanonicalIngressEvent, CanonicalReceiptCoordinator, CanonicalReceiptResult, CanonicalSurfaceBinding,
    _path_identity, request_local_reply_sink,
)
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class _Actor:
    """The bound cached agent, with no model. ``outcome``: completed, malformed (no answer) or raises."""

    compression_in_place = True

    def __init__(self, session_id, db, *, outcome="completed", observe=None):
        self.session_id = session_id
        self._session_db = db
        self.outcome = outcome
        self.observe = observe
        self.calls = []

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        self.calls.append({"text": text, "task_id": task_id, **kwargs})
        if self.observe is not None:
            self.observe(self)
        if self.outcome == "raises":
            raise RuntimeError("actor failure")
        self._persist_user_message_idx = len(conversation_history)
        answer = "terminal" if self.outcome == "completed" else ""
        return {"completed": True, "session_id": task_id, "final_response": answer,
                "messages": [*conversation_history, {"role": "user", "content": text},
                             *([{"role": "assistant", "content": answer}] if answer else [])]}

    def interrupt(self, text=None):
        pass

    def release_clients(self):
        pass


@pytest.fixture
def target(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    db = runner.session_store._db
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    actor = _Actor(entry.session_id, db)

    def cache(agent, session_id=entry.session_id):
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (agent, "exact", 0, session_id)

    def receipts():
        with sqlite3.connect(db.db_path) as conn:
            return conn.execute("SELECT key FROM state_meta WHERE key LIKE 'canonical-receipt:%'").fetchall()

    def sessions():
        with sqlite3.connect(db.db_path) as conn:
            return conn.execute("SELECT id FROM sessions ORDER BY id").fetchall()

    cache(actor)
    event = SimpleNamespace(text="hello", author_id="author", channel_id="channel", event_id="one")
    try:
        yield SimpleNamespace(runner=runner, actor=actor, db=db, entry=entry, binding=binding, event=event,
                              cache=cache, receipts=receipts, sessions=sessions)
    finally:
        runner.session_store.close_all_db_handles()


def _peer():
    return new_peer_metadata(binding="canonical", author_id="author", channel_id="channel", event_id="one",
                             receipt="canonical-receipt:v2:test")


async def _discard(_result):
    return None


def _proof(t, **changes):
    """The proof the receipt coordinator hands the runner for the actor it borrowed."""
    return {"expected_actor": t.actor, "expected_session_db": t.db, "expected_db_path": t.db.db_path,
            "expected_db_identity": _path_identity(t.db.db_path), **changes}


def _run(t, *, binding=None, entry=None, **kwargs):
    kwargs.setdefault("reply_sink", request_local_reply_sink(_discard))
    kwargs.setdefault("peer", _peer())
    return asyncio.run(t.runner.run_bound_existing_turn(binding or t.binding, t.event, entry or t.entry, **kwargs))


# ── An actor or head other than the expected one refuses before the turn starts ─────────────────


@pytest.mark.parametrize("mismatch", ["cache-swapped", "other-db-handle", "db-file-changed"])
def test_an_actor_other_than_the_expected_one_refuses_before_the_turn(target, mismatch):
    other = _Actor(target.entry.session_id, target.db)
    changes = {}
    if mismatch == "cache-swapped":
        target.cache(other)
    elif mismatch == "other-db-handle":
        changes["expected_session_db"] = object()
    else:
        changes["expected_db_identity"] = (0, 0)
    with pytest.raises(ValueError, match="^canonical_agent_replaced$"):
        _run(target, **_proof(target, **changes))
    assert target.actor.calls == [] and other.calls == []


def test_a_retired_run_generation_refuses_before_the_turn(target):
    key = target.entry.session_key
    generation = target.runner._begin_session_run_generation(key)
    target.runner._invalidate_session_run_generation(key, reason="stop")
    with pytest.raises(ValueError, match="^canonical_turn_interrupted$"):
        _run(target, expected_run_generation=generation, **_proof(target))
    assert target.actor.calls == []


@pytest.mark.parametrize("lease_kind", ["other-owner", "released", "other-session"])
def test_a_lease_this_event_does_not_hold_refuses_before_the_turn(target, lease_kind):
    leases, session_id = target.runner._turn_leases, target.entry.session_id
    owner_key = f"canonical:{id(target.event)}"

    async def exercise():
        if lease_kind == "other-owner":
            lease = await leases.acquire(session_id, owner_key="canonical:another-event", generation=0)
        elif lease_kind == "other-session":
            lease = await leases.acquire("another-session", owner_key=owner_key, generation=0)
        else:
            lease = await leases.acquire(session_id, owner_key=owner_key, generation=0)
            leases.release(lease)
        with pytest.raises(ValueError, match="^canonical_turn_busy$"):
            await target.runner.run_bound_existing_turn(
                target.binding, target.event, target.entry, reply_sink=request_local_reply_sink(_discard),
                held_lease=lease, peer=_peer(), **_proof(target))

    asyncio.run(exercise())
    assert target.actor.calls == []


def test_the_lease_this_event_holds_runs_the_expected_actor_once(target):
    """Control for the lease refusals: the same call with this event's own live lease runs the actor."""
    target.actor.outcome = "malformed"  # refused after the actor ran, whatever the completed-turn proof

    async def exercise():
        lease = await target.runner._turn_leases.acquire(
            target.entry.session_id, owner_key=f"canonical:{id(target.event)}", generation=0)
        try:
            with pytest.raises(ValueError, match="^canonical_turn_refused$"):
                await target.runner.run_bound_existing_turn(
                    target.binding, target.event, target.entry, reply_sink=request_local_reply_sink(_discard),
                    held_lease=lease, peer=_peer(), **_proof(target))
            assert lease.lease.lock.locked(), "a held lease belongs to the caller and is not released"
        finally:
            target.runner._turn_leases.release(lease)

    asyncio.run(exercise())
    assert len(target.actor.calls) == 1


@pytest.mark.parametrize("field, value", [
    ("telegram_chat_id", "another-chat"), ("telegram_chat_type", "group"), ("telegram_user_id", "another-user"),
    ("telegram_thread_id", "7"), ("session_key", "agent:main:telegram:dm:another-chat"),
])
def test_an_entry_off_its_binding_refuses_before_any_lease(target, field, value):
    binding = dataclasses.replace(target.binding, **{field: value})
    with pytest.raises(ValueError, match="^canonical_binding_stale$"):
        _run(target, binding=binding, **_proof(target))
    assert target.actor.calls == [] and target.runner._turn_leases._leases == {}


@pytest.mark.parametrize("stale", ["cache-entry", "actor"])
def test_a_cached_actor_serving_another_session_refuses_before_the_turn(target, stale):
    if stale == "cache-entry":
        target.cache(target.actor, session_id="another-session")
    else:
        target.actor.session_id = "another-session"
    with pytest.raises(ValueError, match="^canonical_agent_missing$"):
        _run(target)
    assert target.actor.calls == []


def test_an_actor_swapped_after_the_claim_never_runs(target, monkeypatch):
    """End to end: the receipt coordinator borrowed one actor and claimed the event under it; a
    different actor cached before the turn starts is refused, and neither actor runs."""
    other = _Actor(target.entry.session_id, target.db)
    claim = target.db.claim_meta_once

    def claim_then_swap(key, *args, **kwargs):
        claimed = claim(key, *args, **kwargs)
        if key.startswith("canonical-receipt:"):
            target.cache(other)
        return claimed

    monkeypatch.setattr(target.db, "claim_meta_once", claim_then_swap)
    event = CanonicalIngressEvent.from_json_bytes(json.dumps({
        "binding": "canonical", "event_id": "one", "author_id": "author", "channel_id": "channel",
        "text": "hello"}).encode())
    with pytest.raises(ValueError, match="^canonical_receipt_terminal_unconfirmed$"):
        asyncio.run(CanonicalReceiptCoordinator(target.runner).submit(target.binding, event))
    assert target.actor.calls == [] and other.calls == []


def test_the_admitted_actor_answers_once_and_a_replay_does_not_run_it_again(target):
    event = CanonicalIngressEvent.from_json_bytes(json.dumps({
        "binding": "canonical", "event_id": "one", "author_id": "author", "channel_id": "channel",
        "text": "hello"}).encode())
    result = asyncio.run(CanonicalReceiptCoordinator(target.runner).submit(target.binding, event))
    assert result == CanonicalReceiptResult("terminal", "terminal")
    assert asyncio.run(CanonicalReceiptCoordinator(target.runner).submit(target.binding, event)) == result
    assert len(target.actor.calls) == 1 and len(target.receipts()) == 1


# ── The turn runs as a peer ─────────────────────────────────────────────────────────────────────


def test_the_turn_runs_as_a_peer_with_approvals_denied_and_no_background_review(target, monkeypatch):
    """The owner runs with --yolo; inside the canonical turn every approval is refused, the background
    review is off and the only tool grant is this binding's. Nothing outlives the turn."""
    from tools import approval
    from agent import canonical_peer_tools
    from gateway.session_context import get_session_env
    from tools.approval_context import get_turn_principal

    from gateway.session_context import _CRON_SESSION

    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", True)
    # The caller's turn declared itself non-cron while the process environment says cron: the peer turn
    # carries the bound value over and never falls back on the environment.
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    cron_token = _CRON_SESSION.set("")
    seen = {}

    def observe(agent):
        grant = canonical_peer_tools._grant.get()
        seen.update(
            principal=get_turn_principal(),
            approved=approval.check_dangerous_command("rm -rf /tmp/x", "local")["approved"],
            agent_principal=agent._turn_principal, skip_review=agent.skip_background_review,
            grant=(grant.binding, grant.agent is agent) if grant is not None else None,
            session_key=get_session_env("HERMES_SESSION_KEY"), session_id=get_session_env("HERMES_SESSION_ID"),
            user_id=get_session_env("HERMES_SESSION_USER_ID"), chat_id=get_session_env("HERMES_SESSION_CHAT_ID"),
            cron=get_session_env("HERMES_CRON_SESSION"),
        )

    target.actor.observe, target.actor.outcome = observe, "raises"
    target.actor.skip_background_review = False
    try:
        with pytest.raises(RuntimeError, match="actor failure"):
            _run(target, **_proof(target))
    finally:
        _CRON_SESSION.reset(cron_token)
    assert seen == {
        "principal": "peer", "approved": False, "agent_principal": "peer", "skip_review": True,
        "grant": ("canonical", True), "session_key": target.entry.session_key,
        "session_id": target.entry.session_id, "user_id": "", "chat_id": "chat", "cron": "",
    }
    assert target.actor.skip_background_review is False
    assert getattr(target.actor, "_turn_principal", None) is None
    assert get_turn_principal() == "owner" and canonical_peer_tools._grant.get() is None
    assert approval.check_dangerous_command("rm -rf /tmp/x", "local")["approved"] is True


def test_the_peer_tool_fence_holds_only_while_this_claim_and_head_hold(target, monkeypatch):
    """The grant's fence runs before every peer tool batch and MCP send: it holds on the admitted turn
    and refuses once this event's receipt is gone or the route names another entry."""
    from agent import canonical_peer_tools

    peer, state = _peer(), {"receipt": "claimed", "entry": target.entry}
    get_meta, lookup = target.db.get_meta, target.runner.session_store.lookup_by_session_key_existing
    monkeypatch.setattr(target.db, "get_meta",
                        lambda key: state["receipt"] if key == peer["receipt"] else get_meta(key))
    monkeypatch.setattr(target.runner.session_store, "lookup_by_session_key_existing",
                        lambda key: state["entry"] if key == target.entry.session_key else lookup(key))
    outcomes = []

    def verify():
        try:
            canonical_peer_tools._grant.get().verify()
        except ValueError as exc:
            return str(exc)
        return "held"

    def observe(_agent):
        outcomes.append(verify())
        state["receipt"] = None
        outcomes.append(verify())
        state["receipt"], state["entry"] = "claimed", SimpleNamespace(**vars(target.entry))
        outcomes.append(verify())
        state["entry"] = target.entry

    target.actor.observe, target.actor.outcome = observe, "malformed"
    with pytest.raises(ValueError, match="^canonical_turn_refused$"):
        _run(target, peer=peer, **_proof(target))
    assert outcomes == ["held", "canonical_binding_stale", "canonical_binding_stale"]


# ── The same cached actor on the same session; nothing new ──────────────────────────────────────


def test_the_cached_actor_runs_on_its_own_session_and_nothing_is_built_or_created(target, monkeypatch):
    import run_agent

    def unexpected(*_args, **_kwargs):
        raise AssertionError("a canonical turn must not build an agent, or load, recover or create a session")

    monkeypatch.setattr(run_agent, "AIAgent", unexpected)
    for name in ("get_or_create_session", "lookup_by_session_key", "_route_recover"):
        monkeypatch.setattr(target.runner.session_store, name, unexpected, raising=False)
    with target.runner._agent_cache_lock:
        cached = target.runner._agent_cache[target.entry.session_key]
    sessions = target.sessions()
    peer = _peer()
    target.actor.outcome = "malformed"
    with pytest.raises(ValueError, match="^canonical_turn_refused$"):
        _run(target, peer=peer, **_proof(target))
    [call] = target.actor.calls
    assert call["task_id"] == target.entry.session_id
    assert call["persist_user_message"] == "hello"
    assert call["persist_user_display_kind"] == "canonical_peer"
    assert call["persist_user_display_metadata"] == {"canonical_peer": peer}
    assert call["text"].startswith("[Canonical event principal=peer") and "\nhello\n" in call["text"]
    with target.runner._agent_cache_lock:
        assert target.runner._agent_cache[target.entry.session_key] is cached
    assert target.sessions() == sessions
    assert target.runner.session_store.lookup_by_session_key_existing(target.entry.session_key) is target.entry


def test_a_missing_cached_actor_refuses_and_builds_nothing(target, monkeypatch):
    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", lambda *a, **kw: pytest.fail("no agent is built for a canonical turn"))
    with target.runner._agent_cache_lock:
        target.runner._agent_cache.pop(target.entry.session_key)
    sessions = target.sessions()
    with pytest.raises(ValueError, match="^canonical_agent_missing$"):
        _run(target)
    assert target.actor.calls == [] and target.sessions() == sessions
    with target.runner._agent_cache_lock:
        assert target.entry.session_key not in target.runner._agent_cache
