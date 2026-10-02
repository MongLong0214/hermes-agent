"""A canonical expected identity is checked before a durable event claim."""

import asyncio
import json
import os
import sqlite3

import pytest

from gateway.canonical_surface import (
    CanonicalIngressEvent, CanonicalReceiptCoordinator, CanonicalReceiptResult,
    CanonicalSurfaceBinding,
)
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource
from hermes_state_target_bind import _lineage_root_digest


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

    class Actor:
        compression_in_place = True
        session_id = entry.session_id
        _session_db = db

        def __init__(self):
            self.calls = []

        def run_conversation(self, text, *, conversation_history, task_id, **_kwargs):
            self.calls.append(text)
            self._persist_user_message_idx = len(conversation_history)
            return {"completed": True, "session_id": task_id, "final_response": "terminal",
                    "messages": [*conversation_history, {"role": "user", "content": text},
                                 {"role": "assistant", "content": "terminal"}]}

        def interrupt(self, text=None):
            pass

        def release_clients(self):
            pass

    actor = Actor()
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)

    def receipt_rows():
        with sqlite3.connect(db.db_path) as conn:
            return conn.execute("SELECT key FROM state_meta WHERE key LIKE 'canonical-receipt:%'").fetchall()

    expected = {
        "session_id": entry.session_id,
        "lineage_root_digest": _lineage_root_digest(entry.session_id),
        "process_pid": os.getpid(),
        "process_started_at": __import__("gateway.platforms.api_server_canonical", fromlist=["_process_started_at"])._process_started_at(os.getpid()),
    }
    assert expected["process_started_at"] is not None
    try:
        yield runner, actor, db, entry, binding, expected, receipt_rows
    finally:
        runner.session_store.close_all_db_handles()


def _event(**changes):
    payload = {"binding": "canonical", "event_id": "one", "author_id": "author",
               "channel_id": "channel", "text": "hello", **changes}
    return CanonicalIngressEvent.from_json_bytes(json.dumps(payload).encode())


def test_acp_flat_identity_requires_all_four_valid_fields(target):
    runner, actor, db, entry, binding, expected, receipts = target
    invalid = [
        {"session_id": expected["session_id"]},
        {**expected, "process_pid": True},
        {**expected, "process_pid": 0},
        {**expected, "lineage_root_digest": expected["lineage_root_digest"].upper()},
        {**expected, "process_started_at": ""},
        {**expected, "extra": "unexpected"},
        {"expected_identity": expected},
    ]
    for value in invalid:
        with pytest.raises(ValueError, match="canonical_invalid_request"):
            _event(**value)
    assert _event().expected_identity is None
    assert receipts() == []


def test_held_lease_rechecks_same_head_object_and_refuses_mismatch_without_claim(target, monkeypatch):
    runner, actor, db, entry, binding, expected, receipts = target
    coordinator = CanonicalReceiptCoordinator(runner)
    event = _event(**expected)
    calls = []
    original = runner._turn_leases.acquire

    async def acquire(*args, **kwargs):
        lease = await original(*args, **kwargs)
        # The routing entry is the very same object but its live session id has moved.
        entry.session_id = "different-head"
        return lease

    async def unexpected(*args, **kwargs):
        calls.append("actor")
        raise AssertionError("actor must not run before identity proof")

    monkeypatch.setattr(runner._turn_leases, "acquire", acquire)
    monkeypatch.setattr(runner, "run_bound_existing_turn", unexpected)
    monkeypatch.setattr(db, "claim_meta_once", lambda *a, **kw: calls.append("claim"))
    with pytest.raises(ValueError, match="canonical_binding_stale"):
        asyncio.run(coordinator.submit(binding, event))
    assert calls == [] and receipts() == []


def test_expected_identity_mismatch_refuses_without_claim(target, monkeypatch):
    runner, actor, db, entry, binding, expected, receipts = target
    calls = []
    monkeypatch.setattr(db, "claim_meta_once", lambda *a, **kw: calls.append("claim"))
    for change in (
        {"session_id": "old-session"},
        {"lineage_root_digest": "sha256:" + "0" * 64},
        {"process_pid": os.getpid() + 1},
        {"process_started_at": "darwin-tv:1.000000"},
    ):
        event = _event(**{**expected, **change})
        with pytest.raises(ValueError, match="canonical_binding_stale"):
            asyncio.run(CanonicalReceiptCoordinator(runner).submit(binding, event))
    assert calls == [] and receipts() == []


def test_matching_identity_runs_one_real_cached_actor_turn(target):
    runner, actor, db, entry, binding, expected, receipts = target
    event = _event(**expected)
    result = asyncio.run(CanonicalReceiptCoordinator(runner).submit(binding, event))
    assert result == CanonicalReceiptResult("terminal", "terminal")

    def _is_wrapped_peer_turn(call):
        # The actor now receives the full peer-provenance envelope (render_peer_turn), not the
        # bare event text -- the envelope's nonce is random per render, so an exact-string
        # assertion is neither possible nor meaningful; check the structural contract instead.
        return (call.startswith("[Canonical event principal=peer") and "<<<peer-body:" in call
                and "\nhello\n" in call)

    assert len(actor.calls) == 1 and _is_wrapped_peer_turn(actor.calls[0]) and len(receipts()) == 1
    assert asyncio.run(CanonicalReceiptCoordinator(runner).submit(binding, event)) == result
    assert len(actor.calls) == 1, "a replayed receipt must not run the actor a second time"
