"""The canonical surface admits an event only into an existing, live, bound session, and runs it once.

The resolver and the receipt coordinator run against a real ``GatewayRunner`` and a real SQLite
state.db under a tmp home. Five seams that later packets provide are stood in for on the instance
(see ``gateway_under_test``): the bound-turn runner (``run_bound_existing_turn``), the agent build for
a resolved session, the post-peer queue dispatch, the store's existing-only lookup and its reset
reason, and the writer's proven-handle ``state_meta`` claim / compare-and-set.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from gateway.canonical_surface import (
    CanonicalIngressEvent,
    CanonicalReceiptCoordinator,
    CanonicalReceiptResult,
    CanonicalSurfaceBinding,
    CanonicalTurnResult,
    ExistingCanonicalBindingResolver,
)
from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource

BINDING = "receipt-binding"


def with_receipt_writes(db):
    """Attach the proven-handle ``state_meta`` writes the coordinator claims and settles with to this
    writer: an insert that never overwrites and a compare-and-set on the exact expected value, both
    refused unless the caller names this handle's path and file identity."""

    def require_proof(path, identity):
        if path != db.db_path or identity is None or identity != db._db_file_identity:
            raise sqlite3.ProgrammingError("state metadata write requires a matching proven handle")
        if db.read_only or db._conn is None:
            raise sqlite3.OperationalError("cannot write state metadata on this handle")
        db._raise_if_db_replaced()

    def claim_meta_once(key, value, *, proven_db_path, proven_db_identity):
        with db._lock:
            require_proof(proven_db_path, proven_db_identity)
            cursor = db._conn.execute("INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)", (key, value))
            return cursor.rowcount == 1

    def compare_and_set_meta(key, expected, value, *, proven_db_path, proven_db_identity):
        with db._lock:
            require_proof(proven_db_path, proven_db_identity)
            cursor = db._conn.execute("UPDATE state_meta SET value = ? WHERE key = ? AND value = ?",
                                      (value, key, expected))
            return cursor.rowcount == 1

    db.claim_meta_once = claim_meta_once
    db.compare_and_set_meta = compare_and_set_meta
    return db


class Actor:
    """The bound cached agent: no model call; run n answers ``answer #n`` (n counts this actor's runs)."""

    api_mode = "chat_completions"

    def __init__(self, session_id, db):
        self.session_id = session_id
        self._session_db = db
        self.calls = []
        self.during_turn = None

    def run(self, text):
        self.calls.append(text)
        if self.during_turn is not None:
            self.during_turn()
        return f"answer #{len(self.calls)}"


def gateway_under_test(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    store = runner.session_store
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = store.get_or_create_session(source)
    db = with_receipt_writes(store._db)
    actor = Actor(entry.session_id, db)
    with runner._agent_cache_lock:
        runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)

    created = []

    def refuse_creation(*args, **kwargs):
        created.append(args)
        raise AssertionError("the canonical surface created a session")

    store.lookup_by_session_key_existing = lambda key: store._entries.get(key)
    store._should_reset = lambda _entry, _origin: None
    store.get_or_create_session = refuse_creation
    store.reset_session = refuse_creation

    turns = []

    async def run_bound_existing_turn(binding, event, bound_entry, *, reply_sink, expected_actor,
                                      expected_session_db, expected_db_path, expected_db_identity,
                                      expected_run_generation, held_lease, peer, observation):
        assert bound_entry is entry and expected_actor is actor and expected_session_db is db
        turns.append(event.event_id)
        observation.actor_started = True
        text = expected_actor.run(event.text)
        observation.actor_finished = True
        return CanonicalTurnResult(binding.name, text)

    async def build_agent_for_resolved_session(_entry):
        with runner._agent_cache_lock:
            runner._agent_cache[entry.session_key] = (actor, "exact", 0, entry.session_id)

    runner.run_bound_existing_turn = run_bound_existing_turn
    runner._build_agent_for_resolved_session = build_agent_for_resolved_session
    runner._dispatch_queued_turn_after_peer_turn = lambda _key: None

    binding = CanonicalSurfaceBinding(
        name=BINDING, session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id=source.chat_id, telegram_chat_type=source.chat_type,
        telegram_user_id=source.user_id, telegram_thread_id=source.thread_id,
        allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    return SimpleNamespace(runner=runner, store=store, entry=entry, source=source, db=db, actor=actor,
                           binding=binding, created=created, turns=turns, home=home,
                           coordinator=CanonicalReceiptCoordinator(runner))


@pytest.fixture
def gw(tmp_path, monkeypatch):
    gateway = gateway_under_test(tmp_path, monkeypatch)
    try:
        yield gateway
    finally:
        gateway.store.close_all_db_handles()


def event(text="hello", event_id="evt-1", *, author="author", channel="channel"):
    return CanonicalIngressEvent(BINDING, event_id, author, channel, text)


def canonical_rows(db):
    with db._lock:
        return sorted(db._conn.execute(
            "SELECT key, value FROM state_meta WHERE key LIKE 'canonical-%'").fetchall())


def session_ids(db):
    with db._lock:
        return sorted(row[0] for row in db._conn.execute("SELECT id FROM sessions").fetchall())


def submit(gw, *events):
    async def run():
        return await asyncio.gather(*(gw.coordinator.submit(gw.binding, item) for item in events),
                                    return_exceptions=True)
    results = asyncio.run(run())
    return results if len(results) > 1 else results[0]


def stale_binding(gw, stale):
    if stale == "ended_session":
        gw.db.end_session(gw.entry.session_id, "test")
        return gw.binding
    changes = {"unknown_key": {"session_key": "agent:main:telegram:dm:nobody"},
               "other_chat": {"telegram_chat_id": "another-chat"},
               "other_thread": {"telegram_thread_id": "42"}}[stale]
    return dataclasses.replace(gw.binding, **changes)


# --- the resolver ---------------------------------------------------------------------------------


@pytest.mark.parametrize("stale", ["unknown_key", "other_chat", "other_thread", "ended_session"])
def test_resolver_refuses_an_unknown_or_stale_binding_and_never_creates_a_session(gw, stale):
    before = session_ids(gw.db)
    binding = stale_binding(gw, stale)

    with pytest.raises(ValueError, match="^canonical_binding_stale$"):
        ExistingCanonicalBindingResolver(gw.store).resolve(binding, event())
    with pytest.raises(ValueError, match="^canonical_binding_stale$"):
        asyncio.run(gw.coordinator.submit(binding, event()))

    assert gw.created == [] and session_ids(gw.db) == before
    assert gw.turns == [] and canonical_rows(gw.db) == []


def test_resolver_refuses_an_author_or_channel_the_binding_does_not_admit(gw):
    for refused in (event(author="stranger"), event(channel="another-room")):
        with pytest.raises(ValueError, match="^canonical_principal_rejected$"):
            asyncio.run(gw.coordinator.submit(gw.binding, refused))
    assert gw.turns == [] and canonical_rows(gw.db) == [] and gw.created == []


def test_resolver_answers_the_live_head_with_a_proof_of_its_db_file(gw):
    entry, proof = ExistingCanonicalBindingResolver(gw.store).resolve_with_proof(gw.binding, event())
    assert entry is gw.entry
    assert (proof.session_key, proof.session_id) == (gw.entry.session_key, gw.entry.session_id)
    assert proof.db_identity == gw.db._db_file_identity


# --- the receipt ----------------------------------------------------------------------------------


def test_a_duplicate_event_id_answers_the_same_receipt_and_never_runs_a_second_turn(gw):
    first = submit(gw, event())
    second = submit(gw, event())
    assert first.status == "terminal" and first.terminal_text == "answer #1"
    assert second == first
    assert gw.turns == ["evt-1"]


def test_concurrent_duplicates_run_once_and_answer_only_the_recorded_text(gw):
    results = submit(gw, event(), event(), event())
    assert gw.turns == ["evt-1"]
    # A duplicate that arrives while the turn runs may see it pending; none sees another answer.
    assert not any(isinstance(item, BaseException) for item in results)
    assert all((item.status, item.terminal_text) in {("terminal", "answer #1"), ("pending", None)}
               for item in results)
    assert submit(gw, event()) == CanonicalReceiptResult("terminal", "answer #1")


def test_a_duplicate_after_the_actor_left_the_cache_answers_from_the_receipt_without_a_turn(gw):
    first = submit(gw, event())
    with gw.runner._agent_cache_lock:
        gw.runner._agent_cache.clear()
    assert submit(gw, event()) == first
    assert gw.turns == ["evt-1"]


def test_the_same_event_id_with_changed_text_is_a_conflict_never_a_second_turn(gw):
    submit(gw, event())
    with pytest.raises(ValueError, match="^canonical_receipt_conflict$"):
        asyncio.run(gw.coordinator.submit(gw.binding, event(text="changed")))
    assert gw.turns == ["evt-1"]


def test_a_new_event_id_runs_its_own_turn(gw):
    assert submit(gw, event()).terminal_text == "answer #1"
    assert submit(gw, event(event_id="evt-2")).terminal_text == "answer #2"
    assert gw.turns == ["evt-1", "evt-2"]


# --- the history boundary -------------------------------------------------------------------------


def test_an_unanswered_owner_tail_refuses_with_canonical_history_unanswered_and_claims_nothing(gw):
    gw.db.append_message(gw.entry.session_id, "user", "the owner is still waiting for an answer")
    with pytest.raises(ValueError, match="^canonical_history_unanswered$"):
        asyncio.run(gw.coordinator.submit(gw.binding, event()))
    assert gw.turns == [] and canonical_rows(gw.db) == []
    assert not gw.runner._is_session_running(gw.entry.session_key)

    gw.db.append_message(gw.entry.session_id, "assistant", "answered")
    assert submit(gw, event()).terminal_text == "answer #1"


def test_an_open_tail_of_recorded_delegation_deliveries_is_not_an_unanswered_owner(gw):
    gw.db.append_message(gw.entry.session_id, "user", "[delegation complete]",
                         display_kind="async_delegation_complete", display_metadata={"delegation_id": "d-1"})
    assert submit(gw, event()).terminal_text == "answer #1"
    assert gw.turns == ["evt-1"]


def test_the_turn_lease_and_running_slot_are_released_after_the_turn(gw):
    submit(gw, event())
    assert not gw.runner._is_session_running(gw.entry.session_key)
    lease_free = threading.Event()

    async def take_lease():
        lease = await gw.runner._turn_leases.acquire(gw.entry.session_id, owner_key="probe", generation=0,
                                                    timeout=2)
        lease_free.set()
        gw.runner._turn_leases.release(lease)

    asyncio.run(take_lease())
    assert lease_free.is_set()
