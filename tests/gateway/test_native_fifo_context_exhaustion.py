"""Native messages queued behind a turn that exhausts its context run once, in order, on the SAME
session: exhaustion keeps the chat's conversation (no reset, no re-home, no conversation boundary),
reports itself, and retracts the unanswered ask once the turn lease is released, before the next
queued turn starts. A transport retry of a queued message still dedupes. Real GatewayRunner,
DiscordAdapter, SessionAuthority and drain over SQLite; the agent run is a stand-in. Adapted from
helix4u's #139673 headline test."""
import asyncio
from types import SimpleNamespace

import pytest
import pytest_asyncio

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import list_session_admissions
from plugins.platforms.discord.adapter import DiscordAdapter

_REAL_SCHEDULE = SessionAuthority._schedule  # the fixture parks scheduling; races restore it
EXHAUSTED = {"failed": True, "partial": True, "completed": False, "compression_exhausted": True,
             "failure_reason": "context_overflow", "messages": []}


@pytest_asyncio.fixture
async def native(monkeypatch):
    monkeypatch.setenv("DISCORD_ALLOWED_USERS", "42")
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token", typing_indicator=False))

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="fixture-reply")

    def no_rotation(*_a, **_k):
        raise AssertionError("compression exhaustion must not reset the chat's session")

    monkeypatch.setattr(adapter, "send", send)
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id="default", instance_id="fixture")
    runner._wire_adapter_handlers(adapter)
    monkeypatch.setattr(runner.session_store, "reset_session", no_rotation)
    source = adapter.build_source(chat_id="42", chat_type="dm", user_id="42")
    try:
        yield SimpleNamespace(runner=runner, adapter=adapter, authority=authority, source=source, exhausted=[])
    finally:
        from gateway.session_runtime_workers import mutation_tasks
        await asyncio.gather(*mutation_tasks(authority))
        authority.db.close()


async def _admit(native, text, event=None):
    event = event or MessageEvent(text=text, source=native.source, message_id=text)
    with native_callback(native.runner, event, get_hermes_home()):
        return event, await native.authority.admit_native(event)


def _conversation(db, sid):
    return [(m["role"], m["content"]) for m in db.get_messages_as_conversation(sid)]


async def _exhaust(native, event, entry, session_key, source):
    """The exhausted turn the way production ends it (gateway/run_turn.py): the agent flushed the ask
    under its own turn lease and released the lease on return; the gateway then classifies, keeps the
    session and persists, which retracts the ask. Returns the reply the chat is sent."""
    from agent.turn_failure_copy import site_copy
    from gateway.session_results import execution_result
    runner, db, sid = native.runner, native.authority.db, entry.session_id
    holder = f"agent-turn-{event.text}"
    assert db.acquire_session_turn_lease(sid, holder, wait_seconds=0)
    db.append_message(sid, "user", event.text, turn_lease_holder=holder)
    db.release_session_turn_lease(sid, holder)
    result = dict(EXHAUSTED, final_response=site_copy("context_overflow", model="fixture-model"))
    execution_result.get()["result"] = result
    prepared = SimpleNamespace(history=[], persist_user_message=None, message_text=event.text,
                               persist_user_timestamp=None, persist_user_display_kind=None, persistence_owner=None)
    failed_early, hidden, overflow = runner._hmwa_classify_turn_failure(result, prepared.history, entry)
    response, kept = await runner._hmwa_compression_exhaustion_reset(
        result, result["final_response"], entry, session_key, source)
    assert kept is entry
    await runner._hmwa_persist_turn_transcript(
        event=event, source=source, session_entry=kept, session_key=session_key, agent_result=result,
        agent_messages=[], prepared=prepared, response=response, agent_failed_early=failed_early,
        hidden_reasoning_incomplete=hidden, is_context_overflow_failure=overflow)
    native.exhausted.append(event.text)  # the drain swallows a failed turn: tests check this ran to the end
    return response


@pytest.mark.asyncio
async def test_exhaustion_runs_queued_native_messages_on_the_same_session(native, monkeypatch):
    from gateway.session_results import execution_result
    runner, authority, db = native.runner, native.authority, native.authority.db
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    ref = admitted[0][1].ref
    live = authority.sessions[ref.session_id]
    sessions_before = db.session_count(include_archived=True, include_subagents=True)
    executed, late, replies = [], {}, {}

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        if event.text == "A":
            replies["A"] = await _exhaust(native, event, entry, session_key, source)
            late["D"] = await _admit(native, "D")  # admitted after the exhaustion, behind B and C
            return replies["A"]
        execution_result.get()["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(ref)
    sid = ref.session_id
    assert executed == [("A", sid), ("B", sid), ("C", sid), ("D", sid)] and native.exhausted == ["A"]
    # Session and actor kept: the chat's route still names this session's live record; no new row.
    assert runner.session_store.peek_session_id(live.route) == sid and authority.sessions[sid] is live
    assert late["D"][1].ref.session_id == sid
    assert db.session_count(include_archived=True, include_subagents=True) == sessions_before
    # Exhaustion is reported (the /compress guidance), never as a reset.
    assert "/compress" in replies["A"] and "reset" not in replies["A"].lower()
    assert ("user", "A") not in _conversation(db, sid)  # the unanswered ask was retracted
    rows = list_session_admissions(db, session_id=sid, pending_only=False)
    assert [(r["request_id"], r["status"]) for r in rows] == [(t, "terminal") for t in ("A", "B", "C", "D")]
    assert [r["outcome"] for r in rows[1:]] == ["completed"] * 3
    # Each queued message keeps its admission identity; a transport retry of B dedupes onto it.
    assert [r["admission_id"] for r in rows[1:3]] == [admitted[1][1].admission_id, admitted[2][1].admission_id]
    _, retry = await _admit(native, "B", admitted[1][0])  # the same delivery, redelivered
    assert retry.admission_id == admitted[1][1].admission_id
    assert [t for t, _ in executed].count("B") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("redeliver", [True, False])
async def test_exhaustion_runs_no_conversation_boundary_and_a_redelivery_dedupes(native, monkeypatch, redeliver):
    """No boundary runs on exhaustion: no conversation-scope clear, no topic re-bind, no extra FIFO
    schedule. A redelivery of queued B that arrives while A's exhaustion is being handled dedupes onto
    B's own row on the same session."""
    from gateway.session_results import execution_result
    runner, authority, db = native.runner, native.authority, native.authority.db
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    ref = admitted[0][1].ref
    events, executed, redelivery = [], [], {}
    monkeypatch.setattr(runner, "_clear_conversation_scope", lambda *a, **k: events.append("clear"))
    monkeypatch.setattr(runner, "_sync_telegram_topic_binding", lambda *a, **k: events.append("sync"))
    monkeypatch.setattr(SessionAuthority, "_schedule", lambda self, ref: events.append("schedule"))

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append(event.text)
        if event.text == "A":
            response = await _exhaust(native, event, entry, session_key, source)
            if redeliver:  # the transport redelivers B right after A exhausted
                redelivery["B"] = await _admit(native, "B", admitted[1][0])
            return response
        execution_result.get()["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(ref)
    assert "clear" not in events and "sync" not in events
    if redeliver:
        _, retry = redelivery["B"]
        assert retry.admission_id == admitted[1][1].admission_id and retry.ref.session_id == ref.session_id
    else:
        assert events == []
    assert executed == ["A", "B", "C"] and native.exhausted == ["A"]
    rows = list_session_admissions(db, session_id=ref.session_id, pending_only=False)
    assert [(r["request_id"], r["admission_id"], r["status"]) for r in rows] == [
        (text, receipt.admission_id, "terminal") for (event, receipt), text in zip(admitted, ("A", "B", "C"))]


@pytest.mark.asyncio
async def test_the_next_queued_turn_starts_after_the_unanswered_ask_is_retracted(native, monkeypatch):
    """With the real scheduler, B (queued behind the exhausted A) starts only once A's ask is gone:
    B's turn never sees it merged into its own (#107070), and B and C run on the same session."""
    from gateway.session_results import execution_result
    runner, authority, db = native.runner, native.authority, native.authority.db
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    ref = admitted[0][1].ref
    db.append_message(ref.session_id, "user", "hello")
    db.append_message(ref.session_id, "assistant", "hi there")
    executed, seen = [], {}

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        seen[event.text] = _conversation(db, entry.session_id)
        if event.text == "A":
            return await _exhaust(native, event, entry, session_key, source)
        execution_result.get()["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    monkeypatch.setattr(SessionAuthority, "_schedule", _REAL_SCHEDULE)
    authority._schedule(ref)
    for _ in range(300):
        if len(executed) == 3:
            break
        await asyncio.sleep(0.02)
    sid = ref.session_id
    assert executed == [("A", sid), ("B", sid), ("C", sid)] and native.exhausted == ["A"]
    assert seen["A"][-1] == ("assistant", "hi there")  # the turn-start view, before A's own flush
    assert seen["B"] == seen["C"] == [("user", "hello"), ("assistant", "hi there")]


@pytest.mark.asyncio
async def test_an_admission_reserved_during_exhaustion_commits_on_the_same_session(native, monkeypatch):
    """A retry of queued B that reserved the route while A ran commits after A exhausted, deduped onto
    its own row (same admission id, same session); D, the chat's next input, waits for it at the
    route's admission turn and queues behind it there."""
    import threading
    from gateway.session_results import execution_result
    runner, authority, db = native.runner, native.authority, native.authority.db
    loop = asyncio.get_running_loop()
    admitted = [await _admit(native, text) for text in ("A", "B", "C")]
    ref = admitted[0][1].ref
    reserve_real, held, release = authority._reserve_route, [], threading.Event()

    def held_reserve(src):
        answer = reserve_real(src)
        loop.call_soon_threadsafe(held.append, answer)
        assert release.wait(10)
        return answer

    monkeypatch.setattr(authority, "_reserve_route", held_reserve)
    late = [asyncio.create_task(_admit(native, "B", admitted[1][0])), asyncio.create_task(_admit(native, "D"))]
    for _ in range(250):
        if held:
            break
        await asyncio.sleep(0.02)
    await asyncio.wait({late[1]}, timeout=0.2)
    assert [sid for sid, _ in held] == [ref.session_id]  # D waits behind B's admission
    executed = []

    async def turn(event, source, session_key, run_generation):
        entry = runner.session_store.lookup_by_session_key(session_key)
        executed.append((event.text, entry.session_id))
        if event.text == "A":
            return await _exhaust(native, event, entry, session_key, source)
        execution_result.get()["result"] = {"final_response": "reply", "messages": []}
        return "reply"

    monkeypatch.setattr(runner, "_handle_message_with_agent", turn)
    await authority._drain(ref)  # A exhausts while B/D hold their reservations
    release.set()
    (_, retry), (_, late_d) = await asyncio.gather(*late)
    sid = ref.session_id
    assert retry.admission_id == admitted[1][1].admission_id
    assert retry.ref.session_id == late_d.ref.session_id == sid
    await authority._drain(ref)
    assert executed == [("A", sid), ("B", sid), ("C", sid), ("D", sid)] and native.exhausted == ["A"]
    assert [r["request_id"] for r in list_session_admissions(db, session_id=sid, pending_only=False)] == [
        "A", "B", "C", "D"]


@pytest.mark.asyncio
async def test_exhaustion_leaves_the_adapter_queue_where_it_is(native):
    """A /queue chain deferred behind the exhausted turn (head in the adapter slot, rest in the
    overflow) stays exactly there, once each, for the post-handler drain: no boundary clears it and
    nothing re-queues it."""
    runner, adapter, source = native.runner, native.adapter, native.source
    _, receipt = await _admit(native, "A")
    session_key = runner.session_store.lookup_by_session_key(
        native.authority.sessions[receipt.ref.session_id].route).session_key
    head, second = (MessageEvent(text=t, source=source, message_id=t) for t in ("Q1", "Q2"))
    runner._enqueue_fifo(session_key, head, adapter)
    runner._enqueue_fifo(session_key, second, adapter)
    entry = runner.session_store.lookup_by_session_key(session_key)
    response, kept = await runner._hmwa_compression_exhaustion_reset(
        dict(EXHAUSTED), "context exhausted", entry, session_key, source)
    assert kept is entry and response == "context exhausted"
    assert adapter._pending_messages[session_key] is head
    assert runner._overflow_queue(session_key) == [second]
