"""An owner message that arrives while a canonical peer turn runs is answered to the owner.

Production 2026-11-13 22:27 and 22:32 KST: the CEO session was running a canonical peer turn (a CTO's
Buzz event through ``/v1/canonical-surface/events``) when the owner wrote on Telegram. The busy path
(``busy_input_mode: interrupt``) redirected the text into the running turn; the redirect landed during
a tool call, so it degraded to a steer. The model answered the owner inside the peer turn, that answer
became the canonical answer posted to the CTO's thread, and nothing was sent to Telegram.

A peer turn is the turn the canonical surface runs under its turn lease (owner key ``canonical:``).
While it holds the session, an owner follow-up is never steered, redirected or interrupted into it: it
is queued in the session's FIFO (which a restart flushes and recovers), and when the peer turn
releases the session it hands it to the queue head in the same step, so no other peer event can be
admitted before that owner turn has claimed. Owner-to-owner busy behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from tests.gateway.restart_test_helpers import RestartTestAdapter
from tests.gateway.test_canonical_surface_ingress import (  # noqa: F401 - fixtures
    _TERMINAL,
    _CachedActor,
    _clean_restart,
    _code,
    ingress,
    send,
)
from tests.gateway.test_multiplex_transport_matrix import mux  # noqa: F401 - fixture

_OWNER_TEXT = "owner: are you there?"
_MODES = ("steer", "interrupt", "queue")
_QUEUED_ACK = "⏳ Queued for the next turn"


class _Actor(_CachedActor):
    """The cached actor, with the agent's busy verbs: a steered or redirected text reaches the model
    inside the running turn and so ends up in that turn's answer, as it did in production."""

    _supports_active_turn_redirect = True

    def __init__(self, session_id, db):
        super().__init__(session_id, db)
        self.steered, self.redirected, self.interrupted = [], [], []
        self.hold = None  # (entered, release) events: block the next turn mid-run

    def steer(self, text):
        self.steered.append(text)
        return True

    def redirect(self, text):
        # During a tool call redirect() degrades to steer() (agent.interrupt_control).
        self.redirected.append(text)
        return self.steer(text)

    def interrupt(self, text=None):
        self.interrupted.append(text)

    def get_activity_summary(self):
        return {}

    def run_conversation(self, text, *, conversation_history, task_id, **kwargs):
        hold, self.hold = self.hold, None
        if hold is not None:
            entered, release = hold
            entered.set()
            assert release.wait(10)
        result = super().run_conversation(
            text, conversation_history=conversation_history, task_id=task_id, **kwargs)
        if self.steered:
            # The model read the injected owner text and answered it in this turn.
            answer = f"{_TERMINAL} / answering: " + " | ".join(self.steered)
            result["final_response"] = answer
            result["messages"][-1] = {"role": "assistant", "content": answer}
            self.steered = []
        return result


class _Harness:
    def __init__(self, ingress, send, monkeypatch, mode, *, grace="0"):
        import run_agent

        # Production: the owner wrote 70s into the peer turn, past Telegram's post-start grace queue.
        monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", grace)
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "chat")  # no first-turn /sethome onboarding notice

        self.runner = runner = _clean_restart(ingress, send)
        self.ingress, self.send = ingress, send
        self.key = ingress.entry.session_key
        self.built = []

        def build(**kwargs):
            actor = _Actor(kwargs["session_id"], kwargs["session_db"])
            self.built.append(actor)
            return actor

        monkeypatch.setattr(run_agent, "AIAgent", build)
        monkeypatch.setattr(runner, "_resolve_session_agent_runtime", lambda **kw: (
            "test-model", {"provider": "openrouter", "api_key": "test-key", "base_url": "http://127.0.0.1:9"}))

        self.admission_gates = {}  # text -> asyncio.Event: admission of that message waits on it

        async def admitted(event):
            gate = self.admission_gates.get(event.text)
            if gate is not None:
                await gate.wait()
            return event, event.source, False
        monkeypatch.setattr(runner, "_hm_admit_event", admitted)

        # Holds a message between the running check and its claim, at an await every message passes.
        self.preparation_gates, self.preparing = {}, set()
        dispatch_idle = runner._hm_dispatch_idle_commands

        async def prepared(event, source, quick_key):
            gate = self.preparation_gates.get(event.text)
            if gate is not None:
                self.preparing.add(event.text)
                await gate.wait()
            return await dispatch_idle(event, source, quick_key)
        monkeypatch.setattr(runner, "_hm_dispatch_idle_commands", prepared)

        monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
        runner._busy_input_mode = mode
        runner._busy_text_mode = "interrupt"
        self.adapter = adapter = RestartTestAdapter()
        adapter.gateway_runner = runner
        adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
        runner.adapters[Platform.TELEGRAM] = adapter
        self.entered, self.release = threading.Event(), threading.Event()
        self.handled = []  # every event the runner's message handler was entered with
        handle = runner._handle_message

        async def counted(event):
            self.handled.append(event.text)
            return await handle(event)
        adapter.set_message_handler(counted)

    @property
    def actor(self):
        [actor] = self.built
        return actor

    def owner_event(self, text=_OWNER_TEXT, message_id="m1"):
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=self.ingress.source,
                            message_id=message_id)

    async def warm(self):
        """A first canonical event builds and caches the session's agent (a clean restart)."""
        assert await self.send(event_id="warm") == (200, {"event_id": "warm", "text": _TERMINAL})

    async def wait(self, predicate, what):
        for _ in range(500):
            if predicate():
                return
            await asyncio.sleep(0.01)
        pytest.fail(f"timed out waiting for {what}")

    async def peer_turn_running(self, event_id="peer", text="cto: status?"):
        """Start a canonical peer turn and hold it mid-run."""
        self.entered.clear()
        self.release.clear()
        self.actor.hold = (self.entered, self.release)
        peer = asyncio.ensure_future(self.send(event_id=event_id, text=text))
        await self.wait(self.entered.is_set, "the peer turn to start")
        assert self.runner._canonical_peer_turn_running(self.key)
        return peer

    def owner_replies(self):
        return [(chat, text) for chat, text, _meta in self.adapter.sent_calls]

    def queued(self):
        head = self.adapter._pending_messages.get(self.key)
        overflow = self.runner._overflow_queue(self.key) or []
        return [e.text for e in ([head] if head is not None else []) + list(overflow)]

    async def settle(self):
        for _ in range(500):
            tasks = [t for t in self.adapter._session_tasks.values() if not t.done()]
            if not tasks and not self.runner._is_session_running(self.key):
                return
            await asyncio.sleep(0.01)
        pytest.fail("the owner turn never finished")


def _no_busy_verbs(h):
    assert (h.actor.steered, h.actor.redirected, h.actor.interrupted) == ([], [], [])


@pytest.mark.parametrize("grace", ["0", "30"], ids=["after_grace", "within_grace"])
@pytest.mark.parametrize("mode", _MODES)
def test_owner_text_during_a_peer_turn_is_answered_to_the_owner_never_inside_the_peer_turn(
        ingress, send, monkeypatch, mode, grace):
    h = _Harness(ingress, send, monkeypatch, mode, grace=grace)

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await asyncio.sleep(0.5)  # the follow-up has had every chance to reach the running turn
        _no_busy_verbs(h)
        # Queued in the session's FIFO once, acknowledged, and not re-dispatched while it waits.
        assert h.queued() == [_OWNER_TEXT] and h.handled == [_OWNER_TEXT]
        [(chat, ack)] = h.owner_replies()
        assert chat == ingress.source.chat_id and ack.startswith(_QUEUED_ACK)
        h.release.set()
        status, body = await peer
        await h.settle()
        return status, body

    status, body = asyncio.run(exercise())
    # The peer's canonical answer is the peer turn's own; nothing of the owner's text is in it.
    assert (status, body) == (200, {"event_id": "peer", "text": _TERMINAL})
    # The owner turn ran as its own turn after the peer turn, and its reply went to the owner's chat.
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT]
    assert h.owner_replies()[1:] == [(ingress.source.chat_id, _TERMINAL)]
    assert h.queued() == []


@pytest.mark.parametrize("mode", _MODES)
def test_owner_messages_during_a_peer_turn_run_in_arrival_order_after_it(ingress, send, monkeypatch, mode):
    """The second message reaches the adapter's busy handler (gateway.run_busy) while the first one's
    task still holds the session: it is queued in every mode, never steered or redirected."""
    h = _Harness(ingress, send, monkeypatch, mode)
    first, second = h.owner_event(), h.owner_event("owner: second", message_id="m2")
    h.admission_gates[_OWNER_TEXT] = gate = asyncio.Event()

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(first)  # its task holds the adapter's session guard
        await h.wait(lambda: h.handled, "the first owner message to be handled")
        await h.adapter.handle_message(second)
        gate.set()
        await asyncio.sleep(0.3)
        _no_busy_verbs(h)
        assert sorted(h.queued()) == sorted([_OWNER_TEXT, "owner: second"])
        order = h.queued()
        h.release.set()
        result = await peer
        await h.settle()
        return result, order

    result, order = asyncio.run(exercise())
    assert result == (200, {"event_id": "peer", "text": _TERMINAL})
    assert h.actor.calls == ["hello", "cto: status?", *order]
    assert h.owner_replies()[-2:] == [(ingress.source.chat_id, _TERMINAL)] * 2


def test_a_message_admitted_across_the_hand_off_queues_behind_the_reserved_turn(ingress, send, monkeypatch):
    """A message still in admission when the peer turn hands the session on holds the adapter's
    session guard, so the queue head is only reserved and that task dispatches it as it ends. The
    message itself finds the session reserved and is queued behind the head as its own turn, never
    merged into it."""
    h = _Harness(ingress, send, monkeypatch, "interrupt")
    h.admission_gates["owner: second"] = gate = asyncio.Event()

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.queued() == [_OWNER_TEXT] and all(
            t.done() for t in h.adapter._session_tasks.values()), "the first message to be queued")
        await h.adapter.handle_message(h.owner_event("owner: second", message_id="m2"))
        await h.wait(lambda: "owner: second" in h.handled, "the second message to be in admission")
        h.release.set()
        result = await peer
        reserved = h.runner._is_session_running(h.key) and h.actor.calls[-1] == "cto: status?"
        gate.set()
        await h.settle()
        return result, reserved

    result, reserved = asyncio.run(exercise())
    assert result == (200, {"event_id": "peer", "text": _TERMINAL}) and reserved
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT, "owner: second"]
    assert h.owner_replies()[-2:] == [(ingress.source.chat_id, _TERMINAL)] * 2


def test_steer_command_during_the_peer_turn_is_queued_as_its_own_turn(ingress, send, monkeypatch):
    h = _Harness(ingress, send, monkeypatch, "steer")

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event("/steer owner: look at this"))
        await asyncio.sleep(0.5)
        _no_busy_verbs(h)
        h.release.set()
        result = await peer
        await h.settle()
        return result

    assert asyncio.run(exercise()) == (200, {"event_id": "peer", "text": _TERMINAL})
    assert h.actor.calls == ["hello", "cto: status?", "owner: look at this"]
    replies = h.owner_replies()
    assert replies[0][1] == "A peer turn is running — /steer queued for the next turn."
    assert replies[1:] == [(ingress.source.chat_id, _TERMINAL)]


@pytest.mark.parametrize("path", ["busy_handler", "priority"])
@pytest.mark.parametrize("mode", _MODES)
def test_owner_text_during_an_owner_turn_keeps_its_busy_mode(ingress, send, monkeypatch, mode, path):
    """Control: an owner follow-up during the owner's own turn is steered, redirected or queued
    exactly as before, on both busy entry points."""
    h = _Harness(ingress, send, monkeypatch, mode)
    follow_up = h.owner_event("owner: second", message_id="m2")

    async def exercise():
        await h.warm()
        h.actor.hold = (h.entered, h.release)
        await h.adapter.handle_message(h.owner_event())
        await h.wait(h.entered.is_set, "the owner turn to start")
        state = h.runner._peek_session_state(h.key)
        await h.wait(lambda: state.turn.agent is h.actor, "the owner turn's agent to be published")
        # The owner turn holds the session's turn lease under its own routing key.
        holder = h.runner._turn_leases._leases[ingress.entry.session_id].holder
        assert holder.owner_key == h.key
        if path == "busy_handler":
            await h.adapter.handle_message(follow_up)
        else:
            assert await h.runner._handle_message(follow_up) is None
        seen = (list(h.actor.steered), list(h.actor.redirected))
        queued = h.adapter._pending_messages.get(h.key) is follow_up
        h.release.set()
        await h.settle()
        return seen, queued

    (steered, redirected), queued = asyncio.run(exercise())
    assert _records(ingress) == []  # queued, steered or redirected: settled with the turn that ran it
    if mode == "queue":
        assert (steered, redirected, queued) == ([], [], True)
        assert h.actor.calls == ["hello", _OWNER_TEXT, "owner: second"]
    else:
        assert len(steered) == 1 and steered[0].endswith("owner: second") and not queued
        assert redirected == (steered if mode == "interrupt" else [])
        assert h.actor.calls == ["hello", _OWNER_TEXT]  # it landed inside the running owner turn


def test_the_peer_turn_marker_is_the_canonical_turn_lease(ingress, send, monkeypatch):
    """The marker is the canonical claim's own turn lease: held for the whole peer turn, absent for
    an owner turn on the same actor and once the peer turn has ended."""
    h = _Harness(ingress, send, monkeypatch, "steer")

    async def exercise():
        await h.warm()
        assert not h.runner._canonical_peer_turn_running(h.key)
        peer = await h.peer_turn_running()
        lease = h.runner._turn_leases._leases[ingress.entry.session_id]
        assert lease.holder.owner_key.startswith("canonical:")
        during = h.runner._canonical_peer_turn_running(h.key)
        h.release.set()
        await peer
        return during, h.runner._canonical_peer_turn_running(h.key)

    assert asyncio.run(exercise()) == (True, False)


def test_a_peer_event_arriving_as_the_peer_turn_ends_waits_for_the_queued_owner_turn(
        ingress, send, monkeypatch):
    """Review r1 P1-b: the queued owner turn holds the session from the peer turn's release until it
    claims, however long its preparation awaits. A peer event in between is refused
    canonical_turn_busy instead of running and having its answer invalidated by the owner's claim."""
    h = _Harness(ingress, send, monkeypatch, "interrupt")
    h.preparation_gates[_OWNER_TEXT] = gate = asyncio.Event()

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.queued() == [_OWNER_TEXT], "the owner message to be queued")
        h.release.set()
        first = await peer
        await h.wait(lambda: _OWNER_TEXT in h.preparing, "the owner turn to be preparing")
        second = await h.send(event_id="peer-b", text="cto: next")
        gate.set()
        await h.settle()
        return first, second

    first, second = asyncio.run(exercise())
    assert first == (200, {"event_id": "peer", "text": _TERMINAL})
    assert _code(second) == (409, "canonical_turn_busy")
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT]
    assert h.owner_replies()[1:] == [(ingress.source.chat_id, _TERMINAL)]


def test_an_owner_message_never_overwrites_a_peer_turn_admitted_while_it_was_prepared(
        ingress, send, monkeypatch):
    """Review r1 P1-b, unreserved: an owner message on an idle session awaits before its claim; a
    peer turn admitted in that window keeps the session and its answer, and the owner message is
    queued behind it instead of overwriting its running slot and generation."""
    h = _Harness(ingress, send, monkeypatch, "interrupt")
    h.preparation_gates[_OWNER_TEXT] = gate = asyncio.Event()

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: _OWNER_TEXT in h.preparing, "the owner message to be preparing")
        peer = await h.peer_turn_running()
        gate.set()
        await h.wait(lambda: h.queued() == [_OWNER_TEXT], "the owner message to queue behind the peer turn")
        _no_busy_verbs(h)
        h.release.set()
        result = await peer
        await h.settle()
        return result

    assert asyncio.run(exercise()) == (200, {"event_id": "peer", "text": _TERMINAL})
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT]
    assert h.owner_replies()[-1] == (ingress.source.chat_id, _TERMINAL)


def _capture_shutdown_flush(monkeypatch):
    """The texts the shutdown flush writes out for the next process (pending slots and overflow)."""
    import gateway.shutdown_flush as shutdown_flush

    flushed = []
    monkeypatch.setattr(shutdown_flush, "flush_pending_to_file", lambda pending, **kw: flushed.extend(
        getattr(v, "text", v) for v in pending.values()) or len(pending))
    monkeypatch.setattr(shutdown_flush, "flush_overflow_to_file", lambda overflow, **kw: flushed.extend(
        getattr(v, "text", v) for events in overflow.values() for v in events) or 0)
    return flushed


@pytest.mark.parametrize("when", ["peer_turn_ended", "peer_turn_running"])
def test_a_restart_keeps_the_owner_message_queued_behind_a_peer_turn(ingress, send, monkeypatch, when):
    """Review r1 P1-a: a restart after the owner message was queued never loses it. Stopped right after
    the peer turn handed the session on (the owner turn still preparing) or while the peer turn still
    runs, the message stays in the FIFO and the shutdown flush writes it out for the next process."""
    flushed = _capture_shutdown_flush(monkeypatch)
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.preparation_gates[_OWNER_TEXT] = gate = asyncio.Event()

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.queued() == [_OWNER_TEXT], "the owner message to be queued")
        if when == "peer_turn_ended":
            h.release.set()
            await peer
            await h.wait(lambda: _OWNER_TEXT in h.preparing, "the owner turn to be preparing")
        stop = asyncio.ensure_future(h.runner.stop(restart=True))
        await h.wait(lambda: h.runner._draining, "the restart drain to begin")
        h.release.set()
        gate.set()
        if when == "peer_turn_running":
            await peer
        await asyncio.wait_for(stop, 60)

    asyncio.run(exercise())
    # The restart drain interrupts the running peer turn itself; the owner's text never reaches it.
    assert _OWNER_TEXT not in repr((h.actor.steered, h.actor.redirected, h.actor.interrupted))
    assert h.actor.calls == ["hello", "cto: status?"]  # the owner turn did not run in this process
    assert _OWNER_TEXT in flushed  # and is written out for the next one


def test_a_restart_while_the_handed_on_message_awaits_on_processing_start_keeps_it(
        ingress, send, monkeypatch):
    """Review r2 P1-a: the peer turn ends while the owner's queued ack is still being sent, so the task
    sending it dispatches the reserved head as it ends; a restart while the new task awaits
    on_processing_start (Telegram may await a reaction request there) still leaves the message
    queued for the shutdown flush. The reserved head leaves the queue only at its own claim."""
    flushed = _capture_shutdown_flush(monkeypatch)
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    ack_gate, start_gate, starting = asyncio.Event(), asyncio.Event(), []
    plain_send = h.adapter.send

    async def held_send(chat_id, content, reply_to=None, metadata=None):
        if content.startswith(_QUEUED_ACK):
            await ack_gate.wait()
        return await plain_send(chat_id, content, reply_to=reply_to, metadata=metadata)

    async def on_processing_start(event):
        if event.text == _OWNER_TEXT and ack_gate.is_set():  # the handed-on dispatch
            starting.append(event)
            await start_gate.wait()
    monkeypatch.setattr(h.adapter, "send", held_send)
    monkeypatch.setattr(h.adapter, "on_processing_start", on_processing_start, raising=False)

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.queued() == [_OWNER_TEXT], "the owner message to be queued")
        h.release.set()
        assert (await peer)[0] == 200  # the peer turn ended while the ack is still being sent
        ack_gate.set()
        await h.wait(lambda: starting, "the handed-on message to await on_processing_start")
        await asyncio.wait_for(h.runner.stop(restart=True), 60)

    asyncio.run(exercise())
    assert h.actor.calls == ["hello", "cto: status?"]  # the owner turn did not run in this process
    assert _OWNER_TEXT in flushed  # and is written out for the next one


def test_a_handed_on_message_whose_reservation_is_dropped_before_it_claims_runs_once(
        ingress, send, monkeypatch):
    """The reserved head is dispatched without leaving the queue. If its reservation is released before
    it claims (a reaper or stale-slot eviction), it runs as an ordinary turn and still leaves the
    queue at its claim, so its own turn's queue drain does not run it a second time."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    start_gate, starting = asyncio.Event(), []

    async def on_processing_start(event):
        if event.text == _OWNER_TEXT and h.release.is_set():  # the handed-on dispatch
            starting.append(event)
            await start_gate.wait()
    monkeypatch.setattr(h.adapter, "on_processing_start", on_processing_start, raising=False)

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.queued() == [_OWNER_TEXT] and all(
            t.done() for t in h.adapter._session_tasks.values()), "the owner message to be queued")
        h.release.set()
        assert (await peer)[0] == 200
        await h.wait(lambda: starting, "the handed-on message to await on_processing_start")
        assert h.runner._release_running_agent_state(h.key)  # the reservation is dropped
        start_gate.set()
        await h.settle()

    asyncio.run(exercise())
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT]
    assert h.queued() == []


@pytest.mark.parametrize("path", ["ordinary", "queued_behind_peer"])
def test_a_restart_after_the_claim_before_the_turn_marker_matches_an_ordinary_owner_turn(
        ingress, send, monkeypatch, path):
    """Review r3 comparison: an owner turn that has claimed the session and pauses at topic recovery
    (``_hmwa_resolve_session``, before the turn lease and the durable active-turn marker that
    ``_mark_durable_active_turn`` writes after it) when stop(restart=True) runs. For an ordinary owner
    message on an idle session that window already keeps nothing: no queue entry to flush, no
    marker, no transcript row. The message handed on by a peer turn is measured against it."""
    flushed = _capture_shutdown_flush(monkeypatch)
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    resolve_gate, resolving = asyncio.Event(), []
    resolve = h.runner._hmwa_resolve_session

    async def paused_resolve(event, source):
        if event.text == _OWNER_TEXT:
            resolving.append(event)
            await resolve_gate.wait()
        return await resolve(event, source)
    monkeypatch.setattr(h.runner, "_hmwa_resolve_session", paused_resolve)

    async def exercise():
        await h.warm()
        if path == "queued_behind_peer":
            peer = await h.peer_turn_running()
            await h.adapter.handle_message(h.owner_event())
            await h.wait(lambda: h.queued() == [_OWNER_TEXT], "the owner message to be queued")
            h.release.set()
            assert (await peer)[0] == 200
        else:
            await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: resolving, "the claimed owner turn to pause at topic recovery")
        assert h.runner._is_session_running(h.key) and h.queued() == []
        await asyncio.wait_for(h.runner.stop(restart=True), 60)
        entry = h.runner.session_store.lookup_by_session_key_existing(h.key)
        rows = h.runner.session_store.load_transcript(ingress.entry.session_id)
        return (_OWNER_TEXT in flushed, getattr(entry, "active_turn_token", None) is not None,
                any(_OWNER_TEXT in str(row.get("content")) for row in rows), _OWNER_TEXT in h.actor.calls)

    # (flushed for the next process, durable turn marker, owner transcript row, turn ran)
    assert asyncio.run(exercise()) == (False, False, False, False)


@pytest.mark.parametrize("mode", ["steer", "interrupt"])
def test_witness_without_the_peer_turn_check_the_owner_is_answered_in_the_peer_thread(
        ingress, send, monkeypatch, mode):
    """Mutation witness: with the canonical check disabled the harness reproduces production — the
    owner's text is answered inside the peer turn's canonical answer and nothing reaches Telegram."""
    h = _Harness(ingress, send, monkeypatch, mode)
    monkeypatch.setattr(h.runner, "_canonical_peer_turn_running", lambda session_key: False)

    async def exercise():
        await h.warm()
        h.actor.hold = (h.entered, h.release)
        peer = asyncio.ensure_future(h.send(event_id="peer", text="cto: status?"))
        await h.wait(h.entered.is_set, "the peer turn to start")
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.actor.steered, "the owner text to reach the running turn")
        h.release.set()
        result = await peer
        await h.settle()
        return result

    status, body = asyncio.run(exercise())
    assert status == 200 and _OWNER_TEXT in body["text"]
    assert h.owner_replies() == []


# ── The durable inbound record (gateway/shutdown_flush.py) ───────────────────────────────────────
# An owner message's turn stores its inbound record before it takes the message from the queue and
# claims the session, and deletes it once the turn has ended. A restart that cut the turn short
# leaves the record, and the next process re-admits the message through its adapter as the owner's
# own message, on the bot and chat it came from, instead of an empty auto-resume.

# Every await between the claim (_claim_active_session_slot) and the durable active-turn marker, in the
# order an owner DM reaches them: _hmwa_resolve_session, then _hmwa_prepare_turn up to
# _mark_durable_active_turn. Not on this path: the strict and pinned lookups (internally routed events),
# the topic-binding heal (topic lanes) and the auto-reset notice.
_CLAIM_TO_MARKER_AWAITS = ("topic_recovery", "session_lookup", "topic_lane_check", "heartbeat_owner",
                           "session_open", "turn_lease", "marker_write")
_RESUME_NOTE = "[System note: The previous turn was interrupted"


def _records(ingress):
    """The inbound record files on disk, by name (the name carries the recovery id)."""
    spool = ingress.home / "pending_messages"
    return sorted(path.name for path in spool.glob("inbound-*.json")) if spool.exists() else []


def _record_texts(ingress):
    import json
    spool = ingress.home / "pending_messages"
    return sorted(json.loads(path.read_text())["event"]["text"] for path in spool.glob("inbound-*.json")) \
        if spool.exists() else []


def _pause_owner_turn_at(h, monkeypatch, at):
    """Hold the first call through ``at`` once armed. Returns (arm, paused, release)."""
    import gateway.run_heartbeat_acceptance as heartbeat

    armed, paused, thread_gate, loop_gate = [], [], threading.Event(), asyncio.Event()

    def blocking(fn):  # a call the turn awaits through asyncio.to_thread
        def held(*args, **kwargs):
            if armed and not paused and threading.current_thread() is not threading.main_thread():
                paused.append(at)
                thread_gate.wait(30)
            return fn(*args, **kwargs)
        return held

    def awaited(fn):
        async def held(*args, **kwargs):
            if armed and not paused:
                paused.append(at)
                await loop_gate.wait()
            return await fn(*args, **kwargs)
        return held

    runner, store = h.runner, h.runner.session_store
    owner, name, wrap = {
        "topic_recovery": (runner, "_recover_telegram_topic_thread_id", blocking),
        "session_lookup": (store, "get_or_create_session", blocking),
        "topic_lane_check": (runner, "_is_telegram_topic_lane", blocking),
        "heartbeat_owner": (heartbeat, "resolve_heartbeat_owner", awaited),
        "session_open": (runner, "_hmwa_open_session", awaited),
        "turn_lease": (runner, "_hmwa_acquire_turn_lease", awaited),
        "marker_write": (store, "mark_turn_active", blocking),
        "topic_lobby": (runner, "_is_telegram_topic_root_lobby", blocking),  # before the claim
    }[at]
    monkeypatch.setattr(owner, name, wrap(getattr(owner, name)))

    def release():
        thread_gate.set()
        loop_gate.set()
    return lambda: armed.append(True), paused, release


async def _owner_turn_started(h, path, arm, event=None):
    """Warm the session, then start the owner turn: on an idle session, or queued behind a peer turn
    and handed the session when it ends."""
    await h.warm()
    event = event or h.owner_event()
    if path == "queued_behind_peer":
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(event)
        await h.wait(lambda: h.queued() == [event.text], "the owner message to be queued")
        arm()
        h.release.set()
        assert (await peer)[0] == 200
    else:
        arm()
        await h.adapter.handle_message(event)


async def _settle_all(h):
    for _ in range(1500):
        tasks = [t for t in h.adapter._session_tasks.values() if not t.done()]
        if not tasks and not h.runner._running_agents:
            return
        await asyncio.sleep(0.01)
    pytest.fail("the next process never went idle")


def _next_process(ingress, send, monkeypatch, *, crashed=False, prepare=None, during=None):
    """The next process over the same home: crash recovery when the previous one died (``crashed``),
    then the startup wiring, which re-admits inbound records and runs auto-resume behind the
    startup-restore gate as start() does (its post-connect services are not under test), then the
    shutdown-flush replay gateway/run.py runs after start() returns. ``prepare(nxt)`` runs first;
    ``during(nxt)``, when given, runs instead of waiting for the process to go idle."""
    nxt = _Harness(ingress, send, monkeypatch, "queue")
    # The real admission: it tells internal events apart and honours the startup-restore gate.
    monkeypatch.delattr(nxt.runner, "_hm_admit_event")

    async def no_services(connected_count):
        return None
    monkeypatch.setattr(nxt.runner, "_start_post_connect_services", no_services)
    nxt.runner._startup_restore_in_progress = True  # set by start() before any adapter connects
    nxt.runner._startup_restore_queue, nxt.runner._startup_restore_tasks = [], []
    nxt.runner._running = True  # set by start() before its wiring: boot redelivery sends only while running
    if prepare is not None:
        prepare(nxt)

    async def start():
        # As _start_recover_previous_run: answered records go to the ledger before marker recovery.
        await nxt.runner._ledger_answered_inbound_records()
        if crashed:
            await nxt.runner._recover_unclean_sessions()
        if during is not None:
            wiring = asyncio.ensure_future(nxt.runner._start_finish_wiring(0))
            await during(nxt)
            await asyncio.wait_for(wiring, 60)
            return
        await asyncio.wait_for(nxt.runner._start_finish_wiring(0), 60)
        await _settle_all(nxt)
    asyncio.run(start())
    if during is None:
        from gateway.shutdown_flush import recover_pending_to_db
        recover_pending_to_db(session_resolver=nxt.runner.session_store.resolve_session_id_for_key)
    return nxt


def _answered(ingress, nxt):
    """What the next process did: the user messages its turns ran on, the texts its model calls were
    handed (a resume note over a message does not answer it), the chats its answers went to, and the
    inbound records it left."""
    turns = [text for actor in nxt.built for text in actor.calls]
    inputs = [text for actor in nxt.built for text in actor.model_inputs]
    chats = [chat for chat, text, _meta in nxt.adapter.sent_calls if _TERMINAL in text]
    return turns, inputs, chats, _records(ingress)


def _crash_now(h, monkeypatch):
    """From here on this process writes nothing durable, as a SIGKILL leaves it: the loop that ends
    next only unwinds tasks whose writes no longer land (this runner's store and records only)."""
    store = h.runner.session_store
    for name in ("clear_turn_active", "mark_resume_pending", "clear_resume_pending", "mark_turn_active"):
        monkeypatch.setattr(store, name, lambda *args, **kwargs: False)
    for name in ("_settle_inbound_record", "_settle_inbound_records", "_discard_inbound_records"):
        monkeypatch.setattr(h.runner, name, lambda *args, **kwargs: None, raising=False)


def _persisting_agents(monkeypatch, fragment=None):
    """Agents that write each turn's user row (with its platform message id) and answer, as the real
    agent's turn-start and final flushes do. With ``fragment``, an owner turn writes only that partial
    reply (finish_reason "length"), as a turn cut off mid-answer leaves it."""
    from types import SimpleNamespace

    import run_agent
    from agent.session_persistence import _db_flush_row

    build = run_agent.AIAgent

    def persisting(**kwargs):
        actor = build(**kwargs)
        run = actor.run_conversation

        def writing(text, *, conversation_history, task_id, **kw):
            result = run(text, conversation_history=conversation_history, task_id=task_id, **kw)
            flusher = SimpleNamespace(_persist_user_message_override=kw.get("persist_user_message"),
                                      _persist_user_message_timestamp=None)
            user = {"role": "user", "content": text, "platform_message_id": kw.get("persist_user_platform_id")}
            metadata = kw.get("persist_user_display_metadata") or {}
            recovery_id = metadata.get("inbound_recovery_id")
            if recovery_id:  # the agent writes the turn's display metadata on its user row
                user["display_metadata"] = {key: metadata[key] for key in (
                    "inbound_recovery_id", "inbound_absorbed_ids", "inbound_run") if key in metadata}
            answer = {"role": "assistant", "content": result["final_response"], "finish_reason": "stop"}
            if fragment is not None and recovery_id:  # an owner turn cut off with only a partial reply
                answer = {"role": "assistant", "content": fragment, "finish_reason": "length"}
            actor._session_db.append_messages_batch(task_id, [_db_flush_row(flusher, user, True), answer])
            return result
        actor.run_conversation = writing
        return actor
    monkeypatch.setattr(run_agent, "AIAgent", persisting)


def _owner_rows(ingress):
    import sqlite3
    with sqlite3.connect(ingress.home / "state.db") as conn:
        return conn.execute("SELECT COUNT(*) FROM messages WHERE role='user' AND content=?",
                            (_OWNER_TEXT,)).fetchone()[0]


@pytest.mark.parametrize("path", ["ordinary", "queued_behind_peer"])
def test_an_owner_message_is_durably_stored_before_its_turn_claims_the_session(
        ingress, send, monkeypatch, path):
    """The record is on disk at the claim, and gone once the turn has ended."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    seen, claim = [], h.runner._claim_active_session_slot

    def claiming(session_key, source):
        seen.append(_record_texts(ingress))
        return claim(session_key, source)
    monkeypatch.setattr(h.runner, "_claim_active_session_slot", claiming)

    async def exercise():
        await _owner_turn_started(h, path, lambda: None)
        await h.settle()
    asyncio.run(exercise())
    assert seen == [[_OWNER_TEXT]]
    assert (h.actor.calls[-1], _records(ingress)) == (_OWNER_TEXT, [])


def test_a_record_that_cannot_be_stored_refuses_the_turn_before_its_claim(ingress, send, monkeypatch):
    """A failed store claims nothing and runs nothing; the owner is told to send it again."""
    from gateway import shutdown_flush

    h = _Harness(ingress, send, monkeypatch, "queue")
    claims, claim = [], h.runner._claim_active_session_slot
    monkeypatch.setattr(h.runner, "_claim_active_session_slot",
                        lambda key, source: claims.append(key) or claim(key, source))

    def failing(*args, **kwargs):
        raise OSError("No space left on device")
    monkeypatch.setattr(shutdown_flush, "store_inbound_record", failing, raising=False)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    assert (claims, h.handled, h.actor.calls) == ([], [], ["hello"])  # not even accepted
    [(chat, notice)] = h.owner_replies()
    assert chat == "chat" and "not processed" in notice and "send it again" in notice


@pytest.mark.parametrize("at", _CLAIM_TO_MARKER_AWAITS)
@pytest.mark.parametrize("path", ["ordinary", "queued_behind_peer"])
def test_a_restart_between_the_claim_and_the_turn_marker_keeps_the_owner_message_for_the_next_process(
        ingress, send, monkeypatch, path, at):
    """stop(restart=True) while the claimed owner turn waits at any await before its durable marker:
    the record stays, and the next process runs the message as the owner sent it, once, answered on
    its chat — not a resume note over it (review r2 P1-c)."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    arm, paused, release = _pause_owner_turn_at(h, monkeypatch, at)

    async def cut_off():
        await _owner_turn_started(h, path, arm)
        await h.wait(lambda: paused, f"the claimed owner turn to pause at {at}")
        assert h.runner._is_session_running(h.key) and h.queued() == []
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            release()

    asyncio.run(cut_off())
    assert _OWNER_TEXT not in h.actor.calls and _record_texts(ingress) == [_OWNER_TEXT]
    turns, inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])
    assert not any(_RESUME_NOTE in text for text in inputs)


@pytest.mark.parametrize("stop", ["restart", "crash"])
@pytest.mark.parametrize("at", ["marker_written", "agent_running"])
def test_a_stop_between_the_turn_marker_and_its_transcript_row_keeps_the_owner_message(
        ingress, send, monkeypatch, at, stop):
    """Pins the marker → transcript-row window: the marker is written, the agent has not written the
    user row. A restart or a crash there leaves the record; the next process answers the message once
    as the owner's own, and the marker's resume (an empty turn told to skip unfinished work) never
    runs over it."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    gate, owned, agent_entered, agent_release = asyncio.Event(), [], threading.Event(), threading.Event()
    mark = h.runner._mark_durable_active_turn

    async def marked(event, session_key):
        written = await mark(event, session_key)
        if event.text == _OWNER_TEXT:
            owned.append(written)
            if at == "marker_written":
                await gate.wait()
        return written
    monkeypatch.setattr(h.runner, "_mark_durable_active_turn", marked)

    async def cut_off():
        await h.warm()
        if at == "agent_running":
            h.actor.hold = (agent_entered, agent_release)
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: owned and (at == "marker_written" or agent_entered.is_set()),
                     "the owner turn to own its marker")
        entry = h.runner.session_store.lookup_by_session_key_existing(h.key)
        assert entry.active_turn_token is not None and _owner_rows(ingress) == 0
        if stop == "crash":
            _crash_now(h, monkeypatch)
            return
        await asyncio.wait_for(h.runner.stop(restart=True), 60)

    try:
        asyncio.run(cut_off())
        ran_before = list(h.actor.calls)
    finally:
        agent_release.set()
    assert owned == [True] and _OWNER_TEXT not in ran_before and _record_texts(ingress) == [_OWNER_TEXT]
    nxt = _next_process(ingress, send, monkeypatch, crashed=stop == "crash")
    turns, inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])
    assert not any(_RESUME_NOTE in text for text in inputs)


@pytest.mark.parametrize("path", ["ordinary", "queued_behind_peer"])
def test_a_turn_whose_marker_write_failed_is_not_answered_twice_once_its_agent_ran(
        ingress, send, monkeypatch, path):
    """Review r1 P1-a: the marker write fails transiently and the turn runs anyway; its agent writes the
    user row and the answer, and the restart cuts the turn off while it stops typing. The record is
    still on disk, but the transcript shows the message answered, so the next process consumes it and
    runs no second turn."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    store, failed, stopping, gate = h.runner.session_store, [], [], asyncio.Event()
    mark, stop_typing, armed = store.mark_turn_active, h.runner._hmwa_stop_typing_for_turn, []

    def transiently_failing_mark(session_key):
        if armed and not failed:
            failed.append(session_key)
            raise OSError("state.db is busy")
        return mark(session_key)
    monkeypatch.setattr(store, "mark_turn_active", transiently_failing_mark)

    async def held_stop_typing(event, source):
        if event.text == _OWNER_TEXT:
            stopping.append(event)
            await gate.wait()
        return await stop_typing(event, source)
    monkeypatch.setattr(h.runner, "_hmwa_stop_typing_for_turn", held_stop_typing)
    _persisting_agents(monkeypatch)

    async def cut_off():
        await _owner_turn_started(h, path, lambda: armed.append(True))
        await h.wait(lambda: stopping, "the owner turn to stop typing after its agent ran")
        await asyncio.wait_for(h.runner.stop(restart=True), 60)

    asyncio.run(cut_off())
    assert failed and any(_OWNER_TEXT in actor.calls for actor in h.built)
    assert [text for _chat, text in h.owner_replies() if _TERMINAL in text] == []  # cut off before delivery
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    # Review r1 P1-b: the answer is owed through the delivery ledger before the record goes, and the
    # boot sweep delivers it once; no second model turn.
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)


@pytest.mark.parametrize("path", ["ordinary", "queued_behind_peer"])
def test_an_older_owed_reply_does_not_keep_a_cut_off_owner_message_from_running(
        ingress, send, monkeypatch, path):
    """Review r1 P1-b: the session already owes an older reply in the delivery ledger when a restart
    cuts the owner turn off before its marker. The boot redelivery settles that reply and clears the
    session's resume flag; the record does not depend on that flag."""
    from gateway import delivery_ledger

    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    arm, paused, release = _pause_owner_turn_at(h, monkeypatch, "topic_recovery")
    delivery_ledger.record_crash_left_reply(
        obligation_id="older-reply", session_key=h.key, platform="telegram", chat_id="chat",
        thread_id=None, content="an older reply", since=0)

    async def cut_off():
        await _owner_turn_started(h, path, arm)
        await h.wait(lambda: paused, "the claimed owner turn to pause")
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            release()

    asyncio.run(cut_off())
    claimed, sweep = [], delivery_ledger.sweep_recoverable

    def recorded_sweep(*args, **kwargs):
        rows = sweep(*args, **kwargs)
        claimed.extend(rows)
        return rows
    monkeypatch.setattr(delivery_ledger, "sweep_recoverable", recorded_sweep)
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert [row["obligation_id"] for row in claimed] == ["older-reply"]  # settled at boot
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])


def test_a_first_message_cut_off_before_its_session_exists_is_answered_by_the_next_process(
        ingress, send, monkeypatch):
    """Review r1 P1-c: the first message of a chat with no session yet, cut off before its turn opened
    the session. The next process re-admits it, which opens the session, and answers it once there."""
    from gateway.session import SessionSource

    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    arm, paused, release = _pause_owner_turn_at(h, monkeypatch, "topic_recovery")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat2", chat_type="dm", user_id="user")
    key = h.runner._session_key_for_source(source)
    assert h.runner.session_store.lookup_by_session_key_existing(key) is None

    async def cut_off():
        await h.warm()
        arm()
        await h.adapter.handle_message(MessageEvent(
            text=_OWNER_TEXT, message_type=MessageType.TEXT, source=source, message_id="first"))
        await h.wait(lambda: paused, "the claimed first message to pause")
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            release()

    asyncio.run(cut_off())
    nxt = _next_process(ingress, send, monkeypatch)
    assert nxt.runner.session_store.lookup_by_session_key_existing(key) is not None
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat2"], [])


def test_a_restart_while_the_turn_resolves_its_provider_keeps_the_message(ingress, send, monkeypatch):
    """Review r2 P1-b: the marker write fails transiently, so only the record holds the message while
    the turn resolves its provider runtime on its executor thread, and stop(restart=True) runs then.
    The next process runs the message as the owner sent it."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    store, armed, failed, resolving, unblock = h.runner.session_store, [], [], [], threading.Event()
    mark, resolve = store.mark_turn_active, h.runner._resolve_session_agent_runtime

    def transiently_failing_mark(session_key):
        if armed and not failed:
            failed.append(session_key)
            raise OSError("state.db is busy")
        return mark(session_key)

    def held_resolve(**kwargs):
        if armed and not resolving:
            resolving.append(threading.current_thread() is not threading.main_thread())
            unblock.wait(30)
        return resolve(**kwargs)
    monkeypatch.setattr(store, "mark_turn_active", transiently_failing_mark)
    monkeypatch.setattr(h.runner, "_resolve_session_agent_runtime", held_resolve)

    async def cut_off():
        await h.warm()
        armed.append(True)
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: resolving, "the owner turn to resolve its provider runtime")
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            unblock.set()

    asyncio.run(cut_off())
    assert failed and resolving == [True]
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])


def test_an_internal_event_cut_off_by_a_restart_is_never_run_as_the_owner(ingress, send, monkeypatch):
    """Review r3 P1-a: an internal event's turn gets no inbound record, so nothing re-admits it."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    arm, paused, release = _pause_owner_turn_at(h, monkeypatch, "topic_recovery")
    notice = MessageEvent(text="plugin: build finished", message_type=MessageType.TEXT, source=ingress.source,
                          internal=True, allow_gateway_control=False)

    async def internal_admission(event):
        return event, event.source, bool(event.internal)
    monkeypatch.setattr(h.runner, "_hm_admit_event", internal_admission)

    async def cut_off():
        await h.warm()
        arm()
        await h.adapter.handle_message(notice)
        await h.wait(lambda: paused, "the internal turn to pause")
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            release()

    asyncio.run(cut_off())
    assert _records(ingress) == []
    turns, _inputs, _chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, left) == ([], [])


def test_an_internal_event_queued_behind_a_kept_owner_message_is_not_promoted_to_owner_input(
        ingress, send, monkeypatch):
    """Review r3 P1-a: a plugin notification queued behind an owner turn cut off before its marker.
    The owner message is re-admitted; the notification keeps its flags in the shutdown flush, which
    replays it as a row, and it never runs as the owner's message."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    arm, paused, release = _pause_owner_turn_at(h, monkeypatch, "topic_recovery")
    notice = MessageEvent(text="plugin: build finished", message_type=MessageType.TEXT, source=ingress.source,
                          internal=True, allow_gateway_control=False)

    async def cut_off():
        await _owner_turn_started(h, "ordinary", arm)
        await h.wait(lambda: paused, "the claimed owner turn to pause")
        await h.adapter.handle_message(notice)
        await h.wait(lambda: h.queued() == [notice.text], "the notification to queue behind it")
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            release()

    asyncio.run(cut_off())
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])


def test_the_inbound_record_keeps_the_full_event_with_every_flag(ingress):
    """Source, transport, account, chat, message id, internal and security flags all round-trip."""
    from datetime import datetime

    from gateway import shutdown_flush
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="-100", chat_type="group", user_id="user",
                           user_name="Owner", thread_id="7", scope_id="workspace", profile="team_b",
                           message_id="41", is_bot=True, role_authorized=True)
    source.delivered_via_upstream_relay = True
    event = MessageEvent(
        text="look", message_type=MessageType.PHOTO, source=source, message_id="41", ledger_message_id="42",
        reply_anchor_override="40", platform_update_id=9001, media_urls=["/tmp/a.jpg"],
        media_types=["image/jpeg"], media_text_inlined=[None], reply_to_message_id="39",
        reply_to_text="earlier", reply_to_author_id="bot", reply_to_author_name="Bot",
        reply_to_is_own_message=True, prompt_response={"prompt_id": "p", "option_id": "o"},
        auto_skill=["triage"], channel_prompt="be brief", channel_context="ctx", internal=False,
        metadata={"gateway_session_key": "k", "whatsapp_from_owner": True}, allow_gateway_control=False,
        user_id="user", user_name="Owner", timestamp=datetime(2026, 11, 14, 1, 2, 3))
    event._gateway_pending_stt_text = "transcribed"
    shutdown_flush.store_inbound_record(event, session_key="agent:team_b:telegram:group:-100:7",
                                        transport_profile="team_b", recovery_id="r1", order=(1, 2))
    [record] = shutdown_flush.load_inbound_records()
    assert (record["recovery_id"], record["session_key"], record["transport_profile"]) == (
        "r1", "agent:team_b:telegram:group:-100:7", "team_b")
    back = shutdown_flush.inbound_event_from_record(record)
    for name in ("text", "message_type", "message_id", "ledger_message_id", "reply_anchor_override",
                 "platform_update_id", "media_urls", "media_types", "media_text_inlined", "reply_to_message_id",
                 "reply_to_text", "reply_to_author_id", "reply_to_author_name", "reply_to_is_own_message",
                 "prompt_response", "auto_skill", "channel_prompt", "channel_context", "internal", "metadata",
                 "allow_gateway_control", "user_id", "user_name", "timestamp"):
        assert getattr(back, name) == getattr(event, name), name
    assert back.source.to_dict() == source.to_dict()
    assert (back.source.is_bot, back.source.role_authorized, back.source.delivered_via_upstream_relay) == (
        True, True, True)
    assert back._gateway_pending_stt_text == "transcribed"
    internal = MessageEvent(text="wake", source=source, internal=True)
    shutdown_flush.store_inbound_record(internal, session_key="k", transport_profile=None,
                                        recovery_id="r2", order=(1, 3))
    assert [shutdown_flush.inbound_event_from_record(r).internal for r in shutdown_flush.load_inbound_records()] \
        == [False, True]


def test_the_recovery_backlog_has_no_queue_cap(ingress, send, monkeypatch):
    """Review r3 P1-b: more records than the live FIFO holds for one session. Each is re-admitted in
    order and answered; none is dropped by the busy-queue cap."""
    from gateway import shutdown_flush
    from gateway.run import GatewayRunner

    count = GatewayRunner._BUSY_QUEUE_MAX_PENDING + 2
    texts = [f"owner: item {i}" for i in range(count)]
    for i, text in enumerate(texts):
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=ingress.source,
                             message_id=f"b{i}")
        shutdown_flush.store_inbound_record(event, session_key=ingress.entry.session_key,
                                            transport_profile=None, recovery_id=f"backlog{i:03d}", order=(1, i))
    nxt = _next_process(ingress, send, monkeypatch)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, len(chats), left) == (texts, count, [])


@pytest.mark.parametrize("second_stop", ["before_claim", "after_claim"])
def test_back_to_back_restarts_keep_the_message_and_its_recovery_id(ingress, send, monkeypatch, second_stop):
    """Review r3 P2-a: the next process is restarted again while the re-admitted message awaits
    on_processing_start (before its claim) or topic recovery (after it). The record keeps its file and
    recovery id, and the third process answers the message once."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    arm, paused, release = _pause_owner_turn_at(h, monkeypatch, "topic_recovery")

    async def cut_off():
        await _owner_turn_started(h, "ordinary", arm)
        await h.wait(lambda: paused, "the claimed owner turn to pause")
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            release()

    asyncio.run(cut_off())
    first = _records(ingress)
    holds = {}

    def prepare(nxt):
        nxt.runner._restart_drain_timeout = 0.5
        if second_stop == "after_claim":
            holds["arm"], holds["paused"], holds["release"] = _pause_owner_turn_at(nxt, monkeypatch, "topic_recovery")
            holds["arm"]()
            return
        holds["paused"], holds["gate"] = [], asyncio.Event()

        async def on_processing_start(event):
            if event.text == _OWNER_TEXT:
                holds["paused"].append(event)
                await holds["gate"].wait()
        monkeypatch.setattr(nxt.adapter, "on_processing_start", on_processing_start, raising=False)

    async def restart_again(nxt):
        await nxt.wait(lambda: holds["paused"], "the re-admitted message to pause")
        try:
            await asyncio.wait_for(nxt.runner.stop(restart=True), 60)
        finally:
            holds.get("release", lambda: None)()

    second = _next_process(ingress, send, monkeypatch, prepare=prepare, during=restart_again)
    assert [text for actor in second.built for text in actor.calls] == []
    assert len(first) == 1 and _records(ingress) == first  # same file, same recovery id
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])


@pytest.mark.parametrize("held_at", ["before_claim", "agent_running"])
def test_a_record_is_readmitted_once_however_often_the_resume_scheduler_runs(
        ingress, send, monkeypatch, held_at):
    """The reconnect watcher re-runs the resume scheduler; a record this process already re-admitted
    is not dispatched a second time, whether its message still awaits on_processing_start (the session
    not yet running) or its turn runs."""
    from gateway import shutdown_flush

    event = MessageEvent(text=_OWNER_TEXT, message_type=MessageType.TEXT, source=ingress.source, message_id="m1")
    shutdown_flush.store_inbound_record(event, session_key=ingress.entry.session_key, transport_profile=None,
                                        recovery_id="once", order=(1, 1))
    entered, release, gate = threading.Event(), threading.Event(), asyncio.Event()

    def prepare(nxt):
        if held_at == "before_claim":
            async def on_processing_start(event):
                if event.text == _OWNER_TEXT and not gate.is_set():
                    entered.set()
                    await gate.wait()
            monkeypatch.setattr(nxt.adapter, "on_processing_start", on_processing_start, raising=False)
            return
        import run_agent
        build = run_agent.AIAgent

        def held(**kwargs):
            actor = build(**kwargs)
            actor.hold = (entered, release)
            return actor
        monkeypatch.setattr(run_agent, "AIAgent", held)

    async def rerun_scheduler(nxt):
        await nxt.wait(entered.is_set, "the re-admitted message to be held")
        nxt.runner._schedule_resume_pending_sessions()
        nxt.runner._schedule_resume_pending_sessions(platform=Platform.TELEGRAM)
        await asyncio.sleep(0.2)
        gate.set()
        release.set()
        await _settle_all(nxt)

    try:
        nxt = _next_process(ingress, send, monkeypatch, prepare=prepare, during=rerun_scheduler)
    finally:
        release.set()
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])


def test_a_crash_after_the_turn_completed_before_its_record_was_consumed_runs_no_second_turn(
        ingress, send, monkeypatch):
    """The turn-completion boundary: the turn has written its answer and returned, and the process dies
    before the record is deleted and before the reply is ledgered (the marker is still held). The
    answer is owed once through the ledger, the marker's own recovery finds it owed, and no second
    turn runs."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)
    monkeypatch.setattr(h.runner, "_settle_inbound_record", lambda *args, **kwargs: None, raising=False)
    returned, never = asyncio.Event(), asyncio.Event()

    async def before_delivery(*args, **kwargs):
        returned.set()
        await never.wait()
    monkeypatch.setattr(h.adapter, "_extract_response_content", before_delivery)

    async def run_once():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(returned.is_set, "the owner turn to return its reply to the adapter")
        _crash_now(h, monkeypatch)
    asyncio.run(run_once())
    entry = h.runner.session_store.lookup_by_session_key_existing(h.key)
    assert h.owner_replies() == [] and len(_records(ingress)) == 1 and entry.active_turn_token is not None
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch, crashed=True))
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)


def test_a_restart_during_the_final_send_is_settled_by_redelivery_alone(ingress, send, monkeypatch):
    """The reply-sending boundary: the turn has ended and its final reply is in the delivery ledger when
    stop(restart=True) runs mid-send. The record is still on disk (it goes only once that reply reads
    back delivered). The next process redelivers that reply, runs no second turn, and then settles the
    record."""
    from gateway import delivery_ledger

    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    sending, gate, plain_send = [], asyncio.Event(), h.adapter.send

    async def held_send(chat_id, content, reply_to=None, metadata=None):
        if _TERMINAL in content and not sending:
            sending.append(content)
            await gate.wait()
        return await plain_send(chat_id, content, reply_to=reply_to, metadata=metadata)
    monkeypatch.setattr(h.adapter, "send", held_send)
    _persisting_agents(monkeypatch)

    async def cut_off():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: sending, "the owner turn's final send")
        assert len(_records(ingress)) == 1
        await asyncio.wait_for(h.runner.stop(restart=True), 60)

    asyncio.run(cut_off())
    with delivery_ledger._connect() as conn:  # the process that recorded the reply is gone
        recorded = conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
            "WHERE session_key=? AND content LIKE ? AND state != 'delivered'",
            (h.key, f"%{_TERMINAL}%")).rowcount
        conn.commit()
    assert recorded == 1
    claimed, sweep = [], delivery_ledger.sweep_recoverable

    def recorded_sweep(*args, **kwargs):
        rows = sweep(*args, **kwargs)
        claimed.extend(rows)
        return rows
    monkeypatch.setattr(delivery_ledger, "sweep_recoverable", recorded_sweep)
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert [_TERMINAL in row["content"] for row in claimed] == [True]  # the ledger owns the reply
    assert (turns, chats, left) == ([], ["chat"], [])


def test_a_multiplexed_message_is_readmitted_on_the_bot_that_received_it(mux):
    """Review r3 P2-b: a message received by TEAM_B's bot and routed into the default runtime. Its
    record names TEAM_B as the transport, and the re-admitted event is handed to TEAM_B, on the same
    chat and session key, never to the runtime profile's PRIMARY bot."""
    from gateway import shutdown_flush
    from tests.gateway.test_multiplex_transport_matrix import _live

    source, identity = _live(mux, mux.team_b, "999", transport_profile="team_b")
    assert (identity.transport_profile, identity.runtime_profile) == ("team_b", "default")
    key = mux.team_b._source_session_key(source)
    event = MessageEvent(text=_OWNER_TEXT, message_type=MessageType.TEXT, source=source, message_id="m9")
    assert asyncio.run(mux.runner._store_inbound_record(event, key))
    [record] = shutdown_flush.load_inbound_records()
    assert record["transport_profile"] == "team_b"
    readmitted, adapter = mux.runner._inbound_record_target(record)
    assert adapter is mux.team_b
    assert (readmitted.source.chat_id, readmitted.source.user_id, readmitted.message_id) == ("999", "999", "m9")
    assert mux.team_b._source_session_key(readmitted.source) == key


def test_the_shutdown_flush_and_its_replay_leave_a_stored_message_to_its_record(ingress, caplog):
    """A queued event whose record is on disk is not flushed a second time, and the flush replay leaves
    a record alone (no row, no file deleted, no warning): the record carries it to the next process."""
    import logging

    from gateway import shutdown_flush

    key = ingress.entry.session_key
    stored = MessageEvent(text=_OWNER_TEXT, message_type=MessageType.TEXT, source=ingress.source, message_id="m1")
    shutdown_flush.store_inbound_record(stored, session_key=key, transport_profile=None, recovery_id="kept",
                                        order=(1, 1))
    stored._inbound_record_id = "kept"
    plain = MessageEvent(text="owner: queued", message_type=MessageType.TEXT, source=ingress.source, message_id="m2")
    assert shutdown_flush.flush_pending_to_file({key: stored}) == 0
    assert shutdown_flush.flush_overflow_to_file({key: [stored, plain]}) == 1
    with caplog.at_level(logging.WARNING, logger="gateway.shutdown_flush"):
        assert shutdown_flush.recover_pending_to_db(
            session_resolver=ingress.runner.session_store.resolve_session_id_for_key) == 1
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    rows = [row.get("content") for row in ingress.runner.session_store.load_transcript(ingress.entry.session_id)]
    assert ("owner: queued" in rows, _OWNER_TEXT in rows, _records(ingress)) == (True, False, ["inbound-kept.json"])


def test_a_readmitted_message_the_inbound_path_declines_is_consumed(ingress, send, monkeypatch):
    """The owner was removed from the allowlist before the next start: the re-admitted message is refused
    by the ordinary admission, runs nothing, and its record is consumed instead of returning at every
    start."""
    from gateway import shutdown_flush

    event = MessageEvent(text=_OWNER_TEXT, message_type=MessageType.TEXT, source=ingress.source, message_id="m1")
    shutdown_flush.store_inbound_record(event, session_key=ingress.entry.session_key, transport_profile=None,
                                        recovery_id="declined", order=(1, 1))

    def prepare(nxt):
        monkeypatch.setattr(nxt.runner, "_is_user_authorized_for_source", lambda source: False)
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch, prepare=prepare))
    assert (turns, chats, left) == ([], [], [])


def test_a_turn_the_restart_drain_interrupts_keeps_its_message(ingress, send, monkeypatch):
    """The drain times out and interrupts the running owner turn, which returns (interrupted) rather
    than being cancelled. Its session is still resume_pending, so the record stays, and the next
    process answers the message once instead of an empty resume."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    entered, release, interrupted = threading.Event(), threading.Event(), []

    async def cut_off():
        await h.warm()
        actor = h.actor
        run = actor.run_conversation

        def interruptible(text, *, conversation_history, task_id, **kwargs):
            if kwargs.get("persist_user_message", text) != _OWNER_TEXT:
                return run(text, conversation_history=conversation_history, task_id=task_id, **kwargs)
            entered.set()
            release.wait(30)
            return {"interrupted": True, "interrupt_message": interrupted[0] if interrupted else None,
                    "completed": False, "final_response": "", "messages": list(conversation_history),
                    "api_calls": 0}

        def interrupt(text=None):
            interrupted.append(text)
            release.set()
        monkeypatch.setattr(actor, "run_conversation", interruptible)
        monkeypatch.setattr(actor, "interrupt", interrupt)
        await h.adapter.handle_message(h.owner_event())
        await h.wait(entered.is_set, "the owner turn's agent to run")
        await asyncio.wait_for(h.runner.stop(restart=True), 60)

    try:
        asyncio.run(cut_off())
    finally:
        release.set()
    assert interrupted and _record_texts(ingress) == [_OWNER_TEXT]
    turns, inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])
    assert not any(_RESUME_NOTE in text for text in inputs)


@pytest.mark.parametrize("path", ["arrival", "drained_follow_up"])
def test_a_restart_after_the_adapter_accepted_a_message_before_its_claim_keeps_it(
        ingress, send, monkeypatch, path):
    """Review r1 P1-a, the exact reproduction: the adapter has accepted the message (an arrival on an
    idle session, or a follow-up its queue gave up when the turn ahead of it ended) and the turn waits
    at the topic-lobby check, before its claim, when stop(restart=True) runs. The record was stored
    before the acceptance or the dequeue, so the next process answers the message once."""
    follow_up = "owner: and another thing"
    text = _OWNER_TEXT if path == "arrival" else follow_up
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    arm, paused, release = _pause_owner_turn_at(h, monkeypatch, "topic_lobby")
    message = h.owner_event(text, message_id="m7")
    sending, send_gate, plain_send = [], asyncio.Event(), h.adapter.send

    async def held_send(chat_id, content, reply_to=None, metadata=None):
        if _TERMINAL in content and not sending:
            sending.append(content)
            await send_gate.wait()
        return await plain_send(chat_id, content, reply_to=reply_to, metadata=metadata)
    monkeypatch.setattr(h.adapter, "send", held_send)

    async def cut_off():
        await h.warm()
        if path == "drained_follow_up":
            # The follow-up arrives while the first turn's reply is being sent: the adapter queues it
            # and, once that send is done, takes it from the queue for a turn of its own.
            await h.adapter.handle_message(h.owner_event())
            await h.wait(lambda: sending, "the first owner turn's final send")
            await h.adapter.handle_message(message)
            await h.wait(lambda: h.adapter._pending_messages.get(h.key) is message, "the follow-up to be queued")
            # Queueing never deletes the record; the first message's goes once its reply is delivered.
            assert _record_texts(ingress) == sorted([_OWNER_TEXT, follow_up])
            arm()
            send_gate.set()
        else:
            send_gate.set()
            arm()
            await h.adapter.handle_message(message)
        await h.wait(lambda: paused, "the accepted message to wait at the topic-lobby check")
        assert message._gateway_accepted is True and h.queued() == [] and _record_texts(ingress) == [text]
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            release()

    asyncio.run(cut_off())
    assert text not in h.actor.calls and _record_texts(ingress) == [text]
    turns, inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([text], ["chat"], [])
    assert not any(_RESUME_NOTE in model_input for model_input in inputs)


def test_a_slow_durable_write_never_stalls_the_event_loop(ingress, send, monkeypatch):
    """Review r1 P1-c: the record write (fsync included) takes 2 s. It runs off the loop, so a heartbeat
    coroutine keeps ticking throughout, and the message is accepted and claimed only after it is on
    disk."""
    import time

    from gateway import shutdown_flush

    h = _Harness(ingress, send, monkeypatch, "queue")
    store, accepted_while_writing = shutdown_flush.store_inbound_record, []

    def slow_disk(event, **kwargs):
        accepted_while_writing.append(event._gateway_accepted)
        time.sleep(2.0)
        return store(event, **kwargs)
    monkeypatch.setattr(shutdown_flush, "store_inbound_record", slow_disk)
    seen, claim = [], h.runner._claim_active_session_slot
    monkeypatch.setattr(h.runner, "_claim_active_session_slot",
                        lambda key, source: seen.append(_record_texts(ingress)) or claim(key, source))

    async def exercise():
        await h.warm()
        gaps, stop = [], asyncio.Event()

        async def heartbeat():
            last = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(0.05)
                now = time.monotonic()
                gaps.append(now - last)
                last = now
        beat = asyncio.ensure_future(heartbeat())
        await asyncio.sleep(0.2)  # the heartbeat is ticking before the message arrives
        message = h.owner_event()
        await h.adapter.handle_message(message)
        await h.settle()
        stop.set()
        await beat
        return max(gaps), message._gateway_accepted

    longest_gap, accepted = asyncio.run(exercise())
    assert longest_gap < 0.5, f"the event loop stalled for {longest_gap:.2f}s"
    assert (accepted_while_writing, accepted, seen) == ([False], True, [[_OWNER_TEXT]])
    assert (h.actor.calls[-1], _records(ingress)) == (_OWNER_TEXT, [])


# ── Review r2 of 1600122b ────────────────────────────────────────────────────────────────────────

def _hold_turns(h, monkeypatch, holds):
    """Hold the agent run of each text in *holds* (text -> (entered, release) threading events)."""
    actor = h.actor
    run = actor.run_conversation

    def held(text, *, conversation_history, task_id, **kwargs):
        gate = holds.get(kwargs.get("persist_user_message", text))
        if gate is not None:
            gate[0].set()
            gate[1].wait(30)
        return run(text, conversation_history=conversation_history, task_id=task_id, **kwargs)
    monkeypatch.setattr(actor, "run_conversation", held)


def test_a_queued_follow_up_drained_inside_the_running_turn_keeps_its_record_until_its_turn_completes(
        ingress, send, monkeypatch):
    """Review r2 P1-1: an owner follow-up queued behind a running owner turn is drained inside that turn
    (_run_agent_drain_pending). Queueing and dequeueing keep its record; the first message's record
    goes once its answer reads back delivered. A restart while the follow-up runs leaves only the
    follow-up's record, and the next process answers it once."""
    follow_up = "owner: one more thing"
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    first, second = (threading.Event(), threading.Event()), (threading.Event(), threading.Event())

    async def cut_off():
        await h.warm()
        _hold_turns(h, monkeypatch, {_OWNER_TEXT: first, follow_up: second})
        await h.adapter.handle_message(h.owner_event())
        await h.wait(first[0].is_set, "the first owner turn to run")
        await h.adapter.handle_message(h.owner_event(follow_up, message_id="m2"))
        await h.wait(lambda: h.queued() == [follow_up], "the follow-up to be queued")
        assert _record_texts(ingress) == sorted([_OWNER_TEXT, follow_up])
        first[1].set()
        await h.wait(second[0].is_set, "the follow-up to run inside the turn")
        assert h.queued() == [] and _record_texts(ingress) == [follow_up]
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            second[1].set()

    asyncio.run(cut_off())
    assert [chat for chat, text in h.owner_replies() if _TERMINAL in text] == ["chat"]  # the first answer
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([follow_up], ["chat"], [])


def test_a_queued_follow_up_drained_inside_the_turn_settles_its_record_when_it_completes(
        ingress, send, monkeypatch):
    """The other end of P1-1: drained in-band and run to completion, the follow-up's record is deleted
    with its turn, and nothing is left for a later start."""
    follow_up = "owner: one more thing"
    h = _Harness(ingress, send, monkeypatch, "queue")
    first = (threading.Event(), threading.Event())

    async def exercise():
        await h.warm()
        _hold_turns(h, monkeypatch, {_OWNER_TEXT: first})
        await h.adapter.handle_message(h.owner_event())
        await h.wait(first[0].is_set, "the first owner turn to run")
        await h.adapter.handle_message(h.owner_event(follow_up, message_id="m2"))
        await h.wait(lambda: h.queued() == [follow_up], "the follow-up to be queued")
        first[1].set()
        await h.settle()

    asyncio.run(exercise())
    assert (h.actor.calls[-2:], _records(ingress)) == ([_OWNER_TEXT, follow_up], [])


def test_an_orphan_whose_record_cannot_be_written_is_not_claimed(ingress, send, monkeypatch):
    """Review r2 P1-2: the orphan rescue runs the oldest orphaned FIFO event as this turn. It is made
    durable before the claim like every other path; when its write fails nothing is claimed, the
    orphan stays queued, and the owner is told."""
    from gateway import shutdown_flush

    h = _Harness(ingress, send, monkeypatch, "queue")
    orphan = h.owner_event("owner: orphaned while busy", message_id="o1")
    store = shutdown_flush.store_inbound_record

    def failing_for_the_orphan(event, **kwargs):
        if event is orphan:
            raise OSError("No space left on device")
        return store(event, **kwargs)
    monkeypatch.setattr(shutdown_flush, "store_inbound_record", failing_for_the_orphan)
    claims, claim = [], h.runner._claim_active_session_slot
    monkeypatch.setattr(h.runner, "_claim_active_session_slot",
                        lambda key, source: claims.append(key) or claim(key, source))

    async def exercise():
        await h.warm()
        h.runner._session_state(h.key).conversation.queued_events.append(orphan)  # an orphaned overflow
        await h.adapter.handle_message(h.owner_event())
        await h.settle()

    asyncio.run(exercise())
    assert (claims, h.actor.calls) == ([], ["hello"])
    assert h.runner._overflow_queue(h.key) == [orphan]  # still queued
    [(chat, notice)] = h.owner_replies()
    assert chat == "chat" and "not processed" in notice


def _cut_after_the_agent_wrote(h, monkeypatch, *, message_id, fragment=None):
    """The marker write fails transiently, the agent writes the user row and its answer (or only a
    partial ``fragment``), and stop(restart=True) runs while the turn stops typing, before delivery."""
    store, failed, stopping, gate = h.runner.session_store, [], [], asyncio.Event()
    mark, stop_typing, armed = store.mark_turn_active, h.runner._hmwa_stop_typing_for_turn, []

    def transiently_failing_mark(session_key):
        if armed and not failed:
            failed.append(session_key)
            raise OSError("state.db is busy")
        return mark(session_key)
    monkeypatch.setattr(store, "mark_turn_active", transiently_failing_mark)

    async def held_stop_typing(event, source):
        if event.text == _OWNER_TEXT:
            stopping.append(event)
            await gate.wait()
        return await stop_typing(event, source)
    monkeypatch.setattr(h.runner, "_hmwa_stop_typing_for_turn", held_stop_typing)
    _persisting_agents(monkeypatch, fragment=fragment)

    async def cut_off():
        await h.warm()
        armed.append(True)
        await h.adapter.handle_message(h.owner_event(message_id=message_id))
        await h.wait(lambda: stopping, "the owner turn to stop typing after its agent ran")
        await asyncio.wait_for(h.runner.stop(restart=True), 60)

    asyncio.run(cut_off())
    assert failed and [text for _chat, text in h.owner_replies() if _TERMINAL in text] == []


def test_a_message_without_a_platform_id_is_not_answered_twice(ingress, send, monkeypatch):
    """Review r2 P1-3: the same cut-off as r1 P1-b for a message with no platform message id. The user
    row names the record's recovery id, so the next process finds it answered: the answer is
    delivered once and no second turn runs."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    _cut_after_the_agent_wrote(h, monkeypatch, message_id=None)
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)


def test_a_partial_answer_in_the_transcript_is_not_taken_as_the_answer(ingress, send, monkeypatch):
    """Review r2 P1-4: the cut-off turn's transcript holds only a truncated fragment (finish_reason
    "length"). That is not a completed answer: the next process runs the message again and delivers
    its real answer, never the fragment."""
    fragment = "partial: I was about to"
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    _cut_after_the_agent_wrote(h, monkeypatch, message_id="m1", fragment=fragment)
    nxt = _next_process(ingress, send, monkeypatch)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])
    assert not any(fragment in text for _chat, text, _meta in nxt.adapter.sent_calls)


def test_two_messages_answered_with_the_same_text_are_each_delivered(ingress, send, monkeypatch):
    """Review r2 P1-5: two stored messages of one session were answered with the same text before a
    restart, and neither answer was delivered. Each answer is owed under its own record: both go out."""
    from types import SimpleNamespace

    from agent.session_persistence import _db_flush_row
    from gateway import shutdown_flush

    db, sid, key = ingress.runner.session_store._db, ingress.entry.session_id, ingress.entry.session_key
    rows = []
    for index in (1, 2):
        text, recovery_id = f"owner: question {index}", f"same{index}"
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=ingress.source, message_id=f"m{index}")
        shutdown_flush.store_inbound_record(event, session_key=key, transport_profile=None,
                                            recovery_id=recovery_id, order=(1, index))
        flusher = SimpleNamespace(_persist_user_message_override=None, _persist_user_message_timestamp=None)
        rows += [_db_flush_row(flusher, {"role": "user", "content": text, "platform_message_id": f"m{index}",
                                         "display_metadata": {"inbound_recovery_id": recovery_id}}, True),
                 {"role": "assistant", "content": "OK", "finish_reason": "stop"}]
    db.append_messages_batch(sid, rows)
    nxt = _next_process(ingress, send, monkeypatch)
    delivered = [chat for chat, text, _meta in nxt.adapter.sent_calls if text.endswith("OK")]
    assert (delivered, [text for actor in nxt.built for text in actor.calls], _records(ingress)) == (
        ["chat", "chat"], [], [])


def test_a_message_folded_into_another_keeps_its_record_until_that_message_is_settled():
    """Review r2 P1-1, the other dequeue-free exits: a message merged into the queued head, or replaced
    by a newer one, has no turn of its own. Its record rides the event that absorbed it and is settled
    with that event's turn, never at the merge."""
    from gateway import shutdown_flush
    from gateway.platforms.base import merge_pending_message_event
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")

    def stored(text, recovery_id, message_type=MessageType.TEXT):
        event = MessageEvent(text=text, message_type=message_type, source=source)
        event._inbound_record_id = recovery_id
        return event

    pending = {}
    head, merged = stored("first", "r-head"), stored("second", "r-merged")
    merge_pending_message_event(pending, "k", head)
    merge_pending_message_event(pending, "k", merged, merge_text=True)
    assert pending["k"] is head and shutdown_flush.inbound_record_ids(head) == ["r-head", "r-merged"]
    assert merged._inbound_absorbed is True
    newer = stored("third", "r-newer")
    merge_pending_message_event(pending, "k", newer)  # no text merge: the head is replaced
    assert pending["k"] is newer and shutdown_flush.inbound_record_ids(newer) == ["r-newer", "r-head", "r-merged"]


def test_a_message_steered_into_a_running_turn_keeps_its_record_until_that_turn_ends(
        ingress, send, monkeypatch):
    """Review r2 P1-1 rule for a steered message: it has no turn of its own, so its record rides the
    running turn. A restart that cuts that turn short leaves both records, and the next process
    answers both messages; a turn that completes settles both."""
    steered = "owner: also check the logs"
    h = _Harness(ingress, send, monkeypatch, "steer")
    h.runner._restart_drain_timeout = 0.5
    first = (threading.Event(), threading.Event())

    async def cut_off():
        await h.warm()
        _hold_turns(h, monkeypatch, {_OWNER_TEXT: first})
        await h.adapter.handle_message(h.owner_event())
        await h.wait(first[0].is_set, "the owner turn to run")
        state = h.runner._peek_session_state(h.key)
        await h.wait(lambda: state.turn.agent is h.actor, "the owner turn's agent to be published")
        await h.adapter.handle_message(h.owner_event(steered, message_id="m2"))
        await h.wait(lambda: h.actor.steered, "the follow-up to be steered into the running turn")
        assert _record_texts(ingress) == sorted([_OWNER_TEXT, steered])
        try:
            await asyncio.wait_for(h.runner.stop(restart=True), 60)
        finally:
            first[1].set()

    asyncio.run(cut_off())
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (sorted(turns), chats, left) == (sorted([_OWNER_TEXT, steered]), ["chat", "chat"], [])


# ── Review r3 of 1524ad04 ─────────────────────────────────────────────────────────────────────────


def _failing_marker_write(h, monkeypatch):
    """The next turn's marker write fails (once ``armed`` is set); returns ``(armed, failed)``."""
    store, failed, armed = h.runner.session_store, [], []
    mark = store.mark_turn_active

    def transiently_failing_mark(session_key):
        if armed and not failed:
            failed.append(session_key)
            raise OSError("state.db is busy")
        return mark(session_key)
    monkeypatch.setattr(store, "mark_turn_active", transiently_failing_mark)
    return armed, failed


def test_a_turn_whose_marker_write_failed_and_whose_reply_never_reached_the_ledger_is_answered(
        ingress, send, monkeypatch):
    """Review r3 P1-1: the marker write fails, the agent writes its completed answer, the turn returns
    the reply to the adapter, and the process dies before that reply is in the delivery ledger. With no
    marker, the record is the only thing that can owe the answer, so it stays until the reply's
    obligation exists: the next process delivers the answer once and runs no second turn."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    armed, failed = _failing_marker_write(h, monkeypatch)
    _persisting_agents(monkeypatch)
    returned, never = asyncio.Event(), asyncio.Event()

    async def before_delivery(*args, **kwargs):
        returned.set()
        await never.wait()
    monkeypatch.setattr(h.adapter, "_extract_response_content", before_delivery)

    async def run_once():
        await h.warm()
        armed.append(True)
        await h.adapter.handle_message(h.owner_event())
        await h.wait(returned.is_set, "the owner turn to return its reply to the adapter")
        _crash_now(h, monkeypatch)
    asyncio.run(run_once())
    entry = h.runner.session_store.lookup_by_session_key_existing(h.key)
    assert failed and h.owner_replies() == [] and entry.active_turn_token is None
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch, crashed=True))
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)


@pytest.mark.parametrize("footer", ["", "⏱ test-model · 1s"], ids=["plain", "runtime_footer"])
def test_a_delivered_answer_whose_record_could_not_be_deleted_is_not_sent_again(
        ingress, send, monkeypatch, footer):
    """Review r3 P1-2: the turn completes and its answer is delivered, but deleting its inbound record
    fails (unlink raises). The next process finds the record answered and owes the answer under the
    record, the identity the delivered reply was ledgered under, so that obligation is the delivered
    one: nothing is sent again, no second turn runs, and the record goes. Review r4 P1-2: the same with
    a runtime footer the production footer hook adds at delivery, which the transcript's answer lacks.
    The obligation is the record's and the reply's destination, never the reply's text."""
    import pathlib

    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)
    monkeypatch.setattr(h.runner, "_hmwa_runtime_footer_line", lambda *_args: footer)
    unlink, refused = pathlib.Path.unlink, []

    def failing_unlink(path, *args, **kwargs):
        if path.name.startswith("inbound-"):
            refused.append(path.name)
            raise OSError("read-only file system")
        return unlink(path, *args, **kwargs)

    async def exercise():
        await h.warm()
        monkeypatch.setattr(pathlib.Path, "unlink", failing_unlink)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    assert refused and len(_records(ingress)) == 1
    [(chat, delivered)] = [(chat, text) for chat, text in h.owner_replies() if _TERMINAL in text]
    assert chat == "chat" and delivered.endswith(footer)
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([], [], [])


def test_a_message_merged_into_another_is_answered_by_that_answer_across_a_restart(ingress, send, monkeypatch):
    """Review r3 P1-3, the reviewer's reconstruction: a message merged into the queued head has no turn
    of its own, and the head's turn answers both. The process dies after that answer is written and
    before it is delivered, with both records on disk. The user row names both records, so the next
    process delivers the one answer once and runs neither message again."""
    from types import SimpleNamespace

    from agent.session_persistence import _db_flush_row
    from gateway import shutdown_flush
    from gateway.platforms.base import merge_pending_message_event
    from gateway.run_turn import _inbound_row_metadata

    db, sid, key = ingress.runner.session_store._db, ingress.entry.session_id, ingress.entry.session_key
    events = []
    for index, text in enumerate((_OWNER_TEXT, "owner: and the logs"), start=1):
        event = MessageEvent(text=text, message_type=MessageType.TEXT, source=ingress.source, message_id=f"m{index}")
        shutdown_flush.store_inbound_record(event, session_key=key, transport_profile=None,
                                            recovery_id=f"fold{index}", order=(1, index))
        event._inbound_record_id = f"fold{index}"
        events.append(event)
    pending = {}
    merge_pending_message_event(pending, key, events[0])
    merge_pending_message_event(pending, key, events[1], merge_text=True)
    head = pending[key]
    flusher = SimpleNamespace(_persist_user_message_override=None, _persist_user_message_timestamp=None)
    db.append_messages_batch(sid, [
        _db_flush_row(flusher, {"role": "user", "content": head.text, "platform_message_id": "m1",
                                "display_metadata": _inbound_row_metadata(head)}, True),
        {"role": "assistant", "content": f"{_TERMINAL} / both", "finish_reason": "stop"}])
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([], ["chat"], [])


def _finalized_turn(ingress, monkeypatch, *, failed):
    """The agent's own finalizer closes a turn whose reply had no row of its own (a fallback or
    recovery exit), and the rows it persists (the user row naming the record, and the closing reply)
    are what the next process reads. The record is on disk. Returns the reply text."""
    from types import SimpleNamespace

    from agent.session_persistence import _db_flush_row
    from agent.turn_finalizer import finalize_turn
    from gateway import shutdown_flush
    from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db, sid, key = ingress.runner.session_store._db, ingress.entry.session_id, ingress.entry.session_key
    event = MessageEvent(text=_OWNER_TEXT, message_type=MessageType.TEXT, source=ingress.source, message_id="m1")
    shutdown_flush.store_inbound_record(event, session_key=key, transport_profile=None,
                                        recovery_id="fallback", order=(1, 1))
    reply = f"{_TERMINAL} / " + ("the provider failed" if failed else "recovered from the earlier text")
    agent = FakeAgent()
    finalize_turn(
        agent, final_response=reply, api_call_count=1, interrupted=False, failed=failed,
        messages=[{"role": "user", "content": _OWNER_TEXT, "display_metadata": {"inbound_recovery_id": "fallback"}}],
        conversation_history=[], effective_task_id="task", turn_id="turn", user_message=_OWNER_TEXT,
        original_user_message=_OWNER_TEXT, _should_review_memory=False,
        _turn_exit_reason="fallback_prior_turn_content")
    user, closing = agent.persisted_messages
    assert closing["role"] == "assistant" and closing["content"] == reply and "finish_reason" not in closing
    flusher = SimpleNamespace(_persist_user_message_override=None, _persist_user_message_timestamp=None)
    db.append_messages_batch(sid, [_db_flush_row(flusher, user, True), _db_flush_row(flusher, closing, False)])
    return reply


def test_a_fallback_reply_the_finalizer_closed_a_completed_turn_with_is_answered_once(
        ingress, send, monkeypatch):
    """Review r3 P1-4: a completed turn whose reply came from a fallback exit is closed by the finalizer
    with a row that has no finish_reason. A crash leaves the record and the turn marker. The finalizer
    marked that row as the completed turn's answer, so the record recognises it: the answer is
    delivered once (the marker's recovery finds it already owed) and the message runs no second turn."""
    ingress.runner.session_store.mark_turn_active(ingress.entry.session_key)  # held from the turn's start
    reply = _finalized_turn(ingress, monkeypatch, failed=False)
    entry = ingress.runner.session_store.lookup_by_session_key_existing(ingress.entry.session_key)
    assert entry.active_turn_token is not None
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([], ["chat"], [])
    assert [text for _chat, text, _meta in nxt.adapter.sent_calls if reply in text] != []


def test_a_failed_turn_the_finalizer_closed_is_not_taken_as_the_answer(ingress, send, monkeypatch):
    """The other side of review r3 P1-4: a failed turn's reply is closed by the same finalizer branch,
    unmarked. It is not a completed answer: the next process runs the message again and never sends
    the failed reply as its answer."""
    reply = _finalized_turn(ingress, monkeypatch, failed=True)
    nxt = _next_process(ingress, send, monkeypatch)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])
    assert not any(reply in text for _chat, text, _meta in nxt.adapter.sent_calls)


# ── Review r4 of 36ed681e ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("failure", ["extraction_raises", "ledger_write_and_send_fail"])
def test_a_reply_neither_ledgered_nor_delivered_keeps_its_record_and_marker_for_the_next_process(
        ingress, send, monkeypatch, failure):
    """Review r4 P1-1: the turn completes and writes its answer, then the adapter's delivery fails before
    that reply is in the delivery ledger: extracting the reply raises, or the ledger write fails and the
    send is refused. The process lives on. Neither the record nor the turn marker is released, so the
    next process delivers the stored answer once and runs no second turn."""
    from gateway import delivery_ledger
    from gateway.platforms.base import SendResult

    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)
    armed = []
    if failure == "extraction_raises":
        extract = h.adapter._extract_response_content

        async def failing_extract(*args, **kwargs):
            if armed:
                raise RuntimeError("extraction failed")
            return await extract(*args, **kwargs)
        monkeypatch.setattr(h.adapter, "_extract_response_content", failing_extract)
    else:
        record, send_with_retry = delivery_ledger.record_obligation, h.adapter._send_with_retry

        def failing_record(**kwargs):
            if armed:
                raise OSError("the delivery ledger is read-only")
            return record(**kwargs)

        async def refused_send(*, chat_id, content, **kwargs):
            if armed and _TERMINAL in content:
                return SendResult(success=False, error="Bad Gateway")
            return await send_with_retry(chat_id=chat_id, content=content, **kwargs)
        monkeypatch.setattr(delivery_ledger, "record_obligation", failing_record)
        monkeypatch.setattr(h.adapter, "_send_with_retry", refused_send)

    async def exercise():
        await h.warm()
        armed.append(True)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    armed.clear()
    entry = h.runner.session_store.lookup_by_session_key_existing(h.key)
    assert [chat for chat, text in h.owner_replies() if _TERMINAL in text] == []
    assert len(_records(ingress)) == 1 and entry.active_turn_token is not None
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch, crashed=True))
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)


def test_a_stream_recovered_reply_the_finalizer_filled_into_a_blank_row_is_answered_once(
        ingress, send, monkeypatch):
    """Review r4 P1-4: the incremental flush left a blank assistant row, the terminal completion was
    empty, and the real finalizer filled that row with the streamed text; production's flush then
    updates the row in place. A crash leaves the record and the turn marker. The filled row carries the
    completed-turn mark as an appended closing row does, so the answer is delivered once (the marker's
    recovery finds it already owed) and the message runs no second turn."""
    from types import SimpleNamespace

    from agent.session_persistence import _db_flush_row
    from agent.turn_finalizer import finalize_turn
    from gateway import shutdown_flush
    from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", lambda *_a, **_kw: [])
    db, sid, key = ingress.runner.session_store._db, ingress.entry.session_id, ingress.entry.session_key
    ingress.runner.session_store.mark_turn_active(key)  # held from the turn's start
    event = MessageEvent(text=_OWNER_TEXT, message_type=MessageType.TEXT, source=ingress.source, message_id="m1")
    shutdown_flush.store_inbound_record(event, session_key=key, transport_profile=None,
                                        recovery_id="streamed", order=(1, 1))
    flusher = SimpleNamespace(_persist_user_message_override=None, _persist_user_message_timestamp=None)
    messages = [{"role": "user", "content": _OWNER_TEXT, "display_metadata": {"inbound_recovery_id": "streamed"}},
                {"role": "assistant", "content": ""}]
    rows = [_db_flush_row(flusher, messages[0], True), _db_flush_row(flusher, messages[1], False)]
    db.append_messages_batch(sid, rows)  # the incremental flush: the blank row is on disk
    for message, row in zip(messages, rows):
        message.update(_row_id=row["_row_id"], timestamp=row["timestamp"], _db_persisted=True)
    reply = f"{_TERMINAL} / already streamed to the owner"
    agent = FakeAgent()
    agent._current_streamed_assistant_text = reply

    def flush(persisted, _history):  # rows not yet flushed; the filled blank row is updated in place
        db.append_messages_batch(sid, [_db_flush_row(flusher, m, False) for m in persisted
                                       if not m.get("_db_persisted")])
    agent._persist_session = flush
    finalize_turn(
        agent, final_response="", api_call_count=2, interrupted=False, failed=False, messages=messages,
        conversation_history=[], effective_task_id="task", turn_id="turn", user_message=_OWNER_TEXT,
        original_user_message=_OWNER_TEXT, _should_review_memory=False, _turn_exit_reason="text_response(final)")
    durable = [row for row in db.get_messages_as_conversation(sid) if row.get("role") == "assistant"]
    assert [row["content"] for row in durable] == [reply] and not durable[0].get("finish_reason")
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([], ["chat"], [])


@pytest.mark.parametrize("chain", ["as_is", "rewritten", "raises"])
def test_a_first_response_neither_ledgered_nor_delivered_keeps_its_record_through_the_chain(
        ingress, send, monkeypatch, chain):
    """The in-turn sibling of review r4 P1-1: an owner follow-up is drained inside the running turn, and
    the first message's answer, sent before the follow-up runs, fails with no ledger row (the ledger
    write fails and the send is refused). The process lives on. The first message's record stays
    through the chain's end: after the follow-up is answered, also when admission rewrote the turn's
    event (a pre-dispatch hook) so the record lives on the adapter's event, and when the chain raises
    before the follow-up runs. The next process delivers that answer once and does not run it again.
    After the raise too (the b9a08796 review: the ledger before the transcript): the first message's
    stored answer is still owed and is delivered, never run again, and the follow-up's error notice,
    delivered under the follow-up's own obligation, completed it, so its record is only cleaned up."""
    import dataclasses

    import gateway.run_turn_followup_ack as followup_ack
    from gateway import delivery_ledger
    from gateway.platforms.base import SendResult

    follow_up = "owner: one more thing"
    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)
    if chain == "rewritten":
        async def rewritten(event):
            return dataclasses.replace(event), event.source, False
        monkeypatch.setattr(h.runner, "_hm_admit_event", rewritten)
    elif chain == "raises":
        async def failing_hook(_adapter, _event, hook, *_args):
            raise RuntimeError(f"{hook} failed")
        monkeypatch.setattr(followup_ack, "_run_followup_processing_hook", failing_hook)
    first, failed = (threading.Event(), threading.Event()), []
    record, send_with_retry = delivery_ledger.record_obligation, h.adapter._send_with_retry

    def failing_record(**kwargs):  # the first response's ledger write, then the follow-up's
        if not failed:
            failed.append("ledger")
            raise OSError("the delivery ledger is read-only")
        return record(**kwargs)

    async def refused_send(*, chat_id, content, **kwargs):
        if _TERMINAL in content and failed == ["ledger"]:
            failed.append("send")
            return SendResult(success=False, error="Bad Gateway")
        return await send_with_retry(chat_id=chat_id, content=content, **kwargs)

    import run_agent
    build = run_agent.AIAgent

    def holding(**kwargs):  # every agent the session builds (a persisted row can rebuild it)
        actor = build(**kwargs)
        run = actor.run_conversation

        def held(text, *, conversation_history, task_id, **kw):
            if kw.get("persist_user_message", text) == _OWNER_TEXT:
                first[0].set()
                first[1].wait(30)
            return run(text, conversation_history=conversation_history, task_id=task_id, **kw)
        actor.run_conversation = held
        return actor
    monkeypatch.setattr(run_agent, "AIAgent", holding)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(first[0].is_set, "the first owner turn to run")
        await h.adapter.handle_message(h.owner_event(follow_up, message_id="m2"))
        await h.wait(lambda: h.queued() == [follow_up], "the follow-up to be queued")
        monkeypatch.setattr(delivery_ledger, "record_obligation", failing_record)
        monkeypatch.setattr(h.adapter, "_send_with_retry", refused_send)
        first[1].set()
        await h.settle()
    asyncio.run(exercise())
    ran = [] if chain == "raises" else [follow_up]  # the follow-up's record stays when it never ran
    assert failed == ["ledger", "send"]
    assert [text for actor in h.built for text in actor.calls][-1 - len(ran):] == [_OWNER_TEXT, *ran]
    assert [chat for chat, text in h.owner_replies() if _TERMINAL in text] == ["chat" for _ in ran]
    assert _record_texts(ingress) == sorted([_OWNER_TEXT, *([] if ran else [follow_up])])
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left) == ([], ["chat"], [])


@pytest.mark.parametrize("admitted", ["as_is", "rewritten"])
def test_a_rerun_after_a_restart_is_owed_under_the_run_its_record_fixed(ingress, send, monkeypatch, admitted):
    """Open item 5 of r4 slice 1, under the ledger-first recovery of the b9a08796 review: a stored
    message's earlier run left nothing in the ledger and only a truncated answer in the transcript, so
    the next process runs it again, also through a pre-dispatch hook that rewrites the event. The rerun's
    reply is owed under the record's next run (``rerun#1``), never under the earlier run, whose
    obligation stays free. (With anything in the ledger for the earlier run, the message is not run
    again: test_a_ledger_row_without_components_is_never_taken_as_a_complete_reply.)"""
    import dataclasses

    _stored_turn(ingress, "rerun", "partial: I was about to", finish_reason="length")

    def rewriting(nxt):
        async def rewrite(event, _source):
            return dataclasses.replace(event)
        monkeypatch.setattr(nxt.runner, "_hm_pre_gateway_dispatch_hook", rewrite)
    nxt = _next_process(ingress, send, monkeypatch, prepare=rewriting if admitted == "rewritten" else None)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], [])
    owed = _obligations(ingress)
    assert _inbound_obligation(ingress, "rerun") not in owed
    state, content = owed[_inbound_obligation(ingress, "rerun#1")]
    assert state == "delivered" and _TERMINAL in content


@pytest.mark.parametrize("escape", ["cancelled", "raised"])
def test_a_turn_that_raises_after_its_answer_was_stored_keeps_its_record(ingress, send, monkeypatch, escape):
    """Finding 1 at the turn's own end: the agent wrote its completed answer, then the turn is cancelled
    or raises out of the handler before it returns (here from the post-turn hooks; an exception escapes
    only when the failed-turn reply raises too, as the turn otherwise ends on that notice), so the
    adapter has no reply to deliver or ledger. The record of a message whose answer is stored is not
    deleted with the turn: the next process delivers that answer once and runs no second turn."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)

    async def failing_hooks(*_args, **_kwargs):
        raise asyncio.CancelledError() if escape == "cancelled" else RuntimeError("the post-turn hooks failed")

    async def failing_error_reply(*_args, **_kwargs):
        raise RuntimeError("the failed-turn reply failed")

    async def exercise():
        await h.warm()
        monkeypatch.setattr(h.runner, "_hmwa_post_turn_hooks", failing_hooks)
        monkeypatch.setattr(h.runner, "_hmwa_agent_error_reply", failing_error_reply)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    assert [chat for chat, text in h.owner_replies() if _TERMINAL in text] == []
    assert _record_texts(ingress) == [_OWNER_TEXT]
    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch))
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)


# ── Slice 1 review of ace7c2f5: one settlement boundary ───────────────────────────────────────────────
# An owner message's inbound record and its turn marker are deleted at one boundary only
# (gateway/run_inbound.py, _settle_inbound_records), and only when the obligation of that input, run and
# destination reads back delivered (or the caller states an explicit end that owes nothing). The run is
# fixed in the record before the turn's hooks. Each test below fails at ace7c2f5.


def _inbound_obligation(ingress, answer_ref, kind="answer"):
    """The obligation id of the run's answer, or (``kind="notice"``) of a notice it sent instead."""
    from gateway import delivery_ledger
    source = ingress.source
    return delivery_ledger.inbound_obligation_id(
        ingress.entry.session_key, answer_ref, platform=source.platform.value, chat_id=source.chat_id,
        thread_id=source.thread_id, **({} if kind == "answer" else {"kind": kind}))


def _obligations(ingress):
    """obligation id -> (state, content) of the session's delivery obligations."""
    from gateway import delivery_ledger
    with delivery_ledger._connect() as conn:
        return {oid: (state, content) for oid, state, content in conn.execute(
            "SELECT obligation_id, state, content FROM delivery_obligations WHERE session_key=?",
            (ingress.entry.session_key,)).fetchall()}


def _marker(h):
    return h.runner.session_store.lookup_by_session_key_existing(h.key).active_turn_token


def test_an_answered_message_keeps_its_record_and_marker_until_its_reply_reads_back_delivered(
        ingress, send, monkeypatch):
    """The normal path through the boundary: while the final reply is sent, its obligation is in the
    ledger and the record and the turn marker are still on disk; they go once that obligation reads
    back delivered."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)
    send_with_retry, seen = h.adapter._send_with_retry, []

    async def observed(*, chat_id, content, **kwargs):
        if _TERMINAL in content:
            seen.append((_records(ingress), _marker(h) is not None))
        return await send_with_retry(chat_id=chat_id, content=content, **kwargs)
    monkeypatch.setattr(h.adapter, "_send_with_retry", observed)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    [(records, marked)] = seen
    assert len(records) == 1 and marked  # on disk while the reply is on its way
    recovery_id = records[0][len("inbound-"):-len(".json")]
    assert _obligations(ingress)[_inbound_obligation(ingress, recovery_id)][0] == "delivered"
    assert _records(ingress) == [] and _marker(h) is None


def test_a_cancelled_turn_whose_transcript_cannot_be_read_keeps_its_record_and_marker(ingress, send, monkeypatch):
    """Slice 1 review P1-1: the agent stored its completed answer, then the turn is cancelled and reading
    the transcript fails while it ends. A cancelled turn proves no delivery, and a failed read is not
    "no answer": the record and the marker stay, and the next process delivers the stored answer once."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)
    load = h.runner.async_session_store.load_transcript

    async def unreadable(*_args, **_kwargs):
        raise OSError("transient transcript read failure")

    async def cancelled(*_args, **_kwargs):
        monkeypatch.setattr(h.runner.async_session_store, "load_transcript", unreadable)
        raise asyncio.CancelledError()

    async def exercise():
        await h.warm()
        monkeypatch.setattr(h.runner, "_hmwa_post_turn_hooks", cancelled)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    monkeypatch.setattr(h.runner.async_session_store, "load_transcript", load)
    assert _record_texts(ingress) == [_OWNER_TEXT] and _marker(h) is not None
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)
    assert _marker(nxt) is None


@pytest.mark.parametrize("failure", ["agent_start", "ledger_lookup", "read_back", "run_write"])
def test_a_rerun_is_fixed_before_its_hooks_and_a_failed_lookup_or_read_back_never_runs_it(
        ingress, send, monkeypatch, failure):
    """Slice 1 review P1-2 and P1-3, under the ledger-first recovery of the b9a08796 review: a stored
    message's earlier run left nothing in the ledger and only a truncated answer in the transcript, so
    the next process runs it again, under a run fixed in its record before the run's hooks: an
    agent:start hook that raises makes the rerun's error notice owed under that run (as a notice, its
    own obligation), where it is delivered and, the run having ended on it, settles the record
    (``agent_start``). A ledger lookup that fails at that start runs nothing and keeps the record with
    its run unchanged (``ledger_lookup``: a failed lookup never authorizes a transcript-based run). A
    read-back that fails after the rerun's reply was delivered keeps the record, and the next start
    finds that run's obligation complete and only cleans up (``read_back``). Lookup and read-back are
    one decision (delivery_ledger.inbound_completion). A run that cannot be written into the record
    admits nothing (``run_write``). Once the failure clears, the next start closes the record, sending
    at most one reply in all."""
    import json

    from gateway import delivery_ledger, shutdown_flush

    _stored_turn(ingress, "rerun", "partial: I was about to", finish_reason="length")
    record_path = ingress.home / "pending_messages" / "inbound-rerun.json"
    completion = delivery_ledger.inbound_completion
    reply = "notice" if failure == "agent_start" else "answer"

    def failing(fn, *, once, armed=lambda: True):
        calls = []

        def wrapped(*args, **kwargs):
            if armed() and not (once and calls):
                calls.append(True)
                raise OSError(f"transient {failure} failure")
            return fn(*args, **kwargs)
        return wrapped

    def prepare(nxt):
        _persisting_agents(monkeypatch)
        if failure == "agent_start":
            emit = nxt.runner.hooks.emit

            async def failing_hook(name, *args, **kwargs):
                if name == "agent:start":
                    raise RuntimeError("agent:start hook failed")
                return await emit(name, *args, **kwargs)
            monkeypatch.setattr(nxt.runner.hooks, "emit", failing_hook)
        elif failure == "ledger_lookup":  # every lookup of this process
            monkeypatch.setattr(delivery_ledger, "inbound_completion", failing(completion, once=False))
        elif failure == "read_back":  # the boundary's read-back, once the rerun has run and delivered
            monkeypatch.setattr(delivery_ledger, "inbound_completion", failing(
                completion, once=True, armed=lambda: any(actor.calls for actor in nxt.built)))
        else:
            monkeypatch.setattr(shutdown_flush, "set_inbound_record_run",
                                failing(shutdown_flush.set_inbound_record_run, once=True))

    turns, _inputs, chats, left = _answered(ingress, _next_process(ingress, send, monkeypatch, prepare=prepare))
    monkeypatch.setattr(delivery_ledger, "inbound_completion", completion)
    owed = _obligations(ingress)
    assert _inbound_obligation(ingress, "rerun") not in owed
    if failure == "agent_start":  # its notice is the rerun's own reply, delivered: the record is settled
        assert (turns, chats, left) == ([], [], [])
        assert owed[_inbound_obligation(ingress, "rerun#1", reply)][0] == "delivered"
    elif failure == "read_back":  # answered and delivered, but the read-back failed: kept
        assert (turns, chats, left) == ([_OWNER_TEXT], ["chat"], ["inbound-rerun.json"])
    else:  # nothing run, nothing admitted
        assert (turns, chats, left) == ([], [], ["inbound-rerun.json"])
        assert json.loads(record_path.read_text()).get("run", 0) == 0 and owed == {}
    nxt = _next_process(ingress, send, monkeypatch, prepare=lambda _nxt: _persisting_agents(monkeypatch))
    turns, _inputs, chats, left = _answered(ingress, nxt)
    owed = _obligations(ingress)
    assert left == [] and owed[_inbound_obligation(ingress, "rerun#1", reply)][0] == "delivered"
    if failure in ("ledger_lookup", "run_write"):
        assert (turns, chats) == ([_OWNER_TEXT], ["chat"])
    else:  # nothing sent again, nothing run again
        assert (turns, chats) == ([], [])


@pytest.mark.parametrize("delivery", ["text_refused_image_sent", "text_sent_image_refused"])
def test_a_reply_with_an_attachment_is_settled_only_when_every_component_was_delivered(
        ingress, send, monkeypatch, delivery):
    """Slice 1 review P1-4: the reply is text plus an image. An image that went out never settles text
    that did not (the ledger write failed and the text send was refused), and text that went out never
    settles an image that did not: the record and the marker stay. The next process sends only the
    component still owed (b9a08796 review P1-1): the text the first lost (from the reply's stored
    components, as its own ledger row was never written), never the image it delivered; the second's
    image, never its delivered text."""
    import run_agent
    from gateway import delivery_ledger
    from gateway.platforms.base import SendResult

    h = _Harness(ingress, send, monkeypatch, "queue")
    build = run_agent.AIAgent

    def with_image(**kwargs):
        actor = build(**kwargs)
        run = actor.run_conversation

        def image_result(text, **kw):
            result = run(text, **kw)
            if kw.get("persist_user_message", text) == _OWNER_TEXT:
                result["final_response"] += "\n![chart](https://example.invalid/chart.png)"
                result["messages"][-1]["content"] = result["final_response"]
            return result
        actor.run_conversation = image_result
        return actor
    monkeypatch.setattr(run_agent, "AIAgent", with_image)
    _persisting_agents(monkeypatch)
    record, send_with_retry, armed, recorded = delivery_ledger.record_obligation, h.adapter._send_with_retry, [], []
    text_lost = delivery == "text_refused_image_sent"

    def failing_record(**kwargs):
        if armed and text_lost:
            raise OSError("the delivery ledger is read-only")
        return record(**kwargs)

    async def text_send(*, chat_id, content, **kwargs):
        if armed and text_lost and _TERMINAL in content:
            return SendResult(success=False, error="text send refused")
        return await send_with_retry(chat_id=chat_id, content=content, **kwargs)

    async def image_send(*, chat_id, images, **_kwargs):
        recorded.extend(images)
        if text_lost:
            return SendResult(success=True, message_id="image1")
        return SendResult(success=False, error="image refused")
    monkeypatch.setattr(delivery_ledger, "record_obligation", failing_record)
    monkeypatch.setattr(h.adapter, "_send_with_retry", text_send)
    monkeypatch.setattr(h.adapter, "send_multiple_images", image_send)

    async def exercise():
        await h.warm()
        armed.append(True)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    armed.clear()
    delivered_text = [chat for chat, text in h.owner_replies() if _TERMINAL in text]
    assert recorded == [("https://example.invalid/chart.png", "chart")]
    assert delivered_text == ([] if text_lost else ["chat"])
    assert _record_texts(ingress) == [_OWNER_TEXT] and _marker(h) is not None
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([], ["chat"] if text_lost else [], [])
    # The next process's adapter sends an image through the base batch, which shows its URL as text.
    assert [t for t in _texts_sent(nxt) if "example.invalid/chart.png" in t] == ([] if text_lost else ["chart\nhttps://example.invalid/chart.png"])
    assert _marker(nxt) is None


@pytest.mark.parametrize("failure", ["send_refused", "delivered_unrecorded", "transcript_unreadable"])
def test_a_transient_failure_is_kept_then_recovered_delivered_and_cleaned_up_after_a_restart(
        ingress, send, monkeypatch, failure):
    """Retaining alone is not done. The reply's send is refused (its obligation is in the ledger, failed);
    or it is sent but marking it delivered fails, so the ledger never reads back delivered (a send that
    succeeded is not proof); or it is neither ledgered nor delivered and the next start cannot read the
    transcript (that start neither runs it again nor deletes it). The record and the marker stay. Once
    the failure has cleared, a restart recovers the obligation, delivers it (the unrecorded one again,
    marked as a possible duplicate), reads it back delivered, and only then deletes the record and the
    marker."""
    import json

    from gateway import delivery_ledger
    from gateway.platforms.base import SendResult

    h = _Harness(ingress, send, monkeypatch, "queue")
    _persisting_agents(monkeypatch)
    record, send_with_retry, armed = delivery_ledger.record_obligation, h.adapter._send_with_retry, []
    mark_delivered = delivery_ledger.mark_delivered

    def unrecorded(*args, **kwargs):
        if armed:
            raise OSError("the delivery ledger is read-only")
        return mark_delivered(*args, **kwargs)
    if failure == "delivered_unrecorded":
        monkeypatch.setattr(delivery_ledger, "mark_delivered", unrecorded)

    def failing_record(**kwargs):
        if armed and failure == "transcript_unreadable":
            raise OSError("the delivery ledger is read-only")
        return record(**kwargs)
    store_components = getattr(delivery_ledger, "record_components", None)

    def failing_components(*args, **kwargs):  # the whole ledger write: the reply's components too
        if armed and failure == "transcript_unreadable":
            raise OSError("the delivery ledger is read-only")
        return store_components(*args, **kwargs)

    async def refused(*, chat_id, content, **kwargs):
        if armed and _TERMINAL in content and failure != "delivered_unrecorded":
            return SendResult(success=False, error="Bad Gateway")
        return await send_with_retry(chat_id=chat_id, content=content, **kwargs)
    monkeypatch.setattr(delivery_ledger, "record_obligation", failing_record)
    monkeypatch.setattr(delivery_ledger, "record_components", failing_components, raising=False)
    monkeypatch.setattr(h.adapter, "_send_with_retry", refused)

    async def exercise():
        await h.warm()
        armed.append(True)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    armed.clear()
    [record_file] = _records(ingress)
    recovery_id = record_file[len("inbound-"):-len(".json")]
    sent_first = [c for c, t in h.owner_replies() if _TERMINAL in t]
    assert _marker(h) is not None and sent_first == (["chat"] if failure == "delivered_unrecorded" else [])
    if failure != "transcript_unreadable":
        assert _obligations(ingress)[_inbound_obligation(ingress, recovery_id)][0] == (
            "failed" if failure == "send_refused" else "attempting")
        with delivery_ledger._connect() as conn:  # the process that owned the row is gone
            conn.execute("UPDATE delivery_obligations SET owner_pid=999999999, owner_started_at=1 "
                         "WHERE session_key=?", (h.key,))
            conn.commit()
    else:
        def unreadable(nxt):
            async def failing(*_args, **_kwargs):
                raise OSError("transient transcript read failure")
            monkeypatch.setattr(nxt.runner.async_session_store, "load_transcript", failing)
        turns, _inputs, chats, left = _answered(
            ingress, _next_process(ingress, send, monkeypatch, crashed=True, prepare=unreadable))
        assert (turns, chats, left) == ([], [], [record_file])  # unknown: neither run again nor deleted
        stored = json.loads((ingress.home / "pending_messages" / record_file).read_text())
        assert stored.get("run", 0) == 0  # and no rerun was admitted
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left, _owner_rows(ingress)) == ([], ["chat"], [], 1)
    assert _obligations(ingress)[_inbound_obligation(ingress, recovery_id)][0] == "delivered"
    assert _marker(nxt) is None


def test_an_inbound_obligation_is_never_taken_over_by_another_reply(ingress):
    """The ledger side of P1-2: an obligation that names an input, run and destination is insert-only.
    Another reply under it (an error notice, a retry) raises and records nothing; the same reply again
    leaves the row as it is."""
    from gateway import delivery_ledger
    source = ingress.source
    oid = _inbound_obligation(ingress, "taken")
    row = dict(obligation_id=oid, session_key=ingress.entry.session_key, platform=source.platform.value,
               chat_id=source.chat_id, thread_id=source.thread_id, replace=False)
    delivery_ledger.record_obligation(content="the answer", **row)
    with pytest.raises(delivery_ledger.ObligationTaken):
        delivery_ledger.record_obligation(content="⚠️ an error notice", **row)
    delivery_ledger.record_obligation(content="the answer", **row)
    assert _obligations(ingress)[oid] == ("pending", "the answer")


# ── Slice 1 review of b9a08796: what a reply requires, and the ledger before the transcript ─────────────
# What an owner message's reply requires (its text and each attachment, under component ids stable within
# the reply) is stored under the reply's obligation before any of it is sent
# (delivery_ledger.record_components). The boundary releases the record only when every component reads
# back delivered, on the adapter's final, the queued lane and a later start alike; a later start sends
# only what is still owed. Recovery reads that obligation before it reads the transcript.

_CHART = "![chart](https://example.invalid/chart.png)"
_CHART_IMAGE = ("https://example.invalid/chart.png", "chart")


def _answering_with(monkeypatch, attachment, *, text=True, hold=None):
    """The owner turn's answer carries *attachment* after its text (without ``text``, instead of it), in
    what it returns and in what it stores. With ``hold`` (entered, release) the owner turn waits first."""
    import run_agent
    build = run_agent.AIAgent

    def building(**kwargs):
        actor = build(**kwargs)
        run = actor.run_conversation

        def answering(message, **kw):
            owner = kw.get("persist_user_message", message) == _OWNER_TEXT
            if owner and hold is not None:
                hold[0].set()
                assert hold[1].wait(20)
            result = run(message, **kw)
            if owner:
                result["final_response"] = (result["final_response"] + "\n" if text else "") + attachment
                result["messages"][-1]["content"] = result["final_response"]
            return result
        actor.run_conversation = answering
        return actor
    monkeypatch.setattr(run_agent, "AIAgent", building)
    _persisting_agents(monkeypatch)


def _attachment_sends(monkeypatch, refused, order=None):
    """Every image batch and document any process's adapter sends, as (kind, what, delivered); a kind in
    *refused* (a set the test edits) is refused. Each send's kind is also appended to *order*."""
    from gateway.platforms.base import SendResult
    sent = []

    def outcome(kind, what):
        ok = kind not in refused
        sent.append((kind, what, ok))
        if order is not None:
            order.append(kind)
        return SendResult(success=ok, message_id=f"{kind}-id" if ok else None,
                          error=None if ok else f"{kind} send refused")

    async def images(self, *, chat_id, images, **_kwargs):
        return outcome("image", list(images))

    async def document(self, *, chat_id, file_path, **_kwargs):
        return outcome("document", file_path)
    monkeypatch.setattr(RestartTestAdapter, "send_multiple_images", images)
    monkeypatch.setattr(RestartTestAdapter, "send_document", document)
    return sent


def _components(ingress, answer_ref, kind="answer"):
    from gateway import delivery_ledger
    return [(c["id"], c["kind"], c["state"])
            for c in delivery_ledger.obligation_components(_inbound_obligation(ingress, answer_ref, kind))]


def _recovery_id(record_file):
    return record_file[len("inbound-"):-len(".json")]


def _texts_sent(nxt):
    return [text for _chat, text, _meta in nxt.adapter.sent_calls]


@pytest.mark.parametrize("reply", ["text_only", "attachment_only", "text_and_attachment"])
def test_a_reply_stores_what_it_requires_before_its_first_send_and_is_settled_once_all_of_it_is_delivered(
        ingress, send, monkeypatch, reply):
    """The normal path, text only, an attachment only (review P1-3: it has no text row, so a text check
    can never settle it, and a later start presented it again), and both: the reply's components are in
    the ledger before the first send of any of them, each reads back delivered, and the record and the
    marker go. A later start sends nothing and runs nothing."""
    from gateway import delivery_ledger

    h = _Harness(ingress, send, monkeypatch, "queue")
    if reply == "text_only":
        _persisting_agents(monkeypatch)
    else:
        _answering_with(monkeypatch, _CHART, text=reply == "text_and_attachment")
    order, stored = [], []
    sent = _attachment_sends(monkeypatch, set(), order)
    record_components = getattr(delivery_ledger, "record_components", None)

    def recording(obligation_id, components):
        order.append("stored")
        stored.append((obligation_id, [(c["id"], c["kind"]) for c in components]))
        return record_components(obligation_id, components)
    monkeypatch.setattr(delivery_ledger, "record_components", recording, raising=False)
    send_with_retry = h.adapter._send_with_retry

    async def text_send(*, chat_id, content, **kwargs):
        if _TERMINAL in content:
            order.append("text")
        return await send_with_retry(chat_id=chat_id, content=content, **kwargs)
    monkeypatch.setattr(h.adapter, "_send_with_retry", text_send)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    assert [ok for _kind, _what, ok in sent] == ([] if reply == "text_only" else [True])
    assert _records(ingress) == [] and _marker(h) is None
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left, _texts_sent(nxt)) == ([], [], [], [])
    assert len(sent) == (0 if reply == "text_only" else 1)
    expected = {"text_only": [("text", "text")], "attachment_only": [("image:0", "image")],
                "text_and_attachment": [("text", "text"), ("image:0", "image")]}[reply]
    [(obligation_id, components)] = stored
    assert components == expected and order[0] == "stored"
    assert order[1:] == [kind for _id, kind in expected]
    assert all(c["state"] == "delivered" for c in delivery_ledger.obligation_components(obligation_id))


@pytest.mark.parametrize("refused", ["image", "document"])
def test_a_partly_delivered_reply_is_kept_and_a_restart_sends_only_what_it_still_owes(
        ingress, send, monkeypatch, tmp_path, refused):
    """Review P1-1: the reply is text, an image and a document, and one attachment is refused. The record
    and the marker stay. Once the failure clears, the next start sends that attachment and nothing else
    (neither the delivered text nor the delivered attachment), reads every component back delivered, and
    only then cleans up. The message is not run again."""
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-test")
    h = _Harness(ingress, send, monkeypatch, "queue")
    _answering_with(monkeypatch, f"{_CHART}\nMEDIA:{document}")
    failing = {refused}
    sent = _attachment_sends(monkeypatch, failing)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    assert [(kind, ok) for kind, _what, ok in sent] == [("image", refused != "image"),
                                                         ("document", refused != "document")]
    assert [c for c, t in h.owner_replies() if _TERMINAL in t] == ["chat"]
    [record_file] = _records(ingress)
    assert _marker(h) is not None
    failing.clear()
    first = len(sent)
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([], [], [])
    assert [(kind, ok) for kind, _what, ok in sent[first:]] == [(refused, True)]
    assert all(state == "delivered" for _id, _kind, state in _components(ingress, _recovery_id(record_file)))
    assert _marker(nxt) is None


def test_a_queued_first_reply_whose_attachment_is_refused_keeps_its_record_until_a_restart_sends_it(
        ingress, send, monkeypatch, tmp_path):
    """Review P1-2: the first turn's reply (text and a MEDIA: document) goes out in the queued lane while
    a follow-up waits. Its text is delivered and its document refused. The first message's record stays
    (the follow-up's goes with its own answer). Once the failure clears, the next start sends the document
    alone and cleans up; neither message runs again."""
    document = tmp_path / "chart.pdf"
    document.write_bytes(b"%PDF-test")
    h = _Harness(ingress, send, monkeypatch, "queue")
    hold = threading.Event(), threading.Event()
    _answering_with(monkeypatch, f"MEDIA:{document}", hold=hold)
    failing = {"document"}
    sent = _attachment_sends(monkeypatch, failing)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(hold[0].is_set, "the first turn")
        await h.adapter.handle_message(h.owner_event("owner: followup", message_id="m2"))
        await h.wait(lambda: h.queued() == ["owner: followup"], "the follow-up to queue")
        hold[1].set()
        await h.settle()
    asyncio.run(exercise())
    assert [(kind, what, ok) for kind, what, ok in sent] == [("document", str(document), False)]
    assert sum(_TERMINAL in t for _c, t in h.owner_replies()) == 2  # the first reply's text and the follow-up's
    assert _record_texts(ingress) == [_OWNER_TEXT]
    [record_file] = _records(ingress)
    from gateway import delivery_ledger
    held = [(c["id"], c["kind"], c["state"]) for c in getattr(delivery_ledger, "obligation_components", list)(
        _inbound_obligation(ingress, _recovery_id(record_file)))]
    assert held == [("text", "text", "delivered"), ("media:0", "media", "failed")]  # stored, then recorded
    failing.clear()
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([], [], [])
    assert [(kind, ok) for kind, _what, ok in sent[1:]] == [("document", True)]


@pytest.mark.parametrize("reply", ["error_notice", "answer_with_attachment"])
def test_a_delivered_reply_whose_record_could_not_be_deleted_is_only_cleaned_up_after_a_restart(
        ingress, send, monkeypatch, reply):
    """Review P1-4: agent:start raises, so the turn's reply is its error notice, delivered under the run's
    obligation; deleting the record fails. The next start reads that obligation first and finds it
    complete, so it only cleans up: the failed turn in the transcript never runs the message again, and
    nothing is sent. The same for an answer with an attachment, every component of it delivered."""
    import pathlib

    h = _Harness(ingress, send, monkeypatch, "queue")
    if reply == "error_notice":
        _persisting_agents(monkeypatch)
    else:
        _answering_with(monkeypatch, _CHART)
    sent = _attachment_sends(monkeypatch, set())
    unlink, refused = pathlib.Path.unlink, []

    def failing_unlink(path, *args, **kwargs):
        if path.name.startswith("inbound-"):
            refused.append(path.name)
            raise OSError("temporary delete failure")
        return unlink(path, *args, **kwargs)

    async def exercise():
        await h.warm()
        if reply == "error_notice":
            emit = h.runner.hooks.emit

            async def failing_hook(name, *args, **kwargs):
                if name == "agent:start":
                    raise RuntimeError("temporary hook failure")
                return await emit(name, *args, **kwargs)
            monkeypatch.setattr(h.runner.hooks, "emit", failing_hook)
        monkeypatch.setattr(pathlib.Path, "unlink", failing_unlink)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    if reply == "error_notice":
        assert any("couldn't finish" in text for _chat, text in h.owner_replies())
    else:
        assert [c for c, t in h.owner_replies() if _TERMINAL in t] == ["chat"] and len(sent) == 1
    [record_file] = _records(ingress)
    assert refused and _marker(h) is not None
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, inputs, chats, left = _answered(ingress, nxt)
    assert (turns, inputs, chats, left, _texts_sent(nxt)) == ([], [], [], [], [])
    assert len(sent) == (0 if reply == "error_notice" else 1) and _marker(nxt) is None
    assert all(state == "delivered" for _id, _kind, state in _components(ingress, _recovery_id(record_file)))


def _stored_turn(ingress, recovery_id, answer, *, finish_reason="stop"):
    """A stored message *recovery_id* whose turn's user row names it, answered by *answer*."""
    from types import SimpleNamespace

    from agent.session_persistence import _db_flush_row
    from gateway import shutdown_flush

    db, sid, key, source = (ingress.runner.session_store._db, ingress.entry.session_id,
                            ingress.entry.session_key, ingress.source)
    event = MessageEvent(text=_OWNER_TEXT, message_type=MessageType.TEXT, source=source, message_id="m1")
    shutdown_flush.store_inbound_record(event, session_key=key, transport_profile=None,
                                        recovery_id=recovery_id, order=(1, 1))
    flusher = SimpleNamespace(_persist_user_message_override=None, _persist_user_message_timestamp=None)
    db.append_messages_batch(sid, [
        _db_flush_row(flusher, {"role": "user", "content": _OWNER_TEXT, "platform_message_id": "m1",
                                "display_metadata": {"inbound_recovery_id": recovery_id}}, True),
        {"role": "assistant", "content": answer, "finish_reason": finish_reason}])


@pytest.mark.parametrize("transcript", ["answer_with_attachment", "no_answer"])
def test_a_ledger_row_without_components_is_never_taken_as_a_complete_reply(
        ingress, send, monkeypatch, caplog, transcript):
    """Compatibility: a stored message whose reply is in the ledger as a text row only, written before
    components were stored. It is not presumed text-only and complete. With that run's answer in the
    transcript (text and an image), what else the reply required is read from there: the image is sent,
    the delivered text is not sent again, and the record is cleaned up. Without one (a truncated answer),
    the record is kept and reported, the row is left as it is, and the message is not run again."""
    import logging

    from gateway import delivery_ledger

    answer = f"{_TERMINAL} legacy\n{_CHART}" if transcript == "answer_with_attachment" else "partial: I was"
    _stored_turn(ingress, "legacy", answer, finish_reason="stop" if transcript != "no_answer" else "length")
    source, key = ingress.source, ingress.entry.session_key
    oid = _inbound_obligation(ingress, "legacy")
    delivery_ledger.record_obligation(obligation_id=oid, session_key=key, platform=source.platform.value,
                                      chat_id=source.chat_id, thread_id=source.thread_id,
                                      content=f"{_TERMINAL} legacy" if transcript != "no_answer" else answer)
    if transcript != "no_answer":
        delivery_ledger.mark_delivered(oid)
    sent = _attachment_sends(monkeypatch, set())
    with caplog.at_level(logging.ERROR):
        nxt = _next_process(ingress, send, monkeypatch)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, _texts_sent(nxt)) == ([], [], [])
    if transcript == "answer_with_attachment":
        assert sent == [("image", [_CHART_IMAGE], True)] and left == []
        assert _components(ingress, "legacy") == [("text", "text", "delivered"), ("image:0", "image", "delivered")]
    else:
        assert sent == [] and left == ["inbound-legacy.json"] and _components(ingress, "legacy") == []
        assert _obligations(ingress)[oid] == ("pending", answer)
        assert any("without the components" in r.getMessage() for r in caplog.records)


def test_an_owed_attachment_whose_source_is_gone_is_never_counted_delivered(
        ingress, send, monkeypatch, tmp_path):
    """The text is delivered and the document refused, then the document's file is deleted. A later
    start never counts it delivered: nothing is sent, the component reads missing, the record and the
    marker stay, and the chat is told once; the start after that tells it nothing more. The message is
    never run again."""
    document = tmp_path / "gone.pdf"
    document.write_bytes(b"%PDF-test")
    h = _Harness(ingress, send, monkeypatch, "queue")
    _answering_with(monkeypatch, f"MEDIA:{document}")
    failing = {"document"}
    sent = _attachment_sends(monkeypatch, failing)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    [record_file] = _records(ingress)
    failing.clear()
    document.unlink()
    notices = []
    for _start in range(2):
        nxt = _next_process(ingress, send, monkeypatch, crashed=True)
        turns, _inputs, chats, left = _answered(ingress, nxt)
        assert (turns, chats, left) == ([], [], [record_file])
        notices.append([t for t in _texts_sent(nxt) if "gone.pdf" in t])
    assert len(sent) == 1  # the refused send only: a gone file is never sent
    assert [len(n) for n in notices] == [1, 0]
    assert ("media:0", "media", "missing") in _components(ingress, _recovery_id(record_file))
    assert _marker(nxt) is not None


def test_one_refused_image_of_two_is_owed_alone_though_its_batch_reported_success(ingress, send, monkeypatch):
    """A batch of images reports success when any one of them went out (platforms/base.py,
    send_multiple_images), so it proves nothing about the others: each tracked image goes as its own
    batch. One of two is refused: the record stays, and the next start sends that one alone."""
    from gateway.platforms.base import SendResult

    second = "![table](https://example.invalid/table.png)"
    h = _Harness(ingress, send, monkeypatch, "queue")
    _answering_with(monkeypatch, f"{_CHART}\n{second}")
    failing, shown = {"https://example.invalid/table.png"}, []

    async def image(self, *, chat_id, image_url, **_kwargs):
        ok = image_url not in failing
        shown.append((image_url, ok))
        return SendResult(success=ok, message_id="img" if ok else None, error=None if ok else "image refused")
    monkeypatch.setattr(RestartTestAdapter, "send_image", image)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    assert shown == [("https://example.invalid/chart.png", True), ("https://example.invalid/table.png", False)]
    [record_file] = _records(ingress)
    failing.clear()
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([], [], [])
    assert shown[2:] == [("https://example.invalid/table.png", True)]


# ── Slice 1 review of 2aa38989: one completion decision ────────────────────────────────────────────────
# Whether an owner message's reply is delivered is decided in one place (delivery_ledger.inbound_completion),
# from what the reply is (its answer, or a notice sent instead, each its own obligation), the state of each
# of its components, and its text's retry state. The settlement boundary and every recovery step read it.
# The first five tests are the review's reproductions; each fails at 2aa38989.


@pytest.mark.parametrize("ended", ["cut_short", "answer_stored"])
def test_a_notice_a_restart_cut_short_never_stands_in_for_the_answer_the_run_stored(
        ingress, send, monkeypatch, ended):
    """s1c review P1-1: the agent stored its completed answer, then a post-turn hook raises, so the
    turn's reply is an error notice: while the gateway drains (a run a restart cut short), or with no
    restart at all (a run that ended on the notice after storing its answer). The notice is its own
    obligation (a notice, superseded), never completion evidence for the stored answer: the record and
    the marker stay, and the next start owes that answer under the answer's obligation, delivers it,
    runs nothing again, and only then cleans up. The delivered notice's obligation stays as history."""
    import run_agent

    from gateway import delivery_ledger
    from gateway.shutdown_flush import load_inbound_records

    h = _Harness(ingress, send, monkeypatch, "queue")
    build = run_agent.AIAgent

    def building(**kwargs):  # the agent's own flush: the turn's user row as handed to it, and its answer
        actor = build(**kwargs)
        run = actor.run_conversation

        def writing(text, **kw):
            result = run(text, **kw)
            actor._session_db.append_messages_batch(kw["task_id"], [
                result["messages"][-2], {"role": "assistant", "content": result["final_response"],
                                         "finish_reason": "stop"}])
            return result
        actor.run_conversation = writing
        return actor
    monkeypatch.setattr(run_agent, "AIAgent", building)

    async def draining_hook(*_args, **_kwargs):
        if ended == "cut_short":
            h.runner._draining = True
            await h.runner.async_session_store.mark_resume_pending(h.key)
        raise RuntimeError("post-turn hook failed")

    async def exercise():
        await h.warm()
        monkeypatch.setattr(h.runner, "_hmwa_post_turn_hooks", draining_hook)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    [record] = load_inbound_records()
    rid = record["recovery_id"]
    assert [t for _c, t in h.owner_replies() if _TERMINAL in t] == [] and _marker(h) is not None
    assert _components(ingress, rid) == []  # the notice never took the answer's obligation
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left) == ([], ["chat"], [])
    assert _marker(nxt) is None and _components(ingress, rid) == [("text", "text", "delivered")]
    notice = delivery_ledger.obligation_components(_inbound_obligation(ingress, rid, "notice"))
    flag = delivery_ledger.CUT_SHORT if ended == "cut_short" else delivery_ledger.ANSWER_STORED
    assert [(c["kind"], c["state"], c[delivery_ledger.REPLY], c.get(flag))
            for c in notice] == [("text", "delivered", "notice", True)]  # kept, superseded


def test_a_reply_whose_components_cannot_be_stored_sends_nothing_until_they_are(ingress, send, monkeypatch):
    """s1c review P1-2: storing what an attachment-only reply requires fails. That is not a normal
    result: nothing of the reply is sent (an untracked send could neither prove it delivered nor keep a
    later start from sending it again), and the record and the marker stay. Once the store works again,
    the next start owes the reply from its transcript, sends its image once and cleans up; the message is
    not run again."""
    from gateway import delivery_ledger

    h = _Harness(ingress, send, monkeypatch, "queue")
    _answering_with(monkeypatch, _CHART, text=False)
    sent = _attachment_sends(monkeypatch, set())
    store = delivery_ledger.record_components

    def unavailable(*_args, **_kwargs):
        raise OSError("temporary ledger write failure")

    async def exercise():
        await h.warm()
        monkeypatch.setattr(delivery_ledger, "record_components", unavailable)
        await h.adapter.handle_message(h.owner_event())
        await h.settle()
    asyncio.run(exercise())
    monkeypatch.setattr(delivery_ledger, "record_components", store)
    assert sent == [] and len(_records(ingress)) == 1 and _marker(h) is not None
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, _chats, left = _answered(ingress, nxt)
    assert (turns, left, sent) == ([], [], [("image", [_CHART_IMAGE], True)])
    assert _marker(nxt) is None


@pytest.mark.parametrize("back", ["next_start", "after_the_bound"])
def test_a_stored_answer_whose_attachment_is_gone_keeps_it_owed_as_missing(
        ingress, send, monkeypatch, tmp_path, caplog, back):
    """s1c review P1-3: a completed answer, text and a PDF, was stored before any of its components
    were, then the PDF is deleted. The next start owes both: the text is delivered, the PDF stays one of
    the reply's components, recorded missing (never dropped, never counted delivered), and the record
    stays. Once the file is back, the start after that sends the PDF alone and cleans up. The tries are
    bounded like a text's reactivation: after MAX_REACTIVATIONS starts found it gone, a start sends
    nothing even with the file back; the PDF stays missing, the record stays, and an ERROR says so. The
    message is never run again."""
    import logging

    from gateway import delivery_ledger

    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-test")
    _stored_turn(ingress, "missing", f"{_TERMINAL}\nMEDIA:{document}")
    document.unlink()
    sent = _attachment_sends(monkeypatch, set())
    for start in range(1 if back == "next_start" else delivery_ledger.MAX_REACTIVATIONS):
        nxt = _next_process(ingress, send, monkeypatch, crashed=True)
        turns, _inputs, chats, left = _answered(ingress, nxt)
        assert (turns, chats, left, sent) == ([], ["chat"] if start == 0 else [], ["inbound-missing.json"], [])
        assert _components(ingress, "missing") == [("text", "text", "delivered"), ("media:0", "media", "missing")]
    document.write_bytes(b"%PDF-test")
    with caplog.at_level(logging.ERROR):
        nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    if back == "next_start":
        assert (turns, chats, left) == ([], [], [])
        assert [(kind, ok) for kind, _what, ok in sent] == [("document", True)]
    else:
        assert (turns, chats, left, sent) == ([], [], ["inbound-missing.json"], [])
        assert ("media:0", "media", "missing") in _components(ingress, "missing")
        assert any("not tried again" in r.getMessage() for r in caplog.records)


def test_a_stored_answer_that_is_only_an_attachment_is_delivered_and_never_run_again(
        ingress, send, monkeypatch, tmp_path):
    """s1c review P1-4: the completed answer in the transcript is a MEDIA: file alone. Its display text
    is empty once the directive is stripped, which is no reason to call the message unanswered: the next
    start reads the reply's components from the answer as stored, sends the PDF once and cleans up, and
    runs nothing."""
    document = tmp_path / "completed.pdf"
    document.write_bytes(b"%PDF-test")
    _stored_turn(ingress, "onlyfile", f"MEDIA:{document}")
    sent = _attachment_sends(monkeypatch, set())
    nxt = _next_process(ingress, send, monkeypatch)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left, _texts_sent(nxt)) == ([], [], [], [])
    assert [(kind, ok) for kind, _what, ok in sent] == [("document", True)]
    assert _components(ingress, "onlyfile") == [("media:0", "media", "delivered")]


@pytest.mark.parametrize("cause", ["stale", "attempts", "send_forbidden", "bound_spent"])
def test_a_reply_whose_retry_budget_is_spent_is_reactivated_a_bounded_number_of_times(
        ingress, send, monkeypatch, cause):
    """s1c review P1-5: the stored answer's text row spent its retry budget undelivered (the 24-hour
    cutoff, or the attempts cap) while the record still owes it, and the transport works again. The next
    start reactivates it for one more send, reads it back delivered and cleans up. Reactivation is
    bounded and selective: a row its chat refused for good (send-forbidden), or one already reactivated
    MAX_REACTIVATIONS times, is never revived, not even after retention ran over its old row; it and
    the record stay, and neither of two starts sends anything or runs the message again."""
    import time

    from gateway import delivery_ledger as ledger

    _stored_turn(ingress, "expired", _TERMINAL)
    oid, source = _inbound_obligation(ingress, "expired"), ingress.source
    ledger.record_components(oid, [{"id": "text", "kind": "text", "text": _TERMINAL}])
    ledger.record_obligation(obligation_id=oid, session_key=ingress.entry.session_key, platform="telegram",
                             chat_id=source.chat_id, thread_id=source.thread_id, content=_TERMINAL)
    stale, revived = cause == "stale", cause in ("stale", "attempts")
    # A row that is never revived was abandoned long ago, past the ledger's retention.
    age = ledger.STALE_AFTER_SECONDS + 1 if stale else (0 if revived else 8 * 24 * 60 * 60)
    with ledger._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET state=?, attempts=?, created_at=?, updated_at=?, last_error=?, "
            "owner_pid=NULL, owner_started_at=NULL WHERE obligation_id=?",
            ("failed" if revived else "abandoned", 0 if stale else ledger.MAX_ATTEMPTS, time.time() - age,
             time.time() - age, "Forbidden: bot was blocked by the user" if cause == "send_forbidden" else "Bad Gateway",
             oid))
        if cause == "bound_spent":
            conn.execute("UPDATE delivery_obligations SET reactivations=? WHERE obligation_id=?",
                         (ledger.MAX_REACTIVATIONS, oid))
        conn.commit()
    for start in range(2):
        nxt = _next_process(ingress, send, monkeypatch, crashed=True)
        turns, _inputs, chats, left = _answered(ingress, nxt)
        assert (turns, chats) == ([], ["chat"] if revived and start == 0 else [])
        assert left == ([] if revived else ["inbound-expired.json"])
        # Another reply is recorded meanwhile, so the ledger's retention runs over the old row.
        ledger.record_obligation(obligation_id=f"other-{start}", session_key="other", platform="telegram",
                                 chat_id="other", thread_id=None, content="another reply")
    assert _components(ingress, "expired") == [("text", "text", "delivered" if revived else "pending")]


def test_store_partial_delivery_interruption_restart_recovery_and_cleanup(ingress, send, monkeypatch, tmp_path):
    """End-to-end acceptance. An owner message's answer is text and a PDF. Its components are stored
    before the first send; the text is delivered; while the PDF is on its way the process is killed. The
    record and the marker stay, the PDF still owed. The restart recovers only what is incomplete (the
    PDF: never the delivered text, never a rerun), reads every component back delivered, and only then
    deletes the record and the marker."""
    from gateway import delivery_ledger

    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-test")
    h = _Harness(ingress, send, monkeypatch, "queue")
    _answering_with(monkeypatch, f"MEDIA:{document}")
    order, sent = [], []
    store = delivery_ledger.record_components

    def recording(obligation_id, components):
        order.append("stored")
        return store(obligation_id, components)
    monkeypatch.setattr(delivery_ledger, "record_components", recording)
    send_with_retry = h.adapter._send_with_retry

    async def text_send(*, chat_id, content, **kwargs):
        if _TERMINAL in content:
            order.append("text")
        return await send_with_retry(chat_id=chat_id, content=content, **kwargs)
    monkeypatch.setattr(h.adapter, "_send_with_retry", text_send)
    from gateway.platforms.base import SendResult
    in_flight = []

    async def document_send(self, *, chat_id, file_path, **_kwargs):
        sent.append(file_path)
        if self is h.adapter:  # the first process dies while this send is on its way
            order.append("document")
            in_flight.append(True)
            await asyncio.Event().wait()
        return SendResult(success=True, message_id="doc")
    monkeypatch.setattr(RestartTestAdapter, "send_document", document_send)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: bool(in_flight), "the PDF send")
        _crash_now(h, monkeypatch)  # killed: nothing more this process writes lands
    asyncio.run(exercise())
    [record_file] = _records(ingress)
    rid = _recovery_id(record_file)
    assert order[:3] == ["stored", "text", "document"]
    assert [c for c, t in h.owner_replies() if _TERMINAL in t] == ["chat"] and _marker(h) is not None
    assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "pending")]
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left, _texts_sent(nxt)) == ([], [], [], [])
    assert sent == [str(document), str(document)]  # the interrupted send, then the recovery's alone
    assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "delivered")]
    assert _marker(nxt) is None


def _media_policy(monkeypatch, *, strict, trust_recent=False, allow=None):
    """The media delivery policy every process reads from here on (gateway/media_policy.py): strict or not,
    recent-file trust, and an operator allowlist."""
    monkeypatch.setenv("HERMES_MEDIA_DELIVERY_STRICT", "1" if strict else "0")
    monkeypatch.setenv("HERMES_MEDIA_TRUST_RECENT_FILES", "1" if trust_recent else "0")
    if allow is None:
        monkeypatch.delenv("HERMES_MEDIA_ALLOW_DIRS", raising=False)
    else:
        monkeypatch.setenv("HERMES_MEDIA_ALLOW_DIRS", str(allow))


def _owner_turn_in_lane(ingress, send, monkeypatch, lane, attachment):
    """The owner turn answers with its text and *attachment*, delivered by the adapter's final (``normal``)
    or as the queued lane's first response while a follow-up waits (``queued``). Returns the harness and the
    recovery id of the owner message's record."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    hold = (threading.Event(), threading.Event()) if lane == "queued" else None
    _answering_with(monkeypatch, attachment, hold=hold)

    async def exercise():
        await h.warm()
        await h.adapter.handle_message(h.owner_event())
        if hold is not None:
            await h.wait(hold[0].is_set, "the first turn")
            await h.adapter.handle_message(h.owner_event("owner: followup", message_id="m2"))
            await h.wait(lambda: h.queued() == ["owner: followup"], "the follow-up to queue")
            hold[1].set()
        await h.settle()
    asyncio.run(exercise())
    assert _record_texts(ingress) == [_OWNER_TEXT]  # the follow-up's record went with its own answer
    [record_file] = _records(ingress)
    return h, _recovery_id(record_file)


@pytest.mark.parametrize("lane", ["normal", "queued"])
def test_a_live_reply_whose_attachment_is_gone_keeps_it_owed_as_missing(
        ingress, send, monkeypatch, tmp_path, lane):
    """s1d review P1-1: the answer is text and a PDF that is not on disk when the reply goes out, in the
    adapter's final or in the queued first response. The PDF is stored as one of the reply's components
    before the first send, like any attachment it declares, then recorded missing: never sent, never
    counted delivered. The text is delivered, the record stays, and the chat is told once. Once the file
    is there, the next start sends the PDF alone and cleans up. The message is never run again."""
    document = tmp_path / "gone.pdf"
    sent = _attachment_sends(monkeypatch, set())
    h, rid = _owner_turn_in_lane(ingress, send, monkeypatch, lane, f"MEDIA:{document}")
    assert sent == []
    assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "missing")]
    assert len([t for _c, t in h.owner_replies() if "gone.pdf" in t]) == 1
    if lane == "normal":
        assert _marker(h) is not None
    document.write_bytes(b"%PDF-test")
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left, _texts_sent(nxt)) == ([], [], [], [])
    assert sent == [("document", str(document), True)]
    assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "delivered")]


@pytest.mark.parametrize("case", ["normal_refused", "queued_refused", "normal_restored", "queued_restored",
                                  "stored_restored"])
def test_an_attachment_the_current_media_policy_refuses_is_kept_blocked_and_never_uploaded(
        ingress, send, monkeypatch, tmp_path, caplog, case):
    """s1d review P1-3. Strict media delivery with recent-file trust off, so a PDF outside the allowlist is
    refused. It is refused when the reply goes out (``*_refused``, in the adapter's final or the queued
    first response), or it is restored outside the allowlist after the reply recorded it missing
    (``*_restored``; ``stored_restored`` is the reviewer's reproduction, a stored answer whose PDF was gone
    at the first start). It is never uploaded and never dropped: it stays one of the reply's components,
    recorded blocked, the record stays, and an ERROR says why. The message is never run again."""
    import logging

    from gateway.platforms.base import validate_media_delivery_path

    _media_policy(monkeypatch, strict=True)
    document = tmp_path / "outside-allowlist.pdf"
    lane, when = case.split("_")
    if when == "refused":
        document.write_bytes(b"%PDF-test: outside the media allowlist")
    sent = _attachment_sends(monkeypatch, set())
    if lane == "stored":
        rid = "policy"
        _stored_turn(ingress, rid, f"{_TERMINAL}\nMEDIA:{document}")
        _next_process(ingress, send, monkeypatch, crashed=True)
    else:
        _h, rid = _owner_turn_in_lane(ingress, send, monkeypatch, lane, f"MEDIA:{document}")
    first = "blocked" if when == "refused" else "missing"
    assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", first)]
    if when == "restored":
        document.write_bytes(b"%PDF-test: outside the media allowlist")
    assert validate_media_delivery_path(str(document)) is None  # the real policy refuses it
    with caplog.at_level(logging.ERROR):
        nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats, left, sent) == ([], [], [f"inbound-{rid}.json"], [])
    assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "blocked")]
    assert any("refused by the media delivery policy" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("change", ["allowed", "file_changed", "policy_changed", "symlink_swapped",
                                    "symlink_to_allowed"])
def test_an_owed_attachment_is_checked_again_right_before_each_send(
        ingress, send, monkeypatch, tmp_path, change):
    """The attachment check runs before each send, not only when the reply's components were stored. A PDF
    the policy allows is stored with the reply, and the transport refuses its first upload, so the next
    start still owes it. In between: nothing changes (``allowed``, a normal attachment: the next start
    uploads it and cleans up); the file under its path is replaced by an older one, which the strict
    policy's recent-file trust no longer covers (``file_changed``); the policy turns strict
    (``policy_changed``); or the path is swapped for a symlink to a file outside the allowlist
    (``symlink_swapped``): the next start never uploads it, it is recorded blocked and the record stays.
    Or the path is swapped for a symlink to another file inside the allowlist (``symlink_to_allowed``): what
    is uploaded is the file the check resolved, never the link. The message is never run again."""
    import os
    import time

    allowed, outside = tmp_path / "allowed", tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    document = allowed / "report.pdf"
    document.write_bytes(b"%PDF-test")
    _media_policy(monkeypatch, **{
        "allowed": {"strict": False}, "file_changed": {"strict": True, "trust_recent": True},
        "policy_changed": {"strict": False}, "symlink_swapped": {"strict": True, "allow": allowed},
        "symlink_to_allowed": {"strict": True, "allow": allowed}}[change])
    failing = {"document"}
    sent = _attachment_sends(monkeypatch, failing)
    _h, rid = _owner_turn_in_lane(ingress, send, monkeypatch, "normal", f"MEDIA:{document}")
    assert sent == [("document", str(document), False)]
    assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "failed")]
    failing.clear()
    if change == "file_changed":
        document.unlink()
        document.write_bytes(b"%PDF-test: an older file")
        os.utime(document, (time.time() - 3600, time.time() - 3600))
    elif change == "policy_changed":
        _media_policy(monkeypatch, strict=True)
    elif change.startswith("symlink"):
        target = (outside if change == "symlink_swapped" else allowed) / "target.pdf"
        target.write_bytes(b"%PDF-test: the link's target")
        document.unlink()
        document.symlink_to(target)
    nxt = _next_process(ingress, send, monkeypatch, crashed=True)
    turns, _inputs, chats, left = _answered(ingress, nxt)
    assert (turns, chats) == ([], [])
    if change in ("allowed", "symlink_to_allowed"):
        uploaded = document if change == "allowed" else allowed / "target.pdf"
        assert left == [] and sent[1:] == [("document", str(uploaded), True)] and _texts_sent(nxt) == []
        assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "delivered")]
    else:
        assert left == [f"inbound-{rid}.json"] and sent[1:] == []
        assert _components(ingress, rid) == [("text", "text", "delivered"), ("media:0", "media", "blocked")]
