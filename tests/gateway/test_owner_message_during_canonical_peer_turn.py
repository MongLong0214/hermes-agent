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
    monkeypatch.setattr(h.runner, "_settle_inbound_record", lambda *args, **kwargs: None, raising=False)


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
            recovery_id = (kw.get("persist_user_display_metadata") or {}).get("inbound_recovery_id")
            if recovery_id:  # the agent writes the turn's display metadata on its user row
                user["display_metadata"] = {"inbound_recovery_id": recovery_id}
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
    """The reply-sending boundary: the turn has ended (its record consumed) and its final reply is in
    the delivery ledger when stop(restart=True) runs mid-send. The next process redelivers that reply
    and runs no second turn."""
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
        assert _records(ingress) == []
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
            assert _record_texts(ingress) == [follow_up]  # queueing never deletes the record
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
    goes once its answer is handed to the ledger. A restart while the follow-up runs leaves only the
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
