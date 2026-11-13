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


@pytest.mark.parametrize("when", ["peer_turn_ended", "peer_turn_running"])
def test_a_restart_keeps_the_owner_message_queued_behind_a_peer_turn(ingress, send, monkeypatch, when):
    """Review r1 P1-a: a restart after the owner message was queued never loses it. Stopped right after
    the peer turn handed the session on (the owner turn still preparing) or while the peer turn still
    runs, the message stays in the FIFO and the shutdown flush writes it out for the next process."""
    import gateway.shutdown_flush as shutdown_flush

    flushed = []
    monkeypatch.setattr(shutdown_flush, "flush_pending_to_file", lambda pending, **kw: flushed.extend(
        getattr(v, "text", v) for v in pending.values()) or len(pending))
    monkeypatch.setattr(shutdown_flush, "flush_overflow_to_file", lambda overflow, **kw: flushed.extend(
        getattr(v, "text", v) for events in overflow.values() for v in events) or 0)
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
