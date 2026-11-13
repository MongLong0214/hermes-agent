"""An owner message that arrives while a canonical peer turn runs is answered to the owner.

Production 2026-11-13 22:27 and 22:32 KST: the CEO session was running a canonical peer turn (a CTO's
Buzz event through ``/v1/canonical-surface/events``) when the owner wrote on Telegram. The busy path
(``busy_input_mode: interrupt``) redirected the text into the running turn; the redirect landed during
a tool call, so it degraded to a steer. The model answered the owner inside the peer turn, that answer
became the canonical answer posted to the CTO's thread, and nothing was sent to Telegram.

A peer turn is the turn the canonical surface runs under its turn lease (owner key ``canonical:``).
While it holds the session, an owner follow-up is never steered, redirected or interrupted into it: it
waits as its own owner turn and runs once the peer turn ends, and its reply goes to the owner's chat.
Owner-to-owner busy behaviour is unchanged.
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
    ingress,
    send,
)

_OWNER_TEXT = "owner: are you there?"
_MODES = ("steer", "interrupt", "queue")


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
        self.built = []

        def build(**kwargs):
            actor = _Actor(kwargs["session_id"], kwargs["session_db"])
            self.built.append(actor)
            return actor

        monkeypatch.setattr(run_agent, "AIAgent", build)
        monkeypatch.setattr(runner, "_resolve_session_agent_runtime", lambda **kw: (
            "test-model", {"provider": "openrouter", "api_key": "test-key", "base_url": "http://127.0.0.1:9"}))

        async def admitted(event):
            return event, event.source, False
        monkeypatch.setattr(runner, "_hm_admit_event", admitted)
        monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source: True)
        runner._busy_input_mode = mode
        runner._busy_text_mode = "interrupt"
        self.adapter = adapter = RestartTestAdapter()
        adapter.gateway_runner = runner
        adapter.set_message_handler(runner._handle_message)
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

    async def peer_turn_running(self):
        """Start a canonical peer turn and hold it mid-run."""
        self.actor.hold = (self.entered, self.release)
        peer = asyncio.ensure_future(self.send(event_id="peer", text="cto: status?"))
        await self.wait(self.entered.is_set, "the peer turn to start")
        assert self.runner._is_session_running(self.ingress.entry.session_key)
        return peer

    def owner_replies(self):
        return [(chat, text) for chat, text, _meta in self.adapter.sent_calls]

    async def settle(self):
        for _ in range(500):
            tasks = [t for t in self.adapter._session_tasks.values() if not t.done()]
            if not tasks and not self.runner._is_session_running(self.ingress.entry.session_key):
                return
            await asyncio.sleep(0.01)
        pytest.fail("the owner turn never finished")


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
        assert (h.actor.steered, h.actor.redirected, h.actor.interrupted) == ([], [], [])
        assert h.handled == [_OWNER_TEXT]  # handled once, not re-dispatched while it waits
        assert h.owner_replies() == []  # nothing answered yet: it runs after the peer turn
        h.release.set()
        status, body = await peer
        await h.settle()
        return status, body

    status, body = asyncio.run(exercise())
    # The peer's canonical answer is the peer turn's own; nothing of the owner's text is in it.
    assert (status, body) == (200, {"event_id": "peer", "text": _TERMINAL})
    # The owner turn ran as its own turn after the peer turn, and its reply went to the owner's chat.
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT]
    assert h.owner_replies() == [(ingress.source.chat_id, _TERMINAL)]


@pytest.mark.parametrize("mode", _MODES)
def test_a_second_owner_message_during_the_peer_turn_queues_through_the_busy_handler(
        ingress, send, monkeypatch, mode):
    """The first owner message holds the adapter's session while it waits, so the next one reaches
    the busy handler (gateway.run_busy): it is queued with the queue ack in every mode and runs as
    its own owner turn after the first."""
    h = _Harness(ingress, send, monkeypatch, mode)

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.handled, "the first owner message to be handled")
        await h.adapter.handle_message(h.owner_event("owner: second", message_id="m2"))
        await asyncio.sleep(0.3)
        assert (h.actor.steered, h.actor.redirected, h.actor.interrupted) == ([], [], [])
        [(chat, ack)] = h.owner_replies()
        assert chat == ingress.source.chat_id and ack.startswith("⏳ Queued for the next turn")
        h.release.set()
        result = await peer
        await h.settle()
        return result

    assert asyncio.run(exercise()) == (200, {"event_id": "peer", "text": _TERMINAL})
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT, "owner: second"]
    assert h.owner_replies()[1:] == [(ingress.source.chat_id, _TERMINAL)] * 2


def test_steer_command_during_the_peer_turn_is_queued_as_its_own_turn(ingress, send, monkeypatch):
    h = _Harness(ingress, send, monkeypatch, "steer")

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event("/steer owner: look at this"))
        await asyncio.sleep(0.5)
        assert (h.actor.steered, h.actor.redirected, h.actor.interrupted) == ([], [], [])
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
        state = h.runner._peek_session_state(ingress.entry.session_key)
        await h.wait(lambda: state.turn.agent is h.actor, "the owner turn's agent to be published")
        # The owner turn holds the session's turn lease under its own routing key.
        holder = h.runner._turn_leases._leases[ingress.entry.session_id].holder
        assert holder.owner_key == ingress.entry.session_key
        if path == "busy_handler":
            await h.adapter.handle_message(follow_up)
        else:
            assert await h.runner._handle_message(follow_up) is None
        seen = (list(h.actor.steered), list(h.actor.redirected))
        queued = h.adapter._pending_messages.get(ingress.entry.session_key) is follow_up
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
    key = ingress.entry.session_key

    async def exercise():
        await h.warm()
        assert not h.runner._canonical_peer_turn_running(key)
        peer = await h.peer_turn_running()
        lease = h.runner._turn_leases._leases[ingress.entry.session_id]
        assert lease.holder.owner_key.startswith("canonical:")
        during = h.runner._canonical_peer_turn_running(key)
        h.release.set()
        await peer
        return during, h.runner._canonical_peer_turn_running(key)

    assert asyncio.run(exercise()) == (True, False)


def test_a_drain_ends_the_wait_and_the_follow_up_is_kept_for_after_the_restart(ingress, send, monkeypatch):
    """A follow-up still waiting when the gateway starts draining goes to the busy path's drain
    branch: queued for after the restart (and so in the shutdown flush), never steered."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    key = ingress.entry.session_key

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.handled, "the owner message to be handled")
        h.runner._restart_requested = True
        h.runner._draining = True
        await h.wait(h.owner_replies, "the drain notice")
        kept = h.adapter._pending_messages.get(key)
        h.release.set()
        await peer
        return kept

    kept = asyncio.run(exercise())
    assert kept is not None and kept.text == _OWNER_TEXT
    assert (h.actor.steered, h.actor.redirected, h.actor.interrupted) == ([], [], [])
    assert h.owner_replies() == [
        (ingress.source.chat_id, "⏳ Gateway restarting — queued for the next turn after it comes back.")]


@pytest.mark.parametrize("mode", ["steer", "interrupt"])
def test_witness_without_the_peer_turn_check_the_owner_is_answered_in_the_peer_thread(
        ingress, send, monkeypatch, mode):
    """Mutation witness: with the canonical check disabled the harness reproduces production — the
    owner's text is answered inside the peer turn's canonical answer and nothing reaches Telegram."""
    h = _Harness(ingress, send, monkeypatch, mode)
    monkeypatch.setattr(h.runner, "_canonical_peer_turn_running", lambda session_key: False)

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.actor.steered, "the owner text to reach the running turn")
        h.release.set()
        result = await peer
        await h.settle()
        return result

    status, body = asyncio.run(exercise())
    assert status == 200 and _OWNER_TEXT in body["text"]
    assert h.owner_replies() == []
