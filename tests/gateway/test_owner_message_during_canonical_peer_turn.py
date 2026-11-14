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

import json
import logging
import secrets
from pathlib import Path
from types import SimpleNamespace

from gateway.canonical_surface import CanonicalSurfaceBinding
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _api_request_profile
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from tests.gateway.restart_test_helpers import RestartTestAdapter

_ROUTE = "/v1/canonical-surface/events"
_TERMINAL = "request-owned terminal"


class _CachedActor:
    """Stands in for the bound cached AIAgent: deterministic, no model call (records each turn's text)."""

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
    """Real runner, resolver, coordinator and SQLite under a temp home; no model."""
    home, user_home = tmp_path / "hermes-home", tmp_path / "user"
    home.mkdir()
    user_home.mkdir()
    monkeypatch.setenv("HOME", str(user_home))
    monkeypatch.setattr(Path, "home", lambda: user_home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_state.DEFAULT_DB_PATH", home / "state.db")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    override = set_hermes_home_override(home)
    runner = GatewayRunner(GatewayConfig(sessions_dir=home / "sessions"))
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm", user_id="user")
    entry = runner.session_store.get_or_create_session(source)
    binding = CanonicalSurfaceBinding(
        name="canonical", session_key=entry.session_key, session_id=entry.session_id,
        telegram_chat_id="chat", telegram_chat_type="dm", telegram_user_id="user",
        telegram_thread_id=None, allowed_author_ids=("author",), allowed_channel_ids=("channel",),
    )
    runner.config.canonical_surface_bindings = {binding.name: binding}
    runners = [runner]
    try:
        yield SimpleNamespace(runner=runner, key=secrets.token_hex(32), entry=entry, source=source,
                              binding=binding, home=home, runners=runners)
    finally:
        for each in runners:
            each.session_store.close_all_db_handles()
        reset_hermes_home_override(override)


@pytest.fixture
def send(ingress):
    """POST one event to the registered route handler, as the authenticated caller."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": ingress.key}))
    adapter.gateway_runner = ingress.runner
    [handler] = [h for method, path, h in adapter._http_route_table() if (method, path) == ("POST", _ROUTE)]

    async def post(**changes):
        raw = json.dumps({"binding": "canonical", "event_id": "event", "author_id": "author",
                          "channel_id": "channel", "text": "hello", **changes}).encode()

        async def read():
            return raw
        request = SimpleNamespace(headers={"Authorization": f"Bearer {ingress.key}"}, read=read,
                                  method="POST", path_qs=_ROUTE, transport=None)
        token = _api_request_profile.set(None)
        try:
            response = await handler(request)
        finally:
            _api_request_profile.reset(token)
        return response.status, json.loads(response.text)

    post.adapter = adapter
    try:
        yield post
    finally:
        adapter._response_store.close()


def _code(result):
    return result[0], result[1]["error"]["code"]


def _clean_restart(ingress, send):
    """A clean restart: a new runner over the same home whose agent cache is empty, so the first
    canonical event builds the session's agent."""
    ingress.runner.session_store.close_all_db_handles()
    restarted = GatewayRunner(GatewayConfig(sessions_dir=ingress.home / "sessions"))
    restarted.config.canonical_surface_bindings = {ingress.binding.name: ingress.binding}
    ingress.runners.append(restarted)
    send.adapter.gateway_runner = restarted
    return restarted

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
            return event, event.source, bool(event.internal)
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

        monkeypatch.setattr(runner, "_is_user_authorized_for_source", lambda source, **kw: True)
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
        assert self.peer_lease_held()
        return peer

    def peer_lease_held(self):
        """The canonical claim's own turn lease holds the session (read off the registry directly)."""
        lease = self.runner._turn_leases._leases.get(self.ingress.entry.session_id)
        owner_key = getattr(getattr(lease, "holder", None), "owner_key", None)
        return isinstance(owner_key, str) and owner_key.startswith("canonical:")

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
        await asyncio.sleep(0.5)  # health: allow HX013 -- a window in which nothing may reach the running turn
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
        await h.wait(lambda: len(h.queued()) == 2, "both owner messages to be queued")
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
        await h.wait(lambda: h.owner_replies() and h.queued(), "the /steer reply and its queued turn")
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


def _capture_shutdown_flush(monkeypatch):
    """The texts the shutdown flush writes out for the next process (pending slots and overflow)."""
    from gateway import shutdown_flush

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
    # Queued once and handed on once at most; once kept for the restart it is never dispatched again.
    assert h.handled.count(_OWNER_TEXT) <= 2
    if when == "peer_turn_ended":  # the handed-on turn the drain refused tells the owner it is kept
        assert any("queued for the next turn after it comes back" in text for _chat, text in h.owner_replies())


def test_a_handed_on_message_whose_preparation_fails_releases_the_session(ingress, send, monkeypatch):
    """A handed-on turn that raises before it claims releases the reservation and leaves the queue (as
    a failed ordinary turn does), so the session is not held for a turn that will never claim it."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    dispatch_idle = h.runner._hm_dispatch_idle_commands

    async def failing(event, source, quick_key):
        if event.text == _OWNER_TEXT:
            raise RuntimeError("preparation failed")
        return await dispatch_idle(event, source, quick_key)
    monkeypatch.setattr(h.runner, "_hm_dispatch_idle_commands", failing)

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.queued() == [_OWNER_TEXT], "the owner message to be queued")
        h.release.set()
        assert (await peer)[0] == 200
        await h.settle()
        return h.runner._is_session_running(h.key), h.queued()

    assert asyncio.run(exercise()) == (False, [])
    assert h.actor.calls == ["hello", "cto: status?"]
    assert h.handled == [_OWNER_TEXT, _OWNER_TEXT]  # queued, handed on once; never dispatched again


def test_a_hand_off_whose_task_cannot_start_releases_the_reservation_and_keeps_the_message(
        ingress, send, monkeypatch):
    """The adapter could not start the handed-on turn: the slot is released instead of staying reserved
    for a turn that never comes, and the message stays queued at its head."""
    h = _Harness(ingress, send, monkeypatch, "queue")

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(lambda: h.queued() == [_OWNER_TEXT] and all(
            t.done() for t in h.adapter._session_tasks.values()), "the owner message to be queued")
        monkeypatch.setattr(h.adapter, "_start_session_processing", lambda event, key, **kw: False)
        h.release.set()
        assert (await peer)[0] == 200
        head = h.adapter._pending_messages.get(h.key)
        return h.runner._is_session_running(h.key), getattr(head, "_turn_reservation", None), h.queued()

    assert asyncio.run(exercise()) == (False, None, [_OWNER_TEXT])


async def _hand_on_and_take_the_head(h, gate, *, then_restart=False):
    """A peer turn hands the session to the queued head, which pauses in preparation; something takes
    it out of the adapter's queue meanwhile (``get_pending_message``), with a second message behind."""
    await h.warm()
    peer = await h.peer_turn_running()
    await h.adapter.handle_message(h.owner_event())
    await h.wait(lambda: h.queued() == [_OWNER_TEXT] and all(
        t.done() for t in h.adapter._session_tasks.values()), "the owner message to be queued")
    await h.adapter.handle_message(h.owner_event("owner: second", message_id="m2"))
    await h.wait(lambda: h.queued() == [_OWNER_TEXT, "owner: second"], "the second message to queue")
    h.release.set()
    assert (await peer)[0] == 200
    await h.wait(lambda: _OWNER_TEXT in h.preparing, "the handed-on head to be preparing")
    assert h.adapter.get_pending_message(h.key).text == _OWNER_TEXT
    stop = None
    if then_restart:
        stop = asyncio.ensure_future(h.runner.stop(restart=True))
        await h.wait(lambda: h.runner._draining, "the restart drain to begin")
    gate.set()
    return stop


def test_a_reserved_head_taken_out_of_the_queue_before_its_claim_still_runs_ahead_of_the_rest(
        ingress, send, monkeypatch):
    """The handed-on head claims as itself: the FIFO's orphan rescue never runs a later message in its
    place and parks the head behind it."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.preparation_gates[_OWNER_TEXT] = gate = asyncio.Event()

    async def exercise():
        await _hand_on_and_take_the_head(h, gate)
        await h.wait(lambda: len(h.actor.calls) == 4, "both owner turns to run")
        await h.settle()

    asyncio.run(exercise())
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT, "owner: second"]


def test_a_reserved_head_taken_out_of_the_queue_is_handed_back_when_the_drain_refuses_it(
        ingress, send, monkeypatch):
    """While the gateway drains, the refused head is put back at the queue head (ahead of the message
    behind it) when it is no longer there, so the shutdown flush still writes it out."""
    flushed = _capture_shutdown_flush(monkeypatch)
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 0.5
    h.preparation_gates[_OWNER_TEXT] = gate = asyncio.Event()

    async def exercise():
        stop = await _hand_on_and_take_the_head(h, gate, then_restart=True)
        await asyncio.wait_for(stop, 60)

    asyncio.run(exercise())
    assert h.actor.calls == ["hello", "cto: status?"]
    assert _OWNER_TEXT in flushed and "owner: second" in flushed


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
    # raising=False: the same witness holds on a tree that never had the check.
    monkeypatch.setattr(h.runner, "_canonical_peer_turn_running", lambda session_key: False, raising=False)

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


# --- Target seams (the fork has neither a C13 hand-off error nor a session authority) ---------------


def test_a_canonical_turn_with_nothing_queued_ends_without_an_error(ingress, send, monkeypatch, caplog):
    """C13 carry-over: the canonical surface hands the session on after every peer turn; with nothing
    queued that is a no-op, never an error logged after each canonical turn."""
    h = _Harness(ingress, send, monkeypatch, "interrupt")

    async def exercise():
        await h.warm()
        return await h.send(event_id="peer", text="cto: status?")

    with caplog.at_level(logging.ERROR):
        assert asyncio.run(exercise()) == (200, {"event_id": "peer", "text": _TERMINAL})
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []
    assert h.actor.calls == ["hello", "cto: status?"]


def test_a_wake_admitted_before_the_drain_still_runs_when_the_peer_turn_hands_it_the_session(
        ingress, send, monkeypatch):
    """The target's drain rule (``_admitted_before_drain``): a wake queued behind a peer turn before a
    restart began is handed the session when the peer turn ends and runs, as any such wake does; its
    producer holds a receipt and never resends it. An owner message is refused there instead and kept
    for the restart (test_a_restart_keeps_the_owner_message_queued_behind_a_peer_turn)."""
    h = _Harness(ingress, send, monkeypatch, "queue")
    h.runner._restart_drain_timeout = 5.0
    wake = MessageEvent(text="[wake] background job done", message_type=MessageType.TEXT,
                        source=ingress.source, internal=True)
    wake._admitted_before_drain = True

    async def exercise():
        await h.warm()
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(wake)
        await h.wait(lambda: h.queued() == [wake.text], "the wake to queue behind the peer turn")
        stop = asyncio.ensure_future(h.runner.stop(restart=True))
        await h.wait(lambda: h.runner._draining, "the restart drain to begin")
        h.release.set()
        await peer
        await h.wait(lambda: wake.text in h.actor.calls, "the wake to run")
        await asyncio.wait_for(stop, 60)

    asyncio.run(exercise())
    assert h.actor.calls[:2] == ["hello", "cto: status?"] and wake.text in h.actor.calls[2:]
    # An internal event queues silently behind the peer turn: no "Queued" ack reaches the chat.
    assert not [text for _chat, text in h.owner_replies() if text.startswith(_QUEUED_ACK)]


async def _with_session_authority(h):
    """The shared runtime's session authority, as every started gateway has: a non-command owner
    message is committed to its durable FIFO instead of the in-process one."""
    from gateway.session_authority import initialize_session_authority

    await initialize_session_authority(h.runner, profile_id="default", instance_id="owner")
    assert h.runner.session_authority is not None


@pytest.mark.parametrize("mode", _MODES)
def test_with_the_session_authority_owner_text_during_a_peer_turn_is_answered_after_it(
        ingress, send, monkeypatch, mode):
    """The production path: the authority's busy preempt resolves the follow-up through the same busy
    resolution, so it is never steered, redirected or interrupting into the peer turn; it waits in the
    durable FIFO behind the peer turn and runs as the owner's own turn when that ends."""
    h = _Harness(ingress, send, monkeypatch, mode)

    async def exercise():
        await h.warm()
        await _with_session_authority(h)
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await asyncio.sleep(0.5)  # health: allow HX013 -- a window in which nothing may reach the running turn
        _no_busy_verbs(h)
        h.release.set()
        result = await peer
        await h.wait(lambda: len(h.actor.calls) == 3, "the owner turn to run")
        await h.settle()
        return result

    assert asyncio.run(exercise()) == (200, {"event_id": "peer", "text": _TERMINAL})
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT]
    assert h.owner_replies()[-1] == (ingress.source.chat_id, _TERMINAL)


def test_with_the_session_authority_a_peer_event_waits_for_the_owner_input_committed_before_it(
        ingress, send, monkeypatch):
    """The durable FIFO's hand-off: from the owner input's commit until its turn has claimed, a peer
    event is refused canonical_turn_busy (the bridge retries it). Admitted instead (the FIFO's drain
    claims after the peer turn, then awaits before the owner turn's own claim), the peer turn would be
    overwritten by that claim or have the owner's text steered into it."""
    h = _Harness(ingress, send, monkeypatch, "interrupt")
    h.preparation_gates[_OWNER_TEXT] = gate = asyncio.Event()

    async def exercise():
        await h.warm()
        await _with_session_authority(h)
        peer = await h.peer_turn_running()
        await h.adapter.handle_message(h.owner_event())
        await h.wait(h.owner_replies, "the owner's busy acknowledgement")
        h.release.set()
        first = await peer
        await h.wait(lambda: _OWNER_TEXT in h.preparing, "the owner turn to be preparing")
        second = await h.send(event_id="peer-b", text="cto: next")
        gate.set()
        await h.wait(lambda: len(h.actor.calls) == 3, "the owner turn to run")
        await h.settle()
        return first, second

    first, second = asyncio.run(exercise())
    assert first == (200, {"event_id": "peer", "text": _TERMINAL})
    assert _code(second) == (409, "canonical_turn_busy")
    _no_busy_verbs(h)
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT]
    assert h.owner_replies()[-1] == (ingress.source.chat_id, _TERMINAL)


def test_with_the_session_authority_a_follow_up_the_in_process_fifo_holds_runs_when_handed_on(
        ingress, send, monkeypatch):
    """A non-command message in the in-process FIFO behind a peer turn while an authority exists (an
    executing admission that met a peer turn; the surface's deferral above keeps that from happening)
    runs as the reserved turn itself when handed the session. Re-admitted to the durable FIFO, that
    FIFO's drain would wait on the very reservation it holds, and the turn would never run."""
    h = _Harness(ingress, send, monkeypatch, "interrupt")

    async def exercise():
        await h.warm()
        await _with_session_authority(h)
        peer = await h.peer_turn_running()
        h.runner._queue_behind_peer_turn(h.owner_event(), h.key)
        assert h.queued() == [_OWNER_TEXT]
        h.release.set()
        result = await peer
        await h.wait(lambda: len(h.actor.calls) == 3, "the handed-on follow-up to run")
        await h.settle()
        return result

    assert asyncio.run(exercise()) == (200, {"event_id": "peer", "text": _TERMINAL})
    assert h.actor.calls == ["hello", "cto: status?", _OWNER_TEXT] and h.queued() == []


# --- The adapter's drain sites (gateway.platforms.base) -------------------------------------------


class _HoldingRunner:
    """The two runner seams the adapter's drain sites consult, and nothing else."""

    def __init__(self, *, defers=False, reserved=None):
        self.defers, self.reserved = defers, reserved

    def _defers_pending_drain(self, session_key):
        return self.defers

    def _turn_reservation_of(self, event):
        return ("key", None) if event is self.reserved else None


def _drain_site_adapter(runner):
    adapter = RestartTestAdapter()
    adapter.gateway_runner = runner
    started, spawned = [], []
    adapter._start_session_processing = lambda event, key, **kw: started.append(event.text) or True
    adapter._spawn_drain_task = lambda event, key, delay=0.0: spawned.append(event.text)
    return adapter, started, spawned


def _queued_event(text="queued"):
    return MessageEvent(text=text, message_type=MessageType.TEXT,
                        source=SessionSource(platform=Platform.TELEGRAM, chat_id="chat", chat_type="dm",
                                             user_id="user"))


@pytest.mark.parametrize("defers", [True, False], ids=["held", "ordinary"])
def test_every_drain_site_leaves_a_follow_up_the_runner_holds_queued(defers):
    """While the runner holds a session's queue (a canonical peer turn runs, or the head is kept for the
    restart), no drain site hands its head to a task: the post-command drain, a finished task's late
    arrival, a backed-off drain waking and the busy path's orphan recovery. Otherwise each does."""
    async def exercise():
        results = {}
        for site in ("command", "finish", "backoff", "orphan"):
            adapter, started, spawned = _drain_site_adapter(_HoldingRunner(defers=defers))
            event = _queued_event()
            adapter._pending_messages["key"] = event
            guard = asyncio.Event()
            adapter._active_sessions["key"] = guard
            processed = []

            async def process(ev, key):
                processed.append(ev.text)
            adapter._process_message_background = process
            if site == "command":
                await adapter._drain_pending_after_session_command("key", guard)
            elif site == "finish":
                adapter._session_tasks["key"] = asyncio.current_task()
                adapter._finish_session_task("key", guard)
            elif site == "backoff":
                await adapter._drain_after(event, "key", 0.01, guard)
            else:
                async def busy(ev, key):  # the guard's owner ended during the handler's awaits
                    adapter._active_sessions.pop("key", None)
                    return True
                adapter.set_busy_session_handler(busy)
                await adapter._handle_message_while_active(_queued_event("another"), "key")
            results[site] = (adapter._pending_messages.get("key") is event, started + spawned + processed)
        return results

    results = asyncio.run(exercise())
    if defers:
        assert results == {site: (True, []) for site in ("command", "finish", "backoff", "orphan")}
    else:
        assert results == {site: (False, ["queued"]) for site in ("command", "finish", "backoff", "orphan")}


def test_a_reserved_head_is_dispatched_without_leaving_the_queue():
    """``_pop_pending_for_drain``: the head a canonical peer turn reserved the session for is handed to
    its task and stays queued (until its turn claims, on the runner); any other head is taken out."""
    reserved, other = _queued_event("reserved"), _queued_event("other")
    adapter, _started, _spawned = _drain_site_adapter(_HoldingRunner(reserved=reserved))
    adapter._pending_messages["key"] = reserved
    assert adapter._pop_pending_for_drain("key") is reserved and adapter._pending_messages["key"] is reserved
    adapter._pending_messages["key"] = other
    assert adapter._pop_pending_for_drain("key") is other and "key" not in adapter._pending_messages
