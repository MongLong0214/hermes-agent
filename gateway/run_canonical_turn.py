"""Owned-turn runner for canonical receipts: publish, run, cancel, clear.

A canonical turn runs the bound cached actor on the gateway turn pool while the actor advertises
which processes the turn owns, so /stop, /new and eviction reap only what this turn spawned.
Cancelling the awaiting coroutine must never leave that advertisement behind for a turn that did
not run, nor release the turn while its worker is still inside the actor.

:class:`GatewayCanonicalTurnMixin` is GatewayRunner's side of a canonical receipt (MRO mixin): it
runs one exact cached actor as a peer and selects that turn's answer, and the receipt coordinator in
:mod:`gateway.canonical_surface` decides what is recorded and returned. :func:`peer_replay_row` keeps
a stored peer row a peer row whenever a gateway turn replays it. ``gateway.run`` internals are
imported lazily inside function bodies (import cycle).
"""

from __future__ import annotations

import asyncio
import functools
import threading
from contextlib import suppress
from contextvars import copy_context
from typing import Any, Callable

from agent.interrupt_compat import request_hard_interrupt

_INTERRUPT_REASON_CANCELLED = "Canonical request cancelled"
_INTERRUPT_TOOL_REASON_CANCELLED = "canonical request cancelled"
_INVALIDATION_REASON_CANCELLED = "canonical_request_cancelled"


async def run_owned_turn(
    runner: Any, agent: Any, work: Callable[[], Any], *,
    session_key: str, task_id: str, run_generation: int | None,
) -> Any:
    """Run ``work`` on ``agent`` in the turn pool, owning the processes it spawns until it exits."""
    from tools.process_registry import process_registry

    # Published on the loop so /stop, /new and eviction reap only processes this turn spawns, even
    # when the interrupt lands before the worker thread starts.
    baseline = process_registry.snapshot_running_ids(task_id)
    agent._gateway_turn_process_task_id = task_id
    agent._gateway_turn_process_baseline = baseline

    def clear_ownership() -> None:
        # The identity check keeps a later turn's publication on this actor intact.
        if agent._gateway_turn_process_baseline is baseline:
            agent._gateway_turn_process_task_id = ""
            agent._gateway_turn_process_baseline = frozenset()

    def owned_work() -> Any:
        try:
            return work()
        finally:
            clear_ownership()

    try:
        future = runner._get_executor().submit(copy_context().run, owned_work)
    except BaseException:
        clear_ownership()
        raise
    worker = asyncio.wrap_future(future)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        # cancel() succeeds only while the pool has not handed the item to a thread, and a
        # cancelled item never runs: this is the race-free "never started" answer.
        if future.cancel():
            clear_ownership()
            raise
        if not future.done():
            release_generation = _interrupt_owned_turn(
                runner, agent, session_key=session_key, task_id=task_id,
                baseline=baseline, run_generation=run_generation,
            )
            await _join_worker(worker)
            if release_generation is not None:
                runner._release_running_agent_state(session_key, run_generation=release_generation)
        raise


def _interrupt_owned_turn(
    runner: Any, agent: Any, *, session_key: str, task_id: str, baseline: frozenset,
    run_generation: int | None,
) -> int | None:
    """Interrupt this turn as /stop would; return the generation whose slot it must release."""
    if run_generation is None:
        # A standalone turn claims no running slot for /stop to find, so it is interrupted and
        # reaped here with the same two steps _interrupt_running_turn composes.
        from gateway.run import _reap_gateway_turn_processes

        request_hard_interrupt(
            agent, _INTERRUPT_REASON_CANCELLED, tool_reason=_INTERRUPT_TOOL_REASON_CANCELLED,
        )
        threading.Thread(
            target=copy_context().run,
            args=(_reap_gateway_turn_processes, task_id, baseline),
            kwargs={"source": "gateway_turn_interrupt"},
            name=f"gateway-turn-reaper-{task_id[:12]}",
            daemon=True,
        ).start()
        return None
    state = runner._peek_session_state(session_key)
    if (
        state is None
        or state.turn.agent is not agent
        or not runner._is_session_run_current(session_key, run_generation)
    ):
        # /stop, /new, eviction or shutdown already took the slot; they interrupted this actor
        # and, where they reap, reaped with this turn's baseline.
        return None
    return runner._interrupt_running_turn(
        session_key,
        interrupt_reason=_INTERRUPT_REASON_CANCELLED,
        invalidation_reason=_INVALIDATION_REASON_CANCELLED,
        tool_reason=_INTERRUPT_TOOL_REASON_CANCELLED,
    )


async def _join_worker(worker: asyncio.Future) -> None:
    """Wait until the worker thread has left the actor, however often the waiter is cancelled."""
    while not worker.done():
        with suppress(asyncio.CancelledError):
            await asyncio.wait((worker,))
    if not worker.cancelled():
        # The cancelled caller never reads the abandoned result; retrieve it so asyncio does not
        # report an unobserved exception.
        worker.exception()


def peer_replay_row(msg: Any) -> dict[str, Any] | None:
    """The replay row of a stored canonical peer row; None for every other row.

    A canonical peer row replays as stored: clean body, its provenance, and the model rendering of
    that provenance. No timestamp, cleanup or mirror rewrite touches it, so it can never come back
    as plain owner text; inconsistent provenance raises.

    ``peer_wire_text(msg)`` reads ``msg`` as the loader produced it — already carrying
    PEER_SIDECAR_VERIFIED_KEY when its stored sidecar's digest matched the admission ledger — and
    returns either that authenticated sidecar (Hermes' own appended memory/plugin/surface-switch
    context survives) or a fresh rendering otherwise. The rebuilt row is itself first-party
    (``content``/``api_content`` came straight out of that call, never from an untrusted source), so
    the same trust carries forward for whichever later pass re-derives this row's wire text from it.
    """
    from agent.canonical_peer import PEER_SIDECAR_VERIFIED_KEY, peer_metadata, peer_wire_text

    if peer_metadata(msg) is None:
        return None
    return {
        "role": "user", "content": msg.get("content"), "api_content": peer_wire_text(msg),
        "display_kind": msg["display_kind"], "display_metadata": msg["display_metadata"],
        PEER_SIDECAR_VERIFIED_KEY: True,
        **({"timestamp": msg["timestamp"]} if msg.get("timestamp") is not None else {}),
    }


def _bound_route(binding: Any, entry: Any) -> tuple[str, Any]:
    """``entry``'s session id and origin while it is still ``binding``'s route entry."""
    session_id = getattr(entry, "session_id", None)
    origin = getattr(entry, "origin", None)
    if (
        getattr(entry, "session_key", None) != getattr(binding, "session_key", None)
        or not isinstance(session_id, str)
        or not session_id
        or origin is None
        or getattr(origin, "chat_id", None) != getattr(binding, "telegram_chat_id", None)
        or getattr(origin, "chat_type", None) != getattr(binding, "telegram_chat_type", None)
        or getattr(origin, "user_id", None) != getattr(binding, "telegram_user_id", None)
        or getattr(origin, "thread_id", None) != getattr(binding, "telegram_thread_id", None)
    ):
        raise ValueError("canonical_binding_stale")
    return session_id, origin


def _require_held_lease(registry: Any, lease: Any, session_id: str, owner_key: str) -> None:
    """``lease`` is this event's live turn lease on ``session_id``, held by it right now."""
    lease_state = getattr(lease, "lease", None)
    lock = getattr(lease_state, "lock", None)
    if (
        getattr(lease, "session_id", None) != session_id
        or getattr(lease, "released", True)
        or getattr(lease, "owner_key", None) != owner_key
        or getattr(lease, "generation", None) != 0
        or getattr(lease_state, "holder", None) is not lease
        or not callable(getattr(lock, "locked", None))
        or not lock.locked()
        or getattr(registry, "_leases", {}).get(session_id) is not lease_state
    ):
        raise ValueError("canonical_turn_busy")


def _current_cron_session() -> Any:
    """The cron session var exactly as bound in this context (tri-state, unset included), so a turn
    that rebinds the session vars carries the live value over instead of resetting it."""
    from gateway.session_context import _CRON_SESSION

    return _CRON_SESSION.get()


def _bound_refused_answer(result: Any, *, peer: Any, body: str, rendered: str, history_boundary: Any,
                          expected_session_id: str) -> Any:
    """The refused turn's final answer when it is proven this event's, claim's and turn's, else None."""
    from gateway.canonical_surface import bound_refused_answer

    try:
        return bound_refused_answer(result, peer=peer, body=body, rendered=rendered,
                                    history_boundary=history_boundary, expected_session_id=expected_session_id)
    except Exception:  # health: allow BLE001 -- an answer that cannot be proven is not kept; the refusal stands
        return None


def _turn_binds(result: Any, *, history_boundary: int, peer: Any, body: str, rendered: str) -> bool:
    """Whether the completed turn opened with this claim's own row and carries no other peer row."""
    from gateway.canonical_surface import turn_provenance

    try:
        facts = turn_provenance(result.get("messages"), history_boundary=history_boundary, peer=peer, body=body,
                                rendered=rendered)
    except Exception:  # health: allow BLE001 -- provenance that cannot be read never binds the turn
        return False
    return facts is not None and facts.binds


class _BoundTurn:
    """One canonical turn's fixed expectations: the binding, event, route entry and session it was
    admitted on, and the actor, DB file and run generation the receipt coordinator proved."""

    def __init__(self, runner: Any, binding: Any, event: Any, entry: Any, session_id: str, origin: Any, *,
                 expected_actor: Any, expected_session_db: Any, expected_db_path: Any,
                 expected_db_identity: Any, expected_run_generation: int | None) -> None:
        self.runner, self.binding, self.event, self.entry = runner, binding, event, entry
        self.session_id, self.origin = session_id, origin
        self.expected_actor, self.expected_session_db = expected_actor, expected_session_db
        self.expected_db_path, self.expected_db_identity = expected_db_path, expected_db_identity
        self.expected_run_generation = expected_run_generation

    def require_current_head(self) -> None:
        """The run generation is still current and the route still names this very entry and session."""
        runner, session_key = self.runner, self.binding.session_key
        if (
            self.expected_run_generation is not None
            and not runner._is_session_run_current(session_key, self.expected_run_generation)
        ):
            raise ValueError("canonical_turn_interrupted")
        current_entry = runner.session_store.lookup_by_session_key_existing(session_key)
        if (
            current_entry is not self.entry
            or getattr(self.entry, "session_id", None) != self.session_id
            or getattr(current_entry, "session_id", None) != self.session_id
        ):
            raise ValueError("canonical_binding_stale")

    def require_expected_actor(self) -> None:
        """The expected actor and DB handle are still the ones cached and bound to the proven DB file."""
        from gateway.canonical_surface import bound_actor_db

        if self.expected_actor is None:
            return
        bound = bound_actor_db(self.runner, self.entry.session_key, self.session_id, self.expected_db_path,
                               self.expected_db_identity)
        if bound is None or bound[0] is not self.expected_actor or bound[1] is not self.expected_session_db:
            raise ValueError("canonical_agent_replaced")

    def cached_agent(self) -> Any:
        """The one cached actor this turn may run: never a new one, never one serving another session."""
        from gateway.canonical_surface import cached_actor

        self.require_current_head()
        agent, cached_session_id = cached_actor(self.runner, self.entry.session_key)
        if agent is None:
            raise ValueError("canonical_agent_missing")
        if self.expected_actor is not None and agent is not self.expected_actor:
            raise ValueError("canonical_agent_replaced")
        if cached_session_id != self.session_id or getattr(agent, "session_id", None) != self.session_id:
            raise ValueError("canonical_agent_missing")
        self.require_expected_actor()
        if not bool(getattr(agent, "compression_in_place", True)):
            raise ValueError("canonical_turn_refused")
        return agent

    def require_peer_fence(self, agent: Any, peer: dict, admitted: Any) -> None:
        """Re-run before every peer tool batch and every MCP send: the same run generation and actor on
        the same DB file, still serving this session with this event's receipt claimed, and the COMPLETE
        binding resolution the event was admitted under (principal, route entry, Telegram origin, not
        suspended, session open at its own compression tip, same entry, session and DB file)."""
        from gateway.canonical_surface import CanonicalReceiptCoordinator

        self.require_current_head()
        self.require_expected_actor()
        CanonicalReceiptCoordinator(self.runner)._require_same_resolution(self.binding, self.event, admitted)
        db = getattr(agent, "_session_db", None)
        if (getattr(agent, "session_id", None) != self.session_id or db is None
                or db.get_meta(peer["receipt"]) is None):
            raise ValueError("canonical_binding_stale")

    async def run(self, peer: Any, observation: Any) -> Any:
        """Run the cached actor as this event's peer with every outward callback quarantined."""
        from agent.canonical_peer import render_peer_turn, validate_peer_metadata
        from gateway.canonical_surface import CanonicalBindingProof, _runs_tools_outside_hermes

        agent = self.cached_agent()
        peer = validate_peer_metadata(peer)
        binding, event = self.binding, self.event
        if (peer["binding"], peer["author_id"], peer["channel_id"], peer["event_id"]) != (
                getattr(binding, "name", None), getattr(event, "author_id", None),
                getattr(event, "channel_id", None), getattr(event, "event_id", None)):
            raise ValueError("canonical_peer_provenance_invalid")
        peer_prompt = render_peer_turn(peer, event.text)
        admitted = CanonicalBindingProof(
            entry=self.entry, session_key=binding.session_key, session_id=self.session_id,
            db_path=self.expected_db_path, db_identity=self.expected_db_identity)
        if _runs_tools_outside_hermes(agent):
            raise ValueError("canonical_runtime_refused")
        outward_callbacks = {
            name: value
            for name, value in vars(agent).items()
            if self.runner._is_canonical_outward_callback_attribute(name)
        }
        for name in outward_callbacks:
            setattr(agent, name, None)
        try:
            return await self._run_quarantined(agent, peer, peer_prompt, admitted, observation)
        finally:
            for name, value in outward_callbacks.items():
                setattr(agent, name, value)

    async def _run_quarantined(self, agent: Any, peer: dict, peer_prompt: str, admitted: Any,
                               observation: Any) -> Any:
        from gateway.run import _build_gateway_agent_history

        history = await self.runner.async_session_store.load_transcript(self.session_id)
        agent_history, _ = _build_gateway_agent_history(history)
        self.require_current_head()
        self.runner._init_cached_agent_for_turn(agent, 0)
        work = functools.partial(self._peer_turn, agent, agent_history, peer, peer_prompt, admitted, observation)
        result = await run_owned_turn(
            self.runner, agent, work,
            session_key=self.binding.session_key, task_id=self.session_id,
            run_generation=self.expected_run_generation,
        )
        if observation is not None:
            # Before any check can reject the turn: the caller keeps what the actor returned.
            observation.returned = result
        self.require_current_head()
        self.require_expected_actor()
        return self._select(agent, result, peer, peer_prompt, observation)

    def _peer_turn(self, agent: Any, agent_history: list, peer: dict, peer_prompt: str, admitted: Any,
                   observation: Any) -> Any:
        """The turn itself, on the worker thread that runs it.

        Set on that thread: the agent attribute reaches the tool dispatchers and the Codex runtime
        guard, the context variable the approval gates. Both are restored only once the turn has
        actually returned."""
        from agent.canonical_peer_tools import PeerToolGrant, bind_grant, reset_grant
        from gateway.session_context import clear_session_vars, scoped_current_session_id, set_session_vars
        from tools.approval_context import reset_turn_principal, set_turn_principal

        origin = self.origin
        principal_token = set_turn_principal("peer")
        # The only grant a peer tool batch can run under: this verified binding, this actor, while the
        # fence still holds (agent.canonical_peer_tools). Its sends land in the caller's observation: a
        # failed turn records that scope.
        grant_token = bind_grant(PeerToolGrant(
            self.binding.name, agent, functools.partial(self.require_peer_fence, agent, peer, admitted),
            attempted=observation.attempted if observation is not None else []))
        prior_principal = getattr(agent, "_turn_principal", None)
        prior_skip_review = getattr(agent, "skip_background_review", False)
        agent._turn_principal = "peer"
        agent.skip_background_review = True
        # MCP call provenance names this turn's real session the way an owner turn's does (route, key,
        # session id). No user identity is bound, since the peer is not the owner; the live cron value
        # is carried over as bound (delegation depth has its own context), never reset to fall back on
        # the process environment.
        session_tokens = set_session_vars(
            platform=str(getattr(getattr(origin, "platform", None), "value", "") or ""),
            chat_id=str(getattr(origin, "chat_id", "") or ""),
            session_key=self.entry.session_key,
            parent_chat_id=str(getattr(origin, "parent_chat_id", "") or ""),
            cron_session=_current_cron_session())
        try:
            if observation is not None:
                observation.actor_started = True
            with scoped_current_session_id(self.session_id):
                return agent.run_conversation(
                    peer_prompt, conversation_history=agent_history, task_id=self.session_id,
                    persist_user_message=self.event.text,
                    persist_user_display_kind="canonical_peer",
                    persist_user_display_metadata={"canonical_peer": peer},
                )
        except Exception:
            # The actor's own failure, whatever its type (even a ValueError that reads like a gateway
            # refusal code): its receipt records a terminal failure.
            if observation is not None:
                observation.actor_failed = True
            raise
        finally:
            if observation is not None:
                observation.actor_finished = True
            clear_session_vars(session_tokens)
            agent._turn_principal = prior_principal
            agent.skip_background_review = prior_skip_review
            reset_grant(grant_token)
            reset_turn_principal(principal_token)

    def _select(self, agent: Any, result: Any, peer: dict, peer_prompt: str, observation: Any) -> Any:
        """The completed turn's answer, proven this event's; anything else is the turn's refusal."""
        boundary = getattr(agent, "_persist_user_message_idx", None)
        if isinstance(boundary, bool) or not isinstance(boundary, int):
            raise ValueError("canonical_turn_refused")
        try:
            selected = self.runner._select_canonical_turn_result(
                result,
                binding_name=self.binding.name,
                history_boundary=boundary,
                expected_session_id=self.session_id,
            )
        except ValueError:
            # The refusal stands. Only a final answer proven to be this event's, this claim's and this
            # turn's is handed to the receipt coordinator, to be kept, not delivered.
            if observation is not None:
                observation.bound_answer = _bound_refused_answer(
                    result, peer=peer, body=self.event.text, rendered=peer_prompt,
                    history_boundary=boundary, expected_session_id=self.session_id)
            raise
        # A completed turn belongs to this event only when it opened with this claim's own row and
        # carries no other peer row. Internal recovery/continuation rows do not change that provenance;
        # the same helper decides the refused-answer keep.
        if not _turn_binds(result, history_boundary=boundary, peer=peer, body=self.event.text,
                           rendered=peer_prompt):
            raise ValueError("canonical_turn_refused")
        return selected


class GatewayCanonicalTurnMixin:
    """The bound existing turn of a canonical receipt, for GatewayRunner."""

    @staticmethod
    def _is_canonical_outward_callback_attribute(name: str) -> bool:
        """Whether a cached actor attribute can emit beyond a canonical request."""
        return (
            name == "callback"
            or name.endswith("_callback")
            or name == "_on_session_title"
        )

    @staticmethod
    def _select_canonical_turn_result(
        result: Any,
        *,
        binding_name: str,
        history_boundary: int,
        expected_session_id: str,
    ) -> Any:
        """Select the completed turn's final response after checking tool-call/result pairing.

        Opening-row and foreign-event provenance are checked by ``turn_provenance`` in the caller.
        Recovery and continuation rows are normal parts of a completed turn.
        """
        from gateway.canonical_surface import CanonicalTurnResult

        if (
            not isinstance(result, dict)
            or result.get("completed") is not True
            or any(bool(result.get(flag)) for flag in ("failed", "partial", "interrupted"))
            or result.get("session_id", expected_session_id) != expected_session_id
            or not isinstance(result.get("final_response"), str)
            or not result["final_response"].strip()
            or not isinstance(result.get("messages"), list)
            or isinstance(history_boundary, bool)
            or not isinstance(history_boundary, int)
        ):
            raise ValueError("canonical_turn_refused")
        messages = result["messages"]
        if history_boundary < 0 or history_boundary >= len(messages):
            raise ValueError("canonical_turn_refused")
        suffix = messages[history_boundary:]
        if len(suffix) < 2:
            raise ValueError("canonical_turn_refused")
        user = suffix[0]
        if (
            not isinstance(user, dict)
            or user.get("role") != "user"
            or not isinstance(user.get("content"), str)
            or not user["content"].strip()
        ):
            raise ValueError("canonical_turn_refused")

        # Ids of the open tool-call row still owed a result: one tool row per id, each exactly once,
        # before anything else (executors may answer a batch out of call order).
        owed: set[str] = set()
        for row in suffix[1:]:
            if not isinstance(row, dict) or not isinstance(row.get("role"), str):
                raise ValueError("canonical_turn_refused")
            role = row["role"]
            if owed:
                if (
                    role != "tool"
                    or row.get("tool_call_id") not in owed
                    or not isinstance(row.get("content"), str)
                    or not row["content"].strip()
                ):
                    raise ValueError("canonical_turn_refused")
                owed.discard(row["tool_call_id"])
                continue
            if role == "user":
                continue
            if role != "assistant":
                raise ValueError("canonical_turn_refused")
            if row.get("tool_calls"):
                tool_calls = row["tool_calls"]
                ids = [call.get("id") if isinstance(call, dict) else None
                       for call in (tool_calls if isinstance(tool_calls, list) else ())]
                if (
                    not ids
                    or any(not isinstance(call_id, str) or not call_id.strip() for call_id in ids)
                    or len(set(ids)) != len(ids)
                ):
                    raise ValueError("canonical_turn_refused")
                owed = set(ids)
                continue
        if owed:
            raise ValueError("canonical_turn_refused")
        return CanonicalTurnResult(binding_name=binding_name, terminal_text=result["final_response"])

    async def run_bound_existing_turn(
        self, binding: Any, event: Any, entry: Any, *, reply_sink: Any = None,
        expected_actor: Any = None, expected_session_db: Any = None,
        expected_db_path: Any = None, expected_db_identity: Any = None,
        expected_run_generation: int | None = None,
        held_lease: Any = None, peer: Any = None, observation: Any = None,
    ) -> Any:
        """Run one exact cached actor without creating, routing, or delivering anything.

        The turn runs as a canonical *peer*: ``peer`` is the server-owned provenance the receipt
        coordinator minted for this event. The model gets only its rendering, the body is stored
        clean beside that metadata, and the turn's principal denies approvals, the background review
        and every tool but the configured CEO tools its grant allows (agent.canonical_peer_tools),
        whatever the owner's own settings are."""
        from gateway.canonical_surface import require_request_local_reply_sink

        require_request_local_reply_sink(reply_sink)
        session_id, origin = _bound_route(binding, entry)
        owner_key = f"canonical:{id(event)}"
        if held_lease is None:
            try:
                lease = await self._turn_leases.acquire(session_id, owner_key=owner_key, generation=0)
            except Exception:  # health: allow BLE001 -- any lease failure is this turn's busy refusal
                raise ValueError("canonical_turn_busy") from None
        else:
            lease = held_lease
            _require_held_lease(self._turn_leases, lease, session_id, owner_key)
        try:
            turn = _BoundTurn(
                self, binding, event, entry, session_id, origin,
                expected_actor=expected_actor, expected_session_db=expected_session_db,
                expected_db_path=expected_db_path, expected_db_identity=expected_db_identity,
                expected_run_generation=expected_run_generation,
            )
            return await turn.run(peer, observation)
        finally:
            if held_lease is None:
                self._turn_leases.release(lease)
