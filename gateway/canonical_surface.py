"""Pure admission checks for configured canonical bindings.

This module deliberately does not receive ingress, create sessions, or deliver replies.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from gateway.config import Platform
from gateway.session_persistence import _DB_UNPINNED


_EVENT_FIELDS = frozenset({"binding", "event_id", "author_id", "channel_id", "text"})
_IDENTITY_FIELDS = frozenset({"session_id", "lineage_root_digest", "process_pid", "process_started_at"})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_START_TOKEN = re.compile(r"(?:darwin-tv:[0-9]+\.[0-9]{6}|linux-clk:[0-9]+)\Z")
_MAX_ID_CHARS = 256
_MAX_TEXT_CHARS = 16_384
# A terminal record holds any accepted text: JSON escapes one character into at most six (\uXXXX).
_MAX_RECEIPT_CHARS = 6 * _MAX_TEXT_CHARS + 4_096
# A kept answer is the whole reply the actor returned, never the receipt's summary of an over-cap one,
# plus the failure that explains it (at most 32 listed sends). Any reply up to the answer bound fits
# however it escapes; a record past the kept bound is reported as not kept, never as kept.
_MAX_KEPT_ANSWER_CHARS = 1 << 20
_MAX_KEPT_CHARS = 6 * _MAX_KEPT_ANSWER_CHARS + 4_096 + 16_384
_RESULT_TOO_LARGE = "canonical_result_too_large"
_REFUSAL_CODE = re.compile(r"canonical_[a-z_]{1,64}")
_RECEIPT_NAMESPACE = "canonical-receipt:v2"
# Beside each receipt row, never instead of it: the event's claim record (written before the receipt
# claim, so a receipt row that later goes missing never reads as an unclaimed event), and the answer a
# claimed turn produced whose terminal record did not land (kept, not delivered, never re-run).
_CLAIM_NAMESPACE = "canonical-claim:v1"
_KEPT_NAMESPACE = "canonical-answer-kept:v1"
ANSWER_UNRECORDED = "canonical_answer_unrecorded"

logger = logging.getLogger(__name__)


def _required_text(value: Any, *, limit: int) -> str:
    """Validate text without normalizing caller-provided event identity."""

    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("canonical_invalid_request")
    return value

def _expected_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _IDENTITY_FIELDS:
        raise ValueError("canonical_invalid_request")
    if (
        not isinstance(value["session_id"], str)
        or not value["session_id"].strip()
        or len(value["session_id"]) > 512
        or not isinstance(value["lineage_root_digest"], str)
        or _DIGEST.fullmatch(value["lineage_root_digest"]) is None
        or type(value["process_pid"]) is not int
        or value["process_pid"] <= 0
        or not isinstance(value["process_started_at"], str)
        or _START_TOKEN.fullmatch(value["process_started_at"]) is None
    ):
        raise ValueError("canonical_invalid_request")
    return value


def _path_identity(path: Path | None) -> tuple[int, int] | None:
    """The (device, inode) the path names now, or None when it names nothing."""
    if path is None:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_dev, stat.st_ino


def cached_actor(runner: Any, session_key: str) -> tuple[Any, Any]:
    """The actor cached at ``session_key`` and the session id its cache entry was built for."""
    with runner._agent_cache_lock:
        cached = runner._agent_cache.get(session_key)
    if not cached:
        return None, None
    if isinstance(cached, tuple):
        actor = cached[0]
        return actor, cached[3] if len(cached) > 3 else getattr(actor, "session_id", None)
    return cached, getattr(cached, "session_id", None)


def bound_actor_db(
    runner: Any, session_key: str, session_id: str, db_path: Path | None,
    db_identity: tuple[int, int] | None,
) -> tuple[Any, Any] | None:
    """The cached actor and its DB handle, only while both are bound to the proven DB file.

    Bound: the cache entry at ``session_key`` serves ``session_id``, the actor's ``_session_db``
    was opened on the file ``db_identity`` names, and ``db_path`` still names that file. Paths are
    matched by file identity, never by spelling: the writer registry opens the resolved path while
    the proof keeps the home's own spelling, so a symlinked home must not read as a different DB.
    The claim, the pre-run and the post-run rechecks share this one answer; a missing piece is
    unbound, never an exception.
    """
    actor, cached_session_id = cached_actor(runner, session_key)
    db = getattr(actor, "_session_db", None)
    if (
        actor is None
        or db is None
        or db_identity is None
        or cached_session_id != session_id
        or getattr(actor, "session_id", None) != session_id
        or getattr(db, "_db_file_identity", None) != db_identity
        or _path_identity(db_path) != db_identity
    ):
        return None
    return actor, db


@dataclass(frozen=True)
class CanonicalTurnResult:
    """The terminal selected from one canonical turn, before request readback."""

    binding_name: str
    terminal_text: str


class RequestLocalReplySink(Protocol):
    """Opaque request-scoped capability for accepting exactly one terminal."""

    async def publish(self, result: CanonicalTurnResult) -> None: ...


class _RequestReplySink:
    """One-use sink whose publisher remains private to the creating request."""

    def __init__(
        self, publisher: Callable[[CanonicalTurnResult], Awaitable[None]]
    ) -> None:
        self._publisher = publisher
        self._published = False
        self._lock = asyncio.Lock()

    async def publish(self, result: CanonicalTurnResult) -> None:
        async with self._lock:
            if self._published:
                raise ValueError("canonical_reply_already_published")
            self._published = True
            try:
                await self._publisher(result)
            except Exception:
                raise ValueError("canonical_reply_publish_failed") from None


def request_local_reply_sink(
    publisher: Callable[[CanonicalTurnResult], Awaitable[None]],
) -> RequestLocalReplySink:
    """Create a reply capability that cannot select or retain a destination."""

    return _RequestReplySink(publisher)


def require_request_local_reply_sink(value: Any) -> RequestLocalReplySink:
    """Reject caller-supplied lookalikes in the existing-actor turn path."""

    if not isinstance(value, _RequestReplySink):
        raise ValueError("canonical_reply_sink_missing")
    return value


@dataclass(frozen=True)
class CanonicalSurfaceBinding:
    name: str
    session_key: str
    session_id: str
    telegram_chat_id: str
    telegram_chat_type: str
    telegram_user_id: str | None
    telegram_thread_id: str | None
    allowed_author_ids: tuple[str, ...]
    allowed_channel_ids: tuple[str, ...]
    # The intended (author_id, channel_id) pairs, in declared order. Empty for a binding of one author
    # in one channel, which admits exactly that pair.
    pairs: tuple[tuple[str, str], ...] = ()

    # Bound to the config's home, never the request thread's ambient profile.
    roster: Any = field(default=None, compare=False, repr=False)

    def roster_snapshot(self):
        if self.roster is None:
            return None
        from bridge.canonical.canonical_roster import RosterError
        try:
            return self.roster.read()
        except RosterError as exc:
            # Existing installations without a roster keep their configured pairs. Once observed,
            # disappearance is an outage, not permission to revive a retired config grant.
            if isinstance(exc.__cause__, FileNotFoundError) and self.roster.last_good is None:
                return None
            logger.error("Canonical routing roster rejected: %s", exc)
            raise ValueError("canonical_roster_unavailable") from exc

    def admits(self, author_id: Any, channel_id: Any) -> bool:
        """Only a declared pair, never the cross product of the two lists: a peer admitted in its own
        room is refused in every other room the binding lists. With no pairs declared, only the single
        author in the single channel; more than one of either with no pairs admits nothing."""
        snapshot = self.roster_snapshot()
        if snapshot is not None:
            from bridge.canonical.canonical_roster import RosterError, read_owner
            if channel_id in snapshot.channels:
                # Retirement revokes even an older explicit config grant.
                if snapshot.active(channel_id) is None:
                    return False
                try:
                    owner = read_owner(self.roster.path.parent / "keys" / "owner.pub")
                except RosterError as exc:
                    logger.error("Canonical routing roster rejected: %s", exc)
                    raise ValueError("canonical_roster_unavailable") from exc
                if snapshot.admits(author_id, channel_id, owner):
                    return True
        if author_id not in self.allowed_author_ids or channel_id not in self.allowed_channel_ids:
            return False
        if self.pairs:
            return (author_id, channel_id) in self.pairs
        return len(self.allowed_author_ids) == 1 and len(self.allowed_channel_ids) == 1

    def primary_pair(self) -> tuple[str, str]:
        """The binding's own (author_id, channel_id): the first declared pair, or else the single author
        and channel. A retired primary yields to the first active roster pair, preserving the session.
        IndexError when the binding names neither."""
        primary = self.pairs[0] if self.pairs else (self.allowed_author_ids[0], self.allowed_channel_ids[0])
        snapshot = self.roster_snapshot()
        if snapshot is not None and primary[1] in snapshot.channels and snapshot.active(primary[1]) is None:
            for channel, room in snapshot.channels.items():
                if room["status"] == "active":
                    return room["peer"]["pubkey"], channel
            raise ValueError("canonical_principal_rejected")
        return primary


@dataclass(frozen=True)
class CanonicalBindingProof:
    """The exact live head and physical DB generation a resolver just validated.

    This is evidence for a later existing-actor claimant to re-check. It does
    not reserve the route or make a subsequent turn safe by itself.
    """

    entry: Any
    session_key: str
    session_id: str
    db_path: Path
    db_identity: tuple[int, int]


@dataclass(frozen=True)
class CanonicalIngressEvent:
    """Closed canonical ingress payload with no caller-controlled destination."""

    binding: str
    event_id: str
    author_id: str
    channel_id: str
    text: str
    expected_identity: dict[str, Any] | None = None

    @classmethod
    def from_json_bytes(cls, raw: bytes) -> "CanonicalIngressEvent":
        """Parse the exact canonical event object, rejecting ambiguity at ingress."""

        duplicate = False

        def closed_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            nonlocal duplicate
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    duplicate = True
                result[key] = value
            return result

        try:
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=closed_object)
        except (AttributeError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("canonical_invalid_request") from None
        if (duplicate or not isinstance(payload, dict)
                or set(payload) not in (_EVENT_FIELDS, _EVENT_FIELDS | _IDENTITY_FIELDS)):
            raise ValueError("canonical_invalid_request")
        binding = _required_text(payload["binding"], limit=_MAX_ID_CHARS)
        event_id = _required_text(payload["event_id"], limit=_MAX_ID_CHARS)
        author_id = _required_text(payload["author_id"], limit=_MAX_ID_CHARS)
        channel_id = _required_text(payload["channel_id"], limit=_MAX_ID_CHARS)
        text = payload["text"]
        if isinstance(text, str) and len(text) > _MAX_TEXT_CHARS:
            # Refused before any claim, so nothing ran. The answer stays the plain 400 the bridge tells
            # the sender from; this line makes the refusal visible here too, by correlation, never text.
            logger.warning("Canonical event refused at ingress: reason=text_over_limit binding=%r "
                           "event_id=%r chars=%d limit=%d", binding[:96], event_id[:96], len(text),
                           _MAX_TEXT_CHARS)
        return cls(
            binding=binding,
            event_id=event_id,
            author_id=author_id,
            channel_id=channel_id,
            text=_required_text(text, limit=_MAX_TEXT_CHARS),
            expected_identity=(
                _expected_identity({key: payload[key] for key in _IDENTITY_FIELDS})
                if _IDENTITY_FIELDS <= payload.keys() else None
            ),
        )


class ExistingCanonicalBindingResolver:
    """Resolve a binding only to its current, already-live session-key head."""

    def __init__(self, session_store: Any) -> None:
        self._session_store = session_store

    def _existing_db_path(self, session_key: str) -> Path | None:
        """Prove the owning state.db exists without opening a writable store handle."""
        store = self._session_store
        try:
            pinned_db_for_store = getattr(store, "_pinned_db", None)
            if callable(pinned_db_for_store):
                pinned_db = pinned_db_for_store()
                if pinned_db is not _DB_UNPINNED:
                    pinned_path = getattr(pinned_db, "db_path", None)
                    if pinned_path is None:
                        return None
                    path = Path(pinned_path)
                    return path if path.is_file() else None

            named_profile = getattr(store, "_named_profile_for_key", None)
            if callable(named_profile) and named_profile(session_key) is not None:
                profile_home = getattr(store, "_profile_home_for_key", lambda _key: None)(session_key)
                if profile_home is None:
                    return None
                path = Path(profile_home) / "state.db"
                return path if path.is_file() else None

            routing_home = getattr(store, "_routing_home", None)
            if routing_home is not None:
                path = Path(routing_home) / "state.db"
                return path if path.is_file() else None
        except Exception:
            return None
        return None

    def _read_db_for_existing_path(
        self, session_key: str, *, existing_path: Path | None = None
    ) -> tuple[Any, bool] | tuple[None, bool]:
        """Return a separately-owned read-only DB, or the narrow fake-store compatibility DB."""
        path = existing_path if existing_path is not None else self._existing_db_path(session_key)
        if path is not None:
            from hermes_state import SessionDB

            return SessionDB(path, read_only=True), True
        # Production SessionStore instances provide this writer helper. Do not fall back to _db for
        # them: accessing it can create/open a writable database. Synthetic stores retain _db=fake.
        if callable(getattr(self._session_store, "_db_for_key", None)):
            return None, False
        return getattr(self._session_store, "_db", None), False

    def resolve(self, binding: CanonicalSurfaceBinding, event: Any) -> Any:
        """Return the existing entry exactly as before, without exposing a proof."""
        entry, _ = self._resolve(binding, event, require_proof=False)
        return entry

    def resolve_with_proof(
        self, binding: CanonicalSurfaceBinding, event: Any
    ) -> tuple[Any, CanonicalBindingProof]:
        """Return an existing live entry together with its frozen validation evidence."""
        entry, proof = self._resolve(binding, event, require_proof=True)
        assert proof is not None
        return entry, proof

    def _resolve(
        self, binding: CanonicalSurfaceBinding, event: Any, *, require_proof: bool
    ) -> tuple[Any, CanonicalBindingProof | None]:
        if not binding.admits(getattr(event, "author_id", None), getattr(event, "channel_id", None)):
            raise ValueError("canonical_principal_rejected")

        entry = self._session_store.lookup_by_session_key_existing(binding.session_key)
        if entry is None or getattr(entry, "platform", None) != Platform.TELEGRAM:
            raise ValueError("canonical_binding_stale")
        origin = getattr(entry, "origin", None)
        if origin is None or (
            getattr(origin, "chat_id", None) != binding.telegram_chat_id
            or getattr(origin, "chat_type", None) != binding.telegram_chat_type
            or getattr(origin, "user_id", None) != binding.telegram_user_id
            or getattr(origin, "thread_id", None) != binding.telegram_thread_id
        ):
            raise ValueError("canonical_binding_stale")

        db = None
        close_db = False
        db_path = self._existing_db_path(binding.session_key) if require_proof else None
        db_identity = _path_identity(db_path)
        try:
            if self._session_store._should_reset(entry, origin) is not None:
                raise ValueError("canonical_binding_stale")
            db, close_db = self._read_db_for_existing_path(
                binding.session_key, existing_path=db_path
            )
            session_id = getattr(entry, "session_id", None)
            if db is None or not session_id:
                raise ValueError("canonical_binding_stale")
            row = db.get_session(session_id)
            if not row or row.get("ended_at") is not None or db.get_compression_tip(session_id) != session_id:
                raise ValueError("canonical_binding_stale")
            current_entry = self._session_store.lookup_by_session_key_existing(binding.session_key)
            if current_entry is not entry or getattr(current_entry, "session_id", None) != session_id:
                raise ValueError("canonical_binding_stale")
            if require_proof:
                if (
                    db_path is None
                    or db_identity is None
                    or _path_identity(db_path) != db_identity
                ):
                    raise ValueError("canonical_binding_stale")
                proof: CanonicalBindingProof | None = CanonicalBindingProof(
                    entry=entry,
                    session_key=binding.session_key,
                    session_id=session_id,
                    db_path=db_path,
                    db_identity=db_identity,
                )
            else:
                proof = None
        except ValueError:
            raise ValueError("canonical_binding_stale")
        except Exception:
            raise ValueError("canonical_binding_stale") from None
        finally:
            if close_db and db is not None:
                try:
                    db.close()
                except Exception:
                    pass
        return entry, proof


def _runs_tools_outside_hermes(actor: Any) -> bool:
    """Whether the actor's turn would run on a runtime that executes tools and asks approvals outside
    Hermes' own tool loop (the Codex app-server: exec, patch and MCP tools in its subprocess), where a
    peer turn's no-tools/no-approvals restriction cannot be enforced. The primary runtime counts too:
    a cached agent on a fallback restores it at turn start."""
    primary = getattr(actor, "_primary_runtime", None)
    modes = {getattr(actor, "api_mode", None), primary.get("api_mode") if isinstance(primary, dict) else None}
    return "codex_app_server" in modes


@contextmanager
def _unconfirmed_once_claimed():
    """Report a refusal raised once the event's receipt row exists as an unconfirmed receipt.

    Refusal codes tell a caller the turn did not run and nothing was stored for it. Past
    ``claim_meta_once`` neither is provable (the actor may already have run), so a caller that
    resent under a new event id would run the turn twice; the receipt, not the refusal, answers.
    """
    try:
        yield
    except ValueError as exc:
        raise ValueError("canonical_receipt_terminal_unconfirmed") from exc


@dataclass(frozen=True)
class CanonicalReceiptResult:
    """The durable state visible to an internal canonical caller. A terminal ``failure`` (never
    text) is a turn that did not complete: its code and the side-effect scope observed.

    ``unrecorded``: the turn ran and its answer exists, but its terminal record did not land (or the
    turn was rejected after the actor returned it); the answer is kept beside the receipt or in the
    local fallback beside the state.db (``failure['answer_kept']``, False when neither could hold
    it), never delivered, never re-run.
    ``uncertain``: the event was claimed before and nothing left can say what happened to it."""

    status: str
    terminal_text: str | None = None
    failure: dict | None = None


def _log_record_state(level: int, what: str, *, stage: str, binding: Any, event: Any, key: str,
                      observed: str, delivery: str, **detail: Any) -> None:
    """One log line naming the failing stage, the event correlation (binding, event id, receipt key)
    and the delivery state observed. Never the answer or the event text."""
    logger.log(level, "Canonical event %s: stage=%s binding=%s event_id=%r receipt=%s observed=%s delivery=%s%s",
               what, stage, getattr(binding, "name", None), str(getattr(event, "event_id", ""))[:96], key,
               observed, delivery, "".join(f" {name}={value}" for name, value in detail.items()))


_SIDE_EFFECT_SCOPES = frozenset({"none", "attempted", "unknown"})
_MAX_LISTED_SENDS = 32


@dataclass
class CanonicalTurnObservation:
    """What one canonical turn was seen to do, for its durable failure record: whether the actor's
    turn started, raised and finished under this observation, and every peer tool send that reached
    the transport."""

    actor_started: bool = False
    actor_failed: bool = False
    actor_finished: bool = False
    attempted: list = field(default_factory=list)
    # What the actor's turn returned, set as soon as it returns: a rejection raised after that point
    # (head, actor or generation no longer current) must not discard the answer it holds.
    returned: Any = None
    # The final answer of a turn whose shape the selector refused, set only when
    # ``bound_refused_answer`` proved it is this event's, this claim's and this turn's. It is kept,
    # never delivered and never promoted to the event's answer; None leaves the turn's terminal failure.
    bound_answer: Any = None

    def failure(self, code: str, **detail: Any) -> dict:
        """``side_effects``: ``attempted`` (with the sends, ``settled`` once the actor finished so the
        list is complete) when any send went out; ``none`` only when the actor was seen to finish
        having sent nothing; ``unknown`` otherwise (the scope is unresolved)."""
        if self.attempted:  # the count is exact; the listed sends are capped to fit the receipt
            return {"code": code, "side_effects": "attempted", "settled": self.actor_finished,
                    "attempted_count": len(self.attempted), "attempted": list(self.attempted[:_MAX_LISTED_SENDS]),
                    **detail}
        settled = self.actor_started and self.actor_finished
        return {"code": code, "side_effects": "none" if settled else "unknown", **detail}


def _result_answer(result: Any, observation: CanonicalTurnObservation) -> CanonicalReceiptResult:
    """A completed turn's answer as its receipt states it, or, for a result no receipt may carry, a
    terminal failure summary that fits: its length and digest tie it to the full reply. The summary
    is the receipt's only; an answer that is kept is kept whole (see ``submit``'s ``keep``)."""
    if not isinstance(result, CanonicalTurnResult):
        return CanonicalReceiptResult("terminal", None, observation.failure("canonical_result_invalid"))
    text = result.terminal_text
    if len(text) > _MAX_TEXT_CHARS:
        return CanonicalReceiptResult("terminal", None, observation.failure(
            _RESULT_TOO_LARGE, result_chars=len(text),
            result_sha256=hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()))
    return CanonicalReceiptResult("terminal", text)


def _returned_text(observation: CanonicalTurnObservation) -> str | None:
    """The whole answer text the actor returned before its turn was rejected, or None when it
    returned none."""
    returned = observation.returned
    text = returned.get("final_response") if isinstance(returned, dict) else None
    return text if isinstance(text, str) and text.strip() else None


@dataclass(frozen=True)
class TurnProvenance:
    """The provenance facts of the rows one canonical turn added (``messages[history_boundary:]``), read once
    by ``turn_provenance`` for both the accepted path and the refused-answer keep, so neither is a second
    authority.

    ``opening_is_this_claims``: the opening row is this turn's own peer row — this claim's metadata with its
    fresh nonce and this event's body, or exactly the rendering the model received. ``other_peer_row``: a
    later row, up to and including the last, is marked as a peer row (``display_kind`` ``canonical_peer``,
    or ``display_metadata`` carrying ``canonical_peer``) — another event's, receipt's or nonce's, or this
    one's again. ``interior_user_row``: a row other than the actor's assistant or tool rows (a user row) lies
    between the first and the last, and it is not a recovery nudge of this run: one whose
    ``_recovery_nudge`` marker names this claim's peer nonce as its run, and one of the five nudge kinds
    whose text is its content (``agent.recovery_nudge.is_run_nudge``). These are facts, not verdicts: a row
    with no metadata is never trusted by them, and nothing classifies a row by its text alone."""

    opening_is_this_claims: bool
    other_peer_row: bool
    interior_user_row: bool

    @property
    def binds(self) -> bool:
        """The rule both paths share (H92-03): the turn opened with this claim's row and carries no other
        peer row."""
        return self.opening_is_this_claims and not self.other_peer_row


def turn_provenance(messages: Any, *, history_boundary: Any, peer: Any, body: str,
                    rendered: str) -> TurnProvenance | None:
    """The ``TurnProvenance`` of a turn's rows, or None when they cannot be read (never bound)."""
    from collections.abc import Mapping

    from agent.canonical_peer import PEER_DISPLAY_KIND, PEER_METADATA_KEY, PeerProvenanceError, peer_body, peer_metadata
    from agent.recovery_nudge import is_run_nudge

    if (not isinstance(messages, list) or type(history_boundary) is not int
            or not 0 <= history_boundary < len(messages) - 1):
        return None
    suffix = messages[history_boundary:]
    if not all(isinstance(row, dict) for row in suffix):
        return None
    first = suffix[0]
    opening = False
    if first.get("role") == "user":
        try:
            marked = peer_metadata(first)
            opening = (first.get("content") == rendered if marked is None
                       else marked == peer and peer_body(first) == body)
        except (PeerProvenanceError, ValueError, TypeError):
            opening = False
    other_peer = any(row.get("display_kind") == PEER_DISPLAY_KIND
                     or (isinstance(row.get("display_metadata"), Mapping)
                         and PEER_METADATA_KEY in row["display_metadata"])
                     for row in suffix[1:])
    # This claim's run token is its own peer's nonce, the one source the agent stamps a nudge's marker from.
    run = peer.get("nonce") if isinstance(peer, Mapping) else None
    interior_user = any(row.get("role") not in ("assistant", "tool") and not is_run_nudge(row, run)
                        for row in suffix[1:-1])
    return TurnProvenance(opening, other_peer, interior_user)


def bound_refused_answer(result: Any, *, peer: Any, body: str, rendered: str, history_boundary: Any,
                         expected_session_id: str) -> str | None:
    """The final answer of a turn the selector refused, or None when it is not provably this event's.

    The selector's refusal stands: the turn ends in its terminal failure and this text is never the
    event's answer. It is only kept (not delivered) when all of these hold for the very result this
    claim's run returned:

    * the turn completed, is not failed, partial or interrupted, and ran in the bound session;
    * its rows bind it to this claim (``turn_provenance(...).binds``, the same rule a selected answer
      meets), and, beyond that rule, no user row lies inside the turn but this run's own recovery nudges
      (the Limit below);
    * it ends with one assistant row, without tool calls, whose text is the returned final answer.

    Limit: a user row inside the turn makes the answer unbound unless it is a recovery nudge this run's
    agent loop appended. Matching a nudge's fixed text is no provenance. Each of the five nudges is built by
    ``agent.recovery_nudge.recovery_nudge_row``, which, in a canonical peer turn, marks it
    ``_recovery_nudge: {"kind", "run"}`` with ``run`` read from the structured metadata of the turn's own
    peer opening row: this claim's peer nonce, minted fresh per claim. Only a row whose marker names
    exactly that nonce, one of the five kinds and, after both, that kind's text as its content is accepted
    (``is_run_nudge``). A row without the marker, with another run, kind or text, or any other user row
    still unbinds the answer, and ``binds`` stays required, so a marker never admits a foreign or second
    peer row.

    Anything else, or anything unreadable, is None: the event stays an explicit failure."""
    if (not isinstance(result, dict) or result.get("completed") is not True
            or any(bool(result.get(flag)) for flag in ("failed", "partial", "interrupted"))
            or result.get("session_id", expected_session_id) != expected_session_id):
        return None
    final, messages = result.get("final_response"), result.get("messages")
    if not isinstance(final, str) or not final.strip():
        return None
    facts = turn_provenance(messages, history_boundary=history_boundary, peer=peer, body=body, rendered=rendered)
    if facts is None or not facts.binds or facts.interior_user_row:
        return None
    last = messages[-1]
    if (not isinstance(last, dict) or last.get("role") != "assistant" or "tool_calls" in last
            or last.get("content") != final):
        return None
    return final


def _valid_failure(failure: Any) -> bool:
    return (isinstance(failure, dict) and isinstance(failure.get("code"), str)
            and failure.get("side_effects") in _SIDE_EFFECT_SCOPES
            and (failure["side_effects"] == "attempted") == bool(failure.get("attempted")))


def _turn_failure_code(exc: BaseException, observation: CanonicalTurnObservation) -> str | None:
    """The terminal failure of a turn whose actor ran: it raised, or it ended partial or failed.
    None keeps the receipt pending, as before: nothing ran (refused before the actor started), or
    a binding, actor or DB proof no longer holds and cannot be written to."""
    if observation.actor_failed:
        return "canonical_turn_failed"
    if observation.actor_started and isinstance(exc, ValueError) and str(exc) == "canonical_turn_refused":
        return "canonical_turn_refused"
    return None


_UNRECORDED = "canonical_event_unrecorded"  # internal: no receipt yet; never answered to a caller


class CanonicalReceiptCoordinator:
    """Claim one canonical event on the already-open DB owned by its cached actor.

    This is deliberately an internal coordinator: it neither receives HTTP nor publishes a
    reply.  A pending receipt is deliberately sticky; recovery is outside this narrow path.
    """

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    @staticmethod
    def _fingerprint(binding: CanonicalSurfaceBinding, event: CanonicalIngressEvent) -> str:
        payload = json.dumps(
            {"binding": binding.name, "session_key": binding.session_key,
             "telegram_origin": (binding.telegram_chat_id, binding.telegram_chat_type,
                                 binding.telegram_user_id, binding.telegram_thread_id),
             "event_id": event.event_id, "author_id": event.author_id,
             "channel_id": event.channel_id, "text": event.text},
            ensure_ascii=False, separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _key(binding: CanonicalSurfaceBinding, event: CanonicalIngressEvent) -> str:
        # Digest a length-framed namespace so distinct literal identifiers cannot concatenate.
        fields = (
            binding.name,
            binding.session_key,
            binding.telegram_chat_id,
            binding.telegram_chat_type,
            json.dumps(
                (binding.telegram_user_id, binding.telegram_thread_id),
                ensure_ascii=False, separators=(",", ":"),
            ),
            event.event_id,
            event.author_id,
            event.channel_id,
        )
        framed = b"".join(
            len(value.encode("utf-8")).to_bytes(4, "big") + value.encode("utf-8")
            for value in fields
        )
        return f"{_RECEIPT_NAMESPACE}:{hashlib.sha256(framed).hexdigest()}"

    @staticmethod
    def _beside(namespace: str, key: str) -> str:
        """A record kept beside the receipt ``key``: same event digest, its own namespace."""
        return f"{namespace}:{key.rsplit(':', 1)[1]}"

    def _read_records(self, reader: Any, key: str, home: Path) -> tuple[Any, Any, Any]:
        """(kept answer, receipt, claim record) for the event, read on ``reader``. The kept answer is
        the one beside the receipt or, when the state.db holds none, the one the local fallback in
        ``home`` (the state.db's directory) holds."""
        kept = reader.get_meta(self._beside(_KEPT_NAMESPACE, key))
        if kept is None:
            kept = self._fallback_kept(home, key)
        return kept, reader.get_meta(key), reader.get_meta(self._beside(_CLAIM_NAMESPACE, key))

    @staticmethod
    def _fallback_kept(home: Path, key: str) -> str | None:
        """The kept record the local fallback holds for the event, bound to the claim its name
        carries; unreadable is never read as nothing kept."""
        from gateway.canonical_answer_fallback import kept_for

        try:
            kept = kept_for(home, key.rsplit(":", 1)[1])
        except OSError as exc:
            logger.warning("Canonical answer fallback unreadable: receipt=%s error=%s", key, type(exc).__name__)
            raise ValueError("canonical_receipt_unreadable") from None
        for owner, raw in kept:
            try:
                value = raw.decode("utf-8", "surrogatepass")
                parsed = json.loads(value)
            except Exception:
                parsed = None
            if not isinstance(parsed, dict) or parsed.get("owner") != owner or parsed.get("receipt") != key:
                raise ValueError("canonical_receipt_conflict")
            return value
        return None

    def _answer_records(
        self, binding: CanonicalSurfaceBinding, event: Any, key: str, fingerprint: str,
        records: tuple[Any, Any, Any],
    ) -> CanonicalReceiptResult | None:
        """What a claimed event answers, or None when nothing was ever recorded for it.

        A kept answer comes first, so a replay gives the first request's explicit failure whatever
        became of the receipt row. A receipt answers as always, once its terminal is bound to the
        claim that ran the event: a terminal another holder wrote, or one whose writer is unknown,
        is a conflict, never this event's answer. A claim record without either is an event whose
        receipt row went missing: it was claimed, may have run and been delivered, so it is uncertain
        and never run again. A missing row is never read as an unclaimed event."""
        kept, receipt, claimed = records
        if kept is not None:
            return self._decode_kept(kept, fingerprint)
        if receipt is not None:
            answer = self._decode(receipt, fingerprint)
            claim_owner = self._decode_claim(claimed, fingerprint) if claimed is not None else None
            # A terminal answers only when the claim that ran the event names its owner and the
            # terminal names that same owner. An unknown owner proves no writer, on every path: a claim
            # record from before claims named one, or no claim record at all (its row lost, read cold
            # after a restart). The terminal stays where it is; it is never answered as this event's.
            if answer.status == "terminal" and (not claim_owner or json.loads(receipt).get("owner") != claim_owner):
                _log_record_state(logging.WARNING, "terminal not bound to its claim", stage="replay.owner",
                                  binding=binding, event=event, key=key,
                                  observed="recorded_other" if claim_owner else "owner_unknown",
                                  delivery="not_delivered")
                raise ValueError("canonical_receipt_conflict")
            return answer
        if claimed is None:
            return None
        self._decode_claim(claimed, fingerprint)
        _log_record_state(logging.WARNING, "receipt row missing", stage="claim.receipt_missing",
                          binding=binding, event=event, key=key, observed="absent", delivery="unknown")
        return CanonicalReceiptResult("uncertain")

    @staticmethod
    def _decode_claim(value: Any, fingerprint: str) -> str | None:
        """The owner the claim record names, or None for a claim record that names none."""
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict) or parsed.get("v") != 1 or parsed.get("state") != "claimed" \
                or parsed.get("fingerprint") != fingerprint or not isinstance(parsed.get("owner", ""), str):
            raise ValueError("canonical_receipt_conflict")
        return parsed.get("owner")

    @staticmethod
    def _decode_kept(value: Any, fingerprint: str) -> CanonicalReceiptResult:
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = None
        if (not isinstance(parsed, dict) or parsed.get("v") != 1 or parsed.get("state") != "kept"
                or parsed.get("fingerprint") != fingerprint or not _valid_failure(parsed.get("unrecorded"))
                or parsed["unrecorded"].get("code") != ANSWER_UNRECORDED):
            raise ValueError("canonical_receipt_conflict")
        # Kept means the record holds the answer's text. A length and digest stand for an answer and a
        # failure is no answer at all; neither reads as kept.
        response = (parsed.get("answer") or {}).get("response") if isinstance(parsed.get("answer"), dict) else None
        kept = isinstance(response, dict) and isinstance(response.get("terminal_text"), str)
        return CanonicalReceiptResult("unrecorded", None, {**parsed["unrecorded"], "answer_kept": kept})

    @classmethod
    def _kept_text(cls, value: Any, fingerprint: str) -> str | None:
        """The answer text a kept record holds, read the way a replay reads it; None for none."""
        if value is None or cls._decode_kept(value, fingerprint).failure.get("answer_kept") is not True:
            return None
        return json.loads(value)["answer"]["response"]["terminal_text"]

    @staticmethod
    def _observe_receipt(db: Any, key: str, pending: str, terminal: str | None, fingerprint: str,
                         proof: CanonicalBindingProof) -> str:
        """What the receipt row holds once this request's terminal record did not land."""
        try:
            value = db.get_meta(key)
        except Exception:
            return "unreadable"
        if value is None:
            return "absent"
        if value == pending:
            return "own_pending"
        if terminal is not None and value == terminal:
            # Binding only while the proven file is still the one its path names.
            return "recorded_same" if _path_identity(proof.db_path) == proof.db_identity else "replaced"
        try:
            parsed = json.loads(value)
        except Exception:
            return "unreadable"
        if not isinstance(parsed, dict) or parsed.get("fingerprint") != fingerprint:
            return "conflict"
        return {"pending": "other_pending", "terminal": "recorded_other"}.get(parsed.get("state"), "unreadable")

    @staticmethod
    def _encode(value: dict[str, Any], limit: int = _MAX_RECEIPT_CHARS) -> str:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if len(encoded) > limit:
            raise ValueError("canonical_receipt_invalid")
        return encoded

    @staticmethod
    def _decode(value: Any, fingerprint: str) -> CanonicalReceiptResult:
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict) or parsed.get("v") != 1:
                raise ValueError("invalid")
            if parsed.get("fingerprint") != fingerprint:
                raise ValueError("canonical_receipt_conflict")
            if parsed.get("state") == "pending" and isinstance(parsed.get("owner"), str):
                return CanonicalReceiptResult("pending")
            if parsed.get("state") == "terminal" and _valid_failure(parsed.get("failure")):
                return CanonicalReceiptResult("terminal", None, parsed["failure"])
            response = parsed.get("response")
            if (parsed.get("state") == "terminal" and isinstance(response, dict)
                    and isinstance(response.get("terminal_text"), str)):
                return CanonicalReceiptResult("terminal", response["terminal_text"])
        except ValueError as exc:
            if str(exc) == "canonical_receipt_conflict":
                raise
        except Exception:
            pass
        raise ValueError("canonical_receipt_conflict")

    @staticmethod
    def _borrow_actor_db(runner: Any, proof: CanonicalBindingProof) -> tuple[Any, Any]:
        """Return only the cached actor and its existing writable DB matching resolver proof."""
        bound = bound_actor_db(
            runner, proof.session_key, proof.session_id, proof.db_path, proof.db_identity
        )
        db = bound[1] if bound is not None else None
        if (
            bound is None
            or getattr(db, "read_only", True)
            or not callable(getattr(db, "claim_meta_once", None))
            or not callable(getattr(db, "compare_and_set_meta", None))
            or not callable(getattr(db, "get_meta", None))
        ):
            raise ValueError("canonical_binding_stale")
        return bound

    def _require_same_resolution(
        self, binding: CanonicalSurfaceBinding, event: Any, proof: CanonicalBindingProof
    ) -> None:
        """Resolve the binding again, with every lifecycle check (route entry, Telegram origin, not
        suspended, session row open, compression tip, DB proof), and require the very entry, session
        and DB file ``proof`` named. Anything else is ``canonical_binding_stale``."""
        entry, fresh = ExistingCanonicalBindingResolver(self._runner.session_store).resolve_with_proof(
            binding, event)
        if (entry is not proof.entry or fresh.session_key != proof.session_key
                or fresh.session_id != proof.session_id or fresh.db_identity != proof.db_identity):
            raise ValueError("canonical_binding_stale")

    async def _build_actor_db(self, proof: CanonicalBindingProof, entry: Any) -> tuple[Any, Any]:
        """The agent for a resolved live session that had none cached, built the way an ordinary turn
        builds it and then borrowed exactly as a cached one. Called holding the session's turn lease,
        which ordinary turns hold while they resolve their agent: one cached meanwhile by such a turn
        is borrowed, never built a second time. The resolver already proved the session exists, is
        live and matches the binding; nothing here creates a session, route or binding, and an agent
        that is not bound to the proven session and DB file is refused as stale."""
        if cached_actor(self._runner, proof.session_key)[0] is None:
            await self._runner._build_agent_for_resolved_session(entry)
        actor, db = self._borrow_actor_db(self._runner, proof)
        if _runs_tools_outside_hermes(actor):
            raise ValueError("canonical_runtime_refused")
        return actor, db

    def _recorded_or_refuse(
        self, binding: CanonicalSurfaceBinding, event: Any, key: str, fingerprint: str, code: str,
        proof: CanonicalBindingProof | None = None,
    ) -> CanonicalReceiptResult:
        """Answer an event an earlier request recorded; refuse with ``code`` only when none was.

        A refusal reads as "not run", so a caller resends under a new event id: were the event
        recorded, that runs it twice. Without the cached actor there is no writable handle, and
        without a resolved head (a restarted gateway loads its routing index only after its
        listeners are up) there is no proof, so the receipt is read on a read-only handle to the
        state.db the resolver proves for this binding's own session key and profile scope. "Recorded"
        includes a kept answer and a claim record whose receipt row went missing.
        """
        from hermes_state import SessionDB

        if proof is not None:
            db_path, db_identity = proof.db_path, proof.db_identity
        else:
            db_path = ExistingCanonicalBindingResolver(self._runner.session_store)._existing_db_path(
                binding.session_key
            )
            db_identity = _path_identity(db_path)
        if db_path is None or db_identity is None:
            raise ValueError(code)
        reader = SessionDB(db_path, read_only=True)
        try:
            if getattr(reader, "_db_file_identity", None) != db_identity:
                # The proven file was replaced: whether it held a receipt cannot be known now.
                raise ValueError("canonical_receipt_unreadable")
            records = self._read_records(reader, key, db_path.parent)
        finally:
            reader.close()
        answer = self._answer_records(binding, event, key, fingerprint, records)
        if answer is None:
            raise ValueError(code)
        return answer

    def _require_current_claim_target(
        self, proof: CanonicalBindingProof, actor: Any, db: Any,
        run_generation: int | None = None,
    ) -> None:
        """Recheck the exact head, cached actor, and DB generation before receipt I/O."""
        if run_generation is not None and not self._runner._is_session_run_current(
            proof.session_key, run_generation
        ):
            raise ValueError("canonical_turn_interrupted")
        if (self._runner.session_store.lookup_by_session_key_existing(proof.session_key) is not proof.entry
                or getattr(proof.entry, "session_id", None) != proof.session_id):
            raise ValueError("canonical_binding_stale")
        current_actor, current_db = self._borrow_actor_db(self._runner, proof)
        if current_actor is not actor or current_db is not db:
            raise ValueError("canonical_agent_replaced")

    @staticmethod
    def _require_expected_identity(proof: CanonicalBindingProof, expected: dict[str, Any]) -> None:
        """Compare durable lineage and native process identity under the held turn lease."""
        from gateway.platforms.api_server_canonical import _process_started_at
        from hermes_state_target_bind import _lineage_root_digest, _resolve_lineage_root

        try:
            with closing(sqlite3.connect(proof.db_path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
                conn.row_factory = sqlite3.Row
                if _path_identity(proof.db_path) != proof.db_identity:
                    raise ValueError("canonical_binding_stale")
                root = _resolve_lineage_root(conn, proof.session_id)
            pid = os.getpid()
            if (
                expected["session_id"] != proof.session_id
                or expected["lineage_root_digest"] != _lineage_root_digest(root)
                or expected["process_pid"] != pid
                or (started_at := _process_started_at(pid)) is None
                or expected["process_started_at"] != started_at
            ):
                raise ValueError("canonical_binding_stale")
        except (sqlite3.Error, OSError, RuntimeError):
            raise ValueError("canonical_binding_stale") from None

    async def submit(
        self, binding: CanonicalSurfaceBinding, event: CanonicalIngressEvent
    ) -> CanonicalReceiptResult:
        """Claim, run once, and durably terminalize an already-authenticated canonical event."""
        if event.binding != binding.name:
            raise ValueError("canonical_binding_stale")
        if event.expected_identity is not None:
            _expected_identity(event.expected_identity)
        fingerprint = self._fingerprint(binding, event)
        key = self._key(binding, event)
        # Principal admission is never answered from a receipt; every later refusal before this
        # request's own claim is, when the event was already recorded.
        try:
            entry, proof = ExistingCanonicalBindingResolver(self._runner.session_store).resolve_with_proof(
                binding, event
            )
        except ValueError as refusal:
            if str(refusal) in {"canonical_principal_rejected", "canonical_roster_unavailable"}:
                raise
            return self._recorded_or_refuse(binding, event, key, fingerprint, str(refusal))
        if cached_actor(self._runner, proof.session_key)[0] is None:
            # No agent cached (a clean restart, before any ordinary turn). An event already recorded
            # answers from its receipt here, as it always has; only an unrecorded one goes on to have
            # this resolved session's agent built, under the turn lease, below.
            try:
                return self._recorded_or_refuse(binding, event, key, fingerprint, _UNRECORDED, proof)
            except ValueError as refusal:
                if str(refusal) != _UNRECORDED:
                    raise
        actor = db = None
        try:
            # A cached agent is borrowed as always.
            if cached_actor(self._runner, proof.session_key)[0] is not None:
                actor, db = self._borrow_actor_db(self._runner, proof)
            if self._runner._is_session_running(proof.session_key):
                raise ValueError("canonical_turn_busy")
            if actor is not None and _runs_tools_outside_hermes(actor):
                raise ValueError("canonical_runtime_refused")
            # The event runs as a peer: its verified signer is structured provenance beside the body,
            # and the model sees the body only quoted under a fresh per-turn nonce. A body that
            # already carries the nonce could close the quote, so it is refused here, unclaimed.
            from agent.canonical_peer import new_peer_metadata, render_peer_turn

            peer = new_peer_metadata(binding=binding.name, author_id=event.author_id,
                                     channel_id=event.channel_id, event_id=event.event_id, receipt=key)
            render_peer_turn(peer, event.text)
        except ValueError as refusal:
            return self._recorded_or_refuse(binding, event, key, fingerprint, str(refusal), proof)
        owner = secrets.token_hex(16)
        pending = self._encode({"v": 1, "state": "pending", "owner": owner, "fingerprint": fingerprint})
        try:
            lease = await self._runner._turn_leases.acquire(
                proof.session_id, owner_key=f"canonical:{id(event)}", generation=0,
            )
        except Exception:
            return self._recorded_or_refuse(binding, event, key, fingerprint, "canonical_turn_busy", proof)
        run_generation: int | None = None
        try:
            # Waiting for the turn lease may have exposed a new head, actor, or DB generation, and the
            # session may have been ended, suspended or re-routed meanwhile: the complete resolution
            # runs again before anything is built or claimed, and again after a build.
            try:
                self._require_same_resolution(binding, event, proof)
                if actor is None:
                    actor, db = await self._build_actor_db(proof, entry)
                    self._require_same_resolution(binding, event, proof)
                self._require_current_claim_target(proof, actor, db)
                if event.expected_identity is not None:
                    self._require_expected_identity(proof, event.expected_identity)
                if self._runner._is_session_running(proof.session_key):
                    raise ValueError("canonical_turn_busy")
                # An open user row would be merged with this peer turn by the agent's alternation
                # repair, erasing the boundary between the two principals. Checked under the lease.
                if db.latest_conversation_role(proof.session_id) == "user":
                    raise ValueError("canonical_history_unanswered")
            except ValueError as refusal:
                return self._recorded_or_refuse(binding, event, key, fingerprint, str(refusal), proof)
            run_generation = self._runner._begin_session_run_generation(proof.session_key)
            turn = self._runner._session_state(proof.session_key).turn
            turn.agent = actor
            turn.event = None
            turn.ctx = None
            turn.started_ts = time.time()
            # The handle is bound to the proof by file identity (bound_actor_db), so its own path
            # spelling is the one SessionDB's literal path guard can match; the identity is the proof.
            proven = {"proven_db_path": db.db_path, "proven_db_identity": proof.db_identity}
            # A missing receipt row is not an unclaimed event: the claim record goes first and stays,
            # so neither a re-entry nor a restored row can run a claimed event again. It names the
            # claim's owner, so a terminal is this event's answer only when this claim wrote it.
            claim = self._encode({"v": 1, "state": "claimed", "fingerprint": fingerprint, "owner": owner})
            if not (db.claim_meta_once(self._beside(_CLAIM_NAMESPACE, key), claim, **proven)
                    and db.claim_meta_once(key, pending, **proven)):
                with _unconfirmed_once_claimed():
                    records = self._read_records(db, key, proof.db_path.parent)
                    # get_meta has no generation fence, so its result is usable only after this recheck.
                    self._require_current_claim_target(proof, actor, db, run_generation)
                answer = self._answer_records(binding, event, key, fingerprint, records)
                return answer if answer is not None else CanonicalReceiptResult("uncertain")

            from gateway.canonical_answer_fallback import keep as keep_locally
            from gateway.canonical_surface import request_local_reply_sink

            async def discard(_result: CanonicalTurnResult) -> None:
                return None

            observation = CanonicalTurnObservation()

            def outcome(answer: CanonicalReceiptResult) -> dict[str, Any]:
                return ({"failure": answer.failure} if answer.failure is not None else {"response": {
                    "binding_name": binding.name, "terminal_text": answer.terminal_text}})

            def terminal_of(answer: CanonicalReceiptResult) -> str:
                # Bound to this claim (its owner) and to this one write (a nonce no one holds before it
                # lands): a readback equal to it is this write, never another holder's equal answer.
                return self._encode({"v": 1, "state": "terminal", "fingerprint": fingerprint, "owner": owner,
                                     "write": secrets.token_hex(16), **outcome(answer)})

            def record(answer: CanonicalReceiptResult, generation: int | None) -> None:
                terminal = terminal_of(answer)
                self._require_current_claim_target(proof, actor, db, generation)
                if not db.compare_and_set_meta(key, pending, terminal, **proven):
                    raise ValueError("canonical_receipt_terminal_unconfirmed")

            def keep(answer: CanonicalReceiptResult, whole: str | None, *, stage: str, cause: str,
                     observed: str) -> CanonicalReceiptResult:
                """The one boundary an answer the request will not deliver passes, before the request
                is answered: kept beside the receipt or, when the state.db refuses that, in the local
                fallback beside it, and read back either way. Kept is not delivered and is never resent;
                the turn never runs again. When neither holds it, the failure and the log say so.

                What is kept is ``whole``, the answer text exactly as the actor returned it, never the
                receipt's length-and-digest summary of an over-cap answer: ``answer_kept`` is true only
                when those very bytes read back. A turn that returned no text keeps its failure."""
                failure = observation.failure(ANSWER_UNRECORDED, stage=stage, observed=observed)
                kept_answer = outcome(answer) if whole is None else {
                    "response": {"binding_name": binding.name, "terminal_text": whole}}
                # Ties the log and the kept record to the answer, without the log carrying it.
                digest = hashlib.sha256((whole if whole is not None else json.dumps(
                    answer.failure, sort_keys=True)).encode("utf-8", "surrogatepass")).hexdigest()
                kept_key, preserved, errors, record_kept = self._beside(_KEPT_NAMESPACE, key), None, {}, None
                readback = None
                try:
                    record_kept = self._encode({"v": 1, "state": "kept", "fingerprint": fingerprint, "owner": owner,
                                                "receipt": key, "unrecorded": failure, "answer": kept_answer,
                                                "answer_sha256": digest}, _MAX_KEPT_CHARS)
                    if not db.claim_meta_once(kept_key, record_kept, **proven):
                        errors["keep_error"] = "already_kept"  # never written over
                    elif (readback := db.get_meta(kept_key)) != record_kept:
                        errors["keep_error"] = "readback_mismatch"
                    else:
                        preserved = "receipt_store"
                except Exception as exc:
                    errors["keep_error"] = type(exc).__name__
                if preserved is None and record_kept is None:
                    errors["fallback_error"] = "no_record"  # the record itself could not be built
                elif preserved is None:
                    try:
                        errors["fallback"] = keep_locally(proof.db_path.parent, key.rsplit(":", 1)[1], owner,
                                                          record_kept.encode("utf-8", "surrogatepass"))
                        preserved = "local_fallback"
                        # Read back the way a replay reads the fallback: the record it finds first.
                        readback = self._fallback_kept(proof.db_path.parent, key)
                    except Exception as exc:
                        errors["fallback_error" if preserved is None else "readback_error"] = type(exc).__name__
                # ``answer_kept`` is true only for answer text that was written and then read back, through
                # the replay's own decoding, as these very characters. A kept failure is no kept answer.
                answer_kept = False
                if preserved is not None and whole is not None:
                    try:
                        answer_kept = readback == record_kept and self._kept_text(readback, fingerprint) == whole
                    except Exception as exc:
                        errors["readback_error"] = type(exc).__name__
                    if not answer_kept:
                        errors.setdefault("readback_error", "not_read_back_as_this_answer")
                lost = preserved is None or (whole is not None and not answer_kept)
                _log_record_state(
                    logging.CRITICAL if lost else logging.ERROR,
                    "terminal not recorded and its answer NOT preserved" if lost else "terminal not recorded",
                    stage=stage, binding=binding, event=event, key=key, observed=observed, delivery="not_delivered",
                    cause=cause, answer_kept=answer_kept, preservation=preserved or "failed",
                    answer_sha256=digest, **errors)
                return CanonicalReceiptResult("unrecorded", None, {**failure, "answer_kept": answer_kept})

            def record_or_keep(answer: CanonicalReceiptResult, whole: str | None) -> CanonicalReceiptResult:
                """Record the answer as the event's terminal, or, when that does not land, keep it (its
                ``whole`` text) and end in an explicit failure: it is never delivered unbound, never
                written over another holder's row or answer, and the turn never runs again. Only a
                receipt that holds this very write answers with it; another holder's equal answer is a
                conflict."""
                stage, terminal = "record.encode", None
                try:
                    terminal = terminal_of(answer)
                    stage = "record.proof"
                    self._require_current_claim_target(proof, actor, db, run_generation)
                    stage = "record.compare_and_set"
                    if db.compare_and_set_meta(key, pending, terminal, **proven):
                        return answer
                    cause = "no_row_changed"
                except Exception as exc:
                    code = str(exc)
                    cause = code if isinstance(exc, ValueError) and _REFUSAL_CODE.fullmatch(code) else type(exc).__name__
                observed = self._observe_receipt(db, key, pending, terminal, fingerprint, proof)
                if observed == "recorded_same":
                    _log_record_state(logging.WARNING, "terminal not recorded by this write", stage=stage,
                                      binding=binding, event=event, key=key, observed=observed,
                                      delivery="answered_from_receipt", cause=cause)
                    return answer
                return keep(answer, whole, stage=stage, cause=cause, observed=observed)

            result = whole = None
            with _unconfirmed_once_claimed():
                try:
                    result = await self._runner.run_bound_existing_turn(
                        binding, event, entry, reply_sink=request_local_reply_sink(discard),
                        expected_actor=actor, expected_session_db=db,
                        expected_db_path=proof.db_path, expected_db_identity=proof.db_identity,
                        expected_run_generation=run_generation,
                        held_lease=lease, peer=peer, observation=observation,
                    )
                except asyncio.CancelledError:
                    # Cancelled past the claim (the turn runner first joins the interrupted actor): the
                    # terminal failure with the sends observed is stored, then the cancel goes on. The
                    # cancel itself retired this run's generation, so only head, actor and DB are proved.
                    try:
                        record(CanonicalReceiptResult(
                            "terminal", None, observation.failure("canonical_turn_cancelled")), None)
                    except Exception:
                        pass  # proof lost: the receipt stays pending, as for any unwritable target
                    raise
                except Exception as exc:
                    # The actor raised, or its turn ended partial or failed: a durable terminal failure
                    # with the side effects actually observed, never success and never a replay.
                    code = _turn_failure_code(exc, observation)
                    returned = _returned_text(observation) if code is None else None
                    if returned is not None:
                        # The actor returned an answer and the turn was then rejected (its head, actor or
                        # generation no longer current): the receipt stays pending as before, and the
                        # answer passes the same boundary as one whose terminal record did not land.
                        rejected = str(exc)
                        return keep(_result_answer(CanonicalTurnResult(binding.name, returned), observation),
                                    returned, stage="turn.rejected_after_return",
                                    cause=rejected if isinstance(exc, ValueError) and _REFUSAL_CODE.fullmatch(rejected)
                                    else type(exc).__name__,
                                    observed=self._observe_receipt(db, key, pending, None, fingerprint, proof))
                    if code is None:
                        raise
                    answer = CanonicalReceiptResult("terminal", None, observation.failure(code))
                    bound = observation.bound_answer if code == "canonical_turn_refused" else None
                    if isinstance(bound, str) and bound.strip():
                        # The selector refused the turn's shape, so its text is never the event's answer.
                        # The final answer proven to be this event's, claim's and turn's is kept first
                        # (not delivered), then the refusal is recorded as the terminal failure; the
                        # request answers the kept state, never success.
                        kept = keep(_result_answer(CanonicalTurnResult(binding.name, bound), observation), bound,
                                    stage="turn.refused_bound_answer", cause=code,
                                    observed=self._observe_receipt(db, key, pending, None, fingerprint, proof))
                        try:
                            record(answer, run_generation)
                        except Exception as exc:
                            _log_record_state(logging.WARNING, "refusal not recorded beside its kept answer",
                                              stage="turn.refused_bound_answer", binding=binding, event=event,
                                              key=key, delivery="not_delivered", cause=type(exc).__name__,
                                              observed=self._observe_receipt(db, key, pending, None, fingerprint,
                                                                             proof))
                        return kept
                else:
                    answer = _result_answer(result, observation)
                    whole = result.terminal_text if isinstance(result, CanonicalTurnResult) else None
                answer = record_or_keep(answer, whole)
            # The terminal (or the kept answer) is durable and a replay answers it, so this answer is
            # the same; a head that moves after the compare-and-set cannot change what was recorded.
            return answer
        finally:
            if run_generation is not None:
                self._runner._release_running_agent_state(
                    proof.session_key, run_generation=run_generation
                )
            self._runner._turn_leases.release(lease)
