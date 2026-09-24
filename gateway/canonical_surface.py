"""Pure admission checks for configured canonical bindings.

This module deliberately does not receive ingress, create sessions, or deliver replies.
"""

import asyncio
import hashlib
import json
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from gateway.config import Platform
from gateway.session_persistence import _DB_UNPINNED


_EVENT_FIELDS = frozenset({"binding", "event_id", "author_id", "channel_id", "text"})
_MAX_ID_CHARS = 256
_MAX_TEXT_CHARS = 16_384
_MAX_RECEIPT_CHARS = 32_768
_RECEIPT_NAMESPACE = "canonical-receipt:v2"


def _required_text(value: Any, *, limit: int) -> str:
    """Validate text without normalizing caller-provided event identity."""

    if not isinstance(value, str) or not value.strip() or len(value) > limit:
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
        if duplicate or not isinstance(payload, dict) or set(payload) != _EVENT_FIELDS:
            raise ValueError("canonical_invalid_request")
        return cls(
            binding=_required_text(payload["binding"], limit=_MAX_ID_CHARS),
            event_id=_required_text(payload["event_id"], limit=_MAX_ID_CHARS),
            author_id=_required_text(payload["author_id"], limit=_MAX_ID_CHARS),
            channel_id=_required_text(payload["channel_id"], limit=_MAX_ID_CHARS),
            text=_required_text(payload["text"], limit=_MAX_TEXT_CHARS),
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
        if (
            getattr(event, "author_id", None) not in binding.allowed_author_ids
            or getattr(event, "channel_id", None) not in binding.allowed_channel_ids
        ):
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


@dataclass(frozen=True)
class CanonicalReceiptResult:
    """The durable state visible to an internal canonical caller."""

    status: str
    terminal_text: str | None = None


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
    def _encode(value: dict[str, Any]) -> str:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if len(encoded) > _MAX_RECEIPT_CHARS:
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

    def _require_current_claim_target(
        self, proof: CanonicalBindingProof, actor: Any, db: Any,
        run_generation: int | None = None,
    ) -> None:
        """Recheck the exact head, cached actor, and DB generation before receipt I/O."""
        if run_generation is not None and not self._runner._is_session_run_current(
            proof.session_key, run_generation
        ):
            raise ValueError("canonical_turn_interrupted")
        if self._runner.session_store.lookup_by_session_key_existing(proof.session_key) is not proof.entry:
            raise ValueError("canonical_binding_stale")
        current_actor, current_db = self._borrow_actor_db(self._runner, proof)
        if current_actor is not actor or current_db is not db:
            raise ValueError("canonical_agent_replaced")

    async def submit(
        self, binding: CanonicalSurfaceBinding, event: CanonicalIngressEvent
    ) -> CanonicalReceiptResult:
        """Claim, run once, and durably terminalize an already-authenticated canonical event."""
        if event.binding != binding.name:
            raise ValueError("canonical_binding_stale")
        entry, proof = ExistingCanonicalBindingResolver(self._runner.session_store).resolve_with_proof(
            binding, event
        )
        actor, db = self._borrow_actor_db(self._runner, proof)
        fingerprint = self._fingerprint(binding, event)
        key = self._key(binding, event)
        owner = secrets.token_hex(16)
        pending = self._encode({"v": 1, "state": "pending", "owner": owner, "fingerprint": fingerprint})
        if self._runner._is_session_running(proof.session_key):
            raise ValueError("canonical_turn_busy")
        try:
            lease = await self._runner._turn_leases.acquire(
                proof.session_id, owner_key=f"canonical:{id(event)}", generation=0,
            )
        except Exception:
            raise ValueError("canonical_turn_busy") from None
        run_generation: int | None = None
        try:
            # Waiting for the turn lease may have exposed a new head, actor, or DB generation.
            self._require_current_claim_target(proof, actor, db)
            if self._runner._is_session_running(proof.session_key):
                raise ValueError("canonical_turn_busy")
            run_generation = self._runner._begin_session_run_generation(proof.session_key)
            turn = self._runner._session_state(proof.session_key).turn
            turn.agent = actor
            turn.event = None
            turn.ctx = None
            turn.started_ts = time.time()
            # The handle is bound to the proof by file identity (bound_actor_db), so its own path
            # spelling is the one SessionDB's literal path guard can match; the identity is the proof.
            if not db.claim_meta_once(
                key, pending, proven_db_path=db.db_path, proven_db_identity=proof.db_identity
            ):
                receipt = db.get_meta(key)
                # get_meta has no generation fence, so its result is usable only after this recheck.
                self._require_current_claim_target(proof, actor, db, run_generation)
                return self._decode(receipt, fingerprint)

            from gateway.canonical_surface import request_local_reply_sink

            async def discard(_result: CanonicalTurnResult) -> None:
                return None

            result = await self._runner.run_bound_existing_turn(
                binding, event, entry, reply_sink=request_local_reply_sink(discard),
                expected_actor=actor, expected_session_db=db,
                expected_db_path=proof.db_path, expected_db_identity=proof.db_identity,
                expected_run_generation=run_generation,
                held_lease=lease,
            )
            if not isinstance(result, CanonicalTurnResult) or len(result.terminal_text) > _MAX_TEXT_CHARS:
                raise ValueError("canonical_receipt_invalid")
            terminal = self._encode({
                "v": 1, "state": "terminal", "fingerprint": fingerprint,
                "response": {"binding_name": result.binding_name, "terminal_text": result.terminal_text},
            })
            self._require_current_claim_target(proof, actor, db, run_generation)
            if not db.compare_and_set_meta(
                key, pending, terminal, proven_db_path=db.db_path, proven_db_identity=proof.db_identity
            ):
                raise ValueError("canonical_receipt_terminal_unconfirmed")
            self._require_current_claim_target(proof, actor, db, run_generation)
            return CanonicalReceiptResult("terminal", result.terminal_text)
        finally:
            if run_generation is not None:
                self._runner._release_running_agent_state(
                    proof.session_key, run_generation=run_generation
                )
            self._runner._turn_leases.release(lease)
