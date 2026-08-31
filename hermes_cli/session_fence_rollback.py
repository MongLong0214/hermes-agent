"""One bounded, offline way back off the turn-fence generation barrier.

WHAT THIS IS NOT
    Not a compatibility fallback. Not an automatic trigger drop on open. Not a
    silent admission of an old writer. A binary from before the barrier being
    unable to write a fenced store is INTENTIONAL and required, and nothing in
    this module runs unless a person asks for it by name, on a store nobody is
    using.

WHO CALLS IT, AND WHAT IS CURRENTLY REACHABLE
    ``hermes sessions fence-rollback`` — :mod:`hermes_cli.session_fence_rollback_cmd`
    — and nothing else in the tree. Both entry points below —
    :func:`rollback_turn_fence` and :func:`rehearse_turn_fence_rollback` —
    refuse before they observe the store: their bare call to
    :func:`establish_offline_authority` has no target session/process authority
    bindings and therefore fails closed. The store is never opened, copied or
    written to by the verb.

THIS IS THE REACHABLE CONTRACT ONLY
    An earlier revision of this module also carried the backup/commit engine
    that a real rollback would need once an artifact's offline authority could
    actually be established: exclusive destination acquisition, a
    SQLite-verified backup, a pre-flight surface comparison, an in-transaction
    commit of the trigger drops, and a withdrawal ledger for cleanup. Every one
    of those ~2,600 lines sat behind :func:`establish_offline_authority`, which
    refuses unconditionally in this build — so none of it was reachable from
    the verb, and a test exercising it was maintenance evidence about code
    nothing could call, not a statement about behaviour. It has been removed
    from this reduction along with its dedicated pins; the refusal contract
    below is unchanged by the removal, because the engine never ran. The
    removed engine remains recoverable from this repository's object store
    (commit ``0e7c88685f``, ``fix(c5): fence-rollback verb joins the
    composition with its Family A/B pins``) for whichever future slice
    actually establishes an artifact's offline authority.

WHY THIS EXISTS ANYWAY, EVEN AS A REFUSAL
    Because the alternative that was actually on offer — an operator typing
    ``DROP TRIGGER`` once per surface entry into whatever database they think
    is the right one — is not a rollback story: no backup, no check that the
    store is idle, no check that it is removing the surface it believes it is
    removing, and no atomicity. Naming the refusal, its reason, and the
    surface a real rollback would touch is worth more than either the manual
    route or a success gated on an inference that has a counterexample (see
    :func:`establish_offline_authority`).

THE RETURN LEG
    Rollback is not a one-way door. ``_init_schema`` creates the triggers with
    ``CREATE TRIGGER IF NOT EXISTS`` on every open, so a current binary
    reopening a rolled-back store puts the fence back and the old binary is
    refused again. That is a property of the store, not a favour this module
    does.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import hermes_state_common


class TurnFenceRollbackRefused(RuntimeError):
    """The rollback did not run, and the store was not touched.

    Every raise site in this module happens BEFORE any DDL — so catching this
    means the store is exactly as it was found.

    CARRIES A REASON, NOT JUST A SENTENCE
        Every raise site tags itself with a stable ``reason``. An operator — or
        the script driving them — must be able to tell WHICH refusal they hit,
        and the only thing a caller could otherwise key on is the message text,
        which is exactly the check that goes stale the day someone improves the
        wording. ``preflight`` says how far the pre-flight got before this
        refusal, which is the other half of the same question.
    """

    def __init__(self, message: str, *, reason: str = "refused") -> None:
        super().__init__(message)
        self.reason = reason
        # EMPTY, not all-false. A refusal that never reached the pre-flight has
        # no pre-flight result, and a block of false checks would claim each of
        # them was performed and failed. Nothing in this build ever reaches a
        # pre-flight, so this stays empty on every path.
        self.preflight: dict[str, bool] = {}


def rollback_trigger_names() -> tuple[str, ...]:
    """The triggers a rollback would remove, GENERATED from the declaration.

    Read through the module rather than bound at import so that moving
    ``TURN_FENCE_SURFACE`` moves this with it. Unreachable on the only path
    this build has today (every invocation refuses first), kept because it is
    the generated fact a future success report would need and there is no
    hand-copied list here to go stale.
    """
    declared = getattr(hermes_state_common, "TURN_FENCE_TRIGGERS", None)
    if declared:
        return tuple(declared)
    return tuple(
        hermes_state_common.turn_fence_trigger_name(table, operation)
        for table, operation in hermes_state_common.TURN_FENCE_SURFACE
    )


#: The sidecars SQLite may have left beside the main file. Checked for, never
#: created, by :func:`disqualify_the_target`.
_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

DISQUALIFICATION_REASONS = {
    "namespace": "target-untrusted-namespace",
    "canonical": "canonical-store-target",
    "not-quiesced": "target-not-quiesced",
    "unknown": "offline-authority-unknown",
}


def _canonical_store_paths() -> list:
    """EVERY path this build might mean by "the store", not the first one.

    Canonicalness is a SET, and reducing it to one candidate is how a real
    ``state.db`` gets classified as something nobody recognises. There are at
    least two independent answers and they disagree by design:

    * ``hermes_state._default_db_path()`` — which honours a re-pointed
      ``DEFAULT_DB_PATH`` above everything else;
    * ``get_hermes_home() / "state.db"`` — the profile's home as it stands right
      now.

    A profile override, a re-pointed default, or a test fixture makes those two
    different paths, and the store the operator is actually running on is
    whichever the process would open. If the artifact is ANY of them it is the
    live one.

    Both are asked of the modules that own them rather than reconstructed here,
    for the same reason the trigger names are generated: a second copy of
    somebody else's rule is right until they change it.
    """
    candidates = []
    try:
        from hermes_state import _default_db_path

        candidates.append(Path(_default_db_path()))
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from hermes_constants import get_hermes_home

        candidates.append(Path(get_hermes_home()) / "state.db")
    except Exception:  # pragma: no cover - defensive
        pass
    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def _same_file(left: Path, right: Path) -> bool:
    try:
        a, b = os.stat(left), os.stat(right)
    except OSError:
        return False
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def disqualify_the_target(artifact: Path) -> None:
    """Refuse the targets that are provably wrong, before anything is opened.

    Three facts about the PATHNAME and the INODE, none of which needs the store
    to be read, and each of which is a different next move for the operator. It
    runs first precisely because it must not depend on the store being
    openable: an artifact with a hot journal beside it cannot be probed
    read-only without SQLite trying to roll it back, so a check that waited
    until after a pre-flight would never be reached on the case it is for.

    It only ever ADDS refusals. Nothing here can authorise anything, which is
    why using the canonical location is safe in this direction and would not be
    in the other — "outside HERMES_HOME therefore offline" is an inference, and
    "this IS the store the binary opens by default" is an identity.
    """
    artifact = Path(artifact)
    try:
        info = os.lstat(artifact)
    except FileNotFoundError:
        raise TurnFenceRollbackRefused(
            f"{artifact} is not a file", reason="store-missing"
        ) from None
    except OSError as exc:
        raise TurnFenceRollbackRefused(
            f"{artifact} could not be examined: {exc}", reason="store-unreadable"
        ) from exc
    import stat as _stat

    if not _stat.S_ISREG(info.st_mode):
        # A directory, a symlink, a device. Named here rather than left to a
        # later check, because there is no later check any more: nothing opens
        # the target, so a fact about the pathname is the only fact available.
        raise TurnFenceRollbackRefused(
            f"{artifact} is not a plain file, so it is not a rollback target",
            reason="store-unreadable",
        )
    if info.st_nlink != 1:
        raise TurnFenceRollbackRefused(
            f"{artifact} has {info.st_nlink} hard links, so another name "
            "reaches the same file and nothing this run can see governs it. A "
            "rollback is not permitted on an artifact whose namespace is not "
            "bounded. Nothing was changed",
            reason=DISQUALIFICATION_REASONS["namespace"],
        )
    # BY IDENTITY, against every candidate. A string comparison is what a
    # symlink or an override defeats, which is the same lesson the A/B store
    # swap taught one layer down.
    for canonical in _canonical_store_paths():
        if _same_file(artifact, canonical):
            raise TurnFenceRollbackRefused(
                f"{artifact} IS a canonical store this binary opens on its own "
                f"({canonical}). The canonical store is the live one whatever "
                "it looks like at this instant, and a detached artifact is "
                "what this verb acts on. Nothing was changed",
                reason=DISQUALIFICATION_REASONS["canonical"],
            )
    present = [
        str(artifact.with_name(artifact.name + suffix))
        for suffix in _SIDECAR_SUFFIXES
        if artifact.with_name(artifact.name + suffix).exists()
    ]
    if present:
        raise TurnFenceRollbackRefused(
            f"{artifact} has SQLite sidecars beside it ({', '.join(present)}), "
            "so it is attached to a connection or was interrupted while it "
            "was. A detached artifact has none of these. Nothing was changed",
            reason=DISQUALIFICATION_REASONS["not-quiesced"],
        )


_OFFLINE_AUTHORITY_SCHEMA_VERSION = 1
_OFFLINE_AUTHORITY_KIND = "hermes.offline-session-authority"
_OFFLINE_AUTHORITY_ROOT_KEYS = frozenset(
    {"schema_version", "kind", "issued_at", "nonce", "target", "source", "digest"}
)
_OFFLINE_AUTHORITY_TARGET_KEYS = frozenset(
    {"session_id", "session_generation", "process"}
)
_OFFLINE_AUTHORITY_PROCESS_KEYS = frozenset({"pid", "create_time", "argv"})
_OFFLINE_AUTHORITY_SOURCE_KEYS = frozenset(
    {
        "db",
        "schema_generation",
        "ledger_digest",
        "checkpoint_digest",
        "active_sessions_digest",
    }
)
_OFFLINE_AUTHORITY_DB_KEYS = frozenset({"device", "inode", "size", "sha256"})


def _authority_refusal(reason: str) -> None:
    raise TurnFenceRollbackRefused(
        "offline authority could not be established; nothing was changed",
        reason=reason,
    )


def _canonical_authority_digest(payload: Mapping[str, Any]) -> str:
    """Digest closed-schema authority bytes, excluding only its digest field."""
    unsigned = {key: value for key, value in payload.items() if key != "digest"}
    encoded = json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    """Read one regular-file identity and content digest without authorising a race."""
    try:
        before = os.stat(path)
        if not os.path.isfile(path):
            _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
        data = path.read_bytes()
        after = os.stat(path)
    except OSError:
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        _authority_refusal("offline-authority-stale")
    return {
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "size": int(before.st_size),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read_json_authority(path: Path) -> tuple[list[dict[str, Any]], str]:
    identity = _file_identity(path)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    if not isinstance(parsed, list) or any(not isinstance(item, dict) for item in parsed):
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    if _file_identity(path) != identity:
        _authority_refusal("offline-authority-stale")
    return parsed, identity["sha256"]


def _normalise_process_identity(process_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(process_identity, Mapping):
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    pid = process_identity.get("pid")
    create_time = process_identity.get("create_time")
    argv = process_identity.get("argv")
    if (
        isinstance(pid, bool)
        or not isinstance(pid, int)
        or pid <= 0
        or isinstance(create_time, bool)
        or not isinstance(create_time, (int, float))
        or not math.isfinite(float(create_time))
        or float(create_time) <= 0
        or not isinstance(argv, str)
        or not argv.strip()
    ):
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    return {"pid": pid, "create_time": float(create_time), "argv": argv}


def _read_source_authority(source_db: Path, session_id: str) -> dict[str, Any]:
    """Bind one session row and lease state to one read-only SQLite snapshot."""
    identity_before = _file_identity(source_db)
    try:
        with sqlite3.connect(f"{source_db.resolve().as_uri()}?mode=ro", uri=True) as conn:
            conn.execute("PRAGMA query_only = ON")
            conn.execute("BEGIN")
            schema_generation = conn.execute("PRAGMA schema_version").fetchone()[0]
            row = conn.execute(
                "SELECT id, git_metadata_generation FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            leased = conn.execute(
                "SELECT 1 FROM session_turn_leases WHERE conversation_id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
            conn.rollback()
    except (OSError, sqlite3.Error, ValueError):
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    if (
        row is None
        or not isinstance(row[0], str)
        or row[0] != session_id
        or isinstance(row[1], bool)
        or not isinstance(row[1], int)
        or row[1] < 0
        or leased is not None
    ):
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    identity_after = _file_identity(source_db)
    if identity_after != identity_before:
        _authority_refusal("offline-authority-stale")
    return {
        "db": identity_after,
        "schema_generation": int(schema_generation),
        "session_generation": int(row[1]),
    }


def _default_authority_paths() -> tuple[Path, Path]:
    from hermes_cli import process_identity
    from tools import process_registry as registry_module

    return Path(process_identity._ledger_path()), Path(registry_module.CHECKPOINT_PATH)


def _default_active_sessions() -> tuple[dict[str, Any], ...]:
    from tools.process_registry import process_registry

    with process_registry._lock:
        return tuple(
            {
                "session_id": session.id,
                "parent_session_id": session.parent_session_id,
                "pid": session.pid,
                "create_time": session.host_start_time,
                "argv": session.command,
                "exited": session.exited,
            }
            for session in process_registry._running.values()
        )


def _active_sessions_digest(
    active_sessions: Iterable[Mapping[str, Any]], session_id: str
) -> str:
    """Reject any active entry attached to the target; bind the observed set."""
    observed = []
    for entry in active_sessions:
        if not isinstance(entry, Mapping):
            _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
        if entry.get("parent_session_id") == session_id and not entry.get("exited", False):
            _authority_refusal("offline-authority-active")
        observed.append(
            {
                "session_id": str(entry.get("session_id", "")),
                "parent_session_id": str(entry.get("parent_session_id", "")),
                "pid": entry.get("pid"),
                "create_time": entry.get("create_time"),
                "argv": str(entry.get("argv", "")),
                "exited": bool(entry.get("exited", False)),
            }
        )
    return _canonical_authority_digest({"active_sessions": observed})


def _capture_offline_authority(
    source_db: Path,
    *,
    session_id: str,
    process_identity: Mapping[str, Any],
    ledger_path: Path,
    checkpoint_path: Path,
    active_sessions: Iterable[Mapping[str, Any]],
    liveness: Callable[[int, float], bool | None],
) -> dict[str, Any]:
    """Read, cross-check, and generation-compare all persisted authorities."""
    if not isinstance(session_id, str) or not session_id:
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    process = _normalise_process_identity(process_identity)
    source_before = _read_source_authority(source_db, session_id)
    ledger_before, ledger_digest_before = _read_json_authority(ledger_path)
    checkpoint_before, checkpoint_digest_before = _read_json_authority(checkpoint_path)
    active_digest_before = _active_sessions_digest(active_sessions, session_id)

    matching_ledger = [
        entry
        for entry in ledger_before
        if entry.get("pid") == process["pid"]
        and entry.get("create_time") == process["create_time"]
        and entry.get("argv") == process["argv"]
    ]
    if len(matching_ledger) != 1:
        _authority_refusal("offline-authority-identity-mismatch")
    if any(
        entry.get("parent_session_id") == session_id
        and not entry.get("exited", False)
        for entry in checkpoint_before
    ):
        _authority_refusal("offline-authority-active")

    alive = liveness(process["pid"], process["create_time"])
    if alive is True:
        _authority_refusal("offline-authority-active")
    if alive is not False:
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])

    # No one SQLite transaction can include the ledger/checkpoint files.  Re-read
    # every source and reject any changed generation/identity rather than joining
    # incompatible observations into a plausible-looking authority.
    source_after = _read_source_authority(source_db, session_id)
    ledger_after, ledger_digest_after = _read_json_authority(ledger_path)
    checkpoint_after, checkpoint_digest_after = _read_json_authority(checkpoint_path)
    active_digest_after = _active_sessions_digest(active_sessions, session_id)
    if (
        source_after != source_before
        or ledger_after != ledger_before
        or checkpoint_after != checkpoint_before
        or ledger_digest_after != ledger_digest_before
        or checkpoint_digest_after != checkpoint_digest_before
        or active_digest_after != active_digest_before
    ):
        _authority_refusal("offline-authority-stale")

    return {
        "target": {
            "session_id": session_id,
            "session_generation": source_before["session_generation"],
            "process": process,
        },
        "source": {
            "db": source_before["db"],
            "schema_generation": source_before["schema_generation"],
            "ledger_digest": ledger_digest_before,
            "checkpoint_digest": checkpoint_digest_before,
            "active_sessions_digest": active_digest_before,
        },
    }


def establish_offline_authority(
    artifact: Path,
    *,
    session_id: str | None = None,
    process_identity: Mapping[str, Any] | None = None,
    ledger_path: Path | None = None,
    checkpoint_path: Path | None = None,
    active_sessions: Iterable[Mapping[str, Any]] | None = None,
    liveness: Callable[[int, float], bool | None] | None = None,
    issued_at: float | None = None,
    nonce: str | None = None,
) -> dict[str, Any]:
    """Produce a closed-schema OFFLINE authority, or refuse before rollback.

    The public rollback verb provides none of the target bindings and continues
    to refuse.  A later, separately-authorised consumer must present this
    artifact to :func:`verify_offline_authority`; this producer never mutates a
    store or invokes rollback work.
    """
    artifact = Path(artifact)
    disqualify_the_target(artifact)
    if session_id is None or process_identity is None:
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    if ledger_path is None or checkpoint_path is None:
        ledger_path, checkpoint_path = _default_authority_paths()
    if active_sessions is None:
        active_sessions = _default_active_sessions()
    if liveness is None:
        from hermes_cli.process_identity import _pid_alive_matches

        liveness = _pid_alive_matches
    captured = _capture_offline_authority(
        artifact,
        session_id=session_id,
        process_identity=process_identity,
        ledger_path=Path(ledger_path),
        checkpoint_path=Path(checkpoint_path),
        active_sessions=tuple(active_sessions),
        liveness=liveness,
    )
    if issued_at is None:
        issued_at = time.time()
    if (
        isinstance(issued_at, bool)
        or not isinstance(issued_at, (int, float))
        or not math.isfinite(float(issued_at))
        or not isinstance(nonce, (str, type(None)))
        or (nonce is not None and not nonce)
    ):
        _authority_refusal(DISQUALIFICATION_REASONS["unknown"])
    authority = {
        "schema_version": _OFFLINE_AUTHORITY_SCHEMA_VERSION,
        "kind": _OFFLINE_AUTHORITY_KIND,
        "issued_at": float(issued_at),
        "nonce": nonce if nonce is not None else secrets.token_hex(32),
        **captured,
    }
    authority["digest"] = _canonical_authority_digest(authority)
    return authority


def _closed_authority(authority: object) -> bool:
    if not isinstance(authority, dict) or set(authority) != _OFFLINE_AUTHORITY_ROOT_KEYS:
        return False
    if (
        authority.get("schema_version") != _OFFLINE_AUTHORITY_SCHEMA_VERSION
        or authority.get("kind") != _OFFLINE_AUTHORITY_KIND
        or isinstance(authority.get("issued_at"), bool)
        or not isinstance(authority.get("issued_at"), (int, float))
        or not math.isfinite(float(authority["issued_at"]))
        or not isinstance(authority.get("nonce"), str)
        or not authority["nonce"]
        or not isinstance(authority.get("digest"), str)
        or len(authority["digest"]) != 64
    ):
        return False
    target = authority.get("target")
    source = authority.get("source")
    if not isinstance(target, dict) or set(target) != _OFFLINE_AUTHORITY_TARGET_KEYS:
        return False
    if not isinstance(source, dict) or set(source) != _OFFLINE_AUTHORITY_SOURCE_KEYS:
        return False
    process = target.get("process")
    db = source.get("db")
    if not isinstance(process, dict) or set(process) != _OFFLINE_AUTHORITY_PROCESS_KEYS:
        return False
    if not isinstance(db, dict) or set(db) != _OFFLINE_AUTHORITY_DB_KEYS:
        return False
    try:
        _normalise_process_identity(process)
    except TurnFenceRollbackRefused:
        return False
    return (
        isinstance(target.get("session_id"), str)
        and bool(target["session_id"])
        and isinstance(target.get("session_generation"), int)
        and not isinstance(target.get("session_generation"), bool)
        and target["session_generation"] >= 0
        and isinstance(source.get("schema_generation"), int)
        and not isinstance(source.get("schema_generation"), bool)
        and all(isinstance(source.get(key), str) and len(source[key]) == 64 for key in (
            "ledger_digest", "checkpoint_digest", "active_sessions_digest"
        ))
        and all(isinstance(db.get(key), int) and not isinstance(db.get(key), bool) and db[key] >= 0 for key in (
            "device", "inode", "size"
        ))
        and isinstance(db.get("sha256"), str)
        and len(db["sha256"]) == 64
        and _canonical_authority_digest(authority) == authority["digest"]
    )


def verify_offline_authority(
    authority: object,
    artifact: Path,
    *,
    session_id: str,
    process_identity: Mapping[str, Any],
    ledger_path: Path | None = None,
    checkpoint_path: Path | None = None,
    active_sessions: Iterable[Mapping[str, Any]] | None = None,
    liveness: Callable[[int, float], bool | None] | None = None,
) -> bool:
    """Recompute an authority from live sources without enabling mutation."""
    if not _closed_authority(authority):
        return False
    try:
        expected_process = _normalise_process_identity(process_identity)
    except TurnFenceRollbackRefused:
        return False
    if authority["target"]["session_id"] != session_id or authority["target"]["process"] != expected_process:
        return False
    try:
        current = establish_offline_authority(
            artifact,
            session_id=session_id,
            process_identity=expected_process,
            ledger_path=ledger_path,
            checkpoint_path=checkpoint_path,
            active_sessions=active_sessions,
            liveness=liveness,
            issued_at=authority["issued_at"],
            nonce=authority["nonce"],
        )
    except TurnFenceRollbackRefused:
        return False
    return current == authority


class RollbackOutcome:
    """What the run ESTABLISHED, kept where a failure cannot take it back.

    A report returned at the end cannot describe a run that did not reach the
    end. So the caller owns this object and reads it on every path, including
    the ones where the call raised.

    ``changed`` IS THREE-STATE, and that is the point. ``False`` before a
    commit is attempted, ``True`` once it returned, and ``None`` while it is
    genuinely unknown. In this build every path leaves it ``False``, because
    every path refuses before a commit is attempted.
    """

    def __init__(self) -> None:
        self.outcome = "not-started"
        self.changed = False
        self.backup_created = False
        self.backup_verified = False
        self.backup_durable = False
        # HISTORY above, PRESENCE here, and they are not the same question.
        self.backup_present = False
        self.backup_unlinked_by_this_run = False
        self.backup_absence_durable = False
        self.backup_withdrawn = False
        # A LIST, and never assigned over.
        self.residue = []
        self.backup = None

    @property
    def residue_present(self) -> bool:
        """Derived, so it cannot disagree with what was actually recorded."""
        return bool(self.residue)

    def note_residue(self, record: dict, *, incident: str) -> None:
        """Record ONE unresolved incident. Idempotent per incident.

        Unreachable in this build — nothing here creates anything that could
        leave residue — kept because :class:`RollbackOutcome` is the shared
        bookkeeping shape a future authority-bearing run would still use.
        """
        if record is None:
            return
        entry = dict(record, incident=incident)
        for position, existing in enumerate(self.residue):
            if existing.get("incident") == incident:
                self.residue[position] = entry
                return
        self.residue.append(entry)

    def facts(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "changed": self.changed,
            "backup_created": self.backup_created,
            "backup_verified": self.backup_verified,
            "backup_durable": self.backup_durable,
            "backup_present": self.backup_present,
            "backup_unlinked_by_this_run": self.backup_unlinked_by_this_run,
            "backup_absence_durable": self.backup_absence_durable,
            "backup_withdrawn": self.backup_withdrawn,
            "residue_present": bool(self.residue),
            "residue": list(self.residue),
            "backup": self.backup,
        }


def rehearse_turn_fence_rollback(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path = None,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """Refuses before it derives anything. There is no rehearsal to report.

    Refuses, before a working directory is created, before the source is
    opened, before any image is taken. What comes back is a refusal and the
    reason for it, and the source is byte-for-byte and file-for-file as it was
    found. *work_dir* is accepted and unused: nothing here creates one.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    outcome = outcome if outcome is not None else RollbackOutcome()

    # BEFORE ANYTHING IS CREATED, OPENED OR READ.
    disqualify_the_target(store_path)
    establish_offline_authority(store_path)
    raise AssertionError(  # pragma: no cover - establish_offline_authority raises
        "establish_offline_authority returned; it has no success path"
    )


def rollback_turn_fence(
    store_path: Path,
    *,
    backup_path: Path,
    work_dir: Path = None,
    outcome: "RollbackOutcome" = None,
) -> dict[str, Any]:
    """Remove the turn-fence triggers from a DETACHED artifact, offline.

    OFFLINE IS PROVEN, NOT INFERRED, AND IN THIS BUILD IT CANNOT BE PROVEN
        The pre-flight a real rollback would need refuses a store that records
        a live turn and refuses one it cannot lock, and neither of those is
        evidence that nothing is attached. What permits the operation is
        :func:`establish_offline_authority`, and it refuses every artifact —
        see its docstring for the measurement. So this function has no
        reachable success path today: it refuses BEFORE IT OBSERVES THE SOURCE
        AT ALL. ``disqualify_the_target`` and ``establish_offline_authority``
        are the only two statements that run; the second always raises.
        Nothing else runs: no pre-flight, no copy, no SQLite handle on the
        store, no working directory. That is deliberate and it is the
        acceptance property — opening the source is not free, because it can
        leave ``-wal``/``-shm`` beside the artifact, and a refusal that changed
        the directory it is about to say it left alone is not a refusal.

    Raises :class:`TurnFenceRollbackRefused` on every outcome, leaving the
    store byte-identical — including its directory listing, because nothing
    here opens it. *outcome* is the caller's, so what a run established is
    readable even on the paths where this raises. *backup_path* and
    *work_dir* are accepted for signature compatibility with the future
    authority-bearing implementation; neither is examined or used while this
    build fails closed.
    """
    store_path = Path(store_path)
    backup_path = Path(backup_path)
    outcome = outcome if outcome is not None else RollbackOutcome()

    # BEFORE ANYTHING IS OPENED, COPIED OR CREATED. Under this contract the
    # answer is always "no", so every byte written, every SQLite handle taken
    # and every duplicate of the store made on the way to it is work done for a
    # run that cannot proceed.
    disqualify_the_target(store_path)
    establish_offline_authority(store_path)
    raise AssertionError(  # pragma: no cover - establish_offline_authority raises
        "establish_offline_authority returned; it has no success path"
    )
