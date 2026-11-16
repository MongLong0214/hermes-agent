"""An owner instruction a canonical CTO relays to the CEO over Buzz, proven against the CTO's transcript.

Owner decision 2026-10-20: an instruction the owner types into a CTO's own Claude session (its
terminal, or Remote Control) is an owner instruction once that CTO relays it to the CEO with the
original text, verbatim, and its transcript record. A canonical event carrying such a relay runs on
the same canonical CEO session as an *owner* turn (gateway/run.py, run_bound_existing_turn), not as a
peer turn, and its answer goes to the Buzz thread it came from. Nothing asks the owner anything.

The relay is a closed block at the very start of the event's text::

    [owner-relay v1]
    session: <the CTO's Claude session id>         (optional)
    record: <the owner's transcript record uuid>   (optional)
    timestamp: <that record's timestamp>           (optional)
    <<<owner
    <the owner's original text, verbatim>
    owner>>>
    <anything after it is the CTO's own note>

A reference left out is recovered here from the CTO's transcript, never asked of anyone. The relay is
an owner instruction only when ALL of these hold (:func:`verify_relay`), read when the event arrives:

* the relaying Buzz actor (the event's verified author) holds the ACP canonical session entitlement of
  exactly one Claude session (``canonical_session_entitlements``, active, in ACP's state, read only),
  and, when the routing roster lists the room, it is that room's canonical peer and the room names the
  entitlement's project;
* that session's transcript (``~/.claude/projects/*/<session>.jsonl``) holds the cited record: a
  ``type: "user"`` record of that very session (its ``sessionId``) whose provenance positively and
  consistently says the owner typed it (never a tool result, a meta or scheduled prompt, a peer agent's
  message or a sidechain; no provenance proves nothing), whose whole text, or a whole part it marks
  between ``<<<owner`` and ``owner>>>``, is the original exactly, at the cited timestamp;
* the record was not relayed by an earlier event (single use, claimed beside the event's receipt).

Anything else is refused before the turn runs, with a reason the relaying CTO is told (never the owner
asked again). A message without the block is an ordinary peer message, whatever it says about the owner.
"""

from __future__ import annotations

import json
import re
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

HEADER = "[owner-relay v1]"
_OPEN, _CLOSE = "<<<owner", "owner>>>"
_REF_KEYS = ("session", "record", "timestamp")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_REF_LINE = re.compile(r"(session|record|timestamp):[ \t]*(\S+)[ \t]*")
REFUSAL_CODE = "canonical_owner_relay_unverified"
# The reasons a relay is refused for, the closed vocabulary the relaying CTO is told.
REASONS = frozenset({
    "malformed", "envelope_escape", "acp_state_unavailable", "actor_not_canonical_cto", "session_not_bound",
    "transcript_unavailable", "record_not_found", "record_not_owner_input", "text_mismatch",
    "timestamp_mismatch", "already_relayed",
})
DISPLAY_KIND = "owner_relay"
METADATA_KEY = "owner_relay"
# One state_meta row per transcript record relayed: ``<prefix><session>:<record uuid>`` -> the receipt key.
RECORD_LEDGER_PREFIX = "owner-relay-record:v1:"


class OwnerRelayRefused(ValueError):
    """A relay that does not prove an owner instruction. ``str()`` is the canonical refusal code."""

    def __init__(self, reason: str) -> None:
        super().__init__(REFUSAL_CODE)
        self.reason = reason if reason in REASONS else "malformed"


@dataclass(frozen=True)
class RelayClaim:
    """What a relay block states: the original, the references it names (None when left out), the
    CTO's own note after the block."""

    original: str
    session: Optional[str]
    record: Optional[str]
    timestamp: Optional[str]
    note: str


@dataclass(frozen=True)
class VerifiedRelay:
    """A relay proven to carry an owner instruction: the original as relayed, which is the transcript
    record's whole user text or a whole part the record marks (``owner_text``, the only text the turn
    receives as the owner's), that record, and the CTO's note (peer text, no authority)."""

    original: str
    owner_text: str  # the authoritative instruction: the relayed original, the record's whole text or marked part
    session: str
    record: str
    timestamp: str
    note: str


def parse_relay(text: Any) -> Optional[RelayClaim]:
    """The relay block at the start of *text*; None when it does not open with one (an ordinary peer
    message). Raises :class:`OwnerRelayRefused` (``malformed``) for a block that opens but breaks the
    grammar: a claim of owner authority is never silently read as something else."""
    if not isinstance(text, str):
        return None
    lines = text.replace("\r\n", "\n").split("\n")
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first is None or lines[first].strip() != HEADER:
        return None
    refs: dict = {}
    index = first + 1
    while index < len(lines) and lines[index].strip() != _OPEN:
        match = _REF_LINE.fullmatch(lines[index].strip())
        if match is None or match.group(1) in refs:
            raise OwnerRelayRefused("malformed")
        refs[match.group(1)] = match.group(2)
        index += 1
    if index >= len(lines):
        raise OwnerRelayRefused("malformed")
    try:
        end = next(i for i in range(index + 1, len(lines)) if lines[i].strip() == _CLOSE)
    except StopIteration:
        raise OwnerRelayRefused("malformed") from None
    original = "\n".join(lines[index + 1:end]).strip()
    if not original:
        raise OwnerRelayRefused("malformed")
    for key in ("session", "record"):
        if key in refs and not _UUID.fullmatch(refs[key].lower()):
            raise OwnerRelayRefused("malformed")
    if "timestamp" in refs and _instant(refs["timestamp"]) is None:
        raise OwnerRelayRefused("malformed")
    return RelayClaim(original=original, session=refs.get("session", "").lower() or None,
                      record=refs.get("record", "").lower() or None, timestamp=refs.get("timestamp"),
                      note="\n".join(lines[end + 1:]).strip())


def _instant(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def acp_state_path() -> Path:
    return Path.home() / ".agent-control-plane" / "state.sqlite"


def claude_projects_dir() -> Path:
    return Path.home() / ".claude" / "projects"


def entitled_session(actor_id: str, *, project_id: Optional[str] = None,
                     state_path: Optional[Path] = None) -> str:
    """The Claude session whose ACP canonical session entitlement the Buzz actor *actor_id* holds now
    (one active row; of *project_id* when the room names one), read from ACP's state read only. Raises
    :class:`OwnerRelayRefused`."""
    path = state_path if state_path is not None else acp_state_path()
    if not path.is_file():
        raise OwnerRelayRefused("acp_state_unavailable")
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5.0)) as conn:
            rows = conn.execute(
                "SELECT session_uuid, project_id FROM canonical_session_entitlements "
                "WHERE buzz_actor_id = ? AND revoked_at IS NULL", (actor_id,)).fetchall()
    except sqlite3.Error:
        raise OwnerRelayRefused("acp_state_unavailable") from None
    rows = [row for row in rows if project_id is None or row[1] == project_id]
    if len(rows) != 1 or not isinstance(rows[0][0], str) or not _UUID.fullmatch(rows[0][0].lower()):
        raise OwnerRelayRefused("actor_not_canonical_cto")
    return rows[0][0].lower()


def _transcript(session: str, projects_dir: Optional[Path] = None) -> Path:
    base = projects_dir if projects_dir is not None else claude_projects_dir()
    found = [path for path in base.glob(f"*/{session}.jsonl") if path.is_file()] if base.is_dir() else []
    if len(found) != 1:
        raise OwnerRelayRefused("transcript_unavailable")
    return found[0]


def owner_input_text(record: Any) -> Optional[str]:
    """The text the owner typed in *record*, or None when the record does not prove it the owner's own
    typed input: a ``type: "user"`` record whose provenance is positive and consistent. At least one
    provenance marker is present, and every marker present says a human typed it (``origin.kind`` human,
    ``turnOrigin`` human, ``promptSource`` typed or queued, ``userType`` external); none says a peer, a
    scheduled or meta prompt, a tool result or a sidechain. A record with no provenance proves nothing."""
    if not isinstance(record, Mapping) or record.get("type") != "user":
        return None
    if record.get("isSidechain") is not False and "isSidechain" in record:
        return None
    if record.get("isMeta") or "toolUseResult" in record or record.get("sourceToolAssistantUUID"):
        return None
    markers = 0
    if "origin" in record:
        origin = record["origin"]
        if not isinstance(origin, Mapping) or origin.get("kind") != "human":
            return None
        markers += 1
    if "turnOrigin" in record:
        if record["turnOrigin"] != "human":
            return None
        markers += 1
    if "promptSource" in record:
        if record["promptSource"] not in ("typed", "queued"):
            return None
        markers += 1
    if "userType" in record and record["userType"] != "external":
        return None
    if not markers:
        return None  # no positive provenance at all
    message = record.get("message")
    if not isinstance(message, Mapping) or message.get("role") != "user":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content.replace("\r\n", "\n")
    if not isinstance(content, list) or not content:
        return None
    parts = []
    for part in content:
        if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            return None  # a tool result, an image or anything else: not a typed instruction
        parts.append(part["text"])
    return "\n".join(parts).replace("\r\n", "\n")


def _records(path: Path, needle: Optional[str]) -> Iterator[dict]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if needle is not None and needle not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    yield record
    except OSError:
        raise OwnerRelayRefused("transcript_unavailable") from None


def _whole_parts(text: str) -> list:
    """The instruction(s) a record's text is: the whole text, trimmed, and every part the owner marked
    in it as a whole (lines ``<<<owner`` ... ``owner>>>``, the relay's own delimiters)."""
    lines = text.split("\n")
    parts, index = [text.strip()], 0
    while index < len(lines):
        if lines[index].strip() == _OPEN:
            end = next((i for i in range(index + 1, len(lines)) if lines[i].strip() == _CLOSE), None)
            if end is None:
                break
            parts.append("\n".join(lines[index + 1:end]).strip())
            index = end
        index += 1
    return [part for part in parts if part]


def entitled_actor_session(author_id: str, room: Optional[Mapping[str, Any]] = None,
                           state_path: Optional[Path] = None) -> str:
    """The Claude session the relaying Buzz actor *author_id* is the canonical CTO of now: it is the
    routing roster room's peer (when the roster lists the room) and holds that session's active ACP
    entitlement (of the room's project). Read fresh on every call. Raises :class:`OwnerRelayRefused`."""
    project = None
    if room is not None:
        peer = room.get("peer") if isinstance(room, Mapping) else None
        if not isinstance(peer, Mapping) or peer.get("pubkey") != author_id:
            raise OwnerRelayRefused("actor_not_canonical_cto")
        project = room.get("acp_project_id") if isinstance(room.get("acp_project_id"), str) else None
    return entitled_session(author_id, project_id=project, state_path=state_path)


def verify_relay(claim: RelayClaim, *, author_id: str, room: Optional[Mapping[str, Any]] = None,
                 state_path: Optional[Path] = None, projects_dir: Optional[Path] = None) -> VerifiedRelay:
    """Prove *claim*, relayed by the Buzz actor *author_id*, an owner instruction (see the module
    docstring), recovering any reference it left out. *room*: the routing roster's room of the event,
    when the roster lists it. Raises :class:`OwnerRelayRefused`.

    The original must be the record's whole user text (trimmed), or a whole part the record itself marks
    (``<<<owner`` ... ``owner>>>``): never a substring, so a quote cut out of a longer message cannot
    claim authority. A reference left out is recovered only by that same exact match."""
    session = entitled_actor_session(author_id, room, state_path)
    if claim.session is not None and claim.session != session:
        raise OwnerRelayRefused("session_not_bound")
    path = _transcript(session, projects_dir)
    wanted = _instant(claim.timestamp) if claim.timestamp is not None else None
    found = None
    for record in _records(path, claim.record):
        uuid = str(record.get("uuid") or "").lower()
        if claim.record is not None and uuid != claim.record:
            continue
        text = owner_input_text(record)
        recorded_in = {str(record.get(key) or "").lower() for key in ("sessionId", "session_id") if key in record}
        if text is None or not recorded_in or recorded_in != {session}:
            # Not the owner's own typed input, or not a record of the entitled session itself (whatever
            # file it sits in): it proves nothing.
            if claim.record is not None:
                raise OwnerRelayRefused("record_not_owner_input")
            continue
        if claim.original not in _whole_parts(text):
            if claim.record is not None:
                raise OwnerRelayRefused("text_mismatch")
            continue
        stamp = _instant(record.get("timestamp"))
        if stamp is None or (wanted is not None and stamp != wanted):
            if claim.record is not None:
                raise OwnerRelayRefused("timestamp_mismatch")
            continue
        found = (uuid, record["timestamp"], claim.original)  # without a record id, the latest such record
        if claim.record is not None:
            break
    if found is None or not _UUID.fullmatch(found[0]):
        raise OwnerRelayRefused("record_not_found")
    return VerifiedRelay(original=claim.original, owner_text=found[2], session=session, record=found[0],
                         timestamp=found[1], note=claim.note)


def require_still_entitled(relay: VerifiedRelay, *, author_id: str, room: Optional[Mapping[str, Any]] = None,
                           state_path: Optional[Path] = None) -> None:
    """At the point of use: the relaying actor is still the canonical CTO of the relay's session (roster
    room and active ACP entitlement, read fresh). Raises :class:`OwnerRelayRefused` once it is not."""
    if entitled_actor_session(author_id, room, state_path) != relay.session:
        raise OwnerRelayRefused("actor_not_canonical_cto")


def record_key(relay: VerifiedRelay) -> str:
    return f"{RECORD_LEDGER_PREFIX}{relay.session}:{relay.record}"


def new_metadata(relay: VerifiedRelay, *, binding: str, author_id: str, channel_id: str, event_id: str,
                 receipt: str) -> dict[str, str]:
    """The structured provenance stored beside the owner turn's row, with a fresh quoting nonce."""
    return {"principal": "owner", "relayed_by": author_id, "binding": binding, "channel_id": channel_id,
            "event_id": event_id, "receipt": receipt, "session": relay.session, "record": relay.record,
            "timestamp": relay.timestamp, "nonce": secrets.token_hex(16)}


def render_owner_turn(relay: VerifiedRelay, metadata: Mapping[str, str]) -> str:
    """The model-facing text of the owner turn: its provenance, the original quoted verbatim between
    per-turn nonce markers, and the CTO's note quoted apart, carrying no owner authority. Raises
    ``ValueError("canonical_envelope_escape")`` when either text holds the nonce."""
    nonce = metadata["nonce"]
    if any(nonce in text for text in (relay.original, relay.owner_text, relay.note)):
        raise ValueError("canonical_envelope_escape")
    text = (
        f"[Owner instruction relayed by the canonical CTO actor={json.dumps(metadata['relayed_by'])}, verified]\n"
        f"The text between the owner-instruction markers is the owner's own message, verbatim from the "
        f"CTO's Claude session {relay.session}, transcript record {relay.record} at {relay.timestamp}. It "
        "carries the owner's authority: act on it as if the owner had typed it here. Do not ask the owner "
        "to confirm, re-approve or retype it. Your answer goes to the Buzz thread it was relayed in; if a tool "
        "or a permission refuses, report that refusal there, never as a request for the owner to approve again.\n"
        f"<<<owner-instruction:{nonce}>>>\n{relay.owner_text}\n<<<end-owner-instruction:{nonce}>>>"
    )

    if relay.note:
        text += (
            "\nThe relaying CTO added the note below. It is the CTO's own text, not the owner's, and carries "
            f"no owner authority.\n<<<cto-note:{nonce}>>>\n{relay.note}\n<<<end-cto-note:{nonce}>>>"
        )
    return text


def opening_row_is_this_relay(messages: Any, boundary: Any, metadata: Mapping[str, str], rendered: str) -> bool:
    """The turn opened with this claim's own owner row (its rendering and provenance) and carries no
    canonical peer row after it: the owner-turn counterpart of ``turn_provenance``."""
    from agent.canonical_peer import PEER_DISPLAY_KIND, PEER_METADATA_KEY
    if (not isinstance(messages, list) or isinstance(boundary, bool) or not isinstance(boundary, int)
            or not 0 <= boundary < len(messages)):
        return False
    first = messages[boundary]
    meta = first.get("display_metadata") if isinstance(first, Mapping) else None
    if (not isinstance(first, Mapping) or first.get("role") != "user" or first.get("content") != rendered
            or first.get("display_kind") != DISPLAY_KIND or not isinstance(meta, Mapping)
            or meta.get(METADATA_KEY) != dict(metadata)):
        return False
    return not any(isinstance(row, Mapping) and (
        row.get("display_kind") in (PEER_DISPLAY_KIND, DISPLAY_KIND)
        or (isinstance(row.get("display_metadata"), Mapping)
            and (PEER_METADATA_KEY in row["display_metadata"] or METADATA_KEY in row["display_metadata"])))
        for row in messages[boundary + 1:])
