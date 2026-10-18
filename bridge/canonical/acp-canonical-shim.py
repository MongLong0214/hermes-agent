#!/usr/bin/env python3
"""ACP-stdio agent that submits to Hermes's canonical surface instead of forking a session.

`buzz-acp` speaks ACP over stdio to whatever `--agent-command` names. Pointing it at
`hermes acp` starts a *new* Hermes session, which is the fork the owner prohibited: the CEO's
own transcript — 3,788 sessions — is not what answers.

This stands in that position and forwards each prompt to
`POST /v1/canonical-surface/events`, whose `CanonicalReceiptCoordinator` hands the turn to the
cached actor already bound to `canonical_surface_bindings.<binding>`. A fork is not merely
avoided here, it is unreachable: this process owns no session and can create none.

The endpoint answers the turn on the same connection and keeps a durable receipt, so a resend of
one `event_id` replays that receipt rather than running a second turn.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = os.environ.get("CANONICAL_SURFACE_URL", "http://127.0.0.1:8642")
ROUTE = "/v1/canonical-surface/events"
BINDING = os.environ.get("CANONICAL_BINDING", "acp-canonical-ceo")
AUTHOR = os.environ.get("CANONICAL_AUTHOR_ID", "")
CHANNEL = os.environ.get("CANONICAL_CHANNEL_ID", "")
KEY = os.environ.get("API_SERVER_KEY", "")
# The endpoint runs a full model turn before it answers, so this budget is a turn's, not an RPC's.
TIMEOUT = float(os.environ.get("CANONICAL_TIMEOUT_SECS", "900"))
LOG = os.environ.get("CANONICAL_SHIM_LOG", os.path.expanduser("~/.hermes/logs/acp-canonical-shim.log"))
# 2026-09-27, CEO 지시로 기본값이 quarantine 이다. 이 binding 은 session_key
# agent:main:telegram:dm:<owner-dm-id> 로 해석되므로, 전달된 CTO 이벤트는 CEO 세션에 owner 의
# 텔레그램 턴으로 저장된다. run_bound_existing_turn 은 event.text 만 넘기고 user role 로 적으므로
# 서명된 Buzz pubkey 가 본문과 분리되지 않는다 — 접두사나 heartbeat=0 은 방어가 아니다. 서명자
# 분리가 게이트웨이에 들어오기 전까지 이 자리는 전달하지 않고, 사건을 증거로 남기기만 한다.
FORWARD = os.environ.get("CANONICAL_FORWARD", "quarantine").strip().lower()
# #627 canary (CEO 04f3035f, 2026-10-04): with the default quarantine left in place, exactly one event
# may be forwarded — the first one whose own signer, channel and body nonce match what this file
# arms, before its expiry. The claim renames the file to ``.claimed`` before submitting, so nothing is
# forwarded twice and every other event stays quarantined. Disarm by deleting the file.
CANARY = os.environ.get("CANONICAL_CANARY_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                               "canary.json"))
QUARANTINE = os.environ.get(
    "CANONICAL_QUARANTINE_LOG", os.path.expanduser("~/.hermes/logs/canonical-quarantine.jsonl"))

# #627 canary 1 (2026-10-04) reached the CEO and was answered, but nothing appeared in the thread:
# buzz-acp never publishes an agent's message chunks (agents post with `buzz messages send` themselves)
# and the CEO's canonical turn is a peer turn that only returns text. The shim posts that text as the
# reply through the mention-enforce wrapper rather than buzz-real (its signer, roster and recipient guards
# stay in force), signed
# with the key inherited from buzz-acp. One post per event id, ever: a marker is claimed before the
# send, and a failed or refused send is logged, never retried.
BUZZ_CLI = os.environ.get("CANONICAL_BUZZ_CLI", os.path.expanduser("~/.hermes/bridge/bin/buzz"))
MENTIONS = os.environ.get("CANONICAL_MENTIONS_FILE", os.path.expanduser("~/.hermes/bridge/required-mentions.json"))
PUBLISHED = os.environ.get("CANONICAL_PUBLISHED_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "published"))
PUBLISH_TIMEOUT = float(os.environ.get("CANONICAL_PUBLISH_TIMEOUT_SECS", "60"))

_out = threading.Lock()


def _canary_matches(armed, author, channel, body):
    try:
        live = time.time() < float(armed.get("expires_at", 0))
    except (TypeError, ValueError):
        return False
    nonce = armed.get("nonce") or ""
    return (isinstance(armed, dict) and armed.get("state") == "armed" and live and len(nonce) >= 16
            and nonce in body and author == armed.get("author_id") and channel == armed.get("channel_id"))


def claim_canary(author, channel, body):
    """True for the first event whose signer, channel (from its own header) and body nonce match the
    armed file before expiry — once. The claim is the atomic rename of ``canary.json`` itself, so a
    disarm (deleting it) that lands first wins; the renamed bytes are validated again after the claim,
    so a replacement or an expiry that landed in between refuses rather than forwarding."""
    try:
        with open(CANARY) as handle:
            if not _canary_matches(json.load(handle), author, channel, body):
                return False
    except (OSError, ValueError, AttributeError):
        return False
    # This claimant's own path: an overlapping claimant renames into a different one, so each
    # validates exactly the bytes it took.
    claimed = f"{CANARY}.claimed.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}"
    try:
        os.rename(CANARY, claimed)
    except OSError:
        return False
    try:
        with open(claimed) as handle:
            if not _canary_matches(json.load(handle), author, channel, body):
                return False
    except (OSError, ValueError, AttributeError):
        return False
    # At most one forward per canary, whatever is re-armed later: the first fire wins this marker.
    try:
        fd = os.open(CANARY + ".fired", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except OSError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"claimed": os.path.basename(claimed),
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    return True


def log(message):
    try:
        with open(LOG, "a") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
    except OSError:
        pass


def write(payload):
    line = json.dumps(payload, ensure_ascii=False)
    with _out:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def result(rpc_id, value):
    write({"jsonrpc": "2.0", "id": rpc_id, "result": value})


def error(rpc_id, code, message):
    write({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})


# `[Buzz event: ...]` states who wrote the message and where. The binding's principal check reads
# `author_id`/`channel_id`, so sending this process's own env constants would make that check
# vacuous: every author buzz-acp forwards — owner, orchestrator, anyone later added to the
# allowlist — would arrive at the CEO as the one identity named in the environment. The values are
# read from the event only; the env values are a startup requirement and never stand in for them.
AUTHOR_RE = re.compile(r"hex:\s*([0-9a-f]{64})")
CHANNEL_RE = re.compile(r"#([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})")
EVENT_RE = re.compile(r"Event ID:\s*([0-9a-f]{64})")

# The gateway refuses a turn outright while the bound session is running
# (`canonical_surface.py` raises `canonical_turn_busy`); it never queues. The CEO works for hours,
# so a single attempt drops the message. Retry until buzz-acp's own turn budget is nearly spent.
BUSY_RETRY_SECS = float(os.environ.get("CANONICAL_BUSY_RETRY_SECS", "20"))
BUSY_DEADLINE_SECS = float(os.environ.get("CANONICAL_BUSY_DEADLINE_SECS", "840"))


EVENT_HEADER = "[Buzz event:"
_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
EVENT_TITLE_RE = re.compile(r"\[Buzz event: [^\[\]\n]+\]")
HEADER_EVENT_RE = re.compile(r"Event ID: ([0-9a-f]{64})")
HEADER_CHANNEL_RE = re.compile(r"Channel: [^()#]* \(#(" + _UUID + r")\)")
HEADER_KIND_RE = re.compile(r"Kind: \d{1,10}")
HEADER_FROM_RE = re.compile(r"From: [^()]* \(npub: npub1[02-9ac-hj-np-z]+, hex: ([0-9a-f]{64})\)")
HEADER_TIME_RE = re.compile(r"Time: [0-9][0-9T:.+\-Z]{0,40}")
# What buzz-acp can write on each header line, whatever a channel name or a label holds — wide enough to
# match every header it produces (queue.rs ``format_event_block``: the Channel field ends ``(#<uuid>)``, a
# label is stripped of control characters by ``sanitize_prompt_label``, Time is RFC 3339 or bare seconds),
# so a second reading of the header lines is never missed. They only rule a reading out; nothing is read
# from them.
ANY_CHANNEL_END_RE = re.compile(r".* \(#" + _UUID + r"\)")
ANY_KIND_RE = re.compile(r"Kind: \d+")
ANY_FROM_RE = re.compile(r"From: .*hex: [0-9a-f]{64}\)")
ANY_TIME_RE = re.compile(r"Time: \S+")


def prompt_blocks(prompt):
    """The text of each element of ACP ``params.prompt``, in order; None for one that is not a text
    block. buzz-acp sends each prompt section as its own element (queue.rs ``format_prompt``, acp.rs
    ``build_prompt_params``), so a message body always lies inside one element and can never open
    another: the element boundary is the one delimiter no body can write."""
    if not isinstance(prompt, list):
        return []
    return [block["text"] if isinstance(block, dict) and block.get("type") == "text"
            and isinstance(block.get("text"), str) else None for block in prompt]


def _header_readings(lines):
    """Every offset j at which ``lines[2 + j:7 + j]`` could be buzz-acp's own header — the last line of
    the Channel field, then Kind, From, Time and Content — under the widest forms the producer writes.

    The channel name is the one header field buzz-acp writes unsanitized (pool.rs takes it from the
    channel's ``name`` tag), so it can hold newlines, and with them whole header lines of its own: the
    Kind, From and Time lines the producer writes then come after them, inside what would read as the
    message. The producer's own header is always among these readings, at j = its name's newline count,
    so when 0 is the only one the fixed positions are the producer's; any other reading means the
    boundary it drew cannot be told from the element, and nothing is attributed."""
    return [j for j in range(len(lines) - 6)
            if lines[6 + j].startswith("Content:") and ANY_KIND_RE.fullmatch(lines[3 + j])
            and ANY_TIME_RE.fullmatch(lines[5 + j]) and ANY_FROM_RE.fullmatch(lines[4 + j])
            and ANY_CHANNEL_END_RE.fullmatch(lines[2 + j])]


def envelope(prompt):
    """(author_id, channel_id, event_id, body) of a single-event prompt, or None.

    In buzz-acp's single-event shape the LAST element of ``params.prompt`` is the event section:
    ``[Buzz event: <tag>]`` and then ``format_event_block``'s header lines at fixed positions —
    Event ID, Channel, Kind, From, Time, Content (queue.rs 1081-1147, 1533-1537). Only that element
    is read, only at those positions, and only when it starts with ``[Buzz event: ``. A merged
    (cancel + re-prompt) or batched prompt ends in another element, so it is never taken for a single
    event whatever its bodies quote: no body is searched for a header to promote, and no identity
    stands in for a missing one. A body line opening another event section makes the element
    ambiguous: None. So does a second reading of the header lines (``_header_readings``): a channel
    name carrying a line break and header lines of its own is never taken for the producer's header,
    and neither the channel name, a display name nor a body is ever matched against anything to
    vouch for one."""
    blocks = prompt_blocks(prompt)
    if not blocks or blocks[-1] is None or not blocks[-1].startswith(EVENT_HEADER + " "):
        return None
    # Exactly one event element, right after buzz-acp's own [Context] or its quoted conversation history
    # (queue.rs ``format_prompt``): a second event element, or any element between those and the event,
    # is no single-event prompt buzz-acp writes.
    if sum(isinstance(block, str) and block.startswith(EVENT_HEADER) for block in blocks) != 1:
        return None
    if len(blocks) < 2 or not (isinstance(blocks[-2], str)
                               and (blocks[-2].startswith("[Context]\n") or blocks[-2].startswith(CONVERSATION_HEADERS))):
        return None
    lines = blocks[-1].split("\n")
    if len(lines) < 7 or not EVENT_TITLE_RE.fullmatch(lines[0]):
        return None
    if sum(line.startswith(EVENT_HEADER) for line in lines) != 1:
        return None
    event = HEADER_EVENT_RE.fullmatch(lines[1])
    channel = HEADER_CHANNEL_RE.fullmatch(lines[2])
    author = HEADER_FROM_RE.fullmatch(lines[4])
    # buzz-acp never puts a second ``hex:`` on the From line (CANONICAL-01): a display name quoting
    # one makes the line ambiguous on its own, whichever occurrence the pattern would capture.
    if not (event and channel and author and lines[4].count("hex:") == 1
            and HEADER_KIND_RE.fullmatch(lines[3]) and HEADER_TIME_RE.fullmatch(lines[5])
            and lines[6].startswith("Content:")):
        return None
    if _header_readings(lines) != [0]:
        return None
    body = "\n".join([lines[6][len("Content:"):]] + lines[7:]).strip()
    return author.group(1), channel.group(1), event.group(1), body


CONTEXT_THREAD_ROOT_RE = re.compile(r"Thread root: ([0-9a-f]{64})")
CONTEXT_CHANNEL_HINT = "Hint: Use `buzz messages get --channel <UUID>` for recent messages if needed."
CONTEXT_NEW_THREAD_RE = re.compile(
    r"IMPORTANT: This is a new top-level message\. For ordinary replies in this turn, use `--reply-to "
    r"([0-9a-f]{64})` on `buzz messages send` — the triggering message is the thread root\. Do NOT reply "
    r"into any other \(older\) thread\. If the human explicitly asks for a channel-root, top-level, or "
    r"broadcast post, send that message without `--reply-to`\.")
CONTEXT_PARENT_RE = re.compile(r"Parent: ([0-9a-f]{64})")
CONTEXT_THREAD_HINTS = (
    "Thread context included below. Use `buzz messages thread --channel <UUID> --event <ID>` for full history "
    "if truncated.",
    "Use `buzz messages thread --channel <UUID> --event <ID>` to fetch thread context.")
CONTEXT_THREAD_ANCHOR_RE = re.compile(
    r"IMPORTANT: For ordinary replies in this turn, use `--reply-to ([0-9a-f]{64})` on `buzz messages send` so "
    r"the conversation stays threaded\. If the human explicitly asks for a channel-root, top-level, or broadcast "
    r"post, send that message without `--reply-to`\. If the requested destination is ambiguous, ask before "
    r"sending\.")
# The closing element of a merged prompt and the header of the element before it, which holds the events that
# arrived meanwhile (queue.rs ``MergeFraming``, 1596-1612); a batch's header (1539-1549).
CLOSING_NOTES = (
    "Note: A new message arrived while you were working. Continue your in-progress work and incorporate the new "
    "message if it's relevant; if it's unrelated, you may briefly acknowledge it and carry on.",
    "Note: The previous request was interrupted. Please address the new request.\nIf the new request is unrelated "
    "to the previous one, you may briefly acknowledge the interruption.")
NEW_EVENT_SINGLE = ("[New message — arrived while you were working]", "[New request — supersedes previous]")
NEW_EVENT_HEADERS = NEW_EVENT_SINGLE + ("[New messages — arrived while you were working",
                                        "[New request — supersedes previous")
EVENT_COUNT_RE = re.compile(r"(?:\[Buzz events|\[New messages — arrived while you were working"
                            r"|\[New request — supersedes previous) — (\d{1,6}) events\]")


def _context_form(lines):
    """(scope, root, parent, anchor) when ``lines[3:]`` of a ``[Context]`` element are exactly buzz-acp's
    thread or channel form (queue.rs 1285-1318), else None. Every line is held to its form and the count
    is exact."""
    if lines[1] == "Scope: thread":
        root, rest = CONTEXT_THREAD_ROOT_RE.fullmatch(lines[3]), lines[4:]
        parent = CONTEXT_PARENT_RE.fullmatch(rest[0]) if rest else None
        rest = rest[1:] if parent else rest
        if root is None or not rest or rest[0] not in CONTEXT_THREAD_HINTS or len(rest) > 2:
            return None
        anchor = CONTEXT_THREAD_ANCHOR_RE.fullmatch(rest[1]) if len(rest) == 2 else None
        if len(rest) == 2 and anchor is None:
            return None
        return ("thread", root.group(1), parent.group(1) if parent else None,
                anchor.group(1) if anchor else None)
    if lines[1] == "Scope: channel" and lines[3] == CONTEXT_CHANNEL_HINT and len(lines) in (4, 5):
        anchor = CONTEXT_NEW_THREAD_RE.fullmatch(lines[4]) if len(lines) == 5 else None
        if len(lines) == 5 and anchor is None:
            return None
        return "channel", None, None, anchor.group(1) if anchor else None
    return None


def _last_event_element(blocks):
    """The element holding the event buzz-acp built ``[Context]`` from — its batch's last — told by the
    prompt's layout alone: the last element, or the one before a merged prompt's own closing note."""
    if len(blocks) >= 2 and blocks[-1] in CLOSING_NOTES:
        element, heads = blocks[-2], NEW_EVENT_HEADERS
    else:
        element, heads = blocks[-1] if blocks else None, (EVENT_HEADER + " ", "[Buzz events — ")
    return element.split("\n") if isinstance(element, str) and element.startswith(heads) else None


def _tags_at_end(lines):
    """The tags of the element's last event, from the ``Tags:`` line buzz-acp writes after its message and
    before its optional one-line ``Parsed:`` (queue.rs 1116-1144): the end of an element is the producer's,
    since no message can close one. None when that line is not there in that form."""
    tail = lines[-2] if len(lines) >= 2 and lines[-1].startswith("Parsed: ") else lines[-1]
    if not tail.startswith("Tags: "):
        return None
    try:
        tags = json.loads(tail[len("Tags: "):])
    except Exception:
        return None
    if not (isinstance(tags, list) and all(isinstance(tag, list) and all(isinstance(v, str) for v in tag)
                                           for tag in tags)):
        return None
    return tags


def _thread_of(tags):
    """(root, parent) as buzz-acp's ``parse_thread_tags`` reads NIP-10 markers (queue.rs 854-892)."""
    root = reply = None
    for tag in tags:
        if len(tag) >= 4 and tag[0] == "e":
            if tag[3] == "root":
                root = tag[1]
            elif tag[3] == "reply":
                reply = tag[1]
    if root is not None:
        return root, root if reply is None else reply
    return reply, reply


def _last_event_id(lines, channel_line):
    """The id of the element's last event where buzz-acp writes it, or None. A single event's id is at a
    fixed line. In a list of N events, the ``--- Event N (<tag>) ---`` section is taken only when one line
    alone opens it with buzz-acp's own header under it — the [Context] channel line included — since an
    earlier message could quote one; any other count is no answer."""
    if EVENT_TITLE_RE.fullmatch(lines[0]):
        at = 1
    elif lines[0] in NEW_EVENT_SINGLE:
        at = 3 if len(lines) > 3 and lines[1] == "" and FIRST_EVENT_RE.fullmatch(lines[2]) else None
    else:
        count = EVENT_COUNT_RE.fullmatch(lines[0])
        head = f"--- Event {count.group(1)} (" if count else None
        readings = [i for i in range(1, len(lines) - 6) if head and lines[i].startswith(head)
                    and lines[i].endswith(") ---") and lines[i - 1] == "" and lines[i + 2] == channel_line
                    and HEADER_EVENT_RE.fullmatch(lines[i + 1]) and ANY_KIND_RE.fullmatch(lines[i + 3])
                    and ANY_FROM_RE.fullmatch(lines[i + 4]) and ANY_TIME_RE.fullmatch(lines[i + 5])
                    and lines[i + 6].startswith("Content:")]
        at = readings[0] + 1 if len(readings) == 1 else None
    if at is None or at + 1 >= len(lines) or lines[at + 1] != channel_line:
        return None
    found = HEADER_EVENT_RE.fullmatch(lines[at])
    return found.group(1) if found else None


def _disagreement(blocks, channel_line, channel, scope, root, parent, anchor):
    """Why the ``[Context]`` destination does not agree with the event buzz-acp built it from, or None.

    buzz-acp derives [Context] from the batch's last event (queue.rs ``format_prompt``): its channel, its
    NIP-10 root and parent, and — for a top-level message — its own id as the anchor. That event's tags are
    committed by its id and written, JSON-escaped, at the end of its element, where no message can write.
    The channel must be the event's only ``h`` tag, the thread its root (and parent), and an anchor its
    root or its own id. A Context that is not this event's — whatever its form — names no destination."""
    lines = _last_event_element(blocks)
    tags = _tags_at_end(lines) if lines else None
    if tags is None:
        return "no event section of buzz-acp's ends the prompt to bind the [Context] destination to"
    if {tag[1] for tag in tags if len(tag) >= 2 and tag[0] == "h"} != {channel}:
        return "the [Context] channel is not the last event's own channel"
    event_root, event_parent = _thread_of(tags)
    if scope == "thread":
        if (event_root != root or event_parent != (parent or root) or parent == root
                or anchor not in (None, root)):
            return "the [Context] thread is not the last event's own thread"
    elif event_root is not None:
        return "a channel-scope [Context] for an event in a thread"
    elif anchor is not None and anchor != _last_event_id(lines, channel_line):
        return "the [Context] anchor is not the last event's own id"
    return None


def notice_destination(blocks):
    """(channel, reply_to, None) for an unattributed refusal's notice, or (None, None, why not).

    Read only from ``params.prompt[0]``, where buzz-acp puts its own ``[Context]`` section
    (``format_context_hints``, queue.rs 1238-1319), and only when the WHOLE element is in its exact
    thread or channel form: the thread's root, or for a top-level message the anchor buzz-acp names
    (without one, the channel itself). buzz-acp writes the channel name there unsanitized too, so a name
    carrying line breaks — and with them a ``Channel:`` or ``Thread root:`` line of its own — always
    leaves the producer's ``(#<uuid>)`` at the end of a later line that no form allows.

    Form alone does not make it this prompt's destination: the channel and thread must also agree with
    the event buzz-acp built the element from (``_disagreement``), and the caller then requires the
    channel to be registered with a counterpart. When any of it fails there is no destination, and the
    refusal is journaled as undeliverable — never sent anywhere else. It says where the conversation is,
    never who wrote what: no evidence of an author or an approval. A first element of any other shape (a
    slash command, a legacy ``[Base]``, a DM, a whole prompt sent as one element) names no destination."""
    first = blocks[0] if blocks else None
    lines = first.split("\n") if isinstance(first, str) else []
    channel = HEADER_CHANNEL_RE.fullmatch(lines[2]) if len(lines) >= 4 and lines[0] == "[Context]" else None
    if channel is None:
        return None, None, "params.prompt[0] is not buzz-acp's [Context] block"
    form = _context_form(lines)
    if form is None:
        return None, None, "the [Context] block names no thread or channel in buzz-acp's form"
    scope, root, parent, anchor = form
    why = _disagreement(blocks, lines[2], channel.group(1), scope, root, parent, anchor)
    if why is not None:
        return None, None, why
    return channel.group(1), root if scope == "thread" else anchor, None


CONVERSATION_HEADERS = ("[Thread Context (", "[Conversation Context (")
MERGE_HEADERS = ("[What you were working on]", "[Previous request — interrupted before completion]")


# The elements buzz-acp opens with an event section header and then ``--- Event 1 (<tag>) ---`` (queue.rs
# 1503-1569, ``MergeFraming``): a batch, the interrupted request's events, and what arrived meanwhile.
EVENT_LIST_HEADERS = ("[Buzz events — ",) + MERGE_HEADERS + (
    "[New message — arrived while you were working]", "[New messages — arrived while you were working",
    "[New request — supersedes previous")
FIRST_EVENT_RE = re.compile(r"--- Event 1 \([^\n]*\) ---")
EVENT_N_RE = re.compile(r"--- Event \d{1,6} \([^\n]*\) ---")
# The channel line of a [Context] element: the display name, then the channel id buzz-acp appends.
CONTEXT_CHANNEL_NAME_RE = re.compile(r"Channel: .* (\(#" + _UUID + r"\))")


def _event_ids(text):
    """The event ids an element names where buzz-acp writes them, in order, or None when it names none.

    A single-event element gives the ``Event ID:`` line right under its ``[Buzz event: <tag>]`` title. An
    event list (a batch, the interrupted request's events, what arrived meanwhile) must hold its first
    event's id where buzz-acp writes it, under ``--- Event 1 (<tag>) ---`` right after its header and a
    blank line, and gives the ``Event ID:`` line right under each ``--- Event N (<tag>) ---`` line of it.
    Nothing else of the element is read: not its header or tag, the channel name, a Kind, From, Time,
    Tags or Parsed line, or any label on them.

    An id is the event's own hash, committing its content and tags; the rest of the element is presentation
    buzz-acp renders again on every dispatch (display names: pool.rs ``fetch_prompt_profile_lookup``). A
    message body can quote an event-header line with an id under it, and that pair is taken too — but it
    is committed by the id of the event whose body holds it, which comes before it: the same events give
    the same list on every dispatch, and different events a different one. A label never holds a line
    break (``sanitize_prompt_label``) and a channel name holding one leaves the [Context] out of its form,
    so no label or rename adds, drops or changes a line of either form in a prompt a notice is posted for."""
    lines = text.split("\n")
    if EVENT_TITLE_RE.fullmatch(lines[0]):
        found = HEADER_EVENT_RE.fullmatch(lines[1]) if len(lines) > 1 else None
        return [found.group(1)] if found else None
    if not (len(lines) > 3 and lines[0].startswith(EVENT_LIST_HEADERS) and lines[1] == ""
            and FIRST_EVENT_RE.fullmatch(lines[2]) and HEADER_EVENT_RE.fullmatch(lines[3])):
        return None
    return [found.group(1) for head, line in zip(lines, lines[1:]) if EVENT_N_RE.fullmatch(head)
            for found in [HEADER_EVENT_RE.fullmatch(line)] if found]


def _without_channel_name(text):
    """A ``[Context]`` element with the channel's display name taken off its Channel line and the
    ``(#<uuid>)`` kept: buzz-acp writes the name it holds for the channel on each dispatch."""
    lines = text.split("\n")
    found = CONTEXT_CHANNEL_NAME_RE.fullmatch(lines[2]) if len(lines) > 2 and lines[0] == "[Context]" else None
    if found:
        lines[2] = "Channel: " + found.group(1)
    return "\n".join(lines)


def refusal_key(prompt):
    """The correlation key of an unattributed refusal, and None or why it is this refusal's own.

    A digest of ``params.prompt``, element by element as it was sent, less what buzz-acp renders afresh on
    every dispatch. An element that names its events contributes only their ids (``_event_ids``) and its
    fields other than the text. The quoted conversation history, fetched again each time, is left out.
    ``params.prompt[0]``, buzz-acp's [Context], is kept but for the channel's display name; every other
    element — a closing note, any other text, a non-text element with its type, data and every field —
    is kept whole. So the same events dispatched again (a requeue, a respawned shim, a refreshed profile or
    a renamed channel) give the same key and so one notice; any other difference, an image's bytes
    included, gives another key.

    Only ids buzz-acp wrote give the digest an identity. A prompt with none has nothing that tells a second
    dispatch from a second, identical message, so it gets a key of its own, as does one that cannot be
    serialized at all: that never drops a distinct input, and it does not deduplicate a redelivery."""
    elements = prompt if isinstance(prompt, list) else [prompt]
    kept, identified = [], False
    try:
        for index, element in enumerate(elements):
            if isinstance(element, dict) and element.get("type") == "text" and isinstance(element.get("text"), str):
                text = element["text"]
                if text.startswith(CONVERSATION_HEADERS):
                    continue
                ids = _event_ids(text)
                if ids is not None:
                    identified = True
                    kept.append(["events", ids, {key: value for key, value in element.items() if key != "text"}])
                    continue
                if index == 0:
                    element = dict(element, text=_without_channel_name(text))
            kept.append(["element", element])
        digest = hashlib.sha256(json.dumps(kept, sort_keys=True).encode("ascii")).hexdigest()
    except Exception as exc:  # an element nested past the recursion limit, or not JSON at all
        return "unattributed-" + hashlib.sha256(os.urandom(32)).hexdigest(), (
            f"its elements cannot be serialized ({type(exc).__name__})")
    if not identified:
        return ("unattributed-" + hashlib.sha256(os.urandom(32)).hexdigest(),
                "no element names an event id where buzz-acp writes one")
    return "unattributed-" + digest, None


def prompt_shape(blocks):
    """The buzz-acp shape of an unattributed prompt, by its elements' leading headers. For the journal
    and the log only: it decides nothing."""
    last = blocks[-1] if blocks and isinstance(blocks[-1], str) else ""
    if any(isinstance(block, str) and block.startswith(MERGE_HEADERS) for block in blocks):
        return "merged"
    if last.startswith("[Buzz events — "):
        return "batch"
    return "single_unreadable" if last.startswith(EVENT_HEADER + " ") else "unrecognized"


# 2026-10-10 04:28–04:29 KST: two replies in a long thread (17,550 and 17,079 chars) were refused with
# HTTP 400 canonical_invalid_request, because the gateway refuses `text` longer than its
# `_MAX_TEXT_CHARS` (gateway/canonical_surface.py, counted as `len()` of the decoded str). Nothing told
# the thread, so every later reply there silently never reached the CEO. The gateway limit stays as it
# is; the producer fits the text instead. Text the gateway already accepts is sent unchanged.
GATEWAY_TEXT_LIMIT = 16_384
# Trimming prefers to land below the limit by this margin; any result within the limit is legal.
TEXT_BUDGET = GATEWAY_TEXT_LIMIT - 384
THREAD_ROOT_RE = re.compile(r"Thread root: ([0-9a-f]{64})")
# Digit counts are capped: int() refuses more than 4300 digits, and no thread holds 10,000 messages.
THREAD_HEADER_RE = re.compile(r"\[Thread Context \((\d{1,4}) of (\d{1,6}) messages(?:, truncated)?\)\]")
# buzz-acp's own entry line: `[n] name (author pubkey) (timestamp): first line of the message`. The
# hex is the author's pubkey, not the message's event id; an entry carries no event id at all.
THREAD_ENTRY_RE = re.compile(r"\[(\d{1,4})\] (.*?) \([0-9a-f]{64}\) "
                             r"\((\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:\d\d))\): ")


def _thread_entries(lines, header, end, declared):
    """First line index and label of each of the *declared* entries in ``lines[header + 1:end]``, or
    None when that cannot be told for certain. A message body can hold a line shaped exactly like an
    entry line, so a boundary is taken only when every number 1..declared has exactly one such line,
    in order, the first right under the header; anything else is ambiguous and gets no guess."""
    found = {}
    for i in range(header + 1, end):
        entry = THREAD_ENTRY_RE.match(lines[i])
        if entry and 1 <= int(entry.group(1)) <= declared:
            found.setdefault(int(entry.group(1)), []).append(
                (i, f"[{entry.group(1)}] {entry.group(2)} {entry.group(3)}"))
    if declared < 1 or sorted(found) != list(range(1, declared + 1)):
        return None
    if any(len(hits) != 1 for hits in found.values()):
        return None
    entries = [found[n][0] for n in range(1, declared + 1)]
    if entries[0][0] != header + 1 or any(a[0] >= b[0] for a, b in zip(entries, entries[1:])):
        return None
    return entries


def fit_text(text, channel, event_id):
    """(text to submit, omitted messages, omitted chars), or None when it cannot fit.

    Within the gateway limit the text is returned as it is. Over it, whole ``[Thread Context]``
    entries are dropped oldest first and replaced by one marker naming what was dropped and how to
    read it back; when the entry boundaries are ambiguous, the whole history block goes as one unit.
    The ``[Context]`` block and the ``[Buzz event: …]`` section — envelope, the newest message in
    full, its tags — are never touched, so the result carries the event ``envelope()`` read. The
    fewest drops that land within TEXT_BUDGET win; failing that, the fewest within the gateway limit.
    None when even dropping all history leaves the text over the limit."""
    if len(text) <= GATEWAY_TEXT_LIMIT:
        return text, 0, 0
    # split("\n"), not splitlines(): the kept lines are rejoined and must come back byte for byte.
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if line.startswith(EVENT_HEADER)]
    if len(starts) != 1:
        return None
    event = starts[0]
    headers = [i for i in range(event) if THREAD_HEADER_RE.fullmatch(lines[i].rstrip())]
    if not headers:
        return None  # no thread history to give up
    header = headers[0]  # buzz-acp's own header precedes every quoted message
    end = event
    while end > header + 1 and not lines[end - 1].strip():
        end -= 1  # the blank line before the event section separates it; it is not history
    if end == header + 1:
        return None
    declared = int(THREAD_HEADER_RE.fullmatch(lines[header].rstrip()).group(1))
    roots = [found.group(1) for line in lines[:header]
             for found in [THREAD_ROOT_RE.fullmatch(line.rstrip())] if found]
    root = roots[0] if len(roots) == 1 else event_id
    retrieve = f"in thread {root} — retrieve with `buzz messages thread --channel {channel} --event {root}`]"
    entries = _thread_entries(lines, header, end, declared)
    if entries is None:
        # Boundaries unknown: every line of the history block goes, and the marker says so.
        spans = [(header + 1, end, declared,
                  f"all {declared} messages of this thread context (entry boundaries ambiguous)")]
    else:
        bounds = [i for i, _ in entries[1:]] + [end]
        spans = [(header + 1, bounds[n - 1], n, f"{n} messages, from {entries[0][1]} to {entries[n - 1][1]}")
                 for n in range(1, len(entries) + 1)]
    legal = None
    for cut_from, cut_to, dropped, what in spans:
        chars = sum(len(line) + 1 for line in lines[cut_from:cut_to])
        marker = f"[earlier thread context omitted: {chars} chars, {what} {retrieve}"
        fitted = "\n".join(lines[:cut_from] + [marker] + lines[cut_to:])
        if len(fitted) <= TEXT_BUDGET:
            return fitted, dropped, chars
        if legal is None and len(fitted) <= GATEWAY_TEXT_LIMIT:
            legal = (fitted, dropped, chars)
    return legal


def submit(text, event_id, author, channel):
    """Answer text, or raise. The caller turns a failure into a visible refusal sentence. Author and
    channel are the event's own, from ``envelope()``; nothing stands in for either."""
    body = json.dumps({
        "binding": BINDING, "event_id": event_id,
        "author_id": author, "channel_id": channel, "text": text}).encode()
    request = urllib.request.Request(
        BASE + ROUTE, data=body, method="POST",
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode()).get("text", "")


def _registry_peer(channel):
    """(the channel's registered counterpart handle, None) or (None, why not). Never raises: the reason
    is a fixed category and the exception type, nothing else, as for ``_own_signers``."""
    try:
        with open(MENTIONS, encoding="utf-8") as handle:
            channels = json.load(handle)["channels"]
    except OSError as exc:
        return None, f"reply registry unreadable: {type(exc).__name__}"
    except Exception as exc:  # malformed in any way, nesting past the recursion limit included
        return None, f"reply registry malformed: {type(exc).__name__}"
    try:
        peer = channels[channel]["peer"]["handle"]
    except Exception:
        peer = None
    if peer is None:
        return None, "the [Context] channel has no registered counterpart to post to"
    return peer, None


def _peer_handle(channel):
    return _registry_peer(channel)[0]  # the wrapper refuses an unregistered channel, and adds the handle itself


def _own_signers():
    """(pubkeys this bridge's posts can carry, None) or (None, why not). The wrapper signs with the key
    inherited from buzz-acp and accepts it only as a channel's registered ``ceo``. Every registered
    channel's, not only the event's: the key is one key, and an echo can be read in any channel. When
    the registry cannot be read or names no valid ``ceo``, the caller cannot rule out an echo and
    forwards nothing."""
    try:
        with open(MENTIONS) as handle:
            channels = json.load(handle)["channels"]
        signers = {rule["ceo"]["pubkey"] for rule in channels.values()}
    # The reason is a fixed category and the exception type, nothing else: an OS message or a parser
    # message can carry a path or the registry's own bytes, and the journal is not where those go.
    except OSError as exc:
        return None, f"reply registry unreadable: {type(exc).__name__}"
    except Exception as exc:  # malformed in any way, nesting past the recursion limit included
        return None, f"reply registry malformed: {type(exc).__name__}"
    if not signers or not all(isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key) for key in signers):
        return None, "reply registry names no valid ceo signer"
    return signers, None


def canary_armed_for(author, channel, body):
    """Whether the armed canary would take this event — read only, nothing claimed."""
    try:
        with open(CANARY) as handle:
            return _canary_matches(json.load(handle), author, channel, body)
    except (OSError, ValueError, AttributeError):
        return False


# The wrapper's own refusals state their code on stderr (buzz-mention-enforce.py `fail()`: mention_policy;
# bin/buzz without a key: identity_error). Any other non-zero exit is a send that failed.
WRAPPER_REFUSALS = ("mention_policy", "identity_error")


def _send_state(returncode, stderr):
    if returncode == 0:
        return "sent"
    try:
        code = json.loads(stderr.strip().splitlines()[0]).get("error")
    except Exception:  # anything stderr holds — even JSON nested past the recursion limit — is not a refusal
        code = None
    return "wrapper_refused" if code in WRAPPER_REFUSALS else "send_failed"


# The gateway states a refusal as ``{"error": {"code": "canonical_…", …}}`` (api_server_canonical.py
# `_refusal`). Only that literal code is acted on: a substring match let any refusal whose body
# happened to mention busy be resent as busy, and no other byte of the body ever reaches the sender.
REFUSAL_CODE_RE = re.compile(r"canonical_[a-z_]{1,64}")
REFUSAL_BODY_BYTES = 65_536


def refusal_of(raw):
    """(code, error object) of a gateway refusal body, or (None, {}) when it states no such code."""
    try:
        error = json.loads(raw.decode("utf-8")).get("error")
        code = error.get("code")
    except Exception:  # not JSON, not an object, nested past the recursion limit
        return None, {}
    if not isinstance(code, str) or not REFUSAL_CODE_RE.fullmatch(code):
        return None, {}
    return code, error


def refusal_notice(status, code, error, event_id, chars):
    """The sender's notice for a gateway refusal that ends this event for good, or None for any other.

    Built from fixed text, the gateway's code and state flags, two numbers and the event id only:
    never the message, the gateway's prose, a kept answer or a credential. It goes to the thread
    through the Buzz wrapper (``publish_reply``), not the canonical surface, so it cannot meet the
    limit it reports; one post per event, and a failed post is journaled as that failure."""
    if status == 400 and code == "canonical_invalid_request":
        # Refused at ingress, before any claim: the turn never started. The gateway names no reason
        # and length is the only one a request built here can meet, so both numbers are stated.
        return (f"이 메시지는 CEO 에게 전달되지 않았습니다. canonical surface 가 요청을 거부했습니다"
                f"(HTTP 400 {code}, 제출 {chars}자, 길이 한도 {GATEWAY_TEXT_LIMIT}자). 턴은 실행되지 않았고 "
                f"자동 재전송도 없습니다. 메시지를 나눠 보내거나 새 최상위 스레드로 다시 시작해 주세요. "
                f"(event {event_id})")
    if status == 409 and code == "canonical_answer_unrecorded":
        # The turn ran once and its answer is held at the gateway, unbound to any receipt: it is not
        # delivered, the turn is never re-run, and this event is not resent (#1110).
        delivery = "not_delivered" if error.get("delivery") == "not_delivered" else "unknown"
        kept = error.get("answer_kept")
        if kept is False:
            # Neither the receipt store nor the local fallback holds the answer: said plainly, never
            # implied to be kept somewhere.
            return (f"CEO 턴은 실행됐지만 답변을 기록하지도 보존하지도 못했습니다. 답변은 이 스레드에 "
                    f"전달되지 않았고 gateway 에도 남아 있지 않습니다. 이 턴은 다시 실행되지 않고 자동 "
                    f"재전송도 없습니다. 상태: {code} (delivery={delivery}, answer_kept=false, "
                    f"preservation=failed). (event {event_id})")
        kept = "true" if kept is True else "unknown"
        return (f"CEO 턴은 실행됐지만 답변을 기록하지 못해 이 스레드에 전달되지 않았습니다. 이 턴은 다시 "
                f"실행되지 않고 자동 재전송도 없습니다. 상태: {code} (delivery={delivery}, "
                f"answer_kept={kept}). (event {event_id})")
    return None


def _claim(name):
    """(fd, None) for a fresh O_EXCL claim of the journal entry ``PUBLISHED/<name>``, else (None, why)."""
    try:
        os.makedirs(PUBLISHED, exist_ok=True)
        return os.open(os.path.join(PUBLISHED, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), None
    except FileExistsError:
        return None, "exists"
    except OSError as exc:
        return None, f"marker unavailable: {type(exc).__name__}: {exc}"


def publish_reply(event_id, channel, answer, kind="answer", *, ref=None, reply_to=None):
    """Post *answer* once as the thread reply to *event_id*. True only when the wrapper sent it.

    The delivery journal is ``PUBLISHED/<event id>``: claimed before the send (one post per event,
    ever) and then filled with what happened — ``state`` is ``sent``, ``wrapper_refused``,
    ``send_failed``, ``timeout`` or ``error``, with the ``reason``, and ``delivery`` is ``posted``
    or ``failed``. A failure stays recorded as one and is never retried. An unattributed refusal has
    no event id: its notice passes ``event_id=None``, is journaled under its correlation key *ref*
    and replies to *reply_to*, the thread buzz-acp's ``[Context]`` names (None: the channel)."""
    name, target = (event_id, event_id) if event_id is not None else (ref, reply_to)
    tag = name[:12] if event_id is not None else name
    fd, why = _claim(name)
    if fd is None:
        log(f"prompt {tag}… reply already published once; not posting again" if why == "exists"
            else f"prompt {tag}… reply not posted: {why}")
        return False
    handle = _peer_handle(channel)
    body = f"@{handle} {answer}" if handle and not answer.lstrip().startswith(f"@{handle}") else answer
    try:
        done = subprocess.run(
            [BUZZ_CLI, "messages", "send", "--channel", channel]
            + (["--reply-to", target] if target else []) + ["--content", "-"],
            input=body.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=PUBLISH_TIMEOUT)
    except subprocess.TimeoutExpired:
        done, state, reason = None, "timeout", f"no exit within {PUBLISH_TIMEOUT:g}s"
    except Exception as exc:
        done, state, reason = None, "error", f"{type(exc).__name__}: {exc}"[:300]
    if done is not None:
        try:
            stderr = done.stderr.decode(errors="replace")
            state, reason = _send_state(done.returncode, stderr), f"exit {done.returncode}: {stderr.strip()[:300]}"
        except Exception as exc:  # the claimed marker is filled whatever the classification does
            state = "sent" if done.returncode == 0 else "send_failed"
            reason = f"exit {done.returncode}: stderr unclassifiable ({type(exc).__name__})"
    sent = state == "sent"
    with os.fdopen(fd, "w") as marker:
        marker.write(json.dumps({"sent": sent, "state": state, "delivery": "posted" if sent else "failed",
                                 "kind": kind, "event_id": event_id, "ref": ref, "channel": channel,
                                 "reply_to": target, "reason": str(reason)[:400],
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    log(f"prompt {tag}… {kind} {'POSTED' if sent else 'NOT posted (no retry)'} to thread: "
        f"{state}: {reason}")
    return sent


# Fixed text: it names no sender, quotes no byte of any message and carries only the refusal's key.
UNATTRIBUTED_NOTICE = (
    "이 대화로 온 메시지가 CEO 에게 전달되지 않았습니다. 여러 메시지가 하나로 합쳐진 프롬프트(작업 중 도착한 "
    "메시지 병합 또는 일괄 전달)이거나 단일 이벤트 머리글이 정해진 자리에 없어, 발신자와 이벤트를 하나로 "
    "특정할 수 없었습니다. 누구에게도 귀속하지 않고 거부했으며, 턴은 실행되지 않았고 자동 재전송도 없습니다. "
    "필요하면 메시지를 하나씩 다시 보내 주세요. (ref {ref})")


def journal_undeliverable(ref, shape, reason):
    """Record, once, an unattributed refusal whose notice has no destination: ``PUBLISHED/<ref>`` with
    ``sent`` false, ``state`` ``notice_impossible`` and ``delivery`` ``failed``. It takes the same
    one-entry claim a posted notice would, so the refusal is journaled once and never reads as sent."""
    fd, why = _claim(ref)
    if fd is None:
        log(f"prompt {ref}… refusal already journaled once; not again" if why == "exists"
            else f"prompt {ref}… refusal not journaled: {why}")
        return
    with os.fdopen(fd, "w") as record:
        record.write(json.dumps({"sent": False, "state": "notice_impossible", "delivery": "failed",
                                 "kind": "unattributed_notice", "event_id": None, "ref": ref, "shape": shape,
                                 "channel": None, "reply_to": None, "reason": reason,
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    log(f"prompt {ref}… unattributed_notice NOT posted: {reason}; journaled as a delivery failure")


# The From line buzz-acp writes ends with the author's hex, after the label (which holds no line break).
SENDER_HEX_RE = re.compile(r"From: .*hex: ([0-9a-f]{64})\)")


def _last_event_author(blocks):
    """The author on the From line of the prompt's last event, read at the fixed position buzz-acp writes
    it, or None. For the refusal path only, once ``notice_destination`` has bound the whole
    ``params.prompt[0]`` [Context] to that event: its channel name then holds no line break (any one
    leaves that form), so every header line above the event's message is where buzz-acp put it.

    Only one event alone in its element has its From line at a fixed position: line 4 under ``[Buzz
    event: <tag>]``, or line 6 under a merged prompt's ``[New message — …]`` / ``[New request — …]``
    header, its blank line and ``--- Event 1 (<tag>) ---``. Its Event ID, Channel (the [Context]'s own
    channel line) and Kind lines must be above it, and the author is the hex that ends it. Nothing is
    searched for: the last of two or more events in a list has no fixed position, so None, and no body
    line is ever read. Attribution never uses this; it only decides that no notice answers this bridge."""
    lines = _last_event_element(blocks)
    if lines is None:
        return None
    if EVENT_TITLE_RE.fullmatch(lines[0]):
        at = 1
    elif lines[0] in NEW_EVENT_SINGLE and len(lines) > 2 and lines[1] == "" and FIRST_EVENT_RE.fullmatch(lines[2]):
        at = 3
    else:
        return None
    channel_line = blocks[0].split("\n")[2]
    if not (len(lines) > at + 3 and HEADER_EVENT_RE.fullmatch(lines[at]) and lines[at + 1] == channel_line
            and ANY_KIND_RE.fullmatch(lines[at + 2])):
        return None
    sender = SENDER_HEX_RE.fullmatch(lines[at + 3])
    return sender.group(1) if sender else None


def journal_own_echo(ref, shape, channel, reply_to):
    """Record, once, an unattributed refusal whose last event is this bridge's own post coming back, so
    no notice answers it: ``PUBLISHED/own-echo-<ref's digest>`` with ``sent`` false, ``state``
    ``own_echo`` and ``delivery`` ``suppressed`` — neither a posted notice nor a failed one. Beside the
    refusal's own entry, not in it: *ref*'s entry stays the record of a notice posted or not."""
    fd, why = _claim("own-echo-" + ref[len("unattributed-"):])
    if fd is None:
        log(f"prompt {ref}… own-post echo already journaled once; not again" if why == "exists"
            else f"prompt {ref}… own-post echo not journaled: {why}")
        return
    with os.fdopen(fd, "w") as record:
        record.write(json.dumps({"sent": False, "state": "own_echo", "delivery": "suppressed",
                                 "kind": "unattributed_notice", "event_id": None, "ref": ref, "shape": shape,
                                 "channel": channel, "reply_to": reply_to,
                                 "reason": "the prompt's last event is this bridge's own post",
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    log(f"prompt {ref}… unattributed_notice NOT posted: the last event is this bridge's own post (an echo); "
        f"journaled as own_echo")


def refuse_unattributed(session, prompt, chars):
    """Refuse a prompt that is not one attributable event — a merged or batched dispatch, or an event
    section out of its fixed form — attributing it to no one. The conversation is told once per
    correlation key, in fixed text that names no sender and quotes no body, at the destination
    buzz-acp's own ``[Context]`` element names; when there is none, the reply registry cannot be read
    or parsed (nesting past the recursion limit included), or the post fails, the journal records a
    delivery failure under the refusal's key with a fixed reason, never a sent notice.

    The notice is signed with this bridge's own key and mentions only the channel's registered
    counterpart. buzz-acp drops its own events (``ignore_self``, on unless BUZZ_ACP_NO_IGNORE_SELF;
    lib.rs 2329) and wakes this agent only for mentions of it; an own-signer event that still arrives
    as one event is refused before any submission or post. One refused here is not answered either:
    once the destination is bound to the last event, the From line at that event's fixed position
    (``_last_event_author``) naming this bridge's own signer journals the refusal as an echo and posts
    nothing, and a registry that cannot say whose key that is posts nothing. A notice therefore starts
    no further refusal or reply."""
    blocks = prompt_blocks(prompt)
    (ref, own), shape = refusal_key(prompt), prompt_shape(blocks)
    log(f"prompt refused locally: not one attributable [Buzz event] element ({shape}, {len(blocks)} "
        f"elements, {chars} chars); ref {ref}" + (f" (a key of its own: {own})" if own else ""))
    channel, reply_to, why = notice_destination(blocks)
    if channel is not None:
        # This bridge's own post coming back is never answered. A registry that cannot name the signer
        # cannot rule an echo out, so nothing is posted, as ``handle`` holds such an event.
        signers, unknown = _own_signers()
        if signers is None:
            channel, why = None, unknown
        elif _last_event_author(blocks) in signers:
            journal_own_echo(ref, shape, channel, reply_to)
            return
    if channel is not None:
        # A registry that cannot be read or parsed is an undeliverable notice, journaled as one.
        handle, unknown = _registry_peer(channel)
        if handle is None:
            channel, why = None, unknown
    if channel is None:
        journal_undeliverable(ref, shape, why)
        return
    notice = UNATTRIBUTED_NOTICE.format(ref=ref[:29])
    write({"jsonrpc": "2.0", "method": "session/update", "params": {
        "sessionId": session,
        "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": notice}}}})
    publish_reply(None, channel, notice, kind="unattributed_notice", ref=ref, reply_to=reply_to)


def journal_held(event_id, channel, reason):
    """Record an event held back — not submitted, nothing posted — in the delivery journal, as
    ``PUBLISHED/<event id>.held`` (one line per refusal). Beside the post marker, not in it: a hold
    must not take the one post the event can still earn once the cause is fixed."""
    try:
        os.makedirs(PUBLISHED, exist_ok=True)
        fd = os.open(os.path.join(PUBLISHED, event_id + ".held"), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "a") as record:
            record.write(json.dumps({"sent": False, "state": "held", "event_id": event_id, "channel": channel,
                                     "reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    except OSError as exc:
        log(f"prompt {event_id[:12]}… hold not journaled: {type(exc).__name__}: {exc}")


def hold(session, author, channel, base, text):
    """Quarantine: held, not dropped. The event is recorded so the turn it would have driven can be
    replayed once the signer is separated, and the caller is told plainly that nothing reached the CEO.
    Refusing here leaves the bridge, its session and this job running."""
    try:
        with open(QUARANTINE, "a") as handle:
            handle.write(json.dumps({
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "binding": BINDING,
                "author_id": author, "channel_id": channel, "event_id": base,
                "text": text}, ensure_ascii=False) + "\n")
        os.chmod(QUARANTINE, 0o600)
    except OSError as exc:
        log(f"quarantine write failed: {type(exc).__name__}: {exc}")
    log(f"prompt {base[:12]}… QUARANTINED (CANONICAL_FORWARD={FORWARD}); nothing submitted")
    held = (
        "이 경로는 격리 상태입니다(CEO 지시, 2026-09-27). 서명된 Buzz 발신자가 CEO 세션에서 "
        "owner 의 텔레그램 턴과 분리되기 전까지 전달하지 않습니다. 이벤트는 증거로 "
        f"보존했습니다: {QUARANTINE}"
    )
    write({"jsonrpc": "2.0", "method": "session/update", "params": {
        "sessionId": session,
        "update": {"sessionUpdate": "agent_message_chunk",
                   "content": {"type": "text", "text": held}}}})


HEARTBEAT_HEADER = "[System: Heartbeat]"


def is_heartbeat(prompt):
    """Whether ``params.prompt`` is buzz-acp's own heartbeat, told from its ACP shape alone.

    A heartbeat (lib.rs ``dispatch_heartbeat`` → pool.rs ``run_prompt_task``) is exactly ONE text
    element, ``{"type": "text", "text": …}``, which opens with its ``[System: Heartbeat]`` header — or,
    for a legacy agent, with the ``[Base]`` section prepended to it (pool.rs ``prepend_base_for_legacy``).
    It carries no Buzz message at all. A channel prompt is never one element: ``format_prompt`` always
    writes ``[Context]`` and an event section as elements of their own, and no message body can merge
    or open an element. So the header is trusted only where nothing but buzz-acp writes, and a body
    quoting it never makes a prompt a heartbeat."""
    if not isinstance(prompt, list) or len(prompt) != 1:
        return False
    block = prompt[0]
    if not (isinstance(block, dict) and set(block) == {"type", "text"} and block["type"] == "text"
            and isinstance(block["text"], str)):
        return False
    lines = block["text"].split("\n")
    return lines[0].strip() == HEARTBEAT_HEADER or (
        lines[0] == "[Base]" and any(line.strip() == HEARTBEAT_HEADER for line in lines[1:]))


def prompt_text(params):
    """ACP sends a prompt as a list of content blocks; only text is carried to the CEO."""
    chunks = []
    for block in params.get("prompt") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(str(block.get("text") or ""))
        elif isinstance(block, str):
            chunks.append(block)
    return "\n".join(part for part in chunks if part).strip()


def handle(message):
    rpc_id, method = message.get("id"), message.get("method")
    params = message.get("params") or {}

    if method == "initialize":
        version = params.get("protocolVersion")
        result(rpc_id, {
            "protocolVersion": version if isinstance(version, int) else 1,
            "agentCapabilities": {"loadSession": False, "promptCapabilities": {"image": False, "audio": False, "embeddedContext": False}},
            "authMethods": [],
            "agentInfo": {"name": "acp-canonical-shim", "version": "1.0.0"}})
        return

    if method == "session/new":
        # The session is the binding's, not this process's. Naming it after the binding keeps any
        # log that quotes this id pointing at the real owner of the transcript.
        result(rpc_id, {"sessionId": "canonical:" + BINDING})
        return

    if method == "session/prompt":
        session = params.get("sessionId") or ("canonical:" + BINDING)
        prompt = params.get("prompt")
        if not prompt:
            # Nothing was sent (queue.rs ``format_prompt`` of an empty batch): no message to refuse.
            result(rpc_id, {"stopReason": "end_turn"})
            return
        # buzz-acp wakes its agent every --heartbeat-interval seconds with a prompt that carries no
        # human message and instructs the agent to poll the feed itself. On the other side of this
        # binding is the CEO's live session, which is doing its own work; a heartbeat submitted there
        # is an instruction that nobody sent, arriving as the owner. Refuse it here. 2026-09-27's
        # first heartbeat was only stopped by an HTTP 409 (the session happened to be busy), which is
        # timing, not a guard.
        # 2026-09-27: substring 검사는 실패했다. "routine heartbeat" 는 heartbeat 본문이 아니라 매
        # 프롬프트에 붙는 [Base]/persona 프롬프트 안에 있어서, repo-factory 방에서 온 실제 메시지
        # (3765자)가 거부됐다. 2026-10-18 (#91 r1): 줄 앵커도 본문을 믿었다 — 병합 프롬프트가 인용한
        # 기록 속 한 줄이 거부 알림과 기록을 통째로 지웠다. heartbeat 는 이제 ACP 모양으로만 판정한다.
        if is_heartbeat(prompt):
            log("prompt refused locally: heartbeat, not a message (buzz-acp's one-element heartbeat prompt)")
            result(rpc_id, {"stopReason": "end_turn"})
            return
        text = prompt_text(params)
        parsed = envelope(prompt)
        if parsed is None:
            # No single unambiguous event: nothing can be attributed, so nothing is forwarded in any
            # mode — and the conversation is told, or the journal says it could not be or that the
            # last event is this bridge's own post, which is never answered.
            refuse_unattributed(session, prompt, len(text))
            result(rpc_id, {"stopReason": "end_turn"})
            return
        author, channel, base, body = parsed
        log(f"prompt author={author[:12]}… channel={channel} base={base[:12]}… chars={len(text)}")
        # Every refusal is decided before anything is claimed, so an armed canary is spent only on an
        # event that is actually submitted — not on an echo, an unreadable registry or an oversize text.
        if FORWARD != "live" and not canary_armed_for(author, channel, body):
            hold(session, author, channel, base, text)
            result(rpc_id, {"stopReason": "end_turn"})
            return
        signers, unknown = _own_signers()
        if signers is None:
            log(f"prompt {base[:12]}… HELD, not sent: {unknown}")
            journal_held(base, channel, unknown)
            result(rpc_id, {"stopReason": "end_turn"})
            return
        if author in signers:
            # This bridge's own post — an answer or a not-delivered notice — coming back. buzz-acp drops
            # it first (ignore-self, the author gate, mentions-only) and the binding's author allowlist
            # refuses it after, but nothing here may depend on either: forwarded, it would reach the CEO
            # as a turn and be answered into the thread under a new event id, which no per-event marker
            # stops. (A registry that cannot be read cannot rule an echo out: held above, nothing posted —
            # the wrapper refuses to post without it.)
            log(f"prompt {base[:12]}… refused locally: authored by this bridge's own signer; not forwarded")
            result(rpc_id, {"stopReason": "end_turn"})
            return
        try:
            fitted = fit_text(text, channel, base)
            reason = (f"canonical surface 길이 한도({GATEWAY_TEXT_LIMIT}자)를 넘습니다: 이전 스레드 맥락을 "
                      f"모두 생략해도 이 메시지와 이벤트 머리글만으로 한도를 넘습니다(원문 {len(text)}자)")
        except Exception as exc:  # whatever the text holds, the outcome is a submission or a notice
            log(f"prompt {base[:12]}… fitting {len(text)} chars failed: {type(exc).__name__}: {str(exc)[:200]}")
            fitted = None
            reason = (f"canonical surface 길이 한도({GATEWAY_TEXT_LIMIT}자)에 맞추는 중 오류가 났습니다"
                      f"(원문 {len(text)}자)")
        if fitted is None:
            # Not submitted, and the thread is told so: a refusal that is only logged is what left
            # 2026-10-10's long thread talking to nobody. One post per event id, as for an answer: a
            # notice whose post fails is journaled as that failure and not retried — the marker is
            # claimed before the send because a send that timed out may still have been published, and
            # a wrapper refusal is deterministic, so a retry would duplicate or fail again. The notice
            # carries only fixed text, the limit, the length and the event id — never the message.
            log(f"prompt {base[:12]}… NOT submitted: {len(text)} chars; telling the thread")
            notice = (f"이 메시지는 CEO 에게 전달되지 않았습니다. {reason}. 메시지를 나눠 보내거나 새 "
                      f"최상위 스레드로 다시 시작해 주세요. (event {base})")
            write({"jsonrpc": "2.0", "method": "session/update", "params": {
                "sessionId": session,
                "update": {"sessionUpdate": "agent_message_chunk",
                           "content": {"type": "text", "text": notice}}}})
            publish_reply(base, channel, notice, kind="notice")
            result(rpc_id, {"stopReason": "end_turn"})
            return
        if FORWARD != "live":
            if not claim_canary(author, channel, body):
                hold(session, author, channel, base, text)  # disarmed, expired or taken in between
                result(rpc_id, {"stopReason": "end_turn"})
                return
            log(f"prompt {base[:12]}… CANARY: forwarding this one armed event (CANONICAL_FORWARD={FORWARD})")
        if fitted[0] is not text:
            log(f"prompt {base[:12]}… {len(text)} chars is over the gateway limit {GATEWAY_TEXT_LIMIT}: "
                f"omitted {fitted[1]} thread messages ({fitted[2]} chars), submitting {len(fitted[0])} chars")
        text = fitted[0]
        # Every attempt and every replay of this event uses its own Buzz event id. A busy refusal is
        # raised before the gateway claims the event (it records nothing), so retrying the same id
        # runs it once when the session frees; once claimed, a replay answers from the receipt and
        # never runs it again. A per-attempt id would give each retry its own receipt key, and a later
        # replay of an earlier attempt would run the turn a second time.
        deadline = time.monotonic() + BUSY_DEADLINE_SECS
        attempt = 0
        answer = None
        answered = False
        notice = None
        while True:
            attempt += 1
            event_id = base
            try:
                answer = submit(text, event_id, author, channel)
                answered = True
                break
            except urllib.error.HTTPError as exc:
                raw = exc.read(REFUSAL_BODY_BYTES)
                detail = raw.decode(errors="replace")[:400]
                code, refusal = refusal_of(raw)
                busy = exc.code == 409 and code == "canonical_turn_busy"
                if busy and time.monotonic() + BUSY_RETRY_SECS < deadline:
                    log(f"prompt {event_id} busy; retrying in {BUSY_RETRY_SECS:.0f}s (attempt {attempt})")
                    # Any stdout write resets buzz-acp's --idle-timeout. An unrecognised
                    # `sessionUpdate` deserialises to its Other variant and is not published, so
                    # waiting does not leak a message into the channel.
                    write({"jsonrpc": "2.0", "method": "session/update", "params": {
                        "sessionId": session,
                        "update": {"sessionUpdate": "canonical_wait", "attempt": attempt}}})
                    time.sleep(BUSY_RETRY_SECS)
                    continue
                log(f"prompt {event_id} refused HTTP {exc.code}: {detail}")
                answer = (f"canonical surface refused this turn (HTTP {exc.code}) after {attempt} "
                          f"attempt(s): {detail}")
                notice = refusal_notice(exc.code, code, refusal, base, len(text))
                break
            except Exception as exc:
                log(f"prompt {event_id} failed: {type(exc).__name__}: {exc}")
                answer = f"canonical surface unreachable: {type(exc).__name__}: {exc}"
                break
        write({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": session,
            "update": {"sessionUpdate": "agent_message_chunk",
                       "content": {"type": "text", "text": answer}}}})
        if answered and answer and answer.strip():
            publish_reply(base, channel, answer)
        elif notice is not None:
            # A notice that fails stays journaled as that failure and is the end of it. It is not
            # turned into an error result either: buzz-acp requeues a failed attempt, which would
            # resend this event.
            kept = refusal.get("answer_kept")
            log(f"prompt {base[:12]}… NOT delivered ({code}"
                + ("" if code != "canonical_answer_unrecorded" else
                   ", answer NOT preserved" if kept is False else f", answer_kept={kept}")
                + "); telling the thread")
            publish_reply(base, channel, notice, kind="notice")
        result(rpc_id, {"stopReason": "end_turn"})
        return

    if method == "session/cancel":
        # A turn already handed to the CEO cannot be recalled, so say so rather than imply it stopped.
        log("session/cancel received; the submitted turn continues at the binding")
        if rpc_id is not None:
            result(rpc_id, {})
        return

    if rpc_id is not None:
        error(rpc_id, -32601, f"method not supported by acp-canonical-shim: {method}")


def main():
    missing = [name for name, value in
               (("API_SERVER_KEY", KEY), ("CANONICAL_AUTHOR_ID", AUTHOR), ("CANONICAL_CHANNEL_ID", CHANNEL))
               if not value]
    if missing:
        # Fail before ACP initialize: a shim that answers initialize and then refuses every turn
        # looks to buzz-acp like a working agent producing empty replies.
        sys.stderr.write("acp-canonical-shim: missing " + ", ".join(missing) + "\n")
        raise SystemExit(2)
    log(f"started binding={BINDING} base={BASE}")
    # `for line in sys.stdin` reads ahead into an internal buffer, so a peer that sends one line
    # and keeps the pipe open can have that line withheld until the buffer fills. A pipe closed by
    # the test harness flushes it and hides this entirely; buzz-acp holds the pipe open, and the
    # shim sat through `initialize` without ever seeing it. readline() returns on the newline.
    while True:
        line = sys.stdin.readline()
        if not line:
            log("stdin closed; exiting")
            break
        line = line.strip()
        if not line:
            continue
        log(f"<- {line[:1600]}")
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            log(f"unparsable line dropped: {line[:200]}")
            continue
        try:
            handle(message)
        except Exception as exc:
            log(f"handler crashed on {message.get('method')}: {type(exc).__name__}: {exc}")
            if message.get("id") is not None:
                error(message.get("id"), -32603, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
