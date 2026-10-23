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
import fcntl
import hashlib
import importlib.util
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
# with the key inherited from buzz-acp. One terminal post per event id: a marker is claimed before the
# send, and a failed or refused send is logged, never retried.
BUZZ_CLI = os.environ.get("CANONICAL_BUZZ_CLI", os.path.expanduser("~/.hermes/bridge/bin/buzz"))
MENTIONS = os.environ.get("CANONICAL_MENTIONS_FILE", os.path.expanduser("~/.hermes/bridge/required-mentions.json"))
# Owner broadcasts may wake the CEO only in its paired project channels (2026-10-20).
OWNER = os.environ.get("BUZZ_ACP_AGENT_OWNER", "d046238dd1422955898205bd60507992cb101a1ea2ecfa6e52fd00d5ddca98c4")
PUBLISHED = os.environ.get("CANONICAL_PUBLISHED_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                   "published"))
PUBLISH_TIMEOUT = float(os.environ.get("CANONICAL_PUBLISH_TIMEOUT_SECS", "60"))

# This shim is also installed as a standalone script; load its topical sibling beside it.
_batch_error = None
try:
    _batch_spec = importlib.util.spec_from_file_location(
        "canonical_batch", os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_batch.py"))
    _batch = importlib.util.module_from_spec(_batch_spec)
    _batch_spec.loader.exec_module(_batch)
except Exception as exc:
    # Keep ACP available to refuse each affected event even after a partial/broken deployment.
    _batch = None
    _batch_error = f"{type(exc).__name__}: {exc}"
    print(f"acp-canonical-shim: batch recovery unavailable: {_batch_error}", file=sys.stderr, flush=True)

# The gateway imports the same reader; standalone deployments copy this sibling with the shim.
_roster_spec = importlib.util.spec_from_file_location(
    "canonical_roster", os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_roster.py"))
_roster_module = importlib.util.module_from_spec(_roster_spec)
_roster_spec.loader.exec_module(_roster_module)
_roster_reader = _roster_module.RosterReader(MENTIONS)

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


def _emit(payload):
    line = json.dumps(payload, ensure_ascii=False)
    with _out:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class PromptTurn:
    """One ``session/prompt`` as this process answers it. The stdin loop hands each prompt to its own
    thread (``dispatch``), so a ``session/cancel`` is read while the prompt runs. Exactly one response
    goes out per prompt: from the thread serving it, or earlier from ``cancel_prompts``. Once it is
    answered, that thread writes nothing more to stdout; buzz-acp has moved on."""

    def __init__(self, rpc_id, session):
        self.rpc_id, self.session = rpc_id, session
        self.lock = threading.Lock()
        self.answered = False
        self.cancelled = threading.Event()  # cancelled before anything was submitted
        self.batch = False  # keep draining authenticated siblings after a mid-batch steer
        self.submitted = None  # the event id once handed to the canonical surface


_local = threading.local()  # .prompt: the PromptTurn this thread serves; .submission: its event's turn
_PROMPTS = {}  # id(PromptTurn) -> PromptTurn, for every prompt not yet finished
_PROMPTS_LOCK = threading.Lock()


def write(payload):
    prompt = getattr(_local, "prompt", None)
    if prompt is None:
        _emit(payload)
        return
    with prompt.lock:
        if prompt.answered:
            return
        if "method" not in payload and payload.get("id") == prompt.rpc_id:
            prompt.answered = True
        _emit(payload)


def result(rpc_id, value):
    write({"jsonrpc": "2.0", "id": rpc_id, "result": value})


def error(rpc_id, code, message):
    write({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}})


# `[Buzz event: ...]` states who wrote the message and where. The binding's principal check reads
# `author_id`/`channel_id`, so sending this process's own env constants would make that check
# vacuous: every author buzz-acp forwards — owner, orchestrator, anyone later added to the
# allowlist — would arrive at the CEO as the one identity named in the environment. The values are
# read from the event only; no startup author/channel constants stand in for them.
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
# queue.rs also renders an unlabeled author as `npub1… (hex: …)`.
HEADER_FROM_RE = re.compile(
    r"From: (?:[^()]* \(npub: npub1[02-9ac-hj-np-z]+, |npub1[02-9ac-hj-np-z]+ \()hex: ([0-9a-f]{64})\)")
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
    """The id of the element's last event, or None, from the same confirmed sections the refusal key reads
    (``_confirmed_sections``): no second reading of where buzz-acp wrote an id. *channel_line* is the exact
    [Context]'s own. An element whose sections cannot all be confirmed gives None, so an id a body may have
    quoted never binds a destination; neither does one whose Channel line is not the [Context]'s."""
    sections = _confirmed_sections(lines, channel_line)
    if not sections:
        return None
    at, event_id = sections[-1]
    return event_id if at + 1 < len(lines) and lines[at + 1] == channel_line else None


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


def _exact_context(blocks):
    """(lines, channel id, form, None) of ``params.prompt[0]`` when it is buzz-acp's own ``[Context]`` element
    (``format_context_hints``, queue.rs 1238-1319) in its exact thread or channel form (``_context_form``), else
    (None, None, None, why not). The form leaves no room for a line break in the channel name: buzz-acp writes
    it unsanitized, and a name carrying one always leaves the producer's ``(#<uuid>)`` at the end of a later
    line that no form allows."""
    first = blocks[0] if blocks else None
    lines = first.split("\n") if isinstance(first, str) else []
    channel = HEADER_CHANNEL_RE.fullmatch(lines[2]) if len(lines) >= 4 and lines[0] == "[Context]" else None
    if channel is None:
        return None, None, None, "params.prompt[0] is not buzz-acp's [Context] block"
    form = _context_form(lines)
    if form is None:
        return None, None, None, "the [Context] block names no thread or channel in buzz-acp's form"
    return lines, channel.group(1), form, None


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
    lines, channel, form, why = _exact_context(blocks)
    if why is not None:
        return None, None, why
    scope, root, parent, anchor = form
    why = _disagreement(blocks, lines[2], channel, scope, root, parent, anchor)
    if why is not None:
        return None, None, why
    return channel, root if scope == "thread" else anchor, None


CONVERSATION_HEADERS = ("[Thread Context (", "[Conversation Context (")
MERGE_HEADERS = ("[What you were working on]", "[Previous request — interrupted before completion]")


# The elements buzz-acp opens with an event section header and then ``--- Event 1 (<tag>) ---`` (queue.rs
# 1503-1569, ``MergeFraming``): a batch, the interrupted request's events, and what arrived meanwhile.
EVENT_LIST_HEADERS = ("[Buzz events — ",) + MERGE_HEADERS + (
    "[New message — arrived while you were working]", "[New messages — arrived while you were working",
    "[New request — supersedes previous")
FIRST_EVENT_RE = re.compile(r"--- Event 1 \([^\n]*\) ---")
# Any line that could open section k of an event list; its other lines are held to their form below.
SECTION_RE = re.compile(r"--- Event (\d{1,6}) \(.*\) ---")
# The channel line of a [Context] element: the display name, then the channel id buzz-acp appends.
CONTEXT_CHANNEL_NAME_RE = re.compile(r"Channel: .* (\(#" + _UUID + r"\))")


def _section_candidates(lines, channel_line):
    """(k, line of its Event ID) for every line of an event list that could open buzz-acp's section k:
    ``--- Event k (<tag>) ---`` after an empty line, then Event ID, *channel_line*, Kind, From, Time and
    ``Content:`` at their offsets (queue.rs 1511-1517 and 1550-1557 over ``format_event_block``)."""
    return [(int(head.group(1)), i + 1) for i in range(1, len(lines) - 6)
            if lines[i - 1] == "" and lines[i].startswith("--- Event ")
            for head in [SECTION_RE.fullmatch(lines[i])]
            if head and HEADER_EVENT_RE.fullmatch(lines[i + 1]) and lines[i + 2] == channel_line
            and ANY_KIND_RE.fullmatch(lines[i + 3]) and ANY_FROM_RE.fullmatch(lines[i + 4])
            and ANY_TIME_RE.fullmatch(lines[i + 5]) and lines[i + 6].startswith("Content:")]


def _confirmed_sections(lines, channel_line):
    """The sections of one element whose event ids buzz-acp wrote, as ``[(line of the Event ID, id), …]`` in
    order; ``[]`` when the element opens as an event section but an id in it is not at a position that can be
    confirmed; None when it names no event. The one reading of where an id is, for the refusal key and for
    the destination (``_last_event_id``) alike.

    A single-event element proves its own position: ``[Buzz event: <tag>]``, or a merged prompt's ``[New
    message — …]`` / ``[New request — …]`` header, a blank line and ``--- Event 1 (<tag>) ---``, are written by
    buzz-acp before any message, so the Event ID line under them (line 1, line 3) is the producer's.

    An event list does not: a message body can quote a whole section, and a candidate that has the shape is
    not, on its own, a section of buzz-acp's. Its ids count only under these producer facts (queue.rs
    1081-1147, 1503-1569):

    (a) *channel_line* is the Channel line of ``params.prompt[0]``, given only once ``_exact_context`` has
        found that [Context] in buzz-acp's exact form. That form proves the channel name holds no line break,
        and buzz-acp writes the same name in every section; a label holds none either
        (``sanitize_prompt_label``). So every section buzz-acp wrote is a candidate
        (``_section_candidates``) with that line under its Event ID, and a body can only add candidates.
    (b) A declared count N (``[Buzz events — N events]``, ``[New messages — … — N events]``, ``[New request —
        … — N events]``): buzz-acp writes exactly one section for each k in 1..N, so exactly one candidate for
        each k is that section. A second candidate for any k <= N leaves it unknown which one is real: the
        element is ambiguous. A candidate for k > N is no section of buzz-acp's and is not read.
    (c) A cancelled section (``[What you were working on]`` / ``[Previous request — …]``) declares no count, so
        no k >= 2 can ever be confirmed: its only candidate must be k = 1, and any other makes it ambiguous.
    (d) Without a [Context] in exact form (*channel_line* empty) no list section is confirmed.

    The count and its numbering are necessary; the [Context] form is what makes them sufficient. No body is
    searched for an id, and none is promoted to a section."""
    first = lines[0]
    if EVENT_TITLE_RE.fullmatch(first):
        at = 1
    elif first in NEW_EVENT_SINGLE:
        at = 3 if len(lines) > 3 and lines[1] == "" and FIRST_EVENT_RE.fullmatch(lines[2]) else None
    elif first.startswith((EVENT_HEADER,) + EVENT_LIST_HEADERS):
        count = EVENT_COUNT_RE.fullmatch(first)
        if not channel_line or not (count or first in MERGE_HEADERS):
            return []
        by_k = {}
        for k, where in _section_candidates(lines, channel_line):
            by_k.setdefault(k, []).append(where)
        if count:
            wanted = range(1, int(count.group(1)) + 1)
            if any(len(by_k.get(k, ())) != 1 for k in wanted):
                return []
            ats = [by_k[k][0] for k in wanted]
        elif list(by_k) == [1] and len(by_k[1]) == 1:
            ats = by_k[1]
        else:
            return []
        if not ats or ats != sorted(ats):
            return []
        return [(at, HEADER_EVENT_RE.fullmatch(lines[at]).group(1)) for at in ats]
    else:
        return None
    found = HEADER_EVENT_RE.fullmatch(lines[at]) if at is not None and at < len(lines) else None
    return [(at, found.group(1))] if found else []


def _event_ids(text, channel_line=None):
    """The event ids that identify one element, in order (``_confirmed_sections``); ``[]`` when the element
    names events not all at a confirmed position; None when it names none.

    ``refusal_key`` passes *channel_line*: the Channel line of an exact-form [Context], or ``""`` when there is
    none, and then no list section is confirmed. Called on an element alone (None), its own first Channel line
    (line 4, under ``--- Event 1``) is taken instead: that reads the element's sections against themselves and
    proves nothing about the channel name, so no refusal is keyed and no destination bound that way.

    Nothing else of the element is read: not its header or tag, the channel name, a Kind, From, Time, Tags or
    Parsed line, or any label on them. An id is the event's own hash, committing its content and tags; the
    rest of the element is presentation buzz-acp renders again on every dispatch (display names: pool.rs
    ``fetch_prompt_profile_lookup``), so a refreshed label or a renamed channel changes no confirmed id."""
    lines = text.split("\n")
    if channel_line is None:
        channel_line = lines[4] if len(lines) > 4 else ""
    sections = _confirmed_sections(lines, channel_line)
    return None if sections is None else [event_id for _, event_id in sections]


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
    every dispatch. An element that names its events contributes only the ids at confirmed producer positions
    (``_event_ids`` over ``_confirmed_sections``, against the exact [Context]'s Channel line) and its fields
    other than the text. The quoted conversation history, fetched again each time, is left out.
    ``params.prompt[0]``, buzz-acp's [Context], is kept but for the channel's display name; every other
    element — a closing note, any other text, a non-text element with its type, data and every field —
    is kept whole. So the same events dispatched again (a requeue, a respawned shim, a refreshed profile or
    a renamed channel) give the same key and so one notice; any other difference, an image's bytes
    included, gives another key.

    Only confirmed ids give the digest an identity. A prompt with none, or with any element that names events
    not all at a confirmed position (a body quoting a whole section, a list under no exact-form [Context]),
    is unidentified: it gets a key of its own, as does one that cannot be serialized at all. That never drops
    a distinct input; it does not deduplicate a redelivery, so the same input sent again can be noticed again."""
    elements = prompt if isinstance(prompt, list) else [prompt]
    kept, identified = [], False
    try:
        context = _exact_context(prompt_blocks(prompt))[0]
        channel_line = context[2] if context else ""
        for index, element in enumerate(elements):
            if isinstance(element, dict) and element.get("type") == "text" and isinstance(element.get("text"), str):
                text = element["text"]
                if text.startswith(CONVERSATION_HEADERS):
                    continue
                ids = _event_ids(text, channel_line)
                if ids == []:
                    return ("unattributed-" + hashlib.sha256(os.urandom(32)).hexdigest(),
                            "an element names events not all at a producer position that can be confirmed")
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
    channel are the event's own, from ``envelope()`` or a verified relay event; nothing stands in for either."""
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
    snapshot, why = _read_roster()
    if snapshot is None:
        return None, why
    room = snapshot.active(channel)
    if room is None:
        return None, "the [Context] channel has no registered counterpart to post to"
    return room["peer"]["handle"], None


def _peer_handle(channel):
    return _registry_peer(channel)[0]


def _read_roster():
    try:
        snapshot = _roster_reader.read()
    except _roster_module.RosterError as exc:
        cause = exc.__cause__
        category = "unreadable" if isinstance(cause, OSError) else "malformed"
        log(f"routing {exc}; current event refused, last good roster retained")
        return None, f"reply registry {category}: {type(cause).__name__}"
    return snapshot, None


def _own_signers():
    snapshot, why = _read_roster()
    if snapshot is None:
        return None, why
    signers = {rule["ceo"]["pubkey"] for rule in snapshot.channels.values()
               if rule["status"] == "active"}
    if not signers:
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
    if status == 403 and code == "canonical_principal_rejected":
        # Refused before any claim: this sender is not admitted in this room. The body can name a key
        # or a binding, so none of it is copied — the text is fixed and names only the event.
        return ("이 메시지는 CEO 에게 전달되지 않았습니다. 이 방에서 이 발신자의 메시지는 CEO 가 받지 않도록 "
                "설정되어 있습니다. 턴은 실행되지 않았고 자동 재전송도 없습니다. "
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
    if status == 409 and code in TERMINAL_NOTICES:
        # Answered past the claim: the turn ran, or may have, and nothing of it is delivered or run again.
        # The side-effect scope is the gateway's own record, never assumed; without one it is unknown.
        scope = error.get("side_effects")
        scope = scope if scope in ("none", "attempted", "unknown") else "unknown"
        return (f"{TERMINAL_NOTICES[code]}. 답변은 이 스레드에 전달되지 않았고, 이 턴은 다시 실행되지 않으며 "
                f"자동 재전송도 없습니다. 상태: {code} (HTTP 409, gateway 기록 side_effects={scope}). "
                f"(event {event_id})")
    if code is None and status >= 500:
        return unknown_outcome_notice(f"HTTP {status}", event_id)
    return None


# Codes the gateway answers only once the event was claimed (api_server_canonical.py): a refusal raised
# before the claim means the turn never ran and no answer exists, and stays as it is.
TERMINAL_NOTICES = {
    "canonical_event_failed": "CEO 턴이 실패로 끝났고 gateway 에 실패로 기록됐습니다",
    "canonical_event_uncertain": "이 메시지로 시작된 CEO 턴이 실행됐는지, 어떻게 끝났는지 gateway 가 확정하지 못했습니다",
    "canonical_event_conflict": "이 이벤트의 gateway 기록이 이 요청과 맞지 않아 그 결과를 전달할 수 없습니다",
}


def unknown_outcome_notice(cause, event_id):
    """The sender's notice when the gateway gave no usable answer to a submission: whether the turn ran
    is unknown here, so neither success nor its absence is stated. ``cause`` is an exception type or an
    HTTP status, never a message."""
    return (f"CEO 에게 제출한 이 메시지에 대해 canonical surface 의 응답을 받지 못했습니다({cause}). 턴이 "
            f"실행됐는지 이 브리지는 알 수 없고, 그 결과는 이 스레드에 전달되지 않습니다. 자동 재전송은 "
            f"없습니다. (event {event_id})")


def _claim(name):
    """(fd, None) for a fresh O_EXCL claim of the journal entry ``PUBLISHED/<name>``, else (None, why)."""
    try:
        os.makedirs(PUBLISHED, exist_ok=True)
        return os.open(os.path.join(PUBLISHED, name), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), None
    except FileExistsError:
        return None, "exists"
    except OSError as exc:
        return None, f"marker unavailable: {type(exc).__name__}: {exc}"


def publication_record(name):
    try:
        with open(os.path.join(PUBLISHED, name), encoding="utf-8") as handle:
            record = json.load(handle)
        return record if isinstance(record, dict) else {}
    except (OSError, ValueError):
        return {}


# Set only by the operator's explicit ``recover`` command (#98 r2 H98-03): a notice journaled before
# records carried "journal" (and so before pre-claim proof existed) may yield its event's one later
# answer when an operator explicitly recovers that event. No automatic path ever sets it.
_EXPLICIT_RECOVERY = False


def outcome_key(event_id, channel=None):
    """Only a recorded, proven pre-claim refusal leaves room for a later terminal outcome; so does a
    legacy notice, but only under an explicit operator recovery."""
    # Once the second slot exists it is authoritative in every mode (#100 r2 H100-R2-01): only a proven
    # pre-claim refusal or an explicit recovery ever opened it, and its terminal record (an answer or an
    # unfinished/uncertainty notice) must stop a later live replay from submitting again.
    if os.path.lexists(os.path.join(PUBLISHED, event_id + ".answer")):
        return event_id + ".answer"
    prior = publication_record(event_id)
    if (prior.get("event_id") == event_id and prior.get("kind") == "notice"
            and (prior.get("preclaim") is True or (_EXPLICIT_RECOVERY and "journal" not in prior))
            and (channel is None or prior.get("channel") == channel)):
        return event_id + ".answer"
    return event_id


def publish_reply(event_id, channel, answer, kind="answer", *, ref=None, reply_to=None, preclaim=False):
    """Post *answer* once as the thread reply to *event_id*. True only when the wrapper sent it.

    The delivery journal is ``PUBLISHED/<event id>``: claimed before the send (one post per event,
    ever, except an explicit replay after a proven pre-claim refusal can publish one terminal outcome
    under ``<event id>.answer``). That slot also holds terminal failure/uncertainty notices, never
    another answer after one. The journal records ``state`` as ``sent``, ``wrapper_refused``,
    ``send_failed``, ``timeout`` or ``error``, with the ``reason``, and ``delivery`` is ``posted``
    or ``failed``. A failure stays recorded as one and is never retried. An unattributed refusal has
    no event id: its notice passes ``event_id=None``, is journaled under its correlation key *ref*
    and replies to *reply_to*, the thread buzz-acp's ``[Context]`` names (None: the channel)."""
    name, target = (event_id, event_id) if event_id is not None else (ref, reply_to)
    if event_id is not None and not preclaim:
        name = outcome_key(event_id, channel)
    tag = name[:12] if event_id is not None else name
    fd, why = _claim(name)
    if fd is None:
        log(f"prompt {tag}… reply already published once; not posting again" if why == "exists"
            else f"prompt {tag}… reply not posted: {why}")
        return False
    handle = _peer_handle(channel)
    body = f"@{handle} {answer}" if handle and not answer.lstrip().startswith(f"@{handle}") else answer
    try:
        if handle is None:
            raise _roster_module.RosterError("reply room inactive or roster unavailable")
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
        marker.write(json.dumps({"journal": 2, **({"preclaim": True} if preclaim else {}),
                                 "sent": sent, "state": state, "delivery": "posted" if sent else "failed",
                                 "kind": kind, "event_id": event_id, "ref": ref, "channel": channel,
                                 "roster_generation": getattr(_local, "roster_generation", None),
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
RECOVERY_UNKNOWN_NOTICE = (
    "이 프롬프트의 이벤트 경계를 복구하지 못했습니다. 이전 제출의 CEO 수신 여부와 턴의 결과는 "
    "알 수 없습니다(outcome unknown). 이번 복구에서는 제출하지 않았고 자동 재전송은 없습니다. (ref {ref})")


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


def _single_event_author(blocks):
    """The author of a plain single-event prompt, read from its From line at buzz-acp's fixed line 4, or
    None for every other prompt. For the refusal path only, once ``notice_destination`` has bound the whole
    ``params.prompt[0]`` [Context] to the prompt's event: its channel name then holds no line break (any one
    leaves that form), so every header line above the event's message is where buzz-acp put it.

    Plain means the prompt is exactly buzz-acp's single-event shape (queue.rs ``format_prompt``): [Context],
    an optional quoted conversation history, then ONE ``[Buzz event: <tag>]`` element, and nothing else. Its
    Event ID, Channel (the [Context]'s own channel line) and Kind lines must be above the From line, and the
    author is the hex that ends it. A merged prompt (any cancelled section), a new-event element or an event
    list is never plain, whoever wrote its last event: another author's request can be in it. Nothing is
    searched for and no body line is read. Attribution never uses this; it only decides that no notice
    answers this bridge's own post."""
    if not (len(blocks) in (2, 3) and all(isinstance(block, str) for block in blocks)
            and (len(blocks) == 2 or blocks[1].startswith(CONVERSATION_HEADERS))):
        return None
    lines = blocks[-1].split("\n")
    if not (len(lines) > 4 and EVENT_TITLE_RE.fullmatch(lines[0]) and HEADER_EVENT_RE.fullmatch(lines[1])
            and lines[2] == blocks[0].split("\n")[2] and ANY_KIND_RE.fullmatch(lines[3])):
        return None
    sender = SENDER_HEX_RE.fullmatch(lines[4])
    return sender.group(1) if sender else None


def journal_own_echo(ref, shape, channel, reply_to):
    """Record, once, a refused plain single-event prompt whose one event is this bridge's own post coming
    back, so no notice answers it: ``PUBLISHED/own-echo-<ref's digest>`` with ``sent`` false, ``state``
    ``own_echo`` and ``delivery`` ``suppressed`` — neither a posted notice nor a failed one. Only once the
    destination is whole (channel and registered counterpart): a missing one is a delivery failure, never
    an echo. Beside the refusal's own entry, not in it: *ref*'s entry stays the record of a notice posted
    or not."""
    fd, why = _claim("own-echo-" + ref[len("unattributed-"):])
    if fd is None:
        log(f"prompt {ref}… own-post echo already journaled once; not again" if why == "exists"
            else f"prompt {ref}… own-post echo not journaled: {why}")
        return
    with os.fdopen(fd, "w") as record:
        record.write(json.dumps({"sent": False, "state": "own_echo", "delivery": "suppressed",
                                 "kind": "unattributed_notice", "event_id": None, "ref": ref, "shape": shape,
                                 "channel": channel, "reply_to": reply_to,
                                 "reason": "the prompt's one event is this bridge's own post",
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    log(f"prompt {ref}… unattributed_notice NOT posted: the prompt's one event is this bridge's own post "
        f"(an echo); journaled as own_echo")


def refuse_unattributed(session, prompt, chars, *, notice_template=UNATTRIBUTED_NOTICE):
    """Refuse a prompt that is not one attributable event — a merged or batched dispatch, or an event
    section out of its fixed form — attributing it to no one. The conversation is told once per
    correlation key, in fixed text that names no sender and quotes no body, at the destination
    buzz-acp's own ``[Context]`` element names; when there is none, the reply registry cannot be read
    or parsed (nesting past the recursion limit included), or the post fails, the journal records a
    delivery failure under the refusal's key with a fixed reason, never a sent notice.

    The notice is signed with this bridge's own key and mentions only the channel's registered
    counterpart. It is one notice to the conversation, not one to each author whose message the prompt
    holds. buzz-acp drops its own events (``ignore_self``, on unless BUZZ_ACP_NO_IGNORE_SELF; lib.rs 2329)
    and wakes this agent only for mentions of it; an own-signer event that still arrives as one event is
    refused before any submission or post. One refused here is not answered either, but only in a plain
    single-event prompt (``_single_event_author``) and only once the whole destination is established:
    the [Context] bound to the event and a registered counterpart. Then the From line at the event's fixed
    position naming this bridge's own signer journals the refusal as an echo and posts nothing, and a
    registry that cannot say whose key that is posts nothing. A merged prompt, a new-event element or an
    event list gets its notice whoever wrote its last event, since another author's request can be in it;
    being bridge-signed, that notice starts no further refusal or reply."""
    blocks = prompt_blocks(prompt)
    (ref, own), shape = refusal_key(prompt), prompt_shape(blocks)
    log(f"prompt refused locally: not one attributable [Buzz event] element ({shape}, {len(blocks)} "
        f"elements, {chars} chars); ref {ref}" + (f" (a key of its own: {own})" if own else ""))
    channel, reply_to, why = notice_destination(blocks)
    if channel is not None:
        # The destination is whole only with the channel's registered counterpart; a registry that cannot
        # be read or parsed, or names none, is an undeliverable notice, journaled as one and never an echo.
        handle, unknown = _registry_peer(channel)
        if handle is None:
            channel, why = None, unknown
    if channel is not None:
        # This bridge's own post coming back alone is never answered. A registry that cannot name the signer
        # cannot rule an echo out, so nothing is posted, as ``handle`` holds such an event.
        signers, unknown = _own_signers()
        if signers is None:
            channel, why = None, unknown
        elif _single_event_author(blocks) in signers:
            journal_own_echo(ref, shape, channel, reply_to)
            return
    if channel is None:
        journal_undeliverable(ref, shape, why)
        return
    notice = notice_template.format(ref=ref[:29])
    write({"jsonrpc": "2.0", "method": "session/update", "params": {
        "sessionId": session,
        "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": notice}}}})
    publish_reply(None, channel, notice, kind="unattributed_notice", ref=ref, reply_to=reply_to)


def batch_sections(prompt):
    """Recognized event-list elements under the shim's existing anti-spoof framing contract.

    None leaves unrecognized/ambiguous prompts on the existing refusal path. Presentation headers
    supply no author, mention status, message body or thread context to a canonical submission.
    """
    blocks = prompt_blocks(prompt)
    context, channel, _, why = _exact_context(blocks)
    if why or any(not isinstance(block, str) for block in blocks):
        return None
    rest = blocks[1:]
    if rest and rest[0].startswith(CONVERSATION_HEADERS):
        rest = rest[1:]
    if len(rest) == 1 and EVENT_COUNT_RE.fullmatch(rest[0].split("\n")[0]):
        sections = rest
    elif (len(rest) == 3 and rest[0].split("\n")[0] in MERGE_HEADERS
          and rest[-1] in CLOSING_NOTES
          and (rest[1].split("\n")[0] in NEW_EVENT_SINGLE
               or EVENT_COUNT_RE.fullmatch(rest[1].split("\n")[0]))):
        sections = rest[:2]
    else:
        return None
    return channel, context[2], sections


def refuse_recovery_event(session, prompt, event_id, reason):
    """A recovery refusal owns a separate notice key; it never spends the event's answer marker."""
    state = recovery_delivery_state(event_id)
    if state in ("answered", "pending", "notified"):
        log(f"prompt {event_id} recovery unavailable; existing delivery state {state}; no new notice")
        return
    ref = "unattributed-" + hashlib.sha256(("recovery:" + event_id).encode()).hexdigest()
    log(f"prompt {event_id} recovery refused: {reason}; ref {ref}")
    print(f"acp-canonical-shim: event {event_id} recovery refused: {reason}", file=sys.stderr, flush=True)
    channel, _, why = notice_destination(prompt_blocks(prompt))
    if channel is not None:
        handle, why = _registry_peer(channel)
        signers, unknown = _own_signers()
        if handle is None or signers is None:
            channel, why = None, why or unknown
    if channel is None:
        journal_undeliverable(ref, "event_recovery", why)
        return
    notice = ("원본 이벤트를 인증하여 복구하지 못했습니다. 이전 제출의 CEO 수신 여부와 턴의 결과는 "
              "알 수 없습니다(outcome unknown). 이번 복구에서는 제출하지 않았고 자동 재전송은 없습니다. "
              f"다른 이벤트의 처리는 계속됩니다. (event {event_id}, ref {ref[:29]})")
    write({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": session,
        "update": {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": notice}}}})
    publish_reply(None, channel, notice, kind="unattributed_notice", ref=ref, reply_to=event_id)


def recovery_delivery_state(event_id):
    """Reconcile local answers/outcomes and submission ownership without resubmitting a body.

    The gateway exposes receipt replay through the original attributed POST, not a read-only
    receipt lookup. Without that verified original we cannot safely replay it. Missing local
    evidence therefore means unknown, never proof that no gateway receipt exists.
    """
    records = [publication_record(event_id + suffix) for suffix in ("", ".answer")]
    records = [r for r in records if r.get("event_id") == event_id]
    if any(r.get("kind") == "answer" and r.get("sent") is True for r in records):
        return "answered"
    with _INFLIGHT_LOCK:
        if event_id in _INFLIGHT:
            return "pending"
    if os.path.lexists(os.path.join(PUBLISHED, event_id + INFLIGHT_SUFFIX)):
        return "pending"  # includes abandoned/uncertain ownership; the reaper arbitrates it
    if any(r.get("kind") in ("notice", "unfinished_notice", "answer") for r in records):
        return "notified"
    return "unknown"


def recover_batch(session, prompt):
    if not os.environ.get("BUZZ_RELAY_URL"):
        return False
    batch = batch_sections(prompt)
    if batch is None:
        return False
    if _batch is not None:
        return _batch.deliver_batch(globals(), session, prompt)
    # Only producer-confirmed IDs can receive a per-event notice without the recovery module.
    # Ambiguous sections get a whole-prompt unknown outcome, never guessed identities/non-receipt.
    _, channel_line, sections = batch
    unresolved = False
    ids = []
    for section in sections:
        confirmed = _confirmed_sections(section.split("\n"), channel_line)
        ids.extend(event_id for _, event_id in confirmed or [])
        unresolved |= not confirmed
    for event_id in dict.fromkeys(ids):
        refuse_recovery_event(session, prompt, event_id, "batch module unavailable: " + _batch_error)
    if unresolved:
        refuse_unattributed(session, prompt, len(prompt_text({"prompt": prompt})),
                            notice_template=RECOVERY_UNKNOWN_NOTICE)
    return True


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


# One canonical turn at a time from this process, and a cancel never recalls one (D2 / O2). The stdin
# loop reads on while a prompt's thread waits for the gateway, so buzz-acp's steer ``session/cancel`` is
# answered inside its 5 s drain and this process is not SIGKILLed with a turn in flight. A prompt that
# arrives while an earlier turn still runs waits for this slot: two turns never share the session, and
# an event already in flight is never submitted again.
_TURN_SLOT = threading.Lock()
_INFLIGHT = set()  # event ids this process submitted and has not finished delivering
_INFLIGHT_LOCK = threading.Lock()
_THREADS = []  # prompt threads and the startup reaper, joined when stdin closes
INFLIGHT_SUFFIX = ".inflight"
INFLIGHT_RE = re.compile(r"[0-9a-f]{64}\.inflight")

UNFINISHED_NOTICE = (
    "이 메시지는 CEO 에게 제출됐지만, 이를 제출한 브리지 처리가 gateway 의 응답을 받기 전에 끝났습니다(프로세스 "
    "교체 또는 중단). 턴이 실행됐는지, 어떻게 끝났는지 이 브리지는 알 수 없고, 그 결과는 이 스레드에 게시되지 "
    "않습니다. 이 브리지는 이 메시지를 자동으로 다시 제출하지 않습니다. (event {event_id})")


def _same_file(fd, path):
    """Ownership, checked each time it is relied on: the entry now at *path* is the very file *fd* holds
    locked (same st_dev and st_ino). An unlinked or replaced entry is never ownership."""
    try:
        held, named = os.fstat(fd), os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)


def _claim_inflight(base, channel):
    """(fd, None) holding ``PUBLISHED/<event id>.inflight`` under an exclusive ``flock`` while this process
    owns the event's turn, or (None, why not). The kernel drops the lock when the process dies, which is
    how a later shim tells a replaced owner from a live one.

    The entry is published only once initialized and locked: the owner record is written and fsync'd to
    a private file under the lock, which is then linked to the entry's name (``link`` never replaces a
    name). So no reader ever finds it empty or unlocked while its owner lives. A name already taken — a
    live owner, or an unfinished entry not yet reaped — is no claim, and neither is a name that does not
    end up being this very file."""
    path = os.path.join(PUBLISHED, base + INFLIGHT_SUFFIX)
    private = os.path.join(PUBLISHED, f".{base}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.claim")
    try:
        os.makedirs(PUBLISHED, exist_ok=True)
        fd = os.open(private, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        return None, f"the in-flight entry cannot be written: {type(exc).__name__}"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = (json.dumps({"event_id": base, "channel": channel, "pid": os.getpid(),
                              "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n").encode()
        if os.write(fd, record) != len(record):
            raise OSError("short write")
        os.fsync(fd)
        os.link(private, path)
    except FileExistsError:
        os.close(fd)
        return None, "another shim process holds this event, or its unfinished entry is not yet reaped"
    except OSError as exc:
        os.close(fd)
        return None, f"the in-flight entry was not published: {type(exc).__name__}"
    finally:
        try:
            os.unlink(private)
        except OSError:
            pass
    if not _same_file(fd, path):
        os.close(fd)
        return None, "the in-flight entry was replaced as it was published"
    return fd, None


def _unfinished(base):
    """Why *base* must never be submitted again, or None. The record read is the event's existing
    PUBLISHED journal entry, and only to refuse a resubmission — it never decides or completes delivery,
    and the gateway receipt stays the turn's outcome. An ``unfinished_notice`` entry is the reaper's: a
    shim submitted the event (or was about to) and ended without its outcome, so whether the turn ran is
    unknown, and it stays unresolved rather than being sent again. An entry with no complete record (a
    post cut short) is as unknown. An answer or notice entry is the turn's own outcome: a redelivery is
    submitted as before, and the gateway answers it from the receipt without running it again.
    After a proven pre-claim refusal the terminal record lives at ``<base>.answer`` instead;
    legacy notices without that proof keep their original one-post limit."""
    try:
        with open(os.path.join(PUBLISHED, outcome_key(base)), encoding="utf-8") as handle:
            raw = handle.read(65_536)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"its journal entry is unreadable ({type(exc).__name__})"
    try:
        kind = json.loads(raw.splitlines()[0]).get("kind")
    except Exception:
        kind = None
    if kind in ("answer", "notice"):
        return None
    if kind == "unfinished_notice":
        return "a shim submitted it and ended before its outcome; whether the turn ran is unknown"
    return "its journal entry holds no complete record"


def _wait_for_turn_slot(prompt, session):
    """True once this process's turn slot is held. A wait writes ``canonical_wait`` (it resets
    buzz-acp's idle timeout and is never published) and ends at a cancel or the busy deadline."""
    deadline = time.monotonic() + BUSY_DEADLINE_SECS
    attempt = 0
    while not _TURN_SLOT.acquire(timeout=max(min(BUSY_RETRY_SECS, deadline - time.monotonic()), 0.2)):
        if (prompt is not None and prompt.cancelled.is_set()) or time.monotonic() >= deadline:
            return False
        attempt += 1
        write({"jsonrpc": "2.0", "method": "session/update", "params": {
            "sessionId": session, "update": {"sessionUpdate": "canonical_wait", "attempt": attempt}}})
    return True


def begin_submission(session, base, channel):
    """Take the turn slot and the event's in-flight entry before *base* is submitted. False when it must
    not be submitted: already in flight here or in another live shim process, already submitted by a
    shim that ended without its outcome (``_unfinished``), cancelled while it waited, the slot still taken
    at the busy deadline, or the entry not provably this process's. Past this point a cancel never
    recalls it. Refusing a resubmission is no execution or delivery guarantee: a turn whose shim ended
    stays unknown, and is told once as such by the reaper."""
    prompt = getattr(_local, "prompt", None)
    with _INFLIGHT_LOCK:
        if base in _INFLIGHT:
            log(f"prompt {base[:12]}… already in flight in this process; not submitted again")
            return False
        _INFLIGHT.add(base)
    taken, journal, ok = False, None, False
    try:
        taken = _wait_for_turn_slot(prompt, session)
        if not taken:
            log(f"prompt {base[:12]}… NOT submitted: " + (
                "cancelled while an earlier turn held this process's turn slot"
                if prompt is not None and prompt.cancelled.is_set()
                else f"an earlier turn still held this process's turn slot after {BUSY_DEADLINE_SECS:.0f}s"))
            return False
        journal, why = _claim_inflight(base, channel)
        if journal is None:
            log(f"prompt {base[:12]}… NOT submitted: {why}")
            return False
        why = _unfinished(base)
        if why is None and not _same_file(journal, os.path.join(PUBLISHED, base + INFLIGHT_SUFFIX)):
            why = "this process no longer holds the event's in-flight entry"
        if why is not None:
            log(f"prompt {base[:12]}… NOT submitted again: {why}")
            return False
        if prompt is not None:
            with prompt.lock:
                if prompt.answered and not (prompt.batch and prompt.submitted is not None):
                    log(f"prompt {base[:12]}… NOT submitted: cancelled before submission")
                    return False
                prompt.submitted = base
        _local.submission = {"event": base, "fd": journal, "slot": True}
        ok = True
        return True
    finally:
        if not ok:
            if journal is not None:
                _drop_inflight(base, journal)
            if taken:
                _TURN_SLOT.release()
            with _INFLIGHT_LOCK:
                _INFLIGHT.discard(base)


def _drop_inflight(base, fd):
    path = os.path.join(PUBLISHED, base + INFLIGHT_SUFFIX)
    try:
        if _same_file(fd, path):
            os.unlink(path)
    except OSError:
        pass
    finally:
        os.close(fd)


def release_turn_slot():
    """The canonical surface answered (or no answer will come): another prompt may submit."""
    submission = getattr(_local, "submission", None)
    if submission is not None and submission["slot"]:
        submission["slot"] = False
        _TURN_SLOT.release()


def finish_submission():
    """The event's outcome is posted or journaled: it is no longer in flight."""
    submission = getattr(_local, "submission", None)
    if submission is None:
        return
    release_turn_slot()
    _local.submission = None
    if submission["fd"] is not None:
        _drop_inflight(submission["event"], submission["fd"])
    with _INFLIGHT_LOCK:
        _INFLIGHT.discard(submission["event"])


def abandon_submission():
    """A prompt thread ended before its event's outcome was posted (its handler raised): the slot and
    the lock go, the entry stays linked, so ``reap_unfinished`` tells the thread. True when one stays."""
    submission = getattr(_local, "submission", None)
    if submission is None:
        return False
    release_turn_slot()
    _local.submission = None
    if submission["fd"] is not None:
        os.close(submission["fd"])
    with _INFLIGHT_LOCK:
        _INFLIGHT.discard(submission["event"])
    return submission["fd"] is not None


def _prohibition_persisted(event_id):
    """Whether ``PUBLISHED/<event id>`` now holds the record that refuses this event's resubmission once its
    ``.inflight`` entry is gone (``_unfinished``): readable, complete, naming this very event, of kind
    ``unfinished_notice``, and flushed to disk with its directory entry. A file that merely exists — empty,
    cut short, unreadable, another event's, or any other kind — is not that record. Whether the notice was
    posted is a separate fact: the record may say it was not, and still refuses the resubmission."""
    try:
        fd = os.open(os.path.join(PUBLISHED, outcome_key(event_id)), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            raw = os.read(fd, 65_536)
            record = json.loads(raw.decode("utf-8").splitlines()[0])
            if not (isinstance(record, dict) and record.get("event_id") == event_id
                    and record.get("kind") == "unfinished_notice"):
                return False
            os.fsync(fd)
        finally:
            os.close(fd)
        directory = os.open(PUBLISHED, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        return False
    return True


def reap_unfinished():
    """Tell the thread, once, about each event a shim process submitted and never finished delivering:
    its ``.inflight`` entry is still linked and no live process holds its lock (the owner was replaced
    — buzz-acp SIGKILLs the group on CancelDrainTimeout — or its handler raised). The notice takes the
    event's terminal post (``outcome_key``), so it never goes out beside or after an answer; an
    event that already has its post keeps no second one. The entry is retired only once the event's
    ``unfinished_notice`` record is persisted (``_prohibition_persisted``); until then — the journal could
    not be created or written, or the entry holds another kind — it stays as the guard against a
    resubmission, an explicit unresolved state, and a later reap tries again. Best effort, not crash
    safety: it runs when a shim starts and before each prompt it takes."""
    try:
        names = sorted(name for name in os.listdir(PUBLISHED) if INFLIGHT_RE.fullmatch(name))
    except OSError:
        return
    for name in names:
        path, event_id = os.path.join(PUBLISHED, name), name[:-len(INFLIGHT_SUFFIX)]
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW)
        except OSError:
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if not _same_file(fd, path):
                    continue  # finished and unlinked meanwhile
                raw = os.read(fd, 4096)
            except OSError:
                continue  # a live process holds it, or it is gone
            try:
                record = json.loads(raw.decode())
            except Exception:
                record = None
            channel = record.get("channel") if isinstance(record, dict) else None
            if not os.path.lexists(os.path.join(PUBLISHED, outcome_key(event_id))):
                try:
                    if isinstance(channel, str) and re.fullmatch(_UUID, channel):
                        log(f"prompt {event_id[:12]}… was submitted by a shim that ended before its outcome; "
                            "telling the thread")
                        publish_reply(event_id, channel, UNFINISHED_NOTICE.format(event_id=event_id),
                                      kind="unfinished_notice")
                    else:
                        marker, _ = _claim(outcome_key(event_id))
                        if marker is not None:
                            with os.fdopen(marker, "w") as record_file:
                                record_file.write(json.dumps({
                                    "sent": False, "state": "notice_impossible", "delivery": "failed",
                                    "kind": "unfinished_notice", "event_id": event_id, "channel": None,
                                    "reason": "the in-flight entry names no channel",
                                    "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
                except Exception as exc:  # checked below: an entry claimed and not filled is no record
                    log(f"prompt {event_id[:12]}… unfinished; its journal entry was not completed: "
                        f"{type(exc).__name__}")
            # Until the record that refuses a resubmission is persisted, the in-flight entry is the only guard
            # (``_claim_inflight`` refuses a taken name). It is retired only once that record is read back and
            # flushed; otherwise it stays — an explicit unresolved state, never success or delivery — and a
            # later reap tries again.
            if not _prohibition_persisted(event_id):
                log(f"prompt {event_id[:12]}… UNRESOLVED: unfinished, and no resubmission-refusing journal record "
                    "could be persisted; its in-flight entry stays, so it is not submitted again")
                continue
            try:
                os.unlink(path)
            except OSError:
                pass
        finally:
            os.close(fd)


def cancel_prompts(message):
    """``session/cancel`` (ACP: a notification). A prompt not yet submitted is cancelled
    (``stopReason: cancelled``) and never submitted. A prompt whose event is already at the canonical
    surface cannot be recalled, so it is NOT reported cancelled: it is answered ``end_turn`` with
    ``_meta.canonical`` saying the turn goes on, and its own thread still posts the outcome once — the
    answer, or a failure notice. buzz-acp treats any response inside its drain alike (pool.rs: Ok of
    any stop reason → the batch requeued, this agent kept); none within 5 s is CancelDrainTimeout,
    which SIGKILLs this process and loses the running turn's answer."""
    params = message.get("params") or {}
    session = params.get("sessionId")
    with _PROMPTS_LOCK:
        prompts = [p for p in _PROMPTS.values() if session is None or p.session == session]
    if not prompts:
        log("session/cancel received; no prompt in flight")
    for prompt in prompts:
        with prompt.lock:
            if prompt.answered:
                continue
            prompt.answered = True
            if prompt.submitted is None:
                prompt.cancelled.set()
                value = {"stopReason": "cancelled"}
                log(f"session/cancel: prompt {prompt.rpc_id} cancelled before anything was submitted")
            else:
                value = {"stopReason": "end_turn", "_meta": {"canonical": {
                    "cancelled": False, "turn": "running", "event_id": prompt.submitted,
                    "outcome": "posted once to the thread by this shim: the answer or a failure notice"}}}
                log(f"session/cancel: prompt {prompt.rpc_id} NOT cancelled: event {prompt.submitted[:12]}… is "
                    "at the canonical surface and goes on; answered end_turn, its outcome is posted once")
            _emit({"jsonrpc": "2.0", "id": prompt.rpc_id, "result": value})
    if message.get("id") is not None:
        result(message.get("id"), {})


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


# A transient refusal before any claim (#99 r1 H99-02): the event is held and the ACP prompt fails so
# buzz-acp requeues and resends the same event id; the gateway has recorded nothing for it, so the
# resend runs it once. Ending the turn instead would count the event as delivered and lose it.
RETRY = "retry"


def deliver_event(session, parsed, text, mentioned):
    """Run one attributed event through the existing safety, retry and receipt path. Returns RETRY when
    the event must be resent later (nothing was claimed), otherwise None."""
    author, channel, base, body = parsed
    # One snapshot decides this event (#100 r3 H100-R3-01): reading the roster again for the signers or
    # the room's counterpart could fail between reads and silently drop an owner broadcast.
    snapshot, why = _read_roster()
    _local.roster_generation = snapshot.generation if snapshot is not None else None
    if snapshot is None and not mentioned and author == OWNER:
        journal_held(base, channel, "owner broadcast held: routing roster unavailable; ACP retry requested")
        return RETRY
    if snapshot is not None and channel in snapshot.channels and snapshot.active(channel) is None:
        log(f"prompt {base[:12]}… refused: retired room; roster_generation={snapshot.generation}")
        return
    if snapshot is None:
        signers, unknown = None, why
    else:
        signers = {rule["ceo"]["pubkey"] for rule in snapshot.channels.values()
                   if rule["status"] == "active"} or None
        unknown = None if signers else "reply registry names no valid ceo signer"
    if not mentioned:
        if (author != OWNER or snapshot is None or snapshot.active(channel) is None
                or os.path.exists(os.path.join(PUBLISHED, outcome_key(base, channel)))):
            return
    log(f"prompt author={author[:12]}… channel={channel} base={base[:12]}… chars={len(text)} "
        f"roster_generation={getattr(_local, 'roster_generation', None)}")
    # Every refusal is decided before anything is claimed, so an armed canary is spent only on an
    # event that is actually submitted — not on an echo, an unreadable registry or an oversize text.
    if FORWARD != "live" and not canary_armed_for(author, channel, body):
        hold(session, author, channel, base, text)
        return
    if signers is None:
        # The bridge's own identities are unreadable, so an echo cannot be ruled out: nothing is
        # claimed, and the prompt fails so buzz-acp resends this event once they are readable again
        # (#99 r2 H99-02); ending the turn would count the event as delivered.
        log(f"prompt {base[:12]}… HELD, not sent: {unknown}; ACP retry requested")
        journal_held(base, channel, unknown)
        return RETRY
    if author in signers:
        # This bridge's own post — an answer or a not-delivered notice — coming back. buzz-acp drops
        # it first (ignore-self, the author gate) and the binding's author allowlist
        # refuses it after, but nothing here may depend on either: forwarded, it would reach the CEO
        # as a turn and be answered into the thread under a new event id, which no per-event marker
        # stops. (A registry that cannot be read cannot rule an echo out: held above, nothing posted —
        # the wrapper refuses to post without it.)
        log(f"prompt {base[:12]}… refused locally: authored by this bridge's own signer; not forwarded")
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
        publish_reply(base, channel, notice, kind="notice", preclaim=True)
        return
    if FORWARD != "live":
        if not claim_canary(author, channel, body):
            hold(session, author, channel, base, text)  # disarmed, expired or taken in between
            return
        log(f"prompt {base[:12]}… CANARY: forwarding this one armed event (CANONICAL_FORWARD={FORWARD})")
    if fitted[0] is not text:
        log(f"prompt {base[:12]}… {len(text)} chars is over the gateway limit {GATEWAY_TEXT_LIMIT}: "
            f"omitted {fitted[1]} thread messages ({fitted[2]} chars), submitting {len(fitted[0])} chars")
    text = fitted[0]
    if not begin_submission(session, base, channel):
        return
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
    preclaim = False
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
            if exc.code == 503 and code == "canonical_roster_unavailable":
                log(f"prompt {event_id} refused HTTP 503 canonical_roster_unavailable before any claim; "
                    "held for ACP retry")
                release_turn_slot()
                journal_held(base, channel, "gateway routing roster unavailable; ACP retry requested")
                finish_submission()
                return RETRY
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
            preclaim = ((exc.code, code) in ((400, "canonical_invalid_request"),
                                           (403, "canonical_principal_rejected")))
            break
        except Exception as exc:
            log(f"prompt {event_id} failed: {type(exc).__name__}: {exc}")
            answer = f"canonical surface unreachable: {type(exc).__name__}: {exc}"
            # No gateway answer: the turn may have run, so the thread is told its outcome is unknown.
            code, refusal = None, {}
            notice = unknown_outcome_notice(type(exc).__name__, base)
            break
    release_turn_slot()  # the gateway answered, or never will: another prompt may submit
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
        publish_reply(base, channel, notice, kind="notice", preclaim=preclaim)
    finish_submission()
    return


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
            # Rendered bundle headers cannot establish identity. Recover signed originals separately;
            # an unrecognized or ambiguous framing retains the existing anonymous refusal path.
            recovered = recover_batch(session, prompt)
            if recovered == RETRY:
                error(rpc_id, -32000, "retryable: canonical_roster_unavailable (nothing claimed; resend)")
                return
            if recovered:
                result(rpc_id, {"stopReason": "end_turn"})
                return
            refuse_unattributed(session, prompt, len(text))
            result(rpc_id, {"stopReason": "end_turn"})
            return
        lines = prompt_blocks(prompt)[-1].split("\n")
        tags = _tags_at_end(lines)
        signers, _ = _own_signers()
        mentioned = lines[0] == "[Buzz event: @mention]" or (
            tags is not None and signers is not None
            and any(len(tag) >= 2 and tag[0] == "p" and tag[1] in signers for tag in tags))
        if deliver_event(session, parsed, text, mentioned) == RETRY:
            error(rpc_id, -32000, "retryable: canonical_roster_unavailable (nothing claimed; resend)")
            return
        result(rpc_id, {"stopReason": "end_turn"})
        return

    if method == "session/cancel":
        # A turn already handed to the CEO cannot be recalled, so it is never reported cancelled.
        cancel_prompts(message)
        return

    if rpc_id is not None:
        error(rpc_id, -32601, f"method not supported by acp-canonical-shim: {method}")


def _serve_prompt(prompt, message):
    """One prompt's thread: tell the thread about any unfinished event first, then handle the prompt."""
    _local.prompt = prompt
    try:
        reap_unfinished()
        handle(message)
    except Exception as exc:
        log(f"handler crashed on session/prompt: {type(exc).__name__}: {exc}")
        if prompt.rpc_id is not None:
            error(prompt.rpc_id, -32603, f"{type(exc).__name__}: {exc}")
    finally:
        left = abandon_submission()
        with _PROMPTS_LOCK:
            _PROMPTS.pop(id(prompt), None)
        _local.prompt = None
        if left:
            reap_unfinished()


def dispatch(message):
    """The stdin loop's one call per line. It never waits on a canonical turn: each prompt runs on its
    own thread, so the next line — a ``session/cancel`` above all — is read while the turn runs."""
    method = message.get("method")
    if method == "session/cancel":
        cancel_prompts(message)
        return
    if method != "session/prompt":
        handle(message)
        return
    prompt = PromptTurn(message.get("id"), (message.get("params") or {}).get("sessionId"))
    with _PROMPTS_LOCK:
        _PROMPTS[id(prompt)] = prompt
    thread = threading.Thread(target=_serve_prompt, args=(prompt, message), name=f"prompt-{message.get('id')}")
    _THREADS[:] = [running for running in _THREADS if running.is_alive()] + [thread]
    thread.start()


_EVENT_ID = re.compile(r"[0-9a-f]{64}")


def recover_main(args):
    """``recover <channel-uuid> <event-id>...``: deliver events this shim refused or never received.

    For operators after an outage or a refused merged prompt (2026-10-20: the owner's thread reply
    was refused unattributed). Each original is fetched from the relay once and its hash, Schnorr
    signature and channel tag are verified before the existing per-event delivery runs, so the
    signed author, never this command line, supplies identity. An event already answered is
    skipped, so rerunning never duplicates. One line per event reports its final local state.
    """
    if len(args) < 2:
        sys.stderr.write("usage: acp-canonical-shim.py recover <channel-uuid> <event-id> [<event-id> ...]\n")
        return 2
    channel = args[0].lower()
    if not re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", channel):
        sys.stderr.write(f"acp-canonical-shim: not a channel uuid: {args[0]}\n")
        return 2
    ids = list(dict.fromkeys(arg.lower() for arg in args[1:]))
    bad = [event_id for event_id in ids if not _EVENT_ID.fullmatch(event_id)]
    if bad:
        sys.stderr.write("acp-canonical-shim: not an event id: " + ", ".join(bad) + "\n")
        return 2
    missing = [name for name, value in (("API_SERVER_KEY", KEY),
                                        ("BUZZ_RELAY_URL", os.environ.get("BUZZ_RELAY_URL")),
                                        ("BUZZ_PRIVATE_KEY", os.environ.get("BUZZ_PRIVATE_KEY"))) if not value]
    if missing or _batch is None:
        sys.stderr.write("acp-canonical-shim: recovery unavailable: "
                         + (("missing " + ", ".join(missing)) if missing else _batch_error) + "\n")
        return 2
    from types import SimpleNamespace

    global _EXPLICIT_RECOVERY
    _EXPLICIT_RECOVERY = True
    session = "canonical:" + BINDING
    api = SimpleNamespace(**globals())
    recovered = {}
    # No buzz-acp reads this process's ACP frames; the answer is posted to Buzz and journaled as in a
    # live turn. Stdout carries only the per-event report.
    report, sys.stdout = sys.stdout, open(os.devnull, "w")

    def lookup(event_id):
        if event_id not in recovered:
            try:
                recovered[event_id] = _batch.fetch_event(event_id, channel)
            except Exception as exc:
                recovered[event_id] = exc
        return recovered[event_id]

    status = 0
    for event_id in ids:
        log(f"recover {event_id} channel={channel}")
        prompt = [{"type": "text", "text": f"[Buzz event: recovered]\nEvent ID: {event_id}\nChannel ID: {channel}"}]
        _local.prompt = PromptTurn(None, session)
        try:
            _batch.deliver_member(api, session, prompt, channel, event_id, lookup)
        except Exception as exc:
            status = 1
            log(f"recover {event_id} failed: {_batch.failure_reason(exc)}")
            sys.stderr.write(f"acp-canonical-shim: recover {event_id} failed: {_batch.failure_reason(exc)}\n")
        finally:
            left = abandon_submission()
            _local.prompt = None
            if left:
                reap_unfinished()
        found = recovered.get(event_id)
        state = recovery_delivery_state(event_id)
        if isinstance(found, Exception):
            state = "unavailable: " + _batch.failure_reason(found)
            status = 1
        print(f"{event_id} {state}", file=report, flush=True)
    return status


def main():
    if sys.argv[1:2] == ["recover"]:
        raise SystemExit(recover_main(sys.argv[2:]))
    missing = [name for name, value in
               (("API_SERVER_KEY", KEY),)
               if not value]
    if missing:
        # Fail before ACP initialize: a shim that answers initialize and then refuses every turn
        # looks to buzz-acp like a working agent producing empty replies.
        sys.stderr.write("acp-canonical-shim: missing " + ", ".join(missing) + "\n")
        raise SystemExit(2)
    log(f"started binding={BINDING} base={BASE}")
    reaper = threading.Thread(target=reap_unfinished, name="reap-unfinished")
    _THREADS.append(reaper)
    reaper.start()
    # `for line in sys.stdin` reads ahead into an internal buffer, so a peer that sends one line
    # and keeps the pipe open can have that line withheld until the buffer fills. A pipe closed by
    # the test harness flushes it and hides this entirely; buzz-acp holds the pipe open, and the
    # shim sat through `initialize` without ever seeing it. readline() returns on the newline.
    while True:
        line = sys.stdin.readline()
        if not line:
            running = [thread for thread in _THREADS if thread.is_alive()]
            log("stdin closed; exiting" + (f" once {len(running)} running thread(s) finish" if running else ""))
            for thread in running:
                thread.join()
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
            dispatch(message)
        except Exception as exc:
            log(f"handler crashed on {message.get('method')}: {type(exc).__name__}: {exc}")
            if message.get("id") is not None:
                error(message.get("id"), -32603, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
