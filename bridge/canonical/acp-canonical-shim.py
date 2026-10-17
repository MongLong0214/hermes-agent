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
# read from the event and the env values serve only as a fallback that is logged as such.
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
HEADER_EVENT_RE = re.compile(r"Event ID: ([0-9a-f]{64})")
HEADER_CHANNEL_RE = re.compile(r"Channel: [^()#]* \(#(" + _UUID + r")\)")
HEADER_FROM_RE = re.compile(r"From: [^()]* \(npub: npub1[02-9ac-hj-np-z]+, hex: ([0-9a-f]{64})\)")


def envelope(text):
    """(author_id, channel_id, event_id, body) of the one ``[Buzz event: …]`` section buzz-acp appends
    after its context blocks, or None. The thread context above it quotes other members' ``hex:``
    ids and their text, so signer, channel and id are read only from that section's own header lines
    (each exactly once, before ``Content:``) and the body only after ``Content:``. A second section
    header (a body imitating one) or any missing field makes the event ambiguous: None, never a guess
    and never the env fallback."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(EVENT_HEADER)]
    if len(starts) != 1:
        return None
    section = lines[starts[0] + 1:]
    content = next((i for i, line in enumerate(section) if line.startswith("Content:")), None)
    if content is None:
        return None
    head = section[:content]

    def field(prefix, pattern, *, token=None):
        # The whole line must have buzz-acp's exact shape, carrying one value: a line with a second
        # id, an extra parenthesis or any other variation is ambiguous and refuses the event. For
        # ``From:``, buzz-acp never puts a second ``hex:`` anywhere on the line (CANONICAL-01): a
        # display name quoting one, however it is placed, makes the line ambiguous on its own,
        # independent of which occurrence the pattern's own group would have captured.
        matches = [line for line in head if line.startswith(prefix)]
        if len(matches) != 1:
            return None
        line = matches[0].rstrip()
        if token is not None and line.count(token) != 1:
            return None
        found = pattern.fullmatch(line)
        return found.group(1) if found else None

    event_id = field("Event ID:", HEADER_EVENT_RE)
    channel = field("Channel:", HEADER_CHANNEL_RE)
    author = field("From:", HEADER_FROM_RE, token="hex:")
    if not (event_id and channel and author):
        return None
    body = "\n".join([section[content][len("Content:"):]] + section[content + 1:]).strip()
    return author, channel, event_id, body


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
    full, its tags — are never touched, so ``envelope()`` reads the same event from the result. The
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


def submit(text, event_id, author=None, channel=None):
    """Answer text, or raise. The caller turns a failure into a visible refusal sentence."""
    body = json.dumps({
        "binding": BINDING, "event_id": event_id,
        "author_id": author or AUTHOR, "channel_id": channel or CHANNEL, "text": text}).encode()
    request = urllib.request.Request(
        BASE + ROUTE, data=body, method="POST",
        headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode()).get("text", "")


def _peer_handle(channel):
    try:
        with open(MENTIONS) as handle:
            return json.load(handle)["channels"][channel]["peer"]["handle"]
    except (OSError, ValueError, KeyError, TypeError):
        return None  # the wrapper refuses an unregistered channel, and adds the handle itself


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


def publish_reply(event_id, channel, answer, kind="answer"):
    """Post *answer* once as the thread reply to *event_id*. True only when the wrapper sent it.

    The delivery journal is ``PUBLISHED/<event id>``: claimed before the send (one post per event,
    ever) and then filled with what happened — ``state`` is ``sent``, ``wrapper_refused``,
    ``send_failed``, ``timeout`` or ``error``, with the ``reason``. A failure stays recorded as one
    and is never retried."""
    try:
        os.makedirs(PUBLISHED, exist_ok=True)
        fd = os.open(os.path.join(PUBLISHED, event_id), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        log(f"prompt {event_id[:12]}… reply already published once; not posting again")
        return False
    except OSError as exc:
        log(f"prompt {event_id[:12]}… reply not posted: marker unavailable: {type(exc).__name__}: {exc}")
        return False
    handle = _peer_handle(channel)
    body = f"@{handle} {answer}" if handle and not answer.lstrip().startswith(f"@{handle}") else answer
    try:
        done = subprocess.run(
            [BUZZ_CLI, "messages", "send", "--channel", channel, "--reply-to", event_id, "--content", "-"],
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
        marker.write(json.dumps({"sent": sent, "state": state, "kind": kind, "event_id": event_id,
                                 "channel": channel, "reason": str(reason)[:400],
                                 "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    log(f"prompt {event_id[:12]}… {kind} {'POSTED' if sent else 'NOT posted (no retry)'} to thread: "
        f"{state}: {reason}")
    return sent


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
        text = prompt_text(params)
        if not text:
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
        # (3765자)가 거부됐다. heartbeat 는 자기 섹션 머리글을 한 줄로 갖는다 — 줄에 앵커를 둔다.
        if any(ln.strip() == "[System: Heartbeat]" for ln in text.splitlines()):
            log(f"prompt refused locally: heartbeat, not a message ({len(text)} chars)")
            result(rpc_id, {"stopReason": "end_turn"})
            return
        parsed = envelope(text)
        if parsed is None:
            # No single unambiguous event: nothing can be attributed, so nothing is forwarded in any mode.
            log(f"prompt refused locally: no unambiguous [Buzz event] envelope ({len(text)} chars)")
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
