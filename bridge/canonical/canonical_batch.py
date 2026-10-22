"""Recover batch members by authenticated event ID, never by a rendered sender header.

Installed beside acp-canonical-shim.py. Requires coincurve (see requirements.txt).
Relay reads are one-shot NIP-98 authenticated /query requests triggered by an incoming prompt.
"""
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime
from types import SimpleNamespace


def failure_reason(exc):
    # Import diagnostics name the missing dependency; other errors may contain untrusted bodies.
    return f"{type(exc).__name__}: {exc}" if isinstance(exc, ImportError) else type(exc).__name__


def event_digest(event):
    wire = [0, event["pubkey"], event["created_at"], event["kind"], event["tags"], event["content"]]
    return hashlib.sha256(json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode()).digest()


def verify_event(event, event_id, channel):
    from coincurve import PublicKeyXOnly

    if not (isinstance(event, dict) and event.get("id") == event_id
            and isinstance(event.get("pubkey"), str) and re.fullmatch(r"[0-9a-f]{64}", event["pubkey"])
            and type(event.get("created_at")) is int and event["created_at"] >= 0
            and type(event.get("kind")) is int and event["kind"] in (9, 46010)
            and isinstance(event.get("content"), str) and isinstance(event.get("tags"), list)
            and all(isinstance(tag, list) and all(isinstance(v, str) for v in tag) for tag in event["tags"])
            and {tag[1] for tag in event["tags"] if len(tag) >= 2 and tag[0] == "h"} == {channel}
            and isinstance(event.get("sig"), str) and re.fullmatch(r"[0-9a-f]{128}", event["sig"])):
        raise ValueError("event identity, kind or channel mismatch")
    digest = event_digest(event)
    if digest.hex() != event_id or not PublicKeyXOnly(bytes.fromhex(event["pubkey"])).verify(
            bytes.fromhex(event["sig"]), digest):
        raise ValueError("invalid event hash or signature")
    return event


def fetch_event(event_id, channel):
    """One bounded request; neither a subscription nor a retry/polling loop."""
    from coincurve import PrivateKey

    relay = urllib.parse.urlsplit(os.environ["BUZZ_RELAY_URL"])
    scheme = {"ws": "http", "wss": "https", "http": "http", "https": "https"}.get(relay.scheme)
    if scheme is None or not relay.netloc or relay.username or relay.query or relay.fragment:
        raise ValueError("invalid relay URL")
    url = urllib.parse.urlunsplit((scheme, relay.netloc, relay.path.rstrip("/") + "/query", "", ""))
    data = json.dumps([{"ids": [event_id], "limit": 1}], separators=(",", ":")).encode()
    signer = PrivateKey(bytes.fromhex(os.environ["BUZZ_PRIVATE_KEY"]))
    auth = {"pubkey": signer.public_key_xonly.format().hex(), "created_at": int(time.time()),
            "kind": 27235, "content": "", "tags": [["u", url], ["method", "POST"],
            ["nonce", os.urandom(16).hex()], ["payload", hashlib.sha256(data).hexdigest()]]}
    digest = event_digest(auth)
    auth.update(id=digest.hex(), sig=signer.sign_schnorr(digest).hex())
    headers = {"Content-Type": "application/json", "Authorization": "Nostr " + base64.b64encode(
        json.dumps(auth, ensure_ascii=False, separators=(",", ":")).encode()).decode()}
    if os.environ.get("BUZZ_AUTH_TAG"):
        headers["x-auth-tag"] = os.environ["BUZZ_AUTH_TAG"]
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("relay response too large")
    events = json.loads(raw)
    if not isinstance(events, list) or len(events) != 1:
        raise ValueError("event not found uniquely")
    return verify_event(events[0], event_id, channel)


def rendered_end(lines, at, event_id):
    """Find the Tags line committed by the ID at an already-proven producer position.

    queue.rs format_event_block renders pubkey, timestamp, kind, content and JSON tags: all
    NIP-01 hash inputs. A quoted Tags/delimiter cannot shorten this body without a SHA-256
    second preimage. This proves only the boundary; delivery still requires the signed original.
    Labels (including a label's own 'hex:') are excluded by taking the terminal pubkey field.
    """
    try:
        author = re.search(r"hex: ([0-9a-f]{64})\)$", lines[at + 3]).group(1)
        stamp = lines[at + 4][len("Time: "):]
        created = int(stamp) if stamp.isdecimal() else int(datetime.fromisoformat(
            stamp.replace("Z", "+00:00")).timestamp())
        event = {"pubkey": author, "created_at": created,
                 "kind": int(lines[at + 2][len("Kind: "):])}
        if not lines[at + 5].startswith("Content: "):
            return None
        for end in range(at + 6, len(lines)):
            if not lines[end].startswith("Tags: "):
                continue
            try:
                tags = json.loads(lines[end][len("Tags: "):])
            except (ValueError, RecursionError):
                continue
            event.update(content="\n".join(lines[at + 5:end])[len("Content: "):], tags=tags)
            if event_digest(event).hex() == event_id:
                return end + 1
    except (ValueError, OverflowError, AttributeError, IndexError):
        return None
    return None


def recover_members(api, section, channel_line, lookup):
    """IDs at confirmed positions, or a walk bounded by signed content or a rendered hash match.

    Cancelled lists have no declared count. A delimiter inside a body cannot end an event: the walk
    consumes the content and JSON tags committed by that event's ID before the next delimiter.
    Declared batches also use this walk when a quoted delimiter makes the older ID parser ambiguous.
    """
    lines = section.split("\n")
    confirmed = api._confirmed_sections(lines, channel_line)
    if confirmed:
        return [event_id for _, event_id in confirmed], False
    ids, at, number = [], 3, 1
    declared = api.EVENT_COUNT_RE.fullmatch(lines[0])
    count = int(declared.group(1)) if declared else None
    while at + 5 < len(lines):
        head = api.SECTION_RE.fullmatch(lines[at - 1])
        found = api.HEADER_EVENT_RE.fullmatch(lines[at])
        if not (lines[at - 2] == "" and head and int(head.group(1)) == number
                and found and lines[at + 1] == channel_line):
            break
        event_id = found.group(1)
        ids.append(event_id)
        event = lookup(event_id)
        if isinstance(event, Exception):
            end = rendered_end(lines, at, event_id)
        else:
            tail = ("Content: " + event["content"] + "\nTags: "
                    + json.dumps(event["tags"], ensure_ascii=False, separators=(",", ":"))).split("\n")
            end = at + 5 + len(tail) if lines[at + 5:at + 5 + len(tail)] == tail else None
        if end is None:
            break
        if end < len(lines) and lines[end].startswith("Parsed: "):
            end += 1
        if end == len(lines) and (count is None or count == number):
            return ids, False
        if count is not None and number >= count:
            break
        at, number = end + 2, number + 1
    # A count proves that a unique candidate for each other number is producer-written. Even when
    # one original cannot be fetched, those siblings can still be recovered independently.
    unresolved = True
    if count is not None:
        by_number = {}
        for k, where in api._section_candidates(lines, channel_line):
            by_number.setdefault(k, []).append(where)
        for k in range(number + 1, count + 1):
            positions = by_number.get(k, [])
            if len(positions) == 1:
                ids.append(api.HEADER_EVENT_RE.fullmatch(lines[positions[0]]).group(1))
        unresolved = any(len(by_number.get(k, [])) != 1 for k in range(number + 1, count + 1))
    return list(dict.fromkeys(ids)), unresolved


def already_delivered(api, event_id):
    return api.recovery_delivery_state(event_id) == "answered"


def deliver_batch(namespace, session, prompt):
    api = SimpleNamespace(**namespace)
    # The relay URL is inherited from buzz-acp, never inferred from a message or a default live URL.
    if not os.environ.get("BUZZ_RELAY_URL"):
        return False
    batch = api.batch_sections(prompt)
    if batch is None:
        return False
    channel, channel_line, sections = batch
    recovered = {}

    def lookup(event_id):
        if event_id not in recovered:
            try:
                recovered[event_id] = fetch_event(event_id, channel)
            except Exception as exc:
                recovered[event_id] = exc
        return recovered[event_id]

    ids = []
    unresolved = False
    for section in sections:
        members, incomplete = recover_members(api, section, channel_line, lookup)
        ids.extend(members)
        unresolved |= incomplete
    turn = getattr(api._local, "prompt", None)
    if turn is not None:
        turn.batch = True
    for event_id in dict.fromkeys(ids):
        try:
            deliver_member(api, session, prompt, channel, event_id, lookup)
        except Exception as exc:
            # Isolate lookup, refusal and publication failures too. A submitted event retains the
            # existing unknown-outcome journal; its siblings must still get their own attempt.
            reason = failure_reason(exc)
            api.log(f"prompt {event_id} delivery failed: {reason}")
            print(f"acp-canonical-shim: event {event_id} delivery failed: {reason}",
                  file=sys.stderr, flush=True)
            if api.abandon_submission():
                api.reap_unfinished()
            else:
                api.journal_held(event_id, channel, "recovered event delivery failed")
    if unresolved:
        ref, _ = api.refusal_key(prompt)
        api.journal_undeliverable(ref, "event_recovery", "some event boundaries could not be authenticated; "
                                  "confirmed siblings were processed independently")
    return True


def deliver_member(api, session, prompt, channel, event_id, lookup):
    if already_delivered(api, event_id):
        api.log(f"prompt {event_id} already delivered; skipping recovered event")
        return
    event = lookup(event_id)
    if isinstance(event, Exception):
        api.refuse_recovery_event(session, prompt, event_id, failure_reason(event))
        return
    # No bundle context or framing instructions cross this boundary. Everything here is signed.
    tags = event["tags"]
    root, _ = api._thread_of(tags)
    text = (f"[Buzz event: recovered]\nEvent ID: {event_id}\nChannel ID: {channel}\n"
            f"Author ID: {event['pubkey']}\n"
            + (f"Thread root: {root}\n" if root else "")
            + f"Content: {event['content']}\nTags: "
            + json.dumps(tags, ensure_ascii=False, separators=(",", ":")))
    signers, _ = api._own_signers()
    mentioned = signers is not None and any(
        len(tag) >= 2 and tag[0] == "p" and tag[1] in signers for tag in tags)
    api.deliver_event(session, (event["pubkey"], channel, event_id, event["content"]), text, mentioned)
