"""buzz-acp's own prompt shapes, ported line for line, for the canonical shim's tests (not product code).

Source: ~/projects/buzz, commit 6a17d035f (2026-08-08, the last buzz-acp change before the deployed
binary ~/.hermes/bridge/bin/buzz-acp, built 2026-08-09; its strings carry every header used here).
``crates/buzz-acp/src/queue.rs`` at that commit:

* ``format_event_block``          queue.rs:1081-1147
* ``append_reply_instruction``    queue.rs:1153-1161
* ``append_new_thread_reply_instruction`` queue.rs:1169-1177
* ``format_context_hints``        queue.rs:1238-1319 (thread and channel branches)
* ``format_conversation_context`` queue.rs:1322-1353
* ``format_prompt``               queue.rs:1411-1572 (merged + multi-event shapes 1503-1569)
* ``MergeFraming``                queue.rs:1576-1615
* ``acp.rs:1970 build_prompt_params``: params are ``{"sessionId", "prompt": [text blocks]}`` only.

The #1111 diagnosis (D1) checked this port byte for byte against the live
single-event prompt and the first element of a live thread prompt; the probe is not part of this tree.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

_BECH32 = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _polymod(values):
    gen = (0x3B6A57B2, 0x26508E6D, 0x1EA119FA, 0x3D4233DD, 0x2A1462B3)
    chk = 1
    for v in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ v
        for i in range(5):
            chk ^= gen[i] if ((top >> i) & 1) else 0
    return chk


def npub(hex_pubkey: str) -> str:
    """NIP-19 ``npub`` of a 32-byte hex pubkey (bech32, as ``PublicKey::to_bech32``)."""
    data, acc, bits = [], 0, 0
    for byte in bytes.fromhex(hex_pubkey):
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            data.append((acc >> bits) & 31)
    if bits:
        data.append((acc << (5 - bits)) & 31)
    hrp = "npub"
    expanded = [ord(c) >> 5 for c in hrp] + [0] + [ord(c) & 31 for c in hrp]
    poly = _polymod(expanded + data + [0] * 6) ^ 1
    checksum = [(poly >> 5 * (5 - i)) & 31 for i in range(6)]
    return hrp + "1" + "".join(_BECH32[d] for d in data + checksum)


@dataclass
class Ev:
    event_id: str
    pubkey: str
    created_at: int
    content: str
    tags: list = field(default_factory=list)
    kind: int = 9
    prompt_tag: str = "@mention"


def rfc3339(ts: int) -> str:
    # chrono DateTime<Utc>::to_rfc3339 -> "2026-10-18T11:50:51+00:00"
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def thread_tags(ev: Ev):
    root = reply = None
    mentions = []
    for tag in ev.tags:
        if tag and tag[0] == "e" and len(tag) >= 4:
            if tag[3] == "root":
                root = tag[1]
            elif tag[3] == "reply":
                reply = tag[1]
        elif tag and tag[0] == "p" and len(tag) >= 2:
            mentions.append(tag[1])
    if root and reply:
        return root, reply, mentions
    if root:
        return root, root, mentions
    if reply:
        return reply, reply, mentions
    return None, None, mentions


def event_block(channel_id, channel_name, ev: Ev, labels) -> str:
    label = labels.get(ev.pubkey)
    sender = (f"{label} (npub: {npub(ev.pubkey)}, hex: {ev.pubkey})" if label
              else f"{npub(ev.pubkey)} (hex: {ev.pubkey})")
    block = (f"Event ID: {ev.event_id}\nChannel: {channel_name} (#{channel_id})\nKind: {ev.kind}\n"
             f"From: {sender}\nTime: {rfc3339(ev.created_at)}\nContent: {ev.content}")
    block += "\nTags: " + json.dumps(ev.tags, separators=(",", ":"), ensure_ascii=False)
    root, parent, mentions = thread_tags(ev)
    parts = []
    if parent:
        parts.append(f"parent={parent}")
    if root:
        parts.append(f"root={root}")
    if mentions:
        parts.append("mentions=[" + ", ".join(
            f"{labels[p]} ({p})" if p in labels else p for p in mentions) + "]")
    if parts:
        block += "\nParsed: " + ", ".join(parts)
    return block


def context_hints(channel_id, channel_name, last: Ev, has_context: bool, anchor) -> str:
    root, parent, _ = thread_tags(last)
    if root:
        hint = ("Thread context included below. Use `buzz messages thread --channel <UUID> --event <ID>` "
                "for full history if truncated." if has_context else
                "Use `buzz messages thread --channel <UUID> --event <ID>` to fetch thread context.")
        s = f"[Context]\nScope: thread\nChannel: {channel_name} (#{channel_id})\nThread root: {root}"
        if parent and parent != root:
            s += f"\nParent: {parent}"
        s += f"\n{hint}"
        if anchor:
            s += (f"\nIMPORTANT: For ordinary replies in this turn, use `--reply-to {anchor}` on `buzz "
                  "messages send` so the conversation stays threaded. If the human explicitly asks for a "
                  "channel-root, top-level, or broadcast post, send that message without `--reply-to`. "
                  "If the requested destination is ambiguous, ask before sending.")
        return s
    s = (f"[Context]\nScope: channel\nChannel: {channel_name} (#{channel_id})\n"
         "Hint: Use `buzz messages get --channel <UUID>` for recent messages if needed.")
    if anchor:
        s += (f"\nIMPORTANT: This is a new top-level message. For ordinary replies in this turn, use "
              f"`--reply-to {anchor}` on `buzz messages send` — the triggering message is the thread root. "
              "Do NOT reply into any other (older) thread. If the human explicitly asks for a channel-root, "
              "top-level, or broadcast post, send that message without `--reply-to`.")
    return s


def thread_context(messages, total, truncated, labels) -> str:
    s = f"[Thread Context ({len(messages)} of {total} messages{', truncated' if truncated else ''})]"
    for i, m in enumerate(messages, 1):
        actor = f"{labels[m.pubkey]} ({m.pubkey})" if m.pubkey in labels else m.pubkey
        s += f"\n[{i}] {actor} ({rfc3339(m.created_at)}): {m.content}"
    return s


STEER = ("[What you were working on]", "[New message — arrived while you were working]",
         "[New messages — arrived while you were working",
         "Note: A new message arrived while you were working. Continue your in-progress work and "
         "incorporate the new message if it's relevant; if it's unrelated, you may briefly acknowledge it "
         "and carry on.")
INTERRUPT = ("[Previous request — interrupted before completion]", "[New request — supersedes previous]",
             "[New request — supersedes previous",
             "Note: The previous request was interrupted. Please address the new request.\nIf the new "
             "request is unrelated to the previous one, you may briefly acknowledge the interruption.")


def format_prompt(channel_id, channel_name, events, labels, *, context=None, cancelled=(),
                  framing=STEER, human_facing=True):
    """The content blocks buzz-acp sends (protocol v2 agent: no standing context in the user message)."""
    last = events[-1]
    root, _, _ = thread_tags(last)
    anchor = (root or last.event_id) if human_facing else None
    blocks = [context_hints(channel_id, channel_name, last, context is not None, anchor)]
    if context is not None:
        blocks.append(thread_context(*context, labels))
    prior, single, multi, closing = framing
    if cancelled:
        s = prior
        for i, ev in enumerate(cancelled, 1):
            s += f"\n\n--- Event {i} ({ev.prompt_tag}) ---\n" + event_block(channel_id, channel_name, ev, labels)
        blocks.append(s)
    if len(events) == 1:
        ev = events[0]
        body = event_block(channel_id, channel_name, ev, labels)
        blocks.append(f"{single}\n\n--- Event 1 ({ev.prompt_tag}) ---\n{body}" if cancelled
                      else f"[Buzz event: {ev.prompt_tag}]\n{body}")
    else:
        s = f"{multi} — {len(events)} events]" if cancelled else f"[Buzz events — {len(events)} events]"
        for i, ev in enumerate(events, 1):
            s += f"\n\n--- Event {i} ({ev.prompt_tag}) ---\n" + event_block(channel_id, channel_name, ev, labels)
        blocks.append(s)
    if cancelled:
        blocks.append(closing)
    return blocks


def prompt_params(session_id, blocks):
    """acp.rs:1970 ``build_prompt_params``: no ``_meta``, no sender, no event id outside the text."""
    return {"sessionId": session_id, "prompt": [{"type": "text", "text": b} for b in blocks]}


def section_blocks(text):
    """``params.prompt`` for a prompt a test wrote as one string out of buzz-acp's sections: split
    before the conversation-context header and before the FIRST event header, where buzz-acp starts
    a new element. A blank separator line stays at the end of the element before it, so the shim's
    ``prompt_text`` joins the elements back into exactly *text*."""
    lines = text.split("\n")
    event = next((i for i, line in enumerate(lines) if line.startswith("[Buzz event:")), None)
    if event is None:
        return [{"type": "text", "text": text}]
    context = next((i for i in range(event)
                    if lines[i].startswith(("[Thread Context (", "[Conversation Context ("))), None)
    cuts = [0] + ([context] if context else []) + [event, len(lines)]
    return [{"type": "text", "text": "\n".join(lines[a:b])} for a, b in zip(cuts, cuts[1:])]
