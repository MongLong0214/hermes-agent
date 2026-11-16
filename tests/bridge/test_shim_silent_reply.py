"""The CEO can answer a peer message with exactly ``[SILENT]``: the answer is recorded, never posted.

Live 2026-11-16 20:56-20:59 KST: after the owner asked the CEO to stop the replies that wake the CTO, the
CEO answered two CTO progress reports with nothing at all (reasoning-only, then empty model output). This
path had no way to say "no reply": one event ended as ``409 canonical_event_failed`` with a not-delivered
notice posted to its thread, the other as the turn's "No reply" explanation posted as the answer. The
shim now offers, in buzz-acp's ``[Context]``, exactly ``[SILENT]`` as the answer that means no reply, and
an answer that is exactly that marker is journaled as silent and not posted. Empty or reasoning-only
output stays the gateway's error path; nothing here reads it as silence.

The light cases drive the real shim's ``handle()`` with ``submit`` stubbed; the last one runs it against
the real canonical handler, coordinator and SQLite of an in-process gateway (``test_shim_text_cap_notice``).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

import test_shim_text_cap_notice as cap
from buzz_prompt_port import Ev, format_prompt, prompt_params

_SHIM = Path(os.environ.get("SHIM_PATH") or
             Path(__file__).resolve().parents[2] / "bridge" / "canonical" / "acp-canonical-shim.py")
SIGNER, OTHER, CHANNEL = cap.SIGNER, cap.OTHER, cap.CHANNEL  # the CTO, this bridge, the room
EVENT = "c" * 64
ROOT = "a" * 64
LABELS = {SIGNER: "cto-repo-factory", OTHER: "ceo"}
SILENT = "[SILENT]"

# The in-process gateway and its module home pin, as the other bridge modules reuse them.
gw = cap.gw
_pin_the_module_home = cap._pin_the_module_home


def report(content="@ceo [repo-factory CTO] progress: partial GREEN, next report is the committed head",
           *, event_id=EVENT):
    """buzz-acp's prompt for a new top-level message in a channel (the shape of both live events)."""
    ev = Ev(event_id=event_id, pubkey=SIGNER, created_at=1_789_000_000, content=content,
            tags=[["h", CHANNEL], ["p", OTHER]])
    return format_prompt(CHANNEL, "repo-factory", [ev], LABELS)


def thread_report(*, event_id=EVENT):
    """buzz-acp's prompt for a reply inside a thread, with its quoted history."""
    history = [Ev(event_id=ROOT, pubkey=OTHER, created_at=1_788_999_000, content="earlier decision",
                  tags=[["h", CHANNEL]])]
    ev = Ev(event_id=event_id, pubkey=SIGNER, created_at=1_789_000_000,
            content="@ceo [repo-factory CTO] progress only", tags=[["h", CHANNEL], ["e", ROOT, "", "root"],
                                                                   ["p", OTHER]])
    return format_prompt(CHANNEL, "repo-factory", [ev], LABELS, context=(history, 1, False))


def drive(shim, blocks, rpc_id=1):
    shim.handle({"jsonrpc": "2.0", "id": rpc_id, "method": "session/prompt",
                 "params": prompt_params("s", blocks)})


def joined(blocks):
    return "\n".join(blocks)


def _fake_buzz(root):
    calls = root / "buzz-calls.jsonl"
    (root / "buzz").write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}) + '\\n')\n")
    (root / "buzz").chmod(0o755)
    (root / "required-mentions.json").write_text(json.dumps({"channels": {CHANNEL: {
        "ceo": {"handle": "ceo", "pubkey": OTHER}, "peer": {"handle": "cto-repo-factory", "pubkey": SIGNER}}}}))
    return lambda: [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []


def _journal(root, event_id=EVENT):
    path = root / "published" / event_id
    return json.loads(path.read_text().splitlines()[0]) if path.exists() else None


@pytest.fixture
def shim(tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    posts = _fake_buzz(root)
    for key, value in {"CANONICAL_BUZZ_CLI": str(root / "buzz"),
                       "CANONICAL_MENTIONS_FILE": str(root / "required-mentions.json"),
                       "CANONICAL_PUBLISHED_DIR": str(root / "published"),
                       "CANONICAL_SHIM_LOG": str(root / "shim.log"),
                       "CANONICAL_QUARANTINE_LOG": str(root / "q.jsonl"),
                       "CANONICAL_CANARY_FILE": str(root / "canary.json"),
                       "CANONICAL_BUSY_RETRY_SECS": "0"}.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location(f"silent_reply_shim_{id(tmp_path)}", _SHIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FORWARD", "live")
    monkeypatch.setattr(module, "write", lambda payload: None)  # buzz-acp's stdout, never a post
    module.answer, module.submitted = "answer", []

    def submit(text, event_id, author=None, channel=None):
        module.submitted.append(text)
        return module.answer

    module.submit = submit
    module.posts = posts
    module.journal = lambda event_id=EVENT: _journal(root, event_id)
    module.log_text = lambda: (root / "shim.log").read_text() if (root / "shim.log").exists() else ""
    return module


def _offer_lines(text):
    return [line for line in text.split("\n") if SILENT in line]


# ── The offer: buzz-acp's [Context] tells the CEO how to answer with no reply ────────────────────────


def test_a_channel_report_offers_the_exact_silent_answer_above_the_event(shim):
    blocks = report()
    drive(shim, blocks)
    [sent] = shim.submitted
    [offer] = _offer_lines(sent)
    assert f"exactly {SILENT}" in offer
    lines = sent.split("\n")
    # The last line of buzz-acp's [Context] element, right above the event section.
    assert lines[lines.index(offer) + 1] == "[Buzz event: @mention]"
    # Everything else arrives byte for byte: the line is the only addition.
    assert sent.replace(offer + "\n", "", 1) == joined(blocks)


def test_a_thread_report_gets_the_offer_above_its_quoted_history(shim):
    blocks = thread_report()
    drive(shim, blocks)
    [sent] = shim.submitted
    [offer] = _offer_lines(sent)
    lines = sent.split("\n")
    assert lines[lines.index(offer) + 1].startswith("[Thread Context (")
    assert sent.replace(offer + "\n", "", 1) == joined(blocks)


def test_a_recovered_event_is_sent_without_the_offer(shim):
    """A recovered batch member's first lines and its content are what the gateway's owner-relay check
    reads as signed (gateway/owner_relay.py ``event_content``): nothing is added to that text."""
    text = (f"[Buzz event: recovered]\nEvent ID: {EVENT}\nChannel ID: {CHANNEL}\nAuthor ID: {SIGNER}\n"
            f"Content: @ceo progress only\nTags: [[\"h\",\"{CHANNEL}\"]]")
    shim.deliver_event("s", (SIGNER, CHANNEL, EVENT, "@ceo progress only"), text, True)
    assert shim.submitted == [text]


def test_the_offer_never_pushes_a_text_over_the_gateway_limit(shim):
    """A message that fit the gateway before still goes as it is: no refusal, no trimming for the line."""
    base = joined(report(""))
    blocks = report("x" * (shim.GATEWAY_TEXT_LIMIT - len(base) - 10))
    assert len(joined(blocks)) == shim.GATEWAY_TEXT_LIMIT - 10
    drive(shim, blocks)
    assert shim.submitted == [joined(blocks)]
    [post] = shim.posts()
    assert post["stdin"] == "@cto-repo-factory answer"


# ── The answer: exactly [SILENT] is recorded and not posted ──────────────────────────────────────────


@pytest.mark.parametrize("answer", [SILENT, f"{SILENT}\n", f"  {SILENT}  "])
def test_a_silent_answer_is_journaled_and_never_posted(shim, answer):
    shim.answer = answer
    drive(shim, report())
    assert shim.posts() == []
    record = shim.journal()
    assert (record["kind"], record["sent"], record["state"], record["delivery"]) == (
        "answer", False, "silent", "not_posted")
    assert (record["event_id"], record["channel"]) == (EVENT, CHANNEL)
    assert "answer SILENT" in shim.log_text()
    # Recovery reads the event as answered: never resubmitted, never told "outcome unknown".
    assert shim.recovery_delivery_state(EVENT) == "answered"


def test_a_redelivered_silent_event_still_posts_nothing(shim):
    shim.answer = SILENT
    drive(shim, report())
    drive(shim, report(), rpc_id=2)
    assert shim.posts() == []
    assert shim.journal()["state"] == "silent"


@pytest.mark.parametrize("answer", [f"{SILENT} nothing new", "NO_REPLY", "SILENT", "[silent]",
                                    f"ack\n{SILENT}"])
def test_only_the_exact_marker_is_silent(shim, answer):
    shim.answer = answer
    drive(shim, report())
    [post] = shim.posts()
    assert post["stdin"] == f"@cto-repo-factory {answer}"
    assert shim.journal()["state"] == "sent"


def test_an_ordinary_answer_is_still_posted_once(shim):
    drive(shim, report())
    [post] = shim.posts()
    assert post["stdin"] == "@cto-repo-factory answer"
    assert shim.journal()["sent"] is True


# ── End to end: the real canonical handler records [SILENT] in the receipt; nothing is posted ─────────


@pytest.fixture
def gateway_shim(gw, tmp_path, monkeypatch):
    root = tmp_path / "bridge"
    root.mkdir()
    posts = _fake_buzz(root)
    module = cap._load_shim(monkeypatch, cap._shim_env(root, gw.key))
    module.posts = posts
    module.journal = lambda event_id=EVENT: _journal(root, event_id)
    return module


def _answer_with(actor, answer):
    run = type(actor).run_conversation

    def answering(text, *, conversation_history, task_id, **kwargs):
        result = run(actor, text, conversation_history=conversation_history, task_id=task_id, **kwargs)
        result["final_response"] = answer
        result["messages"][-1] = {"role": "assistant", "content": answer}
        return result

    actor.run_conversation = answering


def test_a_silent_answer_through_the_gateway_is_in_the_receipt_and_not_posted(gw, gateway_shim):
    _answer_with(gw.actor, SILENT)
    drive(gateway_shim, report())

    [sent] = gw.sent()
    assert len(_offer_lines(sent["text"])) == 1
    assert len(gw.runs) == 1
    [receipt] = [json.loads(value) for value in gw.rows("canonical-receipt:").values()]
    assert receipt["state"] == "terminal" and receipt["response"]["terminal_text"] == SILENT
    assert gateway_shim.posts() == []
    assert gateway_shim.journal()["state"] == "silent"
