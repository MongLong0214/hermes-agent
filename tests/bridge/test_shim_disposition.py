"""An inbound that is owed an answer is complete only once a correlated reply is sent and read back.

2026-11-17 live: two owner instructions a CTO relayed, and a repeated review request, reached the CEO's
canonical conversation, the turn ended SILENT and the event was marked complete. Receiving the event, the
transport's end_turn, or a send that exited 0 is none of those things. The gateway names the disposition
it judged for the event (its verified owner-relay provenance; never the message's words); the shim's
journal then keeps the event's reply state until the thread itself shows this bridge's reply to this
event. These drive the shim's real ``handle()`` against the fake ``buzz`` CLI of test_shim_publish_retry.
"""
import hashlib
import importlib.util
import json
import os
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_shim_canary import OTHER
from test_shim_publish_retry import (  # noqa: F401 - the shim fixture is this module's too
    _FAKE, _SHIM, CHANNEL, EVENT_A, EVENT_B, PEER, SIGNER, ask, body, calls, final, log_text, plan, posts, record,
    shim)
# The real coordinator under a tmp home, and the shim wired to it (its fixtures, not copies).
from test_shim_text_cap_notice import _pin_the_module_home  # noqa: F401
from test_shim_text_cap_notice import CHANNEL as GW_CHANNEL
from test_shim_text_cap_notice import EVENT as GW_EVENT
from test_shim_text_cap_notice import OTHER as GW_BRIDGE
from test_shim_text_cap_notice import SIGNER as GW_CTO
from test_shim_text_cap_notice import drive as gw_drive
from test_shim_text_cap_notice import entry as gw_entry
from test_shim_text_cap_notice import gw  # noqa: F401
from test_shim_text_cap_notice import shim as gw_shim  # noqa: F401
from test_shim_text_cap_notice import thread_prompt


class _Gateway(str):
    """The answer text the gateway returned, with what it recorded for the event beside it."""

    def __new__(cls, text, disposition=None, silent=False, resolved=False):
        self = super().__new__(cls, text)
        self.disposition, self.silent, self.resolved = disposition, silent, resolved
        return self


def _gateway_answers(shim, text, disposition=None, silent=False, resolved=False):
    shim.submit = lambda _text, event_id, author=None, channel=None: _Gateway(
        text, disposition, silent, resolved)


@pytest.mark.parametrize("disposition", ["owner_relay", "question", "ruling", "conflict"])
def test_an_owed_inbound_ending_silent_is_never_complete_and_is_told_to_the_thread_once(shim, disposition):
    _gateway_answers(shim, "[SILENT]", disposition, silent=True)
    ask(shim, EVENT_A)
    ask(shim, EVENT_A, rpc_id=2)  # buzz-acp redelivers the event: the gateway answers the same again
    done = final(shim, EVENT_A)
    assert done["reply_state"] == "owed" and done["disposition"] == disposition
    assert (done["kind"], done["preclaim"]) == ("notice", True)  # the slot of its one real reply stays open
    [post] = posts(shim, EVENT_A)  # one visible post, never one per redelivery
    assert "SILENT" not in post["content"] and EVENT_A in post["content"]
    assert not (shim.tmp / "published" / (EVENT_A + ".answer")).exists()


def test_a_no_action_report_that_ends_silent_closes_with_no_reply_and_no_post(shim):
    """CEO 2026-11-17: a real no-action report (not owed) may be answered with nothing at all. It closes: no
    post (the marker is never posted as text), no owed state, and recovery reads it answered."""
    _gateway_answers(shim, "[SILENT]", disposition=None, silent=True)
    ask(shim, EVENT_A)
    done = record(shim, EVENT_A)
    assert (done["kind"], done["disposition"], done["silent"], done.get("reply_state")) == ("answer", None, True, None)
    assert posts(shim, EVENT_A) == [] and [c for c in calls(shim, EVENT_A) if c["cmd"] == "send"] == []
    assert shim.recovery_delivery_state(EVENT_A) == "answered"
    ask(shim, EVENT_A, rpc_id=2)  # redelivered: closed, and still nothing is posted
    assert posts(shim, EVENT_A) == [] and record(shim, EVENT_A) == done


def test_an_owed_answer_completes_once_the_thread_shows_this_reply(shim):
    _gateway_answers(shim, f"answer for {EVENT_A}", "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert [p["content"] for p in posts(shim, EVENT_A)] == [body(EVENT_A)]
    assert [c["cmd"] for c in calls(shim, EVENT_A)] == ["send", "thread"]  # sent, then read back
    assert (done["kind"], done["disposition"], done["reply_state"], done["sent"], done["delivery"],
            done["attempts"]) == ("answer", "owner_relay", "confirmed", True, "posted", 1)


def test_a_duplicate_of_a_completed_event_posts_and_changes_nothing(shim):
    _gateway_answers(shim, f"answer for {EVENT_A}", "question", resolved=True)
    ask(shim, EVENT_A)
    first = final(shim, EVENT_A)
    ask(shim, EVENT_A, rpc_id=2)
    ask(shim, EVENT_A, rpc_id=3)
    assert len(posts(shim, EVENT_A)) == 1 and sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1
    assert record(shim, EVENT_A) == first and first["reply_state"] == "confirmed"
    assert "already published once" in log_text(shim)


def test_a_send_that_fails_leaves_the_event_unprocessed(shim):
    plan(shim, "send", EVENT_A, {"exit": 2, "stderr": "relay closed the connection"})
    _gateway_answers(shim, f"answer for {EVENT_A}", "owner_relay")
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["sent"], done["state"], done["delivery"], done["reply_state"]) == (
        False, "send_failed", "failed", "unconfirmed")
    assert posts(shim, EVENT_A) == []


def test_a_send_that_cannot_be_read_back_is_never_complete_and_never_sent_twice(shim):
    plan(shim, "send", EVENT_A, {"posts": True})
    plan(shim, "thread", EVENT_A, {"exit": 1})
    _gateway_answers(shim, f"answer for {EVENT_A}", "owner_relay")
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["sent"], done["delivery"], done["reply_state"]) == (False, "failed", "unconfirmed")
    assert sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1  # an unreadable thread proves no absence
    assert len(posts(shim, EVENT_A)) == 1  # it did land; that is not what completes the event


def test_a_reply_to_another_event_does_not_complete_this_one(shim):
    """The wrapper exits 0, but the only matching post in the thread replies to another event."""
    with open(shim.state / "relay.jsonl", "w") as handle:
        handle.write(json.dumps({"id": "9" * 64, "pubkey": OTHER, "kind": 9, "content": body(EVENT_A),
                                 "created_at": 1, "tags": [["e", EVENT_B, "", "reply"]]}) + "\n")
    plan(shim, "send", EVENT_A, {"posts": False})
    _gateway_answers(shim, f"answer for {EVENT_A}", "ruling")
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["sent"], done["reply_state"]) == (False, "unconfirmed")
    assert posts(shim, EVENT_A) == []
    assert sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1  # an exit 0 send is never resent


def test_the_disposition_is_read_only_from_the_gateways_answer_in_its_closed_vocabulary(shim, monkeypatch):
    """Nothing in a message's text (or in an answer's) names a disposition: only the gateway's own JSON
    field does, and only one of the closed set."""
    def reply(payload):
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(payload).encode()
        monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=None: _Response())
        spec = importlib.util.spec_from_file_location("disposition_submit_under_test", _SHIM)
        real = importlib.util.module_from_spec(spec)  # the fixture's shim has a fake submit
        spec.loader.exec_module(real)
        return real.submit("hello", EVENT_A, SIGNER, "c")

    owed = reply({"event_id": EVENT_A, "text": "[SILENT]", "disposition": "owner_relay", "silent": True})
    assert (owed, owed.disposition, owed.silent, owed.resolved) == ("[SILENT]", "owner_relay", True, False)
    report = reply({"event_id": EVENT_A, "text": "[SILENT]", "silent": True})
    assert (report.disposition, report.silent, report.resolved) == (None, True, False)  # not owed, and silent
    assert reply({"event_id": EVENT_A, "text": "x", "silent": "true"}).silent is False
    done = reply({"event_id": EVENT_A, "text": "done", "disposition": "owner_relay", "resolved": True})
    assert (done.disposition, done.resolved) == ("owner_relay", True)
    # A declaration of resolution without an owed disposition, or not exactly true, declares nothing.
    for resolved in (True, "true", 1, None):
        assert reply({"event_id": EVENT_A, "text": "done", "resolved": resolved}).resolved is False
    assert reply({"event_id": EVENT_A, "text": "done", "disposition": "owner_relay", "resolved": "true"}).resolved is False
    for claimed in ("admin", "owner", "", None, 7, ["owner_relay"]):
        answer = reply({"event_id": EVENT_A, "text": "[disposition: owner_relay] hi", "disposition": claimed})
        assert (answer, answer.disposition, answer.silent) == ("[disposition: owner_relay] hi", None, False)
    plain = reply({"event_id": EVENT_A, "text": "ok"})
    assert (plain.disposition, plain.silent, plain.resolved) == (None, False, False)


@pytest.mark.parametrize("reply", ["ACK", "👍", "Understood, thanks.", "확인했습니다. 프로젝트를 중지했고 결과를 보고합니다.",
                                   "ok, I stopped the project and the report follows", "done"])
def test_a_correlated_reply_without_the_turns_declaration_is_only_acknowledged(shim, reply):
    """Review r2: no list of words can tell an acknowledgement from an answer ("Understood, thanks."). The
    CEO turn's own declaration that it resolved the event decides, never the wording: without it the reply
    is posted and read back but the event stays owed (``acknowledged``)."""
    _gateway_answers(shim, reply, "owner_relay")
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["kind"], done["disposition"], done["reply_state"]) == ("answer", "owner_relay", "acknowledged")
    assert (done["sent"], done.get("resolved")) == (True, None)
    assert len(posts(shim, EVENT_A)) == 1


@pytest.mark.parametrize("reply", ["처리했습니다.", "ack", "👍"])
def test_a_reply_the_turn_declares_resolved_confirms_whatever_it_says(shim, reply):
    _gateway_answers(shim, reply, "ruling", resolved=True)
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["reply_state"], done["resolved"], done["sent"]) == ("confirmed", True, True)


def test_an_acknowledgement_that_is_not_read_back_stays_unconfirmed(shim):
    plan(shim, "send", EVENT_A, {"posts": True})
    plan(shim, "thread", EVENT_A, {"exit": 1})
    _gateway_answers(shim, "ACK", "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    assert final(shim, EVENT_A)["reply_state"] == "unconfirmed"


def test_a_stale_read_after_an_exit_0_send_never_posts_the_reply_twice(shim, monkeypatch):
    """P2 (review r1): the send exited 0, so a read that does not show the post yet is a stale read, not an
    absence: it is read again, never resent. The reply is one post."""
    reads = []

    def stale_then_fresh(channel, target, digests, _real=shim._read_back):
        reads.append(target)
        return "absent" if len(reads) < 4 else _real(channel, target, digests)

    monkeypatch.setattr(shim, "_read_back", stale_then_fresh)
    _gateway_answers(shim, f"answer for {EVENT_A}", "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert len(posts(shim, EVENT_A)) == 1 and sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1
    assert len(reads) >= 4  # it kept reading
    assert (done["sent"], done["reply_state"], done.get("recovered")) == (True, "confirmed", True)


def test_a_read_that_never_shows_an_exit_0_send_is_left_unconfirmed_not_resent(shim, monkeypatch):
    monkeypatch.setattr(shim, "_read_back", lambda channel, target, digests: "absent")
    _gateway_answers(shim, f"answer for {EVENT_A}", "owner_relay")
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert len(posts(shim, EVENT_A)) == 1 and sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1
    assert (done["sent"], done["reply_state"]) == (False, "unconfirmed")


def test_a_failed_send_is_still_resent_once_a_complete_read_shows_it_absent(shim):
    plan(shim, "send", EVENT_A, {"exit": 2, "stderr": "relay closed the connection"}, {})
    _gateway_answers(shim, f"answer for {EVENT_A}", "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert len(posts(shim, EVENT_A)) == 1 and sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 2
    assert (done["sent"], done["reply_state"]) == (True, "confirmed")


# ---- review r2 -------------------------------------------------------------------------------------------

def _journal(shim, event_id, *, slot="", **fields):
    """A journal entry as a shim left it: the base shim's own shape (no disposition), plus *fields*, in the
    event's journal slot (``slot`` ``.answer``: the slot a refused event's one terminal answer takes)."""
    entry = {"journal": 2, "sent": True, "state": "sent", "delivery": "posted", "kind": "answer",
             "event_id": event_id, "ref": None, "channel": CHANNEL, "roster_generation": None,
             "reply_to": event_id, "reason": "exit 0:", "attempts": 1, "at": "2026-11-16T00:00:00+0000"}
    entry.update(fields)
    os.makedirs(shim.PUBLISHED, exist_ok=True)
    with open(os.path.join(shim.PUBLISHED, event_id + slot), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def _body_sha(text):
    return hashlib.sha256(f"@{PEER} {text}".encode()).hexdigest()


@pytest.mark.parametrize("answer, resolved, expected", [
    ("done", True, "answered"),             # the thread shows a reply the CEO turn declared resolved
    ("done", False, "owed"),                # acknowledged: posted, still owed
    ("[SILENT]", False, "owed"),            # told once, still owed
])
def test_recovery_reads_an_owed_event_as_answered_only_once_it_is_confirmed(shim, answer, resolved, expected):
    _gateway_answers(shim, answer, "owner_relay", silent=answer == "[SILENT]", resolved=resolved)
    ask(shim, EVENT_A)
    final(shim, EVENT_A)
    assert shim.recovery_delivery_state(EVENT_A) == expected
    # What batch recovery and the explicit recover command skip: only an answered event.
    assert shim._batch.already_delivered(shim, EVENT_A) is (expected == "answered")


def test_recovery_keeps_an_unconfirmed_owed_event_recoverable(shim):
    plan(shim, "send", EVENT_A, {"posts": True})
    plan(shim, "thread", EVENT_A, {"exit": 1})
    _gateway_answers(shim, "done", "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    assert final(shim, EVENT_A)["reply_state"] == "unconfirmed"
    assert shim.recovery_delivery_state(EVENT_A) == "owed"


def test_recovery_still_reads_an_event_that_is_not_owed_as_answered(shim):
    _gateway_answers(shim, "hello", None)
    ask(shim, EVENT_B)
    final(shim, EVENT_B)
    assert record(shim, EVENT_B)["disposition"] is None  # judged by the gateway, and not owed
    assert shim.recovery_delivery_state(EVENT_B) == "answered"


def test_an_owed_event_whose_reply_was_only_acknowledged_takes_a_later_substantive_answer(shim):
    """Review r2: the publication claim does not lock an owed event's slot until the reply is confirmed."""
    _gateway_answers(shim, "ACK", "owner_relay")
    ask(shim, EVENT_A)
    assert final(shim, EVENT_A)["reply_state"] == "acknowledged"
    ask(shim, EVENT_A, rpc_id=2)  # the same text redelivered: nothing is posted again
    assert [p["content"] for p in posts(shim, EVENT_A)] == [f"@{PEER} ACK"]
    _gateway_answers(shim, "프로젝트를 중지했고 결과는 다음과 같습니다.", "owner_relay", resolved=True)
    ask(shim, EVENT_A, rpc_id=3)
    done = final(shim, EVENT_A)
    assert (done["reply_state"], done["resolved"], done["sent"]) == ("confirmed", True, True)
    assert [p["content"] for p in posts(shim, EVENT_A)] == [
        f"@{PEER} ACK", f"@{PEER} 프로젝트를 중지했고 결과는 다음과 같습니다."]
    # Confirmed: now locked, whatever is redelivered.
    _gateway_answers(shim, "another answer", "owner_relay", resolved=True)
    ask(shim, EVENT_A, rpc_id=4)
    assert len(posts(shim, EVENT_A)) == 2 and shim.recovery_delivery_state(EVENT_A) == "answered"
    assert "already published once" in log_text(shim)


def test_publish_reply_itself_reopens_an_unconfirmed_owed_slot_but_not_a_pending_or_confirmed_one(shim):
    plan(shim, "send", EVENT_A, {"exit": 2, "stderr": "relay closed the connection"}, {})
    shim.PUBLISH_RETRY_BACKOFF = ()  # the first attempt's failure is the final one
    first = f"answer for {EVENT_A}"
    assert shim.publish_reply(EVENT_A, CHANNEL, first, owed="owner_relay", resolved=True) is False
    done = final(shim, EVENT_A)
    assert (done["state"], done["reply_state"]) == ("send_failed", "unconfirmed") and posts(shim, EVENT_A) == []
    # The same text is protected, whatever state its journal is in.
    assert shim.publish_reply(EVENT_A, CHANNEL, first, owed="owner_relay", resolved=True) is False
    assert sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1
    # A later answer posts, and moves the event to confirmed.
    assert shim.publish_reply(EVENT_A, CHANNEL, "second answer", owed="owner_relay", resolved=True) is True
    done = record(shim, EVENT_A)
    assert (done["sent"], done["reply_state"], done["resolved"]) == (True, "confirmed", True)
    assert [p["content"] for p in posts(shim, EVENT_A)] == [f"@{PEER} second answer"]
    # Confirmed is final.
    assert shim.publish_reply(EVENT_A, CHANNEL, "third answer", owed="owner_relay", resolved=True) is False
    assert len(posts(shim, EVENT_A)) == 1
    # A retry still pending on the claim is never raced.
    _journal(shim, EVENT_B, sent=False, state="timeout", delivery="failed", pending=True,
             disposition="owner_relay", reply_state="unconfirmed", body_sha256=_body_sha("first"))
    assert shim.publish_reply(EVENT_B, CHANNEL, "second", owed="owner_relay", resolved=True) is False
    assert [c for c in calls(shim, EVENT_B) if c["cmd"] == "send"] == []


def test_a_base_journal_that_posted_silence_for_an_event_now_owed_is_reopened_and_told_once(shim):
    """Review r2: the base shim journaled ``[SILENT]`` plus an exit-0 send as the event's answer. The gateway
    now judges the event owed (live, or derived on replay): the answer marker is replaced by ``owed``."""
    _journal(shim, EVENT_A, body_sha256=_body_sha("[SILENT]"))
    _gateway_answers(shim, "[SILENT]", "owner_relay", silent=True)
    ask(shim, EVENT_A)
    done = record(shim, EVENT_A)
    assert (done["kind"], done["preclaim"], done["disposition"], done["reply_state"]) == (
        "notice", True, "owner_relay", "owed")
    assert shim.recovery_delivery_state(EVENT_A) == "owed"
    [post] = posts(shim, EVENT_A)
    assert "SILENT" not in post["content"] and EVENT_A in post["content"]
    ask(shim, EVENT_A, rpc_id=2)  # redelivered: the notice is told once
    assert len(posts(shim, EVENT_A)) == 1 and record(shim, EVENT_A)["reply_state"] == "owed"


def test_a_base_journal_of_a_real_answer_is_not_reopened_by_the_silence_path(shim):
    _journal(shim, EVENT_A, body_sha256=_body_sha("a real answer"))
    _gateway_answers(shim, "[SILENT]", "owner_relay", silent=True)
    ask(shim, EVENT_A)
    assert record(shim, EVENT_A)["kind"] == "answer" and posts(shim, EVENT_A) == []


# ---- review r3 -------------------------------------------------------------------------------------------

def _original(event_id, content="@ceo please do it"):
    """The signed original recovery fetches from the relay by id (``canonical_batch.fetch_event``)."""
    return {"id": event_id, "pubkey": SIGNER, "content": content, "tags": [["h", CHANNEL], ["p", OTHER]]}


def _recover(shim, event_id):
    """What batch recovery and ``recover`` do for one event: skip it when answered, else fetch its original
    and deliver it through the same ``deliver_event``. Returns the ids it fetched."""
    fetched = []

    def lookup(wanted):
        fetched.append(wanted)
        return _original(wanted)

    shim._batch.deliver_member(SimpleNamespace(**vars(shim)), "s", [], CHANNEL, event_id, lookup)
    return fetched


def test_recovery_fetches_the_original_of_an_answer_with_no_disposition_and_reopens_its_silence(shim):
    """Review r3 (P1): a sent answer the base shim journaled has no disposition, so nothing says the event was
    not owed. Recovery skipped it as answered before fetching the original, and the base's ``[SILENT]`` stayed
    the event's answer. It reads as ``unverified``, and its original is fetched and judged like any event."""
    _journal(shim, EVENT_A, body_sha256=_body_sha("[SILENT]"))
    assert shim.recovery_delivery_state(EVENT_A) == "unverified"  # never answered
    assert shim._batch.already_delivered(SimpleNamespace(**vars(shim)), EVENT_A) is False
    _gateway_answers(shim, "[SILENT]", "owner_relay", silent=True)
    assert _recover(shim, EVENT_A) == [EVENT_A]
    done = record(shim, EVENT_A)
    assert (done["kind"], done["disposition"], done["reply_state"]) == ("notice", "owner_relay", "owed")
    assert shim.recovery_delivery_state(EVENT_A) == "owed"
    [post] = posts(shim, EVENT_A)
    assert "SILENT" not in post["content"] and EVENT_A in post["content"]


def test_a_base_silence_in_the_answer_slot_is_reopened_by_redelivery(shim):
    """Review r3 (P1): a refused event's one terminal answer lives in ``<event>.answer``. A base ``[SILENT]``
    there survived redelivery, because only the first slot was looked at."""
    _journal(shim, EVENT_A, kind="notice", preclaim=True, body_sha256=_body_sha("refusal notice"))
    _journal(shim, EVENT_A, slot=".answer", body_sha256=_body_sha("[SILENT]"))
    assert shim.recovery_delivery_state(EVENT_A) == "unverified"
    _gateway_answers(shim, "[SILENT]", "owner_relay", silent=True)
    ask(shim, EVENT_A)
    done = record(shim, EVENT_A + ".answer")
    assert (done["kind"], done["disposition"], done["reply_state"]) == ("notice", "owner_relay", "owed")
    assert record(shim, EVENT_A)["kind"] == "notice" and "disposition" not in record(shim, EVENT_A)
    assert shim.recovery_delivery_state(EVENT_A) == "owed"
    [post] = posts(shim, EVENT_A)
    assert "SILENT" not in post["content"]
    ask(shim, EVENT_A, rpc_id=2)  # redelivered: the notice is told once
    assert len(posts(shim, EVENT_A)) == 1


def test_recovery_settles_a_legacy_answer_once_the_gateway_says_it_was_not_owed(shim):
    _journal(shim, EVENT_A, body_sha256=_body_sha("hello"))
    assert shim.recovery_delivery_state(EVENT_A) == "unverified"
    _gateway_answers(shim, "hello", None)
    assert _recover(shim, EVENT_A) == [EVENT_A]
    assert record(shim, EVENT_A)["disposition"] is None and posts(shim, EVENT_A) == []  # nothing is posted
    assert shim.recovery_delivery_state(EVENT_A) == "answered"
    assert _recover(shim, EVENT_A) == []  # answered now: skipped, its original is not fetched again


def test_a_legacy_answer_with_no_original_to_fetch_is_not_given_a_recovery_notice(shim):
    _journal(shim, EVENT_A, body_sha256=_body_sha("hello"))
    shim.refuse_recovery_event("s", [{"type": "text", "text": "x"}], EVENT_A, "original unavailable")
    assert posts(shim, EVENT_A) == [] and "no new notice" in log_text(shim)


def test_a_legacy_answer_of_an_owed_event_is_rechecked_against_the_thread_not_trusted(shim):
    """The base shim journaled an exit 0 send with no read-back as the answer. Judged owed now, it is read
    back, never reposted: found in the thread, it is confirmed; not found, it is unconfirmed and still owed."""
    text = "중지했고 결과를 보고합니다."
    with open(shim.state / "relay.jsonl", "w") as handle:
        handle.write(json.dumps({"id": "8" * 64, "pubkey": OTHER, "kind": 9, "content": f"@{PEER} {text}",
                                 "created_at": 1, "tags": [["e", EVENT_A, "", "reply"]]}) + "\n")
    _journal(shim, EVENT_A, body_sha256=_body_sha(text))
    _journal(shim, EVENT_B, reply_to=EVENT_B, body_sha256=_body_sha(text))  # the same text, never posted
    _gateway_answers(shim, text, "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    ask(shim, EVENT_B)
    done = record(shim, EVENT_A)
    assert (done["disposition"], done["reply_state"], done["sent"], done["resolved"]) == (
        "owner_relay", "confirmed", True, True)
    assert record(shim, EVENT_B)["reply_state"] == "unconfirmed" and record(shim, EVENT_B)["sent"] is False
    assert [c for c in calls(shim) if c["cmd"] == "send"] == []  # nothing was posted again


def test_redelivering_the_same_body_after_the_readbacks_expired_rechecks_the_thread_and_confirms(shim):
    """Review r3 (P2): the post landed, every read-back failed, the retries ran out. The same body redelivered
    was rejected before any read, so the post that exists stayed unconfirmed."""
    plan(shim, "send", EVENT_A, {"posts": True})
    plan(shim, "thread", EVENT_A, {"exit": 1})
    _gateway_answers(shim, "done", "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    first = final(shim, EVENT_A)
    assert (first["reply_state"], first["sent"]) == ("unconfirmed", False) and len(posts(shim, EVENT_A)) == 1
    ask(shim, EVENT_A, rpc_id=2)  # the thread is still unreadable: nothing changes, nothing is posted
    again = record(shim, EVENT_A)
    assert (again["reply_state"], again["sent"]) == ("unconfirmed", False)
    plan(shim, "thread", EVENT_A, {})  # readable now
    ask(shim, EVENT_A, rpc_id=3)
    done = record(shim, EVENT_A)
    assert (done["reply_state"], done["sent"], done["delivery"], done["resolved"]) == (
        "confirmed", True, "posted", True)
    assert len(posts(shim, EVENT_A)) == 1 and sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1
    assert shim.recovery_delivery_state(EVENT_A) == "answered"


def test_a_same_body_recheck_that_finds_nothing_posts_nothing_and_stays_unconfirmed(shim, monkeypatch):
    plan(shim, "send", EVENT_A, {"posts": False})  # exit 0, and it never lands
    _gateway_answers(shim, "done", "owner_relay", resolved=True)
    ask(shim, EVENT_A)
    assert final(shim, EVENT_A)["reply_state"] == "unconfirmed"
    sends = sum(c["cmd"] == "send" for c in calls(shim, EVENT_A))
    ask(shim, EVENT_A, rpc_id=2)
    assert record(shim, EVENT_A)["reply_state"] == "unconfirmed"
    assert sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == sends and posts(shim, EVENT_A) == []


@pytest.mark.parametrize("answer", ["", "   \n"])
def test_an_owed_event_with_no_answer_text_is_not_posted_and_stays_owed(shim, answer):
    """An owed answer with no text is no answer: it posts nothing of its own, and the thread is told the event
    is still owed."""
    _gateway_answers(shim, answer, "owner_relay")
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["kind"], done["reply_state"]) == ("notice", "owed")
    [post] = posts(shim, EVENT_A)
    assert EVENT_A in post["content"]
    assert shim.recovery_delivery_state(EVENT_A) == "owed"


def _rows(text, kwargs, history, final_text, calls):
    """The rows a turn adds: its own opening row (with the provenance the gateway stamped), the tool calls it
    made (each answered with its result) and its answer."""
    user = {"role": "user", "content": text}
    for key, name in (("display_kind", "persist_user_display_kind"),
                      ("display_metadata", "persist_user_display_metadata")):
        if kwargs.get(name):
            user[key] = kwargs[name]
    rows = [user]
    for name, args, out, call_id in calls:
        rows.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        rows.append({"role": "tool", "tool_call_id": call_id, "name": name, "content": out})
    return [*history, *rows, {"role": "assistant", "content": final_text}]


def _resolve(outcome="done", evidence=None, call_id="resolve-1"):
    args = {"event_id": GW_EVENT, "outcome": outcome,
            "evidence": [{"tool_call_id": "call-1"}] if evidence is None else evidence}
    return ("relay_resolve", args, json.dumps({"declared": True, "event_id": GW_EVENT, "outcome": outcome}), call_id)


def _work(call_id="call-1"):
    return ("run_cancel", {"run_id": "r1"}, '{"ok": true}', call_id)


def _real_path(gw, gw_shim, monkeypatch, tmp_path, *turns, relay=True, kind=None):
    """The shim wired to the real coordinator. Turn n of the cached actor makes the tool calls of
    ``turns[n][1:]`` and ends with ``turns[n][0]`` (the last repeats). *relay*: the event is a relayed owner
    instruction its CTO is entitled to relay; else it is the CTO's own post, tagged *kind* when given."""
    import time

    from tests.gateway.test_canonical_owner_relay import _entitle, _owner_record, _relay, _transcript

    if relay:
        _entitle(actor=GW_CTO)
        _transcript(_owner_record())

    def run_conversation(text, *, conversation_history, task_id, **kwargs):
        gw.runs.append(text)
        final_text, *calls = turns[min(len(gw.runs), len(turns)) - 1]
        gw.actor._persist_user_message_idx = len(conversation_history)
        return {"completed": True, "session_id": task_id, "final_response": final_text,
                "messages": _rows(text, kwargs, conversation_history, final_text, calls)}

    monkeypatch.setattr(gw.actor, "run_conversation", run_conversation)
    # A Buzz CLI that keeps its own thread, so the read-back sees exactly what landed.
    state = tmp_path / "fake"
    state.mkdir()
    cli = Path(gw_shim.BUZZ_CLI)
    cli.write_text(f"#!{sys.executable}\n" + _FAKE.format(state=str(state), signer=GW_BRIDGE, channel=GW_CHANNEL))
    cli.chmod(0o755)
    monkeypatch.setattr(gw_shim, "PUBLISH_RETRY_BACKOFF", (0.05,) * 5)

    def thread():
        path = state / "relay.jsonl"
        return [json.loads(line)["content"] for line in path.read_text().splitlines()] if path.exists() else []

    def journal():
        return json.loads((Path(gw_shim.PUBLISHED) / GW_EVENT).read_text().splitlines()[0])

    def settled():
        end = time.monotonic() + 20
        while journal().get("pending") is True and time.monotonic() < end:
            time.sleep(0.02)
        return journal()

    prompt = thread_prompt([gw_entry(1, 20)], _relay() if relay else "@ceo 확인 부탁드립니다")
    if kind is not None:
        head, _, tags = prompt.rpartition("\nTags: ")
        prompt = head + "\nTags: " + json.dumps(json.loads(tags) + [["acp-kind", kind]], separators=(",", ":"))
    return SimpleNamespace(prompt=prompt, thread=thread, settled=settled, journal=journal)


def test_an_acknowledged_owner_relay_gets_a_reply_only_recovery_turn_and_its_declared_answer_confirms(
        gw, gw_shim, monkeypatch, tmp_path):
    """The real coordinator, its HTTP handler and SQLite, and the shim. The first turn does the work and only
    acknowledges. The event's redelivery runs a reply-only turn (the work is not run again), whose declared
    answer is posted, read back and confirmed."""
    real = _real_path(gw, gw_shim, monkeypatch, tmp_path, ("ACK", _work()),
                      ("중지했고 결과를 보고합니다.", _resolve("done", [{"tool_call_id": "call-1"}])))
    gw_drive(gw_shim, real.prompt)
    assert len(gw.runs) == 1 and real.thread() == ["@cto-repo-factory ACK"]
    assert real.settled()["reply_state"] == "acknowledged"
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "owed"

    gw_drive(gw_shim, real.prompt, rpc_id=2)  # buzz-acp redelivers the event
    assert len(gw.runs) == 2  # the gateway ran a recovery turn instead of replaying the ACK
    assert real.thread() == ["@cto-repo-factory ACK", "@cto-repo-factory 중지했고 결과를 보고합니다."]
    done = real.settled()
    assert (done["reply_state"], done["resolved"], done["sent"], done["disposition"]) == (
        "confirmed", True, True, "owner_relay")
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "answered"

    gw_drive(gw_shim, real.prompt, rpc_id=3)  # resolved: a terminal, replayed, and nothing is posted again
    assert len(gw.runs) == 2 and len(real.thread()) == 2


def test_a_bare_resolve_on_an_acknowledged_request_stays_owed_on_the_real_path(gw, gw_shim, monkeypatch, tmp_path):
    """CEO counterexample (1): an explicit request answered "ACK" with a bare resolve (empty evidence, then
    evidence of nothing the event's turns did) is posted and read back, and is still owed."""
    real = _real_path(gw, gw_shim, monkeypatch, tmp_path, ("ACK", _resolve("done", [])),
                      ("ACK", _resolve("done", [{"tool_call_id": "call-from-another-event"}])))
    gw_drive(gw_shim, real.prompt)
    assert real.settled()["reply_state"] == "acknowledged"
    gw_drive(gw_shim, real.prompt, rpc_id=2)
    assert real.thread() == ["@cto-repo-factory ACK"]  # the same text is not posted twice
    assert real.settled()["reply_state"] == "acknowledged"
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "owed"


def test_a_no_action_report_closes_with_no_reply_and_no_channel_post_on_the_real_path(
        gw, gw_shim, monkeypatch, tmp_path):
    """CEO counterexample (2): a CTO report that asks nothing (``acp-kind: report``) answered with nothing
    posts nothing, and is closed."""
    real = _real_path(gw, gw_shim, monkeypatch, tmp_path, ("[SILENT]",), relay=False, kind="report")
    gw_drive(gw_shim, real.prompt)
    assert len(gw.runs) == 1 and real.thread() == []
    done = real.journal()
    assert (done["kind"], done["disposition"], done["silent"]) == ("answer", None, True)
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "answered"
    gw_drive(gw_shim, real.prompt, rpc_id=2)
    assert len(gw.runs) == 1 and real.thread() == []


@pytest.mark.parametrize("kind", ["question", "ruling_request", "conflict"])
def test_a_declared_question_ruling_or_conflict_runs_judgement_reply_readback_and_close(
        gw, gw_shim, monkeypatch, tmp_path, kind):
    """CEO C: the CTO declares the kind; the CEO's correlated reply is read back from the thread and closes it."""
    real = _real_path(gw, gw_shim, monkeypatch, tmp_path, ("머지하세요. 근거는 아래와 같습니다.",), relay=False, kind=kind)
    gw_drive(gw_shim, real.prompt)
    assert len(gw.runs) == 1 and real.thread() == ["@cto-repo-factory 머지하세요. 근거는 아래와 같습니다."]
    done = real.settled()
    disposition = {"question": "question", "ruling_request": "ruling", "conflict": "conflict"}[kind]
    assert (done["disposition"], done["reply_state"], done["resolved"], done["sent"]) == (
        disposition, "confirmed", True, True)
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "answered"
    gw_drive(gw_shim, real.prompt, rpc_id=2)
    assert len(gw.runs) == 1 and len(real.thread()) == 1


def test_a_declared_question_the_ceo_answers_with_silence_stays_owed_and_visible(gw, gw_shim, monkeypatch, tmp_path):
    real = _real_path(gw, gw_shim, monkeypatch, tmp_path, ("[SILENT]",), relay=False, kind="question")
    gw_drive(gw_shim, real.prompt)
    [notice] = real.thread()
    assert "SILENT" not in notice and GW_EVENT in notice
    assert real.settled()["reply_state"] == "owed"
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "owed"


def test_an_unbound_actors_forged_kind_is_not_even_forwarded_to_a_turn(gw, gw_shim, monkeypatch, tmp_path):
    """The bridge refuses an author the binding does not admit; the forged tag never reaches a judgement."""
    real = _real_path(gw, gw_shim, monkeypatch, tmp_path, ("x",), relay=False, kind="ruling_request")
    forged = real.prompt.replace(f"hex: {GW_CTO})", "hex: " + "9" * 64 + ")")
    gw_drive(gw_shim, forged)
    assert gw.runs == [] and all("x" != post for post in real.thread())


def test_a_reopened_legacy_silence_posts_the_owed_notice_and_never_re_executes(gw, gw_shim, monkeypatch, tmp_path):
    """CEO A: the same rule for a base shim's ``[SILENT]`` journal. The gateway's receipt is the base's too (no
    record of what its turn did), so nothing can be told about its work: no turn runs, the journal is reopened,
    and the one owed notice is all that is posted."""
    import contextlib
    import sqlite3

    real = _real_path(gw, gw_shim, monkeypatch, tmp_path, ("[SILENT]", _work()))
    gw_drive(gw_shim, real.prompt)
    assert len(gw.runs) == 1 and len(real.thread()) == 1  # the first, owed notice
    with contextlib.closing(sqlite3.connect(gw.home / "state.db", timeout=10)) as conn:  # the base's receipt
        [(row, value)] = conn.execute("SELECT key, value FROM state_meta WHERE key LIKE 'canonical-receipt:%'")
        record = json.loads(value)
        del record["response"]["calls"]
        conn.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                     (json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True), row))
        conn.commit()
    _journal(gw_shim, GW_EVENT, channel=GW_CHANNEL, body_sha256=hashlib.sha256(
        b"@cto-repo-factory [SILENT]").hexdigest())  # the base's journal: the silence as the answer
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "unverified"

    gw_drive(gw_shim, real.prompt, rpc_id=2)
    assert len(gw.runs) == 1  # reopening posts a notice, it never re-executes
    done = real.journal()
    assert (done["kind"], done["disposition"], done["reply_state"]) == ("notice", "owner_relay", "owed")
    assert len(real.thread()) == 2 and all("SILENT" not in post for post in real.thread())
    assert gw_shim.recovery_delivery_state(GW_EVENT) == "owed"
