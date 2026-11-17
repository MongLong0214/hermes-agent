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
import urllib.request

import pytest
from test_shim_publish_retry import (  # noqa: F401 - the shim fixture is this module's too
    _SHIM, CHANNEL, EVENT_A, EVENT_B, PEER, SIGNER, ask, body, calls, final, log_text, plan, posts, record, shim)


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


def test_a_no_action_report_may_still_go_silent(shim):
    _gateway_answers(shim, "[SILENT]", disposition=None, silent=False)
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["kind"], done.get("disposition"), done.get("reply_state")) == ("answer", None, None)
    assert not any("owed" in json.dumps(p) for p in posts(shim, EVENT_A))


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
        handle.write(json.dumps({"id": "9" * 64, "pubkey": SIGNER, "kind": 9, "content": body(EVENT_A),
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

def _journal(shim, event_id, **fields):
    """A journal entry as a shim left it: the base shim's own shape, plus *fields*."""
    entry = {"journal": 2, "sent": True, "state": "sent", "delivery": "posted", "kind": "answer",
             "event_id": event_id, "ref": None, "channel": CHANNEL, "roster_generation": None,
             "reply_to": event_id, "reason": "exit 0:", "attempts": 1, "at": "2026-11-16T00:00:00+0000"}
    entry.update(fields)
    os.makedirs(shim.PUBLISHED, exist_ok=True)
    with open(os.path.join(shim.PUBLISHED, event_id), "w", encoding="utf-8") as handle:
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
    _journal(shim, EVENT_A, body_sha256=_body_sha("hello"))  # the base shim's journal: no disposition
    assert shim.recovery_delivery_state(EVENT_A) == "answered"
    _gateway_answers(shim, "hello", None)
    ask(shim, EVENT_B)
    final(shim, EVENT_B)
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
    assert shim.recovery_delivery_state(EVENT_A) == "answered"  # what the base left
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
