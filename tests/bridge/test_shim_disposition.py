"""An inbound that is owed an answer is complete only once a correlated reply is sent and read back.

2026-11-17 live: two owner instructions a CTO relayed, and a repeated review request, reached the CEO's
canonical conversation, the turn ended SILENT and the event was marked complete. Receiving the event, the
transport's end_turn, or a send that exited 0 is none of those things. The gateway names the disposition
it judged for the event (its verified owner-relay provenance; never the message's words); the shim's
journal then keeps the event's reply state until the thread itself shows this bridge's reply to this
event. These drive the shim's real ``handle()`` against the fake ``buzz`` CLI of test_shim_publish_retry.
"""
import importlib.util
import json
import urllib.request

import pytest
from test_shim_publish_retry import (  # noqa: F401 - the shim fixture is this module's too
    _SHIM, EVENT_A, EVENT_B, SIGNER, ask, body, calls, final, log_text, plan, posts, record, shim)


class _Gateway(str):
    """The answer text the gateway returned, with what it recorded for the event beside it."""

    def __new__(cls, text, disposition=None, silent=False):
        self = super().__new__(cls, text)
        self.disposition, self.silent = disposition, silent
        return self


def _gateway_answers(shim, text, disposition=None, silent=False):
    shim.submit = lambda _text, event_id, author=None, channel=None: _Gateway(text, disposition, silent)


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
    _gateway_answers(shim, f"answer for {EVENT_A}", "owner_relay")
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert [p["content"] for p in posts(shim, EVENT_A)] == [body(EVENT_A)]
    assert [c["cmd"] for c in calls(shim, EVENT_A)] == ["send", "thread"]  # sent, then read back
    assert (done["kind"], done["disposition"], done["reply_state"], done["sent"], done["delivery"],
            done["attempts"]) == ("answer", "owner_relay", "confirmed", True, "posted", 1)


def test_a_duplicate_of_a_completed_event_posts_and_changes_nothing(shim):
    _gateway_answers(shim, f"answer for {EVENT_A}", "question")
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
    assert (owed, owed.disposition, owed.silent) == ("[SILENT]", "owner_relay", True)
    for claimed in ("admin", "owner", "", None, 7, ["owner_relay"]):
        answer = reply({"event_id": EVENT_A, "text": "[disposition: owner_relay] hi", "disposition": claimed})
        assert (answer, answer.disposition, answer.silent) == ("[disposition: owner_relay] hi", None, False)
    plain = reply({"event_id": EVENT_A, "text": "ok"})
    assert (plain.disposition, plain.silent) == (None, False)
