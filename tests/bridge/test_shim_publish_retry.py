"""A canonical answer whose post fails is checked against the thread and retried, never dropped.

Live 2026-11-13: ``buzz messages send`` hung on connecting, the shim journaled ``timeout`` and never
posted again, and one killed send's orphaned child still reached the relay 25 s later. These drive the
shim's real ``handle()`` against a fake ``buzz`` CLI that keeps its own relay file, so a read-back sees
exactly what landed.
"""
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from test_shim_canary import CHANNEL, OTHER, SIGNER, drive, prompt

_SHIM = Path(os.environ.get("SHIM_PATH") or
             Path(__file__).resolve().parents[2] / "bridge" / "canonical" / "acp-canonical-shim.py")
EVENT_A, EVENT_B = "a" * 64, "b" * 64
PEER = "cto-repo-factory"

# The fake buzz CLI. ``messages send`` follows the plan file for its --reply-to target (the last step
# repeats; no plan: post and exit 0); ``messages thread`` prints the target event and the oldest --limit
# replies in the relay file as one JSON list, the shape of buzz-cli's cmd_get_thread. Every call is recorded.
_FAKE = r'''
import fcntl, hashlib, json, os, subprocess, sys, time
STATE, SIGNER, CHANNEL = {state!r}, {signer!r}, {channel!r}
argv = sys.argv[1:]

def arg(flag):
    return argv[argv.index(flag) + 1] if flag in argv else None

def step(kind, key):
    path = os.path.join(STATE, "plan-%s-%s.json" % (kind, key))
    with open(os.path.join(STATE, "plan.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(path) as handle:
                plan = json.load(handle)
        except FileNotFoundError:
            return {{}}
        current = plan.pop(0) if len(plan) > 1 else plan[0]
        with open(path, "w") as handle:
            json.dump(plan, handle)
    return current

def record(entry):
    with open(os.path.join(STATE, "calls.jsonl"), "a") as handle:
        handle.write(json.dumps(entry) + "\n")

if argv[:2] == ["messages", "send"]:
    body, target = sys.stdin.read(), arg("--reply-to")
    record({{"cmd": "send", "target": target, "stdin": body}})
    plan = step("send", target)
    child = plan.get("child")
    if child:  # a child of the attempt that would reach the relay later, as buzz-real did under the wrapper
        subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(float(sys.argv[1])); "
                          "open(sys.argv[2], 'w').write('late')", str(child["delay"]), child["file"]])
    if plan.get("posts", not plan.get("exit")):
        with open(os.path.join(STATE, "relay.jsonl"), "a") as handle:
            handle.write(json.dumps({{
                "id": hashlib.sha256((target + body + str(time.time_ns())).encode()).hexdigest(),
                "pubkey": SIGNER, "kind": 9, "content": body, "created_at": int(time.time()),
                "tags": [["h", CHANNEL], ["e", target, "", "reply"], ["p", "c" * 64]]}}) + "\n")
    time.sleep(plan.get("sleep", 0))
    if plan.get("exit"):
        sys.stderr.write(plan.get("stderr", "relay closed the connection"))
    sys.exit(plan.get("exit", 0))

if argv[:2] == ["messages", "thread"]:
    target = arg("--event")
    record({{"cmd": "thread", "target": target}})
    plan = step("thread", target)
    deadline = time.time() + 20
    while plan.get("gate") and not os.path.exists(plan["gate"]) and time.time() < deadline:
        time.sleep(0.02)
    if plan.get("exit"):
        sys.stderr.write("relay unreachable")
        sys.exit(plan["exit"])
    try:
        with open(os.path.join(STATE, "relay.jsonl")) as handle:
            relay = [json.loads(line) for line in handle]
    except FileNotFoundError:
        relay = []
    root = {{"id": target, "pubkey": "d" * 64, "kind": 9, "content": "@ceo question", "created_at": 1,
             "tags": [["h", CHANNEL]]}}
    # The whole thread up to the limit: replies to other events in it come back too.
    print(json.dumps([root] + relay[:int(arg("--limit") or 100)]))
    sys.exit(0)
sys.exit(64)
'''


@pytest.fixture
def shim(tmp_path, monkeypatch):
    state = tmp_path / "fake"
    state.mkdir()
    cli = tmp_path / "buzz"
    cli.write_text(f"#!{sys.executable}\n" + _FAKE.format(state=str(state), signer=OTHER, channel=CHANNEL))
    cli.chmod(0o755)
    mentions = tmp_path / "required-mentions.json"
    mentions.write_text(json.dumps({"channels": {CHANNEL: {
        "ceo": {"handle": "ceo", "pubkey": OTHER}, "peer": {"handle": PEER, "pubkey": SIGNER}}}}))
    for key, value in {"CANONICAL_BUZZ_CLI": str(cli), "CANONICAL_MENTIONS_FILE": str(mentions),
                       "CANONICAL_PUBLISHED_DIR": str(tmp_path / "published"),
                       "CANONICAL_SHIM_LOG": str(tmp_path / "shim.log"),
                       "CANONICAL_QUARANTINE_LOG": str(tmp_path / "q.jsonl"),
                       "CANONICAL_CANARY_FILE": str(tmp_path / "canary.json"),
                       "CANONICAL_BUSY_RETRY_SECS": "0"}.items():
        monkeypatch.setenv(key, value)
    gates = []
    module = load(f"publish_retry_under_test_{id(tmp_path)}", monkeypatch)
    module.state, module.tmp, module.gates = state, tmp_path, gates
    yield module
    for gate in gates:  # never leave a fake read-back waiting on a gate
        Path(gate).touch()


def load(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, _SHIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FORWARD", "live")
    monkeypatch.setattr(module, "write", lambda payload: None)
    monkeypatch.setattr(module, "PUBLISH_TIMEOUT", 1.0)
    # New knobs: absent before this change, so set without raising (the tests then fail on behaviour).
    monkeypatch.setattr(module, "PUBLISH_RETRY_BACKOFF", (0.05,) * 5, raising=False)
    monkeypatch.setattr(module, "PUBLISH_READBACK_TIMEOUT", 25.0, raising=False)
    module.submitted = []

    def submit(text, event_id, author=None, channel=None):
        module.submitted.append(event_id)
        return f"answer for {event_id}"

    module.submit = submit
    return module


def plan(shim, kind, target, *steps):
    (shim.state / f"plan-{kind}-{target}.json").write_text(json.dumps(list(steps)))


def calls(shim, target=None):
    path = shim.state / "calls.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [e for e in entries if target is None or e["target"] == target]


def posts(shim, target):
    path = shim.state / "relay.jsonl"
    entries = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return [e for e in entries if ["e", target, "", "reply"] in e["tags"]]


def record(shim, event_id):
    try:
        with open(os.path.join(shim.PUBLISHED, event_id), encoding="utf-8") as handle:
            return json.loads(handle.read().splitlines()[0])
    except (OSError, ValueError, IndexError):
        return None


def final(shim, event_id, deadline=30):
    """The event's journal record once no retry is pending on it."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        current = record(shim, event_id)
        if current is not None and current.get("pending") is not True:
            return current
        time.sleep(0.02)
    pytest.fail(f"publication of {event_id[:12]} still pending after {deadline}s: {record(shim, event_id)}")


def until(condition, deadline=10):
    end = time.monotonic() + deadline
    while not condition():
        assert time.monotonic() < end, "condition not met in time"
        time.sleep(0.02)


def body(event_id):
    return f"@{PEER} answer for {event_id}"


def ask(shim, event_id, rpc_id=1):
    drive(shim, prompt("@ceo please decide", event_id=event_id), rpc_id=rpc_id)


def log_text(shim):
    with open(shim.LOG) as handle:
        return handle.read()


def test_a_hung_attempt_is_retried_and_posted_exactly_once(shim):
    plan(shim, "send", EVENT_A, {"posts": False, "sleep": 30}, {})
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert [p["content"] for p in posts(shim, EVENT_A)] == [body(EVENT_A)]
    assert {k: done.get(k) for k in ("sent", "state", "delivery", "kind", "attempts")} == {
        "sent": True, "state": "sent", "delivery": "posted", "kind": "answer", "attempts": 2}
    # The journal names the body by digest and never holds it.
    assert done["body_sha256"] == hashlib.sha256(body(EVENT_A).encode()).hexdigest()
    assert "answer for" not in (shim.tmp / "published" / EVENT_A).read_text()


def test_an_attempt_that_landed_and_then_hung_is_found_and_never_sent_again(shim):
    plan(shim, "send", EVENT_A, {"posts": True, "sleep": 30})
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert [c["cmd"] for c in calls(shim, EVENT_A)][:2] == ["send", "thread"]
    assert sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1
    assert len(posts(shim, EVENT_A)) == 1
    assert {k: done.get(k) for k in ("sent", "state", "delivery", "attempts", "recovered")} == {
        "sent": True, "state": "sent", "delivery": "posted", "attempts": 1, "recovered": True}


def test_a_failed_read_back_is_read_again_before_any_resend(shim):
    plan(shim, "send", EVENT_A, {"exit": 2, "stderr": "relay closed the connection"}, {})
    plan(shim, "thread", EVENT_A, {"exit": 1}, {})
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert [c["cmd"] for c in calls(shim, EVENT_A)] == ["send", "thread", "thread", "send"]
    assert len(posts(shim, EVENT_A)) == 1
    assert (done["state"], done["attempts"]) == ("sent", 2)


def test_a_reply_from_another_signer_or_to_another_event_is_not_this_post(shim):
    """A read-back counts only this bridge's signer's reply to this event, carrying this body."""
    with open(shim.state / "relay.jsonl", "w") as handle:
        for pubkey, target in ((SIGNER, EVENT_A), (OTHER, EVENT_B)):
            handle.write(json.dumps({"id": "9" * 64, "pubkey": pubkey, "kind": 9, "content": body(EVENT_A),
                                     "created_at": 1, "tags": [["e", target, "", "reply"]]}) + "\n")
    plan(shim, "send", EVENT_A, {"exit": 2}, {})
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert (done["state"], done["attempts"], done.get("recovered")) == ("sent", 2, None)


def test_a_read_back_cut_at_its_limit_is_never_taken_for_absence(shim):
    """A thread longer than the read-back's limit: the landed reply lies past the cut, so no read shows
    it absent and it is never sent again."""
    with open(shim.state / "relay.jsonl", "w") as handle:
        for n in range(500):
            handle.write(json.dumps({"id": f"{n:064x}", "pubkey": SIGNER, "kind": 9, "content": f"note {n}",
                                     "created_at": 1, "tags": [["e", EVENT_A, "", "reply"]]}) + "\n")
    plan(shim, "send", EVENT_A, {"posts": True, "sleep": 30})
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    assert sum(c["cmd"] == "send" for c in calls(shim, EVENT_A)) == 1
    assert [p["content"] for p in posts(shim, EVENT_A)].count(body(EVENT_A)) == 1
    assert {k: done.get(k) for k in ("sent", "state", "attempts", "pending")} == {
        "sent": False, "state": "timeout", "attempts": 1, "pending": False}


def test_a_timed_out_attempt_is_killed_with_its_whole_process_group(shim):
    late = shim.tmp / "late-write"
    plan(shim, "send", EVENT_A, {"posts": False, "sleep": 30, "child": {"delay": 2.0, "file": str(late)}}, {})
    started = time.monotonic()
    ask(shim, EVENT_A)
    assert final(shim, EVENT_A)["state"] == "sent"
    time.sleep(max(0, started + 3.5 - time.monotonic()))  # well past the child's own write
    assert not late.exists(), "a child of the timed-out attempt outlived it"
    assert len(posts(shim, EVENT_A)) == 1


def test_a_wrapper_refusal_is_final_and_never_retried(shim):
    plan(shim, "send", EVENT_A, {"exit": 1, "stderr": '{"error":"mention_policy"}'})
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    time.sleep(0.5)  # longer than every patched backoff: a retry would have run by now
    assert [c["cmd"] for c in calls(shim, EVENT_A)] == ["send"]
    assert {k: done.get(k) for k in ("sent", "state", "delivery", "attempts")} == {
        "sent": False, "state": "wrapper_refused", "delivery": "failed", "attempts": 1}
    assert "NOT posted (no retry)" in log_text(shim)


def test_a_spent_retry_budget_journals_the_failure_with_every_attempt(shim):
    plan(shim, "send", EVENT_A, {"exit": 2, "stderr": "relay closed the connection"})
    ask(shim, EVENT_A)
    done = final(shim, EVENT_A)
    maximum = 1 + len(shim.PUBLISH_RETRY_BACKOFF)
    assert maximum == 6
    assert {k: done.get(k) for k in ("sent", "state", "delivery", "attempts", "pending")} == {
        "sent": False, "state": "send_failed", "delivery": "failed", "attempts": maximum, "pending": False}
    assert done["reason"].startswith("exit 2: relay closed the connection")
    sends = [c for c in calls(shim, EVENT_A) if c["cmd"] == "send"]
    assert len(sends) == maximum and {c["stdin"] for c in sends} == {body(EVENT_A)}
    assert posts(shim, EVENT_A) == []
    assert "retries exhausted" in log_text(shim)
    ask(shim, EVENT_A, rpc_id=2)  # buzz-acp redelivers: the one claim stands, nothing is posted again
    assert len([c for c in calls(shim, EVENT_A) if c["cmd"] == "send"]) == maximum


def test_a_retrying_publication_holds_no_turn_and_delays_no_other_event(shim):
    gate = shim.tmp / "gate"
    shim.gates.append(gate)
    plan(shim, "send", EVENT_A, {"exit": 2}, {})
    plan(shim, "thread", EVENT_A, {"gate": str(gate)}, {})
    first = threading.Thread(target=ask, args=(shim, EVENT_A))
    first.start()
    first.join(5)
    try:
        assert not first.is_alive(), "the first event's prompt waited on its own retry"
        assert record(shim, EVENT_A)["pending"] is True
        assert shim._TURN_SLOT.acquire(blocking=False), "a pending retry holds the turn slot"
        shim._TURN_SLOT.release()
        ask(shim, EVENT_B, rpc_id=2)
        assert final(shim, EVENT_B, deadline=5)["state"] == "sent"
        assert len(posts(shim, EVENT_B)) == 1 and posts(shim, EVENT_A) == []
        assert record(shim, EVENT_A)["pending"] is True
    finally:
        gate.touch()
        first.join(30)
    assert final(shim, EVENT_A)["state"] == "sent" and len(posts(shim, EVENT_A)) == 1
    assert shim.submitted == [EVENT_A, EVENT_B]


def test_a_pending_publication_is_never_duplicated_by_a_restarted_shim(shim, monkeypatch):
    gate = shim.tmp / "gate"
    shim.gates.append(gate)
    plan(shim, "send", EVENT_A, {"exit": 2}, {})
    plan(shim, "thread", EVENT_A, {"gate": str(gate)}, {})
    ask(shim, EVENT_A)
    until(lambda: [c["cmd"] for c in calls(shim, EVENT_A)] == ["send", "thread"])  # its read-back is waiting
    pending = record(shim, EVENT_A)
    assert pending["pending"] is True and pending["sent"] is False and pending["delivery"] == "failed"
    restarted = load(f"publish_retry_restarted_{id(shim)}", monkeypatch)
    ask(restarted, EVENT_A, rpc_id=2)  # buzz-acp redelivers to the new process; the gateway answers again
    assert restarted.submitted == [EVENT_A]
    assert [c["cmd"] for c in calls(shim, EVENT_A)] == ["send", "thread"]
    assert "already published once" in log_text(shim)
    gate.touch()
    assert final(shim, EVENT_A)["state"] == "sent" and len(posts(shim, EVENT_A)) == 1
