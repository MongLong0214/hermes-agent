"""#92 review round 1 (H92-01, H92-02): the reviewer's witnesses and ours.

The reviewer's (``h92-r1/review_evidence/test_h92_counterexamples.py`` and
``test_h92_lifecycle_controls.py``) are copied below byte for byte, under their two banners; only the
imports and fixture aliases they shared are merged into this one header. Ours follow, synchronized by
events, hooks and process exits — never by sleeping.

H92-01: the shim's ``PUBLISHED/<event>.inflight`` entry is published only once initialized and locked,
every reliance on it confirms the entry at the path is the locked file (st_dev, st_ino), and an event a
shim submitted and then ended without its outcome is never submitted again: the reaper's existing
``unfinished_notice`` journal entry is read for that alone. Refusing resubmission is no execution or
delivery guarantee: such a turn stays unknown.

H92-02: matching a recovery nudge's text is no provenance, and no nudge carries metadata bound to the
run, so any user row inside the turn — and any peer-marked row but the turn's own opening row — leaves
the answer unbound (kept nowhere, one explicit failure notice).
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import test_shim_d2_cancel_keeps_turn as o2
import test_shim_d2_refused_answer_kept as o4

import agent.conversation_loop as loop  # already imported, under its pinned home, by o4

gw = o4.gw
shim = o4.shim
rig = o2.rig
_pin_the_module_home = o4._pin_the_module_home


# ════════════════ reviewer: test_h92_counterexamples.py (verbatim) ════════════════

def test_reaper_cannot_consume_a_live_submitters_new_entry(gw, shim, monkeypatch):
    """Pause after O_CREAT but before flock, precisely where another process may reap."""
    gw.actor.shape = "plain"
    created, resume, done = threading.Event(), threading.Event(), threading.Event()
    real_open = os.open

    def pause_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if str(path).endswith(o4.EVENT + ".inflight") and flags & os.O_CREAT:
            created.set()
            assert resume.wait(10)
        return fd

    monkeypatch.setattr(os, "open", pause_open)
    prompt = {"jsonrpc": "2.0", "id": 9, "method": "session/prompt",
              "params": o4.bp.prompt_params("s", o4.channel_prompt("x" * 300))}
    assert shim.envelope(prompt["params"]["prompt"])[:3] == (o4.SIGNER, o4.CHANNEL, o4.EVENT)

    def drive():
        try:
            shim.handle(prompt)
        finally:
            done.set()

    worker = threading.Thread(target=drive)
    worker.start()
    try:
        # On the base there is no in-flight entry. The normal request simply completes.
        for _ in range(100):
            if created.wait(.02) or done.is_set():
                break
        if created.is_set():
            code = ("import importlib.util; "
                    f"s=importlib.util.spec_from_file_location('reaper', {str(o4._SHIM)!r}); "
                    "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.reap_unfinished()")
            reaper = subprocess.run([sys.executable, "-c", code], env=dict(os.environ),
                                    capture_output=True, timeout=10)
            assert reaper.returncode == 0, reaper.stderr
            record = json.loads((shim.root / "published" / o4.EVENT).read_text())
            assert record["state"] == "notice_impossible"
    finally:
        resume.set()
        worker.join(15)
    assert not worker.is_alive()
    o4._attributed_and_submitted(gw, shim)
    assert len(gw.runs) == 1 and gw.responses[-1][0] == 200
    posts = shim.posts()
    print("REAPER_RACE", json.dumps({"responses": gw.responses, "posts": posts,
        "marker": json.loads((shim.root / "published" / o4.EVENT).read_text())}))
    assert [post["stdin"] for post in posts] == [f"@cto-repo-factory {o4.FINAL}"]


def test_a_foreign_claims_user_row_is_not_a_recovery_nudge(gw, shim, monkeypatch):
    original = gw.actor.run_conversation

    def other_claim(text, **kwargs):
        result = original(text, **kwargs)
        peer = o4.new_peer_metadata(binding="canonical", author_id=o4.SIGNER, channel_id=o4.CHANNEL,
                                   event_id="f" * 64, receipt="canonical-receipt:v2:" + "e" * 64)
        result["messages"][-2] = {"role": "user", "content": o4._CODEX_INCOMPLETE_NUDGE,
                                  "display_kind": "canonical_peer", "display_metadata": {"canonical_peer": peer}}
        return result

    monkeypatch.setattr(gw.actor, "run_conversation", other_claim)
    o4._drive(shim)
    o4._reached_the_refusal(gw, shim)
    print("FOREIGN_NUDGE", json.dumps({"http": gw.responses, "kept": gw.rows("canonical-answer-kept:")}))
    assert gw.rows("canonical-answer-kept:") == {}, "another claim's peer row was treated as a system nudge"


@pytest.mark.parametrize("keep_reason", ["selector_refused", "record_refused"])
def test_competing_record_observation_still_has_our_answer_in_readable_fallback(gw, shim, monkeypatch, keep_reason):
    """Observe the different read sources, without assuming O3 replay recovery is required by O4."""
    claim = gw.db.claim_meta_once

    def competing_record(key, value, **kwargs):
        if key.startswith("canonical-answer-kept:"):
            competing = json.loads(value)
            competing["owner"] = "0" * 32
            competing["answer"]["response"]["terminal_text"] = "OTHER CLAIM'S ANSWER"
            competing["answer_sha256"] = hashlib.sha256(b"OTHER CLAIM'S ANSWER").hexdigest()
            assert claim(key, json.dumps(competing), **kwargs)
        return claim(key, value, **kwargs)

    monkeypatch.setattr(gw.db, "claim_meta_once", competing_record)
    if keep_reason == "record_refused":
        gw.actor.shape = "plain"
        monkeypatch.setattr(gw.db, "compare_and_set_meta", lambda *args, **kwargs: False)
    o4._drive(shim)
    if keep_reason == "selector_refused":
        o4._reached_the_refusal(gw, shim)
    else:
        o4._attributed_and_submitted(gw, shim)
        assert len(gw.runs) == 1 and gw.refusals == []
    [value] = gw.rows("canonical-answer-kept:").values()
    [receipt_key] = gw.rows("canonical-receipt:")
    coordinator = o4.CanonicalReceiptCoordinator(None)
    replay_kept = coordinator._read_records(gw.db, receipt_key, gw.home)[0]
    fingerprint = json.loads(value)["fingerprint"]
    assert coordinator._decode_kept(replay_kept, fingerprint).failure["answer_kept"] is True
    replay_text = json.loads(replay_kept)["answer"]["response"]["terminal_text"]
    print("KEEP_PRECEDENCE", json.dumps({"http": gw.responses, "replay_text": replay_text,
        "fallback": [json.loads(p.read_text()) for p in (gw.home / "canonical-answer-kept").iterdir()]}))
    own_fallback = coordinator._fallback_kept(gw.home, receipt_key)
    assert coordinator._decode_kept(own_fallback, fingerprint).failure["answer_kept"] is True
    assert json.loads(own_fallback)["answer"]["response"]["terminal_text"] == o4.FINAL


# ════════════════ reviewer: test_h92_lifecycle_controls.py (verbatim) ════════════════

@pytest.mark.parametrize("rig", [3.0], indirect=True)
def test_two_live_shims_do_not_submit_the_same_running_event_twice(rig):
    first = rig.spawn("first")
    session = first.start_session()
    first.prompt(2, session, o2.channel_prompt(o2.E1, o2.ROOT_TEXT))
    assert rig.gw.claim_of(o2.E1).wait(10)
    second = rig.spawn("second")
    session2 = second.start_session()
    second.prompt(3, session2, o2.channel_prompt(o2.E1, o2.ROOT_TEXT))
    assert second.wait_result(3, 10) is not None
    first.close()
    second.close()
    assert rig.gw.received == [o2.E1] and rig.gw.runs == [o2.E1]
    assert len(o2._answer_posts(rig, o2.E1)) == 1
    assert o2._no_inflight(rig) == []


@pytest.mark.parametrize("rig", [2.0], indirect=True)
def test_stdin_eof_waits_for_the_inflight_answer_to_post(rig):
    agent = rig.spawn("eof")
    session = agent.start_session()
    agent.prompt(2, session, o2.channel_prompt(o2.E1, o2.ROOT_TEXT))
    assert rig.gw.claim_of(o2.E1).wait(10)
    agent.close()
    assert agent.proc.returncode == 0
    assert rig.gw.received == [o2.E1] and len(o2._answer_posts(rig, o2.E1)) == 1
    assert len(agent.responses_to(2)) == 1


@pytest.mark.parametrize("rig", [7.0], indirect=True)
def test_replacement_does_not_resubmit_the_same_still_running_event(rig):
    first = rig.spawn("before-replacement")
    session = first.start_session()
    first.prompt(2, session, o2.channel_prompt(o2.E1, o2.ROOT_TEXT))
    assert rig.gw.claim_of(o2.E1).wait(10) and o2._attributed(rig, o2.E1) == 1
    inflight_was_recorded = bool(o2._no_inflight(rig))
    first.kill()
    second = rig.spawn("after-replacement")
    session2 = second.start_session()
    if inflight_was_recorded:
        assert o2._wait(lambda: rig.journal(o2.E1) is not None and not o2._no_inflight(rig), 3)
    assert o2._running(rig, o2.E1), "the first turn must still be running"
    # A real, individually attributed event, not the merged prompt H91 refuses first.
    second.prompt(3, session2, o2.channel_prompt(o2.E1, o2.ROOT_TEXT))
    assert second.wait_result(3, 10) is not None
    assert o2._attributed(rig, o2.E1) == 2
    print("REPLACEMENT_REDELIVERY", json.dumps({"received": rig.gw.received,
        "runs": rig.gw.runs, "responses": rig.gw.responses, "posts": rig.posts()}))
    assert rig.gw.received == [o2.E1], "the replacement submitted the still-running event again"


def test_a_prompt_thread_exception_releases_the_slot_and_posts_one_failure(gw, shim, monkeypatch):
    gw.actor.shape = "plain"
    captured = []

    def fail_chunk(payload):
        if payload.get("method") == "session/update":
            raise RuntimeError("controlled output failure")
        captured.append(payload)

    # The fixture suppresses stdout; raise specifically after the real submission completes.
    monkeypatch.setattr(shim, "write", fail_chunk)
    shim.dispatch({"jsonrpc": "2.0", "id": 91, "method": "session/prompt",
                   "params": o4.bp.prompt_params("s", o4.channel_prompt("x" * 300))})
    for worker in shim._THREADS:
        worker.join(15)
        assert not worker.is_alive()
    o4._attributed_and_submitted(gw, shim)
    assert len(gw.runs) == 1 and gw.responses[0][0] == 200
    [post] = shim.posts()
    assert o4.FINAL not in post["stdin"] and o4.EVENT in post["stdin"]
    assert json.loads((shim.root / "published" / o4.EVENT).read_text())["kind"] == "unfinished_notice"
    assert shim._INFLIGHT == set() and not shim._TURN_SLOT.locked()


def test_refusal_kept_before_a_failed_terminal_write_never_becomes_success(gw, shim, monkeypatch):
    monkeypatch.setattr(gw.db, "compare_and_set_meta", lambda *args, **kwargs: False)
    o4._drive(shim)
    o4._reached_the_refusal(gw, shim)
    assert o4._receipt(gw)["state"] == "pending"
    assert o4._held(gw) == [o4.FINAL]
    o4._drive(shim, rpc_id=5)
    assert len(gw.runs) == 1 and len(shim.posts()) == 1
    assert o4._codes(gw) == [(409, "canonical_answer_unrecorded")] * 2
    assert o4.FINAL not in shim.posts()[0]["stdin"]


# ════════════════ ours ════════════════

INFLIGHT = o4.EVENT + ".inflight"


def _log(shim):
    path = shim.root / "shim.log"
    return path.read_text() if path.exists() else ""


def _reaper(shim, env=None):
    """Another shim process's startup reaper, run to completion over the same journal."""
    code = ("import importlib.util; "
            f"s=importlib.util.spec_from_file_location('reaper', {str(o4._SHIM)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.reap_unfinished()")
    done = subprocess.run([sys.executable, "-c", code], env=dict(env or os.environ), capture_output=True,
                          timeout=30)
    assert done.returncode == 0, done.stderr


# ── H92-01 (a): the entry's name never exists before its owner holds it ──────────────────────────────


def test_h92_01a_another_shims_reaper_at_the_instant_the_entry_appears_consumes_nothing(gw, shim, monkeypatch):
    """Paused at the very call after which ``<event>.inflight`` first exists — whichever call that is —
    another process's reaper runs to completion there, then the submitter resumes."""
    gw.actor.shape = "plain"
    path = shim.root / "published" / INFLIGHT
    reached, stop, resume = [], threading.Event(), threading.Event()
    real_open, real_link = os.open, os.link

    def pause_once_visible():
        if not reached and path.exists():
            reached.append(threading.get_ident())
            stop.set()
            assert resume.wait(30)

    def spy_open(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        pause_once_visible()
        return fd

    def spy_link(*args, **kwargs):
        real_link(*args, **kwargs)
        pause_once_visible()

    monkeypatch.setattr(os, "open", spy_open)
    monkeypatch.setattr(os, "link", spy_link)

    def drive():
        try:
            o4._drive(shim)
        finally:
            stop.set()

    worker = threading.Thread(target=drive)
    worker.start()
    try:
        assert stop.wait(30)
        assert reached, "reach: the submitter never made the entry visible"
        _reaper(shim)
        assert path.exists(), "the reaper consumed a live submitter's entry"
        assert not (shim.root / "published" / o4.EVENT).exists(), "the reaper took the event's one post"
    finally:
        resume.set()
        worker.join(30)
    assert not worker.is_alive()
    o4._attributed_and_submitted(gw, shim)
    assert len(gw.runs) == 1 and gw.responses[-1][0] == 200
    assert [post["stdin"] for post in shim.posts()] == [f"@cto-repo-factory {o4.FINAL}"]
    assert not path.exists()


# ── H92-01: ownership is the locked file at the path, checked when relied on ─────────────────────────


def test_h92_01c_an_entry_replaced_after_it_was_claimed_is_no_ownership(gw, shim, monkeypatch):
    gw.actor.shape = "plain"
    claim, replaced = shim._claim_inflight, []

    def claim_then_replace(base, channel):
        got = claim(base, channel)
        path = os.path.join(shim.PUBLISHED, base + ".inflight")
        os.unlink(path)
        with open(path, "w", encoding="utf-8") as other:
            other.write("{}\n")  # another file now holds the name
        replaced.append(path)
        return got

    monkeypatch.setattr(shim, "_claim_inflight", claim_then_replace)
    o4._drive(shim)

    assert replaced and f"base={o4.EVENT[:12]}" in _log(shim), "reach: attributed, and the entry was claimed"
    assert gw.responses == [] and gw.runs == [], "submitted without holding the event's entry"
    assert Path(replaced[0]).read_text() == "{}\n", "another file's entry was removed"


# ── H92-01 (b): an event whose shim ended without its outcome is never submitted again ───────────────


def test_h92_01d_a_shim_stopped_after_its_record_and_before_submitting_stays_unresolved(gw, shim):
    """The record is written and the shim stops before submitting: whether the turn ran is unknown, so a
    redelivery is never submitted — the reaper's one notice says unknown."""
    gw.actor.shape = "plain"
    published = shim.root / "published"
    published.mkdir()
    entry = published / INFLIGHT
    entry.write_text(json.dumps({"event_id": o4.EVENT, "channel": o4.CHANNEL, "pid": 1, "at": "-"}) + "\n")
    entry.chmod(0o600)
    shim.reap_unfinished()  # a later shim: no live process holds the entry
    [notice] = shim.posts()
    assert json.loads((published / o4.EVENT).read_text())["kind"] == "unfinished_notice"
    assert "알 수 없고" in notice["stdin"], "reach: the unresolved state exists"

    o4._drive(shim)  # buzz-acp redelivers the same, valid single event

    assert f"base={o4.EVENT[:12]}" in _log(shim), "reach: the redelivery passed the attribution guard"
    assert gw.responses == [] and gw.runs == [], "an unresolved event was submitted again"
    assert len(shim.posts()) == 1 and "NOT submitted again" in _log(shim)


class GatedGateway:
    """The canonical surface's receipt rules for one session (as ``o2.FakeGateway``), with a turn that
    runs until ``gate`` is set: the test, not a clock, decides when it ends."""

    def __init__(self):
        self.lock, self.gate, self.claimed, self.done = threading.Lock(), *(threading.Event() for _ in range(3))
        self.receipts, self.received, self.runs, self.responses = {}, [], [], []
        gw = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, event_id, status, payload):
                body = json.dumps(payload).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass
                gw.responses.append((event_id, status, payload.get("error", {}).get("code")))

            def do_POST(self):
                event = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                event_id, fp = event["event_id"], hashlib.sha256(event["text"].encode()).hexdigest()
                with gw.lock:
                    gw.received.append(event_id)
                    record = gw.receipts.get(event_id)
                    if record is None and any(r["state"] == "pending" for r in gw.receipts.values()):
                        refusal = "canonical_turn_busy"
                    elif record is None:
                        refusal = None
                        gw.receipts[event_id] = {"fp": fp, "state": "pending"}
                        gw.runs.append(event_id)
                    elif record["state"] == "pending":
                        refusal = "canonical_event_uncertain"
                    else:
                        refusal = ""
                if refusal:
                    self._send(event_id, 409, {"error": {"code": refusal, "message": "x"}})
                    return
                if refusal == "":
                    self._send(event_id, 200, {"event_id": event_id, "text": f"ANSWER for {event_id[:8]}"})
                    return
                gw.claimed.set()
                gw.gate.wait(60)
                with gw.lock:
                    gw.receipts[event_id]["state"] = "terminal"
                self._send(event_id, 200, {"event_id": event_id, "text": f"ANSWER for {event_id[:8]}"})
                gw.done.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def pending(self, event_id):
        with self.lock:
            return self.receipts.get(event_id, {}).get("state") == "pending"


@pytest.fixture
def gated(tmp_path):
    gw = GatedGateway()
    root = tmp_path / "bridge"
    root.mkdir()
    calls = root / "buzz-calls.jsonl"
    (root / "buzz").write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}) + '\\n')\n")
    (root / "buzz").chmod(0o755)
    (root / "required-mentions.json").write_text(json.dumps({"channels": {o2.CHANNEL: {
        "ceo": {"handle": "ceo", "pubkey": o2.OTHER}, "peer": {"handle": "cto-repo-factory", "pubkey": o2.SIGNER}}}}))
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(tmp_path),
        "ACP_RECEIPT_PORT_JS": str(tmp_path / "missing-receipt-port.js"),
        "CANONICAL_SURFACE_URL": gw.url, "API_SERVER_KEY": "h92-placeholder", "CANONICAL_BINDING": "canonical",
        "CANONICAL_AUTHOR_ID": o2.SIGNER, "CANONICAL_CHANNEL_ID": o2.CHANNEL, "CANONICAL_FORWARD": "live",
        "CANONICAL_BUSY_RETRY_SECS": "1", "CANONICAL_BUSY_DEADLINE_SECS": "40",
        "CANONICAL_SHIM_LOG": str(root / "shim.log"), "CANONICAL_QUARANTINE_LOG": str(root / "q.jsonl"),
        "CANONICAL_CANARY_FILE": str(root / "canary.json"), "CANONICAL_PUBLISHED_DIR": str(root / "published"),
        "CANONICAL_MENTIONS_FILE": str(root / "required-mentions.json"), "CANONICAL_BUZZ_CLI": str(root / "buzz"),
    }
    agents = []

    def spawn(label):
        agent = o2.Agent(env, root, label)
        agents.append(agent)
        return agent

    def posts():
        return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []

    try:
        yield type("Gated", (), {"gw": gw, "spawn": staticmethod(spawn), "posts": staticmethod(posts), "root": root,
                                 "env": env})
    finally:
        gw.gate.set()
        for agent in agents:
            agent.close()
        gw.server.shutdown()
        gw.server.server_close()


def test_h92_01b_after_a_replacement_a_redelivery_of_the_still_running_event_is_not_submitted(gated):
    first = gated.spawn("first")
    session = first.start_session()
    first.prompt(2, session, o2.channel_prompt(o2.E1, o2.ROOT_TEXT))
    assert gated.gw.claimed.wait(10), "reach: the event was never submitted"
    first.kill()  # replaced with the turn running: exact PID, and waited for
    _reaper(None, gated.env)  # the next shim's reaper, run to its exit: the event is now unfinished
    record = json.loads((gated.root / "published" / o2.E1).read_text())
    assert record["kind"] == "unfinished_notice" and not (gated.root / "published" / (o2.E1 + ".inflight")).exists()
    second = gated.spawn("second")
    session = second.start_session()
    second.prompt(3, session, o2.channel_prompt(o2.E1, o2.ROOT_TEXT))  # a valid single-event redelivery
    assert second.wait_result(3, 30) is not None
    assert gated.gw.pending(o2.E1), "reach: the first turn still runs while the redelivery is handled"
    received_while_running = list(gated.gw.received)
    gated.gw.gate.set()
    assert gated.gw.done.wait(30)
    second.close()  # joins its threads, the startup reaper's post included

    log = (gated.root / "shim.log").read_text()
    assert log.count(f" base={o2.E1[:12]}") == 2, "reach: both dispatches passed the attribution guard"
    assert received_while_running == [o2.E1], f"the still-running event was submitted again: {gated.gw.responses}"
    assert gated.gw.runs == [o2.E1]
    assert [p for p in gated.posts() if "ANSWER for" in p["stdin"]] == []  # its client was replaced
    assert len([p for p in gated.posts() if o2.E1 in p["stdin"]]) == 1


# ── H92-02: no nudge is provenance; any foreign peer row unbinds the turn ────────────────────────────

NUDGES = ("_CODEX_INCOMPLETE_NUDGE", "_CODEX_ACK_CONTINUATION_NUDGE", "_DEGENERATE_FINAL_NUDGE",
          "_DROPPED_TOOLCALL_NUDGE_CONTENT", "_EMPTY_TOOL_RESPONSE_NUDGE")
# The only metadata the agent loop puts on a nudge row it appends: run-agnostic booleans, two nudges only.
FLAGS = {"_EMPTY_TOOL_RESPONSE_NUDGE": "_empty_recovery_synthetic",
         "_DROPPED_TOOLCALL_NUDGE_CONTENT": "_dropped_toolcall_nudge"}


def _another_events_peer():
    return o4.new_peer_metadata(binding="canonical", author_id=o4.SIGNER, channel_id=o4.CHANNEL,
                                event_id="f" * 64, receipt="canonical-receipt:v2:" + "e" * 64)


def _insert_before_final(gw, monkeypatch, row):
    original = gw.actor.run_conversation

    def with_row(text, **kwargs):
        result = original(text, **kwargs)
        result["messages"].insert(len(result["messages"]) - 1, row)
        return result

    monkeypatch.setattr(gw.actor, "run_conversation", with_row)


def _unbound_and_noticed(gw, shim):
    o4._reached_the_refusal(gw, shim)
    assert gw.rows("canonical-answer-kept:") == {} and o4._held(gw) == [], "an unbound answer was kept"
    assert gw.bound == [None]
    assert o4._codes(gw) == [(409, "canonical_event_failed")]
    [post] = shim.posts()
    assert "canonical_event_failed" in post["stdin"] and o4.FINAL not in post["stdin"]


@pytest.mark.parametrize(("nudge", "variant"), [(n, v) for n in NUDGES for v in ("plain", "another_events_peer")]
                         + [(n, "its_own_flag") for n in FLAGS])
def test_h92_02_a_nudge_inside_the_turn_is_no_provenance(gw, shim, monkeypatch, nudge, variant):
    gw.actor.shape = "unnudged_retry"  # bound and kept without the row (see the control below)
    row = {"role": "user", "content": getattr(loop, nudge)}
    if variant == "another_events_peer":
        row.update(display_kind="canonical_peer", display_metadata={"canonical_peer": _another_events_peer()})
    elif variant == "its_own_flag":
        row[FLAGS[nudge]] = True
    _insert_before_final(gw, monkeypatch, row)
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


def test_h92_02_another_events_peer_marker_on_an_assistant_row_of_the_turn_unbinds_it(gw, shim, monkeypatch):
    gw.actor.shape = "unnudged_retry"
    _insert_before_final(gw, monkeypatch, {"role": "assistant", "content": "", "finish_reason": "incomplete",
                                           "display_kind": "canonical_peer",
                                           "display_metadata": {"canonical_peer": _another_events_peer()}})
    o4._drive(shim)
    _unbound_and_noticed(gw, shim)


def test_h92_02_control_the_same_events_normal_turn_is_recorded_and_posted_once(gw, shim):
    gw.actor.shape = "plain"
    o4._drive(shim)
    o4._attributed_and_submitted(gw, shim)
    assert gw.refusals == [] and gw.bound == [] and o4._held(gw) == [o4.FINAL]
    assert [post["stdin"] for post in shim.posts()] == [f"@cto-repo-factory {o4.FINAL}"]


def test_h92_02_control_a_refused_turn_with_no_row_but_its_own_is_still_kept(gw, shim):
    gw.actor.shape = "unnudged_retry"
    o4._drive(shim)
    o4._reached_the_refusal(gw, shim)
    assert gw.bound == [o4.FINAL] and o4._held(gw) == [o4.FINAL]
    assert o4._codes(gw) == [(409, "canonical_answer_unrecorded")] and len(shim.posts()) == 1
