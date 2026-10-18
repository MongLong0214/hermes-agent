"""D2 / O2: buzz-acp's steer cancel never costs a running canonical turn its answer, and never runs an
event twice or two turns at once.

W1 of DIAG-H1110-1111 D2 (``tests/diag_d2_test_shim_replaced_midturn.py``), ported to the prompt shape
the shim reads since #1111 P1 and #91 r1 (8ccba8b4): buzz-acp's sections as separate ``params.prompt`` elements, built with
``buzz_prompt_port.format_prompt``. Live, event 813415c3…'s turn was in flight when a thread @mention
arrived; under the default ``--multiple-event-handling steer`` buzz-acp sent ``session/cancel`` and
waited ``CONTROL_CANCEL_GRACE`` = 5 s (pool.rs 831, 2013). The shim read stdin only between turns, so
nothing answered; the drain timed out, the agent's process group was SIGKILLed and respawned (lib.rs
3643), and the answer the gateway completed 55 s later was posted to nobody.

The real shim runs as a subprocess. Around it: a driver doing what buzz-acp does; a fake canonical
gateway keeping the live receipt rules (a turn goes on after its client is gone, a replay answers from
the receipt, a pending receipt is ``canonical_event_uncertain``, an unclaimed event while a turn runs is
``canonical_turn_busy``); and a stand-in for the Buzz wrapper that records each post.

Checked at each boundary (normal completion, a cancel race, shim replacement): the event's turn runs at
most once, its answer is posted at most once, and an answer that is not posted ends as an explicit
failure. Crash safety is not claimed: the replacement notice comes from the next shim that starts.
"""

import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import buzz_prompt_port as bp
import pytest

SHIM = Path(os.environ.get("SHIM_PATH") or
            Path(__file__).resolve().parents[2] / "bridge" / "canonical" / "acp-canonical-shim.py")
ROUTE = "/v1/canonical-surface/events"
SIGNER = "73bd7a526adb00a06ed2ec820701e2f761d00cc0dc0ed1fb051f8923d6fdc75a"  # the sender (CTO)
OTHER = "e79dda8ddd8f994b5758a2f8eed4bc76b6b9c418d16ed016f9e437ee7c35ec4d"  # this bridge's signer
CHANNEL = "b9cda259-f165-45ce-afaf-3e8f078a144a"
NAME = "repo-factory"
LABELS = {SIGNER: "cto-repo-factory"}
E1 = "813415c3" + "1" * 56  # the root message whose turn was in flight
E2 = "84aa4fe1" + "2" * 56  # the thread @mention that triggered the steer cancel
E3 = "9a3f00d1" + "3" * 56  # another top-level message, arriving later
CANCEL_GRACE = 5.0  # pool.rs CONTROL_CANCEL_GRACE
TURN_SECS = 9.0  # the gateway turn outlives the grace; live it took 61 s
ROOT_TEXT = "root message without a mention " + "x" * 400


def _ev(event_id, content, *, root=None, tag="message", at=1_792_324_251):
    return bp.Ev(event_id, SIGNER, at, content, [["h", CHANNEL]] + ([["e", root, "", "root"]] if root else []),
                 prompt_tag=tag)


def channel_prompt(event_id, content):
    """buzz-acp's single-event prompt for a top-level message (queue.rs format_prompt)."""
    return bp.format_prompt(CHANNEL, NAME, [_ev(event_id, content)], LABELS)


def merged_steer_prompt(cancelled_id, cancelled, new_id, new):
    """buzz-acp's prompt after a steer cancel: the requeued batch, then the new event, then the note."""
    return bp.format_prompt(CHANNEL, NAME, [_ev(new_id, new, root=cancelled_id, tag="@mention")], LABELS,
                            cancelled=[_ev(cancelled_id, cancelled)], framing=bp.STEER)


class FakeGateway:
    """The canonical surface's receipt rules for one bound session, with a turn of ``turn_secs``."""

    def __init__(self, turn_secs=TURN_SECS):
        self.turn_secs = turn_secs
        self.lock = threading.Lock()
        self.receipts = {}  # event id -> {"fp", "state", "text"}
        self.runs = []  # event ids whose turn ran
        self.received = []  # event ids of every POST
        self.responses = []  # (event id, status, code, client still there)
        self.running = 0
        self.max_running = 0
        self.claimed = {}  # event id -> threading.Event
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def answer_for(self, event_id):
        return f"ANSWER for {event_id[:8]}"

    def claim_of(self, event_id):
        with self.lock:
            return self.claimed.setdefault(event_id, threading.Event())

    def _handler(self):
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
                    self.wfile.flush()
                    delivered = True
                except OSError:
                    delivered = False
                code = payload.get("error", {}).get("code") if status != 200 else None
                gw.responses.append((event_id, status, code, delivered))

            def do_POST(self):
                if self.path != ROUTE:
                    self.send_error(404)
                    return
                event = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                event_id, fp = event["event_id"], hashlib.sha256(event["text"].encode()).hexdigest()
                with gw.lock:
                    gw.received.append(event_id)
                    record = gw.receipts.get(event_id)
                    if record is None and gw.running:
                        refusal = "canonical_turn_busy"
                    elif record is None:
                        refusal = None
                        gw.receipts[event_id] = {"fp": fp, "state": "pending", "text": None}
                        gw.running += 1
                        gw.max_running = max(gw.max_running, gw.running)
                        gw.runs.append(event_id)
                    elif record["fp"] != fp:
                        refusal = "canonical_event_conflict"
                    elif record["state"] == "pending":
                        refusal = "canonical_event_uncertain"
                    else:
                        refusal = ""
                if refusal:
                    self._send(event_id, 409, {"error": {"code": refusal, "message": "x"}})
                    return
                if refusal == "":  # answered from the receipt, never run again
                    self._send(event_id, 200, {"event_id": event_id, "text": record["text"]})
                    return
                gw.claim_of(event_id).set()
                time.sleep(gw.turn_secs)
                with gw.lock:
                    gw.receipts[event_id].update(state="terminal", text=gw.answer_for(event_id))
                    gw.running -= 1
                self._send(event_id, 200, {"event_id": event_id, "text": gw.answer_for(event_id)})

        return Handler

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"


class Agent:
    """One shim process as buzz-acp holds it: stdin/stdout NDJSON, its own process group."""

    def __init__(self, env, cwd, label):
        self.label = label
        self.stderr = open(cwd / f"{label}.stderr", "wb")
        self.proc = subprocess.Popen(
            [sys.executable, str(SHIM), "acp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stderr, env=env, cwd=cwd, start_new_session=True)
        self.lines = queue.Queue()
        self.frames = []
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.proc.stdout:
            try:
                frame = json.loads(raw)
            except ValueError:
                continue
            self.frames.append(frame)
            self.lines.put(frame)

    def send(self, payload):
        self.proc.stdin.write((json.dumps(payload) + "\n").encode())
        self.proc.stdin.flush()

    def wait_result(self, rpc_id, timeout):
        """The response to *rpc_id*, or None when none arrived within *timeout* seconds."""
        deadline = time.monotonic() + timeout
        while True:
            for frame in list(self.frames):
                if frame.get("id") == rpc_id and ("result" in frame or "error" in frame):
                    return frame
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                self.lines.get(timeout=min(left, 0.1))
            except queue.Empty:
                pass

    def responses_to(self, rpc_id):
        return [f for f in self.frames if f.get("id") == rpc_id and ("result" in f or "error" in f)]

    def start_session(self):
        self.send({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 2}})
        assert self.wait_result(0, 10) is not None
        self.send({"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": "/", "mcpServers": []}})
        new = self.wait_result(1, 10)
        assert new is not None
        return new["result"]["sessionId"]

    def prompt(self, rpc_id, session, blocks):
        self.send({"jsonrpc": "2.0", "id": rpc_id, "method": "session/prompt",
                   "params": bp.prompt_params(session, blocks)})

    def cancel(self, session):
        self.send({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session}})

    def kill(self):
        """SIGKILL by this exact PID (buzz-acp kills the group; the shim has no child mid-request)."""
        if self.proc.poll() is None:
            os.kill(self.proc.pid, signal.SIGKILL)
        self.proc.wait(10)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(25)
            except subprocess.TimeoutExpired:
                self.kill()
        self.stderr.close()


@pytest.fixture
def rig(tmp_path, request):
    gw = FakeGateway(getattr(request, "param", TURN_SECS))
    gw.thread.start()
    root = tmp_path / "bridge"
    root.mkdir()
    calls = root / "buzz-calls.jsonl"
    (root / "buzz").write_text(
        f"#!{sys.executable}\nimport json, sys\n"
        f"open({str(calls)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}) + '\\n')\n")
    (root / "buzz").chmod(0o755)
    (root / "required-mentions.json").write_text(json.dumps({"channels": {CHANNEL: {
        "ceo": {"handle": "ceo", "pubkey": OTHER}, "peer": {"handle": "cto-repo-factory", "pubkey": SIGNER}}}}))
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": "/usr/bin:/bin", "HOME": str(home), "TMPDIR": str(tmp_path),
        "ACP_RECEIPT_PORT_JS": str(tmp_path / "missing-receipt-port.js"),
        "CANONICAL_SURFACE_URL": gw.url, "API_SERVER_KEY": "d2-o2-placeholder", "CANONICAL_BINDING": "canonical",
        "CANONICAL_AUTHOR_ID": SIGNER, "CANONICAL_CHANNEL_ID": CHANNEL, "CANONICAL_FORWARD": "live",
        "CANONICAL_BUSY_RETRY_SECS": "1", "CANONICAL_BUSY_DEADLINE_SECS": "40",
        "CANONICAL_SHIM_LOG": str(root / "shim.log"), "CANONICAL_QUARANTINE_LOG": str(root / "q.jsonl"),
        "CANONICAL_CANARY_FILE": str(root / "canary.json"), "CANONICAL_PUBLISHED_DIR": str(root / "published"),
        "CANONICAL_MENTIONS_FILE": str(root / "required-mentions.json"), "CANONICAL_BUZZ_CLI": str(root / "buzz"),
    }
    import pwd  # the account database names the real home even when HOME is redirected

    live = Path(pwd.getpwuid(os.getuid()).pw_dir).joinpath(".hermes").resolve()
    for name, value in env.items():
        if name not in ("CANONICAL_SURFACE_URL", "PATH") and "/" in value and Path(value).resolve().is_relative_to(live):
            pytest.fail(f"{name} names the live Hermes home")
    agents = []

    def spawn(label):
        agent = Agent(env, root, label)
        agents.append(agent)
        return agent

    def posts():
        return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []

    def journal(event_id):
        path = root / "published" / event_id
        return json.loads(path.read_text()) if path.exists() and path.read_text() else None

    try:
        yield type("Rig", (), {"gw": gw, "spawn": staticmethod(spawn), "posts": staticmethod(posts),
                               "journal": staticmethod(journal), "root": root, "agents": agents})
    finally:
        for agent in agents:
            agent.close()
        gw.server.shutdown()
        gw.server.server_close()


def _reply_to(post):
    argv = post["argv"]
    return argv[argv.index("--reply-to") + 1] if "--reply-to" in argv else None


def _answer_posts(rig, event_id):
    return [p for p in rig.posts() if p["stdin"] == f"@cto-repo-factory {rig.gw.answer_for(event_id)}"]


def _wait(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.1)
    return predicate()


def _settled(rig, *events):
    return lambda: not rig.gw.running and all(
        rig.gw.receipts.get(e, {}).get("state") == "terminal" and rig.journal(e) is not None for e in events)


def _no_inflight(rig):
    return sorted(p.name for p in (rig.root / "published").glob("*.inflight"))


def _attributed(rig, event_id):
    """Reach, asserted: #91 r1's attribution guard took a prompt as this one event (the shim logs
    ``prompt author=… base=<id>…`` only past it). A prompt it refuses never reaches the O2 paths."""
    log = rig.root / "shim.log"
    return log.exists() and log.read_text().count(f" base={event_id[:12]}") or 0


def _running(rig, event_id):
    """Reach, asserted: the event's turn is claimed and still running at the gateway."""
    with rig.gw.lock:
        return rig.gw.receipts.get(event_id, {}).get("state") == "pending" and rig.gw.running == 1


def _steer(rig, agent, session, rpc_id, event_id):
    """buzz-acp's steer for a prompt in flight: ``session/cancel``, then CONTROL_CANCEL_GRACE for that
    prompt to end; with no end the agent is SIGKILLed and a new one spawned (CancelDrainTimeout). Reach,
    asserted: the event's turn was running when the cancel was sent. (target, session, response)."""
    assert _running(rig, event_id), "the turn was not running when the cancel was sent"
    agent.cancel(session)
    ended = agent.wait_result(rpc_id, CANCEL_GRACE)
    if ended is not None:
        return agent, session, ended
    agent.kill()
    target = rig.spawn(agent.label + "-respawned")
    return target, target.start_session(), None


def test_every_prompt_these_witnesses_send_is_valid_under_the_shims_own_rules():
    """Reach precondition, asserted against the shim under test (#91 r1-r3): each single-event prompt is
    attributed by ``envelope`` to its own event, and every prompt's [Context] is in exact form and bound to
    its last event's Tags (``notice_destination``). The merged prompt is attributed to no one."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("shim_d2_o2_fixture_check", str(SHIM))
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    for event_id, content in ((E1, ROOT_TEXT), (E3, "a later top-level message " + "y" * 200)):
        prompt = bp.prompt_params("s", channel_prompt(event_id, content))["prompt"]
        assert shim.envelope(prompt)[:3] == (SIGNER, CHANNEL, event_id)
        assert shim.notice_destination(shim.prompt_blocks(prompt)) == (CHANNEL, event_id, None)
    merged = bp.prompt_params("s", merged_steer_prompt(E1, ROOT_TEXT, E2, "@ceo thread question"))["prompt"]
    assert shim.envelope(merged) is None
    assert shim.notice_destination(shim.prompt_blocks(merged)) == (CHANNEL, E1, None)


# ── W1 (ported): a steer cancel while the turn outlives the 5 s drain ─────────────────────────────────
#
# The merged prompt buzz-acp sends after the cancel is refused unattributed by #1111 P1 / #91 r1 before
# any submission (it posts P1's notice to the [Context] thread). It reaches no O2 path and is evidence
# of nothing here; the O2 behaviour W1 reaches is the cancel of a running, attributed single-event turn.


def test_w1_an_answer_completed_after_a_steer_cancel_is_posted_once(rig):
    first = rig.spawn("shim-1")
    session = first.start_session()
    first.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert rig.gw.claim_of(E1).wait(10), "the shim never submitted the in-flight event"
    assert _attributed(rig, E1) == 1

    # The thread @mention arrives: steer. session/cancel, then the 5 s drain window.
    target, session, ended = _steer(rig, first, session, 2, E1)
    target.prompt(3, session, merged_steer_prompt(E1, ROOT_TEXT, E2, "@ceo thread question"))
    target.wait_result(3, 60)

    # The gateway finishes E1's turn whether or not anyone is still listening.
    _wait(_settled(rig, E1), TURN_SECS + 20)
    _wait(lambda: _answer_posts(rig, E1), 10)  # room for a surviving worker to post
    for agent in rig.agents:
        agent.close()

    assert rig.gw.receipts[E1]["state"] == "terminal", "the fake gateway did not finish the turn"
    assert rig.gw.runs.count(E1) == 1, "the turn ran more than once"
    assert rig.gw.received == [E1], "the merged prompt reached the gateway"  # refused first, by H91
    assert len(_answer_posts(rig, E1)) == 1, (
        f"E1's completed answer was posted {len(_answer_posts(rig, E1))} time(s); "
        f"shim replaced={ended is None}, gateway responses={rig.gw.responses}")


def test_a_steer_cancel_of_a_running_turn_is_answered_in_the_drain_and_never_called_cancelled(rig):
    agent = rig.spawn("shim")
    session = agent.start_session()
    agent.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert rig.gw.claim_of(E1).wait(10) and _attributed(rig, E1) == 1
    assert _running(rig, E1)
    agent.cancel(session)
    ended = agent.wait_result(2, CANCEL_GRACE)

    assert ended is not None, "no answer inside buzz-acp's drain: the shim would be SIGKILLed"
    assert _running(rig, E1), "reach: the cancel was answered while the turn still ran"
    assert ended["result"]["stopReason"] == "end_turn"
    assert ended["result"]["_meta"]["canonical"] == {
        "cancelled": False, "turn": "running", "event_id": E1,
        "outcome": "posted once to the thread by this shim: the answer or a failure notice"}
    # buzz-acp requeues the batch merged with the new event; H91 refuses that prompt, so E2 never runs.
    agent.prompt(3, session, merged_steer_prompt(E1, ROOT_TEXT, E2, "@ceo thread question"))
    assert agent.wait_result(3, 10)["result"] == {"stopReason": "end_turn"}
    _wait(_settled(rig, E1), TURN_SECS + 20)  # waits only: the assertions below name what failed
    agent.close()

    assert rig.gw.runs == [E1] and rig.gw.received == [E1] and rig.gw.max_running == 1
    assert len(agent.responses_to(2)) == 1  # one response per prompt, ever
    assert len(_answer_posts(rig, E1)) == 1 and rig.journal(E1)["kind"] == "answer"
    kinds = sorted(json.loads(p.read_text())["kind"] for p in (rig.root / "published").iterdir())
    assert kinds == ["answer", "unattributed_notice"]  # the answer, and H91's notice for the merged prompt
    assert _no_inflight(rig) == []


# ── the other boundaries ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rig", [2.0], indirect=True)
def test_control_a_normal_round_trip_is_answered_end_turn_and_posted_once(rig):
    agent = rig.spawn("shim")
    session = agent.start_session()
    agent.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert agent.wait_result(2, 30)["result"] == {"stopReason": "end_turn"}
    agent.close()

    assert _attributed(rig, E1) == 1 and rig.gw.received == [E1]
    assert rig.gw.runs == [E1] and len(agent.responses_to(2)) == 1
    assert [p["stdin"] for p in rig.posts()] == [f"@cto-repo-factory {rig.gw.answer_for(E1)}"]
    assert _reply_to(rig.posts()[0]) == E1 and _no_inflight(rig) == []


@pytest.mark.parametrize("rig", [1.0], indirect=True)
@pytest.mark.parametrize("offset", [-0.3, 0.0, 0.15, 0.6])
def test_a_cancel_racing_the_turns_end_gets_one_response_and_the_answer_one_post(rig, offset):
    agent = rig.spawn("shim")
    session = agent.start_session()
    agent.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert rig.gw.claim_of(E1).wait(10) and _attributed(rig, E1) == 1
    time.sleep(max(rig.gw.turn_secs + offset, 0))
    agent.cancel(session)
    assert agent.wait_result(2, CANCEL_GRACE) is not None
    _wait(_settled(rig, E1), 20)  # waits only: the assertions below name what failed
    agent.close()

    [response] = agent.responses_to(2)
    assert response["result"]["stopReason"] == "end_turn"  # it was submitted: never "cancelled"
    assert rig.gw.runs == [E1] and len(_answer_posts(rig, E1)) == 1 and len(rig.posts()) == 1
    assert _no_inflight(rig) == []


def test_a_shim_replaced_mid_turn_never_reruns_and_its_unposted_answer_ends_as_an_explicit_failure(rig):
    first = rig.spawn("shim-1")
    session = first.start_session()
    first.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert rig.gw.claim_of(E1).wait(10) and _attributed(rig, E1) == 1
    assert _running(rig, E1), "reach: the turn runs when its shim is replaced"
    first.kill()  # replaced with the turn in flight (a drain, idle or hard timeout, or an exit)

    second = rig.spawn("shim-2")
    session = second.start_session()
    second.prompt(3, session, merged_steer_prompt(E1, ROOT_TEXT, E2, "@ceo thread question"))
    second.wait_result(3, 30)
    _wait(_settled(rig, E1), TURN_SECS + 20)  # waits only: the assertions below name what failed
    second.close()

    assert rig.gw.receipts[E1]["state"] == "terminal", "the fake gateway did not finish the turn"
    assert rig.gw.runs == [E1] and rig.gw.received == [E1], "the turn ran, or was submitted, more than once"
    assert [r[:3] for r in rig.gw.responses] == [(E1, 200, None)]  # answered, to a client that is gone
    assert _answer_posts(rig, E1) == []  # its client is gone: not posted ...
    record = rig.journal(E1)  # ... and that ends as one explicit, journaled failure notice
    assert record is not None, "the unposted answer ended in silence"
    assert (record["kind"], record["state"], record["reply_to"]) == ("unfinished_notice", "sent", E1)
    [notice] = [p for p in rig.posts() if E1 in p["stdin"]]
    assert "알 수 없고" in notice["stdin"] and "다시 제출하지 않습니다" in notice["stdin"]
    assert rig.gw.answer_for(E1) not in notice["stdin"] and _no_inflight(rig) == []


# The next three follow buzz-acp's steer exactly (``_steer``): an unanswered cancel gets the agent
# replaced. With a 7 s turn the next prompt reaches whichever shim holds the session while E1 still runs.


@pytest.mark.parametrize("rig", [7.0], indirect=True)
def test_a_new_event_while_a_turn_runs_waits_its_turn_and_two_turns_never_share_the_session(rig):
    agent = rig.spawn("shim")
    session = agent.start_session()
    agent.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert rig.gw.claim_of(E1).wait(10) and _attributed(rig, E1) == 1
    target, session, _ = _steer(rig, agent, session, 2, E1)
    target.prompt(3, session, channel_prompt(E3, "a later top-level message " + "y" * 200))
    assert _running(rig, E1), "reach: the new event was sent while E1's turn ran"
    assert target.wait_result(3, 40)["result"] == {"stopReason": "end_turn"}
    _wait(_settled(rig, E1, E3), 20)  # waits only: the assertions below name what failed
    for each in rig.agents:
        each.close()

    assert _attributed(rig, E3) == 1
    assert rig.gw.runs == [E1, E3] and rig.gw.max_running == 1
    busy = [r for r in rig.gw.responses if r[2] == "canonical_turn_busy"]
    assert busy == [], f"E3 was put to the shared session while E1 ran: {busy}"
    assert len(_answer_posts(rig, E1)) == 1 and len(_answer_posts(rig, E3)) == 1


@pytest.mark.parametrize("rig", [7.0], indirect=True)
def test_the_same_event_dispatched_again_while_in_flight_is_never_submitted_again(rig):
    agent = rig.spawn("shim")
    session = agent.start_session()
    agent.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert rig.gw.claim_of(E1).wait(10) and _attributed(rig, E1) == 1
    target, session, _ = _steer(rig, agent, session, 2, E1)
    target.prompt(3, session, channel_prompt(E1, ROOT_TEXT))
    assert _running(rig, E1), "reach: the same event was dispatched again while its turn ran"
    assert target.wait_result(3, 30) is not None
    _wait(_settled(rig, E1), 20)  # waits only: the assertions below name what failed
    for each in rig.agents:
        each.close()

    assert _attributed(rig, E1) == 2  # both dispatches passed the attribution guard
    assert rig.gw.received == [E1], f"the event was submitted again: {rig.gw.responses}"
    assert rig.gw.runs == [E1] and len(_answer_posts(rig, E1)) == 1 and len(rig.posts()) == 1


@pytest.mark.parametrize("rig", [7.0], indirect=True)
def test_a_prompt_cancelled_before_it_was_submitted_is_cancelled_and_never_submitted(rig):
    agent = rig.spawn("shim")
    session = agent.start_session()
    agent.prompt(2, session, channel_prompt(E1, ROOT_TEXT))
    assert rig.gw.claim_of(E1).wait(10) and _attributed(rig, E1) == 1
    target, session, _ = _steer(rig, agent, session, 2, E1)
    target.prompt(3, session, channel_prompt(E3, "a later top-level message " + "y" * 200))
    assert _wait(lambda: _attributed(rig, E3) == 1, 5), "reach: E3 never passed the attribution guard"
    assert _running(rig, E1), "reach: E3's cancel is sent while E1's turn holds the session"
    target.cancel(session)
    assert target.wait_result(3, CANCEL_GRACE)["result"] == {"stopReason": "cancelled"}
    _wait(_settled(rig, E1), 20)  # waits only: the assertions below name what failed
    time.sleep(1.5)  # room for a wrongly resumed E3 submission
    for each in rig.agents:
        each.close()

    assert E3 not in rig.gw.received and rig.gw.runs == [E1] and len(target.responses_to(3)) == 1
    assert len(_answer_posts(rig, E1)) == 1 and rig.journal(E3) is None
