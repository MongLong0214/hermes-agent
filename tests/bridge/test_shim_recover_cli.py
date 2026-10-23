"""`acp-canonical-shim.py recover`: operator recovery of refused or missed Buzz events by event ID."""
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from test_shim_batch_delivery import PrivateKey, _crypto, signed
from test_shim_envelope_attribution import CH, _SHIM, _stand_in_wrapper, _calls


def _serve(events, delivered, queried):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/query":
                [event_id] = body[0]["ids"]
                queried.append(event_id)
                assert self.headers["Authorization"].startswith("Nostr ")
                result = [e for e in events if e["id"] == event_id]
            else:
                assert self.path == "/v1/canonical-surface/events"
                delivered.append(body)
                result = {"text": "CEO answer"}
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    return server, thread


def _run(tmp_path, url, *args):
    env = {**_stand_in_wrapper(tmp_path), "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes"),
           "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1", "CANONICAL_SURFACE_URL": url,
           "CANONICAL_FORWARD": "live", "BUZZ_RELAY_URL": url,
           "BUZZ_PRIVATE_KEY": PrivateKey.from_int(77).secret.hex(), "API_SERVER_KEY": "test-only",
           "PYTHONPATH": str(Path(_crypto.__file__).resolve().parent.parent)}
    return subprocess.run([sys.executable, "-X", "dev", str(_SHIM), "recover", *args],
                          text=True, capture_output=True, env=env, cwd=tmp_path, timeout=60)


def test_recover_delivers_each_signed_original_once_and_rerun_skips_answered(tmp_path):
    events = [signed(1), signed(2)]
    delivered, queried = [], []
    server, thread = _serve(events, delivered, queried)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first = _run(tmp_path, url, CH, events[0]["id"], events[1]["id"])
        second = _run(tmp_path, url, CH, events[0]["id"], events[1]["id"])
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
    assert first.returncode == 0, first.stderr
    assert [(d["event_id"], d["author_id"], d["channel_id"]) for d in delivered] == [
        (e["id"], e["pubkey"], CH) for e in events], first.stderr
    assert first.stdout.split("\n")[:2] == [f"{e['id']} answered" for e in events], first.stdout
    assert len(_calls(tmp_path)) == 2  # one answer posted per event
    # A rerun finds both answered: no second gateway turn and no second post.
    assert second.returncode == 0, second.stderr
    assert len(delivered) == 2 and len(_calls(tmp_path)) == 2
    assert second.stdout.split("\n")[:2] == [f"{e['id']} answered" for e in events]


def test_recover_reports_an_unavailable_original_without_touching_the_others(tmp_path):
    present = signed(3)
    absent = signed(4)
    delivered, queried = [], []
    server, thread = _serve([present], delivered, queried)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        done = _run(tmp_path, url, CH, absent["id"], present["id"])
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
    assert done.returncode == 1
    lines = done.stdout.strip().split("\n")
    assert lines[0].startswith(f"{absent['id']} unavailable: ")
    assert lines[1] == f"{present['id']} answered"
    assert [d["event_id"] for d in delivered] == [present["id"]]


@pytest.mark.parametrize("args", [[], [CH], ["not-a-uuid", "0" * 64], [CH, "xyz"]])
def test_recover_refuses_malformed_arguments_before_any_request(tmp_path, args):
    done = _run(tmp_path, "http://127.0.0.1:9", *args)
    assert done.returncode == 2 and done.stderr
    assert not (tmp_path / "published").exists() or not any((tmp_path / "published").iterdir())


def test_explicit_recovery_publishes_one_answer_after_a_legacy_notice(tmp_path):
    """#98 r2 H98-03: a notice journaled before records carried "journal" holds the event's only
    slot. Automatic replays keep that guard; an explicit operator recovery may post one answer."""
    event = signed(5)
    published = tmp_path / "published"
    published.mkdir()
    (published / event["id"]).write_text(json.dumps({
        "sent": True, "state": "sent", "delivery": "posted", "kind": "notice", "event_id": event["id"],
        "ref": None, "channel": CH, "reply_to": event["id"], "reason": "exit 0: ",
        "at": "2026-10-20T19:01:45+0900"}) + "\n")
    delivered, queried = [], []
    server, thread = _serve([event], delivered, queried)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        first = _run(tmp_path, url, CH, event["id"])
        second = _run(tmp_path, url, CH, event["id"])
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
    assert first.returncode == 0, first.stderr
    assert [d["event_id"] for d in delivered] == [event["id"]]
    assert len(_calls(tmp_path)) == 1
    assert json.loads((published / (event["id"] + ".answer")).read_text())["kind"] == "answer"
    # The legacy notice is kept as it was; a rerun finds the answer and posts nothing more.
    assert "journal" not in json.loads((published / event["id"]).read_text())
    assert second.returncode == 0 and len(delivered) == 1 and len(_calls(tmp_path)) == 1


@pytest.mark.parametrize("shape", ["single", "batch", "steer"])
def test_abandoned_explicit_recovery_keeps_its_terminal_guard_for_a_later_live_replay(tmp_path, shape):
    """#100 r2 H100-R2-01: recovery opens the legacy event's second slot and abandons its submission,
    so the reaper persists an unfinished notice there. A later live shim, which runs without the
    explicit-recovery flag, must still read that slot and never submit the event again."""
    from test_shim_batch_delivery import batch, bp, ev
    from test_shim_envelope_attribution import NAME
    event = signed(6)
    base = event["id"]
    published = tmp_path / "published"
    published.mkdir()
    (published / base).write_text(json.dumps({"sent": True, "state": "sent", "kind": "notice",
                                              "event_id": base, "channel": CH}))
    delivered = []

    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/query":
                result = [event]
            else:
                delivered.append(body)
                # A malformed answer makes the recovering handler abandon its submission.
                result = {"text": ["bad-type"]} if len(delivered) == 1 else {"text": "later answer"}
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        recovered = _run(tmp_path, url, CH, base)
        assert recovered.returncode == 1, recovered.stderr
        assert json.loads((published / (base + ".answer")).read_text())["kind"] == "unfinished_notice"
        assert not (published / (base + ".inflight")).exists()
        blocks = {"single": lambda: bp.format_prompt(CH, NAME, [ev(event)], {}),
                  "batch": lambda: batch([event]),
                  "steer": lambda: batch([signed(7)], cancelled=[event], framing=bp.STEER)}[shape]()
        env = {**_stand_in_wrapper(tmp_path), "HOME": str(tmp_path), "HERMES_HOME": str(tmp_path / "hermes"),
               "PATH": "/usr/bin:/bin", "CANONICAL_SURFACE_URL": url, "CANONICAL_FORWARD": "live",
               "BUZZ_RELAY_URL": url, "BUZZ_PRIVATE_KEY": PrivateKey.from_int(77).secret.hex(),
               "API_SERVER_KEY": "test-only", "PYTHONPATH": str(Path(_crypto.__file__).resolve().parent.parent)}
        message = {"jsonrpc": "2.0", "id": 9, "method": "session/prompt", "params": bp.prompt_params("s", blocks)}
        live = subprocess.run([sys.executable, str(_SHIM), "acp"], input=json.dumps(message) + "\n",
                              text=True, capture_output=True, env=env, cwd=tmp_path, timeout=60)
        assert live.returncode == 0, live.stderr
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
    assert [d["event_id"] for d in delivered].count(base) == 1, delivered
