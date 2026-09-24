"""``session/prompt`` with ``_meta.hermes`` fails closed until a terminal-receipt store exists.

The E2E tests drive a real ``hermes acp`` process the way the external controller does: its five
env variables (plus pytest's venv guard), ``initialize`` (id 1), then one ``session/prompt`` (id 2) with no session/new or
session/load first, then stdin closed. A loopback OpenAI-compatible endpoint counts every request the
process makes. The parser tests call the closed-shape parser directly.
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from hermes_state import SessionDB

REPO_ROOT = Path(__file__).resolve().parents[2]
_REFUSED = {"stopReason": "refusal", "_meta": {"hermes": {"acpTerminalReceipt": {"status": "REFUSED"}}}}
_ACP_SESSION = "acp-target"
_TELEGRAM_SESSION = "telegram-target"
_PROMPT_TEXT = "run the release checklist"
# Runs the real entry point and, at exit, reports which checkout it imported and whether the agent
# runtime (the module ``SessionManager._make_agent`` imports to build an AIAgent) was ever loaded.
_PROBE = (
    "import atexit, json, runpy, sys\n"
    "atexit.register(lambda: sys.stderr.write('\\nPROBE=' + json.dumps({"
    "'acp_adapter': getattr(sys.modules.get('acp_adapter'), '__file__', None), "
    "'agent_runtime_loaded': 'run_agent' in sys.modules})))\n"
    "sys.argv = ['hermes', 'acp']\n"
    "runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)\n"
)


def _canonical_digest(value) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _target_bind_receipt(session_id: str) -> dict:
    public = {
        "domain": "hermes.target-bind",
        "version": 1,
        "actor_id": "actor:controller",
        "binding_generation": 3,
        "executor_runtime_identity": "runtime:controller",
        "requested_session_id": session_id,
        "lineage_root_digest": "sha256:" + "1" * 64,
    }
    return {**public, "receipt_digest": _canonical_digest(public)}


def _receipt_identity(text: str) -> dict:
    return {
        "schema": "hermes.acp-terminal-receipt-identity",
        "version": 1,
        "turnRequestId": "turn-1",
        "targetActorId": "actor:controller",
        "promptDigest": _canonical_digest(text),
        "bindingGeneration": 3,
        "targetBindingId": "binding-1",
        "targetAttestationId": "attestation-1",
        "executorSessionId": "executor-1",
        "executorSessionIncarnation": "incarnation-1",
    }


def _terminal_receipt(operation: str, session_id: str, text: str = _PROMPT_TEXT) -> dict:
    return {
        "operation": operation,
        "receiptIdentity": _receipt_identity(text),
        "targetBindReceipt": _target_bind_receipt(session_id),
    }


def _prompt_frame(session_id: str, prompt: list, meta=None) -> bytes:
    params = {"sessionId": session_id, "prompt": prompt}
    if meta is not None:
        params["_meta"] = meta
    return json.dumps({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": params}).encode()


_TEXT_PROMPT = [{"type": "text", "text": _PROMPT_TEXT}]


class _CountingProvider:
    """Loopback OpenAI-compatible endpoint that answers every call and records it."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        provider = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _reply(self, payload: dict, content_type: str = "application/json", raw: bytes | None = None):
                body = raw if raw is not None else json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                provider.requests.append("GET " + self.path)
                self._reply({"object": "list", "data": [{"id": "fixture-model", "object": "model"}]})

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
                provider.requests.append("POST " + self.path)
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                completion = {
                    "id": "c", "object": "chat.completion", "created": 1, "model": "fixture-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "done"},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                if not body.get("stream"):
                    self._reply(completion)
                    return
                completion["object"] = "chat.completion.chunk"
                completion["choices"][0]["delta"] = completion["choices"][0].pop("message")
                self._reply({}, "text/event-stream", ("data: " + json.dumps(completion) + "\n\ndata: [DONE]\n\n").encode())

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base_url = f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def acp_home(tmp_path):
    provider = _CountingProvider()
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"model:\n  provider: custom\n  base_url: {provider.base_url}\n  default: fixture-model\n"
        "  api_mode: chat_completions\nmemory:\n  memory_enabled: false\n  user_profile_enabled: false\n",
        encoding="utf-8",
    )
    db = SessionDB(home / "state.db")
    try:
        db.create_session(session_id=_ACP_SESSION, source="acp", model="fixture-model",
                          model_config={"cwd": str(tmp_path)}, cwd=str(tmp_path))
        db.create_session(session_id=_TELEGRAM_SESSION, source="telegram", model="fixture-model")
        for session_id in (_ACP_SESSION, _TELEGRAM_SESSION):
            db.append_message(session_id, "user", "earlier question")
            db.append_message(session_id, "assistant", "earlier answer")
    finally:
        db.close()
    yield home, provider
    provider.close()


def _message_rows(home: Path) -> list[tuple]:
    conn = sqlite3.connect(home / "state.db")
    try:
        return conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
    finally:
        conn.close()


def _acp_exchange(home: Path, prompt_frame: bytes):
    """initialize → session/prompt → close stdin, exactly as the controller sequences it."""
    env = {
        "HOME": str(home),
        "HERMES_HOME": str(home),
        "HERMES_PROFILE": "default",
        "HERMES_ACP_SKIP_ENV_LOAD": "1",
        "HERMES_ACP_SKIP_CONFIGURED_MCP": "1",
        # Not a client var: keeps the child's interrupted-update recovery from reinstalling into the suite's venv.
        "PYTEST_CURRENT_TEST": os.environ["PYTEST_CURRENT_TEST"],
    }
    initialize ={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": 1, "clientCapabilities": {}, "clientInfo": {"name": "controller", "version": "1"}}}
    proc = subprocess.Popen(
        [sys.executable, "-c", _PROBE], cwd=REPO_ROOT, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    watchdog = threading.Timer(90, proc.kill)
    watchdog.start()
    stderr_chunks: list[bytes] = []
    drain = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
    drain.start()
    try:
        proc.stdin.write(json.dumps(initialize).encode() + b"\n")
        proc.stdin.flush()
        lines = []
        for line in iter(proc.stdout.readline, b""):
            lines.append(line)
            frame_id = json.loads(line).get("id")
            if frame_id == 1:
                proc.stdin.write(prompt_frame + b"\n")
                proc.stdin.flush()
            elif frame_id == 2:
                proc.stdin.close()
        returncode = proc.wait(timeout=60)
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    drain.join(timeout=10)
    stderr = b"".join(stderr_chunks).decode("utf-8", "replace")
    probe = json.loads(stderr.rsplit("PROBE=", 1)[1]) if "PROBE=" in stderr else {}
    return returncode, [json.loads(line) for line in lines], probe, stderr


def _with_raw_edit(frame: bytes, old: bytes, new: bytes) -> bytes:
    assert frame.count(old) == 1, (old, frame)
    return frame.replace(old, new)


_TERMINAL_PROMPTS = {
    "execute-on-acp-session": _prompt_frame(
        _ACP_SESSION, _TEXT_PROMPT, {"hermes": {"acpTerminalReceipt": _terminal_receipt("execute", _ACP_SESSION)}}),
    "status-on-telegram-session": _prompt_frame(
        _TELEGRAM_SESSION, [], {"hermes": {"acpTerminalReceipt": _terminal_receipt("status", _TELEGRAM_SESSION)}}),
    "extra-key": _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, {"hermes": {"acpTerminalReceipt": {
        **_terminal_receipt("execute", _ACP_SESSION), "force": True}}}),
    "operation-abort": _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, {"hermes": {"acpTerminalReceipt": {
        **_terminal_receipt("execute", _ACP_SESSION), "operation": "abort"}}}),
    "receipt-not-an-object": _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, {"hermes": {"acpTerminalReceipt": ["execute"]}}),
    "hermes-not-an-object": _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, {"hermes": "execute"}),
    "hermes-null": _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, {"hermes": None}),
    # The transport keeps the last of duplicate keys; whichever copy survives, the prompt still carries
    # ``_meta.hermes`` and is refused.
    "duplicate-operation": _with_raw_edit(
        _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, {"hermes": {"acpTerminalReceipt": _terminal_receipt("execute", _ACP_SESSION)}}),
        b'"operation": "execute"', b'"operation": "status", "operation": "execute"'),
    "duplicate-hermes": _with_raw_edit(
        _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, {"hermes": {"acpTerminalReceipt": _terminal_receipt("execute", _ACP_SESSION)}}),
        b'"_meta": {', b'"_meta": {"hermes": 1, '),
    "unknown-session": _prompt_frame("no-such-session", _TEXT_PROMPT, {"hermes": {
        "acpTerminalReceipt": _terminal_receipt("execute", "no-such-session")}}),
}


@pytest.mark.parametrize("prompt_frame", list(_TERMINAL_PROMPTS.values()), ids=list(_TERMINAL_PROMPTS))
def test_terminal_receipt_prompt_is_refused_before_any_turn(acp_home, prompt_frame):
    home, provider = acp_home
    messages_before = _message_rows(home)

    returncode, frames, probe, stderr = _acp_exchange(home, prompt_frame)

    observed = {
        "returncode": returncode,
        # Exactly the two responses: no server->client request, no notification, nothing after id 2.
        "frame_ids": [frame.get("id") for frame in frames],
        "prompt_result": frames[-1].get("result") if frames else None,
        "provider_requests": provider.requests,
        "messages_unchanged": _message_rows(home) == messages_before,
        "agent_runtime_loaded": probe.get("agent_runtime_loaded"),
        "this_checkout": Path(probe.get("acp_adapter") or "/").resolve().is_relative_to(REPO_ROOT),
    }
    assert observed == {
        "returncode": 0,
        "frame_ids": [1, 2],
        "prompt_result": _REFUSED,
        "provider_requests": [],
        "messages_unchanged": True,
        "agent_runtime_loaded": False,
        "this_checkout": True,
    }, stderr[-2000:]


@pytest.mark.parametrize("meta", [None, {"clientTrace": "trace-1"}], ids=["no-meta", "other-meta-key"])
def test_prompt_without_hermes_meta_still_runs_the_turn(acp_home, meta):
    home, provider = acp_home
    messages_before = _message_rows(home)

    returncode, frames, probe, stderr = _acp_exchange(home, _prompt_frame(_ACP_SESSION, _TEXT_PROMPT, meta))

    assert returncode == 0, stderr[-2000:]
    assert frames[-1]["id"] == 2 and frames[-1]["result"]["stopReason"] == "end_turn", frames
    assert any(request.startswith("POST ") for request in provider.requests), provider.requests
    assert len(_message_rows(home)) > len(messages_before)
    assert probe["agent_runtime_loaded"] is True, probe
    assert Path(probe["acp_adapter"]).resolve().is_relative_to(REPO_ROOT), probe


def test_closed_shape_is_parsed_into_the_internal_target_form():
    from acp_adapter.terminal_receipts import parse_terminal_receipt_request

    request = parse_terminal_receipt_request(
        {"acpTerminalReceipt": _terminal_receipt("status", _TELEGRAM_SESSION)}, _TELEGRAM_SESSION)

    assert request is not None
    assert request.operation == "status"
    assert request.receipt_identity == _receipt_identity(_PROMPT_TEXT)
    assert request.target_bind_receipt == {
        "schema": "hermes.target-bind-receipt", **_target_bind_receipt(_TELEGRAM_SESSION)}


def _mutated(path: tuple[str, ...], value=..., *, drop=False) -> dict:
    """The well-formed ``_meta.hermes`` with one nested key replaced, added, or dropped."""
    hermes = {"acpTerminalReceipt": _terminal_receipt("execute", _ACP_SESSION)}
    node = hermes
    for key in path[:-1]:
        node = node[key]
    if drop:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return hermes


_MALFORMED = {
    "extra-top-key": _mutated(("trace",), "x"),
    "extra-receipt-key": _mutated(("acpTerminalReceipt", "force"), True),
    "missing-identity": _mutated(("acpTerminalReceipt", "receiptIdentity"), drop=True),
    "operation-abort": _mutated(("acpTerminalReceipt", "operation"), "abort"),
    "operation-not-text": _mutated(("acpTerminalReceipt", "operation"), ["execute"]),
    "identity-not-object": _mutated(("acpTerminalReceipt", "receiptIdentity"), "turn-1"),
    "target-extra-key": _mutated(("acpTerminalReceipt", "targetBindReceipt", "schema"), "hermes.target-bind-receipt"),
    "target-missing-key": _mutated(("acpTerminalReceipt", "targetBindReceipt", "receipt_digest"), drop=True),
    "target-version-true": _mutated(("acpTerminalReceipt", "targetBindReceipt", "version"), True),
    "target-generation-true": _mutated(("acpTerminalReceipt", "targetBindReceipt", "binding_generation"), True),
    "target-generation-zero": _mutated(("acpTerminalReceipt", "targetBindReceipt", "binding_generation"), 0),
    "target-generation-unsafe": _mutated(("acpTerminalReceipt", "targetBindReceipt", "binding_generation"), 2**53),
    "target-other-session": _mutated(("acpTerminalReceipt", "targetBindReceipt", "requested_session_id"), "other"),
    "target-other-domain": _mutated(("acpTerminalReceipt", "targetBindReceipt", "domain"), "hermes.other"),
    "target-uppercase-digest": _mutated(
        ("acpTerminalReceipt", "targetBindReceipt", "lineage_root_digest"), "sha256:" + "A" * 64),
    "target-empty-actor": _mutated(("acpTerminalReceipt", "targetBindReceipt", "actor_id"), ""),
}


@pytest.mark.parametrize("hermes_metadata", list(_MALFORMED.values()), ids=list(_MALFORMED))
def test_anything_but_the_closed_shape_does_not_parse(hermes_metadata):
    from acp_adapter.terminal_receipts import parse_terminal_receipt_request

    assert parse_terminal_receipt_request(hermes_metadata, _ACP_SESSION) is None
