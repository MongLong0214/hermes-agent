"""Causal terminal-receipt coverage through the registered TUI handlers."""

from __future__ import annotations

import hashlib
import threading

import pytest

from hermes_state import SessionDB
from tui_gateway.turn_receipts import TurnReceiptAdapter, request_binding


@pytest.fixture()
def receipt_gateway(tmp_path, monkeypatch):
    """A real SessionDB behind the production handler with narrow edge fakes."""
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    session_key = "receipt-session"
    session_id = "live-session"
    db.create_session(session_key, source="tui")
    image = tmp_path / "attached.png"
    image.write_bytes(b"first attachment bytes")
    events: list[object] = []
    effects = {
        "voice_stop": 0,
        "interrupted": 0,
        "active_slot": 0,
        "busy_queue": 0,
        "inflight": 0,
        "bootstrap": 0,
        "agent_build": 0,
        "agent_run": 0,
    }
    session = {
        "session_key": session_key,
        "history": [{"role": "user", "content": "old", "_row_id": 71}],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": True,
        "attached_images": [str(image)],
        "agent": object(),
        "transport": None,
    }
    server._sessions[session_id] = session
    server._db = db

    monkeypatch.setattr(server, "_emit", lambda *args: events.append(args))
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: True)
    monkeypatch.setattr(
        server,
        "_tts_stream_stop",
        lambda **_kw: effects.__setitem__("voice_stop", effects["voice_stop"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_ensure_active_session_slot",
        lambda *_args: effects.__setitem__(
            "active_slot", effects["active_slot"] + 1
        ) or "must not claim",
    )
    monkeypatch.setattr(
        server,
        "_handle_busy_submit",
        lambda *_args, **_kw: effects.__setitem__(
            "busy_queue", effects["busy_queue"] + 1
        ) or {"result": {"status": "queued"}},
    )
    monkeypatch.setattr(
        server,
        "_start_inflight_turn",
        lambda *_args: effects.__setitem__("inflight", effects["inflight"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_ensure_session_db_row",
        lambda *_args: effects.__setitem__("bootstrap", effects["bootstrap"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_persist_branch_seed",
        lambda *_args: effects.__setitem__("bootstrap", effects["bootstrap"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_start_agent_build",
        lambda *_args: effects.__setitem__("agent_build", effects["agent_build"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kw: effects.__setitem__("agent_run", effects["agent_run"] + 1),
    )

    def expand(text, _task_id):
        events.append(("expand", text))
        return f"expanded::{text}"

    monkeypatch.setattr(server, "_expand_skill_invocation_for_replay", expand)
    monkeypatch.setattr(
        "hermes_cli.input_sanitize.sanitize_user_prompt_text",
        lambda value: f"sanitized::{value.strip()}",
    )
    monkeypatch.setattr("tools.voice_mode.is_voice_stop_phrase", lambda _text: True)
    monkeypatch.setattr(
        "hermes_cli.voice.stop_continuous",
        lambda: effects.__setitem__("voice_stop", effects["voice_stop"] + 1),
    )
    monkeypatch.setattr(
        "tools.tts_streaming.mark_speech_interrupted",
        lambda: effects.__setitem__("interrupted", effects["interrupted"] + 1),
    )

    yield server, db, session_id, session_key, session, image, events, effects

    server._sessions.pop(session_id, None)
    if server._db is db:
        server._db = None
    db.close()


def _params(*, text="/replay original", turn_request_id="request-1"):
    return {
        "session_id": "live-session",
        "text": text,
        "turn_request_id": turn_request_id,
        # This is deliberately attacker-controlled noise, not receipt authority.
        "binding_digest": "sha256:" + "0" * 64,
        "interrupted": True,
        "queued": True,
        "truncate_before_row_id": 71,
        "truncate_before_user_ordinal": 0,
        "confirm_truncate": True,
        "confirm_empty_truncate": True,
    }


def _effective_request(
    session_key, image, *, text="/replay original", turn_request_id="request-1"
):
    return request_binding(
        session_id=session_key,
        turn_request_id=turn_request_id,
        text=f"expanded::sanitized::{text.strip()}",
        display_kind=None,
        attachments=[str(image)],
        truncation={
            "kind": "row_id",
            "row_id": 71,
            "user_ordinal": 0,
            "confirm_truncate": True,
            "confirm_empty_truncate": True,
        },
    )


def test_completed_duplicate_replays_before_voice_interrupt_busy_or_turn_effects(
    receipt_gateway,
):
    server, db, session_id, session_key, session, image, events, effects = receipt_gateway
    request = _effective_request(session_key, image)
    adapter = TurnReceiptAdapter(db)
    adapter.prepare_or_replay(request)
    claimed = adapter.claim_after_lease(request)
    assistant_bytes = "stored assistant 😀 bytes"
    digest = "sha256:" + hashlib.sha256(assistant_bytes.encode()).hexdigest()
    adapter.finish(
        session_key,
        request.turn_request_id,
        request.binding_digest,
        claimed.claim_token,
        assistant_content=assistant_bytes,
        response_digest=digest,
    )

    result = server._methods["prompt.submit"]("rid", _params())

    replay = result["result"]["turn_receipt"]
    assert result["result"]["status"] == "complete"
    assert result["result"]["replayed"] is True
    assert replay["assistantContent"] == assistant_bytes
    assert replay["responseDigest"] == digest
    assert replay == TurnReceiptAdapter(db).completed_replay(request)
    assert effects == {key: 0 for key in effects}
    assert session["running"] is True
    assert session["history"] == [{"role": "user", "content": "old", "_row_id": 71}]
    assert [row["content"] for row in db.get_messages(session_key)] == [assistant_bytes]
    assert events == [
        ("expand", "sanitized::/replay original"),
        ("message.complete", session_id, {
            "text": assistant_bytes,
            "status": "complete",
            "turn_receipt": replay,
            "replayed": True,
        }),
    ]


@pytest.mark.parametrize(
    ("text", "rewrite_attachment"),
    [("/replay altered", False), ("/replay original", True)],
)
def test_conflicting_effective_input_fails_before_all_turn_effects(
    receipt_gateway, text, rewrite_attachment
):
    server, db, _session_id, session_key, session, image, events, effects = receipt_gateway
    request = _effective_request(session_key, image)
    TurnReceiptAdapter(db).prepare_or_replay(request)
    if rewrite_attachment:
        image.write_bytes(b"changed attachment bytes")

    result = server._methods["prompt.submit"]("rid", _params(text=text))

    assert result["error"] == {
        "code": 4091,
        "message": "turn_receipt_binding_conflict",
    }
    assert effects == {key: 0 for key in effects}
    assert session["history"] == [{"role": "user", "content": "old", "_row_id": 71}]
    assert db.get_messages(session_key) == []
    assert events == [("expand", f"sanitized::{text.strip()}")]


def test_prepare_status_and_submit_share_server_effective_input(
    receipt_gateway, monkeypatch
):
    server, db, _session_id, session_key, _session, image, _events, effects = receipt_gateway
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(
        server,
        "_ensure_active_session_slot",
        lambda *_args: effects.__setitem__(
            "active_slot", effects["active_slot"] + 1
        ) or None,
    )
    expected = _effective_request(session_key, image)
    params = _params()

    prepared = server._methods["turn.prepare"]("prepare", params)
    status = server._methods["turn.status"]("status", params)
    submitted = server._methods["prompt.submit"]("submit", params)

    assert prepared["result"]["turn_receipt"]["status"] == "PREPARED"
    assert status["result"]["turn_receipt"] == prepared["result"]["turn_receipt"]
    assert submitted["result"] == {"status": "queued"}
    assert TurnReceiptAdapter(db).status_for(expected) == prepared["result"]["turn_receipt"]
    assert effects["active_slot"] == 1
    assert effects["busy_queue"] == 1
