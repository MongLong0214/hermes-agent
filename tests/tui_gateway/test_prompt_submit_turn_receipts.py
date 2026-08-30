"""Causal terminal-receipt coverage through the registered TUI handlers."""

from __future__ import annotations

import hashlib
import threading
import types

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
        "drain": 0,
        "compute": 0,
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
        "attached_images": [],
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
        "_drain_queued_prompt",
        lambda *_args, **_kw: effects.__setitem__("drain", effects["drain"] + 1),
    )
    monkeypatch.setattr(
        server,
        "_submit_prompt_to_compute_host",
        lambda *_args, **_kw: effects.__setitem__(
            "compute", effects["compute"] + 1
        ) or {"result": {"status": "streaming"}},
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


def _effective_request(session_key, *, text="/replay original", turn_request_id="request-1"):
    return request_binding(
        session_id=session_key,
        turn_request_id=turn_request_id,
        text=f"sanitized::{text.strip()}",
        display_kind=None,
        attachments=[],
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
    server, db, session_id, session_key, session, _image, events, effects = receipt_gateway
    request = _effective_request(session_key)
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

    replay = adapter.completed_replay(request)
    disposition = {"turnRequestId": "request-1", "status": "COMPLETED"}
    assert result["result"]["status"] == "complete"
    assert result["result"]["replayed"] is True
    assert result["result"]["turn_receipt"] == disposition
    assert replay["assistantContent"] == assistant_bytes
    assert replay["responseDigest"] == digest
    assert replay == TurnReceiptAdapter(db).completed_replay(request)
    assert effects == {key: 0 for key in effects}
    assert session["running"] is True
    assert session["history"] == [{"role": "user", "content": "old", "_row_id": 71}]
    assert [row["content"] for row in db.get_messages(session_key)] == [assistant_bytes]
    assert events == [
        ("message.complete", session_id, {
            "text": assistant_bytes,
            "status": "complete",
            "turn_receipt": disposition,
            "replayed": True,
        }),
    ]


def test_conflicting_effective_input_fails_before_all_turn_effects(receipt_gateway):
    text = "/replay altered"
    server, db, _session_id, session_key, session, _image, events, effects = receipt_gateway
    request = _effective_request(session_key)
    TurnReceiptAdapter(db).prepare_or_replay(request)

    result = server._methods["prompt.submit"]("rid", _params(text=text))

    assert result["error"] == {
        "code": 4091,
        "message": "turn_receipt_binding_conflict",
    }
    assert effects == {key: 0 for key in effects}
    assert session["history"] == [{"role": "user", "content": "old", "_row_id": 71}]
    assert db.get_messages(session_key) == []
    assert events == []


def test_receipt_v1_runtime_sends_only_admitted_sanitized_text(
    tmp_path, monkeypatch
):
    """Receipt turns must not consume or prepend any runtime enrichment."""
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    sid, session_key = "receipt-raw-live", "receipt-raw-session"
    db.create_session(session_key, source="tui")
    seen: dict[str, object] = {}
    calls = {"context": 0, "speech": 0, "reactions": 0, "hud": 0}

    def run_conversation(message, **kwargs):
        seen["message"] = message
        seen["persisted"] = kwargs["persist_user_message"]
        return {"final_response": "done"}

    session = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "agent": types.SimpleNamespace(
            session_id=session_key,
            api_mode="chat_completions",
            run_conversation=run_conversation,
            clear_interrupt=lambda: None,
        ),
        "transport": None,
        "cwd": str(tmp_path),
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(server, "_pending_reaction_notes", lambda _session: calls.__setitem__("reactions", calls["reactions"] + 1) or "[reaction]")
    monkeypatch.setattr(server, "_hud_surface_note", lambda _session: calls.__setitem__("hud", calls["hud"] + 1) or "[hud]")
    monkeypatch.setattr(
        "hermes_cli.input_sanitize.sanitize_user_prompt_text",
        lambda value: f"sanitized::{value}",
    )
    monkeypatch.setattr(
        "agent.context_references.preprocess_context_references",
        lambda *_args, **_kwargs: calls.__setitem__("context", calls["context"] + 1),
    )
    monkeypatch.setattr(
        "tools.tts_streaming.take_speech_interrupted",
        lambda: calls.__setitem__("speech", calls["speech"] + 1) or True,
    )
    try:
        result = server._methods["prompt.submit"](
            "rid",
            {
                "session_id": sid,
                "text": "raw @context",
                "surface": "hud",
                "turn_request_id": "raw-runtime-request",
            },
        )

        assert result["result"]["status"] == "streaming"
        assert seen == {
            "message": "sanitized::raw @context",
            "persisted": "sanitized::raw @context",
        }
        assert calls == {"context": 0, "speech": 0, "reactions": 0, "hud": 0}
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_receipt_admission_keeps_late_attachment_for_the_next_ordinary_turn(
    tmp_path, monkeypatch
):
    """A post-admission attachment cannot enter or be consumed by receipt v1."""
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    sid, session_key = "receipt-race-live", "receipt-race-session"
    db.create_session(session_key, source="tui")
    image = tmp_path / "late.png"
    image.write_bytes(b"late attachment")
    runs: list[tuple[object, object]] = []
    image_messages: list[tuple[str, list[str]]] = []

    def run_conversation(message, **kwargs):
        runs.append((message, kwargs["persist_user_message"]))
        return {"final_response": "done"}

    session = {
        "session_key": session_key,
        "history": [],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "agent": types.SimpleNamespace(
            session_id=session_key,
            api_mode="chat_completions",
            run_conversation=run_conversation,
            clear_interrupt=lambda: None,
        ),
        "transport": None,
        "cwd": str(tmp_path),
    }
    server._sessions[sid] = session
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_emit", lambda *_args: None)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_apply_pending_model_switch", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_bot_capabilities", lambda *_args: None)
    monkeypatch.setattr(server, "_session_cwd", lambda _session: str(tmp_path))
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_tts_stream_begin", lambda: None)
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_load_interim_assistant_messages", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.input_sanitize.sanitize_user_prompt_text",
        lambda value: f"sanitized::{value}",
    )
    monkeypatch.setattr(
        server,
        "_build_image_ref_message",
        lambda prompt, paths: image_messages.append((prompt, list(paths)))
        or f"{prompt} [image={paths[0]}]",
    )
    barrier = {"attach": True}

    def wait_until_admitted(*_args):
        if barrier["attach"]:
            barrier["attach"] = False
            session["attached_images"].append(str(image))
        return None

    monkeypatch.setattr(server, "_wait_agent_for_prompt", wait_until_admitted)
    try:
        receipt = server._methods["prompt.submit"](
            "receipt",
            {
                "session_id": sid,
                "text": "receipt text",
                "turn_request_id": "late-attachment-request",
            },
        )

        assert receipt["result"]["status"] == "streaming"
        assert runs == [("sanitized::receipt text", "sanitized::receipt text")]
        assert image_messages == []
        assert session["attached_images"] == [str(image)]

        ordinary = server._methods["prompt.submit"](
            "ordinary", {"session_id": sid, "text": "ordinary text"}
        )

        assert ordinary["result"]["status"] == "streaming"
        assert image_messages == [("sanitized::ordinary text", [str(image)])]
        assert runs[-1][0] == f"sanitized::ordinary text [image={image}]"
        assert str(image) in runs[-1][1]
        assert session["attached_images"] == []
    finally:
        server._sessions.pop(sid, None)
        db.close()


def test_receipt_rpc_boundaries_expose_only_the_safe_disposition(receipt_gateway):
    """Admission, status, in-progress, and replay cannot expose row metadata."""
    server, db, session_id, session_key, session, _image, events, effects = receipt_gateway
    request = _effective_request(session_key)
    expected = {"turnRequestId": "request-1", "status": "PREPARED"}

    prepared = server._methods["turn.prepare"]("prepare", _params())
    status = server._methods["turn.status"]("status", _params())
    in_progress = server._methods["prompt.submit"]("submit", _params())

    assert prepared["result"]["turn_receipt"] == expected
    assert status["result"]["turn_receipt"] == expected
    assert in_progress["result"] == {"status": "in_progress", "turn_receipt": expected}
    assert TurnReceiptAdapter(db).status_for(request)["sessionId"] == session_key
    assert effects == {key: 0 for key in effects}

    claimed = TurnReceiptAdapter(db).claim_after_lease(request)
    assistant_text = "stored assistant reply"
    TurnReceiptAdapter(db).finish(
        session_key,
        request.turn_request_id,
        request.binding_digest,
        claimed.claim_token,
        assistant_content=assistant_text,
        response_digest="sha256:" + "1" * 64,
    )
    replay = server._methods["prompt.submit"]("replay", _params())
    completed = {"turnRequestId": "request-1", "status": "COMPLETED"}

    assert replay["result"] == {
        "status": "complete",
        "turn_receipt": completed,
        "replayed": True,
    }
    assert events[-1] == (
        "message.complete",
        session_id,
        {
            "text": assistant_text,
            "status": "complete",
            "turn_receipt": completed,
            "replayed": True,
        },
    )
    assert set(events[-1][2]["turn_receipt"]) == {"turnRequestId", "status"}
    assert session["history"] == [{"role": "user", "content": "old", "_row_id": 71}]


@pytest.mark.parametrize("handler", ["prompt.submit", "turn.prepare", "turn.status"])
@pytest.mark.parametrize("request_id", [7, True, None, "", " \t\n"])
def test_receipt_handlers_reject_non_string_or_blank_request_ids(
    receipt_gateway, handler, request_id
):
    """Receipt opt-in has one strict request-ID type contract everywhere."""
    server, db, _session_id, session_key, session, image, _events, effects = receipt_gateway
    session["attached_images"] = [str(image)]

    result = server._methods[handler](
        "rid", _params(turn_request_id=request_id)
    )

    assert result["error"] == {
        "code": 4004,
        "message": "turn_request_id must be a non-empty string",
    }
    assert effects == {key: 0 for key in effects}
    assert db.get_messages(session_key) == []
    assert db._conn.execute("SELECT COUNT(*) FROM turn_receipts").fetchone()[0] == 0
    assert session["history"] == [{"role": "user", "content": "old", "_row_id": 71}]


@pytest.mark.parametrize("handler", ["prompt.submit", "turn.prepare", "turn.status"])
@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("truncate_before_row_id", True),
        ("truncate_before_row_id", 71.0),
        ("truncate_before_row_id", "71"),
        ("truncate_before_user_ordinal", True),
        ("truncate_before_user_ordinal", 0.0),
        ("truncate_before_user_ordinal", "0"),
        ("truncate_before_message_id", True),
        ("truncate_before_message_id", 71),
        ("truncate_before_message_id", 71.0),
        ("truncate_before_message_id", ""),
    ],
)
def test_receipt_handlers_reject_malformed_truncation_identifiers_before_mutation(
    receipt_gateway, handler, param, value
):
    """Receipt v1 accepts only JSON integer ids and nonempty string message ids."""
    server, db, _session_id, session_key, session, _image, events, effects = receipt_gateway
    params = _params()
    params.pop("truncate_before_row_id")
    params.pop("truncate_before_user_ordinal")
    params[param] = value

    result = server._methods[handler]("rid", params)

    assert result["error"] == {
        "code": 4004,
        "message": f"{param} has an invalid receipt identifier",
    }
    assert effects == {key: 0 for key in effects}
    assert events == []
    assert db.get_messages(session_key) == []
    assert db._conn.execute("SELECT COUNT(*) FROM turn_receipts").fetchone()[0] == 0
    assert session["history"] == [{"role": "user", "content": "old", "_row_id": 71}]


def test_prompt_submit_without_request_id_keeps_the_ordinary_path(
    receipt_gateway, monkeypatch
):
    """Omitting receipt opt-in remains an ordinary prompt submission."""
    server, db, _session_id, session_key, session, image, _events, effects = receipt_gateway
    session["running"] = False
    session["attached_images"] = [str(image)]
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)

    params = _params()
    params.pop("turn_request_id")
    result = server._methods["prompt.submit"]("rid", params)

    assert result["result"]["status"] == "streaming"
    assert effects["agent_run"] == 1
    assert db._conn.execute("SELECT COUNT(*) FROM turn_receipts").fetchone()[0] == 0
    assert db.get_messages(session_key) == []


@pytest.mark.parametrize("handler", ["prompt.submit", "turn.prepare"])
def test_fresh_session_binding_conflict_does_not_bootstrap_the_new_session(
    tmp_path, monkeypatch, handler
):
    """A global request-ID conflict wins before a fresh session can write."""
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    session_a, session_b = "receipt-session-a", "receipt-session-b"
    request_id = "already-bound-request"
    db.create_session(session_a, source="tui")
    request_a = request_binding(
        session_id=session_a,
        turn_request_id=request_id,
        text="accepted in A",
        display_kind=None,
        attachments=[],
        truncation=None,
    )
    TurnReceiptAdapter(db).prepare_or_replay(request_a)
    session_b_state = {
        "session_key": session_b,
        "history": [{"role": "user", "content": "old", "_row_id": 71}],
        "history_lock": threading.RLock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "agent": object(),
        "transport": None,
        "source": "tui",
        "cwd": str(tmp_path),
    }
    session_b_id = "fresh-session-b"
    server._sessions[session_b_id] = session_b_state
    monkeypatch.setattr(server, "_db", db)
    try:
        result = server._methods[handler](
            "rid",
            {
                "session_id": session_b_id,
                "text": "different input for B",
                "turn_request_id": request_id,
            },
        )

        assert result["error"] == {
            "code": 4091,
            "message": "turn_receipt_binding_conflict",
        }
        assert db.get_session(session_b) is None
        assert db.get_messages(session_b) == []
        assert db._conn.execute(
            "SELECT COUNT(*) FROM turn_receipts WHERE session_id = ?", (session_b,)
        ).fetchone()[0] == 0
        assert session_b_state["history"] == [
            {"role": "user", "content": "old", "_row_id": 71}
        ]
    finally:
        server._sessions.pop(session_b_id, None)
        db.close()


@pytest.mark.parametrize("handler", ["prompt.submit", "turn.prepare"])
def test_receipt_attachments_are_rejected_before_receipt_or_file_effects(
    receipt_gateway, monkeypatch, handler
):
    """Receipt v1 admits text-only turns, never mutable attachment paths."""
    import tui_gateway.turn_receipts as turn_receipts

    server, db, _session_id, session_key, session, image, _events, effects = receipt_gateway
    session["running"] = False
    session["attached_images"] = [str(image)]
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_args: None)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    reads = {"count": 0}

    def observed_attachment_digest(_path):
        reads["count"] += 1
        return "sha256:should-not-be-read"

    monkeypatch.setattr(
        turn_receipts, "_attachment_content_digest", observed_attachment_digest
    )

    result = server._methods[handler]("rid", _params())

    assert result["error"] == {
        "code": 4004,
        "message": "turn_receipt_attachments_unsupported",
    }
    assert reads == {"count": 0}
    assert effects == {key: 0 for key in effects}
    assert db.get_messages(session_key) == []
    assert db._conn.execute("SELECT COUNT(*) FROM turn_receipts").fetchone()[0] == 0
    assert session["attached_images"]


def test_prepared_receipt_while_busy_returns_in_progress_without_dispatch(
    receipt_gateway, monkeypatch
):
    server, db, _session_id, session_key, _session, _image, _events, effects = receipt_gateway
    monkeypatch.setattr(
        server,
        "_ensure_active_session_slot",
        lambda *_args: effects.__setitem__(
            "active_slot", effects["active_slot"] + 1
        ) or None,
    )
    expected = _effective_request(session_key)
    params = _params()

    prepared = server._methods["turn.prepare"]("prepare", params)
    status = server._methods["turn.status"]("status", params)
    submitted = server._methods["prompt.submit"]("submit", params)

    assert prepared["result"]["turn_receipt"]["status"] == "PREPARED"
    assert status["result"]["turn_receipt"] == prepared["result"]["turn_receipt"]
    assert submitted["result"] == {
        "status": "in_progress",
        "turn_receipt": prepared["result"]["turn_receipt"],
    }
    assert TurnReceiptAdapter(db).status_for(expected)["status"] == "PREPARED"
    assert effects == {key: 0 for key in effects}


def test_receipt_submit_rejects_compute_host_before_any_dispatch(
    receipt_gateway, monkeypatch
):
    server, db, _session_id, session_key, session, _image, _events, effects = receipt_gateway
    session["running"] = False
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: True)
    expected = _effective_request(session_key)
    params = _params()

    prepared = server._methods["turn.prepare"]("prepare", params)
    submitted = server._methods["prompt.submit"]("submit", params)

    assert prepared["result"]["turn_receipt"]["status"] == "PREPARED"
    assert submitted["error"] == {
        "code": 4009,
        "message": "turn receipts are unavailable with compute-host isolation",
    }
    assert TurnReceiptAdapter(db).status_for(expected)["status"] == "PREPARED"
    assert effects == {key: 0 for key in effects}


class _InlineThread:
    def __init__(self, *, target, daemon):
        self._target = target

    def start(self):
        self._target()


def test_fresh_receipt_handlers_establish_session_identity_before_receipt_fk(
    tmp_path, monkeypatch
):
    """Both registered admission paths work before an ordinary first turn."""
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    prepared_id, prepared_key = "fresh-prepare", "fresh-prepare-key"
    submitted_id, submitted_key = "fresh-submit", "fresh-submit-key"

    def fresh_session(key):
        return {
            "session_key": key,
            "history": [{"role": "user", "content": "old", "_row_id": 71}],
            "history_lock": threading.RLock(),
            "history_version": 0,
            "running": False,
            "attached_images": [],
            "agent": object(),
            "transport": None,
            "source": "tui",
            "cwd": str(tmp_path),
        }

    server._sessions[prepared_id] = fresh_session(prepared_key)
    server._sessions[submitted_id] = fresh_session(submitted_key)
    monkeypatch.setattr(server, "_db", db)
    monkeypatch.setattr(server, "_voice_mode_enabled", lambda: False)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    monkeypatch.setattr(server, "_session_uses_compute_host", lambda *_args: False)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_args: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args, **_kw: None)
    monkeypatch.setattr(server.threading, "Thread", _InlineThread)
    try:
        prepared = server._methods["turn.prepare"](
            "prepare", {**_params(), "session_id": prepared_id}
        )
        submitted = server._methods["prompt.submit"](
            "submit",
            {
                **_params(turn_request_id="fresh-submit-request"),
                "session_id": submitted_id,
            },
        )

        assert prepared["result"]["turn_receipt"]["status"] == "PREPARED"
        assert submitted["result"]["status"] == "streaming"
        assert db.get_session(prepared_key) is not None
        assert db.get_session(submitted_key) is not None
        assert db._conn.execute(
            "SELECT COUNT(*) FROM turn_receipts WHERE session_id IN (?, ?)",
            (prepared_key, submitted_key),
        ).fetchone()[0] == 2
    finally:
        server._sessions.pop(prepared_id, None)
        server._sessions.pop(submitted_id, None)
        db.close()
