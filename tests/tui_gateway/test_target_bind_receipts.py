"""Target bind receipts through the registered authenticated gateway RPC."""

from __future__ import annotations

import hashlib
import json

import pytest

from hermes_state import SessionDB


def test_target_bind_persists_the_gateway_resolved_lineage_root(tmp_path, monkeypatch):
    """The public receipt commits the actor binding without exposing session internals."""
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    db.create_session("lineage-root", source="tui")
    db.create_session(
        "lineage-tip", source="tui", parent_session_id="lineage-root"
    )
    monkeypatch.setattr(server, "_db", db)
    try:
        response = server._methods["target.bind"](
            "bind-1",
            {
                "session_id": "lineage-tip",
                "actor_id": "acp-actor-7",
                "binding_generation": 4,
                "executor_runtime_identity": "executor-runtime-9",
            },
        )

        receipt = response["result"]["target_bind_receipt"]
        assert set(receipt) == {"domain", "version", "digest"}
        assert receipt["domain"] == "hermes.target-bind"
        assert receipt["version"] == 1
        assert receipt["digest"].startswith("sha256:")
        assert "lineage" not in json.dumps(response)

        records = db.list_meta_prefix("target_bind_receipt:")
        assert len(records) == 1
        stored = json.loads(records[0][1])
        assert stored["actor_id"] == "acp-actor-7"
        assert stored["binding_generation"] == 4
        assert stored["executor_runtime_identity"] == "executor-runtime-9"
        assert stored["requested_session_id"] == "lineage-tip"
        assert stored["lineage_root_id"] == "lineage-root"
        assert stored["digest"] == receipt["digest"]
        canonical = json.dumps(
            {key: value for key, value in stored.items() if key != "digest"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert stored["digest"] == "sha256:" + hashlib.sha256(canonical).hexdigest()
    finally:
        db.close()


def test_target_bind_replays_the_stored_receipt_after_restart(tmp_path, monkeypatch):
    from tui_gateway import server

    path = tmp_path / "state.db"
    params = {
        "session_id": "bound-session",
        "actor_id": "acp-actor-7",
        "binding_generation": 4,
        "executor_runtime_identity": "executor-runtime-9",
    }
    db = SessionDB(path)
    db.create_session("bound-session", source="tui")
    monkeypatch.setattr(server, "_db", db)
    try:
        first = server._methods["target.bind"]("bind-1", params)
    finally:
        db.close()

    reopened = SessionDB(path)
    monkeypatch.setattr(server, "_db", reopened)
    try:
        replay = server._methods["target.bind"]("bind-2", params)

        assert replay["result"]["target_bind_receipt"] == first["result"][
            "target_bind_receipt"
        ]
        assert len(reopened.list_meta_prefix("target_bind_receipt:")) == 1
    finally:
        reopened.close()


def test_target_bind_refuses_a_second_session_for_the_same_binding_identity(
    tmp_path, monkeypatch
):
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    db.create_session("first-session", source="tui")
    db.create_session("second-session", source="tui")
    monkeypatch.setattr(server, "_db", db)
    try:
        first = {
            "session_id": "first-session",
            "actor_id": "acp-actor-7",
            "binding_generation": 4,
            "executor_runtime_identity": "executor-runtime-9",
        }
        assert "result" in server._methods["target.bind"]("bind-1", first)

        collision = server._methods["target.bind"](
            "bind-2", {**first, "session_id": "second-session"}
        )

        assert collision["error"] == {
            "code": 4091,
            "message": "target_bind_receipt_conflict",
        }
        assert len(db.list_meta_prefix("target_bind_receipt:")) == 1
    finally:
        db.close()


@pytest.mark.parametrize(
    "params",
    [
        {
            "session_id": "missing-session",
            "actor_id": "acp-actor-7",
            "binding_generation": 4,
            "executor_runtime_identity": "executor-runtime-9",
        },
        {
            "session_id": "valid-session",
            "actor_id": "",
            "binding_generation": 4,
            "executor_runtime_identity": "executor-runtime-9",
        },
        {
            "session_id": "valid-session",
            "actor_id": "acp-actor-7",
            "binding_generation": True,
            "executor_runtime_identity": "executor-runtime-9",
        },
        {
            "session_id": "valid-session",
            "actor_id": "acp-actor-7",
            "binding_generation": 4,
            "executor_runtime_identity": " ",
        },
        {
            "session_id": "valid-session",
            "actor_id": "acp-actor-7",
            "binding_generation": 4,
            "executor_runtime_identity": "executor-runtime-9",
            "lineage_root_id": "caller-supplied",
        },
        {
            "session_id": "cyclic-session",
            "actor_id": "acp-actor-7",
            "binding_generation": 4,
            "executor_runtime_identity": "executor-runtime-9",
        },
    ],
)
def test_target_bind_rejects_invalid_or_ambiguous_input_without_a_receipt(
    tmp_path, monkeypatch, params
):
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    db.create_session("valid-session", source="tui")
    db.create_session("cyclic-session", source="tui")
    db._conn.execute(
        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
        ("cyclic-session", "cyclic-session"),
    )
    monkeypatch.setattr(server, "_db", db)
    try:
        response = server._methods["target.bind"]("bind-invalid", params)

        assert response["error"] == {
            "code": 4004,
            "message": "target_bind_receipt_invalid",
        }
        assert db.list_meta_prefix("target_bind_receipt:") == []
    finally:
        db.close()
