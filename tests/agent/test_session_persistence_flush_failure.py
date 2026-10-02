"""agent/session_persistence.py: a flush failure must not be reported as persisted (#L4-2).

``_flush_messages_to_session_db`` already fails closed and returns ``False`` on a corrupt/
quarantined state.db (see ``test_flush_diverts_on_corrupt_state_db.py``, which proves that return
value directly). ``_persist_session`` ignored that return value and called ``note_turn_persisted``
unconditionally, so a completed, delivered turn whose only durable write failed was
indistinguishable from one that actually persisted. Downstream, the standard runtime's
``agent_persisted`` default (``gateway/run_turn_runner.py``) relies on the agent having recorded
whether its own flush landed; a silent swallow there meant the gateway's own transcript append
was skipped too, and the reply went out with no durable copy of the turn anywhere.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hermes_state import SessionDB, StateDbCorruptError
from run_agent import AIAgent


def _persist_agent(db, session_id):
    """Same minimal stand-in as test_flush_diverts_on_corrupt_state_db.py, extended with the real
    bound ``_persist_session`` (and its helper) instead of stopping at ``_flush_messages_to_session_db``."""
    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        _persist_disabled=False,
        session_id=session_id,
        _session_persist_lock=None,
        _flushed_db_message_ids=set(),
        _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0,
        _db_flush_scan_prefix=None,
        _persist_user_message_idx=None,
        _persist_user_message_override=None,
        _persist_user_message_timestamp=None,
        _pending_cli_user_message=None,
        _active_session_turn_lease_holder=None,
        _last_persistence_error_cause=None,
        _compression_adoption_failed=False,
    )
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = (
        AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    )
    agent._flush_messages_to_session_db_unlocked = (
        AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    )
    agent._drop_trailing_empty_response_scaffolding = (
        AIAgent._drop_trailing_empty_response_scaffolding.__get__(agent, AIAgent)
    )
    agent._persist_session = AIAgent._persist_session.__get__(agent, AIAgent)
    return agent


def test_persist_session_marks_failure_instead_of_reporting_success(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("live", source="cli")
        agent = _persist_agent(db, "live")

        def _quarantined(self, *, session_id, messages, **kwargs):
            raise StateDbCorruptError("database disk image is malformed (quarantined)")

        monkeypatch.setattr(SessionDB, "append_messages_batch", _quarantined)

        messages = [{"role": "user", "content": "the operator's only copy of this turn"}]
        agent._persist_session(messages, [])

        assert agent._last_persist_succeeded is False
    finally:
        db.close()


def test_persist_session_marks_success_on_a_clean_write(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("live", source="cli")
        agent = _persist_agent(db, "live")

        agent._persist_session([{"role": "user", "content": "hi"}], [])

        assert agent._last_persist_succeeded is True
    finally:
        db.close()
