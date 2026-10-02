"""``_flush_agent_transcript_at_shutdown`` must snapshot on a False return too (R67-3).

``_flush_messages_to_session_db`` can fail WITHOUT raising — its documented contract
(agent/session_persistence.py) is to return ``False`` on a flush it could not land (e.g. a
``_db_flush_failed`` retry budget exhausted). The shutdown helper only handled the ``raise``
case: it caught exceptions and wrote the live transcript to the on-disk recovery snapshot, but a
plain ``False`` return fell straight through with no snapshot at all. Its caller
(``_ensure_persisted_then_release_soft`` / ``_release_evicted_agent_soft``, gateway/run_agent_cache.py)
then unconditionally clears ``agent._session_messages`` right after, so a False-returning writer
produced zero durable copies of the transcript — not in SQLite, not on disk.
"""

from pathlib import Path
from types import SimpleNamespace

from gateway.run_shutdown import GatewayShutdownMixin


def test_false_flush_return_writes_a_recovery_snapshot(tmp_path: Path, monkeypatch):
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)

    messages = [{"role": "user", "content": "the operator's only copy of this turn"}]
    agent = SimpleNamespace(
        _flush_messages_to_session_db=lambda msgs: False,  # fails WITHOUT raising
        _session_messages=messages,
        session_id="live-session",
    )

    GatewayShutdownMixin._flush_agent_transcript_at_shutdown(agent)

    files = list(flush_dir.glob("*.json"))
    assert len(files) == 1, "a False (non-raising) flush failure must still produce a recovery snapshot"

    import json
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert payload["session_id"] == "live-session"
    assert payload["messages"] == messages


def test_true_flush_return_writes_no_snapshot(tmp_path: Path, monkeypatch):
    """Control: a successful flush must not spuriously spool."""
    flush_dir = tmp_path / "pending_messages"
    flush_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("gateway.shutdown_flush._get_flush_dir", lambda: flush_dir)

    agent = SimpleNamespace(
        _flush_messages_to_session_db=lambda msgs: True,
        _session_messages=[{"role": "user", "content": "ok"}],
        session_id="live-session",
    )

    GatewayShutdownMixin._flush_agent_transcript_at_shutdown(agent)

    assert list(flush_dir.glob("*.json")) == []
