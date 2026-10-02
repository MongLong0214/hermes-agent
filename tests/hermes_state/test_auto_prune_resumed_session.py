"""Automatic prune must not finish deleting a session that became live while it was deleting it.

Batched automatic prune deletes an oversized expired session over several write transactions, letting
a reply's transcript write in between. The eligibility filters (ended, inactive past retention) and the
live-write guards (turn lease, compression lock) used to be checked once, in their own transaction,
before the first delete: a turn that took the session's lease, reopened it and appended a message in
the gap after that check lost the session row and every message to the remaining sub-batches. The
pre-batching prune checked and deleted in one transaction, so that gap did not exist.

Real store, real SQLite, real lease/reopen/append: the live turn runs between two of the prune's own
write transactions (the first one the prune commits), which is exactly the gap a reply-path write takes.
"""

from __future__ import annotations

import time

import pytest

import hermes_state_maintenance

_BATCH_ROWS = 15
_MESSAGES = 40


def _mark_old_and_ended(db, session_id: str, old: float) -> None:
    db._execute_write(lambda conn: (
        conn.execute("UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'done', "
                     "last_activity_at = ? WHERE id = ?", (old, old + 1, old, session_id)),
        conn.execute("UPDATE messages SET timestamp = ? WHERE session_id = ?", (old, session_id))))


def _take_lease(db):
    assert db.try_acquire_session_turn_lease("huge", "live-turn")


def _reopen(db):
    db.reopen_session("huge")


def _append(db):
    db.append_message("huge", "user", "fresh turn after resume")


def _resume_turn(db):
    _take_lease(db)
    _reopen(db)
    db.append_message("huge", "user", "fresh turn after resume", turn_lease_holder="live-turn")


@pytest.mark.parametrize("live_turn", [_resume_turn, _take_lease, _reopen, _append],
                         ids=["lease+reopen+append", "lease", "reopen", "append"])
def test_a_session_that_goes_live_mid_prune_keeps_its_row_and_new_messages(tmp_path, monkeypatch, live_turn):
    from hermes_state import SessionDB

    monkeypatch.setattr(hermes_state_maintenance, "_AUTO_PRUNE_BATCH_ROWS", _BATCH_ROWS)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("huge", source="cli")
        db.append_messages_batch("huge", [{"role": "user", "content": f"m{j}"} for j in range(_MESSAGES)])
        _mark_old_and_ended(db, "huge", time.time() - 400 * 86400)

        state = {"pruning": False, "fired": False}
        real_write, real_prune = db._execute_write, db.prune_sessions

        def _write(fn, *args, **kwargs):
            result = real_write(fn, *args, **kwargs)
            if state["pruning"] and not state["fired"]:
                # Between the prune's first committed write transaction and its next one: the gap a
                # reply-path write takes. The live turn's own writes come back through here, unarmed.
                state["fired"] = True
                live_turn(db)
            return result

        def _prune(*args, **kwargs):
            state["pruning"] = True
            try:
                return real_prune(*args, **kwargs)
            finally:
                state["pruning"] = False

        monkeypatch.setattr(db, "_execute_write", _write)
        monkeypatch.setattr(db, "prune_sessions", _prune)

        result = db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0, vacuum=False)

        assert state["fired"], "the prune never committed a write transaction"
        assert result.get("pruned") == 0, result
        session = db.get_session("huge")
        assert session is not None, "the prune deleted a session that went live after its eligibility check"
        contents = [m["content"] for m in db.get_messages("huge")]
        assert contents, "the prune deleted every message of a session that went live mid-prune"
        if live_turn in (_resume_turn, _append):
            assert contents[-1] == "fresh turn after resume", contents[-3:]
        if live_turn is _resume_turn:
            # (A reopen with no new activity is closed again by the stale-open sweep that runs right
            # after the prune — non-destructively, so only the row and its messages are asserted there.)
            assert session["ended_at"] is None
        if live_turn in (_resume_turn, _take_lease):
            # The live holder still owns its lease: the prune neither reclaimed nor fenced it.
            assert db.refresh_session_turn_lease("huge", "live-turn")
    finally:
        db.close()
