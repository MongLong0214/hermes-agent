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
def test_a_session_that_goes_live_mid_prune_is_never_resumable_with_a_shortened_history(
        tmp_path, monkeypatch, live_turn):
    """PR73 R2-1 (round 3): re-checking eligibility per sub-batch protected the REMAINING rows, but the
    rows earlier sub-batches had already committed stayed deleted, so a turn that reopened/appended in
    the gap resumed a silently shortened history (the reviewer saw 26 of 41 messages survive while
    message_count still said 41). The prune now claims the session in the same transaction as its first
    delete; from then on reopen and every transcript write refuse it. Whatever happens in the gap,
    there is never a session that is both part-pruned and resumable/writable."""
    from hermes_state import SessionDB
    from hermes_state_errors import SessionBeingPrunedError

    monkeypatch.setattr(hermes_state_maintenance, "_AUTO_PRUNE_BATCH_ROWS", _BATCH_ROWS)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("huge", source="cli")
        db.append_messages_batch("huge", [{"role": "user", "content": f"m{j}"} for j in range(_MESSAGES)])
        _mark_old_and_ended(db, "huge", time.time() - 400 * 86400)

        state = {"pruning": False, "fired": False, "refused": False}
        real_write, real_prune = db._execute_write, db.prune_sessions

        def _write(fn, *args, **kwargs):
            result = real_write(fn, *args, **kwargs)
            if state["pruning"] and not state["fired"]:
                # Between the prune's first committed write transaction and its next one: the gap a
                # reply-path write takes. The live turn's own writes come back through here, unarmed.
                state["fired"] = True
                try:
                    live_turn(db)
                except SessionBeingPrunedError:
                    state["refused"] = True
            return result

        def _prune(*args, **kwargs):
            state["pruning"] = True
            try:
                return real_prune(*args, **kwargs)
            finally:
                state["pruning"] = False

        monkeypatch.setattr(db, "_execute_write", _write)
        monkeypatch.setattr(db, "prune_sessions", _prune)

        db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0, vacuum=False)
        monkeypatch.setattr(db, "_execute_write", real_write)

        assert state["fired"], "the prune never committed a write transaction"
        if live_turn in (_resume_turn, _reopen, _append):
            assert state["refused"], "a write to a part-pruned session was accepted"
        session = db.get_session("huge")
        if session is None:
            # The prune finished: nothing part-pruned is left behind anywhere.
            assert db.get_messages("huge") == []
        else:
            # The prune stopped (a live lease): what remains is part-pruned and must stay unusable.
            assert len(db.get_messages("huge")) < _MESSAGES
            with pytest.raises(SessionBeingPrunedError):
                db.reopen_session("huge")
            with pytest.raises(SessionBeingPrunedError):
                db.append_message("huge", "user", "late write")
        assert all(m.get("content") != "fresh turn after resume" for m in db.get_messages("huge"))
    finally:
        db.close()


def test_a_prune_stopped_by_a_live_lease_is_finished_by_the_next_pass(tmp_path, monkeypatch):
    """A claimed session the prune had to stop on (a turn held its lease) is finished, claim and all,
    once the lease is gone -- it does not stay a permanently unusable fragment."""
    from hermes_state import SessionDB
    from hermes_state_errors import PRUNE_CLAIM_KEY_PREFIX

    monkeypatch.setattr(hermes_state_maintenance, "_AUTO_PRUNE_BATCH_ROWS", _BATCH_ROWS)
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("huge", source="cli")
        db.append_messages_batch("huge", [{"role": "user", "content": f"m{j}"} for j in range(_MESSAGES)])
        _mark_old_and_ended(db, "huge", time.time() - 400 * 86400)
        real_write = db._execute_write
        armed = {"prune": False, "done": False}
        real_prune = db.prune_sessions

        def _write(fn, *args, **kwargs):
            result = real_write(fn, *args, **kwargs)
            if armed["prune"] and not armed["done"]:
                armed["done"] = True
                _take_lease(db)
            return result

        def _prune(*args, **kwargs):
            armed["prune"] = True
            try:
                return real_prune(*args, **kwargs)
            finally:
                armed["prune"] = False

        monkeypatch.setattr(db, "_execute_write", _write)
        monkeypatch.setattr(db, "prune_sessions", _prune)
        db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0, vacuum=False)
        monkeypatch.setattr(db, "_execute_write", real_write)
        monkeypatch.setattr(db, "prune_sessions", real_prune)
        assert db.get_session("huge") is not None, "the live lease should have stopped this pass"

        db.release_session_turn_lease("huge", "live-turn")
        db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=0, vacuum=False)
        assert db.get_session("huge") is None
        assert db.get_messages("huge") == []
        with db._read_ctx() as conn:
            assert conn.execute("SELECT 1 FROM state_meta WHERE key = ?",
                                (PRUNE_CLAIM_KEY_PREFIX + "huge",)).fetchone() is None
    finally:
        db.close()
