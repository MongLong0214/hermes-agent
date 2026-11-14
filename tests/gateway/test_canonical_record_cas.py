"""A canonical turn whose answer exists but whose terminal record does not land.

The coordinator claims the event's receipt row, runs the turn once, and records the terminal with a
compare-and-set on the pending row it wrote, after re-proving the head, the actor, the DB file and the
run generation. When a newer generation or another holder's row wins, the turn already produced its
answer: it is kept beside the receipt (or in the local fallback beside the state.db), never delivered,
never written over the winner, and the turn never runs again.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from gateway import canonical_answer_fallback
from gateway.canonical_surface import CanonicalReceiptCoordinator, CanonicalReceiptResult
from tests.gateway.test_canonical_receipt_coordinator import event, gateway_under_test, submit

RECEIPT, CLAIM, KEPT = "canonical-receipt:v2:", "canonical-claim:v1:", "canonical-answer-kept:v1:"


@pytest.fixture
def gw(tmp_path, monkeypatch):
    gateway = gateway_under_test(tmp_path, monkeypatch)
    try:
        yield gateway
    finally:
        gateway.store.close_all_db_handles()


def rows(gw, namespace):
    with gw.db._lock:
        found = gw.db._conn.execute("SELECT key, value FROM state_meta WHERE key LIKE ?",
                                    (namespace + "%",)).fetchall()
    assert len(found) <= 1
    return (found[0][0], json.loads(found[0][1])) if found else (None, None)


def write_row(gw, key, value):
    with gw.db._lock:
        gw.db._conn.execute("UPDATE state_meta SET value = ? WHERE key = ?", (value, key))


def delete_row(gw, key):
    with gw.db._lock:
        gw.db._conn.execute("DELETE FROM state_meta WHERE key = ?", (key,))


def restart(gw):
    """Nothing cached: a replay reads the state.db through a read-only handle, as after a restart."""
    with gw.runner._agent_cache_lock:
        gw.runner._agent_cache.clear()


def kept_unrecorded(result, *, stage, observed):
    assert result.status == "unrecorded" and result.terminal_text is None
    assert result.failure["code"] == "canonical_answer_unrecorded"
    assert (result.failure["stage"], result.failure["observed"]) == (stage, observed)
    assert result.failure["side_effects"] == "none" and result.failure["answer_kept"] is True


def test_control_a_normal_record_is_this_claims_terminal_and_replays_without_a_turn(gw):
    first = submit(gw, event())
    assert first == CanonicalReceiptResult("terminal", "answer #1")
    _, receipt = rows(gw, RECEIPT)
    _, claim = rows(gw, CLAIM)
    assert receipt["state"] == "terminal" and receipt["response"]["terminal_text"] == "answer #1"
    assert receipt["owner"] == claim["owner"]
    assert rows(gw, KEPT) == (None, None)
    restart(gw)
    assert submit(gw, event()) == first
    assert gw.turns == ["evt-1"]


def test_a_newer_run_generation_during_the_turn_wins_over_its_terminal_record(gw):
    key = gw.entry.session_key
    gw.actor.during_turn = lambda: gw.runner._invalidate_session_run_generation(key, reason="/new")

    result = submit(gw, event())

    kept_unrecorded(result, stage="record.proof", observed="own_pending")
    _, receipt = rows(gw, RECEIPT)
    assert receipt["state"] == "pending"  # the newer generation's: never settled by the displaced turn
    _, kept = rows(gw, KEPT)
    assert kept["answer"]["response"]["terminal_text"] == "answer #1"
    assert kept["owner"] == receipt["owner"]

    gw.actor.during_turn = None
    assert submit(gw, event()) == result
    restart(gw)
    assert submit(gw, event()) == result
    assert gw.turns == ["evt-1"]


def test_another_holders_newer_receipt_row_is_never_overwritten_by_the_terminal_compare_and_set(gw):
    winner = {}

    def another_holder_claims():
        key, pending = rows(gw, RECEIPT)
        winner["key"] = key
        winner["value"] = json.dumps({**pending, "owner": "f" * 32}, sort_keys=True)
        write_row(gw, key, winner["value"])

    gw.actor.during_turn = another_holder_claims
    result = submit(gw, event())

    kept_unrecorded(result, stage="record.compare_and_set", observed="other_pending")
    with gw.db._lock:
        assert gw.db._conn.execute("SELECT value FROM state_meta WHERE key = ?",
                                   (winner["key"],)).fetchone()[0] == winner["value"]
    gw.actor.during_turn = None
    assert submit(gw, event()) == result
    assert gw.turns == ["evt-1"]


def test_a_receipt_row_deleted_mid_turn_is_never_recreated_and_the_event_never_runs_again(gw):
    gw.actor.during_turn = lambda: delete_row(gw, rows(gw, RECEIPT)[0])
    result = submit(gw, event())
    kept_unrecorded(result, stage="record.compare_and_set", observed="absent")
    assert rows(gw, RECEIPT) == (None, None)

    gw.actor.during_turn = None
    assert submit(gw, event()) == result
    delete_row(gw, rows(gw, KEPT)[0])
    # Only the claim record is left: the event was claimed and may have run, so it is uncertain.
    assert submit(gw, event()) == CanonicalReceiptResult("uncertain")
    restart(gw)
    assert submit(gw, event()) == CanonicalReceiptResult("uncertain")
    assert gw.turns == ["evt-1"]


def refuse_kept_rows(gw):
    """The state.db refuses the kept answer as it refused the terminal; every other claim lands."""
    claim = gw.db.claim_meta_once

    def refuse_kept(key, value, **proof):
        if key.startswith(KEPT):
            raise OSError("disk full")
        return claim(key, value, **proof)

    gw.db.claim_meta_once = refuse_kept


def test_an_answer_the_state_db_refuses_to_keep_is_kept_privately_beside_it_and_read_after_a_restart(gw):
    refuse_kept_rows(gw)
    gw.actor.during_turn = lambda: gw.runner._invalidate_session_run_generation(gw.entry.session_key)
    result = submit(gw, event())

    kept_unrecorded(result, stage="record.proof", observed="own_pending")
    directory = gw.home / canonical_answer_fallback.DIR_NAME
    (only,) = directory.iterdir()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700 and stat.S_IMODE(only.stat().st_mode) == 0o600
    kept = json.loads(only.read_text())
    assert kept["answer"]["response"]["terminal_text"] == "answer #1"
    assert only.name == f"{kept['receipt'].rsplit(':', 1)[1]}.{kept['owner']}.json"

    gw.actor.during_turn = None
    restart(gw)
    assert submit(gw, event()) == result
    assert gw.turns == ["evt-1"]


def test_when_no_store_keeps_the_answer_the_failure_says_so_and_nothing_reruns(gw, monkeypatch):
    refuse_kept_rows(gw)

    def no_space(*_args):
        raise OSError("no space")

    monkeypatch.setattr(canonical_answer_fallback, "keep", no_space)
    gw.actor.during_turn = lambda: gw.runner._invalidate_session_run_generation(gw.entry.session_key)

    result = submit(gw, event())

    assert result.status == "unrecorded" and result.failure["answer_kept"] is False
    gw.actor.during_turn = None
    # Nothing kept and the receipt still pending: a replay says pending, never a second run.
    assert submit(gw, event()).status == "pending"
    assert gw.turns == ["evt-1"]


# --- the local fallback ---------------------------------------------------------------------------


DIGEST, OWNER, OTHER = "a" * 64, "b" * 32, "c" * 32


def test_the_fallback_never_writes_over_a_file_it_did_not_create(tmp_path):
    canonical_answer_fallback.keep(tmp_path, DIGEST, OWNER, b"first")
    with pytest.raises(FileExistsError):
        canonical_answer_fallback.keep(tmp_path, DIGEST, OWNER, b"second")
    canonical_answer_fallback.keep(tmp_path, DIGEST, OTHER, b"another claim")
    assert canonical_answer_fallback.kept_for(tmp_path, DIGEST) == [(OWNER, b"first"), (OTHER, b"another claim")]
    directory = tmp_path / canonical_answer_fallback.DIR_NAME
    assert not [name for name in os.listdir(directory) if name.endswith(".tmp")]


def test_the_fallback_refuses_a_symlinked_directory_and_an_unsafe_file(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    home = tmp_path / "home"
    home.mkdir()
    (home / canonical_answer_fallback.DIR_NAME).symlink_to(elsewhere)
    with pytest.raises(OSError):
        canonical_answer_fallback.keep(home, DIGEST, OWNER, b"answer")
    assert list(elsewhere.iterdir()) == []

    other = tmp_path / "other"
    other.mkdir()
    assert canonical_answer_fallback.kept_for(other, DIGEST) == []
    canonical_answer_fallback.keep(other, DIGEST, OWNER, b"answer")
    (other / canonical_answer_fallback.DIR_NAME / f"{DIGEST}.{OWNER}.json").chmod(0o644)
    with pytest.raises(canonical_answer_fallback.FallbackUnsafe):
        canonical_answer_fallback.kept_for(other, DIGEST)


def test_coordinator_reads_an_unsafe_fallback_as_unreadable_never_as_nothing_kept(gw):
    submit(gw, event())
    key, _ = rows(gw, RECEIPT)
    canonical_answer_fallback.keep(gw.home, key.rsplit(":", 1)[1], OWNER, b"{}")
    (gw.home / canonical_answer_fallback.DIR_NAME / f"{key.rsplit(':', 1)[1]}.{OWNER}.json").chmod(0o644)
    with pytest.raises(ValueError, match="^canonical_receipt_unreadable$"):
        CanonicalReceiptCoordinator._fallback_kept(gw.home, key)
