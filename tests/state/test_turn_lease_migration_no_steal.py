"""The v26 → v27 lease migration, with a REAL legacy owner alive across the TTL.

WHAT WAS UNPINNED
    ``_heal_session_turn_lease_epoch`` carries pre-epoch rows over at
    ``epoch = 0`` and recovers ``owner_pid`` from the holder string. Its
    docstring states why — "dropping them would be the easy migration and the
    wrong one: at upgrade time a live owner's conversation would become
    instantly acquirable by a bystander" — and nothing in the tree checked it
    with an owner that was actually there.

    Every existing liveness pin stands in a foreign process's shoes with
    ``os.getppid()``: a PID that exists, that the test did not start, and whose
    lifetime the test does not control. That is enough to show the predicate
    reads "alive", and not enough to show the MIGRATION preserved the identity
    the predicate reads. These pins start the owner, hold it open across a real
    deadline, and assert on the migrated row's values.

THE DIFFERENCE THE PINS EXIST TO DRAW
    Two recoveries look identical from the outside — the conversation becomes
    writable again — and only one of them is correct:

    * **wrong** — "the TTL expired, so anyone may take it". The deadline says
      the refresher did not run. A laptop that slept, a stopped-world GC and a
      dead owner produce the same row, so reclaiming on it hands a live turn's
      conversation to a contender.
    * **right** — "an operator explicitly force-released it". Deliberate, loud,
      and it keeps the epoch, so the evicted grant can never be replayed.

    So the TTL is allowed to elapse in these checks, for real, with the owner
    process demonstrably alive, and the assertion is that NOTHING changed: same
    holder, same epoch, same transcript. The conversation opens again only when
    :meth:`force_release_session_turn_lease` is called.

WHY THE LEGACY STORE IS BUILT BY DROPPING THE FENCE TRIGGERS
    A v26 binary is a process that never installed the generation triggers, so
    that is how the fixture makes one: it drops the trigger surface (the exact
    published rollback, see ``hermes_cli/session_fence_rollback.py``) and
    rebuilds ``session_turn_leases`` in its four-column shape on a connection
    that registered nothing. The trigger names come from
    ``TURN_FENCE_TRIGGERS``, which is derived from the surface declaration, so
    a table added to the fence later arrives here too.

    Nothing in this file calls ``register_turn_fence_function`` on a writing
    connection. The marker proves "current generation" and nothing else, and
    adding it to a raw writer is how a second admitted door gets minted.
"""

from __future__ import annotations

import os
import pathlib
import sqlite3
import subprocess
import sys
import time

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: Liveness bound for "the deadline should have passed by now". Its expiry is a
#: test FAILURE, never a pass condition — no assertion reads elapsed time.
_LIVENESS_S = 60.0

_LEGACY_LEASE_SHAPE = (
    "CREATE TABLE session_turn_leases ("
    "conversation_id TEXT PRIMARY KEY, "
    "holder TEXT NOT NULL, "
    "acquired_at REAL NOT NULL, "
    "expires_at REAL NOT NULL)"
)


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _lease_values(db, conversation_id="s"):
    """The lease row as VALUES. "Unchanged" is a comparison, not a vibe."""
    row = db.get_session_turn_lease(conversation_id)
    if row is None:
        return None
    return (
        row["holder"],
        int(row["epoch"]),
        row["owner_pid"],
        float(row["acquired_at"]),
        float(row["expires_at"]),
    )


def _start_a_real_owner_process():
    """A process this test starts, keeps alive, and can prove is alive."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
    )


def _build_a_legacy_store(tmpdir, holder: str, *, ttl_seconds: float):
    """A store shaped the way a binary from before the epoch column left it.

    Returns ``(path, acquired_at, expires_at)``. The lease row is written by a
    connection that registered no generation marker — which is precisely what
    an old binary is — after the fence triggers have been dropped.
    """
    from hermes_state import SessionDB
    from hermes_state_common import TURN_FENCE_TRIGGERS

    path = pathlib.Path(tmpdir) / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("s", source="test")
    db.append_message("s", "user", "legacy context")
    db.close()

    conn = sqlite3.connect(str(path))
    try:
        for trigger in TURN_FENCE_TRIGGERS:
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE session_turn_leases")
        conn.execute(_LEGACY_LEASE_SHAPE)
        acquired_at = time.time()
        expires_at = acquired_at + ttl_seconds
        conn.execute(
            "INSERT INTO session_turn_leases "
            "(conversation_id, holder, acquired_at, expires_at) "
            "VALUES (?, ?, ?, ?)",
            ("s", holder, acquired_at, expires_at),
        )
        conn.commit()
    finally:
        conn.close()
    return path, acquired_at, expires_at


def _wait_out_the_deadline(expires_at: float, still_alive) -> None:
    """Let the TTL elapse FOR REAL, and prove the owner outlived it.

    The point of the whole file is that these two facts are simultaneously
    true: the deadline has passed, and the owner is still there. Asserting the
    second one after the first is what makes "expired" mean "the refresher did
    not run" rather than "the owner is gone".
    """
    deadline = time.monotonic() + _LIVENESS_S
    while time.time() <= expires_at:
        assert time.monotonic() < deadline, (
            "the lease deadline never passed; this check would assert "
            "no-steal on a lease that was never expired"
        )
        time.sleep(0.02)
    assert still_alive(), (
        "the owner process died before its lease deadline passed, so this "
        "check is measuring a DEAD owner — the case where reclaiming is right"
    )


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_a_live_legacy_owner_is_not_stolen_when_the_ttl_expires(tmpdir) -> None:
    """The migrated row survives its own deadline while its owner is running.

    Assertions are on ROWS: the lease row is compared value-by-value before and
    after the contender's attempt, and the transcript is compared before and
    after a holderless write. "It returned None" and "it raised" are each half
    of the property; the other half is that nothing moved.
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError

    owner = _start_a_real_owner_process()
    db = None
    try:
        path, _acquired, expires_at = _build_a_legacy_store(
            tmpdir, f"pid={owner.pid}:turn=legacy:platform=telegram",
            ttl_seconds=0.25,
        )
        db = SessionDB(db_path=path)  # opening runs the migration
        before = _lease_values(db)
        assert before is not None, "the migration dropped the carried-over row"

        _wait_out_the_deadline(expires_at, lambda: owner.poll() is None)
        assert before[4] < time.time(), (
            "the lease deadline is still in the future, so no expiry has been "
            "exercised"
        )

        stolen = db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        )
        assert stolen is None, (
            f"a contender took a conversation whose owner (pid {owner.pid}) is "
            f"running right now, because the deadline had passed"
        )
        assert _lease_values(db) == before, (
            f"the lease row moved under a refused contender: "
            f"{_lease_values(db)!r} != {before!r}"
        )

        try:
            db.append_message("s", "assistant", "stolen")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a holderless write landed on a conversation a live legacy "
                "owner still holds"
            )
        assert [m["content"] for m in db.get_messages("s")] == ["legacy context"], (
            "the refused write changed the transcript anyway"
        )
        assert _lease_values(db) == before

        # And the deliberate exit DOES work, keeping the generation monotonic.
        assert db.force_release_session_turn_lease("s") is True
        after_force = _lease_values(db)
        assert after_force is not None and after_force[0] == "", (
            f"force_release did not free the row: {after_force!r}"
        )
        assert after_force[1] == before[1], (
            f"force_release must KEEP the epoch so the evicted grant cannot be "
            f"replayed: {before[1]} -> {after_force[1]}"
        )
        granted = db.try_acquire_session_turn_lease(
            "s", _holder("after-force"), ttl_seconds=300
        )
        assert granted is not None, "force_release did not actually free it"
        assert granted.epoch == before[1] + 1, (
            f"the post-force grant must be a strictly later generation than "
            f"the legacy one: {before[1]} -> {granted.epoch}"
        )
    finally:
        if db is not None:
            db.close()
        owner.terminate()
        owner.wait(timeout=_LIVENESS_S)


def check_the_migration_carries_the_row_and_recovers_the_owner(tmpdir) -> None:
    """Carried over, at epoch 0, with the owner's identity recovered.

    Three values, each load-bearing and each one a different failure if it is
    wrong: the row must exist (dropping it hands the conversation away at
    upgrade time), the epoch must be 0 (a generation no grant can match, so a
    writer presenting a token against it fails closed), and ``owner_pid`` must
    be the pid the holder string names (without it liveness has nothing to
    decide from and the row is unreachable by any evidence at all).
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError

    owner = _start_a_real_owner_process()
    db = None
    try:
        holder = f"pid={owner.pid}:turn=legacy:platform=telegram"
        path, acquired_at, expires_at = _build_a_legacy_store(
            tmpdir, holder, ttl_seconds=900.0
        )
        db = SessionDB(db_path=path)
        row = _lease_values(db)
        assert row is not None, (
            "the migration dropped the carried-over lease; at upgrade time a "
            "live owner's conversation becomes instantly acquirable"
        )
        assert row[0] == holder, f"the holder string was rewritten: {row[0]!r}"
        assert row[1] == 0, (
            f"a carried-over row must sit at the generation no grant can match; "
            f"got epoch {row[1]}"
        )
        assert row[2] == owner.pid, (
            f"owner_pid was not recovered from the holder string: {row[2]!r} != "
            f"{owner.pid}. Without it the row has no identity and no evidence "
            f"can ever free it"
        )
        assert row[3] == pytest.approx(acquired_at)
        assert row[4] == pytest.approx(expires_at)

        # A carried-over row cannot be validated by any grant: epoch 0 is
        # refused explicitly, so a hand-built token cannot ride the migration.
        from hermes_state import SessionTurnLeaseToken

        forged = SessionTurnLeaseToken(holder, 0, "s")
        try:
            db.append_message(
                "s", "assistant", "forged", turn_lease_holder=forged
            )
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError(
                "a token hand-built at the legacy generation was admitted; "
                "epoch 0 has to be unmatchable or the migration is a door"
            )
        assert [m["content"] for m in db.get_messages("s")] == ["legacy context"]
        assert _lease_values(db) == row
    finally:
        if db is not None:
            db.close()
        owner.terminate()
        owner.wait(timeout=_LIVENESS_S)


def check_a_same_pid_live_owner_is_not_stolen_when_the_ttl_expires(tmpdir) -> None:
    """The case a subprocess cannot exercise, on a MIGRATED store.

    After the migration this process takes the conversation for real — the
    generation continues from the legacy 0 to 1 — and then the deadline
    passes while the grant is still held here. ``_turn_lease_owner_is_dead``
    answers "not dead" for our own PID (correctly: the process is right here),
    so same-PID admission falls through to whatever else the predicate
    consults, and the only correct answer is the live-grant registry.
    """
    from hermes_state import SessionDB

    db = None
    # A pid that is PROVABLY gone: this test started it and reaped it, so
    # `psutil.pid_exists` answering False is a fact rather than a guess. The
    # carried row therefore starts out reclaimable, which is what lets this
    # process become the real owner before the deadline part of the check.
    reaped = _start_a_real_owner_process()
    reaped.terminate()
    reaped.wait(timeout=_LIVENESS_S)
    try:
        path, _acquired, _expires = _build_a_legacy_store(
            tmpdir, f"pid={reaped.pid}:turn=legacy:platform=telegram",
            ttl_seconds=-1.0,
        )
        db = SessionDB(db_path=path)
        assert _lease_values(db)[1] == 0, "the migration did not carry the row"

        # This process becomes the real owner. The legacy generation is the
        # floor, so the grant is strictly later than anything carried over.
        held = db.try_acquire_session_turn_lease(
            "s", _holder("owner"), ttl_seconds=0.25
        )
        assert held is not None, (
            f"a carried-over row naming a pid this test reaped ({reaped.pid}) "
            f"should be takeable; if it is not, the migration wedged the store"
        )
        assert held.epoch == 1, (
            f"the generation must continue from the legacy 0: got {held.epoch}"
        )
        before = _lease_values(db)
        _wait_out_the_deadline(before[4], lambda: True)

        contender = db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        )
        assert contender is None, (
            "a contender sharing os.getpid() with the owner took the lease "
            "because the deadline had passed, while this process is still "
            "holding the grant"
        )
        assert _lease_values(db) == before, (
            f"the lease row moved under a refused same-pid contender: "
            f"{_lease_values(db)!r} != {before!r}"
        )
        # The owner is unaffected: its own grant still writes.
        assert db.append_message(
            "s", "assistant", "owner write", turn_lease_holder=held
        )
        assert [m["content"] for m in db.get_messages("s")] == [
            "legacy context", "owner write",
        ]
    finally:
        if db is not None:
            db.close()


def check_an_unknown_legacy_holder_needs_force_release_not_the_clock(tmpdir) -> None:
    """A holder no evidence can reach stays held, and force-release is the exit.

    A pre-``owner_pid`` row whose holder string is not ``pid=<n>`` has no
    identity at all: liveness is UNKNOWN, which is not liveness-DEAD, so
    nothing frees it. That is the largest cost of taking the clock away and it
    is paid deliberately — a wedged conversation is recoverable, an interleaved
    transcript is not. The exit therefore has to exist and has to be the only
    one.
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError

    db = None
    try:
        path, _acquired, expires_at = _build_a_legacy_store(
            tmpdir, "gateway-writer-no-identity", ttl_seconds=0.25
        )
        db = SessionDB(db_path=path)
        before = _lease_values(db)
        assert before is not None and before[2] is None, (
            f"this check needs a row with NO recorded identity; got {before!r}"
        )
        _wait_out_the_deadline(expires_at, lambda: True)

        assert db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        ) is None, (
            "an unknown legacy holder was reclaimed by its deadline; "
            "'cannot prove alive' is not 'dead'"
        )
        try:
            db.append_message("s", "assistant", "stolen")
        except SessionTurnLeaseLostError as exc:
            assert "force_release_session_turn_lease" in str(exc), (
                f"the refusal does not name the exit an operator has, so the "
                f"exit is unfindable: {exc}"
            )
        else:
            raise AssertionError("a holderless write landed on a wedged row")
        assert _lease_values(db) == before
        assert [m["content"] for m in db.get_messages("s")] == ["legacy context"]

        assert db.force_release_session_turn_lease("s") is True
        granted = db.try_acquire_session_turn_lease(
            "s", _holder("after-force"), ttl_seconds=300
        )
        assert granted is not None and granted.epoch == before[1] + 1, (
            f"force_release is the only exit and it did not work: {granted!r}"
        )
    finally:
        if db is not None:
            db.close()


PINS = {
    "check_a_live_legacy_owner_is_not_stolen_when_the_ttl_expires":
        check_a_live_legacy_owner_is_not_stolen_when_the_ttl_expires,
    "check_the_migration_carries_the_row_and_recovers_the_owner":
        check_the_migration_carries_the_row_and_recovers_the_owner,
    "check_a_same_pid_live_owner_is_not_stolen_when_the_ttl_expires":
        check_a_same_pid_live_owner_is_not_stolen_when_the_ttl_expires,
    "check_an_unknown_legacy_holder_needs_force_release_not_the_clock":
        check_an_unknown_legacy_holder_needs_force_release_not_the_clock,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_migration_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


# ---------------------------------------------------------------------------
# The mutation table.
#
# The guard being removed here is unusual: for three of the four rows it is the
# ABSENCE of a clock in the admission predicate. So the mutation puts the old
# rule back — "expired ⇒ reclaimable" — which is exactly the contract this work
# replaced, and requires the pin to die. A pin that survives the restoration of
# the rule it exists to forbid is asserting nothing.
# ---------------------------------------------------------------------------

_CLOCK_ANCHOR = (
    '        if row is None:\n'
    '            return True\n'
    '        holder = row["holder"] or ""\n'
)
_CLOCK_MUTANT = (
    '        if row is None:\n'
    '            return True\n'
    '        if float(row["expires_at"] or 0) <= time.time():\n'
    '            return True\n'
    '        holder = row["holder"] or ""\n'
)

SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_live_legacy_owner_is_not_stolen_when_the_ttl_expires",
        module="hermes_state.py",
        find="        if owner_pid_start is None:\n            return False\n",
        replace="        if owner_pid_start is None:\n            return True\n",
        why="a carried-over row has NO recorded start time, so treating "
            "'unknown start' as proof of death declares every migrated owner "
            "dead and hands its conversation away at upgrade time",
    ),
    Mutation(
        pin="check_the_migration_carries_the_row_and_recovers_the_owner",
        module="hermes_state_schema.py",
        find="        for row in carried:\n",
        replace="        for row in []:\n",
        why="dropping the rows is the easy migration and the wrong one: at "
            "upgrade time a live owner's conversation becomes acquirable by "
            "anyone",
    ),
    Mutation(
        pin="check_the_migration_carries_the_row_and_recovers_the_owner",
        module="hermes_state_schema.py",
        find="            owner_pid = int(match.group(1)) if match else None\n",
        replace="            owner_pid = None\n",
        why="without recovering the pid from the holder string the carried row "
            "has no identity, so no evidence can ever decide its liveness",
    ),
    Mutation(
        pin="check_a_same_pid_live_owner_is_not_stolen_when_the_ttl_expires",
        module="hermes_state.py",
        find="        if owner_pid and int(owner_pid) == os.getpid():\n"
             "            return live_turn_grant(self.db_path, conversation_id) is None\n",
        replace="        if owner_pid and int(owner_pid) == os.getpid():\n"
                "            return True\n",
        why="the live-grant registry is the only thing that distinguishes 'our "
            "own turn is running' from 'our own turn died without releasing'; "
            "without it a same-pid contender takes a live conversation",
    ),
    Mutation(
        pin="check_an_unknown_legacy_holder_needs_force_release_not_the_clock",
        module="hermes_state.py",
        find=_CLOCK_ANCHOR,
        replace=_CLOCK_MUTANT,
        why="this restores the rule the contract replaced — expired means "
            "reclaimable — which frees a holder nothing can prove anything "
            "about and makes force_release redundant",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS,
    ids=[f"{m.pin}-{i}" for i, m in enumerate(SOURCE_MUTATIONS)],
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    """Clean, mutated, restored. A pin that survives its mutation is not a pin."""
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path)


def test_every_pin_has_a_mutation_that_kills_it():
    """No pin without a killer, and no killer without a pin."""
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
