"""The v26 → v27 lease migration, with the EXACT BASE BINARY holding the lease.

WHAT AN EARLIER DRAFT OF THIS FILE GOT WRONG, AND WHY IT MATTERED
    It started ``python -c 'time.sleep(600)'`` and then wrote a lease row whose
    holder string NAMED that sleeper's pid. That process never imported the
    base object, never opened the store, never acquired anything. It was a live
    pid with the right number in a string the test wrote.

    What that establishes is real and narrow — a migrated row whose recorded
    owner is a live pid is treated as live — and it is kept below under a name
    that says exactly that. What it does NOT establish is the acceptance: that
    a v26 owner which opened the store and took the lease with the base code
    keeps it across the migration. Calling the first thing the second is the
    defect, because the next reader takes the item as closed.

WHAT THE BASE-PROCESS PINS DO INSTEAD
    ``261a4efb90d7dbe4e71786861858f721b4ab730c`` — the exact commit, extracted
    with ``git archive`` — runs in a child interpreter whose ``sys.path`` and
    cwd are the extract, and which asserts its own ``hermes_state.__file__``
    lives there so the current tree cannot leak in. THAT child creates the
    store, and THAT child acquires the lease through the base's own
    ``try_acquire_session_turn_lease``. The store is therefore v26 because a
    v26 binary made it, not because a fixture reshaped one.

    The child then emits a structured READY line carrying the pid, the holder
    and the lease row AS IT OBSERVES THEM, and blocks on stdin. The parent
    never guesses when the child is up and never sleeps to find out.

THE TWO EXERCISES ARE SEPARATE ON PURPOSE
    ``check_the_base_owner_keeps_its_lease_when_the_ttl_expires`` lets the
    deadline pass FOR REAL and then proves the owner is still there with an
    application-level round trip — the parent sends PING and the child answers
    — before asserting nothing moved.

    ``check_the_base_owner_is_taken_over_only_once_it_is_provably_dead`` never
    lets the deadline pass: the lease is taken with an hour of TTL, so the
    clock says HELD throughout. The child is killed and REAPED (a returncode,
    not a timeout), and only then does takeover succeed. Conflating the two is
    the original defect in a new costume — "the TTL passed so anyone may take
    it" is precisely what this item says is wrong, and a single test that
    expires the lease AND kills the owner cannot tell which one freed it.

FAILURE MODES ARE ERRORS, NOT REDS
    An unreachable base object, a child that cannot import, or a child that
    dies before READY are raised as failures naming what happened. None of them
    can read as a proven property, and none of them is silently tolerated.

    The one tolerated absence is having no git repository at all, which is
    reported as a skip by the shared helper; the pins below that need no base
    object still run there.

NOTHING HERE REGISTERS THE FENCE FUNCTION ON A WRITING CONNECTION
    The base child needs no help: at ``261a4efb`` the triggers do not exist, so
    it writes its own store with its own code. The marker proves "current
    generation" and nothing else, and adding it to a raw writer is how a second
    admitted door gets minted.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap
import time

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)
from tests.state.test_turn_lease_generation_trigger import (
    BASE_COMMIT,
    BASE_TREE_PATHSPEC,
    _git_dir,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: Liveness bound only. Every use is asserted to have been reached; an expiry
#: is reported as "the child never got there", never as a passing observation.
_LIVENESS_S = 60.0


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


# ---------------------------------------------------------------------------
# The exact-base owner process.
# ---------------------------------------------------------------------------

#: The child program. It is the BASE binary doing the work: creating the store,
#: seeding a transcript and taking the lease through the base's own API. The
#: parent supplies only a path and a TTL.
_BASE_OWNER_PROGRAM = textwrap.dedent(
    """
    import json, os, pathlib, sqlite3, sys

    tree = pathlib.Path(sys.argv[1]).resolve()
    store = pathlib.Path(sys.argv[2])
    ttl = float(sys.argv[3])
    sys.path.insert(0, str(tree))

    import hermes_state
    loaded = pathlib.Path(hermes_state.__file__).resolve()
    if not loaded.is_relative_to(tree):
        raise SystemExit("the owner imported %s, not the base extract" % loaded)

    db = hermes_state.SessionDB(db_path=store)
    db.create_session("s", source="legacy")
    db.append_message("s", "user", "legacy context")
    # The holder string the SHIPPED base binary writes, quoted from
    # run_agent.py:8625 at 261a4efb (`_durable_holder`). It is composed here
    # rather than imported because the base has no make_turn_lease_holder --
    # that helper arrived with this work -- and run_agent.py is not part of the
    # minimal extract. This is the exact format, and `pid=<n>:` at the front is
    # the part the migration reads.
    holder = "pid=%d:turn=relay-legacy:platform=telegram" % os.getpid()
    acquired = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=ttl)

    probe = sqlite3.connect(str(store))
    probe.row_factory = sqlite3.Row
    row = probe.execute(
        "SELECT conversation_id, holder, acquired_at, expires_at "
        "FROM session_turn_leases WHERE conversation_id = 's'"
    ).fetchone()
    columns = [
        c[1] for c in probe.execute(
            'PRAGMA table_info("session_turn_leases")'
        ).fetchall()
    ]
    probe.close()

    print(json.dumps({
        "ready": True,
        "pid": os.getpid(),
        "module": str(loaded),
        "holder": holder,
        "acquired": bool(acquired),
        "columns": columns,
        "row": None if row is None else dict(row),
    }), flush=True)

    # The barrier. The owner holds its SessionDB open and does nothing until
    # the parent says so; every ordering fact in the parent is carried by this
    # round trip rather than by elapsed time.
    for line in sys.stdin:
        command = line.strip()
        if command == "PING":
            print(json.dumps({"alive": True, "pid": os.getpid()}), flush=True)
        elif command == "EXIT":
            break
    """
)


class _BaseOwner:
    """A live ``261a4efb`` process holding the turn lease on a store it made."""

    def __init__(self, process, ready: dict, store: pathlib.Path):
        self.process = process
        self.ready = ready
        self.store = store

    @property
    def pid(self) -> int:
        return int(self.ready["pid"])

    @property
    def holder(self) -> str:
        return str(self.ready["holder"])

    def ping(self) -> dict:
        """Prove the owner is alive by making it answer, not by asking the OS.

        ``pid_exists`` is what the code under test consults, so using it here
        as the parent's independent evidence would be checking the predicate
        against itself. A reply is evidence the process is running.
        """
        assert self.process.poll() is None, (
            f"the base owner exited with {self.process.returncode} before it "
            f"was asked anything"
        )
        self.process.stdin.write("PING\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        assert line, (
            "the base owner closed its output instead of answering; it is not "
            "alive, so nothing below is measuring a live owner"
        )
        answer = json.loads(line)
        assert answer.get("alive") is True and answer.get("pid") == self.pid
        return answer

    def kill_and_reap(self) -> int:
        """Kill the owner and observe its exit. Death is a returncode here.

        Inferring death from a timeout would be the same mistake the whole
        contract is about: "we waited and nothing happened" is not evidence.
        """
        self.process.kill()
        returncode = self.process.wait(timeout=_LIVENESS_S)
        assert returncode is not None, "the base owner was never reaped"
        assert self.process.poll() is not None
        return returncode

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(timeout=_LIVENESS_S)
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except Exception:  # pragma: no cover - teardown only
                pass


def _extract_base_tree(tmpdir: pathlib.Path) -> pathlib.Path:
    """``BASE_COMMIT``'s tree, or a loud failure naming what went wrong."""
    git_dir = _git_dir()
    if git_dir is None:
        pytest.skip(
            "no git repository to read the base commit from; set "
            "HERMES_BASE_TREE_GIT_DIR to one. Nothing here reports a property "
            "as proven in that state."
        )
    out = pathlib.Path(tmpdir) / "base"
    out.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "-C", git_dir, "archive", BASE_COMMIT, "--", *BASE_TREE_PATHSPEC],
        capture_output=True,
    )
    assert archive.returncode == 0, (
        f"the base object {BASE_COMMIT} could not be read out of {git_dir}: "
        f"{archive.stderr.decode(errors='replace')}. This is an ERROR, not a "
        f"failing property — no claim below has been measured."
    )
    extract = subprocess.run(
        ["tar", "-x", "-C", str(out)], input=archive.stdout, capture_output=True
    )
    assert extract.returncode == 0, extract.stderr.decode(errors="replace")
    assert (out / "hermes_state.py").is_file(), (
        "the base extract has no hermes_state.py, so no base binary can be run"
    )
    return out


def _start_base_owner(tmpdir, *, ttl_seconds: float) -> _BaseOwner:
    """Run ``BASE_COMMIT`` until it has the lease, and return it holding it."""
    tmpdir = pathlib.Path(tmpdir)
    tree = _extract_base_tree(tmpdir)
    store = tmpdir / "legacy" / "state.db"
    store.parent.mkdir(parents=True, exist_ok=True)
    errors = tmpdir / "base-owner.stderr"

    with open(errors, "w") as stderr:
        process = subprocess.Popen(
            [
                sys.executable, "-c", _BASE_OWNER_PROGRAM,
                str(tree), str(store), str(ttl_seconds),
            ],
            cwd=str(tree),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            env={
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "HOME": str(tmpdir),
                "PYTHONPATH": str(tree),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

    line = process.stdout.readline()
    if not line:
        process.wait(timeout=_LIVENESS_S)
        raise AssertionError(
            f"the base owner produced no READY line and exited "
            f"{process.returncode}. This is an ERROR — the fixture never ran, "
            f"so nothing was measured.\n"
            f"{errors.read_text(errors='replace')}"
        )
    ready = json.loads(line)
    assert ready.get("ready") is True
    assert ready.get("acquired") is True, (
        f"the base binary could not take its own lease: {ready!r}"
    )
    assert "epoch" not in ready.get("columns", []), (
        f"the store the base binary created already has the epoch column, so "
        f"there is no v26 → v27 migration left to exercise: {ready['columns']!r}"
    )
    assert ready.get("row") is not None, "the base binary left no lease row"
    return _BaseOwner(process, ready, store)


def _wait_out_the_deadline(expires_at: float) -> None:
    """Let the TTL elapse FOR REAL, with a liveness bound that is not a clock.

    The bound's expiry is a FAILURE ("the deadline never passed"), never a pass
    condition; no assertion anywhere reads elapsed time.
    """
    deadline = time.monotonic() + _LIVENESS_S
    while time.time() <= expires_at:
        assert time.monotonic() < deadline, (
            "the lease deadline never passed, so this check would assert "
            "no-steal on a lease that was never expired"
        )
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_the_base_owner_keeps_its_lease_when_the_ttl_expires(tmpdir) -> None:
    """A v26 process holds its lease across the migration and across the TTL.

    The owner is the exact base object. It made the store, it took the lease,
    and it is still answering when the assertions run. The deadline passes for
    real in between, which is the whole point: a lease that expires while its
    owner is demonstrably running is the case where reclaiming hands one
    conversation to two writers.
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError

    owner = _start_base_owner(tmpdir, ttl_seconds=0.4)
    db = None
    try:
        owner.ping()
        db = SessionDB(db_path=owner.store)  # opening runs the migration
        before = _lease_values(db)
        assert before is not None, (
            "the migration dropped the row a live v26 owner was holding; at "
            "upgrade time its conversation becomes acquirable by anyone"
        )
        assert before[0] == owner.holder, (
            f"the migration rewrote the holder: {before[0]!r} != {owner.holder!r}"
        )
        assert before[1] == 0, (
            f"a carried-over row must sit at the generation no grant can "
            f"match; got epoch {before[1]}"
        )
        assert before[2] == owner.pid, (
            f"owner_pid was not recovered from the v26 holder string: "
            f"{before[2]!r} != {owner.pid}"
        )

        _wait_out_the_deadline(before[4])
        assert before[4] < time.time(), "no expiry has been exercised"
        owner.ping()  # the owner outlived its own deadline, and says so

        stolen = db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        )
        assert stolen is None, (
            f"a contender took the conversation while the v26 owner "
            f"(pid {owner.pid}) was still answering, because the deadline had "
            f"passed"
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
                "a holderless write landed on a conversation a live v26 owner "
                "still holds"
            )
        assert [m["content"] for m in db.get_messages("s")] == ["legacy context"], (
            "the refused write changed the transcript anyway"
        )
        assert _lease_values(db) == before
        owner.ping()  # ...and it was alive for every one of those assertions

        # The deliberate exit is the ONLY thing that opens it, and it keeps the
        # generation monotonic so the evicted grant can never be replayed.
        assert db.force_release_session_turn_lease("s") is True
        after_force = _lease_values(db)
        assert after_force is not None and after_force[0] == "", (
            f"force_release did not free the row: {after_force!r}"
        )
        assert after_force[1] == before[1], (
            f"force_release must KEEP the epoch: {before[1]} -> {after_force[1]}"
        )
        granted = db.try_acquire_session_turn_lease(
            "s", _holder("after-force"), ttl_seconds=300
        )
        assert granted is not None, "force_release did not actually free it"
        assert granted.epoch == before[1] + 1, (
            f"the post-force grant must be a strictly later generation: "
            f"{before[1]} -> {granted.epoch}"
        )
    finally:
        if db is not None:
            db.close()
        owner.close()


def check_the_base_owner_is_taken_over_only_once_it_is_provably_dead(
    tmpdir,
) -> None:
    """Death frees the lease. The clock never gets the chance to.

    The lease is taken with an hour of TTL and the deadline is asserted to be
    in the FUTURE at every step, so "expired" is never available as an
    explanation. The owner is killed and reaped — a returncode, not a timeout —
    and the takeover happens only after that.
    """
    from hermes_state import SessionDB

    owner = _start_base_owner(tmpdir, ttl_seconds=3600.0)
    db = None
    try:
        owner.ping()
        db = SessionDB(db_path=owner.store)
        before = _lease_values(db)
        assert before is not None and before[2] == owner.pid
        assert before[4] > time.time(), (
            "the lease is already expired, so a later takeover could be the "
            "clock's doing and this check would not be able to tell"
        )

        # Alive: refused.
        assert db.try_acquire_session_turn_lease(
            "s", _holder("early-contender"), ttl_seconds=300
        ) is None, (
            "a contender took the conversation while the v26 owner was alive "
            "and its lease was not even expired"
        )
        assert _lease_values(db) == before

        returncode = owner.kill_and_reap()
        assert returncode is not None

        # Dead: takeover, with the deadline still in the future.
        assert before[4] > time.time(), (
            "the deadline passed during the kill, so this check can no longer "
            "attribute the takeover to the owner's death"
        )
        recovered = db.try_acquire_session_turn_lease(
            "s", _holder("recovery"), ttl_seconds=300
        )
        assert recovered is not None, (
            f"the conversation is wedged: its v26 owner (pid {owner.pid}) has "
            f"been reaped with returncode {returncode} and the lease does not "
            f"expire for an hour, so no signal frees it"
        )
        assert recovered.epoch == before[1] + 1, (
            f"a takeover must advance the generation so the dead owner's row "
            f"cannot be replayed: {before[1]} -> {recovered.epoch}"
        )
        assert db.append_message(
            "s", "assistant", "after recovery", turn_lease_holder=recovered
        )
        assert [m["content"] for m in db.get_messages("s")] == [
            "legacy context", "after recovery",
        ]
    finally:
        if db is not None:
            db.close()
        owner.close()


def check_a_migrated_row_naming_a_live_pid_is_treated_as_live(tmpdir) -> None:
    """A NARROW unit property, named for what it actually establishes.

    There is no v26 binary here and no owner. A process is started so that its
    pid is a live number, and a carried-over row is made to name it. What that
    shows is one link in the chain — the migration's recovered ``owner_pid`` is
    what liveness reads, and a live number is not reclaimable — with none of
    the process-level claims the base-owner pins above make.

    It is kept because it isolates that link: the base-owner pins would also
    pass if liveness were decided by something else about the child, and this
    one cannot.
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError
    from hermes_state_common import TURN_FENCE_TRIGGERS
    import sqlite3 as _sqlite3

    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    db = None
    try:
        path = pathlib.Path(tmpdir) / "state.db"
        seed = SessionDB(db_path=path)
        seed.create_session("s", source="test")
        seed.append_message("s", "user", "legacy context")
        seed.close()

        # Downgrade to the pre-epoch shape the way the published rollback does:
        # drop the derived trigger surface, then rebuild the table. The row is
        # then written by a connection that registered nothing, which is what
        # an old binary is.
        conn = _sqlite3.connect(str(path))
        try:
            for trigger in TURN_FENCE_TRIGGERS:
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP TABLE session_turn_leases")
            conn.execute(
                "CREATE TABLE session_turn_leases ("
                "conversation_id TEXT PRIMARY KEY, holder TEXT NOT NULL, "
                "acquired_at REAL NOT NULL, expires_at REAL NOT NULL)"
            )
            now = time.time()
            expires_at = now + 0.25
            conn.execute(
                "INSERT INTO session_turn_leases VALUES (?, ?, ?, ?)",
                ("s", f"pid={alive.pid}:turn=legacy:platform=telegram",
                 now, expires_at),
            )
            conn.commit()
        finally:
            conn.close()

        db = SessionDB(db_path=path)
        before = _lease_values(db)
        assert before is not None and before[2] == alive.pid, (
            f"owner_pid was not recovered from the holder string: {before!r}"
        )
        _wait_out_the_deadline(before[4])
        assert alive.poll() is None, (
            "the pid stopped being live before the deadline passed, so this "
            "check is measuring a dead one"
        )

        assert db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        ) is None
        assert _lease_values(db) == before
        try:
            db.append_message("s", "assistant", "stolen")
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError("a holderless write landed on a held row")
        assert [m["content"] for m in db.get_messages("s")] == ["legacy context"]
    finally:
        if db is not None:
            db.close()
        alive.terminate()
        alive.wait(timeout=_LIVENESS_S)


def check_a_same_pid_live_owner_is_not_stolen_when_the_ttl_expires(tmpdir) -> None:
    """The case a subprocess cannot exercise, on a store a v26 binary made.

    The base owner is killed and reaped first, so its row is reclaimable; THIS
    process then becomes the real owner, the generation continues from the
    legacy 0 to 1, and the deadline passes while the grant is still held here.
    ``_turn_lease_owner_is_dead`` answers "not dead" for our own pid — correctly,
    the process is right here — so same-pid admission falls through to whatever
    else the predicate consults, and the only correct answer is the live-grant
    registry.
    """
    from hermes_state import SessionDB

    owner = _start_base_owner(tmpdir, ttl_seconds=3600.0)
    db = None
    try:
        owner.ping()
        owner.kill_and_reap()
        db = SessionDB(db_path=owner.store)
        assert _lease_values(db)[1] == 0, "the migration did not carry the row"

        held = db.try_acquire_session_turn_lease(
            "s", _holder("owner"), ttl_seconds=0.3
        )
        assert held is not None, (
            "a carried-over row whose owner has been reaped should be "
            "takeable; if it is not, the migration wedged the store"
        )
        assert held.epoch == 1, (
            f"the generation must continue from the legacy 0: got {held.epoch}"
        )
        before = _lease_values(db)
        _wait_out_the_deadline(before[4])

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
        assert db.append_message(
            "s", "assistant", "owner write", turn_lease_holder=held
        )
        assert [m["content"] for m in db.get_messages("s")] == [
            "legacy context", "owner write",
        ]
    finally:
        if db is not None:
            db.close()
        owner.close()


def check_an_unknown_legacy_holder_needs_force_release_not_the_clock(
    tmpdir,
) -> None:
    """A holder no evidence can reach stays held, and force-release is the exit.

    A pre-``owner_pid`` row whose holder string is not ``pid=<n>`` has no
    identity at all: liveness is UNKNOWN, which is not liveness-DEAD, so
    nothing frees it. That is the largest cost of taking the clock away and it
    is paid deliberately — a wedged conversation is recoverable, an interleaved
    transcript is not. The exit therefore has to exist and has to be the only
    one.
    """
    from hermes_state import SessionDB, SessionTurnLeaseLostError
    from hermes_state_common import TURN_FENCE_TRIGGERS
    import sqlite3 as _sqlite3

    db = None
    path = pathlib.Path(tmpdir) / "state.db"
    seed = SessionDB(db_path=path)
    seed.create_session("s", source="test")
    seed.append_message("s", "user", "legacy context")
    seed.close()

    conn = _sqlite3.connect(str(path))
    try:
        for trigger in TURN_FENCE_TRIGGERS:
            conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
        conn.execute("DROP TABLE session_turn_leases")
        conn.execute(
            "CREATE TABLE session_turn_leases ("
            "conversation_id TEXT PRIMARY KEY, holder TEXT NOT NULL, "
            "acquired_at REAL NOT NULL, expires_at REAL NOT NULL)"
        )
        now = time.time()
        expires_at = now + 0.25
        conn.execute(
            "INSERT INTO session_turn_leases VALUES (?, ?, ?, ?)",
            ("s", "gateway-writer-no-identity", now, expires_at),
        )
        conn.commit()
    finally:
        conn.close()

    try:
        db = SessionDB(db_path=path)
        before = _lease_values(db)
        assert before is not None and before[2] is None, (
            f"this check needs a row with NO recorded identity; got {before!r}"
        )
        _wait_out_the_deadline(before[4])

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
    "check_the_base_owner_keeps_its_lease_when_the_ttl_expires":
        check_the_base_owner_keeps_its_lease_when_the_ttl_expires,
    "check_the_base_owner_is_taken_over_only_once_it_is_provably_dead":
        check_the_base_owner_is_taken_over_only_once_it_is_provably_dead,
    "check_a_migrated_row_naming_a_live_pid_is_treated_as_live":
        check_a_migrated_row_naming_a_live_pid_is_treated_as_live,
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
# Every row edits the code that DECIDES WHETHER A CONTENDER MAY TAKE A HELD
# LEASE — `_turn_lease_row_is_free` and the liveness predicate it calls — or the
# migration statement that gives that code something to decide from. None of
# them edits a fixture or a claim, because a row that only dies when the test's
# own scaffolding changes is measuring the scaffolding.
#
# For the no-steal rows the guard being removed is unusual: it is the ABSENCE of
# a clock in the admission predicate. So the mutation puts the old rule back —
# "expired ⇒ reclaimable" — which is exactly the contract this work replaced.
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
        pin="check_the_base_owner_keeps_its_lease_when_the_ttl_expires",
        module="hermes_state.py",
        find=_CLOCK_ANCHOR,
        replace=_CLOCK_MUTANT,
        why="this is the seam that decides whether a contender may take a held "
            "lease. Restoring 'expired ⇒ reclaimable' hands the conversation "
            "away while its v26 owner is still answering",
    ),
    Mutation(
        pin="check_the_base_owner_keeps_its_lease_when_the_ttl_expires",
        module="hermes_state.py",
        find="        if owner_pid_start is None:\n            return False\n",
        replace="        if owner_pid_start is None:\n            return True\n",
        why="a carried-over row has NO recorded start time, so treating "
            "'unknown start' as proof of death declares every migrated owner "
            "dead at the moment of upgrade",
    ),
    Mutation(
        pin="check_the_base_owner_is_taken_over_only_once_it_is_provably_dead",
        module="hermes_state.py",
        find="            if not psutil.pid_exists(pid):\n                return True\n",
        replace="            if not psutil.pid_exists(pid):\n                return False\n",
        why="the kernel's answer about the reaped owner is the ONLY evidence "
            "available here — the lease does not expire for an hour — so "
            "without it the conversation is wedged forever",
    ),
    Mutation(
        pin="check_a_migrated_row_naming_a_live_pid_is_treated_as_live",
        module="hermes_state_schema.py",
        find="            owner_pid = int(match.group(1)) if match else None\n",
        replace="            owner_pid = None\n",
        why="without recovering the pid from the v26 holder string the carried "
            "row has no identity, so liveness has nothing to read and the "
            "narrow property this pin isolates does not exist",
    ),
    Mutation(
        pin="check_a_migrated_row_naming_a_live_pid_is_treated_as_live",
        module="hermes_state_schema.py",
        find="        for row in carried:\n",
        replace="        for row in []:\n",
        why="dropping the rows is the easy migration and the wrong one: at "
            "upgrade time a live owner's conversation becomes acquirable by "
            "anyone",
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
        why="restoring the old rule frees a holder nothing can prove anything "
            "about, which is precisely the case force_release exists for",
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
