"""The crash matrix: real processes, real kills, and a real handover.

WHY THIS IS NOT THE EXISTING PROCESS-BARRIER FILE
    ``test_turn_lease_process_barriers`` establishes that two processes cannot
    hold one conversation, with ``multiprocessing`` Events carrying every
    ordering fact. It stops there: both of its children exit normally, and the
    only handover it exercises is a clean release.

    Nothing exercised a CRASH. That is the case the whole liveness contract was
    rewritten for — the old rule healed a crashed owner after one TTL, the new
    rule refuses to heal on a clock and heals on evidence instead — and the
    evidence path had never been run against a process that actually died.

    It is also the case where the two possible explanations are easiest to
    confuse. A test that both expires the lease AND kills the owner cannot say
    which one freed it. So the matrix separates them:

        owner alive, deadline ahead      contender refused
        owner alive, deadline PASSED     contender refused   <- not the clock
        owner KILLED, deadline ahead     contender admitted  <- the evidence
        owner killed, restart            new generation, and it writes

WHY subprocess AND NOT multiprocessing
    Two reasons, and the second is the load-bearing one.

    A ``spawn`` child re-imports its target by qualified module name. These
    pins also run inside the mutation harness, where the pin module is loaded
    by path as ``pins_under_mutation`` and exists in no importable location, so
    a spawn target cannot be resolved there at all. A program string has no
    such problem, and it is why the existing barrier file has no mutation
    table.

    And the barrier here is stronger than an Event. The child ANSWERS: the
    parent sends PING and reads a reply, so "the owner is inside its region" is
    something the owner said, not something the parent inferred. Every ordering
    fact is a round trip; ``_LIVENESS_S`` bounds them and its expiry is always
    a failure, never a pass.

DEATH IS A RETURNCODE
    The owner is SIGKILLed — no release, no cleanup, exactly the crash the
    contract exists for — and then REAPED, so "it is gone" is an observed exit
    status. Inferring death from a timeout is the same mistake as inferring it
    from a deadline.

CROSS-ROOT IDENTITY IS HERE TOO
    A grant carries ``(root, holder, epoch)``. Holder and epoch order
    acquisitions WITHIN one conversation and say nothing across them — a first
    grant is epoch 1 everywhere — so a check that gives two conversations
    different holders is refused by the holder comparison and never exercises
    the root at all. That mistake has already been made once on this branch.
    The check below uses ONE holder string and ONE epoch, and it covers the two
    surfaces the compression-lineage pins do not: the transcript write, and
    release.
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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: Liveness bound only. Every use is asserted to have been reached; an expiry
#: is reported as "the child never got there", never as a passing observation.
_LIVENESS_S = 60.0


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


#: The owner program. It runs the tree under test, takes the lease and then
#: does only what it is told, one command per line, answering each one.
_OWNER_PROGRAM = textwrap.dedent(
    """
    import json, os, pathlib, sys

    tree = pathlib.Path(sys.argv[1]).resolve()
    store = pathlib.Path(sys.argv[2])
    ttl = float(sys.argv[3])
    tag = sys.argv[4]
    sys.path.insert(0, str(tree))

    import hermes_state
    loaded = pathlib.Path(hermes_state.__file__).resolve()
    if not loaded.is_relative_to(tree):
        raise SystemExit("the owner imported %s, not the tree under test" % loaded)

    db = hermes_state.SessionDB(db_path=store)
    if db.get_session("s") is None:
        db.create_session("s", source="test")
        db.append_message("s", "user", "seed")
    holder = hermes_state.make_turn_lease_holder(tag)
    token = db.try_acquire_session_turn_lease("s", holder, ttl_seconds=ttl)
    print(json.dumps({
        "ready": True,
        "pid": os.getpid(),
        "module": str(loaded),
        "holder": holder,
        "acquired": token is not None,
        "epoch": None if token is None else token.epoch,
    }), flush=True)

    for line in sys.stdin:
        command = line.strip()
        if command == "PING":
            print(json.dumps({"alive": True, "pid": os.getpid()}), flush=True)
        elif command.startswith("WRITE "):
            db.append_message(
                "s", "assistant", command[6:], turn_lease_holder=token
            )
            print(json.dumps({"wrote": command[6:]}), flush=True)
        elif command == "EXIT":
            break
    """
)


class _Owner:
    """A live process holding the turn lease, answering on a pipe."""

    def __init__(self, process, ready: dict, errors: pathlib.Path):
        self.process = process
        self.ready = ready
        self.errors = errors

    @property
    def pid(self) -> int:
        return int(self.ready["pid"])

    @property
    def holder(self) -> str:
        return str(self.ready["holder"])

    def _command(self, text: str) -> dict:
        assert self.process.poll() is None, (
            f"the owner exited with {self.process.returncode} before being "
            f"asked {text!r}:\n{self.errors.read_text(errors='replace')}"
        )
        self.process.stdin.write(f"{text}\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        assert line, (
            f"the owner closed its output instead of answering {text!r}:\n"
            f"{self.errors.read_text(errors='replace')}"
        )
        return json.loads(line)

    def ping(self) -> None:
        """Alive because it SAID so, not because a clock has not run out."""
        answer = self._command("PING")
        assert answer.get("alive") is True and answer.get("pid") == self.pid

    def write(self, text: str) -> None:
        assert self._command(f"WRITE {text}").get("wrote") == text

    def kill_and_reap(self) -> int:
        """SIGKILL: no release, no cleanup. Then observe the exit status."""
        self.process.kill()
        returncode = self.process.wait(timeout=_LIVENESS_S)
        assert returncode is not None, "the owner was never reaped"
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


def _tree_under_test() -> pathlib.Path:
    """The tree the imported ``hermes_state`` came from.

    Derived from the loaded module rather than from ``REPO_ROOT`` so that under
    the mutation harness the child runs the MUTATED copy — a child that quietly
    ran the pristine repository would make every mutation row pass.
    """
    import hermes_state

    return pathlib.Path(hermes_state.__file__).resolve().parent


def _start_owner(tmpdir, *, ttl_seconds: float, tag: str = "owner") -> _Owner:
    tmpdir = pathlib.Path(tmpdir)
    store = tmpdir / "state.db"
    errors = tmpdir / f"{tag}.stderr"
    tree = _tree_under_test()

    with open(errors, "w") as stderr:
        process = subprocess.Popen(
            [
                sys.executable, "-c", _OWNER_PROGRAM,
                str(tree), str(store), str(ttl_seconds), tag,
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
            f"the owner produced no READY line and exited {process.returncode}. "
            f"This is an ERROR — nothing was measured.\n"
            f"{errors.read_text(errors='replace')}"
        )
    ready = json.loads(line)
    assert ready.get("acquired") is True, (
        f"the owner process could not take the lease: {ready!r}"
    )
    return _Owner(process, ready, errors)


def _store(tmpdir):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / "state.db")


def _lease(db, conversation_id="s"):
    row = db.get_session_turn_lease(conversation_id)
    if row is None:
        return None
    return (row["holder"], int(row["epoch"]), row["owner_pid"],
            float(row["expires_at"]))


def _wait_out_the_deadline(expires_at: float) -> None:
    deadline = time.monotonic() + _LIVENESS_S
    while time.time() <= expires_at:
        assert time.monotonic() < deadline, (
            "the lease deadline never passed, so no expiry was exercised"
        )
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_a_live_owner_process_survives_the_contender_and_its_deadline(
    tmpdir,
) -> None:
    """Two rows of the matrix: alive-and-fresh, and alive-and-expired.

    The second is the one the contract turns on. The deadline passing says the
    refresher did not run — a starved thread, a stopped-world GC, a laptop that
    slept — and the owner answering a PING one line later says it is still
    there. Reclaiming on the first fact while the second is true is how one
    conversation gets two writers.
    """
    owner = _start_owner(tmpdir, ttl_seconds=0.5)
    db = None
    try:
        db = _store(tmpdir)
        before = _lease(db)
        assert before is not None and before[2] == owner.pid, before
        assert before[3] > time.time(), "the lease was already expired"

        # Row 1: alive, deadline ahead.
        owner.ping()
        assert db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        ) is None, "a contender took a conversation from a live owner"
        assert _lease(db) == before

        # Row 2: alive, deadline PASSED.
        _wait_out_the_deadline(before[3])
        owner.ping()
        assert db.try_acquire_session_turn_lease(
            "s", _holder("contender"), ttl_seconds=300
        ) is None, (
            f"a contender took the conversation because the deadline had "
            f"passed, while the owner (pid {owner.pid}) was still answering"
        )
        assert _lease(db) == before, (
            f"the lease row moved under a refused contender: {_lease(db)!r} != "
            f"{before!r}"
        )

        # And the owner is unaffected: its own grant still writes.
        owner.write("owner still writing")
        assert [m["content"] for m in db.get_messages("s")] == [
            "seed", "owner still writing",
        ]
        owner.ping()
    finally:
        if db is not None:
            db.close()
        owner.close()


def check_a_sigkilled_owner_is_recovered_by_a_restart_not_by_the_clock(
    tmpdir,
) -> None:
    """The other two rows: killed-and-reaped, then restarted.

    The lease is taken with an hour of TTL and the deadline is asserted to be
    in the FUTURE at every step, so "it expired" is never available as an
    explanation for the recovery. The owner is SIGKILLed — no release, no
    cleanup — and reaped, and only then is the conversation takeable.

    The restart is a second REAL process, not this one calling acquire: a
    crashed gateway comes back as a new process, and what it has to be able to
    do is take the conversation and write to it.
    """
    owner = _start_owner(tmpdir, ttl_seconds=3600.0, tag="owner")
    restarted = None
    db = None
    try:
        db = _store(tmpdir)
        before = _lease(db)
        assert before is not None and before[2] == owner.pid
        assert before[3] > time.time()
        owner.ping()
        owner.write("last words")

        # Alive: refused, with an hour left on the clock.
        assert db.try_acquire_session_turn_lease(
            "s", _holder("early"), ttl_seconds=300
        ) is None
        assert _lease(db) == before

        returncode = owner.kill_and_reap()
        assert returncode is not None
        assert before[3] > time.time(), (
            "the deadline passed during the kill, so this check can no longer "
            "attribute the recovery to the owner's death"
        )

        # Dead: the restart takes it, at a strictly later generation.
        restarted = _start_owner(tmpdir, ttl_seconds=3600.0, tag="restarted")
        after = _lease(db)
        assert after is not None, "the restart left no lease row"
        assert after[2] == restarted.pid, (
            f"the lease still names the killed owner: {after!r}"
        )
        assert after[1] == before[1] + 1, (
            f"a takeover after a crash must advance the generation so the dead "
            f"owner's grant cannot be replayed: {before[1]} -> {after[1]}"
        )
        assert restarted.ready["epoch"] == after[1]

        restarted.write("after the crash")
        assert [m["content"] for m in db.get_messages("s")] == [
            "seed", "last words", "after the crash",
        ], "the restarted owner could not write the conversation it took"
        restarted.ping()
    finally:
        if db is not None:
            db.close()
        if restarted is not None:
            restarted.close()
        owner.close()


def check_a_cross_root_grant_can_neither_write_nor_release(tmpdir) -> None:
    """One holder, one epoch, two conversations — only the root can tell them apart.

    Both grants are the first in their conversation, so both are epoch 1, and
    both are composed by the same process for the same purpose, so both are the
    same string. If the root were ignored, a holder that legitimately owns A
    could append to B and could free B's lease, and neither the holder
    comparison nor the epoch comparison would notice.

    Release is the sharper half: it is a destructive authority transfer, so a
    misdirected one does not merely write in the wrong place, it makes somebody
    else's live conversation acquirable by anyone.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        for sid in ("a", "b"):
            db.create_session(sid, "test")
            db.append_message(sid, "user", f"{sid} context")
        one_identity = _holder("shared")
        token_a = db.try_acquire_session_turn_lease(
            "a", one_identity, ttl_seconds=600
        )
        token_b = db.try_acquire_session_turn_lease(
            "b", one_identity, ttl_seconds=600
        )
        assert token_a and token_b
        assert str(token_a) == str(token_b), (
            "the two grants must be the same holder string, or the holder "
            "comparison refuses everything and the ROOT is never exercised"
        )
        assert token_a.epoch == token_b.epoch == 1, (
            "a first grant is epoch 1 in every conversation — that is the point"
        )
        before_a, before_b = _lease(db, "a"), _lease(db, "b")

        # Write.
        try:
            db.append_message(
                "b", "assistant", "cross-root", turn_lease_holder=token_a
            )
        except SessionTurnLeaseLostError:
            pass
        else:
            raise AssertionError("a grant for conversation a appended to b")
        assert [m["content"] for m in db.get_messages("b")] == ["b context"]
        assert [m["content"] for m in db.get_messages("a")] == ["a context"]

        # Release.
        db.release_session_turn_lease("b", token_a)
        assert _lease(db, "b") == before_b, (
            f"a grant for conversation a freed b's lease: {_lease(db, 'b')!r} "
            f"!= {before_b!r}. Nothing owns b now, and anyone may take it"
        )
        assert _lease(db, "a") == before_a, (
            "the misdirected release also disturbed the grant's own lease"
        )

        # Each owner is unaffected.
        assert db.append_message(
            "b", "assistant", "b owner", turn_lease_holder=token_b
        )
        assert db.append_message(
            "a", "assistant", "a owner", turn_lease_holder=token_a
        )
        assert [m["content"] for m in db.get_messages("b")] == [
            "b context", "b owner",
        ]
    finally:
        db.close()


PINS = {
    "check_a_live_owner_process_survives_the_contender_and_its_deadline":
        check_a_live_owner_process_survives_the_contender_and_its_deadline,
    "check_a_sigkilled_owner_is_recovered_by_a_restart_not_by_the_clock":
        check_a_sigkilled_owner_is_recovered_by_a_restart_not_by_the_clock,
    "check_a_cross_root_grant_can_neither_write_nor_release":
        check_a_cross_root_grant_can_neither_write_nor_release,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_crash_matrix_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_live_owner_process_survives_the_contender_and_its_deadline",
        module="hermes_state.py",
        find="        if pid <= 0 or pid == os.getpid() or psutil is None:\n"
             "            return False\n",
        replace="        if pid <= 0 or pid == os.getpid() or psutil is None:\n"
                "            return True\n",
        why="this is the conservative half of the liveness predicate — "
            "'cannot decide' answers not-dead. Flipping it declares every "
            "owner this host cannot probe to be gone, and the contender walks "
            "in while the owner is still answering",
    ),
    Mutation(
        pin="check_a_live_owner_process_survives_the_contender_and_its_deadline",
        module="hermes_state.py",
        find='        if row is None:\n            return True\n'
             '        holder = row["holder"] or ""\n',
        replace='        if row is None:\n            return True\n'
                '        if float(row["expires_at"] or 0) <= time.time():\n'
                '            return True\n'
                '        holder = row["holder"] or ""\n',
        why="restores the rule the contract replaced — expired means "
            "reclaimable — which is exactly the second row of the matrix: the "
            "owner is answering and the deadline has passed",
    ),
    Mutation(
        pin="check_a_sigkilled_owner_is_recovered_by_a_restart_not_by_the_clock",
        module="hermes_state.py",
        find="            if not psutil.pid_exists(pid):\n                return True\n",
        replace="            if not psutil.pid_exists(pid):\n                return False\n",
        why="the kernel's answer about the reaped owner is the ONLY evidence "
            "available — the lease has an hour left — so without it a crashed "
            "gateway wedges its conversation until a human intervenes",
    ),
    Mutation(
        pin="check_a_sigkilled_owner_is_recovered_by_a_restart_not_by_the_clock",
        module="hermes_state.py",
        find='            epoch = int(row["epoch"] or 0) + 1\n',
        replace='            epoch = int(row["epoch"] or 0)\n',
        why="recovering a crashed owner's conversation without advancing the "
            "generation leaves the dead owner's grant valid; anything that "
            "still holds it can write into the restarted turn",
    ),
    Mutation(
        pin="check_a_cross_root_grant_can_neither_write_nor_release",
        module="hermes_state.py",
        find="        if granted_root != conversation_id:\n            return None\n",
        replace="        if False:\n            return None\n",
        why="the root comparison is the only cross-conversation check; with it "
            "gone the same holder at the same epoch writes into, and frees, a "
            "conversation it never owned",
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
