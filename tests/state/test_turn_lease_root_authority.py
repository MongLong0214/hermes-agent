"""The canonical lease ROOT is the authority, on every path that names one.

Three surfaces resolve "which conversation is this" from a mutable session id
where the answer is already known — and each one of them gets a different
answer than the grant it is checking against.

B1 — THE RELEASE PATH FREES A CONVERSATION IT NEVER NAMES, OR NOTHING AT ALL
    :meth:`SessionDB.session_turn_lease` acquires against ``root(session_id)``
    and stamps that root into the token. Its ``finally`` hands the CALLER'S id
    back to :meth:`release_session_turn_lease`, which re-derives the root from
    it. When the body deleted that row — which is exactly what both production
    delete callers do — ``_session_turn_lease_key_on_conn`` returns the unknown
    id unchanged, the derived root is no longer the granted root, and the
    release is skipped behind a ``logger.debug``.

    The conversation is then held forever. Not for one TTL: this same design
    deliberately removed the clock from ``_turn_lease_row_is_free``, so a missed
    release went from a self-healing blip to a permanent wedge that only
    ``force_release_session_turn_lease`` clears — and that has no CLI verb.

    The process that leaked it cannot notice. Its own registry entry was
    unregistered, so ``_turn_lease_row_is_free`` takes the "ours, unheld" branch
    and reports the row FREE to the leaker while every other process reads the
    live ``owner_pid`` and is refused. Which is why the check below asks a
    SECOND PROCESS. A same-process re-acquire passes against the defect.

    ``_authorize_turn_lease_token``'s own docstring already states the rule the
    release path breaks: *"the grant's root is immutable because it was stamped
    when the lease was issued."* The token carries it. The release ignored it.

B2 — FOUR FLAG WRITERS GUARD ONE CONVERSATION AND WRITE A DIFFERENT ONE
    ``set_session_archived`` / ``_pinned`` / ``_hidden`` / ``_read`` admit
    through ``_check_session_flag_write``, which checks the NAMED session. The
    ``WITH RECURSIVE descendants`` walk they then run descends on
    ``parent.end_reason = 'compression'`` alone. A ``/branch`` child of a
    compression parent satisfies that join and is NOT a continuation — it is its
    own lease root (``_is_explicit_fork_child_row``), a separate conversation
    that the named grant never covered and that the guard therefore never saw.

    So naming the parent writes a flag into a conversation another process is
    mid-turn on, with no refusal anywhere, while naming that child directly is
    refused. The control leg below asserts both halves, because "it wrote" and
    "it may not write" are only the same property when the refusal still works.

    ``archived`` is not cosmetic here: ``prune_sessions`` and
    ``count_empty_sessions`` read it as a DO-NOT-COLLECT marker, which
    ``_check_session_flag_write`` says in its own docstring is why these writers
    are fenced at all. Clearing it on a live conversation hands that
    conversation to the next sweep.

B3 — THE MULTI-TARGET ADMITTERS CLASSIFY BY ID WHERE THE RULE IS ROOTS
    ``_admit_on_connection`` and ``_admit_routing_write`` find the ONE id whose
    root matches the grant, then send ``[sid for sid in ids if sid != named]``
    to the freeness check. Every OTHER id in the same conversation is a
    different string, so the caller's own grant is checked against it as though
    it were a bystander's — and the conversation is owned, by the caller, so it
    refuses.

    ``_affected_session_ids`` warns about this in the file already: *"a caller
    that wants 'everything except what my grant covers' filters by root, because
    id equality is the wrong comparison for a lineage."*
    ``_refuse_unless_reached_conversations_are_free`` does filter by root. These
    two do not.

    This is NOT latent at the routing site. ``save_gateway_routing_entry`` is a
    shipped public method and its ordinary compression-rotation write — the key
    pointed at the root, the turn repoints it to the continuation — sends both
    ids in at once and is refused against the grant it is holding.

WHY THE PINS ARE MODULE-LEVEL FUNCTIONS
    The mutation harness re-runs the same assertions in a subprocess against a
    MUTATED copy of the tree, so each property has to be callable there. A pin
    and the thing being mutation-tested cannot drift apart, because they are the
    same function.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _compression_chain(db, parent="A", child="B"):
    """*parent* compressed into *child*: one conversation, root ``parent``."""
    db.create_session(parent, "test")
    db.append_message(parent, "user", f"{parent} context")
    db.end_session(parent, "compression")
    db.create_session(child, "test", parent_session_id=parent)
    db.append_message(child, "user", f"{child} context")
    assert db._session_turn_lease_key(child) == parent, (
        "the fixture is not exercising a compression chain: the continuation "
        "is its own lease root, so the caller's id and the grant's root would "
        "agree and no path under test could diverge"
    )


def _branch_child(db, parent: str, child: str) -> None:
    """A ``/branch`` child of *parent*: a SEPARATE conversation, its own root.

    ``model_config`` is handed over as a dict, not as JSON text. The insert
    serializes it, so a pre-serialized string is stored double-encoded, the
    marker never parses, and the child silently resolves to its parent's root —
    a fixture that reports the gap closed because it never opened it.
    """
    db.create_session(
        child, "test", parent_session_id=parent,
        model_config={"_branched_from": parent},
    )
    db.append_message(child, "user", f"{child} context")
    assert db._session_turn_lease_key(child) == child, (
        "the fixture is not exercising the gap: the branch child resolves to "
        "its parent's root, so the parent's own guard already covers it"
    )


def _lease_row(db, session_id):
    row = db.get_session_turn_lease(session_id)
    return None if row is None else dict(row)


#: The four lineage flag writers, as (name, setter, "is it marked" reader).
#:
#: ``read`` is included deliberately even though it is not a 0/1 column —
#: ``set_session_read`` stamps ``last_read_at`` (a timestamp, or 0 for
#: explicitly-unread) through the SAME recursive walk as the other three. A
#: table that skipped it because its column has a different shape would leave
#: one of the four writers uncovered while reporting the family done.
_FLAG_WRITERS = (
    ("archived", "set_session_archived", "archived"),
    ("pinned", "set_session_pinned", "pinned"),
    ("hidden", "set_session_hidden", "hidden"),
    ("read", "set_session_read", "last_read_at"),
)


def _marked(db, session_id, column) -> bool:
    """Whether *column* is set on *session_id*, normalised across the four."""
    row = db.get_session(session_id)
    assert row is not None, f"no session row for {session_id!r}"
    return bool(row[column])


def _acquire_from_another_process(db_path, session_id) -> str:
    """Take the lease on *session_id* from a SEPARATE process.

    The whole point of the B1 pin. A leaked row names THIS process, and
    ``_turn_lease_row_is_free`` answers "ours, unheld → free" for the leaker
    once its registry entry is gone. Asking here would pass against the defect;
    only a foreign pid reads the row the way every other writer does.

    The interpreter and the import path are taken from the LOADED module rather
    than from the environment, so this works identically under pytest and inside
    the mutation harness's extracted tree.
    """
    import hermes_state

    tree = str(pathlib.Path(hermes_state.__file__).resolve().parent)
    probe = (
        "import pathlib, os, sys\n"
        f"sys.path.insert(0, {tree!r})\n"
        "from hermes_state import SessionDB\n"
        f"db = SessionDB(db_path=pathlib.Path({str(db_path)!r}))\n"
        "grant = db.try_acquire_session_turn_lease(\n"
        f"    {session_id!r}, 'pid=%d:turn=other:platform=test' % os.getpid(),\n"
        "    ttl_seconds=60,\n"
        ")\n"
        "print('ACQUIRED' if grant else 'REFUSED')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = tree
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    run = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=120, env=env,
    )
    assert run.returncode == 0, (
        "the second-process probe crashed, so this check measured nothing "
        f"about the lease:\n{run.stdout}\n{run.stderr}"
    )
    assert "ACQUIRED" in run.stdout or "REFUSED" in run.stdout, (
        f"the second-process probe printed no verdict:\n{run.stdout}\n{run.stderr}"
    )
    return "ACQUIRED" if "ACQUIRED" in run.stdout else "REFUSED"


def _delete_through_the_production_path(db, session_id, sessions_dir=None):
    """The two lines both shipped delete callers run, verbatim.

    ``hermes_cli/web_routers/sessions.py`` (``DELETE /api/sessions/{id}``) and
    ``hermes_cli/sessions_cmd.py`` (``hermes sessions delete``) both open the
    scope on the id the operator named and pass ``lease.token`` into the delete.
    Entering anywhere else — calling ``release_session_turn_lease`` directly,
    say — would be a check at a layer production does not use.
    """
    from hermes_state import make_turn_lease_holder

    with db.session_turn_lease(
        session_id,
        make_turn_lease_holder("sessions-delete"),
        ttl_seconds=30.0,
        reload_messages=False,
    ) as lease:
        return db.delete_session(
            session_id,
            sessions_dir=sessions_dir,
            turn_lease_holder=lease.token,
        )


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_deleting_a_continuation_tip_leaves_the_conversation_acquirable(
    tmpdir,
) -> None:
    """B1. The delete both shipped callers perform must release what it took.

    The tip is the id the desktop and every listing show for a compressed
    conversation, so a non-root id here is the ROUTINE case, not an edge one.
    """
    db = _store(tmpdir)
    try:
        _compression_chain(db, "A", "B")
        assert _delete_through_the_production_path(db, "B") is True, (
            "the delete itself did not happen, so nothing below is about the "
            "release path"
        )

        row = _lease_row(db, "A")
        assert row is not None, "no lease row on the root at all"
        assert row["holder"] == "", (
            "the turn lease on the conversation root was NOT released by the "
            "delete that took it. The scope acquired against root 'A' and "
            "stamped it into the token, then released against the caller's id "
            "'B' — whose row the body had just deleted, so the root re-derived "
            "from it is 'B', the grant does not authorize 'B', and the release "
            "was skipped. There is no clock left in _turn_lease_row_is_free, so "
            "this row never frees itself.\n"
            f"  lease row still held by: {row['holder']!r} (epoch {row['epoch']})"
        )

        verdict = _acquire_from_another_process(db.db_path, "A")
        assert verdict == "ACQUIRED", (
            "another process cannot take the conversation after the delete "
            "that leaked it. The row names this still-live process, so "
            "_turn_lease_owner_is_dead answers False and the row is never free "
            "for anyone else — permanently, since force_release_session_turn_"
            "lease has no CLI verb to run. Every later turn on this "
            "conversation waits out its whole budget and then refuses."
        )
    finally:
        db.close()


def check_a_lineage_flag_write_leaves_an_owned_branch_child_alone(
    tmpdir,
) -> None:
    """B2. All four lineage flag writers, and the refusal that proves the guard.

    Every one of the four is exercised: they carry character-identical walks,
    so a check that drove only ``archived`` would report the other three covered
    on the strength of them looking similar.
    """
    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        _compression_chain(db, "A", "B")
        _branch_child(db, "A", "C")

        for name, setter, column in _FLAG_WRITERS:
            getattr(db, setter)("C", True)
            assert _marked(db, "C", column), (
                f"the fixture never set {name!r} on the branch child, so the "
                f"clear below could not have shown anything"
            )

        grant = db.try_acquire_session_turn_lease(
            "C", _holder("owner-of-the-branch"), ttl_seconds=600
        )
        assert grant, "could not take the lease on the branch child"
        assert getattr(grant, "conversation_id", None) == "C", (
            "the branch child is not its own lease root in this fixture"
        )

        for name, setter, column in _FLAG_WRITERS:
            getattr(db, setter)("A", False)
            assert _marked(db, "C", column), (
                f"set_session_{name}('A', False) cleared {name!r} on branch "
                f"child 'C', a SEPARATE conversation that a live turn owns, and "
                f"refused nothing. _check_session_flag_write admitted the write "
                f"by checking the NAMED session 'A'; the recursive descendants "
                f"walk it guards then descended into 'C', because that walk "
                f"tests only parent.end_reason = 'compression' and never asks "
                f"whether the child is a continuation. The exclusion that "
                f"answers this already exists on the sibling walks in "
                f"record_gateway_session_peer and resolve_resume_session_id."
            )
            assert not _marked(db, "A", column), (
                f"set_session_{name}('A', False) did not clear {name!r} on the "
                f"conversation it named, so the walk is now excluding more than "
                f"the non-continuation children"
            )
            assert not _marked(db, "B", column), (
                f"set_session_{name}('A', False) did not reach continuation "
                f"'B'; the compression lineage must still be flipped as a unit"
            )

        # The control. Naming the branch child directly IS refused, which is
        # what makes the leg above a bypass rather than a missing fence.
        for name, setter, column in _FLAG_WRITERS:
            with pytest.raises(SessionTurnLeaseLostError):
                getattr(db, setter)("C", False)
            assert _marked(db, "C", column), (
                f"the refused direct write on 'C' still changed {name!r}"
            )
    finally:
        db.close()


def check_a_routing_rotation_within_the_granted_root_is_admitted(
    tmpdir,
) -> None:
    """B3. Same-root ids are the grant's own conversation, not bystanders.

    Driven through ``save_gateway_routing_entry`` — a shipped public writer —
    on its ordinary compression-rotation write, and through the
    ``admit_on_connection`` public API on a foreign connection, which is the
    documented way an outside module borrows this rule.
    """
    import sqlite3

    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        _compression_chain(db, "A", "B")
        grant = db.try_acquire_session_turn_lease(
            "B", _holder("turn"), ttl_seconds=600
        )
        assert grant, "could not take the lease this check depends on"
        assert getattr(grant, "conversation_id", None) == "A", (
            "the grant was not stamped with the conversation root"
        )

        # The key currently routes to the root; the turn rotates it onto the
        # continuation. Both ids are this grant's own conversation.
        db.save_gateway_routing_entry(
            "chat-1", json.dumps({"session_id": "A"}),
            scope="sc", turn_lease_holder=grant,
        )
        try:
            db.save_gateway_routing_entry(
                "chat-1", json.dumps({"session_id": "B"}),
                scope="sc", turn_lease_holder=grant,
            )
        except SessionTurnLeaseLostError as exc:
            raise AssertionError(
                "the routing rotation was refused against the grant that owns "
                "BOTH ids. The overwritten entry names 'A' and the incoming "
                "entry names 'B'; root('A') == root('B') == 'A' == the grant's "
                "stamped root, so this is one conversation repointing its own "
                "key. _admit_routing_write picked one id as `named` and sent "
                "the other to the freeness check by STRING inequality, where it "
                "was refused for being owned — by this very caller.\n"
                f"  {exc}"
            ) from exc

        with db._read_ctx() as conn:
            routed = conn.execute(
                "SELECT entry_json FROM gateway_routing "
                "WHERE scope = ? AND session_key = ?", ("sc", "chat-1"),
            ).fetchone()
        assert json.loads(routed[0])["session_id"] == "B", (
            "the rotation was admitted but did not land"
        )

        # The borrowed-rule API, on a foreign connection, as documented.
        conn = sqlite3.connect(str(db.db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                db.admit_on_connection(
                    conn, ["A", "B"], because="probe",
                    turn_lease_holder=grant,
                )
            except SessionTurnLeaseLostError as exc:
                raise AssertionError(
                    "admit_on_connection refused a multi-id write whose ids "
                    "are ALL in the grant's own conversation. Its own contract "
                    "says 'the caller's grant for the conversation it names, "
                    "freeness for every other', and _affected_session_ids warns "
                    "in this same file that 'id equality is the wrong "
                    "comparison for a lineage'.\n"
                    f"  {exc}"
                ) from exc
            conn.rollback()
        finally:
            conn.close()

        # And the rule it must keep: a genuinely foreign owned root refuses.
        _branch_child(db, "A", "C")
        foreign = db.try_acquire_session_turn_lease(
            "C", _holder("owner-of-the-branch"), ttl_seconds=600
        )
        assert foreign, "could not take the lease on the branch child"
        conn = sqlite3.connect(str(db.db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            with pytest.raises(SessionTurnLeaseLostError):
                db.admit_on_connection(
                    conn, ["A", "C"], because="probe",
                    turn_lease_holder=grant,
                )
            conn.rollback()
        finally:
            conn.close()
    finally:
        db.close()


def check_the_three_root_confusions_compose(tmpdir) -> None:
    """One conversation, all three surfaces, in the order they are reached.

    Separately each defect reads as a local slip. Composed they are one rule
    being resolved from a mutable id three times: the delete leaks the root, so
    the conversation can never be handed over; the flag walk then reaches a
    branch child the leak never covered; and the admitters cannot even tell the
    leaked root apart from the ids inside it. This is the counterexample the
    three fixes have to close together — under any one of them it dies.
    """
    import sqlite3

    from hermes_state import SessionTurnLeaseLostError

    db = _store(tmpdir)
    try:
        # A ──compression──▶ B ──compression──▶ D   (one conversation, root A)
        #  └──/branch──▶ C                           (a second conversation)
        _compression_chain(db, "A", "B")
        db.end_session("B", "compression")
        db.create_session("D", "test", parent_session_id="B")
        db.append_message("D", "user", "D context")
        assert db._session_turn_lease_key("D") == "A"
        _branch_child(db, "A", "C")

        # B1 — the conversation survives the delete of its own tip as something
        # another process can still take. D is the id every listing shows.
        assert _delete_through_the_production_path(db, "D") is True
        assert _acquire_from_another_process(db.db_path, "A") == "ACQUIRED", (
            "step 1 of the composition failed: after the production delete of "
            "the continuation tip, no other process can take the conversation "
            "root. Every later step here would be asserting things about a "
            "conversation nothing can ever own again."
        )

        # Mark the branch child BEFORE anyone owns it — once the owner below
        # holds it, this write is refused, which is the property step 2 relies
        # on and not a way to set up its fixture.
        db.set_session_archived("C", True)

        # A live owner on the BRANCH child — a separate conversation hanging off
        # the very parent row the flag walk descends from.
        branch_owner = db.try_acquire_session_turn_lease(
            "C", _holder("owner-of-the-branch"), ttl_seconds=600
        )
        assert branch_owner, "could not take the lease on the branch child"

        # B2 — naming the root must not reach into that conversation.
        db.set_session_archived("A", False)
        assert _marked(db, "C", "archived"), (
            "step 2 of the composition failed: the flag write on the root "
            "reached the branch child a live turn owns"
        )

        # B3 — the root and the continuation inside it are ONE conversation to
        # the admitters; the branch child is not. Both halves under the grant
        # this process now holds on the root the delete handed back.
        own = db.try_acquire_session_turn_lease(
            "A", _holder("later-turn"), ttl_seconds=600
        )
        assert own is not None, (
            "step 3 of the composition failed: a later turn could not take the "
            "conversation root even though step 1 showed it acquirable"
        )
        conn = sqlite3.connect(str(db.db_path))
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                db.admit_on_connection(
                    conn, ["A", "B"], because="probe", turn_lease_holder=own,
                )
            except SessionTurnLeaseLostError as exc:
                raise AssertionError(
                    "step 3 of the composition failed: a multi-id write over "
                    "the root and its own continuation was refused against the "
                    "grant that owns both.\n"
                    f"  {exc}"
                ) from exc
            with pytest.raises(SessionTurnLeaseLostError):
                db.admit_on_connection(
                    conn, ["A", "C"], because="probe", turn_lease_holder=own,
                )
            conn.rollback()
        finally:
            conn.close()
    finally:
        db.close()


PINS = {
    "check_deleting_a_continuation_tip_leaves_the_conversation_acquirable":
        check_deleting_a_continuation_tip_leaves_the_conversation_acquirable,
    "check_a_lineage_flag_write_leaves_an_owned_branch_child_alone":
        check_a_lineage_flag_write_leaves_an_owned_branch_child_alone,
    "check_a_routing_rotation_within_the_granted_root_is_admitted":
        check_a_routing_rotation_within_the_granted_root_is_admitted,
    "check_the_three_root_confusions_compose":
        check_the_three_root_confusions_compose,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_root_authority_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_deleting_a_continuation_tip_leaves_the_conversation_acquirable",
        module="hermes_state.py",
        find=(
            '            self.release_session_turn_lease(\n'
            '                self._scope_release_target(session_id, token), token\n'
            '            )\n'
        ),
        replace='            self.release_session_turn_lease(session_id, token)\n',
        why="re-deriving the root from the caller's id is the defect: the body "
            "may have deleted that row, the derived root then differs from the "
            "stamped one, and the release is skipped behind a logger.debug — "
            "permanently, because _turn_lease_row_is_free no longer has a clock",
        kills_by="the turn lease on the conversation root was NOT released",
    ),
    Mutation(
        pin="check_a_lineage_flag_write_leaves_an_owned_branch_child_alone",
        module="hermes_state.py",
        find=(
            '    _CONTINUATION_DESCENDANT_FILTER_SQL = (\n'
            '        "      AND json_extract(COALESCE(child.model_config, \'{}\'),"\n'
            '        " \'$._branched_from\') IS NULL\\n"\n'
            '        "      AND json_extract(COALESCE(child.model_config, \'{}\'),"\n'
            '        " \'$._delegate_from\') IS NULL\\n"\n'
            '        "      AND COALESCE(child.source, \'\') != \'tool\'\\n"\n'
            '    )\n'
        ),
        replace='    _CONTINUATION_DESCENDANT_FILTER_SQL = ""\n',
        why="without the exclusion the four flag writers' descendants walk "
            "admits branch, delegate and tool children — separate conversations "
            "the guard on the named session never checked",
        kills_by="cleared 'archived' on branch child 'C'",
    ),
    Mutation(
        pin="check_a_routing_rotation_within_the_granted_root_is_admitted",
        module="hermes_state.py",
        find='        rest = [sid for sid in ids if roots[sid] != granted_root]\n',
        replace='        rest = [sid for sid in ids if sid != named]\n',
        why="id equality sends every OTHER id of the grant's own conversation "
            "to the freeness check, where it is refused for being owned by the "
            "caller doing the asking",
        kills_by="the routing rotation was refused against the grant that owns",
    ),
    Mutation(
        pin="check_the_three_root_confusions_compose",
        module="hermes_state.py",
        find=(
            '            self.release_session_turn_lease(\n'
            '                self._scope_release_target(session_id, token), token\n'
            '            )\n'
        ),
        replace='            self.release_session_turn_lease(session_id, token)\n',
        why="the composition is aimed at the head of the chain: with the "
            "release still re-derived from the caller's id, the conversation "
            "is wedged at step 1 and the two later properties are being "
            "asserted about a conversation nothing can ever own again",
        kills_by="step 1 of the composition failed",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    """Clean, mutated, restored. A pin that survives its mutation is not a pin."""
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path)


def test_every_pin_has_a_mutation_that_kills_it():
    """No pin without a killer, and no killer without a pin."""
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)


def test_every_mutation_row_names_the_assertion_it_dies_at():
    """A row without ``kills_by`` scores discrimination while discriminating none."""
    missing = [m.pin for m in SOURCE_MUTATIONS if not m.kills_by]
    assert not missing, (
        f"mutation rows with no target assertion: {missing}. Six pins in the "
        f"C5 table carry a crash detector as their own assert, so 'an "
        f"AssertionError fired' does not tell a kill from a crash."
    )
