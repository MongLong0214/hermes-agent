"""The rest of the `sessions` surface, and the routing index next to it.

WHY ONE FILE FOR MANY WRITERS
    The families before this one each closed a column pair with a named
    counterexample. What was left after them was not a hole with a story — it
    was the LONG TAIL: twenty-odd single-target mutators, three sweeps and four
    gateway-routing primitives, every one of which writes `sessions` (or the
    index that routes into it) with no admission at all.

    A long tail is where a fence dies, because each member looks individually
    harmless. The census's answer is the right one: the set is DERIVED, and a
    member leaves it only by reaching a fence form. So the pins here are
    REPRESENTATIVES — one per mechanism, each with a killer — and the
    derivation in ``test_turn_lease_writer_census`` /
    ``test_turn_lease_replay_column_writers`` is what keeps the twenty-odd from
    drifting back out.

WHAT EACH MECHANISM IS

    the cwd pair          ``update_session_cwd`` / ``publish_session_git_metadata``
                          move ``cwd`` / ``git_branch`` / ``git_repo_root``,
                          which the assembled prompt carries, and bump the
                          generation counter the publish path fences against.
    the routing identity  ``record_gateway_session_peer`` rewrites
                          ``session_key`` / ``chat_id`` / ``thread_id`` — and
                          with ``include_compression_ancestors`` it rewrites
                          them across the WHOLE lineage.
    the compression dials ``record_compression_failure_cooldown`` and its four
                          siblings decide whether the owner's next turn is
                          allowed to compress at all. A bystander arming the
                          cooldown wedges a conversation into context overflow.
    the activity labels   ``touch_session_activity`` / the label clear are what
                          the session list and the resume banner read.
    the handoff state     ``request_handoff`` / ``claim_handoff`` /
                          ``complete_handoff`` / ``fail_handoff`` decide which
                          platform drives the conversation next.
    the list flags        ``set_session_archived`` / ``pinned`` / ``hidden`` /
                          ``read`` and the title pair.
    the sweeps            ``backfill_repo_roots``, ``prune_empty_ghost_sessions``,
                          ``retag_kanban_worker_sessions`` — victims come from a
                          filter, so they SKIP rather than refuse.
    the routing index     ``save_gateway_routing_entry`` and its three siblings
                          key by ``session_key``, not ``session_id``: the
                          conversation they affect is named INSIDE
                          ``entry_json``, which is why a fence written against
                          the method's parameters would have found nothing to
                          check.

WHAT THE PINS ASSERT
    Rows and values, and each refusal paired with the same write LANDING —
    from the owner with its grant, or on the unaffected member of the same
    sweep.
"""

from __future__ import annotations

import json
import os
import pathlib
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


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _store(tmpdir, name="state.db"):
    from hermes_state import SessionDB

    return SessionDB(db_path=pathlib.Path(tmpdir) / name)


def _row(db, session_id="s"):
    return db.get_session(session_id) or {}


def _owned(db, session_id="s", *, tag="owner", **create):
    db.create_session(session_id, "test", **create)
    db.append_message(session_id, "user", f"{session_id} context")
    grant = db.try_acquire_session_turn_lease(
        session_id, _holder(tag), ttl_seconds=600
    )
    assert grant, f"could not take the lease on {session_id!r}"
    return grant


def _refused(call):
    from hermes_state import SessionTurnLeaseLostError

    try:
        result = call()
    except SessionTurnLeaseLostError:
        return
    raise AssertionError(
        f"the write was admitted while another writer held the "
        f"conversation's turn lease (returned {result!r})"
    )


# ---------------------------------------------------------------------------
# The pins.
# ---------------------------------------------------------------------------

def check_a_bystander_cannot_move_the_owners_cwd(tmpdir) -> None:
    """``cwd`` and the git pair ride in the assembled prompt."""
    db = _store(tmpdir)
    try:
        grant = _owned(db, cwd="/work/original")
        before = (_row(db)["cwd"], _row(db)["git_branch"])

        _refused(lambda: db.update_session_cwd("s", "/work/hijacked", "attacker"))
        assert (_row(db)["cwd"], _row(db)["git_branch"]) == before, (
            f"the refused write moved the working directory anyway: "
            f"{(_row(db)['cwd'], _row(db)['git_branch'])!r} != {before!r}"
        )

        generation = db.update_session_cwd(
            "s", "/work/moved", "main", turn_lease_holder=grant
        )
        assert generation is not None and _row(db)["cwd"] == "/work/moved", (
            "the owner's own cwd move was refused"
        )
    finally:
        db.close()


def check_a_bystander_cannot_publish_git_metadata_for_the_owner(tmpdir) -> None:
    """The generation-fenced publish is still a bystander write."""
    db = _store(tmpdir)
    try:
        grant = _owned(db, cwd="/work/original")
        generation = db.update_session_cwd(
            "s", "/work/original", "main", turn_lease_holder=grant
        )
        assert generation is not None
        before = _row(db)["git_branch"]

        _refused(
            lambda: db.publish_session_git_metadata(
                "s", "/work/original", generation, "attacker-branch"
            )
        )
        assert _row(db)["git_branch"] == before, (
            f"the refused publish moved the branch anyway: "
            f"{_row(db)['git_branch']!r} != {before!r}"
        )

        assert db.publish_session_git_metadata(
            "s", "/work/original", generation, "owner-branch",
            turn_lease_holder=grant,
        ) is True, "the owner's own git publish was refused"
        assert _row(db)["git_branch"] == "owner-branch"
    finally:
        db.close()


def check_a_bystander_cannot_repoint_the_owners_gateway_identity(tmpdir) -> None:
    """``record_gateway_session_peer`` rewrites where replies are delivered."""
    db = _store(tmpdir)
    try:
        grant = _owned(db, session_key="chat:owner", chat_id="1")
        before = (_row(db)["session_key"], _row(db)["chat_id"])

        _refused(
            lambda: db.record_gateway_session_peer(
                "s", source="telegram", session_key="chat:attacker", chat_id="999"
            )
        )
        assert (_row(db)["session_key"], _row(db)["chat_id"]) == before, (
            f"the refused write repointed the conversation: "
            f"{(_row(db)['session_key'], _row(db)['chat_id'])!r} != {before!r}"
        )

        db.record_gateway_session_peer(
            "s", source="telegram", session_key="chat:owner", chat_id="2",
            turn_lease_holder=grant,
        )
        assert _row(db)["chat_id"] == "2", (
            "the owner's own peer record was refused"
        )
    finally:
        db.close()


def check_a_bystander_cannot_arm_the_owners_compression_cooldown(tmpdir) -> None:
    """The cooldown decides whether the owner's next turn may compress."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        assert _row(db)["compression_failure_cooldown_until"] is None

        _refused(
            lambda: db.record_compression_failure_cooldown(
                "s", time.time() + 9_999, "injected"
            )
        )
        assert _row(db)["compression_failure_cooldown_until"] is None, (
            "a bystander wedged the conversation out of compression, which "
            "walks it straight into context overflow"
        )

        db.record_compression_failure_cooldown(
            "s", time.time() + 5, "real", turn_lease_holder=grant
        )
        assert _row(db)["compression_failure_cooldown_until"] is not None, (
            "the owner's own cooldown record was refused"
        )
    finally:
        db.close()


def check_a_bystander_cannot_relabel_the_owners_activity(tmpdir) -> None:
    """The activity label is what the session list and resume banner read."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        db.touch_session_activity(
            "s", 1000.0, description="owner working", turn_lease_holder=grant
        )
        before = _row(db)["last_activity_description"]
        assert before == "owner working"

        _refused(
            lambda: db.touch_session_activity(
                "s", 2000.0, description="injected"
            )
        )
        assert _row(db)["last_activity_description"] == before, (
            f"the refused heartbeat relabelled the conversation: "
            f"{_row(db)['last_activity_description']!r}"
        )

        db.touch_session_activity(
            "s", 3000.0, description="owner still working",
            turn_lease_holder=grant,
        )
        assert _row(db)["last_activity_description"] == "owner still working", (
            "the owner's own heartbeat was refused, which is every turn"
        )
    finally:
        db.close()


def check_a_bystander_cannot_hand_off_the_owners_session(tmpdir) -> None:
    """Handoff state decides which platform drives the conversation next."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        assert _row(db)["handoff_state"] is None

        _refused(lambda: db.request_handoff("s", "attacker-platform"))
        assert _row(db)["handoff_state"] is None, (
            "a bystander queued the owner's conversation for a handoff"
        )

        assert db.request_handoff(
            "s", "owner-platform", turn_lease_holder=grant
        ) is True, "the owner's own handoff request was refused"
        assert _row(db)["handoff_platform"] == "owner-platform"
    finally:
        db.close()


def check_a_bystander_cannot_flip_the_owners_list_flags(tmpdir) -> None:
    """Archive / pin / hide / read, and the title pair, on one owned row."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        db.set_session_title("s", "the owner's title", turn_lease_holder=grant)
        before = (
            _row(db)["archived"], _row(db)["pinned"],
            _row(db)["hidden"], _row(db)["title"],
        )

        _refused(lambda: db.set_session_archived("s", True))
        _refused(lambda: db.set_session_pinned("s", True))
        _refused(lambda: db.set_session_hidden("s", True))
        _refused(lambda: db.set_session_read("s", True))
        _refused(lambda: db.set_session_title("s", "injected title"))

        after = (
            _row(db)["archived"], _row(db)["pinned"],
            _row(db)["hidden"], _row(db)["title"],
        )
        assert after == before, (
            f"a refused flag write landed anyway: {after!r} != {before!r}"
        )

        assert db.set_session_archived(
            "s", True, turn_lease_holder=grant
        ) is True, "the owner's own archive was refused"
        assert _row(db)["archived"] == 1
    finally:
        db.close()


def check_a_repo_root_backfill_skips_the_owned_conversation(tmpdir) -> None:
    """A sweep over `cwd` keys still has to answer for the rows it moves."""
    db = _store(tmpdir)
    try:
        grant = _owned(db, "owned", cwd="/work/shared")
        db.create_session("free", "test", cwd="/work/shared")

        db.backfill_repo_roots({"/work/shared": "/work/shared/root"})

        assert _row(db, "owned")["git_repo_root"] in (None, ""), (
            f"the sweep filled the owned conversation's repo root: "
            f"{_row(db, 'owned')['git_repo_root']!r}"
        )
        assert _row(db, "free")["git_repo_root"] == "/work/shared/root", (
            "the sweep skipped the UNOWNED row too, so it repairs nothing"
        )
    finally:
        db.close()


def check_a_ghost_prune_skips_the_owned_conversation(tmpdir) -> None:
    """Deleting a ghost row is still deleting a row someone may own."""
    db = _store(tmpdir)
    try:
        # The collector's own predicate: source='tui', ended, no title, no
        # messages, older than 24h. Built exactly, so a pin that stops
        # collecting says something about the fence rather than the filter.
        for sid in ("owned", "free"):
            db.create_session(sid, "tui")
            db.end_session(sid, "done")
        grant = db.try_acquire_session_turn_lease(
            "owned", _holder("owner"), ttl_seconds=600
        )
        assert grant
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id IN ('owned', 'free')",
            (time.time() - 1_000_000,),
        )
        db._conn.commit()

        collected = db.prune_empty_ghost_sessions()

        assert db.get_session("owned") is not None, (
            "the ghost prune deleted a conversation a live turn owns"
        )
        assert db.get_session("free") is None, (
            "the ghost prune skipped the UNOWNED ghost too, so it collects "
            "nothing"
        )
        assert collected == 1, f"expected exactly one ghost, got {collected}"
    finally:
        db.close()


def check_a_bystander_cannot_repoint_the_owners_routing_entry(tmpdir) -> None:
    """The routing index names its conversation INSIDE ``entry_json``.

    A fence written against ``save_gateway_routing_entry``'s parameters would
    have found no session id to check at all — the parameter is a
    ``session_key``.
    """
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        db.save_gateway_routing_entry(
            "chat:1", json.dumps({"session_id": "s"}), scope="tg",
            turn_lease_holder=grant,
        )
        before = db.load_gateway_routing_entries(scope="tg")
        assert before.get("chat:1")

        _refused(
            lambda: db.save_gateway_routing_entry(
                "chat:1", json.dumps({"session_id": "hijack"}), scope="tg"
            )
        )
        assert db.load_gateway_routing_entries(scope="tg") == before, (
            f"a bystander repointed the owned conversation's routing entry: "
            f"{db.load_gateway_routing_entries(scope='tg')!r}"
        )

        db.save_gateway_routing_entry(
            "chat:1", json.dumps({"session_id": "s", "v": 2}), scope="tg",
            turn_lease_holder=grant,
        )
        assert db.load_gateway_routing_entries(scope="tg") != before, (
            "the owner's own routing write was refused"
        )
    finally:
        db.close()


def check_a_routing_replace_keeps_the_owned_conversations_entry(tmpdir) -> None:
    """A wholesale index rewrite is a SWEEP: it skips, it does not refuse."""
    db = _store(tmpdir)
    try:
        grant = _owned(db)
        db.create_session("other", "test")
        db.save_gateway_routing_entry(
            "chat:owned", json.dumps({"session_id": "s"}), scope="tg",
            turn_lease_holder=grant,
        )
        db.save_gateway_routing_entry(
            "chat:other", json.dumps({"session_id": "other"}), scope="tg"
        )
        owned_before = db.load_gateway_routing_entries(scope="tg")["chat:owned"]

        db.replace_gateway_routing_entries(
            {"chat:other": json.dumps({"session_id": "other", "v": 2})},
            scope="tg",
        )

        after = db.load_gateway_routing_entries(scope="tg")
        assert after.get("chat:owned") == owned_before, (
            f"the index rewrite dropped the owned conversation's route: "
            f"{after!r}"
        )
        assert json.loads(after["chat:other"])["v"] == 2, (
            "the rewrite skipped the UNOWNED entry too, so a gateway restart "
            "could never rebuild its index"
        )
    finally:
        db.close()


PINS = {
    "check_a_bystander_cannot_move_the_owners_cwd":
        check_a_bystander_cannot_move_the_owners_cwd,
    "check_a_bystander_cannot_publish_git_metadata_for_the_owner":
        check_a_bystander_cannot_publish_git_metadata_for_the_owner,
    "check_a_bystander_cannot_repoint_the_owners_gateway_identity":
        check_a_bystander_cannot_repoint_the_owners_gateway_identity,
    "check_a_bystander_cannot_arm_the_owners_compression_cooldown":
        check_a_bystander_cannot_arm_the_owners_compression_cooldown,
    "check_a_bystander_cannot_relabel_the_owners_activity":
        check_a_bystander_cannot_relabel_the_owners_activity,
    "check_a_bystander_cannot_hand_off_the_owners_session":
        check_a_bystander_cannot_hand_off_the_owners_session,
    "check_a_bystander_cannot_flip_the_owners_list_flags":
        check_a_bystander_cannot_flip_the_owners_list_flags,
    "check_a_repo_root_backfill_skips_the_owned_conversation":
        check_a_repo_root_backfill_skips_the_owned_conversation,
    "check_a_ghost_prune_skips_the_owned_conversation":
        check_a_ghost_prune_skips_the_owned_conversation,
    "check_a_bystander_cannot_repoint_the_owners_routing_entry":
        check_a_bystander_cannot_repoint_the_owners_routing_entry,
    "check_a_routing_replace_keeps_the_owned_conversations_entry":
        check_a_routing_replace_keeps_the_owned_conversations_entry,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_broad_writer_property(name, tmp_path):
    PINS[name](tmp_path)


def _guard_block(comment: str) -> str:
    """The admission call as it appears in ONE mutator, keyed by its comment."""
    return (
        f"            # {comment}\n"
        "            self._check_turn_lease_guard(\n"
        "                conn,\n"
        "                session_id,\n"
        "                turn_lease_holder,\n"
        "                turn_lease_ttl_seconds=turn_lease_ttl_seconds,\n"
        "            )\n"
    )


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_a_bystander_cannot_move_the_owners_cwd",
        module="hermes_state.py",
        find=_guard_block(
            "cwd and the git pair ride in the assembled prompt."
        ),
        replace="",
        why="the working directory and branch are part of what the next turn "
            "is dispatched with, and this also bumps the generation counter "
            "the publish path fences against",
    ),
    Mutation(
        pin="check_a_bystander_cannot_publish_git_metadata_for_the_owner",
        module="hermes_state.py",
        find=_guard_block(
            "A generation fence is not a lease: it orders writers, it does not admit them."
        ),
        replace="",
        why="the generation counter serializes publishers against each other; "
            "it says nothing about whether this publisher may write at all",
    ),
    Mutation(
        pin="check_a_bystander_cannot_repoint_the_owners_gateway_identity",
        module="hermes_state.py",
        find=_guard_block(
            "This rewrites where the conversation's replies are delivered."
        ),
        replace="",
        why="with include_compression_ancestors it rewrites the routing "
            "identity of the whole lineage, not just the row named",
    ),
    Mutation(
        pin="check_a_bystander_cannot_arm_the_owners_compression_cooldown",
        module="hermes_state.py",
        find=_guard_block(
            "The cooldown decides whether the owner's next turn may compress."
        ),
        replace="",
        why="arming the cooldown from outside wedges a conversation out of "
            "compression and into context overflow",
    ),
    Mutation(
        pin="check_a_bystander_cannot_relabel_the_owners_activity",
        module="hermes_state.py",
        find=_guard_block(
            "The activity label is what the session list and resume banner read."
        ),
        replace="",
        why="an observation-only write is still a write into somebody's "
            "conversation",
    ),
    Mutation(
        pin="check_a_bystander_cannot_hand_off_the_owners_session",
        module="hermes_state.py",
        find=_guard_block(
            "Handoff state decides which platform drives this conversation next."
        ),
        replace="",
        why="a queued handoff moves the conversation to another platform "
            "underneath the turn that is running on this one",
    ),
    Mutation(
        pin="check_a_bystander_cannot_flip_the_owners_list_flags",
        module="hermes_state.py",
        find=(
            "    def _check_session_flag_write(\n"
            "        self, conn, session_id, turn_lease_holder, turn_lease_ttl_seconds\n"
            "    ) -> None:\n"
        ),
        replace=(
            "    def _check_session_flag_write(\n"
            "        self, conn, session_id, turn_lease_holder, turn_lease_ttl_seconds\n"
            "    ) -> None:\n"
            "        return\n"
        ),
        why="the flag writers share one admission helper; short-circuiting it "
            "leaves archive, pin, hide, read and the title pair unfenced "
            "together",
    ),
    Mutation(
        pin="check_a_repo_root_backfill_skips_the_owned_conversation",
        module="hermes_state.py",
        find=(
            "                targets = self._skip_leased_conversations(conn, targets)[0]\n"
                "                if not targets:\n"
                "                    continue\n"
        ),
        replace="",
        why="without the per-row sweep admission the cwd-keyed UPDATE writes "
            "every matching row, owned conversations included",
    ),
    Mutation(
        pin="check_a_ghost_prune_skips_the_owned_conversation",
        module="hermes_state.py",
        find="            ids = self._skip_leased_conversations(conn, ids)[0]\n",
        replace="",
        why="the ghost collector deletes rows; a row inside an owned "
            "conversation is not garbage",
    ),
    Mutation(
        pin="check_a_bystander_cannot_repoint_the_owners_routing_entry",
        module="hermes_state.py",
        find=(
            "            self._admit_routing_write(\n"
            "                conn, [(scope, session_key)], entry_json,\n"
            "                turn_lease_holder, turn_lease_ttl_seconds,\n"
            "            )\n"
        ),
        replace="",
        why="the routing index names its conversation inside entry_json, so "
            "removing this leaves the whole index outside the fence",
    ),
    Mutation(
        pin="check_a_routing_replace_keeps_the_owned_conversations_entry",
        module="hermes_state.py",
        find=(
            "            # conversation a live turn owns are KEPT, not replaced.\n"
            "            keep = self._owned_routing_keys(conn, scope)\n"
        ),
        replace=(
            "            # conversation a live turn owns are KEPT, not replaced.\n"
            "            keep = set()\n"
        ),
        why="a wholesale index rewrite that keeps nothing drops the owned "
            "conversation's route, which is how a live turn's replies stop "
            "being delivered",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    assert_mutation_kills_the_pin(mutation, str(_SELF), tmp_path)


def test_every_pin_has_a_mutation_that_kills_it():
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
