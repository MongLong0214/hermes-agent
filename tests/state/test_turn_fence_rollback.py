"""A bounded, offline, fail-closed way back off the generation fence.

WHY THIS EXISTS AND WHAT IT IS NOT
    A binary from before the turn fence cannot write a store that has it — not
    the transcript, and not ``_init_schema``'s own ``UPDATE messages SET
    active = 1 WHERE active IS NULL``, so it cannot even finish a writable
    open. That is INTENTIONAL and required, and nothing here softens it. There
    is no compatibility fallback, no automatic trigger drop on open, and no
    silent admission of an old writer.

    What is not acceptable is that the only way back was a human typing
    ``DROP TRIGGER`` nine times into a live store. That is not a rollback
    story; it is an invitation to do it wrong at 3am, on the wrong file, with a
    turn in flight, and with no copy of the data if it goes badly.

    So: ONE bounded operation, offline, on a store nobody is using, that makes
    an executable backup first, verifies exactly what it is about to remove
    before it removes anything, and removes all of it or none of it.

WHAT THIS FILE PINS
    * refusal while any conversation in the store is owned by a live turn, with
      the ROWS and the trigger set both asserted unchanged afterwards;
    * refusal when the backup precondition cannot be MET — not documented, met:
      the backup is written and read back before a trigger is touched;
    * the trigger set is GENERATED from ``TURN_FENCE_SURFACE``. Not a nine-name
      list in the rollback module, and the test proves it by moving the
      declaration and watching the rollback follow;
    * verification BEFORE mutation: a store whose installed surface is not the
      exact expected one is refused whole, with every trigger still in place;
    * the acceptance run: snapshot every ``sessions`` / ``messages`` /
      ``session_turn_leases`` row, roll back, prove the rows are byte-identical,
      then run the EXACT binary at 261a4efb and prove it can open AND write;
    * and the return leg — a current binary reopening the rolled-back copy
      reinstalls the fence, after which the same old binary is refused again.

    Temp databases only. No live checkout, no ``state.db``, no service.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import sqlite3
import subprocess
import sys

import pytest

import hermes_state_common
from hermes_state import SessionDB
from tests.state import test_turn_lease_generation_trigger as _base_binary

#: Reused rather than re-derived: one place decides which commit "the exact old
#: binary" means, and one fixture knows how to extract it. Bound by assignment
#: so the fixture name is also a plain module attribute, which keeps the test
#: parameters below from reading as redefinitions of an import.
BASE_COMMIT = _base_binary.BASE_COMMIT
base_binary_tree = _base_binary.base_binary_tree


def _rollback_module():
    """Import the rollback slice, or fail with what is missing and why.

    Imported inside the test rather than at module scope so an absent module is
    a behavioural failure of these tests instead of a collection error. A
    collection error proves the symbol is absent; it does not exercise the
    property, and a RED that cannot be collected is not a RED.
    """
    try:
        from hermes_cli import session_fence_rollback
    except ImportError as exc:
        pytest.fail(
            "there is no bounded rollback slice "
            f"(hermes_cli.session_fence_rollback): {exc}. The only way off "
            "the generation fence is a human typing DROP TRIGGER at a live "
            "store, which is not a rollback story"
        )
    return session_fence_rollback


def _holder(tag: str) -> str:
    return f"pid={os.getpid()}:turn={tag}:platform=test"


def _installed_triggers(path: pathlib.Path) -> list[str]:
    conn = sqlite3.connect(str(path))
    try:
        return sorted(
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'hermes_turn_fence_%'"
            )
        )
    finally:
        conn.close()


def _canonical_rows(path: pathlib.Path) -> dict[str, list[tuple]]:
    """Every row of the three tables the fence covers, as plain tuples."""
    conn = sqlite3.connect(str(path))
    try:
        return {
            "sessions": [
                tuple(r) for r in conn.execute(
                    "SELECT id, source, model, model_config, system_prompt_hash, "
                    "title, end_reason, ended_at, parent_session_id, "
                    "message_count FROM sessions ORDER BY id"
                )
            ],
            "messages": [
                tuple(r) for r in conn.execute(
                    "SELECT id, session_id, role, content, timestamp, active "
                    "FROM messages ORDER BY id"
                )
            ],
            "session_turn_leases": [
                tuple(r) for r in conn.execute(
                    "SELECT conversation_id, holder, expires_at, epoch "
                    "FROM session_turn_leases ORDER BY conversation_id"
                )
            ],
        }
    finally:
        conn.close()


def _fenced_store(path: pathlib.Path, *, leave_lease_live: bool):
    """A v27 store this generation created, with real rows in all three tables."""
    db = SessionDB(db_path=path)
    try:
        db.create_session("keep", "test")
        grant = db.try_acquire_session_turn_lease(
            "keep", _holder("keep"), ttl_seconds=600
        )
        assert grant, "could not take the lease this fixture depends on"
        db.append_message(
            session_id="keep", role="user", content="irreplaceable",
            turn_lease_holder=grant,
        )
        db.append_message(
            session_id="keep", role="assistant", content="also irreplaceable",
            turn_lease_holder=grant,
        )
        if not leave_lease_live:
            db.release_session_turn_lease("keep", grant)
    finally:
        db.close()
    assert _installed_triggers(path) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    )
    return grant


def _private_work_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    work = tmp_path / f"work-{os.urandom(4).hex()}"
    work.mkdir()
    return work


def _rehearsal_refusal(rollback, store, backup, tmp_path):
    """MAINTENANCE ONLY. Drive the machinery and hand back how it refused.

    These properties used to be driven through ``rollback_turn_fence``. Both
    library entry points now refuse before they observe the store, so a test
    driven through either reaches ``offline-authority-unknown`` and stops
    testing what its name says — failing for the wrong reason is the same
    defect as passing for the wrong one, and both stop being evidence.

    The properties themselves did not stop mattering, so they move behind the
    direct private-machinery driver. Nothing here is evidence about the verb's
    boundary; it keeps the machinery from rotting until an artifact whose
    coherence its producer establishes reopens the path.
    """
    work = _private_work_dir(tmp_path)
    try:
        prepared = rollback.prepare_the_private_copy(store, work_dir=work)
    except rollback.TurnFenceRollbackRefused as exc:
        return exc
    try:
        rollback._commit_the_rollback(
            prepared, backup, sorted(rollback.rollback_trigger_names()),
            report_as=store,
        )
    except rollback.TurnFenceRollbackRefused as exc:
        return exc
    finally:
        prepared.close()
    raise AssertionError("the machinery did not refuse at all")


def _roll_back_the_private_object(rollback, store, backup, tmp_path):
    """Perform the real rollback on the private object; return report + image.

    The same boundary the rehearsal drives, entered directly so the test can
    keep the resulting database. ``serialize()`` materializes it into bytes THE
    TEST then writes to a temp file — the production path never writes a
    rolled-back store anywhere, and this does not make it. It is how a probe
    gets something to open.
    """
    prepared = rollback.prepare_the_private_copy(
        store, work_dir=_private_work_dir(tmp_path)
    )
    try:
        report = rollback._commit_the_rollback(
            prepared, backup, sorted(rollback.rollback_trigger_names()),
            report_as=store,
        )
        return report, prepared.connection.serialize()
    finally:
        prepared.close()


def _materialize(image: bytes, where: pathlib.Path) -> pathlib.Path:
    where.write_bytes(image)
    return where

def test_rollback_refuses_while_a_conversation_is_live_owned(tmp_path):
    """Offline-only, stated as rows and as triggers.

    A rollback with a turn in flight hands the conversation to a binary that
    has never heard of the lease — which is precisely the interleave the fence
    was installed to stop, arranged by the tool that removes it. The assertion
    is that NOTHING moved: not the rows, not the trigger set.
    """
    rollback = _rollback_module()
    store = tmp_path / "state.db"
    _fenced_store(store, leave_lease_live=True)
    before_rows = _canonical_rows(store)
    before_triggers = _installed_triggers(store)

    refusal = _rehearsal_refusal(rollback, store, tmp_path / "backup.db", tmp_path)

    assert refusal.reason == "live-turn", (
        f"the live turn was not what refused this: {refusal.reason} — {refusal}"
    )
    assert "keep" in str(refusal)
    assert _canonical_rows(store) == before_rows
    assert _installed_triggers(store) == before_triggers


def test_rollback_refuses_when_the_backup_cannot_be_made(tmp_path):
    """The backup precondition is EXECUTED, not documented.

    ``backup_path`` already exists, so the copy cannot be made without
    destroying whatever is there. The rollback must stop at that, before any
    DDL — a docstring that says "take a backup first" is not a precondition.
    """
    rollback = _rollback_module()
    store = tmp_path / "state.db"
    _fenced_store(store, leave_lease_live=False)
    before_rows = _canonical_rows(store)
    before_triggers = _installed_triggers(store)

    occupied = tmp_path / "backup.db"
    occupied.write_bytes(b"not a database, and not to be clobbered")

    refusal = _rehearsal_refusal(rollback, store, occupied, tmp_path)

    assert refusal.reason == "backup-exists", (
        f"the occupied destination was not what refused this: {refusal.reason}"
    )
    assert "backup" in str(refusal).lower()
    assert occupied.read_bytes() == b"not a database, and not to be clobbered"
    assert _canonical_rows(store) == before_rows
    assert _installed_triggers(store) == before_triggers


def test_the_backup_is_a_readable_copy_of_the_rows_before_any_ddl(tmp_path):
    """A backup that is not readable is not data preservation.

    Written AND read back: the copy is opened and its three tables compared
    against the source, so "the file exists" cannot pass for "the data
    survived".
    """
    rollback = _rollback_module()
    store = tmp_path / "state.db"
    _fenced_store(store, leave_lease_live=False)
    before_rows = _canonical_rows(store)

    backup = tmp_path / "backup.db"
    report, _image = _roll_back_the_private_object(
        rollback, store, backup, tmp_path
    )

    # `_commit_the_rollback` returns the backup report itself, not a wrapper.
    assert report["verified"] is True
    assert report["durable"] is True
    assert backup.is_file()
    assert _canonical_rows(backup) == before_rows
    # And the source is untouched: the object rolled back was the private copy.
    assert _installed_triggers(store) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    )
    assert _canonical_rows(store) == before_rows


def test_rollback_verifies_the_installed_surface_before_changing_anything(
    tmp_path,
):
    """All-or-nothing, and the check runs first.

    One trigger is removed out of band, so the store's installed surface is no
    longer the expected one. The rollback must refuse the WHOLE operation and
    leave the other triggers in place: a rollback that drops what it recognises
    and shrugs at the rest leaves a store that is fenced against some writes
    and not others, which is the worst of both.
    """
    rollback = _rollback_module()
    store = tmp_path / "state.db"
    _fenced_store(store, leave_lease_live=False)

    victim = hermes_state_common.turn_fence_trigger_name("messages", "INSERT")
    conn = sqlite3.connect(str(store), isolation_level=None)
    try:
        conn.execute(f"DROP TRIGGER {victim}")
    finally:
        conn.close()

    remaining = _installed_triggers(store)
    # Counted off the DECLARATION, not typed. A literal here was right while the
    # surface had nine entries and silently wrong the moment it grew: the assert
    # would have failed on a store that was in exactly the intended state.
    expected_after_drop = len(rollback.rollback_trigger_names()) - 1
    assert victim not in remaining and len(remaining) == expected_after_drop, (
        f"dropping {victim} out of band should leave every other declared "
        f"trigger in place: expected {expected_after_drop}, found "
        f"{len(remaining)} — {sorted(remaining)}"
    )

    refusal = _rehearsal_refusal(rollback, store, tmp_path / "backup.db", tmp_path)

    assert refusal.reason == "surface-mismatch", (
        f"the half-installed surface was not what refused this: {refusal.reason}"
    )
    assert victim in str(refusal)
    assert _installed_triggers(store) == remaining, (
        "the rollback refused and still removed triggers — it is not "
        "all-or-nothing"
    )


def test_the_real_path_refuses_before_it_observes_or_mutates_the_source(
    tmp_path, monkeypatch
):
    """The in-place entry point refuses, and refuses BEFORE it looks.

    Every property above now runs against a private object, so one test has to
    hold the thing that made that necessary: ``rollback_turn_fence`` is offered
    an ordinary idle fenced store and must decline it without opening,
    copying or preparing anything. Asserted as EVENTS, because bytes agreeing
    would also be consistent with an open that happened to leave no trace on
    this build — and on a WAL build a read-only probe leaves ``-wal`` and
    ``-shm`` behind.
    """
    rollback = _rollback_module()
    store = tmp_path / "state.db"
    _fenced_store(store, leave_lease_live=False)
    before_rows = _canonical_rows(store)
    before_triggers = _installed_triggers(store)
    listing_before = sorted(entry.name for entry in tmp_path.iterdir())

    seen = {"preflight": 0, "bound": 0, "prepared": 0, "opened": []}
    real_sqlite_connect = rollback.sqlite3.connect

    def _watch(name, real):
        def _counted(*args, **kwargs):
            seen[name] += 1
            return real(*args, **kwargs)
        return _counted

    monkeypatch.setattr(
        rollback, "preflight_turn_fence_rollback",
        _watch("preflight", rollback.preflight_turn_fence_rollback),
    )
    monkeypatch.setattr(
        rollback, "BoundTarget", _watch("bound", rollback.BoundTarget)
    )
    monkeypatch.setattr(
        rollback, "prepare_the_private_copy",
        _watch("prepared", rollback.prepare_the_private_copy),
    )

    class _WatchingSqlite:
        def __getattr__(self, name):
            return getattr(sqlite3, name)

        def connect(self, target, *args, **kwargs):
            seen["opened"].append(str(target))
            return real_sqlite_connect(target, *args, **kwargs)

    monkeypatch.setattr(rollback, "sqlite3", _WatchingSqlite())

    with pytest.raises(rollback.TurnFenceRollbackRefused) as caught:
        rollback.rollback_turn_fence(store, backup_path=tmp_path / "backup.db")

    assert caught.value.reason == "offline-authority-unknown", (
        f"the in-place path refused for another reason: {caught.value.reason}"
    )
    assert seen["preflight"] == 0, "the real path ran the pre-flight"
    assert seen["bound"] == 0, "the real path opened the store with BoundTarget"
    assert seen["prepared"] == 0, (
        "the real path made a private copy for a run that cannot proceed"
    )
    assert [t for t in seen["opened"] if str(store) in t] == [], (
        f"the real path opened the operator's store: {seen['opened']!r}"
    )
    assert _canonical_rows(store) == before_rows
    assert _installed_triggers(store) == before_triggers
    assert sorted(entry.name for entry in tmp_path.iterdir()) == listing_before, (
        "the refusal added or removed files beside the store"
    )
    assert not (tmp_path / "backup.db").exists()

def test_the_rollback_trigger_set_is_generated_from_the_declaration(
    tmp_path, monkeypatch
):
    """Move the declaration; the rollback must follow it.

    The failure this rules out is a rollback module carrying its own list of
    nine trigger names, which is right on the day it is written and silently
    incomplete on the day the surface grows. So the surface is moved here and
    the rollback is required to notice — a hand-copied list cannot pass this.
    """
    rollback = _rollback_module()

    extra_surface = tuple(hermes_state_common.TURN_FENCE_SURFACE) + (
        ("system_prompts", "INSERT"),
    )
    monkeypatch.setattr(
        hermes_state_common, "TURN_FENCE_SURFACE", extra_surface, raising=True
    )
    monkeypatch.setattr(
        hermes_state_common,
        "TURN_FENCE_TRIGGERS",
        tuple(
            hermes_state_common.turn_fence_trigger_name(table, operation)
            for table, operation in extra_surface
        ),
        raising=True,
    )

    derived = rollback.rollback_trigger_names()
    assert "hermes_turn_fence_system_prompts_insert" in derived
    assert sorted(derived) == sorted(hermes_state_common.TURN_FENCE_TRIGGERS)


def _old_binary_probe(tree: pathlib.Path, store: pathlib.Path, tmp_path):
    """Run the exact base binary against *store*; report open + write."""
    probe = f'''
import pathlib, sys
import hermes_state

here = pathlib.Path({str(tree)!r}).resolve()
loaded = pathlib.Path(hermes_state.__file__).resolve()
assert loaded.is_relative_to(here), (
    "this subprocess imported %s, not the extracted base tree under %s"
    % (loaded, here)
)
print("LOADED", loaded)
try:
    db = hermes_state.SessionDB(pathlib.Path({str(store)!r}))
except Exception as exc:
    print("OPEN-REFUSED", type(exc).__name__, exc)
    raise SystemExit(0)
print("OPEN-OK")
try:
    db.append_message(session_id="keep", role="assistant", content="OLD BINARY")
except Exception as exc:
    print("WRITE-REFUSED", type(exc).__name__, exc)
else:
    print("WRITE-OK")
finally:
    try:
        db.close()
    except Exception:
        pass
'''
    home = tmp_path / f"home-{os.urandom(4).hex()}"
    home.mkdir()
    return subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(tree),
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(home),
            "PYTHONPATH": str(tree),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True, text=True, timeout=180,
    )


def test_temp_db_acceptance_rows_survive_and_the_exact_old_binary_writes(
    tmp_path, base_binary_tree
):
    """The acceptance run, on a copy, with the exact binary at 261a4efb.

    Three claims in one place because separating them lets each pass while the
    sequence fails:

    1. the old binary CANNOT write the fenced store (otherwise the rollback is
       measuring nothing);
    2. rollback leaves every ``sessions`` / ``messages`` / lease row identical;
    3. the same old binary can then OPEN and WRITE the rolled-back result.

    WHERE THE ROLLED-BACK STORE COMES FROM. The rollback runs on a private
    object with no pathname, so there is no file for a probe to open until this
    test makes one: ``serialize()`` gives the bytes and the TEST writes them
    into its own temp directory. Production writes a rolled-back store nowhere,
    and this does not change that — it is a probe needing something to open,
    not a success path being manufactured. What is being measured is unchanged:
    the old binary meets exactly the database the rollback produced.
    """
    rollback = _rollback_module()

    original = tmp_path / "state.db"
    _fenced_store(original, leave_lease_live=False)

    fenced_copy = tmp_path / "fenced-copy.db"
    shutil.copyfile(original, fenced_copy)
    fenced = _old_binary_probe(base_binary_tree, fenced_copy, tmp_path)
    assert fenced.returncode == 0, fenced.stderr
    assert "WRITE-OK" not in fenced.stdout, (
        "the old binary wrote a FENCED store, so this rollback proves nothing:"
        f"\n{fenced.stdout}\n{fenced.stderr}"
    )

    before_rows = _canonical_rows(original)
    report, image = _roll_back_the_private_object(
        rollback, original, tmp_path / "rollback-backup.db", tmp_path
    )
    assert report["verified"] is True and report["durable"] is True

    rolled_back = _materialize(image, tmp_path / "rolled-back.db")
    assert _installed_triggers(rolled_back) == []
    assert _canonical_rows(rolled_back) == before_rows, (
        "the rollback changed user rows; it is only allowed to remove triggers"
    )
    # And the source it was prepared from is untouched.
    assert _installed_triggers(original) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    )

    old = _old_binary_probe(base_binary_tree, rolled_back, tmp_path)
    assert old.returncode == 0, old.stderr
    assert "OPEN-OK" in old.stdout, (
        f"the binary at {BASE_COMMIT[:10]} still could not OPEN the "
        f"rolled-back store:\n{old.stdout}\n{old.stderr}"
    )
    assert "WRITE-OK" in old.stdout, (
        f"the binary at {BASE_COMMIT[:10]} opened the rolled-back store but "
        f"could not write it:\n{old.stdout}\n{old.stderr}"
    )


def test_a_current_binary_reinstalls_the_fence_on_a_rolled_back_store(
    tmp_path, base_binary_tree
):
    """The return leg. Rollback is not a one-way door, and not a permanent one.

    Reopening the rolled-back result with the current binary must put the fence
    back — it is ``CREATE TRIGGER IF NOT EXISTS`` in schema init, so this is a
    property of the store, not of the rollback tool — and the same old binary
    must then be refused again. Without this, a rolled-back store stays
    unfenced forever and the rollback is a downgrade with no way home.
    """
    rollback = _rollback_module()

    store = tmp_path / "state.db"
    _fenced_store(store, leave_lease_live=False)
    _report, image = _roll_back_the_private_object(
        rollback, store, tmp_path / "backup.db", tmp_path
    )
    rolled_back = _materialize(image, tmp_path / "rolled-back.db")
    assert _installed_triggers(rolled_back) == []

    reopened = SessionDB(db_path=rolled_back)
    reopened.close()
    assert _installed_triggers(rolled_back) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), "reopening with the current binary did not reinstall the fence"

    again = _old_binary_probe(base_binary_tree, rolled_back, tmp_path)
    assert again.returncode == 0, again.stderr
    assert "WRITE-OK" not in again.stdout, (
        "the old binary wrote a store the current binary had re-fenced:\n"
        f"{again.stdout}\n{again.stderr}"
    )
