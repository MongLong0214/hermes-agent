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

    with pytest.raises(rollback.TurnFenceRollbackRefused) as caught:
        rollback.rollback_turn_fence(store, backup_path=tmp_path / "backup.db")

    assert "keep" in str(caught.value)
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

    with pytest.raises(rollback.TurnFenceRollbackRefused) as caught:
        rollback.rollback_turn_fence(store, backup_path=occupied)

    assert "backup" in str(caught.value).lower()
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
    report = rollback.rollback_turn_fence(store, backup_path=backup)

    assert report["backup"]["verified"] is True
    assert backup.is_file()
    assert _canonical_rows(backup) == before_rows


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


def test_rollback_verifies_the_installed_surface_before_changing_anything(
    tmp_path,
):
    """All-or-nothing, and the check runs first.

    One trigger is removed out of band, so the store's installed surface is no
    longer the expected one. The rollback must refuse the WHOLE operation and
    leave the other eight in place: a rollback that drops what it recognises
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
    assert victim not in remaining and len(remaining) == 8

    with pytest.raises(rollback.TurnFenceRollbackRefused) as caught:
        rollback.rollback_turn_fence(store, backup_path=tmp_path / "backup.db")

    assert victim in str(caught.value)
    assert _installed_triggers(store) == remaining, (
        "the rollback refused and still removed triggers — it is not "
        "all-or-nothing"
    )


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
    3. the same old binary can then OPEN and WRITE the rolled-back copy.
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

    rolled_back = tmp_path / "rolled-back.db"
    shutil.copyfile(original, rolled_back)
    before_rows = _canonical_rows(rolled_back)

    report = rollback.rollback_turn_fence(
        rolled_back, backup_path=tmp_path / "rollback-backup.db"
    )
    assert sorted(report["dropped_triggers"]) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    )
    assert _installed_triggers(rolled_back) == []
    assert _canonical_rows(rolled_back) == before_rows, (
        "the rollback changed user rows; it is only allowed to remove triggers"
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

    Reopening the rolled-back copy with the current binary must put the fence
    back — it is ``CREATE TRIGGER IF NOT EXISTS`` in schema init, so this is a
    property of the store, not of the rollback tool — and the same old binary
    must then be refused again. Without this, a rolled-back store stays
    unfenced forever and the rollback is a downgrade with no way home.
    """
    rollback = _rollback_module()

    store = tmp_path / "state.db"
    _fenced_store(store, leave_lease_live=False)
    rollback.rollback_turn_fence(store, backup_path=tmp_path / "backup.db")
    assert _installed_triggers(store) == []

    reopened = SessionDB(db_path=store)
    reopened.close()
    assert _installed_triggers(store) == sorted(
        hermes_state_common.TURN_FENCE_TRIGGERS
    ), "reopening with the current binary did not reinstall the fence"

    again = _old_binary_probe(base_binary_tree, store, tmp_path)
    assert again.returncode == 0, again.stderr
    assert "WRITE-OK" not in again.stdout, (
        "the old binary wrote a store the current binary had re-fenced:\n"
        f"{again.stdout}\n{again.stderr}"
    )
