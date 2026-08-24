"""There is no atomic "check identity and delete" primitive on this platform.

Three rounds tried to build one anyway, in ``_AcquiredDestinations._release``:
close-then-check ordering, a second lstat immediately before the unlink, a
content seal against ABA inode reuse. Each closed one hole and left another —
fd-close-before-check, double-release re-entry, ABA inode reuse, a
``BaseException`` mid-work, ``suffix=""`` bypassing the seal — because every
version was still "identity-check, THEN unlink", and the interval between the
check and the unlink cannot be closed on this API: POSIX has no call that
unlinks "this exact inode at this name" atomically, and holding the descriptor
open across the unlink is not on the table either, because the close has to
happen first (the same ordering Windows requires).

OWNER RULING: stop narrowing the window. Remove the unlink. ``_release`` now
closes the descriptor — always safe, always done, on every path including a
``BaseException`` — and, if the reserved name still resolves to a file
afterward, reports it as ``cleanup_required`` at its concrete path instead of
ever deleting it. This is a single-trusted-owner local side project: an
occasional leftover reserved-name file for the owner to delete by hand is far
cheaper than a wrong-target delete.

Every pin below proves the NEGATIVE directly — the unlink spy shows ZERO
calls — rather than merely checking the reported outcome, because a correct
outcome computed by a code path that still happens to call unlink is not the
property this file exists to hold.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from tests.state.lease_mutation_harness import (
    Mutation,
    assert_every_pin_has_a_killer,
    assert_mutation_kills_the_pin,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

#: The verb lives in the CLI package and drives the state layer.
_EXTRA_EXTRACT = (".",)


def _load_verb_pins():
    """The shipped pin module, reused for its store and boundary fixtures."""
    import importlib.util
    import sys

    path = REPO_ROOT / "tests/hermes_cli/test_sessions_fence_rollback_verb.py"
    spec = importlib.util.spec_from_file_location("verbpins_for_identity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _UnlinkSpy:
    """Forwards every ``os`` call except counting ``unlink``.

    Installed as the module's ``os`` so the count reflects exactly what the
    library under test invoked, not a mock of the library's own claims.
    """

    def __init__(self):
        self.unlink_calls: list = []

    def __getattr__(self, name):
        return getattr(os, name)

    def unlink(self, path, *args, **kwargs):
        self.unlink_calls.append(pathlib.Path(path))
        return os.unlink(path, *args, **kwargs)


def check_the_real_boundary_never_calls_unlink_and_reports_leftovers(
    tmpdir,
) -> None:
    """Driven through the real backup boundary: the ordinary, successful run.

    Nothing between ``acquire()`` and ``release_the_sidecars()`` ever writes to
    or removes the reserved ``-wal``/``-shm``/``-journal`` placeholders, so on
    an ordinary, non-adversarial run they are still on disk when release runs.
    Under the removed-unlink contract that is not a failure: the rollback
    still backs up and commits, and each placeholder is reported as
    ``cleanup_required`` residue rather than silently deleted or silently
    dropped.
    """
    pins = _load_verb_pins()
    from hermes_cli import session_fence_rollback as lib

    where = pathlib.Path(tmpdir)
    pins._sandbox_home(where)
    store = where / "state.db"
    pins._fenced_store(store, leave_lease_live=False)
    work_dir = where / "work"
    work_dir.mkdir()

    import hermes_state_common

    backup = work_dir.parent / "backup.db"
    outcome = lib.RollbackOutcome()
    spy = _UnlinkSpy()
    returned, crash = None, ""
    had_os = hasattr(lib, "os")
    previous = getattr(lib, "os", None)
    lib.os = spy
    try:
        copy = lib.prepare_the_private_copy(store, work_dir=work_dir)
        returned = lib._commit_the_rollback(
            copy, backup, sorted(hermes_state_common.TURN_FENCE_TRIGGERS),
            report_as=store, outcome=outcome,
        )
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        crash = f"{type(exc).__name__}: {exc}"
    finally:
        if had_os:
            lib.os = previous
        else:
            del lib.os

    assert not crash, f"the boundary crashed, so nothing below was measured: {crash}"
    assert not spy.unlink_calls, (
        f"the boundary called os.unlink {len(spy.unlink_calls)} time(s) on the "
        f"sidecar family: {spy.unlink_calls!r}. There is no atomic check-and-"
        f"delete on this platform, so nothing in this path may unlink a "
        f"reserved sidecar any more"
    )
    assert returned is not None, (
        "an ordinary run did not produce a backup — removing the unlink must "
        f"not break the common path: {returned!r}"
    )
    assert outcome.outcome == "committed", (
        f"an ordinary run did not commit: {outcome.outcome!r}"
    )

    sidecars = ("-wal", "-shm", "-journal")
    surviving = [
        backup.with_name(backup.name + suffix) for suffix in sidecars
        if backup.with_name(backup.name + suffix).exists()
    ]
    assert surviving, (
        "the fixture left no reserved sidecar behind, so this pin measures "
        "nothing about leftover reporting"
    )
    reported = {record["path"]: record for record in outcome.facts()["residue"]}
    for member in surviving:
        assert str(member) in reported, (
            f"{member} survived the release and is not in the residue report: "
            f"{outcome.facts()['residue']!r}"
        )
        assert reported[str(member)]["error"] == "cleanup_required", (
            f"a leftover reserved sidecar is not reported as cleanup_required: "
            f"{reported[str(member)]!r}"
        )


def check_a_swap_between_close_and_the_existence_check_is_never_deleted(
    tmpdir,
) -> None:
    """A name swapped for a stranger's file the instant the descriptor closes.

    ``_release`` closes the descriptor and then looks at the name. Nothing
    pins the inode from the close onward, so a real actor with write access to
    the parent directory can replace the name in exactly that gap. Under the
    removed-unlink contract this can no longer matter for safety — nothing
    past the close ever deletes anything — and this pin proves that directly:
    the unlink spy is watched, not inferred from the outcome.
    """
    import hermes_cli.session_fence_rollback as sfr
    from hermes_cli.session_fence_rollback import _AcquiredDestinations

    where = pathlib.Path(tmpdir)
    backup = where / "backup.db"
    reservation = _AcquiredDestinations(backup)
    reservation.acquire()
    try:
        member = reservation._member("-wal")
        wal_handle = reservation.handles["-wal"]
        stranger = b"a different file that arrived in the close-to-check window"

        real_close = os.close
        spy = _UnlinkSpy()

        class _SwappingOs(_UnlinkSpy):
            def close(self, handle, *args, **kwargs):
                result = real_close(handle, *args, **kwargs)
                if handle == wal_handle:
                    # THE WINDOW: the descriptor is gone, and nothing else has
                    # looked at the name yet.
                    member.unlink()
                    member.write_bytes(stranger)
                return result

        swapping = _SwappingOs()
        had_lib_os = hasattr(sfr, "os")
        previous_lib_os = getattr(sfr, "os", None)
        sfr.os = swapping
        try:
            problem = reservation._release("-wal")
        finally:
            if had_lib_os:
                sfr.os = previous_lib_os
            else:
                del sfr.os

        assert not swapping.unlink_calls, (
            f"the release called os.unlink {len(swapping.unlink_calls)} "
            f"time(s) after a stranger's file was swapped in: "
            f"{swapping.unlink_calls!r}"
        )
        assert member.exists() and member.read_bytes() == stranger, (
            f"the swapped-in stranger did not survive the release: "
            f"exists={member.exists()!r}. A name that changed after the "
            f"descriptor closed was deleted anyway.\n  "
            f"release() returned: {problem!r}"
        )
        assert problem is not None and problem.get("error") == "cleanup_required", (
            f"the release did not report the surviving stranger as needing "
            f"cleanup: {problem!r}"
        )
    finally:
        reservation.close()


def check_a_same_inode_swap_after_close_is_never_deleted(
    tmpdir,
) -> None:
    """A coincidental ``(st_dev, st_ino)`` match after close proves nothing.

    sol's counterexample: an inode NUMBER is a small, densely-reused index, not
    an identity, once nothing pins it — the kernel could in principle hand a
    just-freed number straight back to an unrelated file created at the same
    well-known name. Earlier rounds answered this with a content seal compared
    at release time. That comparison is gone along with the unlink it used to
    authorise: this pin swaps in a stranger AND makes the post-close ``lstat``
    report the ORIGINAL ``(st_dev, st_ino)`` for it, and proves the release
    still never deletes anything — it does not need to tell the coincidence
    apart from the truth any more, because it acts on neither.
    """
    import hermes_cli.session_fence_rollback as sfr
    from hermes_cli.session_fence_rollback import _AcquiredDestinations

    where = pathlib.Path(tmpdir)
    backup = where / "backup.db"
    reservation = _AcquiredDestinations(backup)
    reservation.acquire()
    try:
        suffix = "-wal"
        member = reservation._member(suffix)
        identity = reservation.identities[suffix]
        wal_handle = reservation.handles[suffix]
        stranger = b"a different file that reused the just-freed inode number"

        real_close = os.close
        swapped = {"done": False}

        class _FakeStat:
            def __init__(self, dev, ino, real_result):
                self.st_dev = dev
                self.st_ino = ino
                self._real = real_result

            def __getattr__(self, name):
                return getattr(self._real, name)

        class _SameInodeCoincidenceOs(_UnlinkSpy):
            def close(self, handle, *args, **kwargs):
                result = real_close(handle, *args, **kwargs)
                if handle == wal_handle:
                    member.unlink()
                    member.write_bytes(stranger)
                    swapped["done"] = True
                return result

            def lstat(self, path, *args, **kwargs):
                real_result = os.lstat(path, *args, **kwargs)
                if pathlib.Path(path) == member and swapped["done"]:
                    return _FakeStat(identity[0], identity[1], real_result)
                return real_result

        fake = _SameInodeCoincidenceOs()
        had_lib_os = hasattr(sfr, "os")
        previous_lib_os = getattr(sfr, "os", None)
        sfr.os = fake
        try:
            problem = reservation._release(suffix)
        finally:
            if had_lib_os:
                sfr.os = previous_lib_os
            else:
                del sfr.os

        assert swapped["done"], (
            "the stand-in close never swapped in the stranger's file, so this "
            "probe measures nothing"
        )
        assert not fake.unlink_calls, (
            f"the release called os.unlink {len(fake.unlink_calls)} time(s) "
            f"even though the swapped-in file coincidentally matched the "
            f"recorded (st_dev, st_ino): {fake.unlink_calls!r}"
        )
        assert member.exists() and member.read_bytes() == stranger, (
            f"the swapped-in stranger did not survive the release: "
            f"exists={member.exists()!r}. A (st_dev, st_ino) match that was "
            f"only a coincidence of inode-number reuse must not authorise "
            f"deleting a file this run never created.\n  "
            f"release() returned: {problem!r}"
        )
        assert problem is not None and problem.get("error") == "cleanup_required", (
            f"the release did not report the surviving stranger as needing "
            f"cleanup: {problem!r}"
        )
    finally:
        reservation.identities.clear()
        reservation.close()


def check_a_base_exception_mid_release_leaves_the_suffix_claimed(tmpdir) -> None:
    """A ``BaseException`` raised inside ``_release``'s work must not look resolved.

    The descriptor close is unconditional and happens FIRST, so a
    ``KeyboardInterrupt`` (the real-world case) raised during the existence
    check that follows it still leaves the descriptor closed — no leak — while
    ``self.identities`` still shows the suffix claimed, because the pop only
    happens after ``_resolve_release`` returns normally. Nothing here can call
    ``os.unlink`` any more, so this no longer needs to prove a file survived —
    only that closing precedes claiming, and that no interrupted attempt at
    resolving a suffix quietly deletes anything either.
    """
    import hermes_cli.session_fence_rollback as sfr
    from hermes_cli.session_fence_rollback import _AcquiredDestinations

    where = pathlib.Path(tmpdir)
    backup = where / "backup.db"
    reservation = _AcquiredDestinations(backup)
    reservation.acquire()
    try:
        suffix = "-wal"
        member = reservation._member(suffix)
        real_lstat = os.lstat

        class _InterruptingOs(_UnlinkSpy):
            """Forwards everything except the lstat of the member under test."""

            def lstat(self, path, *args, **kwargs):
                if pathlib.Path(path) == member:
                    raise KeyboardInterrupt("simulated interrupt mid-release")
                return real_lstat(path, *args, **kwargs)

        interrupting = _InterruptingOs()
        had_lib_os = hasattr(sfr, "os")
        previous_lib_os = getattr(sfr, "os", None)
        sfr.os = interrupting
        interrupted = False
        try:
            reservation._release(suffix)
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if had_lib_os:
                sfr.os = previous_lib_os
            else:
                del sfr.os

        assert interrupted, (
            "the stand-in lstat did not raise, so this probe measures nothing "
            "about what happens when a BaseException interrupts the release"
        )
        assert not interrupting.unlink_calls, (
            f"os.unlink was called {len(interrupting.unlink_calls)} time(s) "
            f"during an interrupted release: {interrupting.unlink_calls!r}"
        )
        assert suffix not in reservation.handles, (
            f"the descriptor for {suffix!r} was not closed before the "
            f"interrupt reached the caller. Closing must happen "
            f"unconditionally and first, before the suffix can lose tracked "
            f"status, or a KeyboardInterrupt during the existence check that "
            f"follows would leak the handle"
        )
        assert suffix in reservation.identities, (
            f"a BaseException raised mid-release popped {suffix!r} out of "
            f"self.identities anyway, so the suffix looks resolved to "
            f"remove_only_what_we_created even though its fate was never "
            f"observed"
        )
        assert member.exists(), (
            "the member vanished even though nothing in this path may ever "
            "unlink it — the stand-in is not measuring the window it claims to"
        )
    finally:
        reservation.identities.clear()
        reservation.close()


def check_a_second_release_of_an_already_resolved_suffix_is_a_safe_no_op(
    tmpdir,
) -> None:
    """Re-entry for a suffix ``_release`` already resolved must not crash or reclaim.

    The presence check (``suffix in self.identities``) plus the pop on the way
    out make a second, ordinary call return ``None`` immediately rather than
    re-running anything against a descriptor this run already closed. With the
    unlink gone there is no unsound second identity check left to run, but a
    second call must still be inert: no double-close, no re-reported residue,
    no unlink, ever.
    """
    import hermes_cli.session_fence_rollback as sfr
    from hermes_cli.session_fence_rollback import _AcquiredDestinations

    where = pathlib.Path(tmpdir)
    backup = where / "backup.db"
    reservation = _AcquiredDestinations(backup)
    reservation.acquire()
    try:
        suffix = "-wal"
        member = reservation._member(suffix)
        spy = _UnlinkSpy()
        had_lib_os = hasattr(sfr, "os")
        previous_lib_os = getattr(sfr, "os", None)
        sfr.os = spy
        try:
            first = reservation._release(suffix)
            second = reservation._release(suffix)
        finally:
            if had_lib_os:
                sfr.os = previous_lib_os
            else:
                del sfr.os

        assert not spy.unlink_calls, (
            f"os.unlink was called {len(spy.unlink_calls)} time(s) across two "
            f"releases of the same suffix: {spy.unlink_calls!r}"
        )
        assert first is not None and first.get("error") == "cleanup_required", (
            f"the first release of a surviving member did not report it: "
            f"{first!r}"
        )
        assert second is None, (
            f"a second release of an already-resolved suffix re-ran and "
            f"reported something: {second!r}. It must be a no-op — the "
            f"suffix's fate was already decided"
        )
        assert suffix not in reservation.identities, (
            "the first release did not drop the suffix's claimed status"
        )
        assert member.exists(), (
            "the member vanished — nothing in this path may ever unlink it"
        )
    finally:
        reservation.identities.clear()
        reservation.close()


def check_a_sidecar_removed_by_something_else_before_release_is_reported_clean(
    tmpdir,
) -> None:
    """The ordinary ABSENT case: gone by the time release looks, and clean.

    Whether it was this run's own descriptor's owner closing out naturally
    (a checkpoint that removes a ``-wal``, say) or an operator's own cleanup,
    ``_release`` must report NOTHING WRONG when the name is already gone — no
    residue, no claim of agency, and certainly no attempt to unlink a name
    that resolves to nothing. Removing the unlink must not break this: the
    common ABSENT case is still a clean, non-adversarial result.
    """
    import hermes_cli.session_fence_rollback as sfr
    from hermes_cli.session_fence_rollback import _AcquiredDestinations

    where = pathlib.Path(tmpdir)
    backup = where / "backup.db"
    reservation = _AcquiredDestinations(backup)
    reservation.acquire()
    try:
        suffix = "-wal"
        member = reservation._member(suffix)
        # SOMETHING ELSE removed it first — this run performed no unlink to
        # get here, which is the whole point of the scenario.
        member.unlink()

        spy = _UnlinkSpy()
        had_lib_os = hasattr(sfr, "os")
        previous_lib_os = getattr(sfr, "os", None)
        sfr.os = spy
        try:
            problem = reservation._release(suffix)
        finally:
            if had_lib_os:
                sfr.os = previous_lib_os
            else:
                del sfr.os

        assert not spy.unlink_calls, (
            f"os.unlink was called {len(spy.unlink_calls)} time(s) resolving "
            f"an already-absent name: {spy.unlink_calls!r}"
        )
        assert problem is None, (
            f"a name that resolves to nothing was reported as residue: "
            f"{problem!r}. Absence is the obligation met, not an incident"
        )
    finally:
        reservation.identities.clear()
        reservation.close()


PINS = {
    "check_the_real_boundary_never_calls_unlink_and_reports_leftovers":
        check_the_real_boundary_never_calls_unlink_and_reports_leftovers,
    "check_a_swap_between_close_and_the_existence_check_is_never_deleted":
        check_a_swap_between_close_and_the_existence_check_is_never_deleted,
    "check_a_same_inode_swap_after_close_is_never_deleted":
        check_a_same_inode_swap_after_close_is_never_deleted,
    "check_a_base_exception_mid_release_leaves_the_suffix_claimed":
        check_a_base_exception_mid_release_leaves_the_suffix_claimed,
    "check_a_second_release_of_an_already_resolved_suffix_is_a_safe_no_op":
        check_a_second_release_of_an_already_resolved_suffix_is_a_safe_no_op,
    "check_a_sidecar_removed_by_something_else_before_release_is_reported_clean":
        check_a_sidecar_removed_by_something_else_before_release_is_reported_clean,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_sidecar_release_identity_property(name, tmp_path):
    """The pin. Asserted against the tree under test."""
    PINS[name](tmp_path)


#: THE ONE MUTATION EVERY UNLINK-SPY PIN MUST DIE BY. Reintroducing exactly
#: the call this whole file exists to remove, at exactly the place it used to
#: run. Registered once per pin that watches ``os.unlink`` directly, because
#: each pin's own assertion — not a shared crash detector — is what has to
#: fire.
_REINTRODUCE_UNLINK_FIND = (
    "        self._close_handle(suffix)\n"
    "        try:\n"
    "            os.lstat(member)\n"
    "        except FileNotFoundError:\n"
    "            return None\n"
    "        except OSError as exc:\n"
    '            return {"path": str(member), "files": 1,\n'
    '                    "error": f"{type(exc).__name__}: {exc}"}\n'
    '        return {"path": str(member), "files": 1, "error": "cleanup_required"}\n'
)
_REINTRODUCE_UNLINK_REPLACE = (
    "        self._close_handle(suffix)\n"
    "        try:\n"
    "            os.lstat(member)\n"
    "        except FileNotFoundError:\n"
    "            return None\n"
    "        except OSError as exc:\n"
    '            return {"path": str(member), "files": 1,\n'
    '                    "error": f"{type(exc).__name__}: {exc}"}\n'
    "        try:\n"
    "            os.unlink(member)\n"
    "        except OSError:\n"
    "            pass\n"
    "        return None\n"
)
_REINTRODUCE_UNLINK_WHY = (
    "reintroducing the unlink is exactly the defect this whole file exists "
    "to catch: whatever is at the reserved name after the descriptor closes "
    "gets deleted again"
)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_the_real_boundary_never_calls_unlink_and_reports_leftovers",
        module="hermes_cli/session_fence_rollback.py",
        find=_REINTRODUCE_UNLINK_FIND,
        replace=_REINTRODUCE_UNLINK_REPLACE,
        why=_REINTRODUCE_UNLINK_WHY + " — including every reserved sidecar "
            "placeholder on an ordinary, successful run",
        kills_by="the boundary called os.unlink",
    ),
    Mutation(
        pin="check_a_swap_between_close_and_the_existence_check_is_never_deleted",
        module="hermes_cli/session_fence_rollback.py",
        find=_REINTRODUCE_UNLINK_FIND,
        replace=_REINTRODUCE_UNLINK_REPLACE,
        why=_REINTRODUCE_UNLINK_WHY + ", including a stranger's file that "
            "arrived in the close-to-check window",
        kills_by="the swapped-in stranger did not survive the release",
    ),
    Mutation(
        pin="check_a_same_inode_swap_after_close_is_never_deleted",
        module="hermes_cli/session_fence_rollback.py",
        find=_REINTRODUCE_UNLINK_FIND,
        replace=_REINTRODUCE_UNLINK_REPLACE,
        why=_REINTRODUCE_UNLINK_WHY + ", including a stranger's file whose "
            "(st_dev, st_ino) only coincidentally matched the recorded one",
        kills_by="the release called os.unlink",
    ),
    Mutation(
        pin="check_a_base_exception_mid_release_leaves_the_suffix_claimed",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "        if suffix not in self.identities:\n"
            "            return None\n"
            "        member = self._member(suffix)\n"
            "        problem = self._resolve_release(suffix, member)\n"
            "        self.identities.pop(suffix, None)\n"
            "        return problem\n"
        ),
        replace=(
            "        identity = self.identities.pop(suffix, None)\n"
            "        if identity is None:\n"
            "            return None\n"
            "        member = self._member(suffix)\n"
            "        return self._resolve_release(suffix, member)\n"
        ),
        why="popping the suffix out of self.identities BEFORE the "
            "close/existence-check work runs is the exact hole this row "
            "exists to catch: a BaseException raised anywhere in that work (a "
            "KeyboardInterrupt is the real case) then leaves the suffix "
            "already gone from self.identities, looking resolved to "
            "remove_only_what_we_created even though its fate was never "
            "observed",
        kills_by="a BaseException raised mid-release popped",
    ),
    Mutation(
        pin="check_a_second_release_of_an_already_resolved_suffix_is_a_safe_no_op",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "        if suffix not in self.identities:\n"
            "            return None\n"
            "        member = self._member(suffix)\n"
            "        problem = self._resolve_release(suffix, member)\n"
            "        self.identities.pop(suffix, None)\n"
            "        return problem\n"
        ),
        replace=(
            "        member = self._member(suffix)\n"
            "        return self._resolve_release(suffix, member)\n"
        ),
        why="removing the presence check and the pop is what lets a second, "
            "ordinary call re-run the close/existence-check work against a "
            "descriptor this run already closed and popped from self.handles, "
            "instead of returning None as a no-op",
        kills_by="a second release of an already-resolved suffix re-ran and reported something",
    ),
    Mutation(
        pin="check_a_sidecar_removed_by_something_else_before_release_is_reported_clean",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "        try:\n"
            "            os.lstat(member)\n"
            "        except FileNotFoundError:\n"
            "            return None\n"
        ),
        replace=(
            "        try:\n"
            "            os.lstat(member)\n"
            "        except FileNotFoundError:\n"
            '            return {"path": str(member), "files": 1, "error": "vanished"}\n'
        ),
        why="a name that resolves to nothing is the obligation met, not an "
            "incident. Reporting it as residue sends the operator to a "
            "directory to remove a file that is not there",
        kills_by="a name that resolves to nothing was reported as residue",
    ),
)


@pytest.mark.parametrize(
    "mutation", SOURCE_MUTATIONS, ids=[m.pin for m in SOURCE_MUTATIONS]
)
def test_each_pin_dies_when_its_own_guard_is_removed(mutation, tmp_path):
    """Clean, mutated, restored. A pin that survives its mutation is not a pin."""
    assert_mutation_kills_the_pin(
        mutation, str(_SELF), tmp_path, *_EXTRA_EXTRACT
    )


def test_every_pin_has_a_mutation_that_kills_it():
    """No pin without a killer, and no killer without a pin."""
    assert_every_pin_has_a_killer(PINS, SOURCE_MUTATIONS)
