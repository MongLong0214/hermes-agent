"""``(st_dev, st_ino)`` identifies a file only while a descriptor pins it.

``_AcquiredDestinations._release`` hands back one reserved NAME. It must not
delete a file this run did not create, so it compares the ``(st_dev, st_ino)``
recorded at ``acquire`` against what the name resolves to now, and refuses with
``ownership-lost`` when they differ.

The comparison is sound only while the descriptor from ``acquire`` is still
open. An inode NUMBER is not an identity — it is an index into a table, and the
kernel is free to hand it to the next file created once nothing refers to it.
An open descriptor is what refers to it. ``_release`` closes first:

    handle = self.handles.pop(suffix, None)
    if handle is not None:
        os.close(handle)            # <- the inode is now free for reuse
    info = os.lstat(member)         # <- and this may be a DIFFERENT file
    if identity is not None and (info.st_dev, info.st_ino) != identity:
        return ...ownership-lost
    os.unlink(member)               # <- deleting a file this run did not create

MEASURED, NOT ARGUED — a fork-only CI probe on ubuntu-24.04 / ext4:

    reserved (descriptor OPEN)  dev=2049 ino=9209326
    stranger while fd OPEN      dev=2049 ino=9209327  SAME_INO=False
    reserved2 (before close)    dev=2049 ino=9209326
    stranger after fd CLOSED    dev=2049 ino=9209326  SAME_INO=True

and on that platform the deleting call stack is production's own:

    _commit_the_rollback -> _make_verified_backup -> release_the_sidecars
      -> _release -> unlink

WHY THIS FILE PINS AN ORDER RATHER THAN THE OUTCOME
    The harm — a stranger's file deleted — only reproduces where the filesystem
    recycles inode numbers. APFS does not, so on macOS the shipped order looks
    correct and an outcome pin is GREEN here for a reason that has nothing to
    do with the code being right. A pin that can only fail on one platform is
    not much of a pin, and "it passed locally" would keep meaning nothing.

    The defect itself is not platform-specific: it is the ORDER of close and
    check, and that is observable everywhere. So this asserts the order, on the
    real object, driven through the real boundary. It fails on macOS and Linux
    alike before the fix, and the ext4 measurement above is what says why the
    order matters.
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


def check_the_identity_check_runs_while_the_descriptor_still_pins_the_inode(
    tmpdir,
) -> None:
    """For every reserved name: lstat BEFORE close, never after.

    Recorded on the real ``_release``, driven through the real backup boundary,
    by watching the ``os`` the library actually calls.
    """
    pins = _load_verb_pins()
    from hermes_cli import session_fence_rollback as lib

    where = pathlib.Path(tmpdir)
    pins._sandbox_home(where)
    store = where / "state.db"
    pins._fenced_store(store, leave_lease_live=False)
    work_dir = where / "work"
    work_dir.mkdir()

    #: (operation, suffix) in the order the library performed them.
    events: list = []
    sidecars = ("", "-wal", "-shm", "-journal")
    backup_name = "backup.db"

    def _suffix_of(path) -> str:
        text = str(path)
        for suffix in ("-wal", "-shm", "-journal"):
            if text.endswith(backup_name + suffix):
                return suffix
        return "" if text.endswith(backup_name) else None

    real_close, real_lstat = os.close, os.lstat
    handles: dict = {}

    class _Watched:
        """Records the calls that matter and forwards everything else."""

        def __getattr__(self, name):
            return getattr(os, name)

        def open(self, path, *args, **kwargs):
            handle = os.open(path, *args, **kwargs)
            suffix = _suffix_of(path)
            if suffix is not None:
                handles[handle] = suffix
            return handle

        def close(self, handle, *args, **kwargs):
            suffix = handles.pop(handle, None)
            if suffix is not None:
                events.append(("close", suffix))
            return real_close(handle, *args, **kwargs)

        def lstat(self, path, *args, **kwargs):
            suffix = _suffix_of(path)
            if suffix is not None:
                events.append(("lstat", suffix))
            return real_lstat(path, *args, **kwargs)

    # The boundary is driven here rather than through
    # ``_drive_the_boundary_with_unlink_failing``, which installs its OWN
    # ``library.os`` and would silently replace this watcher — the observation
    # would then be empty and the ordering assertion vacuously true.
    import hermes_state_common

    backup = work_dir.parent / backup_name
    outcome = lib.RollbackOutcome()
    returned, crash = None, ""
    had_os = hasattr(lib, "os")
    previous = getattr(lib, "os", None)
    lib.os = _Watched()
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
    assert returned is not None, (
        "the clean path did not produce a backup, so the release under test "
        "never ran"
    )

    # The fixture has to have SEEN both operations, or the ordering assertion
    # below is vacuously true — the failure mode this whole family exists for.
    seen = {operation for operation, _ in events}
    assert {"close", "lstat"} <= seen, (
        f"the watcher never observed both a close and an lstat on the backup "
        f"family, so it is not measuring the release at all: {events!r}"
    )

    for suffix in sidecars:
        order = [operation for operation, which in events if which == suffix]
        if "lstat" not in order or "close" not in order:
            continue
        assert order.index("lstat") < order.index("close"), (
            f"for the reserved name {backup_name + suffix!r}, the release "
            f"CLOSED its descriptor before checking identity. An inode number "
            f"is not an identity — it is reusable the moment nothing refers to "
            f"it, and the descriptor is what refers to it. Measured on "
            f"ubuntu-24.04/ext4: a file created at that name after the close "
            f"gets the SAME inode number (SAME_INO=True), so the "
            f"(st_dev, st_ino) comparison answers 'still ours' about a "
            f"stranger's file and the unlink below it destroys that file. With "
            f"the descriptor still open the number cannot be recycled "
            f"(SAME_INO=False). APFS never recycles, which is the only reason "
            f"this looks correct on macOS.\\n"
            f"  observed order for {suffix!r}: {order}"
        )


def check_a_double_release_never_checks_identity_with_the_descriptor_already_closed(
    tmpdir,
) -> None:
    """A suffix ``release_the_sidecars`` already resolved must not be re-entered.

    ``release_the_sidecars`` raises on the first problem it meets, and the
    caller's ``except BaseException`` then calls ``remove_only_what_we_created``
    on the SAME reservation. Before the suffix was claimed the instant a call
    entered ``_release``, that method ran a second time for a suffix the first
    call had already resolved — with the descriptor the first call closed
    already gone from ``self.handles``, so the second call's identity check
    ran with nothing left pinning the inode. That is not a smaller version of
    the earlier close-after-check fix; it is the same defect one call later,
    because the object that fix depends on (an open descriptor) had already
    been given up.

    Counted at the METHOD, not by the syscalls a single call happens to make:
    the shipped fix for the swap-in-the-window hole (a second, revalidating
    ``lstat`` right before the unlink) means even ONE correct call now does
    two ``lstat``s, so counting syscalls cannot tell "one call, two checks"
    from "two calls, two checks" apart. Counting entries into ``_release``
    itself for the suffix under test can.

    Driven through the REAL double-release path: a pinned ``-wal`` sidecar
    makes ``release_the_sidecars`` fail, and the exception handling that
    follows is production's own, not a simulation of it.
    """
    pins = _load_verb_pins()
    from hermes_cli import session_fence_rollback as lib

    where = pathlib.Path(tmpdir)
    pins._sandbox_home(where)
    store = where / "state.db"
    pins._fenced_store(store, leave_lease_live=False)
    work_dir = where / "work"
    work_dir.mkdir()

    backup_name = "backup.db"

    def _suffix_of(path) -> str:
        text = str(path)
        for suffix in ("-wal", "-shm", "-journal"):
            if text.endswith(backup_name + suffix):
                return suffix
        return "" if text.endswith(backup_name) else None

    real_unlink = os.unlink

    class _PinnedWalUnlink:
        """Pins the ``-wal`` unlink shut, forwarding everything else.

        The pinned unlink is what forces ``release_the_sidecars`` to fail for
        exactly one suffix, which is what drives the exception handler into
        calling ``remove_only_what_we_created`` for the same reservation — the
        real double-release path, not a synthetic call into ``_release``.
        """

        def __getattr__(self, name):
            return getattr(os, name)

        def unlink(self, path, *args, **kwargs):
            if _suffix_of(path) == "-wal":
                raise PermissionError("pinned sidecar (probe)")
            return real_unlink(path, *args, **kwargs)

    import hermes_state_common

    release_calls: list = []
    real_release = lib._AcquiredDestinations._release

    def _counting_release(self, suffix):
        release_calls.append(suffix)
        return real_release(self, suffix)

    backup = work_dir.parent / backup_name
    outcome = lib.RollbackOutcome()
    returned, crash = None, ""
    had_os = hasattr(lib, "os")
    previous = getattr(lib, "os", None)
    lib.os = _PinnedWalUnlink()
    lib._AcquiredDestinations._release = _counting_release
    try:
        copy = lib.prepare_the_private_copy(store, work_dir=work_dir)
        try:
            returned = lib._commit_the_rollback(
                copy, backup, sorted(hermes_state_common.TURN_FENCE_TRIGGERS),
                report_as=store, outcome=outcome,
            )
        except lib.TurnFenceRollbackRefused:
            returned = None
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        crash = f"{type(exc).__name__}: {exc}"
    finally:
        lib._AcquiredDestinations._release = real_release
        if had_os:
            lib.os = previous
        else:
            del lib.os

    assert not crash, f"the boundary crashed, so nothing below was measured: {crash}"
    assert returned is None, (
        "a pinned sidecar did not refuse the run, so release_the_sidecars "
        f"never failed and the double-release path never ran: {returned!r}"
    )

    wal_entries = release_calls.count("-wal")
    assert wal_entries >= 1, (
        f"_release was never entered for '-wal', so this pin measures "
        f"nothing: {release_calls!r}"
    )
    assert wal_entries == 1, (
        f"_release was entered {wal_entries} times for the SAME suffix "
        f"'-wal': release_the_sidecars() made the first attempt and failed "
        f"(the sidecar is pinned), and the exception-cleanup path "
        f"(remove_only_what_we_created) then re-entered _release for a "
        f"suffix release_the_sidecars had ALREADY resolved. The first call "
        f"already closed that suffix's descriptor, so the second call's "
        f"identity check runs with nothing pinning the inode — exactly the "
        f"state the earlier close-after-check fix was supposed to make "
        f"unreachable.\n  entries: {release_calls!r}"
    )


def check_a_swap_between_the_identity_check_and_the_unlink_is_still_caught(
    tmpdir,
) -> None:
    """A name swapped for a stranger AFTER the check, before the unlink, survives.

    ``_release`` runs its identity check ONCE, while the descriptor from
    ``acquire`` still pins the inode, then closes that descriptor — mandatory
    ahead of the unlink, since an open handle blocks the delete on some
    platforms — and only then unlinks by PATH. Nothing pins the inode from the
    close onward, so a name swapped in that window used to reach the unlink
    unexamined: the FIRST check, made while the swap had not happened yet,
    was the only one ever made.

    FAKE-OS PROBE, same shape as the swap sol demonstrated: real files, a real
    ``_release`` call, and a swap timed to land exactly between the close and
    the unlink by hooking ``os.close`` for the one descriptor under test — the
    step that already has to run there on every call, not a step invented for
    the probe.
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
        stranger = b"a different file that arrived in the check-to-unlink window"

        real_os = os
        real_close = os.close

        class _FakeOS:
            def __getattr__(self, name):
                return getattr(real_os, name)

            def close(self, handle, *args, **kwargs):
                result = real_close(handle, *args, **kwargs)
                if handle == wal_handle:
                    # THE WINDOW: the (only) identity check already ran, with
                    # THIS descriptor still open, and matched. Nothing pins
                    # the inode from here to the unlink below — a real actor
                    # with write access to the parent directory can replace
                    # the name in exactly this gap.
                    member.unlink()
                    member.write_bytes(stranger)
                return result

        had_lib_os = hasattr(sfr, "os")
        previous_lib_os = getattr(sfr, "os", None)
        sfr.os = _FakeOS()
        try:
            problem = reservation._release("-wal")
        finally:
            if had_lib_os:
                sfr.os = previous_lib_os
            else:
                del sfr.os

        assert member.exists() and member.read_bytes() == stranger, (
            f"the swapped-in stranger did not survive the release: "
            f"exists={member.exists()!r}. A name that changed between the "
            f"check and the unlink was deleted anyway — the reservation "
            f"deletes by PATH, and a path is not the file it named a moment "
            f"earlier.\n  release() returned: {problem!r}"
        )
        assert problem is not None and problem.get("error") == "ownership-lost", (
            f"the release did not report the swap as a lost ownership: "
            f"{problem!r}"
        )
    finally:
        reservation.close()


PINS = {
    "check_the_identity_check_runs_while_the_descriptor_still_pins_the_inode":
        check_the_identity_check_runs_while_the_descriptor_still_pins_the_inode,
    "check_a_double_release_never_checks_identity_with_the_descriptor_already_closed":
        check_a_double_release_never_checks_identity_with_the_descriptor_already_closed,
    "check_a_swap_between_the_identity_check_and_the_unlink_is_still_caught":
        check_a_swap_between_the_identity_check_and_the_unlink_is_still_caught,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_sidecar_release_identity_property(name, tmp_path):
    """The pin. Asserted against the tree under test."""
    PINS[name](tmp_path)


SOURCE_MUTATIONS = (
    Mutation(
        pin="check_the_identity_check_runs_while_the_descriptor_still_pins_the_inode",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "        info = self._identity_check_target(suffix, member)\n"
        ),
        replace=(
            "        self._close_handle(suffix)\n"
            "        info = self._identity_check_target(suffix, member)\n"
        ),
        why="closing the descriptor before the identity check is the defect: "
            "the inode number becomes reusable, ext4 hands it straight back to "
            "the next file created at that name, and the comparison then "
            "authorises deleting a file this run did not create",
        kills_by="CLOSED its descriptor before checking identity",
    ),
    Mutation(
        pin="check_a_double_release_never_checks_identity_with_the_descriptor_already_closed",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "        identity = self.identities.pop(suffix, None)\n"
            "        if identity is None:\n"
            "            return None\n"
            "        member = self._member(suffix)\n"
        ),
        replace=(
            "        identity = self.identities.get(suffix)\n"
            "        member = self._member(suffix)\n"
        ),
        why="claiming the suffix by popping it out of self.identities the "
            "instant _release is entered is what makes a second call for the "
            "same suffix a no-op. Reverting to a plain .get() lets the "
            "exception-cleanup path (remove_only_what_we_created) re-run the "
            "identity check for a suffix release_the_sidecars already closed "
            "the descriptor for — an lstat with nothing left pinning the "
            "inode, the exact state the earlier close-after-check fix was "
            "supposed to make unreachable",
        kills_by="then re-entered _release for a suffix release_the_sidecars had ALREADY resolved",
    ),
    Mutation(
        pin="check_a_swap_between_the_identity_check_and_the_unlink_is_still_caught",
        module="hermes_cli/session_fence_rollback.py",
        find=(
            "        # A SECOND check, immediately before the unlink. The descriptor closed\n"
            "        # above no longer pins the inode, and the unlink below still names the\n"
            "        # file by PATH — POSIX has no call that unlinks \"this exact inode at\n"
            "        # this name\" atomically, and holding the descriptor open across the\n"
            "        # unlink is not on the table either: the close has to happen first,\n"
            "        # which is the same ordering Windows requires. So this does not CLOSE\n"
            "        # the window between the first check and the unlink — nothing built on\n"
            "        # unlink-by-path can — it SHRINKS it, from \"close, a comparison and a\n"
            "        # dict pop\" down to the instant right before the call that destroys\n"
            "        # the file. A name that changed since the FIRST check is caught here\n"
            "        # instead of silently unlinked.\n"
            "        try:\n"
            "            revalidated = os.lstat(member)\n"
            "        except FileNotFoundError:\n"
            "            return None\n"
            "        except OSError as exc:\n"
            "            return {\"path\": str(member), \"files\": 1,\n"
            "                    \"error\": f\"{type(exc).__name__}: {exc}\"}\n"
            "        if (revalidated.st_dev, revalidated.st_ino) != identity:\n"
            "            return {\"path\": str(member), \"files\": 1, \"error\": \"ownership-lost\"}\n"
        ),
        replace="",
        why="the second, immediately-before-the-unlink identity check is the "
            "only thing standing in the window between the (single, "
            "fd-pinned) first check and the unlink-by-path. Removing it "
            "restores the shape sol's probe found: a name swapped after the "
            "first check and before the unlink is deleted without a second "
            "look, because nothing looks a second time",
        kills_by="the swapped-in stranger did not survive the release",
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
