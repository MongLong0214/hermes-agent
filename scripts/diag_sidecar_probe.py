"""Fork-CI-only probe: WHO deletes the foreign file at the reserved sidecar name.

DO NOT MERGE. Diagnostic only, never promoted.

``check_a_foreign_file_at_a_reserved_sidecar_is_never_deleted`` passes on macOS
and is reported failing on Linux. Before anything is changed, the question is
which code path performs the deletion there, because the two candidate answers
lead to different seams:

  A. production's ``_AcquiredDestinations._release`` unlinks it, meaning the
     identity check matched when it should not have — a real wrong-target
     deletion in a destructive verb;
  B. nothing in production unlinks it and the pin fails for another reason
     entirely — a fixture that does not hold on that platform.

Measured on macOS, for comparison, and printed again here so the two platforms
are read off the same instrument:

    every unlink of *backup.db-wal came from the TEST HOOK, not production
    the clean-path -wal at the backup destination is a 0-byte reservation whose
      inode never changes between acquire and release
    swap-at-every-lstat and swap-once-at-first-lstat give identical results

Nothing here asserts. It prints, and exits 0 whatever it finds, so a difference
is read from the log rather than inferred from a red X.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sqlite3
import sys
import tempfile
import traceback

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PIN_FILE = REPO_ROOT / "tests/hermes_cli/test_sessions_fence_rollback_verb.py"


def _load_pins():
    spec = importlib.util.spec_from_file_location("verbpins", PIN_FILE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["verbpins"] = module
    spec.loader.exec_module(module)
    return module


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def q0_wal_support() -> None:
    section("Q0  does this platform's SQLite accept WAL, and keep a -wal file")
    print("sqlite3.sqlite_version:", sqlite3.sqlite_version)
    tmp = pathlib.Path(tempfile.mkdtemp())
    path = tmp / "probe.db"
    conn = sqlite3.connect(str(path))
    mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    print("PRAGMA journal_mode=WAL ->", mode)
    conn.execute("CREATE TABLE t(x)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    print("-wal present after write ->", (tmp / "probe.db-wal").exists())
    conn.close()
    print("-wal present after close ->", (tmp / "probe.db-wal").exists())


def q1_store_journal_mode(pins) -> None:
    section("Q1  what journal mode does the FIXTURE STORE actually run in")
    tmp = pathlib.Path(tempfile.mkdtemp())
    pins._sandbox_home(tmp)
    store = tmp / "state.db"
    pins._fenced_store(store, leave_lease_live=False)
    conn = sqlite3.connect(str(store))
    print("store PRAGMA journal_mode ->", conn.execute("PRAGMA journal_mode").fetchone()[0])
    conn.close()
    for suffix in ("-wal", "-shm", "-journal"):
        sibling = store.with_name(store.name + suffix)
        print(f"  store{suffix} present ->", sibling.exists())


def q2_clean_path_timeline(pins) -> None:
    """The -wal inode at the BACKUP destination, across acquire -> release."""
    section("Q2  clean path: is a genuine -wal ever at the backup destination")
    from hermes_cli import session_fence_rollback as lib

    tmp = pathlib.Path(tempfile.mkdtemp())
    pins._sandbox_home(tmp)
    store = tmp / "state.db"
    pins._fenced_store(store, leave_lease_live=False)
    work_dir = tmp / "work"
    work_dir.mkdir()
    wal = work_dir.parent / "backup.db-wal"

    def snap(tag):
        if wal.exists():
            st = wal.lstat()
            print(f"  [{tag:34s}] -wal ino={st.st_ino} size={st.st_size}")
        else:
            print(f"  [{tag:34s}] -wal ABSENT")

    target = lib._AcquiredDestinations
    original_acquire = target.acquire
    original_release = target.release_the_sidecars
    original_release_one = target._release

    def acquire(self):
        snap("before acquire")
        original_acquire(self)
        snap("after acquire")
        print("     reserved -wal identity:", self.identities.get("-wal"))

    def release(self, *, outcome=None):
        snap("before release_the_sidecars")
        result = original_release(self, outcome=outcome)
        snap("after release_the_sidecars")
        return result

    def release_one(self, suffix):
        if suffix == "-wal":
            current = wal.lstat() if wal.exists() else None
            print(
                "     _release('-wal') reserved=",
                self.identities.get("-wal"),
                " onDisk=",
                (current.st_dev, current.st_ino) if current else None,
            )
        out = original_release_one(self, suffix)
        if suffix == "-wal":
            print("     _release('-wal') ->", out)
        return out

    target.acquire = acquire
    target.release_the_sidecars = release
    target._release = release_one
    try:
        run = pins._drive_the_boundary_with_unlink_failing(
            lib, store, work_dir, lambda path, op: None
        )
        snap("after the boundary returned")
        print("  returned a backup:", run["result"]["returned"] is not None)
        print("  crash:", run["result"]["crash"][:200])
        print("  residue:", pins._residue_errors(run["outcome"].facts()))
    finally:
        target.acquire = original_acquire
        target.release_the_sidecars = original_release
        target._release = original_release_one


def q3_who_unlinks(pins) -> None:
    """THE question. Every unlink of the sidecar name, with its call stack."""
    section("Q3  under the pin's own sabotage, WHO unlinks *backup.db-wal")
    unlinks = []
    real_unlink = os.unlink

    def spy(path, *args, **kwargs):
        text = str(path)
        if text.endswith("backup.db-wal"):
            stack = [
                f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}"
                for f in traceback.extract_stack()[-6:-1]
            ]
            unlinks.append(stack)
        return real_unlink(path, *args, **kwargs)

    os.unlink = spy
    verdict = ""
    try:
        tmp = pathlib.Path(tempfile.mkdtemp())
        pins.check_a_foreign_file_at_a_reserved_sidecar_is_never_deleted(tmp)
        verdict = "PIN PASSED"
    except AssertionError as exc:
        verdict = f"PIN FAILED: {exc}"
    except BaseException as exc:  # noqa: BLE001 - carried, not swallowed
        verdict = f"PIN CRASHED: {type(exc).__name__}: {exc}"
    finally:
        os.unlink = real_unlink

    print(verdict)
    print(f"\nunlink() calls on *backup.db-wal: {len(unlinks)}")
    for stack in unlinks:
        print("   <-", " | ".join(stack[-3:]))
    # CLASSIFY BY WHAT IS PRESENT, NOT BY WHAT IS ABSENT. The first version of
    # this excluded any stack mentioning the test file, and the harness's
    # _SabotagedOs.unlink wrapper lives in that file and sits on EVERY
    # production call — so production always looked like the hook and the
    # verdict line said B while the raw stacks said A.
    production = [
        s for s in unlinks
        if any("session_fence_rollback.py" in frame for frame in s)
    ]
    print(f"\nOF THOSE, FROM PRODUCTION (not the test hook): {len(production)}")
    for stack in production:
        print("   <-", " | ".join(stack))
    print(
        "\nVERDICT: "
        + (
            "A - production unlinked it (wrong-target deletion)"
            if production
            else "B - production never unlinked it on this platform"
        )
    )



def q4_inode_reuse() -> None:
    """Does unlink-then-recreate REUSE the inode number on this filesystem?

    This is the hypothesis the Linux stack points at. ``_release`` closes its
    descriptor FIRST, then lstats the name and compares ``(st_dev, st_ino)``
    against what it recorded at acquisition. Closing the descriptor is what
    lets the inode be recycled: while it was open the number could not be
    reused, so the comparison meant something; once closed, a different file
    created at that name can inherit the same number and the comparison
    silently answers "still ours".
    """
    section("Q4  does unlink+recreate reuse the inode number here")
    tmp = pathlib.Path(tempfile.mkdtemp())
    target = tmp / "reserve.db-wal"

    handle = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    reserved = os.fstat(handle)
    print(f"  reserved (descriptor OPEN)  dev={reserved.st_dev} ino={reserved.st_ino}")

    # A: swap while the descriptor is still open.
    target.unlink()
    target.write_bytes(b"a stranger, written while our fd is open")
    while_open = target.lstat()
    print(f"  stranger while fd OPEN      dev={while_open.st_dev} ino={while_open.st_ino}"
          f"  SAME_INO={while_open.st_ino == reserved.st_ino}")
    os.close(handle)
    target.unlink()

    # B: the shipped order — close first, then swap.
    handle = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    reserved2 = os.fstat(handle)
    os.close(handle)
    target.unlink()
    target.write_bytes(b"a stranger, written after our fd was closed")
    after_close = target.lstat()
    print(f"  reserved2 (before close)    dev={reserved2.st_dev} ino={reserved2.st_ino}")
    print(f"  stranger after fd CLOSED    dev={after_close.st_dev} ino={after_close.st_ino}"
          f"  SAME_INO={after_close.st_ino == reserved2.st_ino}")
    print("\n  IF SAME_INO is True in the closed case, (st_dev, st_ino) does not")
    print("  identify the file, and _release deletes a stranger believing it is ours.")

def main() -> int:
    print("platform:", sys.platform)
    print("python:", sys.version.split()[0])
    print("repo HEAD tree under test:", REPO_ROOT)
    if not PIN_FILE.exists():
        print("PIN FILE ABSENT ON THIS REF:", PIN_FILE)
        return 0
    pins = _load_pins()
    for step in (q0_wal_support, q4_inode_reuse):
        try:
            step()
        except BaseException as exc:  # noqa: BLE001
            print(f"{step.__name__} raised: {type(exc).__name__}: {exc}")
    for step in (q1_store_journal_mode, q2_clean_path_timeline, q3_who_unlinks):
        try:
            step(pins)
        except BaseException as exc:  # noqa: BLE001
            print(f"{step.__name__} raised: {type(exc).__name__}: {exc}")
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
