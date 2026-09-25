"""Forward-migration admission for SessionDB's writable open, and for every raw state.db writer.

Migrating a store another build owns (fence swap, lineage stamp, data migrations) locks that build
out of every governed table, and a gateway of that build may still be running on the profile.
Every gateway holds its home's ``gateway.lock`` (an OS lock the kernel drops with the process) for
as long as it runs, whatever its build, so such a migration runs only under a lease on that lock
(``gateway.status_migration_lease``), and a held lock refuses the open before any byte is written.
A raw opener (``hermes_state_fence.open_fenced_state_connection``: the delivery ledger, async
delegation, A2A) runs DDL and writes on the same file, so it takes the same lease the same way.

The home is the store's own directory: a gateway's lock and its ``state.db`` sit side by side, so
no profile scope has to be bound to find it. A database with another name has no gateway.

The lease is checked again right before the open's first DDL and under the write lock that
admits the fence swap. The lease writes no gateway record, so an unscoped ``get_running_pid()``
of any build unlinks the lock file it holds as stale, and the next gateway claims a new file.
A lease lost that way is re-taken on the new file, and refuses the open while another process
holds it.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from hermes_state_fence import (
    LINEAGE_FENCED, TURN_FENCE_GENERATION, StoreLineage, decode_store_lineage, probe_store_lineage,
)

FORWARD_MIGRATION_ADMISSION_BLOCKED = "STATE_DB_FORWARD_MIGRATION_ADMISSION_BLOCKED"
_STATE_DB_NAME = "state.db"


class ForwardSchemaMigrationAdmissionError(RuntimeError):
    """A gateway holds the runtime lock of a store this open would migrate away from its build."""

    code = FORWARD_MIGRATION_ADMISSION_BLOCKED

    def __init__(self, lineage: StoreLineage, lock_path: Path):
        from hermes_constants import profile_name_for_home

        self.lineage, self.lock_path = lineage, lock_path
        self.actual_generation, self.expected_generation = lineage.stored, TURN_FENCE_GENERATION
        name = profile_name_for_home(lock_path.parent)
        profile_arg = f"-p {name} " if name and name != "default" else ""
        foreign = sorted(lineage.fence_literals - {TURN_FENCE_GENERATION}) if lineage.lineage == LINEAGE_FENCED else []
        fences = f", turn-fence generation {', '.join(map(str, foreign))}" if foreign else ""
        # classify_persistence_error keys on "running Hermes gateway owns", which survives the
        # "Type: message" text _last_init_error keeps. No "locked"/"busy": text-matching SQLite
        # contention retries must not spin on a refusal that lasts until the gateway stops.
        super().__init__(
            "A running Hermes gateway owns this profile's session history, which is still in another "
            f"build's format ({lineage.lineage} lineage, stored generation {lineage.stored}{fences}); this "
            "version will not upgrade it while that gateway runs, and nothing was changed (its runtime "
            f"lock is held: {lock_path}). Stop that gateway first (`hermes {profile_arg}gateway stop`), "
            "then start this version again."
        )


def _runtime_lock_path(db_path: Path) -> Optional[Path]:
    if Path(db_path).name != _STATE_DB_NAME:
        return None
    from gateway.status_migration_lease import gateway_runtime_lock_path_for_home

    return gateway_runtime_lock_path_for_home(Path(db_path).parent)


def _acquire_lease(db_path: Path, lineage: StoreLineage, *, reprobe: bool):
    if not lineage.owned_by_another_build:
        return None
    lock_path = _runtime_lock_path(db_path)
    if lock_path is None:
        return None
    from gateway.status_migration_lease import (
        acquire_gateway_runtime_migration_lease, release_gateway_runtime_migration_lease,
    )

    lease = acquire_gateway_runtime_migration_lease(lock_path)
    if lease is None:
        raise ForwardSchemaMigrationAdmissionError(lineage, lock_path)
    if not reprobe:
        return lease
    try:
        # The first read only chose this branch: another opener may have migrated the store before
        # the lease was taken, and then there is nothing left to hold the lock for.
        if probe_store_lineage(db_path).owned_by_another_build:
            return lease
    except BaseException:
        release_gateway_runtime_migration_lease(lease)
        raise
    release_gateway_runtime_migration_lease(lease)
    return None


def admit_forward_migration(db, lineage: StoreLineage, *, reprobe: bool = True) -> None:
    """Hold *db*'s admission lease when opening it migrates *lineage* away from another build.

    ``reprobe=False`` is for a lineage decoded on the open's own connection, which is already
    current. A lease already held counts only while its lock is on the file now at the lock path;
    one whose file was unlinked is re-taken on the new file. Raises
    ForwardSchemaMigrationAdmissionError while another process (that build's gateway) holds it."""
    lease = db._forward_migration_lease
    if lease is not None:
        from gateway.status_migration_lease import lease_holds_lock_file

        if not lease_holds_lock_file(lease):
            release_forward_migration_lease(db)
    if db._forward_migration_lease is None:
        db._forward_migration_lease = _acquire_lease(db.db_path, lineage, reprobe=reprobe)


def readmit_forward_migration(db, cursor) -> None:
    """Right before the open's first DDL: a lease lost since the probe is re-taken, or the open is
    refused with no byte written. The fence transaction admits once more under the write lock."""
    if db._forward_migration_lease is not None:
        admit_forward_migration(db, decode_store_lineage(cursor), reprobe=False)


def release_forward_migration_lease(db) -> None:
    lease, db._forward_migration_lease = db._forward_migration_lease, None
    if lease is not None:
        from gateway.status_migration_lease import release_gateway_runtime_migration_lease

        release_gateway_runtime_migration_lease(lease)


class _RawOpen:
    """The lease slot SessionDB keeps for its open, kept for one raw opener's open."""

    def __init__(self, db_path):
        self.db_path, self._forward_migration_lease = Path(db_path), None


@contextmanager
def raw_open_admission(db_path, lineage: StoreLineage):
    """SessionDB's admission for a raw state.db opener: taken before its connect when *lineage* is
    another build's, held to the end of its initialize. Yields ``readmit(cursor)`` for right before
    the first DDL. Raises ForwardSchemaMigrationAdmissionError while another process holds the lock.
    The raw open itself never migrates (only its SessionDB bootstrap does); its DDL is additive."""
    opener = _RawOpen(db_path)
    admit_forward_migration(opener, lineage)
    try:
        yield lambda cursor: readmit_forward_migration(opener, cursor)
    finally:
        release_forward_migration_lease(opener)
