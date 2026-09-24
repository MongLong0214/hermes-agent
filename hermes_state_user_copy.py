"""Plain-language copy for "session storage is unavailable / could not be written" notices.

One table keyed by ``classify_persistence_error``'s cause bucket feeds every surface (CLI banner,
gateway home-channel warning, TUI/Desktop RPC errors) so they agree on what happened, what to do,
and a machine-readable ``code`` a GUI can attach a "Run doctor" button to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hermes_state_errors import (
    SCHEMA_CAUSE_BUILD_TOO_OLD, SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, SCHEMA_CAUSE_VERSION_UNREADABLE,
    STORAGE_RECOVERY_DOCS_URL, classify_persistence_error, incompatible_schema_cause, is_disk_full_error,
)


@dataclass(frozen=True)
class StorageFailure:
    cause: str      # classify_persistence_error bucket
    code: str       # machine-readable, stable: storage_locked | storage_readonly | storage_corrupt | disk_full | ...
    gloss: str      # what happened, one clause, lowercase start
    action: str     # what to do, one sentence naming the exact command


_DOCTOR = "Run `hermes {profile_arg}doctor --fix` to diagnose and repair."

# cause -> (code, gloss, action). "disk" is split by is_disk_full_error at lookup time.
_STORAGE_FAILURES: dict[str, tuple[str, str, str]] = {
    "locked": (
        "storage_locked",
        "the session database is locked by another Hermes process",
        "Wait a moment and try again; if it persists, stop the other Hermes process "
        "(`hermes {profile_arg}gateway stop`).",
    ),
    "disk_full": (
        "disk_full",
        "the disk is full",
        "Free some disk space, then try again.",
    ),
    "disk": (
        "storage_readonly",
        "the session database file is read-only or not writable",
        _DOCTOR,
    ),
    "corrupt": (
        "storage_corrupt",
        "the session database file is damaged",
        _DOCTOR + " Recovery: `hermes {profile_arg}sessions recover --source <state.db> --inspect-only`.",
    ),
    "fts_index": (
        "storage_index_corrupt",
        "the session search index is damaged (the messages themselves are intact)",
        "Run `hermes {profile_arg}doctor --fix` (or `hermes {profile_arg}sessions repair`) to rebuild it.",
    ),
    "replaced": (
        "storage_replaced",
        "the session database file was replaced while Hermes was running",
        "Stop Hermes (`hermes {profile_arg}gateway stop`), run `hermes {profile_arg}doctor`, then start it again.",
    ),
    # Code stays `storage_replaced` (GUI clients key on it); the copy names the real remedy: every writer
    # on the profile must stop, doctor names the ones still holding the retired log (#110054).
    "deleted_wal": (
        "storage_replaced",
        "another Hermes process still holds an old copy of the session database's write-ahead log, "
        "so Hermes stopped writing to keep the file safe",
        "Nothing is lost. Quit every Hermes process on this profile (Desktop app, "
        "`hermes {profile_arg}gateway stop`, dashboard, cron), run `hermes {profile_arg}doctor` — it names "
        "any process still holding the log — then start Hermes again. Do not run `doctor --fix` or delete "
        "any state.db files while they run. Guide: " + STORAGE_RECOVERY_DOCS_URL,
    ),
    "compression": (
        "storage_busy",
        "another process is compressing this session",
        "Send your message again once compression finishes.",
    ),
    "compression_closed": (
        "storage_session_rotated",
        "this session was rotated by context compression",
        "Refresh the client (or start a new turn) and send your message again.",
    ),
    "turn_lease": (
        "storage_busy",
        "another Hermes process took over this session",
        "Wait for it to finish, then send your message again.",
    ),
    "unknown": (
        "storage_unavailable",
        "the session database could not be opened",
        _DOCTOR,
    ),
    # "schema_incompatible" is split by the refusal's own cause at lookup time. The store is healthy for
    # the build that wrote it, so no remedy is `doctor --fix` or `sessions repair`: the first only warns on
    # such a store and the second reports it clean, which sends the owner in a circle (72e525838f).
    SCHEMA_CAUSE_BUILD_TOO_OLD: (
        "storage_schema_incompatible",
        "this profile's session history was saved by a newer version of Hermes, so this version "
        "will not open it and has changed nothing",
        "Update Hermes (`hermes update`) or switch back to the newer version.",
    ),
    SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH: (
        "storage_schema_incompatible",
        "this profile's session history belongs to a different Hermes version, so this version "
        "will not write to it and has changed nothing",
        "Use the Hermes version that last ran on this profile, or update this one (`hermes update`). "
        "`hermes {profile_arg}doctor` shows the details; repair commands will not change this.",
    ),
    SCHEMA_CAUSE_VERSION_UNREADABLE: (
        "storage_schema_incompatible",
        "this profile's session history does not record a readable format version, so no version "
        "of Hermes will open it; nothing in it was changed",
        "Check it with `hermes {profile_arg}sessions recover --source <state.db> --inspect-only`, then "
        "rebuild it into a new database with `--output <new-file>`. "
        "`hermes {profile_arg}sessions repair` does not fix this.",
    ),
}


def schema_incompatibility_cause(exc_or_str) -> Optional[str]:
    """The refusal cause when *exc_or_str* is this build refusing a store another build owns, else None.

    A write the store's fence aborted, or one made without the fence function, is a generation
    mismatch: the store's fences name a build other than this one."""
    cause = incompatible_schema_cause(exc_or_str)
    if cause is None:
        from hermes_state_fence import fence_refusal_verdict

        if exc_or_str is not None and fence_refusal_verdict(exc_or_str) is not None:
            cause = SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH
    return cause


def _storage_failure(cause: str, key: str) -> StorageFailure:
    code, gloss, action = _STORAGE_FAILURES.get(key, _STORAGE_FAILURES["unknown"])
    # Pin the copy-pasteable command to the failing profile — see profile_cli_selector.
    from hermes_constants import profile_cli_selector

    return StorageFailure(
        cause=cause, code=code, gloss=gloss, action=action.replace("{profile_arg}", profile_cli_selector())
    )


def describe_schema_refusal(refusal_cause: str) -> StorageFailure:
    """The copy for one ``IncompatibleSchemaError`` cause. A retry repeats the refusal, so a surface
    that knows it has one shows this instead of its own retry text."""
    return _storage_failure("schema_incompatible", refusal_cause)


def describe_storage_failure(exc_or_str) -> StorageFailure:
    """Plain-language description of a persistence failure (never raises)."""
    cause = classify_persistence_error(exc_or_str)
    if cause == "schema_incompatible":
        return describe_schema_refusal(
            schema_incompatibility_cause(exc_or_str) or SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH)
    key = "disk_full" if cause == "disk" and is_disk_full_error(exc_or_str) else cause
    return _storage_failure(cause, key)


def storage_failure_details(exc_or_str, limit: int = 200) -> str:
    """Raw cause for a trailing, secondary "Details:" line (never the lead sentence)."""
    text = " ".join(str(exc_or_str or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
