"""One copy table for "session storage unavailable / not written" across CLI, gateway and RPC.

Contract: the user gets a plain cause + the exact repair command, never the raw sqlite text as the
lead, plus a stable machine-readable `code` a GUI can key a "Run doctor" button on.
"""

import sqlite3

import pytest

from hermes_state_user_copy import describe_storage_failure, storage_failure_details


@pytest.mark.parametrize(
    ("exc", "code", "command"),
    [
        (sqlite3.OperationalError("database is locked"), "storage_locked", "try again"),
        (sqlite3.OperationalError("attempt to write a readonly database"), "storage_readonly", "hermes doctor --fix"),
        (sqlite3.DatabaseError("database disk image is malformed"), "storage_corrupt", "hermes doctor --fix"),
        (OSError(28, "No space left on device"), "disk_full", "Free some disk space"),
        (None, "storage_unavailable", "hermes doctor --fix"),
    ],
)
def test_each_cause_has_a_stable_code_and_an_action(exc, code, command):
    failure = describe_storage_failure(exc)
    assert failure.code == code
    assert command in failure.action
    assert failure.gloss
    # The raw sqlite wording never leaks into the user-facing gloss.
    assert "sqlite" not in failure.gloss.lower() and "OperationalError" not in failure.gloss


def test_a_refused_store_gets_its_cause_remedy_from_the_typed_exception(tmp_path):
    """A store this build refuses is not "could not be opened, run doctor --fix": each refusal cause
    has its own copy — but only from the typed exception. Gateway, TUI and CLI all now keep the
    exception object alongside its text (L4-4) instead of collapsing it to ``str(e)`` first, because
    the text alone can only ever QUOTE these phrases, never confirm them."""
    from hermes_state import SessionDB
    from tests.hermes_state.fork_store_fixture import REFUSED_KINDS, build_store

    unknown = describe_storage_failure(None)
    glosses = {}
    for kind, cause in REFUSED_KINDS.items():
        with pytest.raises(Exception) as caught:
            SessionDB(db_path=build_store(tmp_path / kind / "state.db", kind)).close()
        failure = describe_storage_failure(caught.value)
        assert failure.cause == "schema_incompatible" and failure.code != unknown.code, (kind, failure)
        assert failure.gloss[0].islower() and "doctor --fix" not in failure.action, (kind, failure)
        glosses.setdefault(cause, set()).add(failure.gloss)
    # Kinds that share a cause share its copy; different causes never do.
    assert all(len(g) == 1 for g in glosses.values()), glosses
    assert len({next(iter(g)) for g in glosses.values()}) == len(glosses), glosses


def test_a_refused_stores_text_alone_is_unknown_not_schema_incompatible(tmp_path):
    """Once the exception crosses to bare ``str(e)`` (no type, no __cause__ chain left), it can only
    ever QUOTE one of the refusal phrases above — it must not get the same cause/remedy as the real
    typed exception. This is the L4-4 fix: text alone is "unknown", a real refusal is
    "schema_incompatible"."""
    from hermes_state import SessionDB
    from tests.hermes_state.fork_store_fixture import REFUSED_KINDS, build_store

    for kind in REFUSED_KINDS:
        with pytest.raises(Exception) as caught:
            SessionDB(db_path=build_store(tmp_path / kind / "state.db", kind)).close()
        text_only_failure = describe_storage_failure(str(caught.value))
        assert text_only_failure.cause == "unknown", (kind, text_only_failure)


def test_undetermined_schema_refusal_is_not_defaulted_to_fence_mismatch(monkeypatch):
    """``describe_storage_failure`` used to default an undetermined schema sub-cause (the typed
    classifier returning None) straight to FENCE_GENERATION_MISMATCH, telling the user their store
    "belongs to a different Hermes version" even when nothing confirmed that. When the typed
    classifier cannot determine the sub-cause, the caller must get an explicit unknown/generic
    cause instead of the fence-specific one."""
    import hermes_state_user_copy
    from hermes_state_errors import SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH

    monkeypatch.setattr(hermes_state_user_copy, "classify_persistence_error", lambda _exc: "schema_incompatible")
    monkeypatch.setattr(hermes_state_user_copy, "schema_incompatibility_cause", lambda _exc: None)

    failure = describe_storage_failure(RuntimeError("some schema-incompatible-shaped error"))

    fence_mismatch_failure = hermes_state_user_copy.describe_schema_refusal(SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH)
    assert failure != fence_mismatch_failure, (
        "an undetermined schema cause must not be defaulted to the fence-generation-mismatch copy")
    assert "different hermes version" not in failure.gloss.lower()


def test_details_line_is_flattened_and_bounded():
    details = storage_failure_details("line one\n   line two " + "x" * 400, limit=60)
    assert "\n" not in details and len(details) == 60 and details.endswith("...")


def test_action_command_is_pinned_to_the_failing_profile(monkeypatch, tmp_path):
    """The action names the profile whose store failed (multi-profile backends serve sessions
    whose state.db is not the process default; a bare ``hermes`` follows active_profile)."""
    from hermes_constants import profile_cli_selector

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "research"))
    selector = profile_cli_selector()
    assert selector.strip()
    for exc in (sqlite3.OperationalError("database is locked"), sqlite3.DatabaseError("malformed"), None):
        action = describe_storage_failure(exc).action
        assert "{profile_arg}" not in action
        assert f"`hermes {selector}" in action and "`hermes doctor" not in action and "`hermes gateway" not in action
