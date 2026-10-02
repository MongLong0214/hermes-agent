"""Plain-language contracts for hermes_state user-facing errors (CLI UX message campaign, cluster D)."""

import hermes_state
from hermes_state import format_session_db_unavailable
from hermes_state_errors import (
    SCHEMA_CAUSE_BUILD_TOO_OLD,
    SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH,
    SCHEMA_CAUSE_VERSION_UNREADABLE,
    IncompatibleSchemaError,
)








def test_db_unavailable_is_one_line_for_chat_surfaces_by_default():
    hermes_state._set_last_init_error("OperationalError: database is locked")
    try:
        text = format_session_db_unavailable(prefix="Cannot resume")
    finally:
        hermes_state._set_last_init_error(None)
    assert "\n" not in text
    assert "Details:" not in text
    assert text.startswith("Cannot resume:")






def test_db_unavailable_commands_are_pinned_to_the_failing_profile(monkeypatch, tmp_path):
    """Both fallbacks that bypass the shared cause table (no cause; network-drive gloss) name the
    profile whose store failed, like the table's actions do."""
    from hermes_constants import profile_cli_selector

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes" / "profiles" / "research"))
    selector = profile_cli_selector()
    assert selector.strip()
    hermes_state._set_last_init_error(None)
    no_cause = format_session_db_unavailable()
    hermes_state._set_last_init_error("OperationalError: locking protocol")
    try:
        network = format_session_db_unavailable()
    finally:
        hermes_state._set_last_init_error(None)
    for text in (no_cause, network):
        assert f"`hermes {selector}doctor`" in text and "`hermes doctor`" not in text
        assert "{profile_arg}" not in text


def test_db_unavailable_details_line_carries_raw_cause_only_below_the_lead():
    """CLI banner (details=True): the raw SQLite text is kept for bug reports on
    its own ``Details:`` line and never leaks into the plain-language lead; with
    no recorded cause there is nothing to detail."""
    hermes_state._set_last_init_error("OperationalError: database is locked")
    try:
        text = format_session_db_unavailable(details=True)
    finally:
        hermes_state._set_last_init_error(None)
    lead, *rest = text.splitlines()
    assert "OperationalError" not in lead
    assert len(rest) == 1 and rest[0].startswith("Details: ") and "database is locked" in rest[0]

    no_cause = format_session_db_unavailable(details=True)
    assert "\n" not in no_cause and "Details:" not in no_cause


def test_db_unavailable_network_drive_cause_does_not_offer_doctor_fix():
    """A locking-protocol failure means the DB sits on a network filesystem that
    cannot host SQLite's WAL; `hermes doctor --fix` cannot repair a mount, so it
    must not be the offered action (only moving the file helps)."""
    hermes_state._set_last_init_error("OperationalError: locking protocol")
    try:
        text = format_session_db_unavailable(details=True)
    finally:
        hermes_state._set_last_init_error(None)
    lead, details = text.splitlines()
    assert "doctor --fix" not in lead
    assert "locking protocol" not in lead and "locking protocol" in details


# L4-4: bare text that only QUOTES a fence/version phrase must not be read as a confirmed
# schema-version refusal — only a real typed exception (IncompatibleSchemaError, or a confirmed
# fence abort) earns the "switch/update Hermes" copy. Exercises the actual constructor/formatter
# path: _set_last_init_error (what SessionDB.__init__ calls on failure) then
# format_session_db_unavailable (what the CLI/tools surface reads).
_SCHEMA_HEAD_CASES = [
    (SCHEMA_CAUSE_BUILD_TOO_OLD, 1030, 2000),
    (SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, 1030, 29),
    (SCHEMA_CAUSE_VERSION_UNREADABLE, 1030, 0),
]


def _schema_refusal_text(cause, expected, actual):
    return str(IncompatibleSchemaError(cause=cause, expected_generation=expected, actual_generation=actual))


def test_bare_text_quoting_a_schema_phrase_does_not_advise_switching_hermes_versions():
    for cause, expected, actual in _SCHEMA_HEAD_CASES:
        quoted = _schema_refusal_text(cause, expected, actual)
        # Constructor stored only text, no exception (what any RPC/init-error slot does when
        # nothing but a string survives).
        hermes_state._set_last_init_error(f"OperationalError: {quoted}")
        try:
            text = format_session_db_unavailable()
        finally:
            hermes_state._set_last_init_error(None)
        assert "switch" not in text.lower(), (cause, text)
        assert "hermes update" not in text.lower() and "update hermes" not in text.lower(), (cause, text)
        assert "another version" not in text.lower() and "newer version" not in text.lower(), (cause, text)


def test_real_typed_schema_refusal_still_advises_switching_or_updating_hermes():
    for cause, expected, actual in _SCHEMA_HEAD_CASES:
        exc = IncompatibleSchemaError(cause=cause, expected_generation=expected, actual_generation=actual)
        # Constructor kept the exception object alongside the text (the fix for L4-4).
        hermes_state._set_last_init_error(f"{type(exc).__name__}: {exc}", exc=exc)
        try:
            text = format_session_db_unavailable()
        finally:
            hermes_state._set_last_init_error(None)
        if cause == SCHEMA_CAUSE_VERSION_UNREADABLE:
            # This cause's remedy is recover/rebuild, not switch/update — assert its own wording.
            assert "recover" in text.lower(), (cause, text)
        else:
            assert "switch" in text.lower() or "update" in text.lower(), (cause, text)
