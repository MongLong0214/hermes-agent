"""Schema incompatibility must name the action that works, and no exception text.

`#784` closed the loud half: one refusal at open instead of 32 opaque write
failures, and the owner sees a schema message rather than "unexpected error".
This file covers the half that stayed open. Nine sites raise
`IncompatibleSchemaError` for five different failures and only two of them are
fixed by a different build; the other three are damage, and the old single
message told the owner to go find a release for those too.

So each case asserts the advice, not merely that a schema message appeared.
A test that only checks for the word "schema" passes on advice that cannot work.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource
from hermes_state import IncompatibleSchemaError


@pytest.mark.parametrize(
    "case",
    [
        "build_too_old",
        "fence_mismatch",
        "store_damaged",
        "schema_absent",
        "not_open",
        "unknown_cause",
        "generic",
    ],
)
def test_schema_incompatible_owner_message(monkeypatch, tmp_path, case):
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {})
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length", lambda *a, **kw: 100_000
    )
    runner = gateway_run.GatewayRunner(GatewayConfig())
    runner.adapters = {}
    runner._session_db = MagicMock()
    runner._set_session_env = lambda context: None
    runner._recover_telegram_topic_thread_id = lambda source: None
    runner._cache_session_source = lambda key, source: None
    runner._is_session_run_current = lambda key, generation: True
    runner._reply_anchor_for_event = lambda event: None
    runner._get_guild_id = lambda event: None
    runner._should_send_voice_reply = lambda *a, **kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key="schema-owner-message", session_id="schema-owner-message",
        created_at=datetime.now(), updated_at=datetime.now(),
        platform=Platform.TELEGRAM, chat_type="dm",
    )
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_platform_message_id.return_value = False
    secret = "PRIVATE_EXCEPTION_SENTINEL"
    if case == "build_too_old":
        error = IncompatibleSchemaError(
            cause=IncompatibleSchemaError.BUILD_TOO_OLD,
            expected_generation=41,
            actual_generation=47,
        )
    elif case == "fence_mismatch":
        error = IncompatibleSchemaError(
            cause=IncompatibleSchemaError.FENCE_GENERATION_MISMATCH,
            expected_generation=41,
            actual_generation=47,
        )
    elif case == "store_damaged":
        error = IncompatibleSchemaError(
            cause=IncompatibleSchemaError.STORE_DAMAGED
        )
    elif case == "schema_absent":
        error = IncompatibleSchemaError(
            cause=IncompatibleSchemaError.SCHEMA_ABSENT
        )
    elif case == "not_open":
        error = IncompatibleSchemaError(cause=IncompatibleSchemaError.NOT_OPEN)
    elif case == "unknown_cause":
        # A site added later that states a cause this mapping does not know
        # must fall back to the old wording, not crash and not claim damage.
        error = IncompatibleSchemaError(cause="a_cause_added_after_this_test")
    else:
        # An exception that merely *looks* like the typed one must stay
        # generic: `code` and the two generations are not provenance.
        error = RuntimeError(secret)
        error.cause = IncompatibleSchemaError.BUILD_TOO_OLD
        error.expected_generation = 41
        error.actual_generation = 47
        error.code = "STATE_DB_SCHEMA_INCOMPATIBLE"
    # Even typed exceptions may carry private build/path details in their text.
    error.args = (secret,)
    runner._run_agent = AsyncMock(side_effect=error)
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="owner", chat_type="dm", user_id="owner"
    )
    response = asyncio.run(runner._handle_message_with_agent(
        MessageEvent(text="hello", source=source), source, "schema-owner-message", 1
    ))
    runner._run_agent.assert_awaited_once()
    assert secret not in response

    if case == "generic":
        # An untyped failure keeps the ordinary advice, `/reset` included —
        # a fresh session is a real remedy when the store is fine.
        assert "unexpected error" in response
        assert "Hermes build" not in response
        assert "sessions repair" not in response
        assert "41" not in response and "47" not in response
        return

    # `/reset` is wrong for every schema cause: a new session opens the same
    # store. The old single message already avoided it; keep that.
    assert "/reset" not in response
    assert "None" not in response

    if case in ("build_too_old", "fence_mismatch"):
        # A build skew is the only cause a different build fixes, and the
        # owner needs the number to pick one.
        assert "expected generation 41" in response
        assert "actual generation 47" in response
        assert "Hermes build" in response
        assert "sessions repair" not in response
        if case == "fence_mismatch":
            # Either direction is possible, so the message must not assert
            # that the store is newer.
            assert "turn-fence" in response
            assert "newer" not in response
        else:
            assert "newer" in response
    elif case in ("store_damaged", "schema_absent"):
        # No build opens a damaged store. Sending the owner after a release
        # is the wrong action, and it hides the right one.
        assert "hermes sessions repair --check-only" in response
        assert "damaged" in response
        assert "no Hermes build will open it" in response
        assert not any(char.isdigit() for char in response)
    elif case == "not_open":
        assert "hermes gateway restart" in response
        assert "sessions repair" not in response
        assert not any(char.isdigit() for char in response)
    else:
        assert case == "unknown_cause"
        assert "compatible Hermes build" in response
        assert "sessions repair" not in response
        assert not any(char.isdigit() for char in response)
