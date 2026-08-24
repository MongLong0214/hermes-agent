"""Regression tests for two gaps left in the R-5 /branch create_failed fix.

R-5 (fb9c3f39ba) made ``_handle_branch_command`` check ``get_session(new_id)``
before reporting ``create_failed`` when ``create_session`` raises, so a row
that was actually committed (e.g. a post-commit maintenance step raised
after the INSERT) is no longer misreported as a failure.

Two gaps remain in that fix:

Gap 1: the confirming ``get_session()`` call can itself raise (e.g. during
its own token-count flush or read). The current code treats *that*
exception exactly like "the row doesn't exist" and reports
``create_failed`` anyway -- which can be the same lie the R-5 fix exists to
prevent, just one level deeper: we genuinely don't know whether the row
exists, but we tell the caller it doesn't.

Gap 2: once past the create_session/get_session check, two further
sub-steps -- copying history (``append_messages_batch``) and setting the
title (``set_session_title``) -- each wrap their call in a bare
``except Exception: pass``. If either raises, the failure leaves zero
trace (not even a log line) and the command still returns its normal
"branched successfully" text.
"""

from __future__ import annotations

import logging
import sqlite3
import unittest.mock as mock

import pytest

from agent.i18n import t
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB, PartialBatchInsertError, SessionDB


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Real SessionStore backed by a real SessionDB (SQLite in tmp_path)."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=config)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="170829464",
        chat_id="170829464",
        chat_type="dm",
        thread_id="544520",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


def _make_branch_runner(store: SessionStore):
    """Minimal GatewayRunner stub wired to a REAL session_store/session_db."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner.config = {}
    runner._background_tasks = set()
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._busy_ack_ts = {}
    runner._pending_approvals = {}
    runner._update_prompt_pending = {}
    runner._agent_cache_lock = None
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    runner._pending_skills_reload_notes = {}
    return runner


class TestBranchGap1ConfirmGetSessionRaises:
    """get_session() raising while confirming must not be read as "absent"."""

    @pytest.mark.asyncio
    async def test_get_session_raising_during_confirm_is_not_reported_as_create_failed(
        self, store
    ):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        real_get_session = store._db.get_session
        captured: dict[str, str] = {}

        def _commit_then_raise(session_id, source, **kwargs):
            # Real insert, identical to production, then simulate the
            # post-commit maintenance step raising (same shape as the R-5
            # test).
            real_create_session(session_id, source, **kwargs)
            captured["id"] = session_id
            raise RuntimeError("simulated post-commit maintenance failure")

        def _get_session_raises_for_new_row(session_id):
            # Simulate get_session() itself raising during its own
            # flush/read -- but only for the branch row being confirmed, so
            # unrelated get_session calls elsewhere are unaffected.
            if captured.get("id") and session_id == captured["id"]:
                confirm_exception = RuntimeError(
                    "simulated get_session flush/read failure"
                )
                captured["confirm_exception"] = confirm_exception
                raise confirm_exception
            return real_get_session(session_id)

        with mock.patch.object(
            store._db, "create_session", side_effect=_commit_then_raise
        ), mock.patch.object(
            store._db, "get_session", side_effect=_get_session_raises_for_new_row
        ):
            with pytest.raises(RuntimeError) as excinfo:
                await runner._handle_branch_command(_make_event("/branch"))

        # The propagated exception must be the EXACT confirm-side failure
        # object (from get_session()), not merely an exception with the same
        # message -- those are two different bugs (re-raising the original
        # create_session exception ``e`` by accident could coincidentally
        # carry an equal-looking message) and only an identity check tells
        # them apart.
        confirm_exception = captured.get("confirm_exception")
        assert confirm_exception is not None, "get_session was never invoked for the new row"
        assert excinfo.value is confirm_exception, (
            f"expected the exact confirm-side exception instance to propagate, "
            f"got a different object: {excinfo.value!r}"
        )

        new_session_id = captured.get("id")
        assert new_session_id, "create_session was never called"

        # The row really was committed -- proving that reporting
        # create_failed (confirmed absent) would have been a lie.
        row = real_get_session(new_session_id)
        assert row is not None, "the branch row must actually be committed to state.db"


class TestBranchGap2SwallowedSubStepFailures:
    """History-copy and title-set failures must not vanish without a trace."""

    @pytest.mark.asyncio
    async def test_history_copy_failure_is_not_silently_swallowed(self, store, caplog):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        with mock.patch.object(
            store._db,
            "append_messages_batch",
            side_effect=RuntimeError("simulated history-copy failure"),
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch"))

        assert "Branched to" in result, f"expected the best-effort branch to still succeed, got: {result!r}"

        matching = [
            r for r in caplog.records
            if "simulated history-copy failure" in r.getMessage()
            or (r.exc_info and "simulated history-copy failure" in str(r.exc_info[1]))
        ]
        assert matching, (
            "history-copy failure left no trace in the logs -- it was "
            "silently swallowed even though the returned message claims success"
        )

    @pytest.mark.asyncio
    async def test_title_set_failure_is_not_silently_swallowed(self, store, caplog):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        with mock.patch.object(
            store._db,
            "set_session_title",
            side_effect=RuntimeError("simulated title-set failure"),
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch"))

        assert "Branched to" in result, f"expected the best-effort branch to still succeed, got: {result!r}"

        matching = [
            r for r in caplog.records
            if "simulated title-set failure" in r.getMessage()
            or (r.exc_info and "simulated title-set failure" in str(r.exc_info[1]))
        ]
        assert matching, (
            "title-set failure left no trace in the logs -- it was "
            "silently swallowed even though the returned message claims success"
        )


class TestBranchGap2ResponseTextHonesty:
    """Logging the failure (above) is not enough -- the user-facing text
    itself must not claim the same full success it would have claimed if
    copy/title had actually completed.
    """

    @pytest.mark.asyncio
    async def test_both_copy_and_title_failure_text_differs_from_full_success(
        self, store, caplog
    ):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        captured: dict[str, str] = {}

        def _capture_create(session_id, source, **kwargs):
            captured["id"] = session_id
            return real_create_session(session_id, source, **kwargs)

        with mock.patch.object(
            store._db, "create_session", side_effect=_capture_create
        ), mock.patch.object(
            store._db,
            "append_messages_batch",
            side_effect=RuntimeError("simulated history-copy failure"),
        ), mock.patch.object(
            store._db,
            "set_session_title",
            side_effect=RuntimeError("simulated title-set failure"),
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch mybranch"))

        new_session_id = captured["id"]
        assert new_session_id, "create_session was never called"

        # What the buggy code always returned regardless of outcome: the
        # plain full-success text claiming 1 message was copied.
        full_success_text = t(
            "gateway.branch.branched_one",
            title="mybranch",
            count=1,
            parent=parent_entry.session_id,
            new=new_session_id,
        )
        assert result != full_success_text, (
            "response text is identical to the full-success message even "
            "though both history-copy and title-set failed -- the caller "
            "cannot tell the branch is incomplete"
        )
        # A plain (non-PartialBatchInsertError) exception with zero rows
        # copied is a real, total failure -- must get the harsher
        # "did not complete" combined note, not merely "some other text".
        incomplete_note = t("gateway.branch.incomplete_copy_and_title")
        assert incomplete_note in result, (
            f"expected the combined copy+title failure note {incomplete_note!r} "
            f"in the response, got: {result!r}"
        )
        # The best-effort branch must still be reported as created (not
        # discarded/rolled back just because sub-steps failed).
        assert new_session_id in result

    @pytest.mark.asyncio
    async def test_title_set_returning_false_is_detected_and_reported(
        self, store, caplog
    ):
        """set_session_title() can return False on failure instead of
        raising -- that must be treated as a failure too, not just the
        exception path.
        """
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        store._db.append_message(parent_entry.session_id, role="user", content="hello")

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        captured: dict[str, str] = {}

        def _capture_create(session_id, source, **kwargs):
            captured["id"] = session_id
            return real_create_session(session_id, source, **kwargs)

        with mock.patch.object(
            store._db, "create_session", side_effect=_capture_create
        ), mock.patch.object(
            store._db, "set_session_title", return_value=False
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch mybranch2"))

        new_session_id = captured["id"]
        assert new_session_id, "create_session was never called"

        # Copy succeeded here, so the count is still accurate -- only the
        # title-set silently returned False. The message must still differ
        # from plain full success.
        full_success_text = t(
            "gateway.branch.branched_one",
            title="mybranch2",
            count=1,
            parent=parent_entry.session_id,
            new=new_session_id,
        )
        assert result != full_success_text, (
            "set_session_title() returned False (not an exception) but the "
            "response text still claims full success -- a False return was "
            "not detected"
        )
        # Copy fully succeeded, so only the title-only note is correct here
        # -- assert the actual key/text, not just "differs from success".
        incomplete_note = t("gateway.branch.incomplete_title")
        assert incomplete_note in result, (
            f"expected the title-only incomplete note {incomplete_note!r} in "
            f"the response, got: {result!r}"
        )

        matching = [
            r for r in caplog.records
            if "mybranch2" in r.getMessage() and new_session_id in r.getMessage()
        ]
        assert matching, (
            "set_session_title() returning False left no trace in the logs"
        )


class TestBranchGap3ChunkedPartialCopyCount:
    """A chunked copy that commits some rows before a later chunk fails
    must report the real count, and use the milder partial-copy note --
    not the same "0 messages copied" / "did not complete" text used for a
    copy that landed nothing at all.
    """

    @pytest.mark.asyncio
    async def test_partial_batch_insert_error_reports_actual_count_and_milder_note(
        self, store, caplog
    ):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        # 3 user + 2 assistant messages, in order.
        store._db.append_message(parent_entry.session_id, role="user", content="m1")
        store._db.append_message(parent_entry.session_id, role="assistant", content="a1")
        store._db.append_message(parent_entry.session_id, role="user", content="m2")
        store._db.append_message(parent_entry.session_id, role="assistant", content="a2")
        store._db.append_message(parent_entry.session_id, role="user", content="m3")

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        captured: dict[str, str] = {}

        def _capture_create(session_id, source, **kwargs):
            captured["id"] = session_id
            return real_create_session(session_id, source, **kwargs)

        # Simulate: first 3 rows (m1, a1, m2) committed in earlier chunks,
        # then the next chunk failed -- 2 of the 3 original user messages
        # actually landed.
        def _partial_copy(*args, **kwargs):
            raise PartialBatchInsertError(3, RuntimeError("simulated later-chunk failure"))

        with mock.patch.object(
            store._db, "create_session", side_effect=_capture_create
        ), mock.patch.object(
            store._db, "append_messages_batch", side_effect=_partial_copy
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch partialbranch"))

        new_session_id = captured["id"]
        assert new_session_id, "create_session was never called"

        # 2 of the 3 user messages are in the committed prefix (m1, a1, m2)
        # -- the reported count must reflect that, not 0.
        expected_partial_text = t(
            "gateway.branch.branched_many",
            title="partialbranch",
            count=2,
            parent=parent_entry.session_id,
            new=new_session_id,
        )
        assert result.startswith(expected_partial_text), (
            f"expected the headline to report 2 messages copied (the actual "
            f"committed prefix), got: {result!r}"
        )

        # Milder note: some data landed, so this must NOT be the harsh
        # "did not complete" text used when nothing committed at all.
        harsh_note = t("gateway.branch.incomplete_copy")
        assert harsh_note not in result, (
            f"partial copy (2/3 landed) was reported with the same harsh "
            f"note as a total failure: {result!r}"
        )
        # The note's saved-count must be driven by copied_rows (the real
        # committed count, 3) -- NOT by re-deriving a user-only msg_count
        # from the committed prefix (which happens to be 2 here, since only
        # 2 of the 3 committed rows are user-role). Asserting count=2 here
        # was itself the gap-A bug baked into the test: it locked in
        # "report the user-filtered count" as correct behavior, when the
        # actually-durable count is 3.
        milder_note = t("gateway.branch.incomplete_copy_partial", count=3)
        assert milder_note in result, (
            f"expected the milder partial-copy note to report the real "
            f"committed row count (3, from copied_rows) rather than the "
            f"re-derived user-only msg_count (2), got: {result!r}"
        )

    @pytest.mark.asyncio
    async def test_partial_copy_with_zero_user_rows_among_committed_is_still_partial(
        self, store, caplog
    ):
        """A committed chunk made entirely of non-user rows must still be
        classified as a partial success (not a total failure), and the
        note's saved-count must be the real committed count.

        Before the gap-A fix, ``copy_partial``/``copy_total_failure`` were
        decided from ``msg_count`` (re-filtered to user-role rows only from
        the committed prefix), not from ``copied_rows`` (the real committed
        count). If the committed prefix happens to contain zero user-role
        rows, msg_count comes out 0 even though rows really landed -- and
        the old code misreported that as a total failure.
        """
        source = _make_source()
        parent_entry = store.get_or_create_session(source)
        # The committed prefix (first 3 rows, per PartialBatchInsertError
        # (inserted=3, ...) below) must be entirely non-user-role -- zero
        # user-role rows among what actually landed. Three consecutive
        # ``assistant`` rows would NOT survive to reach the branch copy
        # logic unchanged: _handle_branch_command loads history via
        # load_transcript(..., repair_alternation=True), which runs
        # repair_message_sequence() over the loaded list and merges any
        # run of consecutive same-role ``assistant`` messages into one
        # (belt-and-suspenders repair for malformed provider histories --
        # see agent/agent_runtime_helpers.py). Three plain "a1"/"a2"/"a3"
        # assistant rows would collapse to a single merged assistant
        # message before copying, shrinking the 4-row history to 2 rows
        # and defeating the copied_rows=3 / zero-user-rows setup this
        # test needs. Using assistant(tool_call) -> tool(result) ->
        # assistant instead keeps three real rows across the merge (a
        # ``tool`` row breaks the "consecutive assistant" adjacency the
        # merge pass looks for), verified directly against
        # repair_message_sequence() to survive with 0 repairs.
        store._db.append_message(
            parent_entry.session_id,
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "x", "arguments": "{}"},
                }
            ],
        )
        store._db.append_message(
            parent_entry.session_id,
            role="tool",
            tool_call_id="call_1",
            tool_name="x",
            content="tool result 1",
        )
        store._db.append_message(parent_entry.session_id, role="assistant", content="a3")
        store._db.append_message(parent_entry.session_id, role="user", content="m1")

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        captured: dict[str, str] = {}

        def _capture_create(session_id, source, **kwargs):
            captured["id"] = session_id
            return real_create_session(session_id, source, **kwargs)

        def _partial_copy(*args, **kwargs):
            raise PartialBatchInsertError(3, RuntimeError("simulated later-chunk failure"))

        with mock.patch.object(
            store._db, "create_session", side_effect=_capture_create
        ), mock.patch.object(
            store._db, "append_messages_batch", side_effect=_partial_copy
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch zerouserbranch"))

        new_session_id = captured["id"]
        assert new_session_id, "create_session was never called"

        # 3 rows actually committed (all assistant-role) -- must be
        # reported as a partial success using the milder note, NOT the
        # harsh total-failure note.
        harsh_note = t("gateway.branch.incomplete_copy")
        assert harsh_note not in result, (
            f"3 rows actually committed (copied_rows=3) but were reported "
            f"with the harsh total-failure note, as if nothing landed: "
            f"{result!r}"
        )
        milder_note = t("gateway.branch.incomplete_copy_partial", count=3)
        assert milder_note in result, (
            f"expected the milder partial-copy note reporting the real "
            f"committed count (3), got: {result!r}"
        )
        assert new_session_id in result


class TestBranchGap5RealChunkedCopyPersistence:
    """The partial-copy count must reflect what is REALLY durable in the
    child session's DB, proven end-to-end -- not just whatever count a
    mocked ``append_messages_batch``/``PartialBatchInsertError`` happens to
    claim. A real (non-mocked) chunk boundary against a real (temp) SQLite
    DB closes a gap a fully-mocked test can't: an implementation that just
    trusts the claimed ``inserted`` count without it actually matching what
    landed would still pass a mocked-exception test, but not this one.
    """

    @pytest.mark.asyncio
    async def test_real_second_chunk_failure_persists_exactly_the_first_chunk(
        self, store, caplog
    ):
        source = _make_source()
        parent_entry = store.get_or_create_session(source)

        # Real chunk size used by the production code (see
        # gateway/slash_commands.py's append_messages_batch(..., chunk_rows=500)
        # call). Two full chunks worth + a partial third chunk's start, so
        # the failure below lands inside the SECOND real chunk.
        chunk_rows = 500
        total = chunk_rows + 251
        # Alternating user/assistant roles -- load_transcript() runs
        # repair_message_sequence() over the loaded history, which collapses
        # consecutive same-role rows as alternation violations. A history of
        # all-user rows would get merged down to a single turn before the
        # branch copy ever sees it, defeating the >500-row real-chunk setup
        # this test needs.
        history_msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(total)
        ]
        # Real (unchunked) write to seed the parent's history -- fixture
        # setup, not the mechanism under test.
        store._db.append_messages_batch(parent_entry.session_id, history_msgs)

        real_insert = SessionDB._insert_message_rows
        calls = {"n": 0}

        def flaky_insert(self_db, conn, session_id, messages):
            calls["n"] += 1
            if calls["n"] == 2:
                raise sqlite3.OperationalError("simulated real second-chunk failure")
            return real_insert(self_db, conn, session_id, messages)

        runner = _make_branch_runner(store)

        real_create_session = store._db.create_session
        captured: dict[str, str] = {}

        def _capture_create(session_id, source, **kwargs):
            captured["id"] = session_id
            return real_create_session(session_id, source, **kwargs)

        with mock.patch.object(
            store._db, "create_session", side_effect=_capture_create
        ), mock.patch.object(
            SessionDB, "_insert_message_rows", flaky_insert
        ):
            with caplog.at_level(logging.ERROR, logger="gateway.run"):
                result = await runner._handle_branch_command(_make_event("/branch realchunk"))

        new_session_id = captured["id"]
        assert new_session_id, "create_session was never called"

        # (a) the child DB's ACTUAL durable row count -- queried directly
        # from state.db, not trusted from whatever the exception claims --
        # must be exactly the first real chunk (chunk_rows rows). Nothing
        # here mocks append_messages_batch itself; the failure is injected
        # one layer below inside a real transaction, so a broken
        # implementation that just trusts a claimed count without it
        # actually being backed by real commits cannot pass this by luck.
        real_row_count = len(store._db.get_messages(new_session_id))
        assert real_row_count == chunk_rows, (
            f"expected exactly the first real chunk ({chunk_rows} rows) to "
            f"be durably committed to the child session's DB, found "
            f"{real_row_count}"
        )

        # (b) the response text must reflect that REAL count.
        milder_note = t("gateway.branch.incomplete_copy_partial", count=chunk_rows)
        assert milder_note in result, (
            f"expected the response to report the real committed count "
            f"({chunk_rows}), got: {result!r}"
        )
        harsh_note = t("gateway.branch.incomplete_copy")
        assert harsh_note not in result, (
            f"{chunk_rows} rows actually committed but were reported with "
            f"the harsh total-failure note: {result!r}"
        )
        assert new_session_id in result
