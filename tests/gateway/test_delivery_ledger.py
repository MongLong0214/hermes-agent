"""Tests for the gateway delivery-obligation ledger (gateway/delivery_ledger.py).

State machine, dead-owner claiming, attempts cap, stale cutoff, retention,
id stability, and the startup redelivery sweep's contract:
- pending rows redeliver plainly (send never started, no dup risk)
- attempting/failed rows carry the recovered-reply marker (honest
  at-least-once; ambiguity is labeled, never silently resent)
- rows owned by a LIVE process are never claimed
- poison rows abandon at the attempts cap / stale cutoff
"""

import logging
import time
import sqlite3
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway import delivery_ledger as dl


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Isolated state.db per test (autouse HERMES_HOME isolation already
    redirects get_hermes_home; make the redirect explicit and per-test)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(dl, "_db_path", lambda: home / "state.db")
    yield


def _record(oid="ob-1", session_key="agent:main:slack:channel:C1", **kw):
    dl.record_obligation(
        obligation_id=oid,
        session_key=session_key,
        platform=kw.get("platform", "slack"),
        chat_id=kw.get("chat_id", "C1"),
        thread_id=kw.get("thread_id", "171.001"),
        content=kw.get("content", "the final answer"),
        adapter_profile=kw.get("adapter_profile"),
    )


def _row(oid):
    with dl._connect() as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")
        }
        select_token = "runtime_claim_token, " if "runtime_claim_token" in columns else ""
        r = conn.execute(
            f"""SELECT state, attempts, owner_pid, owner_started_at, content, last_error,
                       {select_token}NULL
               FROM delivery_obligations WHERE obligation_id=?""",
            (oid,),
        ).fetchone()
    return None if r is None else {
        "state": r[0], "attempts": r[1], "owner_pid": r[2],
        "owner_started_at": r[3], "content": r[4], "last_error": r[5],
        "runtime_claim_token": r[6],
    }


def _runtime_claim_call(method, claim, **kwargs):
    """Call a runtime mutator with claim authority when the ledger supplies it.

    The fallback keeps this real durable-ledger regression runnable against the
    predecessor, whose claimed rows did not yet carry a claim authority.
    """
    token = claim.get("runtime_claim_token")
    if isinstance(token, str) and token:
        kwargs["claim_token"] = token
    return method(claim["obligation_id"], **kwargs)


def _blocking_probe():
    """Return a blocking ledger call and an event-loop progress witness."""
    ledger_started = threading.Event()
    event_loop_progressed = threading.Event()
    blocked_event_loop = []

    def _slow_ledger_call(*args, **kwargs):
        ledger_started.set()
        # Generous timeout: a genuinely blocked loop can never set the event
        # (the witness coroutine cannot run), so a longer wait only guards
        # against loaded-CI scheduling flake, not against missing the bug.
        if not event_loop_progressed.wait(timeout=5.0):
            blocked_event_loop.append(True)

    async def _event_loop_witness():
        import asyncio

        deadline = asyncio.get_running_loop().time() + 10
        while not ledger_started.is_set():
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("ledger call never started")
            await asyncio.sleep(0)
        event_loop_progressed.set()

    return _slow_ledger_call, _event_loop_witness, blocked_event_loop


def _orphan(oid):
    """Make the row look like it belongs to a dead process."""
    with dl._connect() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET owner_pid=999999999, "
            "owner_started_at=1 WHERE obligation_id=?",
            (oid,),
        )


class TestSchemaMigration:
    def test_existing_null_or_blank_profiles_normalize_to_default_and_scope_runtime_claims(
        self, monkeypatch
    ):
        """Legacy multiplex rows are default-owned; explicit profiles stay fenced."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        now = time.time()
        conn = sqlite3.connect(dl._db_path())
        try:
            conn.execute(
                """CREATE TABLE delivery_obligations (
                    obligation_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT,
                    content TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    owner_pid INTEGER,
                    owner_started_at INTEGER,
                    last_error TEXT,
                    adapter_profile TEXT
                )"""
            )
            conn.executemany(
                """INSERT INTO delivery_obligations VALUES
                   (?, 'agent:main:telegram:channel:C1', 'telegram', 'C1', NULL,
                    'answer', 'failed', 0, ?, ?, 4242, 1700000000,
                    'send_path_degraded', ?)""",
                [
                    ("legacy-null", now, now, None),
                    ("legacy-empty", now, now, ""),
                    ("legacy-whitespace", now, now, "  "),
                    ("default", now, now, "default"),
                    ("explicit", now, now, "profile-a"),
                ],
            )
            conn.commit()
            dl._initialize_schema(conn)
            dl._initialize_schema(conn)  # migration is idempotent
            profiles = dict(
                conn.execute(
                    "SELECT obligation_id, adapter_profile FROM delivery_obligations"
                )
            )
        finally:
            conn.close()

        assert profiles == {
            "legacy-null": "default",
            "legacy-empty": "default",
            "legacy-whitespace": "default",
            "default": "default",
            "explicit": "profile-a",
        }
        assert {
            row["obligation_id"]
            for row in dl.peek_failed_for_runtime("telegram", profile="default")
        } == {"legacy-null", "legacy-empty", "legacy-whitespace", "default"}
        assert {
            row["obligation_id"]
            for row in dl.peek_failed_for_runtime("telegram", profile="profile-a")
        } == {"explicit"}
        assert dl.peek_failed_for_runtime("telegram", profile="profile-b") == []


class TestStateMachine:
    def test_record_starts_pending(self):
        _record()
        assert _row("ob-1")["state"] == "pending"


class TestObligationId:
    def test_stable_and_distinct(self):
        a = dl.compute_obligation_id("sk1", "msg1", "hello")
        assert a == dl.compute_obligation_id("sk1", "msg1", "hello")
        # Different thread (baked into session_key) → different id. This is
        # the cron-topic collision class from the earlier outbox attempt.
        assert a != dl.compute_obligation_id("sk1:threadB", "msg1", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg2", "hello")
        assert a != dl.compute_obligation_id("sk1", "msg1", "other")
        assert len(a) == 24


class TestSweep:
    def test_live_owner_rows_never_claimed(self):
        _record()  # owner = this (live) process
        assert dl.sweep_recoverable() == []

    def test_dead_owner_pending_claimed_without_marker(self):
        _record()
        _orphan("ob-1")
        claimed = dl.sweep_recoverable()
        assert len(claimed) == 1
        assert claimed[0]["needs_marker"] is False
        assert claimed[0]["attempts"] == 1
        # Claim re-stamps ownership: a second sweep in the same (live)
        # process must not double-claim.
        assert dl.sweep_recoverable() == []


class TestRuntimeFailedSweep:
    def test_claims_current_process_transient_failure_for_default_profile(self, monkeypatch):
        """Reconnect recovery must atomically claim the live default adapter's row."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")

        claimed = dl.sweep_failed_for_runtime("telegram", profile="default")

        assert [row["obligation_id"] for row in claimed] == ["ob-1"]
        assert claimed[0]["profile"] == "default"
        assert claimed[0]["runtime_recovery"] is True
        assert claimed[0]["needs_marker"] is True
        assert dl.sweep_failed_for_runtime("telegram", profile="default") == []
        assert _row("ob-1")["state"] == "attempting"
        assert _row("ob-1")["attempts"] == 1

    def test_profile_scope_never_claims_another_adapter_identity(self, monkeypatch):
        """A same-platform reconnect must never recover another bot's delivery."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(oid="default-row", platform="telegram")
        _record(
            oid="profile-a-row",
            session_key="agent:profile-a:telegram:channel:C2",
            platform="telegram",
            chat_id="C2",
            adapter_profile="profile-a",
        )
        dl.mark_failed("default-row", "send_path_degraded")
        dl.mark_failed("profile-a-row", "send_path_degraded")

        claimed = dl.sweep_failed_for_runtime("telegram", profile="profile-a")

        assert [row["obligation_id"] for row in claimed] == ["profile-a-row"]
        assert claimed[0]["profile"] == "profile-a"
        assert _row("default-row")["state"] == "failed"
        assert _row("profile-a-row")["state"] == "attempting"

    def test_release_only_restores_the_current_process_runtime_claim(self, monkeypatch):
        """A stale owner cannot release a newer claimant's attempting row."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        claim_a = dl.sweep_failed_for_runtime("telegram")[0]

        assert _runtime_claim_call(
            dl.release_runtime_claim, claim_a, error="send_path_degraded"
        ) is True
        assert _row("ob-1")["state"] == "failed"
        assert _row("ob-1")["attempts"] == 0

        claim_b = dl.sweep_failed_for_runtime("telegram")[0]
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET owner_pid=5000, "
                "owner_started_at=1800000000 WHERE obligation_id='ob-1'"
            )
        assert _runtime_claim_call(dl.release_runtime_claim, claim_b) is False
        assert _row("ob-1")["state"] == "attempting"
        assert _row("ob-1")["attempts"] == 1

    def test_interrupted_send_settlement_is_fenced_to_current_runtime_owner(
        self, monkeypatch
    ):
        """A stale cancellation handler cannot settle a replacement owner's claim."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        claim = dl.claim_failed_for_runtime("ob-1", "telegram")
        assert claim is not None

        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_obligations SET owner_pid=5151,
                   owner_started_at=1800000000 WHERE obligation_id='ob-1'"""
            )
        assert _runtime_claim_call(
            dl.settle_runtime_claim_after_send_started, claim
        ) is False
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"]) == ("attempting", 1)

        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_obligations SET owner_pid=4242,
                   owner_started_at=1700000000 WHERE obligation_id='ob-1'"""
            )
        assert _runtime_claim_call(
            dl.settle_runtime_claim_after_send_started, claim
        ) is True
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"], row["last_error"]) == (
            "failed",
            1,
            "send_path_degraded",
        )

    def test_runtime_peek_and_claim_recheck_profile_owner_and_error(self, monkeypatch):
        """The advisory peek and authority claim share the runtime boundary."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(oid="eligible", platform="telegram")
        _record(
            oid="wrong-profile",
            platform="telegram",
            adapter_profile="profile-a",
        )
        _record(oid="wrong-owner", platform="telegram")
        _record(oid="permanent-error", platform="telegram")
        _record(
            oid="changed-after-peek",
            session_key="agent:main:telegram:channel:C2",
            platform="telegram",
            chat_id="C2",
        )
        for oid, error in (
            ("eligible", "send_path_degraded"),
            ("wrong-profile", "send_path_degraded"),
            ("wrong-owner", "send_path_degraded"),
            ("permanent-error", "permanent rejection"),
            ("changed-after-peek", "send_path_degraded"),
        ):
            dl.mark_failed(oid, error)
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_obligations SET owner_pid=5151,
                   owner_started_at=1800000000 WHERE obligation_id='wrong-owner'"""
            )
        candidates = dl.peek_failed_for_runtime("telegram", profile="default")
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_obligations SET last_error='permanent rejection'
                   WHERE obligation_id='changed-after-peek'"""
            )

        assert {candidate["obligation_id"] for candidate in candidates} == {
            "eligible", "changed-after-peek"
        }
        row = _row("eligible")
        assert row is not None
        assert row["state"] == "failed"
        assert row["attempts"] == 0
        claim = dl.claim_failed_for_runtime("eligible", "telegram", profile="default")
        assert claim is not None
        assert claim["obligation_id"] == "eligible"
        assert (
            dl.claim_failed_for_runtime(
                "changed-after-peek", "telegram", profile="default"
            )
            is None
        )
        for oid in (
            "wrong-profile",
            "wrong-owner",
            "permanent-error",
            "changed-after-peek",
        ):
            row = _row(oid)
            assert row is not None
            assert row["state"] == "failed"
            assert row["attempts"] == 0


class TestPrune:
    def test_old_delivered_rows_pruned(self):
        _record()
        dl.mark_delivered("ob-1")
        with dl._connect() as conn:
            conn.execute(
                "UPDATE delivery_obligations SET updated_at=? WHERE obligation_id=?",
                (time.time() - dl._RETENTION_SECONDS - 60, "ob-1"),
            )
        dl._prune()
        assert _row("ob-1") is None


class TestLedgerEnabled:
    def test_default_on(self):
        assert dl.ledger_enabled({}) is True
        assert dl.ledger_enabled({"gateway": {}}) is True


class TestGatewayRedeliverySweep:
    """Drive the real GatewayRunner._redeliver_pending_obligations."""

    @staticmethod
    def _runner(adapter=None):
        from gateway.config import Platform
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {Platform.SLACK: adapter} if adapter else {}
        runner._profile_adapters = {}
        runner._active_profile_name = lambda: "default"
        runner._running = True
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @staticmethod
    def _adapter(success=True):
        adapter = MagicMock()
        adapter.send = AsyncMock(
            return_value=MagicMock(success=success, error="" if success else "nope")
        )
        return adapter

    @pytest.mark.asyncio
    async def test_pending_redelivers_plain_and_clears_resume(self):
        _record()  # pending
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        n = await runner._redeliver_pending_obligations()

        assert n == 1
        sent = adapter.send.call_args.kwargs
        assert sent["content"] == "the final answer"  # no marker
        assert sent["metadata"] == {"thread_id": "171.001"}
        assert _row("ob-1")["state"] == "delivered"
        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )

    @pytest.mark.asyncio
    async def test_attempting_redelivers_with_marker(self):
        _record()
        dl.mark_attempting("ob-1")
        _orphan("ob-1")
        adapter = self._adapter()
        runner = self._runner(adapter)

        await runner._redeliver_pending_obligations()

        sent = adapter.send.call_args.kwargs
        assert sent["content"].startswith(dl.RECOVERED_MARKER)
        assert sent["content"].endswith("the final answer")

    @pytest.mark.asyncio
    async def test_runtime_redelivery_clears_resume_and_uses_reconnect_marker(self):
        from gateway.config import Platform

        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)

        redelivered = await runner._redeliver_failed_obligations_for_platform(
            Platform.SLACK
        )

        assert redelivered == 1
        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )
        assert adapter.send.call_args.kwargs["content"].startswith(
            dl.RECONNECTED_MARKER
        )
        assert _row("ob-1")["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_runtime_redelivery_success_log_excludes_delivery_identifiers(
        self, caplog
    ):
        from gateway.config import Platform

        obligation_id = "obligation-raw-redelivery-id"
        chat_id = "C-RAW-REDELIVERY-CHAT"
        profile = "profile-private-redelivery"
        session_key = "agent:session-private-redelivery:slack:channel:C-RAW-REDELIVERY-CHAT"
        content = "private redelivery content must not reach the success log"
        _record(
            oid=obligation_id,
            session_key=session_key,
            platform="slack",
            chat_id=chat_id,
            content=content,
            adapter_profile=profile,
        )
        dl.mark_failed(obligation_id, "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        runner._profile_adapters = {profile: {Platform.SLACK: adapter}}

        with caplog.at_level(logging.INFO):
            assert (
                await runner._redeliver_failed_obligations_for_platform(
                    Platform.SLACK, profile=profile
                )
                == 1
            )

        for private_value in (
            chat_id,
            obligation_id,
            profile,
            session_key,
            content,
            f"slack:{chat_id}",
        ):
            assert private_value not in caplog.text
        assert "Recovered delivery succeeded (attempt 1)" in caplog.text

    @pytest.mark.parametrize("release_raises", [False, True])
    @pytest.mark.asyncio
    async def test_runtime_clear_failure_refunds_claim_and_next_reconnect_settles_once(
        self, monkeypatch, release_raises
    ):
        """A pre-send clear failure leaves this signal with nothing to send."""
        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="slack", adapter_profile="default")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        release_calls = []

        def unavailable_release(*_args, **_kwargs):
            release_calls.append(True)
            if release_raises:
                raise RuntimeError("release unavailable")
            return False

        monkeypatch.setattr(dl, "release_runtime_claim", unavailable_release)
        failed_clear = AsyncMock(side_effect=RuntimeError("temporary resume-store failure"))
        monkeypatch.setattr(
            runner._async_session_store,
            "clear_resume_pending",
            failed_clear,
        )

        redelivered = await runner._redeliver_failed_obligations_for_platform(
            Platform.SLACK, profile="default"
        )

        row = _row("ob-1")
        assert row is not None
        assert redelivered == 0
        adapter.send.assert_not_awaited()
        assert row["state"] == "failed"
        assert row["attempts"] == 0
        assert release_calls == []
        failed_clear.assert_awaited_once_with("agent:main:slack:channel:C1")

        # A separate later reconnect can claim the refunded row and settle it once.
        monkeypatch.setattr(
            runner._async_session_store,
            "clear_resume_pending",
            AsyncMock(),
        )
        assert await runner._redeliver_failed_obligations_for_platform(
            Platform.SLACK, profile="default"
        ) == 1
        adapter.send.assert_awaited_once()
        final_row = _row("ob-1")
        assert final_row is not None
        assert final_row["state"] == "delivered"

    @pytest.mark.parametrize(
        "initial_outcome",
        ["raises_before_update", "updates_then_raises", "returns_false"],
    )
    @pytest.mark.asyncio
    async def test_runtime_claim_return_cancellation_retains_exact_refund_for_next_signal(
        self, monkeypatch, initial_outcome
    ):
        """A cancelled claim keeps its exact unsent refund authority for retry."""
        import asyncio

        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        loop = asyncio.get_running_loop()
        claim_committed = asyncio.Event()
        release_worker = threading.Event()
        worker_returned = threading.Event()
        release_tokens = []
        actual_claim = dl.claim_failed_for_runtime
        actual_release = dl.release_runtime_claim

        def claim_then_wait_for_return(*args, **kwargs):
            row = actual_claim(*args, **kwargs)
            loop.call_soon_threadsafe(claim_committed.set)
            try:
                release_worker.wait()
                return row
            finally:
                worker_returned.set()

        def release_with_initial_uncertainty(obligation_id, *, claim_token):
            release_tokens.append(claim_token)
            if len(release_tokens) != 1:
                return actual_release(obligation_id, claim_token=claim_token)
            if initial_outcome == "raises_before_update":
                raise RuntimeError("release unavailable before update")
            if initial_outcome == "updates_then_raises":
                assert actual_release(obligation_id, claim_token=claim_token) is True
                raise RuntimeError("release unavailable after update")
            assert initial_outcome == "returns_false"
            return False

        monkeypatch.setattr(dl, "claim_failed_for_runtime", claim_then_wait_for_return)
        monkeypatch.setattr(dl, "release_runtime_claim", release_with_initial_uncertainty)
        cancelled_signal = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await claim_committed.wait()
        cancelled_signal.cancel()
        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_signal
        await asyncio.to_thread(worker_returned.wait)

        row = _row("ob-1")
        assert row is not None
        if initial_outcome == "updates_then_raises":
            assert (row["state"], row["attempts"]) == ("failed", 0)
        else:
            assert (row["state"], row["attempts"]) == ("attempting", 1)
        adapter.send.assert_not_awaited()

        # The next signal retries the same token once.  A post-commit initial
        # exception returns False here, proving that stale authority cannot
        # refund the newly claimed generation that follows it.
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        assert len(release_tokens) == 2
        assert release_tokens[0] == release_tokens[1]
        adapter.send.assert_awaited_once()
        final_row = _row("ob-1")
        assert final_row is not None
        assert (final_row["state"], final_row["attempts"]) == ("delivered", 1)

    @pytest.mark.asyncio
    async def test_runtime_cancelled_claim_discards_stale_refund_without_touching_replacement(
        self, monkeypatch
    ):
        """A stale refund token cannot change or send a replacement attempting row."""
        import asyncio

        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        loop = asyncio.get_running_loop()
        claim_committed = asyncio.Event()
        release_worker = threading.Event()
        worker_returned = threading.Event()
        release_tokens = []
        actual_claim = dl.claim_failed_for_runtime
        actual_release = dl.release_runtime_claim

        def claim_then_wait_for_return(*args, **kwargs):
            row = actual_claim(*args, **kwargs)
            loop.call_soon_threadsafe(claim_committed.set)
            try:
                release_worker.wait()
                return row
            finally:
                worker_returned.set()

        def release_then_fail_once(obligation_id, *, claim_token):
            release_tokens.append(claim_token)
            if len(release_tokens) == 1:
                raise RuntimeError("release unavailable")
            return actual_release(obligation_id, claim_token=claim_token)

        monkeypatch.setattr(dl, "claim_failed_for_runtime", claim_then_wait_for_return)
        monkeypatch.setattr(dl, "release_runtime_claim", release_then_fail_once)
        cancelled_signal = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await claim_committed.wait()
        cancelled_signal.cancel()
        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_signal
        await asyncio.to_thread(worker_returned.wait)

        stranded = _row("ob-1")
        assert stranded is not None
        stale_token = stranded["runtime_claim_token"]
        assert isinstance(stale_token, str)
        assert actual_release("ob-1", claim_token=stale_token) is True
        replacement_claim = actual_claim("ob-1", "slack")
        assert replacement_claim is not None
        replacement_before = _row("ob-1")
        assert replacement_before is not None
        assert (replacement_before["state"], replacement_before["attempts"]) == (
            "attempting",
            1,
        )
        assert replacement_before["runtime_claim_token"] == replacement_claim[
            "runtime_claim_token"
        ]

        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        replacement_after = _row("ob-1")
        assert replacement_after == replacement_before
        assert len(release_tokens) == 2
        assert release_tokens[0] == release_tokens[1]
        adapter.send.assert_not_awaited()

        # False discarded only the stale authority; it cannot keep issuing
        # release attempts against the replacement on later signals.
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        assert len(release_tokens) == 2
        adapter.send.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_runtime_cancelled_claim_retries_uncertain_refund_once_per_signal(
        self, monkeypatch
    ):
        """A retry exception retains authority without sending or retry storms."""
        import asyncio

        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        loop = asyncio.get_running_loop()
        claim_committed = asyncio.Event()
        release_worker = threading.Event()
        worker_returned = threading.Event()
        release_tokens = []
        actual_claim = dl.claim_failed_for_runtime
        actual_release = dl.release_runtime_claim

        def claim_then_wait_for_return(*args, **kwargs):
            row = actual_claim(*args, **kwargs)
            loop.call_soon_threadsafe(claim_committed.set)
            try:
                release_worker.wait()
                return row
            finally:
                worker_returned.set()

        def release_then_recover(obligation_id, *, claim_token):
            release_tokens.append(claim_token)
            if len(release_tokens) < 3:
                raise RuntimeError("release unavailable")
            return actual_release(obligation_id, claim_token=claim_token)

        monkeypatch.setattr(dl, "claim_failed_for_runtime", claim_then_wait_for_return)
        monkeypatch.setattr(dl, "release_runtime_claim", release_then_recover)
        cancelled_signal = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await claim_committed.wait()
        cancelled_signal.cancel()
        release_worker.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_signal
        await asyncio.to_thread(worker_returned.wait)

        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"]) == ("attempting", 1)
        assert len(release_tokens) == 2
        adapter.send.assert_not_awaited()

        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        assert len(release_tokens) == 3
        assert release_tokens[0] == release_tokens[1] == release_tokens[2]
        adapter.send.assert_awaited_once()
        final_row = _row("ob-1")
        assert final_row is not None
        assert (final_row["state"], final_row["attempts"]) == ("delivered", 1)

    @pytest.mark.asyncio
    async def test_runtime_cancellation_during_clear_leaves_row_for_later_signal(
        self, monkeypatch
    ):
        """Cancellation before a send claim must not strand the durable row."""
        import asyncio

        from gateway.config import Platform

        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        clear_entered = asyncio.Event()
        unblock_clear = asyncio.Event()

        async def block_clear(_session_key):
            clear_entered.set()
            await unblock_clear.wait()

        monkeypatch.setattr(
            runner._async_session_store, "clear_resume_pending", block_clear
        )
        task = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await asyncio.wait_for(clear_entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        row = _row("ob-1")
        assert row is not None
        assert row["state"] == "failed"
        assert row["attempts"] == 0
        adapter.send.assert_not_awaited()

        monkeypatch.setattr(
            runner._async_session_store, "clear_resume_pending", AsyncMock()
        )
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        adapter.send.assert_awaited_once()
        final_row = _row("ob-1")
        assert final_row is not None
        assert final_row["state"] == "delivered"

    @pytest.mark.asyncio
    async def test_runtime_cancellation_during_second_clear_leaves_all_rows_unclaimed(
        self, monkeypatch
    ):
        """No candidate may be claimed while any resume clearance can await."""
        import asyncio

        from gateway.config import Platform

        _record(oid="ob-1", platform="slack", chat_id="C1")
        _record(
            oid="ob-2",
            session_key="agent:main:slack:channel:C2",
            platform="slack",
            chat_id="C2",
        )
        dl.mark_failed("ob-1", "send_path_degraded")
        dl.mark_failed("ob-2", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        second_clear_entered = asyncio.Event()
        unblock_clear = asyncio.Event()
        clear_calls = 0

        async def block_second_clear(_session_key):
            nonlocal clear_calls
            clear_calls += 1
            if clear_calls == 2:
                second_clear_entered.set()
                await unblock_clear.wait()

        monkeypatch.setattr(
            runner._async_session_store, "clear_resume_pending", block_second_clear
        )
        task = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await asyncio.wait_for(second_clear_entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for oid in ("ob-1", "ob-2"):
            row = _row(oid)
            assert row is not None
            assert row["state"] == "failed"
            assert row["attempts"] == 0
        adapter.send.assert_not_awaited()

        monkeypatch.setattr(
            runner._async_session_store, "clear_resume_pending", AsyncMock()
        )
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 2
        assert adapter.send.await_count == 2
        assert sorted(
            call.kwargs["chat_id"] for call in adapter.send.await_args_list
        ) == ["C1", "C2"]

    @pytest.mark.asyncio
    async def test_runtime_cancellation_during_first_send_settles_only_current_claim(
        self, monkeypatch
    ):
        """An entered send is ambiguous; it keeps its attempt and aborts this signal."""
        import asyncio

        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(oid="ob-1", platform="slack", chat_id="C1")
        _record(
            oid="ob-2",
            session_key="agent:main:slack:channel:C2",
            platform="slack",
            chat_id="C2",
        )
        dl.mark_failed("ob-1", "send_path_degraded")
        dl.mark_failed("ob-2", "send_path_degraded")
        send_entered = asyncio.Event()
        sent_on_cancelled_signal = []

        async def block_first_send(**kwargs):
            sent_on_cancelled_signal.append(kwargs["chat_id"])
            send_entered.set()
            await asyncio.Event().wait()

        adapter = MagicMock()
        adapter.send = block_first_send
        runner = self._runner(adapter)
        task = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await asyncio.wait_for(send_entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert sent_on_cancelled_signal == ["C1"]
        first = _row("ob-1")
        second = _row("ob-2")
        assert first is not None
        assert second is not None
        assert (first["state"], first["attempts"], first["last_error"]) == (
            "failed",
            1,
            "send_path_degraded",
        )
        assert (second["state"], second["attempts"]) == ("failed", 0)

        adapter.send = AsyncMock(return_value=MagicMock(success=True, error=""))
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 2
        assert sorted(
            call.kwargs["chat_id"] for call in adapter.send.await_args_list
        ) == ["C1", "C2"]
        first = _row("ob-1")
        second = _row("ob-2")
        assert first is not None
        assert second is not None
        assert (first["state"], first["attempts"]) == ("delivered", 2)
        assert (second["state"], second["attempts"]) == ("delivered", 1)

    @pytest.mark.asyncio
    async def test_runtime_result_control_exception_settles_current_claim_for_later_signal(
        self, monkeypatch
    ):
        """A control exception while inspecting a sent result remains recoverable."""
        from gateway.config import Platform

        class ResultInspectionAbort(BaseException):
            pass

        abort = ResultInspectionAbort()

        class ExplodingResult:
            @property
            def success(self):
                raise abort

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = MagicMock()
        adapter.send = AsyncMock(return_value=ExplodingResult())
        runner = self._runner(adapter)

        with pytest.raises(ResultInspectionAbort) as caught:
            await runner._redeliver_failed_obligations_for_platform(Platform.SLACK)

        assert caught.value is abort
        adapter.send.assert_awaited_once()
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"], row["last_error"]) == (
            "failed",
            1,
            "send_path_degraded",
        )

        # The interrupted signal makes no second attempt; a distinct reconnect
        # gets exactly one retry, spending the second attempt.
        adapter.send = AsyncMock(return_value=MagicMock(success=True, error=""))
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        adapter.send.assert_awaited_once()
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"]) == ("delivered", 2)

    @pytest.mark.asyncio
    async def test_runtime_post_settlement_control_exception_keeps_delivered_state(
        self, monkeypatch
    ):
        """A control exception after durable settlement cannot revert delivery."""
        from gateway.config import Platform

        class PostSettlementAbort(BaseException):
            pass

        abort = PostSettlementAbort()
        settlements = []
        actual_settlement = dl.settle_runtime_claim
        actual_fallback = dl.settle_runtime_claim_after_send_started

        def settle_then_abort(obligation_id, *, claim_token, delivered, error=""):
            assert (
                actual_settlement(
                    obligation_id,
                    claim_token=claim_token,
                    delivered=delivered,
                    error=error,
                )
                is True
            )
            raise abort

        def record_fallback(obligation_id, *, claim_token):
            settlements.append(obligation_id)
            return actual_fallback(obligation_id, claim_token=claim_token)

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        monkeypatch.setattr(dl, "settle_runtime_claim", settle_then_abort)
        monkeypatch.setattr(
            dl, "settle_runtime_claim_after_send_started", record_fallback
        )
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        runner = self._runner(self._adapter())

        with pytest.raises(PostSettlementAbort) as caught:
            await runner._redeliver_failed_obligations_for_platform(Platform.SLACK)

        assert caught.value is abort
        assert settlements == ["ob-1"]
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"]) == ("delivered", 1)

    @pytest.mark.parametrize("send_success", [True, False])
    @pytest.mark.asyncio
    async def test_runtime_normal_settlement_cannot_mutate_a_newer_attempting_owner(
        self, monkeypatch, send_success
    ):
        """A stale post-send task may not settle a replacement runtime claim."""
        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")

        async def send_after_replacement(**_kwargs):
            # This stands in for a newer owner taking over after our send has
            # entered but before the stale task classifies and settles it.
            with dl._DB_LOCK, dl._transaction() as conn:
                conn.execute(
                    """UPDATE delivery_obligations
                       SET state='attempting', owner_pid=5151,
                           owner_started_at=1800000000,
                           last_error='send_path_degraded'
                       WHERE obligation_id='ob-1'"""
                )
            return MagicMock(success=send_success, error="definitive rejection")

        adapter = MagicMock()
        adapter.send = send_after_replacement
        runner = self._runner(adapter)

        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"]) == ("attempting", 1)
        assert (row["owner_pid"], row["owner_started_at"]) == (5151, 1800000000)

    @pytest.mark.parametrize("stale_delivered", [True, False])
    def test_same_process_stale_settlement_cannot_mutate_new_claim(
        self, monkeypatch, stale_delivered
    ):
        """A delayed A worker must not settle same-PID/start attempt B."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        claim_a = dl.claim_failed_for_runtime("ob-1", "telegram")
        assert claim_a is not None

        worker_entered = threading.Event()
        release_worker = threading.Event()
        worker_results = []
        actual_settle = dl._settle_runtime_claim

        def pause_before_atomic_write(obligation_id, *, state, error, **kwargs):
            is_stale_normal_result = state == "delivered" or error == "ordinary failure"
            if is_stale_normal_result:
                worker_entered.set()
                assert release_worker.wait(timeout=2)
            return actual_settle(obligation_id, state=state, error=error, **kwargs)

        monkeypatch.setattr(dl, "_settle_runtime_claim", pause_before_atomic_write)

        def delayed_a_normal_settlement():
            worker_results.append(
                _runtime_claim_call(
                    dl.settle_runtime_claim,
                    claim_a,
                    delivered=stale_delivered,
                    error="ordinary failure",
                )
            )

        worker = threading.Thread(target=delayed_a_normal_settlement)
        worker.start()
        assert worker_entered.wait(timeout=2)

        # Cancellation/fallback wins while A's normal worker is still alive.
        assert _runtime_claim_call(
            dl.settle_runtime_claim_after_send_started, claim_a
        ) is True
        claim_b = dl.claim_failed_for_runtime("ob-1", "telegram")
        assert claim_b is not None

        release_worker.set()
        worker.join(timeout=2)
        assert not worker.is_alive()

        # On f8ca this is True: stale A settles B under the same PID/start.
        assert worker_results == [False]
        attempting_b = _row("ob-1")
        assert attempting_b is not None
        assert (attempting_b["state"], attempting_b["attempts"]) == ("attempting", 2)
        assert claim_a["runtime_claim_token"] != claim_b["runtime_claim_token"]
        assert attempting_b["runtime_claim_token"] == claim_b["runtime_claim_token"]

        assert _runtime_claim_call(
            dl.settle_runtime_claim, claim_b, delivered=True
        ) is True
        settled_b = _row("ob-1")
        assert settled_b is not None
        assert (settled_b["state"], settled_b["attempts"]) == ("delivered", 2)
        assert settled_b["runtime_claim_token"] is None

    def test_same_process_stale_release_and_fallback_cannot_refund_new_claim(
        self, monkeypatch
    ):
        """Neither stale A fallback nor its pre-send refund can touch B."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="telegram")
        dl.mark_failed("ob-1", "send_path_degraded")
        claim_a = dl.claim_failed_for_runtime("ob-1", "telegram")
        assert claim_a is not None
        assert _runtime_claim_call(
            dl.settle_runtime_claim_after_send_started, claim_a
        ) is True
        claim_b = dl.claim_failed_for_runtime("ob-1", "telegram")
        assert claim_b is not None
        before = _row("ob-1")

        # On f8ca this returns True and refunds B's attempt.
        assert _runtime_claim_call(dl.release_runtime_claim, claim_a) is False
        assert _runtime_claim_call(
            dl.settle_runtime_claim_after_send_started, claim_a
        ) is False
        after = _row("ob-1")
        assert after == before
        assert after is not None
        assert (after["state"], after["attempts"]) == ("attempting", 2)
        assert after["runtime_claim_token"] == claim_b["runtime_claim_token"]

        assert _runtime_claim_call(
            dl.settle_runtime_claim_after_send_started, claim_b
        ) is True
        settled_b = _row("ob-1")
        assert settled_b is not None
        assert (settled_b["state"], settled_b["attempts"]) == ("failed", 2)
        assert settled_b["runtime_claim_token"] is None

    def test_runtime_claim_token_migrates_and_dead_owner_recovery_retires_it(
        self, monkeypatch
    ):
        """Legacy NULL is recoverable; a dead runtime claim cannot retain authority."""
        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        now = time.time()
        conn = sqlite3.connect(dl._db_path())
        try:
            conn.execute(
                """CREATE TABLE delivery_obligations (
                    obligation_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT,
                    content TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    owner_pid INTEGER,
                    owner_started_at INTEGER,
                    last_error TEXT,
                    adapter_profile TEXT NOT NULL DEFAULT 'default'
                )"""
            )
            conn.execute(
                """INSERT INTO delivery_obligations VALUES
                   ('legacy', 'agent:main:telegram:channel:C1', 'telegram', 'C1',
                    NULL, 'answer', 'failed', 0, ?, ?, 4242, 1700000000,
                    'send_path_degraded', 'default')""",
                (now, now),
            )
            conn.commit()
            dl._initialize_schema(conn)
            dl._initialize_schema(conn)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(delivery_obligations)")}
            assert "runtime_claim_token" in columns
            assert conn.execute(
                "SELECT runtime_claim_token FROM delivery_obligations WHERE obligation_id='legacy'"
            ).fetchone() == (None,)
        finally:
            conn.close()

        claim_a = dl.claim_failed_for_runtime("legacy", "telegram")
        assert claim_a is not None
        assert isinstance(claim_a["runtime_claim_token"], str)
        assert claim_a["runtime_claim_token"]
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_obligations SET owner_pid=999999999,
                   owner_started_at=1 WHERE obligation_id='legacy'"""
            )
        recovered = dl.sweep_recoverable(deliverable_platforms={"telegram"})
        assert [row["obligation_id"] for row in recovered] == ["legacy"]
        recovered_row = _row("legacy")
        assert recovered_row is not None
        assert recovered_row["runtime_claim_token"] is None

        dl.mark_failed("legacy", "send_path_degraded")
        claim_b = dl.claim_failed_for_runtime("legacy", "telegram")
        assert claim_b is not None
        assert claim_b["runtime_claim_token"] != claim_a["runtime_claim_token"]

    @pytest.mark.asyncio
    async def test_cancelled_runner_worker_cannot_settle_same_owner_successor(
        self, monkeypatch
    ):
        """Cancellation leaves A's to_thread worker unable to affect signal B."""
        import asyncio

        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]
        finished = [threading.Event(), threading.Event()]
        normal_settlements = 0
        actual_settle = dl._settle_runtime_claim

        def pause_each_normal_worker(obligation_id, *, state, error, **kwargs):
            nonlocal normal_settlements
            if state == "delivered":
                slot = normal_settlements
                normal_settlements += 1
                entered[slot].set()
                assert release[slot].wait(timeout=2)
                try:
                    return actual_settle(obligation_id, state=state, error=error, **kwargs)
                finally:
                    finished[slot].set()
            return actual_settle(obligation_id, state=state, error=error, **kwargs)

        monkeypatch.setattr(dl, "_settle_runtime_claim", pause_each_normal_worker)
        first_signal = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await asyncio.wait_for(asyncio.to_thread(entered[0].wait), timeout=2)
        first_signal.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_signal

        after_a_fallback = _row("ob-1")
        assert after_a_fallback is not None
        assert (after_a_fallback["state"], after_a_fallback["attempts"]) == ("failed", 1)

        second_signal = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        try:
            await asyncio.wait_for(asyncio.to_thread(entered[1].wait), timeout=2)
            attempting_b = _row("ob-1")
            assert attempting_b is not None
            assert (attempting_b["state"], attempting_b["attempts"]) == ("attempting", 2)

            release[0].set()
            await asyncio.wait_for(asyncio.to_thread(finished[0].wait), timeout=2)
            # On f8ca this row is already delivered by stale A's delayed worker.
            assert _row("ob-1") == attempting_b

            release[1].set()
            assert await asyncio.wait_for(second_signal, timeout=2) == 1
        finally:
            release[0].set()
            release[1].set()
            if not second_signal.done():
                await asyncio.wait_for(second_signal, timeout=2)

        final = _row("ob-1")
        assert final is not None
        assert (final["state"], final["attempts"]) == ("delivered", 2)
        assert final["runtime_claim_token"] is None
        assert adapter.send.await_count == 2

    @pytest.mark.asyncio
    async def test_runtime_prewrite_normal_settlement_failure_recovers_without_leaking(
        self, monkeypatch, caplog
    ):
        """A normal settlement error must release this spent claim for a later signal."""
        from gateway.config import Platform

        class PrivateSettlementError(Exception):
            pass

        actual_settlement = getattr(dl, "settle_runtime_claim", None)
        actual_fallback = dl.settle_runtime_claim_after_send_started
        fallback_calls = []

        def fail_before_mutation(*_args, **_kwargs):
            raise PrivateSettlementError(
                "private-token /tmp/secret-profile session=ob-1 actor=alice"
            )

        def record_fallback(obligation_id, *, claim_token):
            fallback_calls.append(obligation_id)
            return actual_fallback(obligation_id, claim_token=claim_token)

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        monkeypatch.setattr(
            dl, "settle_runtime_claim", fail_before_mutation, raising=False
        )
        monkeypatch.setattr(dl, "settle_runtime_claim_after_send_started", record_fallback)
        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)

        with caplog.at_level(logging.DEBUG):
            assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0

        adapter.send.assert_awaited_once()
        assert fallback_calls == ["ob-1"]
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"], row["last_error"]) == (
            "failed",
            1,
            "send_path_degraded",
        )
        for private_value in (
            "private-token",
            "/tmp/secret-profile",
            "session=ob-1",
            "actor=alice",
            "Traceback",
        ):
            assert private_value not in caplog.text

        assert actual_settlement is not None
        monkeypatch.setattr(dl, "settle_runtime_claim", actual_settlement)
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        assert adapter.send.await_count == 2
        row = _row("ob-1")
        assert row is not None
        assert (row["state"], row["attempts"]) == ("delivered", 2)

    @pytest.mark.asyncio
    async def test_runtime_raced_whitespace_profile_handoff_uses_default_adapter(
        self, monkeypatch
    ):
        """A row inserted after migration still hands legacy blank to default."""
        from gateway.config import Platform

        monkeypatch.setattr(dl, "_owner_stamp", lambda: (4242, 1700000000))
        _record(oid="raced", platform="slack", chat_id="C1")
        _record(
            oid="explicit",
            session_key="agent:profile-a:slack:channel:C2",
            platform="slack",
            chat_id="C2",
            adapter_profile="profile-a",
        )
        dl.mark_failed("raced", "send_path_degraded")
        dl.mark_failed("explicit", "send_path_degraded")

        # Simulate a legacy writer racing immediately after every connection's
        # transactional migration, before its runtime query receives the row.
        actual_initialize_schema = dl._initialize_schema

        def inject_raced_legacy_profile(conn):
            actual_initialize_schema(conn)
            conn.execute(
                "UPDATE delivery_obligations SET adapter_profile='  ' "
                "WHERE obligation_id='raced'"
            )
            conn.commit()

        monkeypatch.setattr(dl, "_initialize_schema", inject_raced_legacy_profile)
        default_adapter = self._adapter()
        wrong_adapter = self._adapter()
        runner = self._runner(default_adapter)
        runner._profile_adapters = {
            "  ": {},
            "profile-a": {Platform.SLACK: wrong_adapter},
        }

        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        default_adapter.send.assert_awaited_once()
        wrong_adapter.send.assert_not_awaited()
        row = _row("raced")
        assert row is not None
        assert (row["state"], row["attempts"]) == ("delivered", 1)
        explicit_row = _row("explicit")
        assert explicit_row is not None
        assert explicit_row["state"] == "failed"

        # Every runtime handoff reports the legacy blank as default, while a
        # nonblank profile stays exact and cannot be claimed by default.
        with dl._connect() as conn:
            conn.execute(
                """UPDATE delivery_obligations
                   SET state='failed', attempts=0, last_error='send_path_degraded'
                   WHERE obligation_id='raced'"""
            )
        assert dl.peek_failed_for_runtime("slack", profile="default") == [
            {
                "obligation_id": "raced",
                "session_key": "agent:main:slack:channel:C1",
                "profile": "default",
            }
        ]
        claim = dl.claim_failed_for_runtime("raced", "slack", profile="default")
        assert claim is not None
        assert claim["profile"] == "default"
        assert _runtime_claim_call(dl.release_runtime_claim, claim) is True
        swept = dl.sweep_failed_for_runtime("slack", profile="default")
        assert len(swept) == 1
        assert swept[0]["profile"] == "default"
        assert _runtime_claim_call(dl.release_runtime_claim, swept[0]) is True
        assert dl.claim_failed_for_runtime("explicit", "slack", profile="default") is None
        assert dl.peek_failed_for_runtime("slack", profile="profile-a") == [
            {
                "obligation_id": "explicit",
                "session_key": "agent:profile-a:slack:channel:C2",
                "profile": "profile-a",
            }
        ]

    @pytest.mark.asyncio
    async def test_concurrent_runtime_signals_claim_each_row_once(self, monkeypatch):
        """Two reconnect signals may both clear, but only one claims each row."""
        import asyncio

        from gateway.config import Platform

        _record(oid="ob-1", platform="slack", chat_id="C1")
        _record(
            oid="ob-2",
            session_key="agent:main:slack:channel:C2",
            platform="slack",
            chat_id="C2",
        )
        dl.mark_failed("ob-1", "send_path_degraded")
        dl.mark_failed("ob-2", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)
        clear_barrier = asyncio.Barrier(2)

        async def synchronize_clear(_session_key):
            await clear_barrier.wait()

        monkeypatch.setattr(
            runner._async_session_store, "clear_resume_pending", synchronize_clear
        )
        results = await asyncio.gather(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK),
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK),
        )

        assert sum(results) == 2
        assert adapter.send.await_count == 2
        assert sorted(
            call.kwargs["chat_id"] for call in adapter.send.await_args_list
        ) == ["C1", "C2"]
        for oid in ("ob-1", "ob-2"):
            row = _row(oid)
            assert row is not None
            assert row["state"] == "delivered"

    @pytest.mark.parametrize("interruption", ["shutdown", "adapter_disappears"])
    @pytest.mark.asyncio
    async def test_runtime_claim_releases_without_send_when_interrupted_before_send(
        self, interruption, monkeypatch
    ):
        """A claimed runtime replay is refunded if its sender is no longer usable."""
        from gateway.config import Platform

        _record(platform="slack")
        dl.mark_failed("ob-1", "send_path_degraded")
        adapter = self._adapter()
        runner = self._runner(adapter)

        async def interrupt_before_send(_session_key):
            if interruption == "shutdown":
                runner._running = False
            else:
                runner.adapters.clear()

        monkeypatch.setattr(
            runner._async_session_store,
            "clear_resume_pending",
            AsyncMock(side_effect=interrupt_before_send),
        )

        redelivered = await runner._redeliver_failed_obligations_for_platform(
            Platform.SLACK
        )

        row = _row("ob-1")
        assert row is not None
        assert redelivered == 0
        adapter.send.assert_not_awaited()
        assert row["state"] == "failed"
        assert row["attempts"] == 0

    @pytest.mark.parametrize(
        ("send_success", "ledger_method"),
        [(True, "mark_delivered"), (False, "mark_failed")],
    )
    @pytest.mark.asyncio
    async def test_slow_state_update_does_not_block_event_loop(
        self, send_success, ledger_method
    ):
        import asyncio

        _record()
        _orphan("ob-1")
        runner = self._runner(self._adapter(success=send_success))
        slow_update, event_loop_witness, blocked_event_loop = _blocking_probe()

        with patch.object(dl, ledger_method, side_effect=slow_update):
            await asyncio.gather(
                runner._redeliver_pending_obligations(), event_loop_witness()
            )

        assert blocked_event_loop == []

    @pytest.mark.asyncio
    async def test_clear_resume_pending_before_send_so_a_hang_cannot_also_resume(
        self,
    ):
        """A hung redelivery send must still clear resume_pending.

        Otherwise a timed-out startup-restore gate would schedule resume and
        replay a turn whose answer is already in the ledger (#91969).
        """
        import asyncio

        _record()
        _orphan("ob-1")
        hang = asyncio.Event()

        async def hanging_send(**_kwargs):
            await hang.wait()
            return MagicMock(success=True, error="")

        adapter = MagicMock()
        adapter.send = hanging_send
        runner = self._runner(adapter)
        task = asyncio.create_task(runner._redeliver_pending_obligations())

        deadline = asyncio.get_running_loop().time() + 2
        while runner._async_session_store.clear_resume_pending.await_count == 0:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("resume_pending was not cleared before send")
            await asyncio.sleep(0)

        runner._async_session_store.clear_resume_pending.assert_awaited_once_with(
            "agent:main:slack:channel:C1"
        )
        assert not task.done()

        hang.set()
        assert await task == 1

    @pytest.mark.asyncio
    async def test_runtime_compensation_cap_blocks_over_capacity_claims_after_500_uncertain_retries_across_scopes(
        self, monkeypatch
    ):
        """Retained exact authorities exhaust one runner-wide admission budget."""
        import asyncio

        from gateway.config import Platform

        runner = self._runner(self._adapter())
        scope = (Platform.SLACK.value, "default")
        runner._runtime_claim_pending_compensations = {
            (f"pending-{index}", f"{index:032x}", scope) for index in range(500)
        }
        retry_calls = []
        claim_calls = []
        peek_barrier = threading.Barrier(2)

        def release_with_uncertainty(obligation_id, *, claim_token):
            retry_calls.append((obligation_id, claim_token))
            raise RuntimeError("release outcome unknown")

        def concurrent_peek(platform, *, profile=None):
            assert profile is None
            peek_barrier.wait()
            return [{"obligation_id": f"fresh-{platform}", "session_key": ""}]

        def record_claim(obligation_id, platform, *, profile=None):
            claim_calls.append((obligation_id, platform, profile))
            return None

        monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
        monkeypatch.setattr(dl, "release_runtime_claim", release_with_uncertainty)
        monkeypatch.setattr(dl, "peek_failed_for_runtime", concurrent_peek)
        monkeypatch.setattr(dl, "claim_failed_for_runtime", record_claim)

        assert await asyncio.gather(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK),
            runner._redeliver_failed_obligations_for_platform(Platform.TELEGRAM),
        ) == [0, 0]
        assert len(retry_calls) == 500
        assert claim_calls == []
        assert len(runner._runtime_claim_pending_compensations) == 500
        assert runner._runtime_claim_compensation_reservations == set()

    @pytest.mark.parametrize("released", [True, False])
    @pytest.mark.asyncio
    async def test_runtime_compensation_exact_retry_frees_one_concurrent_admission_slot(
        self, monkeypatch, released
    ):
        """Either exact terminal release frees only one held admission slot."""
        import asyncio

        from gateway.config import Platform

        runner = self._runner(self._adapter())
        scope = (Platform.SLACK.value, "default")
        released_compensation = ("released", "r" * 32, scope)
        runner._runtime_claim_pending_compensations = {
            (f"pending-{index}", f"{index:032x}", scope) for index in range(499)
        } | {released_compensation}
        loop = asyncio.get_running_loop()
        first_claim_entered = asyncio.Event()
        release_first_claim = threading.Event()
        claim_calls = []
        peek_calls = 0

        def release_exactly_one(obligation_id, *, claim_token):
            if obligation_id == "released":
                assert claim_token == released_compensation[1]
                return released
            raise RuntimeError("release outcome unknown")

        def candidates_for_generation(platform, *, profile=None):
            nonlocal peek_calls
            assert (platform, profile) == (Platform.SLACK.value, None)
            peek_calls += 1
            obligation_id = "first" if peek_calls == 1 else "second"
            return [{"obligation_id": obligation_id, "session_key": ""}]

        def block_first_claim(obligation_id, platform, *, profile=None):
            assert (platform, profile) == (Platform.SLACK.value, None)
            claim_calls.append(obligation_id)
            if obligation_id == "first":
                loop.call_soon_threadsafe(first_claim_entered.set)
                release_first_claim.wait()
            return {
                "obligation_id": obligation_id,
                "runtime_claim_token": f"{obligation_id}-" + "t" * 32,
            }

        monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
        monkeypatch.setattr(dl, "release_runtime_claim", release_exactly_one)
        monkeypatch.setattr(dl, "peek_failed_for_runtime", candidates_for_generation)
        monkeypatch.setattr(dl, "claim_failed_for_runtime", block_first_claim)
        monkeypatch.setattr(runner, "_redeliver_claimed_obligations", AsyncMock(return_value=1))

        first_generation = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await first_claim_entered.wait()
        try:
            # The reservation for first fills the one slot just freed by the
            # exact retry, so a concurrent generation cannot claim second.
            assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
            assert claim_calls == ["first"]
        finally:
            release_first_claim.set()

        assert await first_generation == 1
        assert runner._runtime_claim_compensation_reservations == set()
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        assert claim_calls == ["first", "second"]
        assert len(runner._runtime_claim_pending_compensations) == 499
        assert released_compensation not in runner._runtime_claim_pending_compensations

    @pytest.mark.parametrize("claim_outcome", ["none", "raises"])
    @pytest.mark.asyncio
    async def test_runtime_compensation_claim_failure_releases_admission_reservation(
        self, monkeypatch, claim_outcome
    ):
        """An unsuccessful claim gives its admission slot back exactly once."""
        from gateway.config import Platform

        runner = self._runner(self._adapter())
        scope = (Platform.SLACK.value, "default")
        runner._runtime_claim_pending_compensations = {
            (f"pending-{index}", f"{index:032x}", scope) for index in range(499)
        }
        claim_calls = []
        peek_calls = 0

        def release_with_uncertainty(*_args, **_kwargs):
            raise RuntimeError("release outcome unknown")

        def candidates_for_generation(platform, *, profile=None):
            nonlocal peek_calls
            assert (platform, profile) == (Platform.SLACK.value, None)
            peek_calls += 1
            obligation_id = "fails" if peek_calls == 1 else "after-failure"
            return [{"obligation_id": obligation_id, "session_key": ""}]

        def fail_then_claim(obligation_id, platform, *, profile=None):
            assert (platform, profile) == (Platform.SLACK.value, None)
            claim_calls.append(obligation_id)
            if obligation_id == "fails":
                if claim_outcome == "none":
                    return None
                raise RuntimeError("claim failed")
            return {
                "obligation_id": obligation_id,
                "runtime_claim_token": "a" * 32,
            }

        monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
        monkeypatch.setattr(dl, "release_runtime_claim", release_with_uncertainty)
        monkeypatch.setattr(dl, "peek_failed_for_runtime", candidates_for_generation)
        monkeypatch.setattr(dl, "claim_failed_for_runtime", fail_then_claim)
        monkeypatch.setattr(runner, "_redeliver_claimed_obligations", AsyncMock(return_value=1))

        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        assert runner._runtime_claim_compensation_reservations == set()
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 1
        assert claim_calls == ["fails", "after-failure"]
        assert runner._runtime_claim_compensation_reservations == set()

    @pytest.mark.asyncio
    async def test_runtime_compensation_cancellation_transfers_reservation_to_pending_without_double_charge(
        self, monkeypatch
    ):
        """An uncertain cancellation release keeps one pending slot, not two."""
        import asyncio

        from gateway.config import Platform

        runner = self._runner(self._adapter())
        scope = (Platform.SLACK.value, "default")
        runner._runtime_claim_pending_compensations = {
            (f"pending-{index}", f"{index:032x}", scope) for index in range(499)
        }
        loop = asyncio.get_running_loop()
        claim_entered = asyncio.Event()
        return_claim = threading.Event()
        claim_calls = []
        peek_calls = 0
        cancelled_token = "c" * 32

        def release_with_uncertainty(*_args, **_kwargs):
            raise RuntimeError("release outcome unknown")

        def candidates_for_generation(platform, *, profile=None):
            nonlocal peek_calls
            assert (platform, profile) == (Platform.SLACK.value, None)
            peek_calls += 1
            obligation_id = "cancelled" if peek_calls == 1 else "over-cap"
            return [{"obligation_id": obligation_id, "session_key": ""}]

        def claim_then_return_after_cancellation(obligation_id, platform, *, profile=None):
            assert (platform, profile) == (Platform.SLACK.value, None)
            claim_calls.append(obligation_id)
            if obligation_id == "cancelled":
                loop.call_soon_threadsafe(claim_entered.set)
                return_claim.wait()
                return {
                    "obligation_id": obligation_id,
                    "runtime_claim_token": cancelled_token,
                }
            return None

        monkeypatch.setattr(dl, "ledger_enabled", lambda: True)
        monkeypatch.setattr(dl, "release_runtime_claim", release_with_uncertainty)
        monkeypatch.setattr(dl, "peek_failed_for_runtime", candidates_for_generation)
        monkeypatch.setattr(dl, "claim_failed_for_runtime", claim_then_return_after_cancellation)

        cancelled_generation = asyncio.create_task(
            runner._redeliver_failed_obligations_for_platform(Platform.SLACK)
        )
        await claim_entered.wait()
        cancelled_generation.cancel()
        return_claim.set()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_generation

        cancelled_compensation = ("cancelled", cancelled_token, scope)
        assert len(runner._runtime_claim_pending_compensations) == 500
        assert cancelled_compensation in runner._runtime_claim_pending_compensations
        assert runner._runtime_claim_compensation_reservations == set()
        assert await runner._redeliver_failed_obligations_for_platform(Platform.SLACK) == 0
        assert claim_calls == ["cancelled"]


class TestAttemptsOnlySpentOnRealSends:
    """``attempts`` is the redelivery budget — it must buy a send.

    ``self.adapters`` only holds a platform after its ``connect()`` succeeded,
    and the sweep claimed every dead-owner row regardless. A platform that
    failed to connect this boot therefore burned one attempt per boot while
    the caller's ``adapter is None`` branch skipped it without sending — so
    after MAX_ATTEMPTS boots the row abandoned having never been sent once,
    losing exactly the response the ledger exists to guarantee. That failure
    correlates with the crash that created the obligation: the network
    trouble that killed the send tends to still be there on the next boot.
    """

    def test_absent_platform_does_not_burn_attempts(self):
        _record(platform="telegram")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            assert dl.sweep_recoverable(deliverable_platforms={"discord"}) == []

        row = dl.debug_rows()
        assert "abandoned" not in row
        with dl._connect() as conn:
            state, attempts = conn.execute(
                "SELECT state, attempts FROM delivery_obligations "
                "WHERE obligation_id=?", ("ob-1",),
            ).fetchone()
        assert attempts == 0, "an unsendable boot must not spend the budget"
        assert state == "attempting"

    def test_row_still_delivers_once_its_platform_returns(self):
        _record(platform="telegram")
        for _ in range(dl.MAX_ATTEMPTS + 2):
            _orphan("ob-1")
            dl.sweep_recoverable(deliverable_platforms={"discord"})

        _orphan("ob-1")
        claimed = dl.sweep_recoverable(deliverable_platforms={"telegram"})
        assert len(claimed) == 1
        assert claimed[0]["attempts"] == 1


class TestUnconnectedPlatformKeepsItsBudget:
    """End-to-end through the real runner: boots where the platform failed to
    connect must not consume the row's redelivery budget."""

    @staticmethod
    def _runner_without_slack():
        from gateway.run import GatewayRunner

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}  # slack failed to connect this boot
        _store = MagicMock()
        _store.clear_resume_pending = AsyncMock()
        _store._store = None
        runner.session_store = None
        runner._async_session_store = _store
        return runner

    @pytest.mark.asyncio
    async def test_row_survives_boots_where_its_platform_is_down(self):
        _record(platform="slack")
        dl.mark_attempting("ob-1")

        for _ in range(dl.MAX_ATTEMPTS + 1):
            _orphan("ob-1")
            runner = self._runner_without_slack()
            assert await runner._redeliver_pending_obligations() == 0

        assert _row("ob-1")["state"] != "abandoned", (
            "the obligation was abandoned without a single send being attempted"
        )
        assert _row("ob-1")["attempts"] == 0



class TestOwnerAlivePidProbe:
    """_owner_alive's no-start-time fallback must route through
    gateway.status._pid_exists, never a raw ``os.kill(pid, 0)`` probe.

    On Windows ``os.kill(pid, 0)`` is NOT a no-op: CPython maps sig=0 to
    ``GenerateConsoleCtrlEvent(0, pid)`` (bpo-14484), so probing a LIVE pid
    whose start time psutil could not read would Ctrl+C its console group.
    Pattern per the windows-native-support reference: patch
    ``gateway.status._pid_exists``, not ``os.kill``.
    """

    def _no_start_time(self, monkeypatch):
        from gateway import status

        monkeypatch.setattr(status, "get_process_start_time", lambda pid: None)

    def test_alive_when_pid_exists(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: True)
        assert dl._owner_alive(12345, 999) is True

    def test_dead_when_pid_gone(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
        assert dl._owner_alive(12345, 999) is False

    def test_raw_os_kill_probe_never_used(self, monkeypatch):
        """Regression guard: the probe must not touch os.kill when
        gateway.status._pid_exists is importable (i.e. always in-tree)."""
        from gateway import status

        self._no_start_time(monkeypatch)
        calls = []
        monkeypatch.setattr(status, "_pid_exists", lambda pid: calls.append(pid) or True)
        monkeypatch.setattr(
            dl.os, "kill", lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw os.kill probe used"))
        )
        assert dl._owner_alive(4242, 999) is True
        assert calls == [4242]

    def test_probe_exception_means_dead(self, monkeypatch):
        from gateway import status

        self._no_start_time(monkeypatch)

        def boom(pid):
            raise RuntimeError("probe blew up")

        monkeypatch.setattr(status, "_pid_exists", boom)
        assert dl._owner_alive(12345, 999) is False
