"""
Tests for Telegram polling network error recovery.

Specifically tests the fix for #3173 — when start_polling() fails after a
network error, the adapter must self-reschedule the next reconnect attempt
rather than silently leaving polling dead.
"""

import ast
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402
from gateway.run import GatewayRunner  # noqa: E402


@pytest.fixture(autouse=True)
def _no_auto_discovery(monkeypatch):
    """Disable DoH auto-discovery so connect() uses the plain builder chain."""
    async def _noop():
        return []
    monkeypatch.setattr("plugins.platforms.telegram.adapter.discover_fallback_ips", _noop)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


async def _complete_current_polling_generation(adapter: TelegramAdapter) -> None:
    verifier = adapter._polling_progress_verifier_task
    adapter._record_polling_progress(adapter._polling_generation)
    if verifier is not None:
        await verifier


@pytest.mark.asyncio
async def test_reconnect_self_schedules_on_start_polling_failure():
    """
    When start_polling() raises during a network error retry, the adapter must
    schedule a new _handle_polling_network_error task — otherwise polling stays
    dead with no further error callbacks to trigger recovery.

    Regression test for #3173: gateway becomes unresponsive after Telegram 502.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock(side_effect=Exception("Timed out"))

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # A retry task must have been added to _background_tasks
    pending = [t for t in adapter._background_tasks if not t.done()]
    assert len(pending) >= 1, (
        "Expected at least one self-rescheduled retry task in _background_tasks "
        f"after start_polling failure, got {len(pending)}"
    )

    # Clean up — cancel the pending retry so it doesn't run after the test
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


@pytest.mark.asyncio
async def test_retry_exhaustion_queues_reconnect_before_child_disconnect(tmp_path):
    """Fatal teardown must not cancel the gateway's reconnect handoff.

    The gateway runs ``disconnect()`` in a bounded child task.  If the current
    polling-recovery owner remains in ``_polling_error_task``, Telegram teardown
    cancels that parent while it is still awaiting the fatal handler, so the
    handler never gets to queue background reconnection.
    """
    config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="test-token")
        },
        sessions_dir=tmp_path / "sessions",
    )
    runner = GatewayRunner(config)
    adapter = _make_adapter()
    adapter._polling_network_error_count = 10  # MAX_NETWORK_RETRIES
    adapter.set_fatal_error_handler(runner._handle_adapter_fatal_error)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.delivery_router.adapters = runner.adapters

    recovery_task = asyncio.create_task(
        adapter._handle_polling_network_error(Exception("still failing"))
    )
    adapter._polling_error_task = recovery_task
    result = await asyncio.gather(recovery_task, return_exceptions=True)

    assert result == [None]
    assert runner.adapters == {}
    assert Platform.TELEGRAM in runner._failed_platforms
    assert runner._failed_platforms[Platform.TELEGRAM]["attempts"] == 0


# ---------------------------------------------------------------------------
# Connection pool drain tests (PR #16466 salvage)
# ---------------------------------------------------------------------------

def _make_mock_app():
    """Build a mock Application with an explicit polling request object."""
    mock_polling_req = AsyncMock()
    mock_polling_req.shutdown = AsyncMock()
    mock_polling_req.initialize = AsyncMock()

    mock_bot = MagicMock()
    mock_bot._request = (mock_polling_req, MagicMock())  # (getUpdates, general)

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot = mock_bot
    return mock_app, mock_polling_req


@pytest.mark.asyncio
async def test_initialize_still_runs_when_shutdown_fails():
    """If shutdown() raises, initialize() must still be attempted.

    This prevents a failed shutdown from leaving the request pool in a
    permanently closed state.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_app, mock_polling_req = _make_mock_app()
    mock_polling_req.shutdown = AsyncMock(side_effect=Exception("shutdown boom"))
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    # initialize MUST be called even though shutdown raised
    mock_polling_req.initialize.assert_called_once()
    mock_app.updater.start_polling.assert_called_once()


@pytest.mark.asyncio
async def test_reconnect_continues_if_drain_hangs(monkeypatch):
    """If the polling request drain HANGS (wedged httpx pool close on a
    CLOSE-WAIT socket), the reconnect ladder must still advance rather than
    freezing the tracked _polling_error_task forever.

    Regression test for #66377: an unbounded ``shutdown()`` /
    ``initialize()`` in ``_drain_polling_connections`` leaves the handler
    task pending, which gates every escalation path and silently kills the
    gateway. The drain awaits are bounded by ``_DRAIN_TIMEOUT``, so the
    handler must complete and reach ``start_polling`` within a hard bound.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_app, mock_polling_req = _make_mock_app()

    async def _hang(*args, **kwargs):
        await asyncio.Event().wait()  # never returns

    # Both drain awaits wedge indefinitely.
    mock_polling_req.shutdown = AsyncMock(side_effect=_hang)
    mock_polling_req.initialize = AsyncMock(side_effect=_hang)
    adapter._app = mock_app

    # Keep the drain timeout tiny so the test stays fast; the real default
    # is generous enough not to truncate healthy closes.
    monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 0.01, raising=False)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        # Hard outer bound: on unfixed code the drain hangs forever and this
        # trips; with the fix the inner wait_for releases well before it.
        await asyncio.wait_for(
            adapter._handle_polling_network_error(Exception("Timed out")),
            timeout=5,
        )

    # Ladder advanced past the wedged drain despite it never returning.
    mock_app.updater.start_polling.assert_called_once()
    assert adapter._polling_network_error_count == 2
    # The tracked task must not be stuck pending — otherwise every
    # escalation path stays gated behind an in-flight guard.
    assert (
        adapter._polling_error_task is None
        or adapter._polling_error_task.done()
    )


@pytest.mark.asyncio
async def test_reconnect_abandons_cancellation_resistant_polling_shutdown(monkeypatch):
    """A cancellation-resistant polling close must not wedge recovery.

    ``asyncio.wait_for`` cancels a timed-out close and then waits for that
    cancellation to finish. A PTB/httpcore close can swallow CancelledError
    while its cleanup waits on a lifecycle barrier, which otherwise leaves the
    tracked reconnect task pending forever.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1
    mock_app, mock_polling_req = _make_mock_app()
    adapter._app = mock_app

    release_shutdown = asyncio.Event()
    shutdown_cancelled = asyncio.Event()
    shutdown_finished = asyncio.Event()

    async def _cancellation_resistant_shutdown():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            shutdown_cancelled.set()
            await release_shutdown.wait()
        finally:
            shutdown_finished.set()

    mock_polling_req.shutdown = AsyncMock(
        side_effect=_cancellation_resistant_shutdown
    )
    monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 0.05)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        recovery = asyncio.create_task(
            adapter._handle_polling_network_error(Exception("Timed out"))
        )
        adapter._polling_error_task = recovery
        try:
            done, _ = await asyncio.wait({recovery}, timeout=2.0)

            assert recovery in done, (
                "reconnect remained blocked waiting for cancellation-resistant "
                "polling request shutdown() cleanup"
            )
            assert shutdown_cancelled.is_set()
            mock_polling_req.initialize.assert_awaited_once()
            mock_app.updater.start_polling.assert_awaited_once()
            assert (
                adapter._polling_error_task is None
                or adapter._polling_error_task.done()
            )
        finally:
            release_shutdown.set()
            await asyncio.wait_for(shutdown_finished.wait(), timeout=2.0)
            await asyncio.gather(recovery, return_exceptions=True)
            await _complete_current_polling_generation(adapter)
            assert not [task for task in adapter._background_tasks if not task.done()]


@pytest.mark.asyncio
async def test_drain_logs_sanitized_late_failure_from_abandoned_shutdown(monkeypatch, caplog):
    """An abandoned shutdown failure is observed and logged without its data."""
    adapter = _make_adapter()
    mock_app, mock_polling_req = _make_mock_app()
    adapter._app = mock_app

    release_shutdown = asyncio.Event()
    shutdown_cancelled = asyncio.Event()
    observer_finished = asyncio.Event()
    marker = (
        "token=123456:TOP_SECRET /private/telegram/session/4242 "
        "payload=untrusted chat_id=777 traceback"
    )

    class LateShutdownFailure(RuntimeError):
        pass

    async def _cancellation_resistant_shutdown():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            shutdown_cancelled.set()
            await release_shutdown.wait()
            raise LateShutdownFailure(marker)

    mock_polling_req.shutdown = AsyncMock(
        side_effect=_cancellation_resistant_shutdown
    )
    monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 0.01)

    created_tasks = []
    original_ensure_future = asyncio.ensure_future

    def _capture_task(awaitable, *, loop=None):
        task = original_ensure_future(awaitable, loop=loop)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(tg_adapter.asyncio, "ensure_future", _capture_task)

    with caplog.at_level("DEBUG", logger=tg_adapter.__name__):
        await adapter._drain_polling_connections()
        await asyncio.wait_for(shutdown_cancelled.wait(), timeout=2.0)
        shutdown_task = created_tasks[0]
        # This callback is registered after the production observer, so its
        # barrier proves the exception has been retrieved before assertions.
        shutdown_task.add_done_callback(lambda _task: observer_finished.set())
        release_shutdown.set()
        await asyncio.wait_for(observer_finished.wait(), timeout=2.0)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == tg_adapter.__name__
        and "Abandoned Telegram task failed after timeout" in record.getMessage()
    ]
    assert messages == [
        "Abandoned Telegram task failed after timeout (LateShutdownFailure)"
    ]
    logged = "\n".join(messages)
    for forbidden in (
        "TOP_SECRET",
        "/private/telegram/session/4242",
        "payload=untrusted",
        "token=123456",
        "chat_id=777",
        "traceback",
    ):
        assert forbidden not in logged
    mock_polling_req.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_drain_outer_cancellation_cancels_and_observes_resistant_shutdown(
    monkeypatch, caplog
):
    """Outer cancellation must not strand an unobserved drain child."""
    import gc

    adapter = _make_adapter()
    mock_app, mock_polling_req = _make_mock_app()
    adapter._app = mock_app

    shutdown_started = asyncio.Event()
    shutdown_cancelled = asyncio.Event()
    release_shutdown = asyncio.Event()
    observer_finished = asyncio.Event()

    class LateShutdownFailure(RuntimeError):
        pass

    async def _cancellation_resistant_shutdown():
        shutdown_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            shutdown_cancelled.set()
            await release_shutdown.wait()
            raise LateShutdownFailure("late child failure")

    mock_polling_req.shutdown = AsyncMock(
        side_effect=_cancellation_resistant_shutdown
    )
    monkeypatch.setattr(tg_adapter, "_DRAIN_TIMEOUT", 60.0)

    created_tasks = []
    original_ensure_future = asyncio.ensure_future

    def _capture_task(awaitable, *, loop=None):
        task = original_ensure_future(awaitable, loop=loop)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(tg_adapter.asyncio, "ensure_future", _capture_task)

    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    unobserved_task_contexts = []

    def _capture_unobserved_task(_loop, context):
        if context.get("message") == "Task exception was never retrieved":
            unobserved_task_contexts.append(context)
            return
        if prior_handler is not None:
            prior_handler(_loop, context)
        else:
            _loop.default_exception_handler(context)

    loop.set_exception_handler(_capture_unobserved_task)
    outer = asyncio.create_task(adapter._drain_polling_connections())
    cancellation_delivery = None
    try:
        with caplog.at_level("DEBUG", logger=tg_adapter.__name__):
            await asyncio.wait_for(shutdown_started.wait(), timeout=2.0)
            outer.cancel()
            done, _ = await asyncio.wait({outer}, timeout=2.0)
            assert outer in done, "outer drain cancellation must propagate immediately"
            with pytest.raises(asyncio.CancelledError):
                outer.result()

            cancellation_delivery = asyncio.create_task(shutdown_cancelled.wait())
            done, _ = await asyncio.wait({cancellation_delivery}, timeout=2.0)
            assert cancellation_delivery in done, (
                "outer drain cancellation must cancel its shutdown child"
            )

            pending_children = [task for task in created_tasks if not task.done()]
            assert len(pending_children) == 1
            shutdown_task = pending_children[0]
            # This callback is registered after the production observer, so its
            # barrier proves the exception has been retrieved before assertion.
            shutdown_task.add_done_callback(lambda _task: observer_finished.set())
            release_shutdown.set()
            await asyncio.wait_for(observer_finished.wait(), timeout=2.0)

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == tg_adapter.__name__
            and "Abandoned Telegram task failed after timeout" in record.getMessage()
        ]
        assert messages == [
            "Abandoned Telegram task failed after timeout (LateShutdownFailure)"
        ]
        assert not [task for task in created_tasks if not task.done()]
        created_tasks.clear()
        del shutdown_task
        gc.collect()
        assert unobserved_task_contexts == []
    finally:
        release_shutdown.set()
        if cancellation_delivery is not None and not cancellation_delivery.done():
            cancellation_delivery.cancel()
            await asyncio.gather(cancellation_delivery, return_exceptions=True)
        if not outer.done():
            outer.cancel()
            await asyncio.gather(outer, return_exceptions=True)
        for task in created_tasks:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        loop.set_exception_handler(prior_handler)


@pytest.mark.asyncio
async def test_strict_start_cancellation_cleans_abandoned_polling_app(monkeypatch):
    """A cancelled strict start must still release its abandoned PTB app."""
    adapter = _make_adapter()
    app = MagicMock()
    app.updater = MagicMock()
    adapter._app = app

    start_started = asyncio.Event()
    start_cancelled = asyncio.Event()
    start_finished = asyncio.Event()
    release_start = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_calls = []

    async def _cancellation_resistant_start(**_kwargs):
        start_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            start_cancelled.set()
            await release_start.wait()
        finally:
            start_finished.set()

    async def _cleanup(abandoned_app):
        cleanup_calls.append(abandoned_app)
        cleanup_started.set()

    app.updater.start_polling = _cancellation_resistant_start
    monkeypatch.setattr(tg_adapter, "_shutdown_abandoned_app", _cleanup)
    monkeypatch.setattr(tg_adapter, "_UPDATER_START_TIMEOUT", 60.0)

    owner = asyncio.create_task(
        adapter._start_polling_resilient(
            drop_pending_updates=False,
            error_callback=None,
            require_progress=True,
        )
    )
    try:
        await asyncio.wait_for(start_started.wait(), timeout=2.0)
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        await asyncio.wait_for(start_cancelled.wait(), timeout=2.0)
        await asyncio.wait_for(cleanup_started.wait(), timeout=0.1)
        assert cleanup_calls == [app]
    finally:
        release_start.set()
        await asyncio.wait_for(start_finished.wait(), timeout=2.0)


@pytest.mark.asyncio
async def test_connect_timeout_owns_late_initialize_cleanup_once(monkeypatch):
    """A timed-out initialize has one owner that shuts down after it finishes."""
    adapter = _make_adapter()
    initialize_started = asyncio.Event()
    initialize_cancelled = asyncio.Event()
    release_initialize = asyncio.Event()
    initialize_finished = asyncio.Event()
    release_shutdown = asyncio.Event()
    cleanup_finished = asyncio.Event()
    second_shutdown_started = asyncio.Event()

    old_app = MagicMock()
    old_app.bot = MagicMock()
    old_app.bot._request = ()
    old_app.shutdown_calls = 0
    old_app.state = "new"
    old_app.shutdown_saw_initialized = []

    async def _late_initialize():
        initialize_started.set()
        try:
            await release_initialize.wait()
        except asyncio.CancelledError:
            initialize_cancelled.set()
            await release_initialize.wait()
        old_app.state = "initialized"
        initialize_finished.set()

    async def _shutdown_old_app():
        old_app.shutdown_calls += 1
        if old_app.shutdown_calls == 2:
            second_shutdown_started.set()
        await release_shutdown.wait()
        old_app.shutdown_saw_initialized.append(old_app.state == "initialized")
        old_app.state = "shutdown"
        cleanup_finished.set()

    old_app.initialize = _late_initialize
    old_app.shutdown = _shutdown_old_app

    new_app = MagicMock()
    new_app.bot = MagicMock()
    new_app.bot._request = ()
    new_app.initialize = AsyncMock()
    new_app.start = AsyncMock()

    class _Builder:
        def token(self, _token):
            return self

        def request(self, _request):
            return self

        def get_updates_request(self, _request):
            return self

        def build(self):
            return apps.pop(0)

    class _Application:
        @staticmethod
        def builder():
            return _Builder()

    class _Request:
        def __init__(self, **_kwargs):
            pass

    apps = [old_app, new_app]

    async def _heartbeat():
        await asyncio.Event().wait()

    monkeypatch.setattr(tg_adapter, "Application", _Application)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _Request)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
    monkeypatch.setattr(adapter, "_register_handlers", lambda _app: None)
    monkeypatch.setattr(adapter, "_instrument_polling_request", lambda request: request)
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_start_polling_resilient", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", _heartbeat)
    monkeypatch.setenv("HERMES_TELEGRAM_INIT_TIMEOUT", "0.01")

    connect_task = None
    second_shutdown_waiter = None
    try:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            connect_task = asyncio.create_task(adapter.connect())
            await initialize_started.wait()
            await initialize_cancelled.wait()
            second_shutdown_waiter = asyncio.create_task(second_shutdown_started.wait())
            done, _ = await asyncio.wait(
                {connect_task, second_shutdown_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )

            assert old_app.shutdown_calls <= 1, (
                "a timed-out Application.initialize() must have exactly one "
                "old-app cleanup owner"
            )
            assert connect_task in done, (
                "reconnect must advance while cancellation-resistant initialize "
                "is still pending"
            )
            assert await connect_task is True
            new_app.initialize.assert_awaited_once()
            assert old_app.shutdown_calls == 0

            cleanup_tasks = list(adapter._abandoned_initialize_owners.values())
            assert len(cleanup_tasks) == 1
            cleanup_task = cleanup_tasks[0]
            release_initialize.set()
            await initialize_finished.wait()
            release_shutdown.set()
            await cleanup_task

        assert old_app.shutdown_calls == 1
        assert old_app.shutdown_saw_initialized == [True]
        assert old_app.state == "shutdown"
        assert not adapter._abandoned_initialize_owners
        assert not adapter._background_tasks
    finally:
        release_initialize.set()
        release_shutdown.set()
        if second_shutdown_waiter is not None and not second_shutdown_waiter.done():
            second_shutdown_waiter.cancel()
            await asyncio.gather(second_shutdown_waiter, return_exceptions=True)
        if connect_task is not None:
            await asyncio.gather(connect_task, return_exceptions=True)
        heartbeat_task = adapter._polling_heartbeat_task
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        pending = [task for task in adapter._background_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_connect_rebuild_failure_retires_abandoned_app_before_disconnect(
    monkeypatch,
):
    """A failed rebuild cannot make defensive disconnect own the old app."""
    adapter = _make_adapter()
    initialize_started = asyncio.Event()
    initialize_cancelled = asyncio.Event()
    release_initialize = asyncio.Event()
    shutdown_finished = asyncio.Event()

    old_app = MagicMock()
    old_app.bot = MagicMock()
    old_app.bot._request = ()
    old_app.updater = MagicMock()
    old_app.updater.running = False
    old_app.running = False
    old_app.shutdown_calls = 0
    old_app.state = "new"

    async def _late_initialize():
        initialize_started.set()
        try:
            await release_initialize.wait()
        except asyncio.CancelledError:
            initialize_cancelled.set()
            await release_initialize.wait()
        old_app.state = "initialized"

    async def _shutdown_old_app():
        old_app.shutdown_calls += 1
        old_app.state = "shutdown"
        shutdown_finished.set()

    old_app.initialize = _late_initialize
    old_app.shutdown = _shutdown_old_app

    class BuildFailure(RuntimeError):
        pass

    class _Builder:
        def token(self, _token):
            return self

        def request(self, _request):
            return self

        def get_updates_request(self, _request):
            return self

        def build(self):
            result = builds.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result

    class _Application:
        @staticmethod
        def builder():
            return _Builder()

    class _Request:
        def __init__(self, **_kwargs):
            pass

    builds = [old_app, BuildFailure("rebuild failed")]
    monkeypatch.setattr(tg_adapter, "Application", _Application)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _Request)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
    monkeypatch.setattr(adapter, "_register_handlers", lambda _app: None)
    monkeypatch.setattr(adapter, "_instrument_polling_request", lambda request: request)
    monkeypatch.setenv("HERMES_TELEGRAM_INIT_TIMEOUT", "0.01")

    connect_task = asyncio.create_task(adapter.connect())
    try:
        await initialize_started.wait()
        await initialize_cancelled.wait()
        assert await connect_task is False

        # Gateway's failed-connect path calls disconnect() defensively.  The
        # transferred app must already be retired, so this cannot shut it down.
        await adapter.disconnect()
        assert adapter._app is not old_app
        assert old_app.shutdown_calls == 0

        release_initialize.set()
        await shutdown_finished.wait()
        assert old_app.shutdown_calls == 1
        assert old_app.state == "shutdown"
    finally:
        release_initialize.set()
        await asyncio.gather(connect_task, return_exceptions=True)
        registry = getattr(adapter, "_abandoned_initialize_owners", {})
        cleanup_tasks = list(registry.values()) if isinstance(registry, dict) else []
        cleanup_tasks.extend(adapter._background_tasks)
        for task in cleanup_tasks:
            if not task.done():
                task.cancel()
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_teardown_retains_never_terminal_abandoned_initialize_ownership(
    monkeypatch,
):
    """Base task clearing cannot untrack the retained old-app cleanup owner."""
    from gateway.platforms import base as gateway_base

    adapter = _make_adapter()
    initialize_started = asyncio.Event()
    initialize_cancelled = asyncio.Event()
    release_initialize = asyncio.Event()
    initialize_finished = asyncio.Event()
    old_shutdown_finished = asyncio.Event()

    old_app = MagicMock()
    old_app.bot = MagicMock()
    old_app.bot._request = ()
    old_app.shutdown_calls = 0
    old_app.state = "new"

    async def _never_terminal_initialize():
        initialize_started.set()
        try:
            await release_initialize.wait()
        except asyncio.CancelledError:
            initialize_cancelled.set()
            await release_initialize.wait()
        old_app.state = "initialized"
        initialize_finished.set()

    async def _shutdown_old_app():
        old_app.shutdown_calls += 1
        old_app.state = "shutdown"
        old_shutdown_finished.set()

    old_app.initialize = _never_terminal_initialize
    old_app.shutdown = _shutdown_old_app

    new_app = MagicMock()
    new_app.bot = MagicMock()
    new_app.bot._request = ()
    new_app.updater = MagicMock()
    new_app.updater.running = False
    new_app.running = False
    new_app.initialize = AsyncMock()
    new_app.start = AsyncMock()
    new_app.shutdown = AsyncMock()

    class _Builder:
        def token(self, _token):
            return self

        def request(self, _request):
            return self

        def get_updates_request(self, _request):
            return self

        def build(self):
            return apps.pop(0)

    class _Application:
        @staticmethod
        def builder():
            return _Builder()

    class _Request:
        def __init__(self, **_kwargs):
            pass

    apps = [old_app, new_app]
    created_tasks = []
    original_ensure_future = asyncio.ensure_future

    def _capture_task(awaitable, *, loop=None):
        task = original_ensure_future(awaitable, loop=loop)
        created_tasks.append(task)
        return task

    async def _heartbeat():
        await asyncio.Event().wait()

    monkeypatch.setattr(tg_adapter, "Application", _Application)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _Request)
    monkeypatch.setattr(tg_adapter.asyncio, "ensure_future", _capture_task)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
    monkeypatch.setattr(adapter, "_register_handlers", lambda _app: None)
    monkeypatch.setattr(adapter, "_instrument_polling_request", lambda request: request)
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_start_polling_resilient", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", _heartbeat)
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    monkeypatch.setattr(adapter, "_cancel_pending_delivery_tasks", AsyncMock())
    monkeypatch.setenv("HERMES_TELEGRAM_INIT_TIMEOUT", "0.01")

    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    unobserved_task_contexts = []

    def _capture_unobserved_task(_loop, context):
        if context.get("message") == "Task exception was never retrieved":
            unobserved_task_contexts.append(context)
            return
        if prior_handler is not None:
            prior_handler(_loop, context)
        else:
            _loop.default_exception_handler(context)

    loop.set_exception_handler(_capture_unobserved_task)
    connect_task = asyncio.create_task(adapter.connect())
    cleanup_task = None
    try:
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await initialize_started.wait()
            await initialize_cancelled.wait()
            assert await connect_task is True

        cleanup_task = next(
            task
            for task in created_tasks
            if "_shutdown_after_initialize" in task.get_coro().__qualname__
        )
        assert cleanup_task is not created_tasks[0]
        assert not cleanup_task.done()
        cleanup_owner_observed = asyncio.Event()
        cleanup_task.add_done_callback(lambda _task: cleanup_owner_observed.set())

        async def _force_base_timeout(awaitable, timeout):
            if not awaitable.done():
                awaitable.cancel()
            await asyncio.gather(awaitable, return_exceptions=True)
            raise asyncio.TimeoutError()

        # Drive base's five-second branch without waiting: it cancels and clears
        # generic tasks, exactly as gateway teardown does for a straggler.
        with monkeypatch.context() as base_patch:
            base_patch.setattr(gateway_base.asyncio, "wait_for", _force_base_timeout)
            await adapter.cancel_background_tasks()

        assert not adapter._background_tasks
        owners = getattr(adapter, "_abandoned_initialize_owners", {})
        assert cleanup_task in owners.values(), (
            "a never-terminal abandoned initializer must stay in the adapter-owned "
            "registry after generic background-task clearing"
        )
        assert len(owners) == 1
        assert not cleanup_task.done()

        # A new call fails closed instead of adding a second retained owner; the
        # first real reconnect already advanced to ``new_app`` without waiting.
        assert await adapter.connect() is False
        assert list(owners.values()) == [cleanup_task]

        await adapter.disconnect()
        assert not cleanup_task.done()

        release_initialize.set()
        await initialize_finished.wait()
        await old_shutdown_finished.wait()
        await cleanup_task
        await cleanup_owner_observed.wait()
        assert old_app.shutdown_calls == 1
        assert old_app.state == "shutdown"
        assert owners == {}
        assert unobserved_task_contexts == []
    finally:
        release_initialize.set()
        await asyncio.gather(connect_task, return_exceptions=True)
        if cleanup_task is not None and not cleanup_task.done():
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
        heartbeat_task = adapter._polling_heartbeat_task
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        loop.set_exception_handler(prior_handler)


@pytest.mark.asyncio
async def test_cleanup_scheduling_failure_uses_fallback_owner_without_error_text(
    monkeypatch, caplog
):
    """A failed primary cleanup registration transfers to one fallback owner."""
    adapter = _make_adapter()
    initialize_started = asyncio.Event()
    initialize_cancelled = asyncio.Event()
    release_initialize = asyncio.Event()
    shutdown_finished = asyncio.Event()
    secret = "token=123456:TOP_SECRET /private/telegram/session/4242"

    old_app = MagicMock()
    old_app.bot = MagicMock()
    old_app.bot._request = ()
    old_app.shutdown_calls = 0
    old_app.state = "new"

    async def _late_initialize():
        initialize_started.set()
        try:
            await release_initialize.wait()
        except asyncio.CancelledError:
            initialize_cancelled.set()
            await release_initialize.wait()
        old_app.state = "initialized"

    async def _shutdown_old_app():
        old_app.shutdown_calls += 1
        old_app.state = "shutdown"
        shutdown_finished.set()

    old_app.initialize = _late_initialize
    old_app.shutdown = _shutdown_old_app

    new_app = MagicMock()
    new_app.bot = MagicMock()
    new_app.bot._request = ()
    new_app.initialize = AsyncMock()
    new_app.start = AsyncMock()

    class _Builder:
        def token(self, _token):
            return self

        def request(self, _request):
            return self

        def get_updates_request(self, _request):
            return self

        def build(self):
            return apps.pop(0)

    class _Application:
        @staticmethod
        def builder():
            return _Builder()

    class _Request:
        def __init__(self, **_kwargs):
            pass

    class CleanupSchedulingFailure(RuntimeError):
        pass

    apps = [old_app, new_app]
    original_ensure_future = asyncio.ensure_future

    def _fail_primary_cleanup_once(awaitable, *, loop=None):
        qualname = getattr(getattr(awaitable, "cr_code", None), "co_qualname", "")
        if "_shutdown_after_initialize" in qualname:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise CleanupSchedulingFailure(secret)
        return original_ensure_future(awaitable, loop=loop)

    async def _heartbeat():
        await asyncio.Event().wait()

    monkeypatch.setattr(tg_adapter, "Application", _Application)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _Request)
    monkeypatch.setattr(tg_adapter.asyncio, "ensure_future", _fail_primary_cleanup_once)
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *_args: True)
    monkeypatch.setattr(adapter, "_register_handlers", lambda _app: None)
    monkeypatch.setattr(adapter, "_instrument_polling_request", lambda request: request)
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_start_polling_resilient", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_start_post_connect_housekeeping", lambda: None)
    monkeypatch.setattr(adapter, "_polling_heartbeat_loop", _heartbeat)
    monkeypatch.setenv("HERMES_TELEGRAM_INIT_TIMEOUT", "0.01")

    loop = asyncio.get_running_loop()
    prior_handler = loop.get_exception_handler()
    unobserved_task_contexts = []

    def _capture_unobserved_task(_loop, context):
        if context.get("message") == "Task exception was never retrieved":
            unobserved_task_contexts.append(context)
            return
        if prior_handler is not None:
            prior_handler(_loop, context)
        else:
            _loop.default_exception_handler(context)

    loop.set_exception_handler(_capture_unobserved_task)
    connect_task = asyncio.create_task(adapter.connect())
    cleanup_task = None
    try:
        with caplog.at_level("DEBUG", logger=tg_adapter.__name__), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            await initialize_started.wait()
            await initialize_cancelled.wait()
            assert await connect_task is True

        messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == tg_adapter.__name__
            and "Abandoned Telegram init cleanup scheduling failed" in record.getMessage()
        ]
        assert messages == [
            "Abandoned Telegram init cleanup scheduling failed (CleanupSchedulingFailure)"
        ]
        assert secret not in "\n".join(messages)
        assert adapter._app is new_app
        assert old_app.shutdown_calls == 0

        owners = adapter._abandoned_initialize_owners
        assert len(owners) == 1
        cleanup_task = next(iter(owners.values()))
        assert not cleanup_task.done()
        cleanup_owner_observed = asyncio.Event()
        cleanup_task.add_done_callback(lambda _task: cleanup_owner_observed.set())

        release_initialize.set()
        await shutdown_finished.wait()
        await cleanup_task
        await cleanup_owner_observed.wait()
        assert old_app.shutdown_calls == 1
        assert old_app.state == "shutdown"
        assert owners == {}
        assert unobserved_task_contexts == []
    finally:
        release_initialize.set()
        await asyncio.gather(connect_task, return_exceptions=True)
        if cleanup_task is not None and not cleanup_task.done():
            cleanup_task.cancel()
            await asyncio.gather(cleanup_task, return_exceptions=True)
        heartbeat_task = adapter._polling_heartbeat_task
        if heartbeat_task is not None and not heartbeat_task.done():
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        loop.set_exception_handler(prior_handler)


@pytest.mark.asyncio
async def test_reconnect_stop_deadline_does_not_wait_for_cancel_cleanup(monkeypatch):
    """A cancellation-resistant PTB stop must not freeze the retry ladder.

    ``asyncio.wait_for`` waits for the cancelled coroutine to finish.  AnyIO's
    cancellation-shielded httpcore cleanup can therefore leave ``stop()``
    pending forever after the timeout fires: the gateway process stays alive,
    but no later Telegram retry runs.  The wall-clock deadline must abandon
    that task and escalate to a fresh adapter without reusing the Updater.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    release_stop = asyncio.Event()
    stop_cancelled = asyncio.Event()
    lifecycle_lock = asyncio.Lock()

    async def _cancellation_resistant_stop():
        async with lifecycle_lock:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                stop_cancelled.set()
                await release_stop.wait()

    async def _start_polling_with_same_lock(*args, **kwargs):
        async with lifecycle_lock:
            return None

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock(side_effect=_cancellation_resistant_stop)
    mock_updater.start_polling = AsyncMock(side_effect=_start_polling_with_same_lock)

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot = MagicMock()
    mock_app.bot._request = ()
    adapter._app = mock_app
    adapter._notify_fatal_error = AsyncMock()

    monkeypatch.setattr(tg_adapter, "_UPDATER_STOP_TIMEOUT", 0.01)
    with patch("asyncio.sleep", new_callable=AsyncMock):
        recovery = asyncio.create_task(
            adapter._handle_polling_network_error(Exception("Timed out"))
        )
        done, _ = await asyncio.wait({recovery}, timeout=0.2)

    try:
        assert recovery in done, (
            "reconnect remained blocked waiting for cancellation-shielded "
            "updater.stop() cleanup"
        )
        assert stop_cancelled.is_set()
        assert adapter.has_fatal_error
        adapter._notify_fatal_error.assert_awaited_once()
        mock_updater.start_polling.assert_not_awaited()
    finally:
        release_stop.set()
        if not recovery.done():
            recovery.cancel()
        await asyncio.gather(recovery, return_exceptions=True)


@pytest.mark.asyncio
async def test_heartbeat_force_escalates_wedged_recovery_task(monkeypatch):
    """#66377: the heartbeat is an independent, cause-agnostic watchdog.

    Every recovery path (ladder re-entry, pending-update probe, PTB error
    callback) gates new recovery on ``_polling_error_task.done()``. If that task
    wedges on ANY hung await — not just the drain closed by #66492 — the gateway
    stays alive but deaf with nothing retrying. The heartbeat must detect a
    recovery task that stays in-flight past ``_POLLING_ERROR_TASK_STUCK_TIMEOUT``
    and force a retryable-fatal so the background reconnector rebuilds the
    adapter.
    """
    adapter = _make_adapter()

    async def _wedged():
        await asyncio.Event().wait()  # never completes — simulates the hang

    wedged_task = asyncio.ensure_future(_wedged())
    adapter._polling_error_task = wedged_task

    mock_bot = MagicMock()
    mock_bot.get_me = AsyncMock()
    mock_app = MagicMock()
    mock_app.bot = mock_bot
    adapter._app = mock_app
    adapter._probe_pending_updates = AsyncMock()
    adapter._notify_fatal_error = AsyncMock()

    # Controllable monotonic clock advanced by each (mocked) heartbeat sleep so
    # the same wedged task is observed across the stuck threshold deterministically.
    clock = [1000.0]

    async def _fake_sleep(*_a, **_k):
        clock[0] += 200.0

    monkeypatch.setattr(tg_adapter.time, "monotonic", lambda: clock[0])

    with patch("asyncio.sleep", new=AsyncMock(side_effect=_fake_sleep)):
        await asyncio.wait_for(adapter._polling_heartbeat_loop(), timeout=5)

    assert adapter.has_fatal_error, "wedged recovery task must force a fatal escalation"
    adapter._notify_fatal_error.assert_awaited()

    wedged_task.cancel()
    try:
        await wedged_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_conflict_retry_also_drains_polling_connections():
    """_handle_polling_conflict must also drain the polling pool on retry."""
    adapter = _make_adapter()
    adapter._polling_conflict_count = 0

    mock_app, mock_polling_req = _make_mock_app()
    adapter._app = mock_app

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_conflict(Exception("Conflict: terminated by other getUpdates"))

    # Polling request must be drained during conflict retry too
    mock_polling_req.shutdown.assert_called_once()
    mock_polling_req.initialize.assert_called_once()
    mock_app.updater.start_polling.assert_called_once()


@pytest.mark.asyncio
async def test_drain_helper_noop_without_app():
    """_drain_polling_connections must be a no-op when _app is None."""
    adapter = _make_adapter()
    adapter._app = None
    # Should not raise
    await adapter._drain_polling_connections()


# ── Heartbeat probe ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_heartbeat_probe_reenters_ladder_when_updater_not_running(monkeypatch):
    """
    If Updater.running is False at the progress deadline, re-enter recovery.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = False

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock()
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()
    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    mock_app.bot.get_me.assert_not_called()
    # Recovery is scheduled through _schedule_polling_recovery (#63243), so
    # the ladder runs as the tracked _polling_error_task.
    task = adapter._polling_error_task
    assert task is not None
    await task
    adapter._handle_polling_network_error.assert_awaited_once()
    err = adapter._handle_polling_network_error.await_args.args[0]
    assert isinstance(err, RuntimeError)
    assert "not running" in str(err).lower()


@pytest.mark.asyncio
async def test_heartbeat_probe_ignores_auth_errors(monkeypatch):
    """
    Auth/validation failures from the post-reconnect probe must not enter the
    network-reconnect ladder (#63243): a revoked token would otherwise churn
    through stop/drain/start_polling cycles that mask the real failure.
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True

    # Name-shaped like PTB's InvalidToken; _looks_like_network_error excludes
    # it by class name, matching real PTB semantics.
    invalid_token = type("InvalidToken", (Exception,), {})("token revoked")

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(side_effect=invalid_token)
    adapter._app = mock_app

    adapter._handle_polling_network_error = AsyncMock()
    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    assert adapter._polling_error_task is None
    adapter._handle_polling_network_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_heartbeat_probe_defers_to_inflight_recovery(monkeypatch):
    """
    A probe failure while another recovery is mid-flight must not start a
    second concurrent stop/drain/start_polling sequence (#63243) — overlapping
    recoveries produce dueling getUpdates sessions (self-inflicted 409s).
    """
    adapter = _make_adapter()

    mock_updater = MagicMock()
    mock_updater.running = True

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(side_effect=ConnectionError("pool wedged"))
    adapter._app = mock_app

    inflight = MagicMock()
    inflight.done.return_value = False
    adapter._polling_error_task = inflight

    adapter._handle_polling_network_error = AsyncMock()
    generation, progress = adapter._begin_polling_generation()
    monkeypatch.setattr(tg_adapter, "_POLLING_PROGRESS_TIMEOUT", 0)

    await adapter._verify_polling_after_reconnect(generation, progress)

    assert adapter._polling_error_task is inflight
    adapter._handle_polling_network_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconnect_schedules_heartbeat_probe_on_success():
    """
    After a successful start_polling() in the reconnect path, a probe task
    must be added to _background_tasks. Without it, a wedged Updater would
    sit silent indefinitely with no further error_callback to advance the
    reconnect ladder.
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 1

    mock_updater = MagicMock()
    mock_updater.running = True
    mock_updater.stop = AsyncMock()
    mock_updater.start_polling = AsyncMock()  # succeeds

    mock_app = MagicMock()
    mock_app.updater = mock_updater
    mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
    adapter._app = mock_app

    initial_count = len(adapter._background_tasks)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await adapter._handle_polling_network_error(Exception("Bad Gateway"))

    assert len(adapter._background_tasks) > initial_count, (
        "Expected a heartbeat probe task to be scheduled after a successful "
        "reconnect's start_polling()"
    )

    # Clean up.
    pending = [t for t in adapter._background_tasks if not t.done()]
    for t in pending:
        t.cancel()
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass


# ── Persistent heartbeat loop (_polling_heartbeat_loop) ──────────────────────
#
# These tests cover the continuous CLOSE-WAIT detection loop that fixes the bug
# (#48495) where a dead Telegram TCP socket caused the gateway to stop receiving
# messages silently. The _verify_polling_after_reconnect tests above cover the
# one-shot post-reconnect probe; these cover the background loop that runs for
# the gateway's full lifetime in polling mode.
#
# Loop structure: while True: sleep(INTERVAL) → fatal/app checks → get_me().
# So with cancel raised on the Nth patched sleep, get_me() fires (N-1) times.


@pytest.mark.asyncio
async def test_heartbeat_loop_skips_reconnect_if_already_in_progress():
    """If a reconnect task is already running, the heartbeat must not spawn another."""
    adapter = _make_adapter()

    # Simulate an already-running reconnect task.
    existing_task = asyncio.get_event_loop().create_task(asyncio.sleep(0.2))
    adapter._polling_error_task = existing_task
    adapter._handle_polling_network_error = AsyncMock()

    mock_app = MagicMock()
    adapter._app = mock_app

    sleep_call = 0

    async def fast_sleep(seconds):
        nonlocal sleep_call
        sleep_call += 1
        if sleep_call >= 3:
            raise asyncio.CancelledError()

    async def timeout_wait_for(coro, timeout):
        if asyncio.iscoroutine(coro):
            coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.sleep", side_effect=fast_sleep):
        with patch("plugins.platforms.telegram.adapter.asyncio.wait_for", side_effect=timeout_wait_for):
            await adapter._polling_heartbeat_loop()

    # _handle_polling_network_error must NOT have been called — existing task still running.
    adapter._handle_polling_network_error.assert_not_awaited()

    existing_task.cancel()
    try:
        await existing_task
    except (asyncio.CancelledError, Exception):
        pass


async def _heartbeat_exception_case(exc, *, pending_probe=False):
    adapter = _make_adapter()
    reconnect_handler = AsyncMock()
    adapter._handle_polling_network_error = reconnect_handler  # type: ignore[method-assign]
    mock_app = MagicMock()
    mock_app.updater.running = True
    if pending_probe:
        mock_app.bot.get_me = AsyncMock(return_value=MagicMock())
        mock_app.bot.get_webhook_info = AsyncMock(side_effect=exc)
    else:
        mock_app.bot.get_me = AsyncMock(side_effect=exc)
    adapter._app = mock_app

    sleep_calls = 0

    async def fast_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError()

    with patch("asyncio.sleep", side_effect=fast_sleep):
        await adapter._polling_heartbeat_loop()
    await asyncio.sleep(0)
    return adapter


def _calls_shared_network_classifier(node):
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "_looks_like_network_error"
        for child in ast.walk(node)
    )


# ── Bootstrap degradation: keep polling alive during outages (#47508) ────


@pytest.mark.asyncio
async def test_polling_bootstrap_conflict_schedules_conflict_recovery_task():
    """Initial 409 polling conflict should also be recovered in background."""
    adapter = _make_adapter()
    mock_updater = MagicMock()
    mock_updater.start_polling = AsyncMock(
        side_effect=Exception("Conflict: terminated by other getUpdates request")
    )
    mock_app = MagicMock()
    mock_app.updater = mock_updater
    adapter._app = mock_app
    adapter._handle_polling_conflict = AsyncMock()

    result = await adapter._start_polling_resilient(
        drop_pending_updates=True,
        error_callback=lambda error: None,
    )

    assert result is False
    pending = [t for t in adapter._background_tasks if not t.done()]
    assert pending, "expected background conflict recovery task"
    for task in pending:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
    assert not adapter.has_fatal_error


@pytest.mark.asyncio
async def test_handle_polling_network_error_updater_stop_timeout():
    """updater.stop() hanging (CLOSE-WAIT) must not block the reconnect ladder.

    When the underlying TCP connection is in CLOSE-WAIT, PTB's polling task is
    blocked on epoll on the dead socket.  updater.stop() awaits that task and
    therefore hangs indefinitely.  The wall-clock deadline abandons the stop
    task and escalates to fresh-adapter recovery instead of calling
    start_polling() while PTB's shared lifecycle lock may still be held.

    This test simulates the hang by making stop() outlive the deadline and
    verifies that the current Updater is not drained or restarted afterward.
    Refs: NousResearch/hermes-agent#58270
    """
    adapter = _make_adapter()
    adapter._polling_network_error_count = 0

    # Build a fake app whose updater.stop() can make no progress until it is
    # cancelled. An Event, rather than a short sleep, prevents scheduler delay
    # from letting the double finish before the wall-clock deadline is observed.
    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = True
    stop_started = asyncio.Event()
    stop_cancelled = asyncio.Event()
    release_stop = asyncio.Event()

    async def _hanging_stop():
        stop_started.set()
        try:
            await release_stop.wait()
        except asyncio.CancelledError:
            stop_cancelled.set()
            raise

    app.updater.stop = _hanging_stop
    app.updater.start_polling = AsyncMock()
    adapter._app = app
    adapter._notify_fatal_error = AsyncMock()

    drain_called = []

    async def _fake_drain():
        drain_called.append(True)

    adapter._drain_polling_connections = _fake_drain

    start_polling_called = []

    async def _fake_start_polling(**kwargs):
        start_polling_called.append(True)

    app.updater.start_polling = AsyncMock(side_effect=_fake_start_polling)

    # Shrink the stop() watchdog bound so the test completes fast instead of
    # waiting the full _UPDATER_STOP_TIMEOUT. Patching the named constant is
    # cleaner than monkeypatching asyncio.wait_for process-wide.
    import plugins.platforms.telegram.adapter as _mod

    recovery = None
    try:
        with patch.object(_mod, "_UPDATER_STOP_TIMEOUT", 0.05), patch(
            "asyncio.sleep", new_callable=AsyncMock
        ):
            recovery = asyncio.create_task(
                adapter._handle_polling_network_error(OSError("CLOSE-WAIT test"))
            )
            adapter._polling_error_task = recovery
            assert adapter._polling_error_task is recovery
            await asyncio.wait_for(stop_started.wait(), timeout=2.0)
            await asyncio.wait_for(stop_cancelled.wait(), timeout=2.0)
            await asyncio.wait_for(recovery, timeout=2.0)

        # A timed-out stop may still hold PTB's lifecycle lock. Reusing this
        # Updater would wedge start_polling() behind it, so recovery must hand
        # the runner a retryable fatal and rebuild the adapter instead.
        assert recovery.done()
        assert adapter._polling_error_task is None
        assert adapter.has_fatal_error
        adapter._notify_fatal_error.assert_awaited_once()
        assert not drain_called
        assert not start_polling_called
        assert not [task for task in adapter._background_tasks if not task.done()]
    finally:
        release_stop.set()
        if recovery is not None and not recovery.done():
            recovery.cancel()
        if recovery is not None:
            await asyncio.gather(recovery, return_exceptions=True)


@pytest.mark.asyncio
async def test_disconnect_releases_token_lock_before_wedged_app_shutdown(monkeypatch):
    """#80598: token lock must drop even when app.shutdown() never returns.

    The reconnect watcher creates a fresh adapter that re-acquires the bot-token
    lock. If disconnect only releases the lock after a wedged PTB shutdown, the
    watcher fails forever with a lock conflict while the process stays alive.
    """
    adapter = _make_adapter()
    released = []
    monkeypatch.setattr(
        adapter, "_release_platform_lock", lambda: released.append(True)
    )
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    monkeypatch.setattr(adapter, "_cancel_pending_delivery_tasks", AsyncMock())

    app = MagicMock()
    app.updater = MagicMock()
    app.updater.running = False
    app.running = True
    app.stop = AsyncMock()

    async def _hanging_shutdown():
        await asyncio.Event().wait()

    app.shutdown = _hanging_shutdown
    adapter._app = app
    adapter._bot = MagicMock()

    monkeypatch.setattr(tg_adapter, "_DISCONNECT_STEP_TIMEOUT", 0.01)

    await asyncio.wait_for(adapter.disconnect(), timeout=1.0)

    assert released, "token lock must be released before wedged shutdown"
    assert adapter._app is None


@pytest.mark.asyncio
async def test_disconnect_advances_past_cancellation_swallowing_lifecycle(monkeypatch):
    """#80598: lifecycle tasks that swallow CancelledError must not wedge disconnect."""
    adapter = _make_adapter()
    monkeypatch.setattr(adapter, "_release_platform_lock", MagicMock())
    monkeypatch.setattr(adapter, "_set_status_indicator", AsyncMock())
    monkeypatch.setattr(adapter, "_cancel_pending_delivery_tasks", AsyncMock())

    release = asyncio.Event()

    async def swallow_cancel():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    wedged = asyncio.create_task(swallow_cancel())
    adapter._polling_error_task = wedged
    adapter._app = None
    adapter._bot = None

    monkeypatch.setattr(tg_adapter, "_DISCONNECT_STEP_TIMEOUT", 0.01)

    await asyncio.wait_for(adapter.disconnect(), timeout=1.0)
    assert adapter._polling_error_task is None

    release.set()
    await asyncio.wait({wedged}, timeout=0.2)
