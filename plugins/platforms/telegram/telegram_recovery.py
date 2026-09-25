"""Telegram delivery recovery: replies refused while polling was degraded go back to the runner when
polling recovers IN PLACE, and an ``initialize()`` whose connect was cancelled keeps one owner until its
transports are closed."""

from __future__ import annotations

import asyncio
import contextvars
import logging

logger = logging.getLogger(__name__)

# How long an abandoned initialize() may unwind before its transports are closed under it: long enough for
# a cancellation-shielded httpcore connect to return, bounded so a wedged one cannot hold the pools open.
_ABANDONED_INIT_UNWIND_TIMEOUT = 15.0


def schedule_in_place_redelivery(adapter, generation: int, generation_context: contextvars.ContextVar) -> None:
    """Hand this process's ``send_path_degraded`` rows back to the runner after an in-place recovery.

    Such rows are reconnect-only: ``pending_retries`` keeps them off the redelivery timer and
    ``_finalize_delivery_obligation`` replays them only when a reconnect installed a DIFFERENT adapter.
    Polling that recovers inside the same instance swaps nothing, so without this the reply waits for
    the next restart."""
    runner = getattr(adapter, "gateway_runner", None)
    if runner is None:
        return
    # Spawned from inside the getUpdates poll: the handoff must not carry that poll's generation (the
    # same reset ``_generation_error_callback`` applies to PTB's error callback).
    context = contextvars.copy_context()
    context.run(generation_context.set, None)
    task = asyncio.get_running_loop().create_task(
        _redeliver_after_in_place_recovery(adapter, runner, generation), context=context)
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)


async def _redeliver_after_in_place_recovery(adapter, runner, generation: int) -> None:
    # The edge is re-read on the loop: by now it may belong to a fenced or newer generation.
    if (adapter._teardown_started or adapter.has_fatal_error or adapter._send_path_degraded
            or generation != adapter._polling_generation
            or getattr(adapter, "gateway_runner", None) is not runner or not getattr(runner, "_running", False)):
        return
    from gateway.run import _async_profile_runtime_scope

    profile = getattr(adapter, "_owner_profile", None)
    # Rows live in the OWNING profile's state.db; this runs outside any turn, so bind it explicitly
    # (``None`` binds the launch profile once the process multiplexes).
    home = runner._routed_profile_home(profile) if profile else None
    try:
        async with runner._async_scope_or_null(_async_profile_runtime_scope, home):
            await runner._redeliver_failed_obligations_for_platform(adapter.platform, profile=profile)
    except Exception:
        logger.warning("[%s] Redelivery after in-place polling recovery failed", adapter.name, exc_info=True)


def retain_abandoned_initialize(adapter, init_task: asyncio.Future, app) -> None:
    """Own an ``initialize()`` abandoned by its caller's cancellation until the app's transports close.

    ``run_bounded_async`` cancels and abandons the child when the CALLER is cancelled (runner connect
    timeout, startup abort) but runs no ``on_abandon`` there, and ``app.shutdown()`` no-ops for an app
    that never finished initializing, so nothing else closes its httpx pools. The owner lets the child
    unwind first: ``HTTPXRequest.initialize`` rebuilds a closed client, so closing under a child still
    inside ``Bot.initialize`` leaves a pool no-one owns. The owner is kept out of ``_background_tasks``
    (base teardown cancels those) and the app is detached, so ``disconnect()`` waits on the owner
    instead of racing it."""
    owners = getattr(adapter, "_abandoned_initialize_owners", None)
    if owners is None:
        owners = adapter._abandoned_initialize_owners = set()
    owner = asyncio.get_running_loop().create_task(_close_after_initialize_exits(init_task, app))
    owners.add(owner)
    owner.add_done_callback(owners.discard)
    if adapter._app is app:
        adapter._app = adapter._bot = None


async def _close_after_initialize_exits(init_task: asyncio.Future, app) -> None:
    await asyncio.wait({init_task}, timeout=_ABANDONED_INIT_UNWIND_TIMEOUT)
    from plugins.platforms.telegram.adapter import _shutdown_abandoned_app

    await _shutdown_abandoned_app(app)


async def await_abandoned_initialize_owners(adapter, timeout: float) -> None:
    """Give retained initialize owners ``timeout`` to finish; never cancel them (they are the only
    thing that closes those transports)."""
    owners = [task for task in getattr(adapter, "_abandoned_initialize_owners", ()) if not task.done()]
    if owners:
        await asyncio.wait(owners, timeout=timeout)
