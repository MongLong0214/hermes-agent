"""Shared daemon-thread ThreadPoolExecutor.

Stdlib workers are non-daemon AND registered in ``_threads_queues``, whose atexit
hook joins every worker even after ``shutdown(wait=False)`` — one wedged worker
(tool blocked on network I/O, hung provider, stuck subagent) blocks interpreter
exit forever. This variant spawns daemon workers and skips that registration.
Use it for best-effort/interruptible work that must never hold the process open;
NOT for work that must complete before exit (durable writes belong on foreground
threads with explicit bounded joins).
"""

from __future__ import annotations

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import thread as _cf_thread
from concurrent.futures.thread import _worker
from contextvars import copy_context

__all__ = ["DaemonThreadPoolExecutor"]


class DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor variant whose workers do not block process exit."""

    def submit(self, fn, /, *args, **kwargs):
        """Submit a callable, propagating the caller's contextvars. Stdlib only does
        this from 3.14; on 3.11-3.13 a bare worker starts with an EMPTY Context and
        drops profile secret scope / HERMES_HOME override — under the multiplexed
        gateway a credential read then fails closed with ``UnscopedSecretError``.
        Unconditional: on 3.14+ ``ctx.run`` re-applies the same context (no-op)."""
        ctx = copy_context()

        def _run_with_context(*call_args, **call_kwargs):
            return ctx.run(fn, *call_args, **call_kwargs)
        # A caller that loses its Future to a submit() failure (the stdlib enqueues before it
        # tries to start a worker thread) must still be able to find its own item sitting in
        # ``_work_queue`` by identity. ``fn`` is what it actually submitted, so expose it the way
        # ``functools.wraps`` would -- ``inspect.unwrap`` then sees through this layer too.
        _run_with_context.__wrapped__ = fn
        # Reimplemented (not delegated to ``super().submit``) so a caller whose submit() raises
        # can tell a PROVEN-safe failure from an ambiguous one. Anything raised inside this
        # ``try`` happened before the work item ever reached ``_work_queue`` (construction, or
        # the ``put`` itself) -- the item will never run, full stop -- and is tagged
        # ``exc.never_enqueued = True``. A failure from ``_adjust_thread_count()`` below happens
        # AFTER a successful ``put``: the item is irreversibly queued and an existing idle
        # thread may dequeue and run it before the caller's except block even starts, so it is
        # left untagged (the caller must scan ``_work_queue`` by identity instead, see
        # ``async_delegation._discard_queued_work_item``).
        try:
            with self._shutdown_lock, _cf_thread._global_shutdown_lock:
                if self._broken:
                    raise _cf_thread.BrokenThreadPool(self._broken)
                if self._shutdown:
                    raise RuntimeError('cannot schedule new futures after shutdown')
                if _cf_thread._shutdown:
                    raise RuntimeError('cannot schedule new futures after interpreter shutdown')
                future = _cf_thread._base.Future()
                work_item = _cf_thread._WorkItem(future, _run_with_context, args, kwargs)
                self._work_queue.put(work_item)
        except Exception as exc:  # noqa: BLE001 — tag-and-reraise, not a handled error
            exc.never_enqueued = True
            raise
        self._adjust_thread_count()
        return future

    def _adjust_thread_count(self) -> None:
        # Mirrors CPython's implementation with two changes:
        # daemon=True and no _threads_queues registration.
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)
        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            executor_ref = weakref.ref(self, weakref_cb)
            if hasattr(self, "_create_worker_context"):
                # Python 3.14 replaced _initializer/_initargs with a factory
                # that supplies the worker's initializer context.
                worker_args = (
                    executor_ref,
                    self._create_worker_context(),
                    self._work_queue,
                )
            else:
                worker_args = (
                    executor_ref,
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                )
            # Carry the active profile into the review thread so MEMORY.md / skill review writes land in the
            # right profile (#54937).
            t = threading.Thread(
                name=thread_name, target=_worker, daemon=True,
                args=worker_args,
            )
            t.start()
            self._threads.add(t)
