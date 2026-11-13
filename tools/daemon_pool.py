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
        """Keep each task in its caller's profile scope, even on a reused worker.

        Thread-start context cannot track later submissions from other profiles
        (#54937). The stdlib worker context manages initialization, not contextvars.

        Mirrors ``ThreadPoolExecutor.submit`` so a caller can tell a safe failure from an
        ambiguous one. Anything that raises before the item reaches ``_work_queue`` is tagged
        ``exc.never_enqueued = True``: that item can never run. A failure in
        ``_adjust_thread_count()`` comes after the put, when an idle worker may already be
        running the item, so it stays untagged and the caller must look for its item in
        the queue (the wrapper's ``__wrapped__`` names the callable it was given).
        """
        try:
            ctx = copy_context()

            def _run_with_context(*call_args, **call_kwargs):
                return ctx.run(fn, *call_args, **call_kwargs)
            _run_with_context.__wrapped__ = fn
        except Exception as exc:  # tag and re-raise
            exc.never_enqueued = True
            raise
        with self._shutdown_lock, _cf_thread._global_shutdown_lock:
            try:
                if self._broken:
                    raise getattr(self, "BROKEN", _cf_thread.BrokenThreadPool)(self._broken)
                if self._shutdown:
                    raise RuntimeError('cannot schedule new futures after shutdown')
                if _cf_thread._shutdown:
                    raise RuntimeError('cannot schedule new futures after interpreter shutdown')
                future = _cf_thread._base.Future()
                if hasattr(self, "_resolve_work_item_task"):
                    # Python 3.14: a work item holds one resolved task, not fn/args/kwargs.
                    work_item = _cf_thread._WorkItem(
                        future, self._resolve_work_item_task(_run_with_context, args, kwargs))
                else:
                    work_item = _cf_thread._WorkItem(future, _run_with_context, args, kwargs)
                self._work_queue.put(work_item)
            except Exception as exc:  # tag and re-raise
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
            thread_name = f'{self._thread_name_prefix or self}_{num_threads:d}'
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
            t = threading.Thread(
                name=thread_name, target=_worker, daemon=True,
                args=worker_args,
            )
            t.start()
            self._threads.add(t)
