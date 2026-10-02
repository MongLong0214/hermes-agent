"""Tests for tools.daemon_pool.DaemonThreadPoolExecutor.

The daemon pool exists so abandoned workers (interrupted/timed-out tool
batches, wedged memory-provider syncs) can never block interpreter exit:
stdlib ThreadPoolExecutor workers are non-daemon AND registered in
concurrent.futures.thread._threads_queues, whose atexit hook joins every
worker unconditionally — even after shutdown(wait=False).
"""

import subprocess
import sys
import threading
import time

from concurrent.futures.thread import _threads_queues

import tools.daemon_pool as daemon_pool
from tools.daemon_pool import DaemonThreadPoolExecutor


def test_workers_are_daemon_threads():
    pool = DaemonThreadPoolExecutor(max_workers=2)
    try:
        info = pool.submit(
            lambda: (threading.current_thread().daemon, threading.current_thread())
        ).result(timeout=10)
        is_daemon, worker = info
        assert is_daemon is True
        # Not registered with concurrent.futures' atexit join hook.
        assert worker not in _threads_queues
    finally:
        pool.shutdown(wait=True)


def test_idle_worker_reuse():
    pool = DaemonThreadPoolExecutor(max_workers=4)
    try:
        tid1 = pool.submit(threading.get_ident).result(timeout=10)
        time.sleep(0.05)  # let the worker park on the idle semaphore
        tid2 = pool.submit(threading.get_ident).result(timeout=10)
        assert tid1 == tid2
    finally:
        pool.shutdown(wait=True)


def test_wedged_worker_does_not_block_interpreter_exit():
    """A worker stuck in a long sleep must not hold the process open.

    With stdlib ThreadPoolExecutor this subprocess hangs until the sleep
    finishes (the atexit hook joins the worker); with the daemon pool it
    exits as soon as the main thread returns.
    """
    script = (
        "import sys; sys.path.insert(0, %r)\n"
        "from tools.daemon_pool import DaemonThreadPoolExecutor\n"
        "import time\n"
        "pool = DaemonThreadPoolExecutor(max_workers=1)\n"
        "pool.submit(time.sleep, 120)\n"
        "time.sleep(0.3)\n"
        "pool.shutdown(wait=False)\n"
        "print('main-done', flush=True)\n"
    ) % (str(_repo_root()),)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
    assert "main-done" in proc.stdout


def test_submit_propagates_caller_contextvars():
    """Pool workers inherit contextvars set in the submitting context.

    Stdlib ThreadPoolExecutor snapshots the caller's context with
    ``copy_context()``; some bundled CPython runtime builds strip that, so
    the daemon pool restores it explicitly.  Without the fix this returns
    the default because the worker runs in a bare context.
    """
    from contextvars import ContextVar

    var = ContextVar("daemon_pool_test_var", default="unset")

    pool = DaemonThreadPoolExecutor(max_workers=1)
    try:
        token = var.set("hello")
        try:
            seen = pool.submit(var.get).result(timeout=10)
        finally:
            var.reset(token)
        assert seen == "hello"
    finally:
        pool.shutdown(wait=True)


def _capture_worker_args(monkeypatch, pool):
    """Swap the stdlib worker for one that records its args and resolves one item.

    The stdlib ``_worker`` signature differs between interpreters, so the fake
    accepts anything and completes the work item's future directly — the test
    then runs on 3.11 and 3.14 alike and asserts only on the arg shape chosen.
    """
    seen = []

    def fake_worker(*args):
        seen.append(args)
        pool._work_queue.get().future.set_result("done")

    monkeypatch.setattr(daemon_pool, "_worker", fake_worker)
    return seen


def test_worker_gets_context_when_executor_builds_worker_contexts(monkeypatch):
    """3.14+ shape (#58596, #111813): the executor exposes ``_create_worker_context``
    and no ``_initializer``/``_initargs``; the worker must receive
    ``(executor_ref, ctx, work_queue)`` — reading the legacy fields raised
    ``AttributeError`` on every pool spawn."""
    pool = DaemonThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(pool, "_create_worker_context", lambda: "worker-context", raising=False)
    monkeypatch.delattr(pool, "_initializer", raising=False)
    monkeypatch.delattr(pool, "_initargs", raising=False)
    seen = _capture_worker_args(monkeypatch, pool)
    try:
        assert pool.submit(lambda: None).result(timeout=10) == "done"
    finally:
        pool.shutdown(wait=True)
    ((executor_ref, ctx, work_queue),) = seen
    assert executor_ref() is pool
    assert ctx == "worker-context"
    assert work_queue is pool._work_queue


def test_worker_gets_initializer_when_executor_stores_initializer_fields(monkeypatch):
    """3.11–3.13 shape: no ``_create_worker_context``; the worker must receive
    ``(executor_ref, work_queue, initializer, initargs)``."""

    def init(*_):
        return None

    pool = DaemonThreadPoolExecutor(max_workers=1)
    monkeypatch.delattr(pool, "_create_worker_context", raising=False)
    monkeypatch.setattr(pool, "_initializer", init, raising=False)
    monkeypatch.setattr(pool, "_initargs", (1, 2), raising=False)
    seen = _capture_worker_args(monkeypatch, pool)
    try:
        assert pool.submit(lambda: None).result(timeout=10) == "done"
    finally:
        pool.shutdown(wait=True)
    ((executor_ref, work_queue, initializer, initargs),) = seen
    assert executor_ref() is pool
    assert work_queue is pool._work_queue
    assert (initializer, initargs) == (init, (1, 2))


def _gate_worker_starts(monkeypatch, prefix, on_start):
    """Run *on_start(thread)* just before each real ``Thread.start`` of a worker named *prefix*_N (then
    start it for real). Only scheduling is controlled; the executor's own code runs unmodified."""
    real_start = threading.Thread.start

    def gated_start(self):
        if self.name.startswith(prefix + "_"):
            on_start(self)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", gated_start)


def test_concurrent_submissions_never_create_more_workers_than_max_workers(monkeypatch):
    """R-EXECUTOR-LOCKING: ``submit()`` must create and register its worker under the same locks as the
    enqueue, the way the stdlib does. Hold the first worker's start open: a second, concurrent submit()
    that is not serialized behind it sees zero registered threads and spawns a second worker even though
    ``max_workers=1``."""
    pool = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="serial-probe")
    starts = []
    first_start = threading.Event()
    second_start = threading.Event()

    def on_start(thread):
        starts.append(thread)
        if len(starts) == 1:
            first_start.set()
            second_start.wait(1.0)  # a serialized second submit() can never get here while we wait
        else:
            second_start.set()

    _gate_worker_starts(monkeypatch, "serial-probe", on_start)
    release = threading.Event()
    futures = []
    submitters = [threading.Thread(target=lambda: futures.append(pool.submit(release.wait, 10)))
                  for _ in range(2)]
    try:
        submitters[0].start()
        assert first_start.wait(5)
        submitters[1].start()
        for t in submitters:
            t.join(10)
        assert len(futures) == 2
        release.set()
        assert all(f.result(timeout=10) is True for f in futures)
        assert len(starts) == 1, f"max_workers=1 but {len(starts)} worker threads were started"
        assert len(pool._threads) == 1
    finally:
        release.set()
        pool.shutdown(wait=True)


def test_shutdown_wait_waits_for_an_admitted_submission(monkeypatch):
    """R-EXECUTOR-LOCKING: once submit() has admitted an item, ``shutdown(wait=True)`` must not return
    until that item's worker exists and has run it. With worker creation outside the shutdown lock,
    shutdown slipped in between the enqueue and the worker's registration, found no threads to join,
    and returned with the admitted item still unrun."""
    pool = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="shutdown-probe")
    in_start = threading.Event()
    gate = threading.Event()

    def on_start(_thread):
        in_start.set()
        assert gate.wait(10)

    _gate_worker_starts(monkeypatch, "shutdown-probe", on_start)
    ran = threading.Event()
    observed = {}

    def do_shutdown():
        pool.shutdown(wait=True)
        observed["ran_when_shutdown_returned"] = ran.is_set()

    submitter = threading.Thread(target=pool.submit, args=(ran.set,))
    stopper = threading.Thread(target=do_shutdown)
    try:
        submitter.start()
        assert in_start.wait(5), "the admitted submission never reached its worker start"
        stopper.start()
        stopper.join(0.5)  # a correctly serialized shutdown is still blocked behind the submission here
    finally:
        gate.set()
        submitter.join(10)
        stopper.join(10)
    assert not stopper.is_alive()
    assert observed == {"ran_when_shutdown_returned": True}, (
        "shutdown(wait=True) returned before the admitted submission's worker ran it")


def test_failure_before_enqueue_is_tagged_never_enqueued(monkeypatch):
    """Every step of submit() that can fail before the item reaches ``_work_queue`` -- including the
    pool's own context capture, the very first thing it does -- must tag the exception
    ``never_enqueued`` so a caller can release what it reserved for the item."""
    pool = DaemonThreadPoolExecutor(max_workers=1)

    def failing_copy_context():
        raise MemoryError("injected: context capture")

    monkeypatch.setattr(daemon_pool, "copy_context", failing_copy_context)
    try:
        try:
            pool.submit(lambda: None)
        except MemoryError as exc:
            assert getattr(exc, "never_enqueued", False) is True
        else:
            raise AssertionError("submit() did not raise")
        assert pool._work_queue.empty()
        assert not pool._threads
    finally:
        pool.shutdown(wait=True)


def test_worker_start_failure_after_enqueue_is_not_tagged_never_enqueued(monkeypatch):
    """The opposite case: a worker-start failure happens AFTER the item is queued (an idle or later
    worker can still run it), so it must stay untagged -- callers fall back to scanning the queue."""
    pool = DaemonThreadPoolExecutor(max_workers=1, thread_name_prefix="start-fail-probe")

    def on_start(_thread):
        raise RuntimeError("injected: can't start new thread")

    _gate_worker_starts(monkeypatch, "start-fail-probe", on_start)
    try:
        try:
            pool.submit(lambda: None)
        except RuntimeError as exc:
            assert not getattr(exc, "never_enqueued", False)
        else:
            raise AssertionError("submit() did not raise")
        assert pool._work_queue.qsize() == 1, "the item was queued before the worker start failed"
    finally:
        pool.shutdown(wait=False)


def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]
