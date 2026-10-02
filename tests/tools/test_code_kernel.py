#!/usr/bin/env python3
"""Tests for execute_code's session kernel mode.

``code_execution.kernel_mode: session`` keeps one Python child alive per
(task, mode, interpreter, cwd, tool-set) so state survives across calls.
These tests pin the contract:

  - default stays per-call (no state carries over unless opted in)
  - state persists across cells and reset=true discards it
  - a raised exception keeps the kernel (and its state) alive
  - a timeout kills the kernel; the next call gets a fresh one
  - fd-level output from user-spawned subprocesses reaches the result
  - sys.exit() inside a cell ends the kernel deliberately

Mode is sourced from ``code_execution.kernel_mode`` in config.yaml only;
tests patch ``_load_config`` directly, mirroring test_code_execution_modes.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

os.environ["TERMINAL_ENV"] = "local"


@pytest.fixture(autouse=True)
def _force_local_terminal(monkeypatch):
    """Mirror test_code_execution.py — guarantee local backend under xdist."""
    monkeypatch.setenv("TERMINAL_ENV", "local")


from tools.code_execution_tool import execute_code
from tools.code_kernel import _KERNELS, shutdown_all_kernels


@contextmanager
def _kernel_config(**overrides):
    """Pin code_execution config; strict mode keeps the test hermetic."""
    config = {"mode": "strict", "kernel_mode": "session", "timeout": 30}
    config.update(overrides)
    with patch("tools.code_execution_tool._load_config", return_value=config):
        yield


@pytest.fixture(autouse=True)
def _fresh_kernel_registry():
    shutdown_all_kernels()
    yield
    shutdown_all_kernels()


def _run(code, **kwargs):
    return json.loads(execute_code(code, task_id="kernel-test", **kwargs))


class TestSessionStatePersistence(unittest.TestCase):
    def test_state_persists_across_cells(self):
        with _kernel_config():
            first = _run("x = 41")
            self.assertEqual(first["status"], "success", first)
            self.assertEqual(first["kernel"]["reused"], False)
            second = _run("print(x + 1)")
        self.assertEqual(second["status"], "success", second)
        self.assertIn("42", second["output"])
        self.assertEqual(second["kernel"]["reused"], True)
        self.assertEqual(second["kernel"]["execution_count"], 2)

    def test_reset_discards_state(self):
        with _kernel_config():
            _run("x = 41")
            second = _run("print(x + 1)", reset=True)
        self.assertEqual(second["status"], "error", second)
        self.assertIn("NameError", second.get("error", ""))
        self.assertEqual(second["kernel"]["state_reset"], True)

    def test_exception_keeps_the_kernel_alive(self):
        with _kernel_config():
            _run("a = 7")
            boom = _run("1 / 0")
            self.assertEqual(boom["status"], "error")
            self.assertIn("ZeroDivisionError", boom["error"])
            after = _run("print(a)")
        self.assertEqual(after["status"], "success", after)
        self.assertIn("7", after["output"])
        self.assertEqual(after["kernel"]["reused"], True)

    def test_imports_persist(self):
        with _kernel_config():
            _run("import json as _j")
            second = _run("print(_j.dumps({'k': 1}))")
        self.assertIn('{"k": 1}', second["output"])


class TestKernelLifecycle(unittest.TestCase):
    def test_kernel_exits_when_its_backend_parent_dies(self):
        """A kernel must not outlive the host that spawned it, even when the
        host dies without cleanup (SIGKILL/OOM/crash). Windows: inherited
        SYNCHRONIZE handle; POSIX: inherited death pipe. Both are proven the
        same way — kill the host mid-cell, the kernel is gone within seconds."""
        import psutil

        repo_root = str(Path(__file__).resolve().parents[2])
        host_src = textwrap.dedent(f"""
            import json, os, sys, time
            os.environ["HERMES_HOME"] = sys.argv[1]
            sys.path.insert(0, {repo_root!r})
            from tools.code_kernel import SessionKernel, _spawn
            k = SessionKernel(("parent-death",))
            _spawn(k, task_id="parent-death", child_python=sys.executable,
                   child_cwd="", sandbox_tools=frozenset(), max_tool_calls=1)
            cell = json.dumps({{"id": "x", "code": "import os, time\\n"
                "assert 'HERMES_KERNEL_PARENT_PROCESS_HANDLE' not in os.environ\\n"
                "assert 'HERMES_KERNEL_PARENT_DEATH_FD' not in os.environ\\n"
                "time.sleep(300)"}}) + "\\n"
            k.proc.stdin.write(cell.encode()); k.proc.stdin.flush()
            print(k.proc.pid, flush=True)
            time.sleep(600)
        """)
        with tempfile.TemporaryDirectory() as home:
            host = subprocess.Popen(
                [sys.executable, "-c", host_src, home],
                stdout=subprocess.PIPE, text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                kernel = psutil.Process(int(host.stdout.readline()))
                time.sleep(0.5)
                self.assertTrue(kernel.is_running(), "kernel never came up")
                host.kill()
                host.wait(timeout=10)
                try:
                    kernel.wait(timeout=10)
                except psutil.TimeoutExpired:
                    kernel.kill()
                    self.fail("session kernel survived its backend parent")
            finally:
                if host.poll() is None:
                    host.kill()

    def test_timeout_kills_the_kernel_and_reports_state_loss(self):
        with _kernel_config(timeout=1):
            slow = _run("import time\ntime.sleep(30)")
            self.assertEqual(slow["status"], "timeout", slow)
            self.assertIn("state was lost", slow["error"])
        self.assertEqual(len(_KERNELS), 0)
        with _kernel_config():
            fresh = _run("print('alive')")
        self.assertEqual(fresh["status"], "success", fresh)
        self.assertEqual(fresh["kernel"]["reused"], False)
        self.assertIn("alive", fresh["output"])

    def test_sys_exit_ends_the_kernel(self):
        with _kernel_config():
            done = _run("import sys\nsys.exit(0)")
            self.assertEqual(done["kernel"].get("ended"), True, done)
            self.assertEqual(len(_KERNELS), 0)
            fresh = _run("print('respawned')")
        self.assertEqual(fresh["kernel"]["reused"], False)
        self.assertIn("respawned", fresh["output"])

    def test_subprocess_fd_output_reaches_the_result(self):
        code = (
            "import subprocess, sys\n"
            "subprocess.run([sys.executable, '-c', \"print('raw-passthrough')\"])\n"
        )
        with _kernel_config():
            result = _run(code)
        self.assertEqual(result["status"], "success", result)
        self.assertIn("raw-passthrough", result["output"])


class TestModelFacingReset(unittest.TestCase):
    def test_reset_is_reachable_from_a_model_call_despite_stale_kernel_mode(self):
        """Session kernels are always on (#96787), so ``reset`` is the model's only
        way out of poisoned state. A stale ``kernel_mode: per-call`` key must not
        drop it from the schema, and a model-shaped call routed through the
        registered handler must actually discard the kernel's state."""
        from tools.code_execution_tool import _execute_code_handler, build_execute_code_schema

        with _kernel_config(kernel_mode="per-call"):
            schema = build_execute_code_schema(mode="strict")
            self.assertEqual(schema["parameters"]["properties"]["reset"]["type"], "boolean")
            _execute_code_handler({"code": "x = 41"}, task_id="kernel-test")
            kept = json.loads(_execute_code_handler({"code": "print(x + 1)"}, task_id="kernel-test"))
            self.assertIn("42", kept["output"], kept)
            reset = json.loads(_execute_code_handler(
                {"code": "print(x + 1)", "reset": True}, task_id="kernel-test"))
        self.assertEqual(reset["status"], "error", reset)
        self.assertIn("NameError", reset.get("error", ""))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


class TestKernelOwnershipAndLifecycle(unittest.TestCase):
    """The kernel belongs to the conversation, and its lifetime is bounded.

    run_agent mints a fresh task id per top-level turn, so a task-keyed
    kernel would neither survive the next user turn nor ever be disposed
    with anything. The owner is the approval session key; disposal rides
    the same session boundary that clears approval/yolo state, idle
    kernels are reaped, and the process-wide live count is capped (the
    lifecycle shape carried forward from hermes-agent#88637).
    """

    def _run_as(self, session_key, code, task_id, **kwargs):
        from tools.approval_context import reset_current_session_key, set_current_session_key

        token = set_current_session_key(session_key)
        try:
            return json.loads(execute_code(code, task_id=task_id, **kwargs))
        finally:
            reset_current_session_key(token)

    def test_state_survives_across_turns_of_one_conversation(self):
        # Two top-level turns: same session, different per-turn task ids.
        with _kernel_config():
            first = self._run_as("conv-a", "x = 41", task_id="turn-1")
            self.assertEqual(first["status"], "success", first)
            second = self._run_as("conv-a", "print(x + 1)", task_id="turn-2")
        self.assertEqual(second["status"], "success", second)
        self.assertIn("42", second["output"])
        self.assertEqual(second["kernel"]["reused"], True)

    def test_sessions_are_isolated_from_each_other(self):
        # Same task id, different sessions: no state may cross.
        with _kernel_config():
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            other = self._run_as("conv-b", "print(x + 1)", task_id="turn-1")
        self.assertEqual(other["status"], "error", other)
        self.assertIn("NameError", other.get("error", ""))

    def test_delegated_children_get_their_own_kernels(self):
        """A delegated child runs in a COPY of the parent's context and
        inherits the parent's approval session key — the naive owner
        resolution attached the child to the parent's kernel and leaked
        in-memory state across the delegation boundary (both directions,
        verified live). The owner must be qualified for child contexts."""
        from agent.delegation_context import delegated_child_context

        with _kernel_config():
            self._run_as("conv-a", "parent_secret = 'p'", task_id="turn-1")
            with delegated_child_context("child-1"):
                leak = self._run_as(
                    "conv-a",
                    "print(globals().get('parent_secret', 'ISOLATED'))",
                    task_id="child-task",
                )
                self._run_as("conv-a", "child_secret = 'c'", task_id="child-task")
            back = self._run_as(
                "conv-a",
                "print(globals().get('child_secret', 'ISOLATED'))",
                task_id="turn-2",
            )
        self.assertIn("ISOLATED", leak.get("output", ""), leak)
        self.assertIn("ISOLATED", back.get("output", ""), back)

    def test_two_delegated_children_are_isolated_from_each_other(self):
        """Sibling children in one batch must not share a kernel either —
        each child context carries its own delegation session id."""
        from agent.delegation_context import delegated_child_context

        with _kernel_config():
            with delegated_child_context("child-A"):
                self._run_as("conv-a", "sibling_secret = 'A'", task_id="t")
            with delegated_child_context("child-B"):
                peek = self._run_as(
                    "conv-a",
                    "print(globals().get('sibling_secret', 'ISOLATED'))",
                    task_id="t",
                )
        self.assertIn("ISOLATED", peek.get("output", ""), peek)

    def test_live_children_keep_their_kernels_past_the_lru_cap(self):
        """A fan-out wider than max_session_kernels used to evict LIVE children's kernels (each
        child's execute_code spawned a kernel, the cap reaped the oldest sibling's), so a child's
        second call hit NameError on state its first call had set — 48 NameErrors across 28 lanes,
        while the schema promised persistence. A live child's kernel is pinned for the child's life."""
        import contextvars

        from agent.delegation_context import delegated_child_context

        with _kernel_config(max_session_kernels=2):
            contexts = []
            for index in range(5):
                def _set(index=index):
                    with delegated_child_context(f"child-{index}"):
                        self._run_as("conv", f"v = {index}", task_id=f"child-{index}")
                ctx = contextvars.copy_context()
                ctx.run(_set)
                contexts.append(ctx)
            outcomes = {}
            for index, ctx in enumerate(contexts):
                def _read(index=index):
                    with delegated_child_context(f"child-{index}"):
                        outcomes[index] = self._run_as("conv", "print(v)", task_id=f"child-{index}")
                ctx.run(_read)
        for index, outcome in outcomes.items():
            self.assertEqual(outcome["status"], "success", outcome)
            self.assertTrue(outcome["kernel"]["reused"], outcome)
            self.assertIn(str(index), outcome["output"])

    def test_finished_children_release_their_kernels(self):
        """The pin is not a leak: when the child is torn down (the delegate_task cleanup path calls
        ``shutdown_kernels_for_delegated_child``) its kernels die and stop counting."""
        from agent.delegation_context import delegated_child_context
        from tools.code_kernel import shutdown_kernels_for_delegated_child

        with _kernel_config():
            with delegated_child_context("child-done"):
                self._run_as("conv", "v = 1", task_id="child-done")
            with delegated_child_context("child-live"):
                self._run_as("conv", "v = 2", task_id="child-live")
            doomed = [k for k in _KERNELS.values() if k.owner.endswith("::child::child-done")]
            self.assertEqual(len(doomed), 1)
            shutdown_kernels_for_delegated_child("child-done")
            self.assertEqual([k for k in _KERNELS.values() if k.owner.endswith("::child::child-done")], [])
            doomed[0].proc.wait(timeout=10)
            self.assertFalse(doomed[0].alive())
            # The sibling's kernel is untouched.
            with delegated_child_context("child-live"):
                still = self._run_as("conv", "print(v)", task_id="child-live")
        self.assertIn("2", still["output"])

    def test_session_clear_disposes_the_owners_kernels(self):
        from tools.approval import clear_session

        with _kernel_config():
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            self.assertEqual(len(_KERNELS), 1)
            kernel = next(iter(_KERNELS.values()))
            self.assertTrue(kernel.alive())
            clear_session("conv-a")
            self.assertEqual(len(_KERNELS), 0)
            kernel.proc.wait(timeout=10)
            self.assertFalse(kernel.alive())
            # The next turn in a cleared session starts fresh.
            after = self._run_as("conv-a", "print('x' in dir())", task_id="turn-2")
        self.assertEqual(after["status"], "success", after)
        self.assertIn("False", after["output"])

    def test_live_kernels_are_capped_lru_across_owners(self):
        with _kernel_config(max_session_kernels=2):
            kernels = []
            for index in range(4):
                self._run_as(f"conv-{index}", "x = 1", task_id=f"turn-{index}")
                kernels.append(list(_KERNELS.values()))
            self.assertLessEqual(len(_KERNELS), 2)
            live_owners = {key[0] for key in _KERNELS}
            # The two most recently used owners survive.
            self.assertEqual(live_owners, {"conv-2", "conv-3"})
        # Evicted kernels are actually dead, not orphaned.
        evicted = [
            kernel
            for snapshot in kernels
            for kernel in snapshot
            if kernel.key not in _KERNELS
        ]
        for kernel in evicted:
            kernel.proc.wait(timeout=10)
            self.assertFalse(kernel.alive())

    def test_idle_kernels_are_reaped(self):
        import time as time_module

        with _kernel_config(kernel_idle_timeout=1):
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            stale = next(iter(_KERNELS.values()))
            time_module.sleep(1.2)
            # Any owner's next call sweeps expired kernels process-wide.
            self._run_as("conv-b", "y = 1", task_id="turn-2")
            self.assertNotIn(stale.key, _KERNELS)
            stale.proc.wait(timeout=10)
            self.assertFalse(stale.alive())

    def test_parallel_cells_share_one_kernel_process(self):
        """Parallel cells for one owner race the first spawn. Each racer
        used to see proc=None as 'dead', replace the registry entry, and
        orphan the winner's process — 110 live kernels under a 4-capped
        process (Sep 2026). Every kernel process must stay registry-owned."""
        import subprocess
        import threading

        results = []
        with _kernel_config():
            def _cell():
                results.append(self._run_as("conv-a", "import time; time.sleep(0.3)", task_id="t"))
            threads = [threading.Thread(target=_cell) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual([r["status"] for r in results], ["success"] * 6)
        self.assertEqual(len(_KERNELS), 1)
        live = subprocess.run(
            ["pgrep", "-fc", "-P", str(os.getpid()), "hermes_kernel_runner"],
            capture_output=True, text=True,
        ).stdout.strip()
        self.assertEqual(live, "1")


class TestPerCellRpcAuthority(unittest.TestCase):
    """Interpreter state persists across cells; RPC authority must not."""

    def _recorder(self, seen):
        def _handle(tool_name, tool_args, task_id=None):
            from tools.thread_context import _callback_api

            (get_approval, _set_a), *_rest = _callback_api()
            seen.append(
                {
                    "tool": tool_name,
                    "task_id": task_id,
                    "approval_cb": get_approval(),
                }
            )
            return json.dumps({"ok": True})

        return _handle

    def test_a_later_cells_rpc_runs_under_that_cells_authority(self):
        from tools.terminal_tool import set_approval_callback

        seen = []
        cell = "import hermes_tools\nhermes_tools.web_search(query='q')\n"
        with _kernel_config(), patch(
            "model_tools.handle_function_call", new=self._recorder(seen)
        ):
            def cb_one():
                return "one"

            def cb_two():
                return "two"

            set_approval_callback(cb_one)
            try:
                first = _run(cell)
                set_approval_callback(cb_two)
                second = _run(cell)
            finally:
                set_approval_callback(None)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(len(seen), 2)
        self.assertIs(seen[0]["approval_cb"], cb_one)
        self.assertIs(seen[1]["approval_cb"], cb_two)
        self.assertEqual(seen[0]["task_id"], "kernel-test")

    def test_cross_cell_alias_dispatches_under_the_current_cell(self):
        # Adversarial cross-cell dataflow: a callable captured in cell 1 and
        # invoked by an opaque global name in cell 2 still crosses the RPC
        # boundary — under cell 2's authority, allow-list, and budget — the
        # operative enforcement a per-script static scan cannot provide once
        # state persists (composition contract with the execute-code guard).
        from tools.terminal_tool import set_approval_callback

        seen = []
        with _kernel_config(), patch(
            "model_tools.handle_function_call", new=self._recorder(seen)
        ):
            def cb_one():
                return "one"

            def cb_two():
                return "two"

            set_approval_callback(cb_one)
            try:
                first = _run("import hermes_tools\nalias = hermes_tools.web_search\n")
                set_approval_callback(cb_two)
                second = _run("alias(query='q')\n")
            finally:
                set_approval_callback(None)
        self.assertEqual(first["status"], "success", first)
        self.assertEqual(second["status"], "success", second)
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0]["approval_cb"], cb_two)

    def test_a_settled_cells_authority_refuses_dispatch(self):
        from tools.code_kernel import CellAuthority

        authority = CellAuthority("turn-1")
        authority.retire()
        result = authority.dispatch("web_search", {"query": "q"})
        self.assertIn("No active execute_code cell", result)

    def test_each_cell_installs_a_fresh_authority(self):
        with _kernel_config():
            _run("x = 1")
            kernel = next(iter(_KERNELS.values()))
            first_authority = kernel.cell_authority
            self.assertFalse(first_authority.active)
            _run("y = 2")
            self.assertIsNot(kernel.cell_authority, first_authority)
            self.assertFalse(kernel.cell_authority.active)


class TestBackgroundIdleReaper(unittest.TestCase):
    """#117169: the idle sweep must not depend on the next kernel acquire — a host
    that stays alive but wedged (e.g. pids exhaustion fail-closing every tool call)
    never acquires again, so a background reaper reapplies the acquire-path criteria
    on its own schedule, and staging dirs that outlived a dead host are swept by age."""

    def _run_as(self, session_key, code, task_id, **kwargs):
        from tools.approval_context import reset_current_session_key, set_current_session_key

        token = set_current_session_key(session_key)
        try:
            return json.loads(execute_code(code, task_id=task_id, **kwargs))
        finally:
            reset_current_session_key(token)

    def test_reap_once_sweeps_idle_kernels_without_a_new_acquire(self):
        import time as time_module

        from tools.code_kernel import _reap_once

        with _kernel_config(kernel_idle_timeout=1):
            self._run_as("conv-a", "x = 41", task_id="turn-1")
            stale = next(iter(_KERNELS.values()))
            time_module.sleep(1.2)
            # No conv-b acquire here: the reaper pass alone must retire the kernel.
            _reap_once()
            self.assertNotIn(stale.key, _KERNELS)
            stale.proc.wait(timeout=10)
            self.assertFalse(stale.alive())

    def test_reap_once_spares_attached_and_fresh_kernels(self):
        from tools.code_kernel import _reap_once

        with _kernel_config(kernel_idle_timeout=1):
            fresh = self._run_as("conv-fresh", "x = 1", task_id="turn-1")
            self.assertEqual(fresh["status"], "success", fresh)
            kernel = next(iter(_KERNELS.values()))
            kernel.attached += 1  # a cell is mid-flight: reaping must skip it
            try:
                _reap_once()
                self.assertIn(kernel.key, _KERNELS)
                self.assertTrue(kernel.alive())
            finally:
                kernel.attached -= 1

class TestStaleStagingDirSweep(unittest.TestCase):
    def test_week_old_kernel_dirs_go_and_fresh_ones_stay(self):
        import time as time_module

        from tools.code_kernel import _sweep_stale_staging_dirs

        with tempfile.TemporaryDirectory() as tmp:
            with patch("tools.code_kernel.tempfile.gettempdir", return_value=tmp):
                old = Path(tmp, "hermes_kernel_old")
                young = Path(tmp, "hermes_kernel_young")
                bystander = Path(tmp, "unrelated_dir")
                for path in (old, young, bystander):
                    path.mkdir()
                week_and_a_bit = time_module.time() - 8 * 86400
                os.utime(old, (week_and_a_bit, week_and_a_bit))
                removed = _sweep_stale_staging_dirs()
                # Asserted inside the TemporaryDirectory: cleanup would flatten everything.
                self.assertEqual(removed, 1)
                self.assertFalse(old.exists())
                self.assertTrue(young.exists())
                self.assertTrue(bystander.exists())


class TestLifecycleGuardAuditHook(unittest.TestCase):
    """L7-2: execute_code's pre-dispatch guard only scans the cell's SOURCE TEXT for a gateway
    lifecycle command, so a cell that builds the argv at runtime (``a = "launch"; b = "ctl"``)
    never looks like a literal string to that scan and — with ``approvals.mode: off`` — the cell
    then runs unchecked. The kernel's bootstrap installs a ``sys.addaudithook`` that re-applies
    ``cron.lifecycle_guard.contains_gateway_lifecycle_command`` to the REAL argv right before
    CPython would create the process, so terminal and Python enforce the same rule.

    R70-2 scoped the hook to the supervised gateway only (matching the SOURCE-TEXT scan's existing
    condition), so every test here that exercises blocking behavior needs that precondition set up
    explicitly -- this class is not itself the supervised gateway process."""

    def setUp(self):
        from tools import process_registry
        patcher = patch.object(process_registry, "_is_supervised_gateway_process", lambda: True)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _stub_launchctl(bin_dir: Path, record_path: Path) -> None:
        bin_dir.mkdir(parents=True, exist_ok=True)
        stub = bin_dir / "launchctl"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "with open(" + repr(str(record_path)) + ", 'a') as f:\n"
            "    f.write(json.dumps(sys.argv) + chr(10))\n"
        )
        stub.chmod(0o755)

    def _run_with_stub_launchctl(self, code: str):
        # R70-3: a plain ``with tempfile.TemporaryDirectory() as tmp: ... return result, record_path``
        # unwinds the ``with`` block (deleting the directory) as PART of executing that ``return`` --
        # before the caller ever sees the value -- so every ``record_path.exists()`` check below was
        # checking an already-removed path regardless of whether the stub had run. `addCleanup` defers
        # removal until after the whole test method (including its assertions) has run.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        tmp_path = Path(tmp)
        bin_dir = tmp_path / "bin"
        record_path = tmp_path / "launchctl_argv.jsonl"
        self._stub_launchctl(bin_dir, record_path)
        patched_path = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        with _kernel_config(), patch.dict(os.environ, {"PATH": patched_path}):
            result = _run(code)
        return result, record_path

    def test_dynamically_built_lifecycle_argv_never_reaches_launchctl(self):
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import subprocess
            a = "launch"
            b = "ctl"
            subprocess.run([a + b, "bootout", "gui/501/ai.hermes.gateway"], check=False)
            """))
        self.assertEqual(result["status"], "error", result)
        self.assertIn("Blocked: cannot restart or stop the gateway", result["error"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    def test_literal_lifecycle_argv_still_blocked(self):
        result, record_path = self._run_with_stub_launchctl(
            'import subprocess\n'
            'subprocess.run(["launchctl", "bootout", "gui/501/ai.hermes.gateway"], check=False)\n'
        )
        # A literal argv like this one is ALSO recognized by the pre-existing SOURCE-TEXT scan
        # (tools/code_execution_tool.py), which returns a bare ``{"error": ...}`` with no "status"
        # key -- a dict shared with the kernel's own error shape only through "blocked" semantics,
        # not key-for-key. Either layer catching it is correct; this test asserts the command is
        # still blocked, not which layer caught it.
        self.assertEqual(result.get("status", "error"), "error", result)
        self.assertIn("Blocked: cannot restart or stop the gateway", result["error"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    def test_ordinary_subprocess_use_still_works(self):
        result, record_path = self._run_with_stub_launchctl(
            'import subprocess\n'
            'r = subprocess.run(["echo", "ok"], capture_output=True, text=True)\n'
            'print(r.stdout.strip())\n'
        )
        self.assertEqual(result["status"], "success", result)
        self.assertIn("ok", result["output"])
        self.assertFalse(record_path.exists())

    def test_record_path_reflects_a_genuine_invocation(self):
        """R70-3: ``_run_with_stub_launchctl`` returned ``record_path`` from inside the
        ``TemporaryDirectory`` ``with`` block AFTER that block had already exited (``return``
        unwinds the ``with`` before the caller gets the value), so the directory -- and
        ``record_path`` -- no longer existed by the time any caller checked it. Every
        ``assertFalse(record_path.exists(), ...)`` above held vacuously, regardless of whether the
        stub had actually run. This command carries no hermes-gateway label, so the guard allows
        it and the stub genuinely runs; ``record_path`` must still answer ``True`` for it."""
        result, record_path = self._run_with_stub_launchctl(
            'import subprocess\n'
            'subprocess.run(["launchctl", "bootout", "unrelated-label"], check=False)\n'
        )
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(record_path.exists(),
                         "the stub launchctl should have run for an unrelated label")

    def test_os_system_bytes_argv_still_blocked(self):
        """R70-1(a): on this interpreter ``os.system()`` raises its ``os.system`` audit event with
        a BYTES argument even when called with a ``str`` (verified directly: ``os.system('echo
        hi')`` audits ``args[0]`` as ``b'echo hi'``). The hook passed that straight to the
        classifier, which raised ``TypeError`` on bytes input; the classifier's own fail-OPEN
        ``except Exception`` swallowed it, so ``os.system(<a lifecycle command>)`` reached the real
        ``launchctl`` unblocked. The command is built at runtime (as in
        ``test_dynamically_built_lifecycle_argv_never_reaches_launchctl``) so the pre-existing
        SOURCE-TEXT scan cannot catch it first -- this isolates the audit hook itself."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import os
            a = "launch"
            b = "ctl"
            os.system(a + b + " bootout gui/501/ai.hermes.gateway")
            """))
        self.assertEqual(result["status"], "error", result)
        self.assertIn("Blocked: cannot restart or stop the gateway", result["error"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    def test_quoted_sh_c_payload_still_blocked(self):
        """R70-1(b): the hook reconstructs a Popen argv into one text blob with ``shlex.join`` and
        classifies it with the single-pass ``contains_gateway_lifecycle_command``. A quoted
        executable/verb INSIDE the ``sh -c`` payload (``sh -c '"launchctl" "bootout" ...'``) leaves
        literal ``"`` characters sitting between ``launchctl`` and ``bootout`` even after that
        reconstruction, so the single-pass regex/tokenizer never resolves them into one word and
        the match fails -- while the terminal tool's guard (``scan_gateway_lifecycle``, which
        re-parses an ``sh -c`` payload as its own fresh command line and so strips exactly those
        quotes) recognizes the identical command. Confirmed directly against
        ``cron.lifecycle_guard``: ``contains_gateway_lifecycle_command`` returns False on this
        reconstructed text, ``scan_gateway_lifecycle`` returns True."""
        result, record_path = self._run_with_stub_launchctl(
            'import subprocess\n'
            'subprocess.run(["sh", "-c", \'"launchctl" "bootout" gui/501/ai.hermes.gateway\'], check=False)\n'
        )
        self.assertEqual(result["status"], "error", result)
        self.assertIn("Blocked: cannot restart or stop the gateway", result["error"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    def test_global_classifier_override_does_not_disable_the_guard(self):
        """R70-1(c): the hook's classifier was looked up off a module-level global
        (``_HERMES_LIFECYCLE_CHECK``) in the runner's own ``__main__`` namespace, and a cell shares
        that exact namespace with the runner's bootstrap -- ``exec(cell_code, GLOBALS)`` runs in a
        separate dict, but ``import __main__`` resolves to the REAL running module
        (``sys.modules["__main__"]``), not that dict. A cell that reassigns
        ``__main__._HERMES_LIFECYCLE_CHECK = lambda _: False`` therefore disabled the check for
        every later cell in the kernel's life. This cell disables it and THEN issues the blocked
        command (built at runtime so the pre-existing SOURCE-TEXT scan cannot catch it first,
        isolating the audit hook) in the same cell; the guard must still fire."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import __main__
            import subprocess
            try:
                __main__._HERMES_LIFECYCLE_CHECK = lambda _: False
            except Exception:
                pass
            a = "launch"
            b = "ctl"
            subprocess.run([a + b, "bootout", "gui/501/ai.hermes.gateway"], check=False)
            """))
        self.assertEqual(result["status"], "error", result)
        self.assertIn("Blocked: cannot restart or stop the gateway", result["error"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    def test_patched_shlex_join_does_not_disable_the_guard(self):
        """R70-1 round 2 (a): the hook flattened a list argv with ``shlex.join(parts)`` -- a name
        resolved through the runner's ``__main__`` globals and then the shared ``shlex`` module on
        EVERY call. ``import __main__; __main__.shlex.join = lambda _: "echo safe"`` turns every
        later argv into the text ``echo safe`` before the classifier sees it, so the dynamically
        built lifecycle argv below reached the stub."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import shlex  # the very module object ``__main__.shlex`` named in the runner
            import subprocess
            shlex.join = lambda _: "echo safe"
            a = "launch"
            b = "ctl"
            subprocess.run([a + b, "bootout", "gui/501/ai.hermes.gateway"], check=False)
            """))
        self.assertEqual(result["status"], "error", result)
        self.assertIn("Blocked:", result["error"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    def test_patched_classifier_dependency_does_not_disable_the_guard(self):
        """R70-1 round 2 (a), sibling site: ``scan_gateway_lifecycle`` resolves its own helpers
        through shared modules at call time -- e.g. ``from tools.shell_heredoc import
        strip_inert_heredoc_bodies`` runs inside the scan, so a cell that replaces that function
        first makes the classifier read an empty command and answer "safe"."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import subprocess
            import tools.shell_heredoc
            tools.shell_heredoc.strip_inert_heredoc_bodies = lambda text, *a, **k: ""
            a = "launch"
            b = "ctl"
            subprocess.run([a + b, "bootout", "gui/501/ai.hermes.gateway"], check=False)
            """))
        self.assertEqual(result["status"], "error", result)
        self.assertIn("Blocked:", result["error"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    # A child interpreter that builds the lifecycle argv itself: nothing in the PARENT's argv
    # (``python -c <this text>``) names the command, so only a hook inside the child can stop it.
    _GRANDCHILD = (
        "import subprocess; a = 'launch'; b = 'ctl'; "
        "subprocess.run([a + b, 'bootout', 'gui/501/ai.hermes.gateway'], check=False)"
    )

    def _assert_grandchild_blocked(self, result, record_path):
        self.assertEqual(result["status"], "success", result)
        self.assertIn("Blocked: cannot restart or stop the gateway", result["output"])
        self.assertFalse(record_path.exists(),
                          "the stub launchctl must never have been invoked")

    def test_python_c_grandchild_is_guarded(self):
        """R70-1 round 2 (b): ``sys.addaudithook`` is per-interpreter, so a ``python -c`` child the
        cell starts ran with no Hermes hook at all and reached the stub."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import subprocess, sys
            r = subprocess.run([sys.executable, "-c", %r], capture_output=True, text=True)
            print(r.returncode, r.stderr)
            """ % self._GRANDCHILD))
        self._assert_grandchild_blocked(result, record_path)

    def test_python_c_grandchild_with_scrubbed_env_is_guarded(self):
        """(b) sibling: the child is armed through its environment, so a cell passing an explicit
        env without the Hermes variables must not be enough to start an unguarded child."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import os, subprocess, sys
            env = {k: v for k, v in os.environ.items()
                   if k != "PYTHONPATH" and not k.startswith("HERMES_KERNEL_")}
            r = subprocess.run([sys.executable, "-c", %r], env=env, capture_output=True, text=True)
            print(r.returncode, r.stderr)
            """ % self._GRANDCHILD))
        self._assert_grandchild_blocked(result, record_path)

    def test_python_c_grandchild_after_bootstrap_overwrite_is_guarded(self):
        """(b) sibling: the child-side bootstrap is a file in the kernel's own (cell-writable)
        staging dir; emptying it must not leave the next child unguarded."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import os, subprocess, sys
            site = os.environ.get("HERMES_KERNEL_LIFECYCLE_GUARD_SITE", "")
            if site:
                with open(os.path.join(site, "sitecustomize.py"), "w") as f:
                    f.write("")
            r = subprocess.run([sys.executable, "-c", %r], capture_output=True, text=True)
            print(r.returncode, r.stderr)
            """ % self._GRANDCHILD))
        self._assert_grandchild_blocked(result, record_path)

    def test_posix_spawn_with_an_explicit_env_runs_without_lifecycle_content(self):
        """R70-4: os.posix_spawn/os.exec* convert the env before auditing it, so this hook cannot
        arm it in place; that must not block an ordinary (non-lifecycle) call -- only a genuine
        lifecycle command is ever refused."""
        result, _record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import os
            pid = os.posix_spawn("/bin/echo", ["echo", "posix-spawn-ok"], {"PATH": "/usr/bin:/bin"})
            os.waitpid(pid, 0)
            print("spawned")
            """))
        self.assertEqual(result["status"], "success", result)
        self.assertIn("spawned", result["output"])

    def test_popen_with_an_immutable_env_runs_without_lifecycle_content(self):
        """R70-4 sibling: a read-only env mapping (MappingProxyType) cannot be armed in place
        either; it must not block an ordinary subprocess call."""
        result, _record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import subprocess
            from types import MappingProxyType
            r = subprocess.run(["/bin/echo", "ok"], env=MappingProxyType({"PATH": "/usr/bin:/bin"}),
                                capture_output=True, text=True)
            print(r.returncode, r.stdout.strip())
            """))
        self.assertEqual(result["status"], "success", result)
        self.assertIn("0 ok", result["output"])

    def test_ordinary_python_grandchild_and_its_own_sitecustomize_still_work(self):
        """No regression from arming children: an ordinary ``python -c`` child runs, and a
        ``sitecustomize`` the cell put on the child's own PYTHONPATH still runs too (the Hermes
        bootstrap sits first on the path and chains to it rather than shadowing it)."""
        result, record_path = self._run_with_stub_launchctl(textwrap.dedent("""\
            import os, subprocess, sys, tempfile
            user_site = tempfile.mkdtemp()
            with open(os.path.join(user_site, "sitecustomize.py"), "w") as f:
                f.write("import os; os.environ['USER_SITE_RAN'] = '1'")
            env = dict(os.environ, PYTHONPATH=user_site)
            r = subprocess.run(
                [sys.executable, "-c",
                 "import os, subprocess; print('child-ok', os.environ.get('USER_SITE_RAN')); "
                 "print(subprocess.run(['echo', 'gc-ok'], capture_output=True, text=True).stdout.strip())"],
                env=env, capture_output=True, text=True)
            print(r.returncode, r.stdout, r.stderr)
            """))
        self.assertEqual(result["status"], "success", result)
        self.assertIn("child-ok 1", result["output"])
        self.assertIn("gc-ok", result["output"])
        self.assertNotIn("Error in sitecustomize", result["output"])
        self.assertFalse(record_path.exists())


class TestLifecycleGuardSupervisedGatewayOnly(unittest.TestCase):
    """R70-2: ``tools/code_execution_env.py::_build_child_env`` wired the audit-hook guard into
    EVERY local kernel unconditionally, discarding the existing supervised-gateway condition the
    SOURCE-TEXT scan still honors (``tools/code_execution_tool.py``,
    ``_is_supervised_gateway_process()``). The terminal sibling explicitly preserves lifecycle
    operations from CLI, ``hermes serve``, and unsupervised foreground contexts -- the kernel's
    audit hook must too."""

    @staticmethod
    def _stub_launchctl(bin_dir: Path, record_path: Path) -> None:
        bin_dir.mkdir(parents=True, exist_ok=True)
        stub = bin_dir / "launchctl"
        stub.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "with open(" + repr(str(record_path)) + ", 'a') as f:\n"
            "    f.write(json.dumps(sys.argv) + chr(10))\n"
        )
        stub.chmod(0o755)

    def test_dynamically_built_lifecycle_argv_allowed_outside_supervised_gateway(self):
        from tools import process_registry

        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        tmp_path = Path(tmp)
        bin_dir = tmp_path / "bin"
        record_path = tmp_path / "launchctl_argv.jsonl"
        self._stub_launchctl(bin_dir, record_path)
        patched_path = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
        with patch.object(process_registry, "_is_supervised_gateway_process", lambda: False):
            with _kernel_config(), patch.dict(os.environ, {"PATH": patched_path}):
                result = _run(textwrap.dedent("""\
                    import subprocess
                    a = "launch"
                    b = "ctl"
                    subprocess.run([a + b, "bootout", "gui/501/ai.hermes.gateway"], check=False)
                    """))
        self.assertEqual(result["status"], "success", result)
        self.assertTrue(record_path.exists(),
                         "outside a supervised gateway the audit hook must not block this cell")
