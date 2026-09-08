"""Gateway event-loop freeze backstops for issue #69089."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.shutdown_watchdog import (
    _arm_loop_floor_timer,
    _write_loop_liveness_watchdog_dump,
    start_loop_liveness_watchdog,
)


def _immediate_loop() -> MagicMock:
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    loop.call_soon_threadsafe.side_effect = lambda callback: callback()
    return loop


def test_loop_liveness_final_strike_persists_private_all_thread_dump_before_exit(
    tmp_path, monkeypatch
):
    """The final strike must leave durable evidence even when stderr is discarded."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    events = []
    dump_calls = []
    exit_code = 71
    target = tmp_path / "logs" / "gateway-loop-liveness-watchdog.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def fake_dump_traceback(*, file=None, all_threads):
        dump_calls.append({"file_supplied": file is not None, "all_threads": all_threads})
        events.append("dump")
        if file is not None:
            file.write("synthetic all-thread stack\\n")

    def fake_mark_exited(code, *, reason):
        events.append("mark_exited")
        assert (code, reason) == (exit_code, "loop_liveness_watchdog")

    def fake_exit(code):
        events.append("hard_exit")
        assert code == exit_code

    with (
        patch(
            "gateway.shutdown_watchdog.faulthandler.dump_traceback",
            side_effect=fake_dump_traceback,
        ),
        patch(
            "gateway.lifecycle_ledger.mark_exited", side_effect=fake_mark_exited
        ),
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
    ):
        handle = start_loop_liveness_watchdog(
            loop,
            probe_interval=0.01,
            probe_timeout=0.01,
            max_strikes=1,
            exit_code=exit_code,
        )
        assert handle is not None
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    assert dump_calls == [{"file_supplied": True, "all_threads": True}]
    assert events == ["dump", "mark_exited", "hard_exit"]
    assert target.is_file()
    assert stat.S_IMODE(target.stat().st_mode) == 0o600

    header = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert set(header) == {
        "event_kind",
        "pid",
        "strike_count",
        "probe_interval_s",
        "probe_timeout_s",
        "utc_timestamp",
        "runtime_state",
    }
    assert header["event_kind"] == "loop_liveness_watchdog_final_strike"
    assert header["pid"] == os.getpid()
    assert header["strike_count"] == 1
    assert header["probe_interval_s"] == 0.01
    assert header["probe_timeout_s"] == 0.01
    assert header["utc_timestamp"].endswith("+00:00")
    assert header["runtime_state"] == {"loop_liveness": "unresponsive"}
    assert "synthetic all-thread stack" in target.read_text(encoding="utf-8")


def test_loop_liveness_final_strike_fsyncs_replaced_dump_directory_before_exit(
    tmp_path, monkeypatch
):
    """A replaced final-strike dump must have its directory entry persisted."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    events = []
    directory_fds = set()
    exit_code = 71
    target = tmp_path / "logs" / "gateway-loop-liveness-watchdog.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real_open = os.open
    real_fsync = os.fsync
    real_replace = os.replace
    real_close = os.close

    def spy_open(path, flags, mode=0o777, *, dir_fd=None):
        kwargs = {} if dir_fd is None else {"dir_fd": dir_fd}
        fd = real_open(path, flags, mode, **kwargs)
        if Path(path) == target.parent:
            directory_fds.add(fd)
            events.append("open_directory")
        return fd

    def spy_fsync(fd):
        events.append("directory_fsync" if fd in directory_fds else "file_fsync")
        return real_fsync(fd)

    def spy_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    def spy_close(fd):
        try:
            return real_close(fd)
        finally:
            if fd in directory_fds:
                events.append("close_directory")

    def fake_dump_traceback(*, file=None, all_threads):
        assert file is not None
        assert all_threads is True
        events.append("temp_content_write")
        file.write("synthetic all-thread stack\\n")

    def fake_mark_exited(code, *, reason):
        assert (code, reason) == (exit_code, "loop_liveness_watchdog")
        events.append("mark_exited")

    def fake_exit(code):
        assert code == exit_code
        events.append("hard_exit")

    with (
        patch(
            "gateway.shutdown_watchdog.faulthandler.dump_traceback",
            side_effect=fake_dump_traceback,
        ),
        patch("gateway.shutdown_watchdog.os.open", side_effect=spy_open),
        patch("gateway.shutdown_watchdog.os.fsync", side_effect=spy_fsync),
        patch("gateway.shutdown_watchdog.os.replace", side_effect=spy_replace),
        patch("gateway.shutdown_watchdog.os.close", side_effect=spy_close),
        patch(
            "gateway.lifecycle_ledger.mark_exited", side_effect=fake_mark_exited
        ),
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
    ):
        handle = start_loop_liveness_watchdog(
            loop,
            probe_interval=0.01,
            probe_timeout=0.01,
            max_strikes=1,
            exit_code=exit_code,
        )
        assert handle is not None
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    assert events == [
        "temp_content_write",
        "file_fsync",
        "replace",
        "open_directory",
        "directory_fsync",
        "close_directory",
        "mark_exited",
        "hard_exit",
    ]


def test_loop_liveness_dump_tightens_stale_temp_mode_before_dump_bytes(
    tmp_path, monkeypatch
):
    """A stale fixed temp file is private before any dump bytes are written."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "logs" / "gateway-loop-liveness-watchdog.log"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.parent.mkdir()
    temporary.touch()
    temporary.chmod(0o666)
    write_modes = []
    stack_modes = []
    real_fdopen = os.fdopen

    class ModeCheckingFile:
        def __init__(self, fh):
            self._fh = fh

        def write(self, content):
            mode = stat.S_IMODE(os.fstat(self._fh.fileno()).st_mode)
            write_modes.append(mode)
            return self._fh.write(content)

        def flush(self):
            return self._fh.flush()

        def fileno(self):
            return self._fh.fileno()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self._fh.close()
            return False

    def fake_fdopen(fd, *args, **kwargs):
        return ModeCheckingFile(real_fdopen(fd, *args, **kwargs))

    def fake_dump_traceback(*, file=None, all_threads):
        assert file is not None
        assert all_threads is True
        mode = stat.S_IMODE(os.fstat(file.fileno()).st_mode)
        stack_modes.append(mode)
        file.write("synthetic all-thread stack\\n")

    with (
        patch("gateway.shutdown_watchdog.os.fdopen", side_effect=fake_fdopen),
        patch(
            "gateway.shutdown_watchdog.faulthandler.dump_traceback",
            side_effect=fake_dump_traceback,
        ),
    ):
        assert _write_loop_liveness_watchdog_dump(
            target,
            strikes=1,
            probe_interval=0.01,
            probe_timeout=0.01,
        )

    assert write_modes
    assert set(write_modes) == {0o600}
    assert stack_modes == [0o600]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_loop_liveness_dump_atomically_replaces_latest_private_file(tmp_path, monkeypatch):
    """Repeated final strikes retain one private latest dump, not an unbounded log."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "logs" / "gateway-loop-liveness-watchdog.log"
    dump_count = 0
    real_replace = os.replace

    def fake_dump_traceback(*, file=None, all_threads):
        nonlocal dump_count
        assert file is not None
        assert all_threads is True
        dump_count += 1
        file.write(f"synthetic all-thread stack {dump_count}\\n")

    with (
        patch(
            "gateway.shutdown_watchdog.faulthandler.dump_traceback",
            side_effect=fake_dump_traceback,
        ),
        patch("gateway.lifecycle_ledger.mark_exited"),
        patch("gateway.shutdown_watchdog.os._exit"),
        patch(
            "gateway.shutdown_watchdog.os.replace", side_effect=real_replace
        ) as replace,
    ):
        for _ in range(2):
            loop = MagicMock(spec=asyncio.AbstractEventLoop)
            handle = start_loop_liveness_watchdog(
                loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
            )
            assert handle is not None
            handle.join(timeout=2.0)
            assert not handle.is_alive()

    assert replace.call_count == 2
    temporary = target.with_name(f".{target.name}.tmp")
    assert [Path(call.args[0]) for call in replace.call_args_list] == [
        temporary,
        temporary,
    ]
    assert list((tmp_path / "logs").iterdir()) == [target]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    contents = target.read_text(encoding="utf-8")
    assert "synthetic all-thread stack 1" not in contents
    assert "synthetic all-thread stack 2" in contents


def test_loop_liveness_dump_failure_warns_safely_and_still_hard_exits(
    tmp_path, monkeypatch
):
    """A private-sink failure never prevents recovery or leaks its details."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    events = []

    def fake_mark_exited(*_args, **_kwargs):
        events.append("mark_exited")

    def fake_exit(*_args, **_kwargs):
        events.append("hard_exit")

    with (
        patch(
            "gateway.shutdown_watchdog.os.open",
            side_effect=OSError("write failed at /private/path/with-token"),
        ),
        patch(
            "gateway.lifecycle_ledger.mark_exited", side_effect=fake_mark_exited
        ),
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
        patch("gateway.shutdown_watchdog.logger.warning") as warning,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    assert events == ["mark_exited", "hard_exit"]
    warning.assert_called_once_with(
        "Gateway loop liveness watchdog durable dump failed; "
        "continuing forced recovery."
    )


@pytest.mark.parametrize(
    "failure_stage", ["directory_open", "directory_fsync", "directory_close"]
)
def test_loop_liveness_directory_sync_failure_preserves_replaced_dump_and_exits(
    tmp_path, monkeypatch, failure_stage
):
    """Directory durability failures remain fail-open after replacing the dump."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    target = tmp_path / "logs" / "gateway-loop-liveness-watchdog.log"
    events = []
    directory_fds = set()
    raw_failure = "directory sync failed at /private/path/with-token"
    real_open = os.open
    real_fsync = os.fsync
    real_replace = os.replace
    real_close = os.close

    def fail_directory_open(path, flags, mode=0o777, *, dir_fd=None):
        kwargs = {} if dir_fd is None else {"dir_fd": dir_fd}
        if Path(path) == target.parent:
            if failure_stage == "directory_open":
                raise OSError(raw_failure)
            fd = real_open(path, flags, mode, **kwargs)
            directory_fds.add(fd)
            return fd
        return real_open(path, flags, mode, **kwargs)

    def fail_directory_fsync(fd):
        if failure_stage == "directory_fsync" and fd in directory_fds:
            raise OSError(raw_failure)
        return real_fsync(fd)

    def fail_directory_close(fd):
        if failure_stage == "directory_close" and fd in directory_fds:
            real_close(fd)
            raise OSError(raw_failure)
        return real_close(fd)

    def record_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    def fake_dump_traceback(*, file=None, all_threads):
        assert file is not None
        assert all_threads is True
        events.append("dump")
        file.write("synthetic all-thread stack\\n")

    def fake_mark_exited(*_args, **_kwargs):
        events.append("mark_exited")

    def fake_exit(*_args, **_kwargs):
        events.append("hard_exit")

    with (
        patch("gateway.shutdown_watchdog.os.open", side_effect=fail_directory_open),
        patch("gateway.shutdown_watchdog.os.fsync", side_effect=fail_directory_fsync),
        patch("gateway.shutdown_watchdog.os.close", side_effect=fail_directory_close),
        patch("gateway.shutdown_watchdog.os.replace", side_effect=record_replace),
        patch(
            "gateway.shutdown_watchdog.faulthandler.dump_traceback",
            side_effect=fake_dump_traceback,
        ),
        patch(
            "gateway.lifecycle_ledger.mark_exited", side_effect=fake_mark_exited
        ),
        patch("gateway.shutdown_watchdog.os._exit", side_effect=fake_exit),
        patch("gateway.shutdown_watchdog.logger.warning") as warning,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    assert events == ["dump", "replace", "mark_exited", "hard_exit"]
    assert target.is_file()
    assert "synthetic all-thread stack" in target.read_text(encoding="utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    warning.assert_called_once_with(
        "Gateway loop liveness watchdog durable dump failed; "
        "continuing forced recovery."
    )
    assert raw_failure not in str(warning.call_args)
    assert str(target) not in str(warning.call_args)


def test_loop_liveness_watchdog_stop_during_dump_disarms_hard_exit(
    tmp_path, monkeypatch
):
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    handle_ready = threading.Event()
    handle_ref = {}
    exit_codes = []
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def stop_during_dump(*_args, **_kwargs) -> None:
        assert handle_ready.wait(timeout=2.0)
        handle_ref["handle"].stop()

    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch(
            "gateway.shutdown_watchdog.faulthandler.dump_traceback",
            side_effect=stop_during_dump,
        ) as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append),
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        handle_ref["handle"] = handle
        handle_ready.set()
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    critical.assert_called_once()
    dump.assert_called_once()
    assert dump.call_args.kwargs["all_threads"] is True
    assert dump.call_args.kwargs["file"] is not None
    assert (tmp_path / "logs" / "gateway-loop-liveness-watchdog.log").is_file()
    assert exit_codes == []


def test_loop_liveness_watchdog_stop_before_dump_disarms_dump_and_hard_exit(
    tmp_path, monkeypatch
):
    """A disarm in the final-strike window must not create evidence or exit."""
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    handle_ready = threading.Event()
    handle_ref = {}
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    def stop_before_dump(*_args, **_kwargs) -> None:
        assert handle_ready.wait(timeout=2.0)
        handle_ref["handle"].stop()

    with (
        patch("gateway.shutdown_watchdog.logger.critical", side_effect=stop_before_dump),
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit") as hard_exit,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        handle_ref["handle"] = handle
        handle_ready.set()
        handle.join(timeout=2.0)

    assert not handle.is_alive()
    dump.assert_not_called()
    hard_exit.assert_not_called()
    assert not (tmp_path / "logs" / "gateway-loop-liveness-watchdog.log").exists()


def test_loop_liveness_watchdog_stop_during_final_miss_disarms_hard_exit():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    probe_scheduled = threading.Event()
    release_probe = threading.Event()
    probe_event_ref = {}
    handle_ref = {}
    exit_codes = []

    class FinalStrikeLimit:
        def __gt__(self, _strikes: int) -> bool:
            # If strike evaluation is reached, keep recheck #2 from masking a
            # missing post-probe recheck #1 in this boundary test.
            handle_ref["handle"]._stop_event.clear()
            return False

    def hold_scheduled_probe(callback) -> None:
        probe_event_ref["event"] = callback.__self__
        probe_scheduled.set()
        assert release_probe.wait(timeout=2.0)

    loop.call_soon_threadsafe.side_effect = hold_scheduled_probe
    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit", side_effect=exit_codes.append),
    ):
        handle = start_loop_liveness_watchdog(
            loop,
            probe_interval=0.01,
            probe_timeout=0.01,
            max_strikes=FinalStrikeLimit(),
        )
        assert handle is not None
        handle_ref["handle"] = handle
        assert probe_scheduled.wait(timeout=2.0), "watchdog did not schedule a probe"

        def stop_during_miss() -> bool:
            handle.stop()
            return False

        probe_event_ref["event"].is_set = stop_during_miss
        release_probe.set()
        handle.join(timeout=1.0)

    assert not handle.is_alive()
    assert exit_codes == []
    critical.assert_not_called()
    dump.assert_not_called()


def test_loop_liveness_watchdog_stop_after_first_recheck_skips_final_actions():
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    probe_scheduled = threading.Event()
    release_probe = threading.Event()

    def hold_scheduled_probe(callback) -> None:
        probe_scheduled.set()
        assert release_probe.wait(timeout=2.0)

    loop.call_soon_threadsafe.side_effect = hold_scheduled_probe
    with (
        patch("gateway.shutdown_watchdog.logger.critical") as critical,
        patch("gateway.shutdown_watchdog.faulthandler.dump_traceback") as dump,
        patch("gateway.shutdown_watchdog.os._exit") as hard_exit,
    ):
        handle = start_loop_liveness_watchdog(
            loop, probe_interval=0.01, probe_timeout=0.01, max_strikes=1
        )
        assert handle is not None
        assert probe_scheduled.wait(timeout=2.0), "watchdog did not schedule a probe"

        original_is_set = handle._stop_event.is_set
        is_set_calls = 0

        def stop_on_final_recheck() -> bool:
            nonlocal is_set_calls
            is_set_calls += 1
            # With the forced immediate timeout: _wait_for_probe is call 1,
            # recheck #1 is call 2, and recheck #2 is call 3.
            if is_set_calls == 3:
                handle.stop()
            return original_is_set()

        handle._stop_event.is_set = stop_on_final_recheck
        with patch(
            "gateway.shutdown_watchdog.time.monotonic", side_effect=[0.0, 1.0]
        ):
            release_probe.set()
            handle.join(timeout=1.0)

    assert is_set_calls == 3
    assert not handle.is_alive()
    critical.assert_not_called()
    dump.assert_not_called()
    hard_exit.assert_not_called()


def test_gateway_config_loop_watchdog_round_trip():
    """loop_watchdog is a config.yaml knob: default on, nested-gateway form honored."""
    from gateway.config import GatewayConfig

    assert GatewayConfig.from_dict({}).loop_watchdog is True
    assert GatewayConfig.from_dict({"loop_watchdog": False}).loop_watchdog is False
    assert (
        GatewayConfig.from_dict(
            {"gateway": {"loop_watchdog": "off"}}
        ).loop_watchdog
        is False
    )
    config = GatewayConfig.from_dict({"loop_watchdog": False})
    assert config.to_dict()["loop_watchdog"] is False


def test_gateway_config_loop_watchdog_tuning_round_trip():
    """Watchdog tolerance knobs parse, serialize, and clamp malformed values."""
    from gateway.config import GatewayConfig

    # Defaults
    default = GatewayConfig.from_dict({})
    assert default.loop_watchdog is True
    assert default.loop_watchdog_probe_interval_s == 30.0
    assert default.loop_watchdog_probe_timeout_s == 10.0
    assert default.loop_watchdog_max_strikes == 3

    # Explicit values round-trip
    cfg = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": 45,
            "loop_watchdog_probe_timeout_s": 15,
            "loop_watchdog_max_strikes": 12,
        }
    )
    assert cfg.loop_watchdog_probe_interval_s == 45.0
    assert cfg.loop_watchdog_probe_timeout_s == 15.0
    assert cfg.loop_watchdog_max_strikes == 12
    d = cfg.to_dict()
    assert d["loop_watchdog_probe_interval_s"] == 45.0
    assert d["loop_watchdog_probe_timeout_s"] == 15.0
    assert d["loop_watchdog_max_strikes"] == 12

    # Nested gateway.* form honored
    nested = GatewayConfig.from_dict(
        {
            "gateway": {
                "loop_watchdog_probe_interval_s": 60,
                "loop_watchdog_probe_timeout_s": 20,
                "loop_watchdog_max_strikes": 20,
            }
        }
    )
    assert nested.loop_watchdog_probe_interval_s == 60.0
    assert nested.loop_watchdog_probe_timeout_s == 20.0
    assert nested.loop_watchdog_max_strikes == 20

    # Malformed / degenerate values fall back to safe defaults
    clamped = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": 0,
            "loop_watchdog_probe_timeout_s": -5,
            "loop_watchdog_max_strikes": 0,
        }
    )
    assert clamped.loop_watchdog_probe_interval_s == 30.0
    assert clamped.loop_watchdog_probe_timeout_s == 10.0
    assert clamped.loop_watchdog_max_strikes == 3


def test_gateway_config_loop_watchdog_nonfinite_values_degrade():
    """NaN/Inf tuning values fall back to defaults instead of reaching the
    watchdog's Event.wait loop (or aborting config load via int(inf))."""
    from gateway.config import GatewayConfig

    cfg = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": float("inf"),
            "loop_watchdog_probe_timeout_s": float("nan"),
            "loop_watchdog_max_strikes": float("inf"),  # int() would raise
        }
    )
    assert cfg.loop_watchdog_probe_interval_s == 30.0
    assert cfg.loop_watchdog_probe_timeout_s == 10.0
    assert cfg.loop_watchdog_max_strikes == 3

    # Oversized-but-finite values also clamp to defaults.
    big = GatewayConfig.from_dict(
        {
            "loop_watchdog_probe_interval_s": 86400,
            "loop_watchdog_probe_timeout_s": 7200,
            "loop_watchdog_max_strikes": 10**9,
        }
    )
    assert big.loop_watchdog_probe_interval_s == 30.0
    assert big.loop_watchdog_probe_timeout_s == 10.0
    assert big.loop_watchdog_max_strikes == 3


def test_load_gateway_config_bridges_loop_watchdog_keys(tmp_path, monkeypatch):
    """The real startup loader must honor gateway.loop_watchdog* from
    config.yaml — from_dict's nested fallback never sees the yaml gateway
    section because load_gateway_config builds gw_data flat."""
    from gateway.config import load_gateway_config

    (tmp_path / "config.yaml").write_text(
        "gateway:\n"
        "  loop_watchdog: false\n"
        "  loop_watchdog_probe_interval_s: 45\n"
        "  loop_watchdog_probe_timeout_s: 15\n"
        "  loop_watchdog_max_strikes: 12\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("gateway.config.get_hermes_home", lambda: tmp_path)

    cfg = load_gateway_config()
    assert cfg.loop_watchdog is False
    assert cfg.loop_watchdog_probe_interval_s == 45.0
    assert cfg.loop_watchdog_probe_timeout_s == 15.0
    assert cfg.loop_watchdog_max_strikes == 12


def test_gateway_runner_liveness_guards_start_and_stop():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._loop_floor_timer_handle = None
    runner._loop_liveness_watchdog = None
    runner.config = None
    loop = MagicMock(spec=asyncio.AbstractEventLoop)
    floor_timer = MagicMock()
    watchdog = MagicMock()
    watchdog.is_alive.return_value = True

    with (
        patch(
            "gateway.run._arm_loop_floor_timer", return_value=floor_timer
        ) as arm_floor,
        patch(
            "gateway.run.start_loop_liveness_watchdog", return_value=watchdog
        ) as start_watchdog,
    ):
        runner._start_loop_liveness_guards(loop)

    arm_floor.assert_called_once_with(loop)
    start_watchdog.assert_called_once_with(
        loop,
        probe_interval=30.0,
        probe_timeout=10.0,
        max_strikes=3,
    )
    assert runner._loop_floor_timer_handle is floor_timer
    assert runner._loop_liveness_watchdog is watchdog

    runner._stop_loop_liveness_guards()

    watchdog.stop.assert_called_once_with()
    floor_timer.cancel.assert_called_once_with()
    assert runner._loop_liveness_watchdog is None
    assert runner._loop_floor_timer_handle is None
