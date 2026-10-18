"""The canonical peer tool tests can never write the user's real Hermes config.

Regression for an incident: run outside ``tests/`` (so without the hermetic conftest), the peer
tool test helpers resolved ``HERMES_HOME`` to the real home and replaced its ``config.yaml``. This
reproduces that exact run in a subprocess whose HOME is a stand-in "real home" and requires that
nothing at all is written under it."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_OUTSIDE_RUNS = {
    "tests/agent/test_canonical_peer_tools.py": "owner_turns_ignore_the_peer_policy",
    "tests/gateway/test_canonical_peer_tool_grant.py": "sends_its_real_session",
    "tests/gateway/test_canonical_record_cas.py": "control_a_normal_record",
    "tests/gateway/test_canonical_text_cap_ingress.py": "over_the_cap_is_refused",
    "tests/bridge/test_shim_text_cap_notice.py": "short_thread_reply_reaches_the_gateway",
}
# Modules whose pins come from tests/hermes_home_pin.py: a run must find each pin put back.
_PUT_BACK = ("tests/gateway/test_canonical_record_cas.py",
             "tests/gateway/test_canonical_text_cap_ingress.py",
             "tests/bridge/test_shim_text_cap_notice.py")
_EACH_PIN_CHECKED_TWICE = {"env": 2, "override": 2, "home_caches": 2}  # the import pin and the test pin

# Runs one test module with pytest, then reports, still before interpreter exit, whether HOME,
# HERMES_HOME and the override are what the process started with and how many pin exits checked
# that. Two probes report the same at exit: one runs before every hook the run registered, one after.
# With argv[1] == "block-browser-cleanup", exactly the browser tool's two exit hooks are unregistered
# once the run is over, in this child only; no other hook is touched.
_RUN_THEN_REPORT = r'''
import atexit, json, sys
import pytest
from tests import hermes_home_pin as pin

start = pin.home_state()
def probe(when):
    return lambda: print(f"EXIT-HOME {when} {json.dumps(pin.home_state() == start)}", flush=True)
atexit.register(probe("last"))
rc = pytest.main(sys.argv[2:])
blocked = []
lifecycle = sys.modules.get("tools.browser_tool_lifecycle")
if sys.argv[1] == "block-browser-cleanup" and lifecycle is not None:
    for hook in (lifecycle._emergency_cleanup_all_sessions, lifecycle._stop_browser_cleanup_thread):
        atexit.unregister(hook)
        blocked.append(hook.__name__)
atexit.register(probe("first"))
print("PIN-REPORT " + json.dumps({"restored": pin.home_state() == start, "checks": pin.RESTORE_CHECKS,
                                  "blocked": blocked}), flush=True)
sys.exit(rc)
'''


def _run_outside_tests(tmp_path, module, *, home, tmpdir, browser_cleanup):
    """Run ``module``'s selected test from a copy outside tests/ (no conftest, as in the incident)."""
    outside = tmp_path / "outside-tests"
    outside.mkdir()
    copy = outside / Path(module).name
    shutil.copyfile(_REPO / module, copy)
    tmpdir.mkdir()
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "PYTHONPATH": str(_REPO),
           "TMPDIR": str(tmpdir), "LANG": "C.UTF-8", "TZ": "UTC",
           "SHIM_PATH": str(_REPO / "bridge" / "canonical" / "acp-canonical-shim.py")}
    mode = "run-browser-cleanup" if browser_cleanup else "block-browser-cleanup"
    run = subprocess.run([sys.executable, "-c", _RUN_THEN_REPORT, mode, str(copy), "-q", "-p", "no:cacheprovider",
                          "-k", _OUTSIDE_RUNS[module]], cwd=_REPO, env=env, capture_output=True, text=True,
                         timeout=240)
    out = run.stdout[-3000:] + run.stderr[-3000:]
    assert run.returncode == 0 and re.search(r"\b1 passed\b", run.stdout), out
    [report] = [json.loads(line.split(" ", 1)[1]) for line in run.stdout.splitlines() if line.startswith("PIN-REPORT ")]
    probes = re.findall(r"^EXIT-HOME (first|last) (true|false)$", run.stdout, re.M)
    return report, sorted(probes), out


@pytest.mark.parametrize("module", list(_OUTSIDE_RUNS))
def test_a_run_outside_the_hermetic_conftest_writes_nothing_in_the_real_home(tmp_path, module):
    """The browser tool's exit cleanup, which resolves the home only at exit, is not under test here
    and is unregistered for this run; every other exit hook still runs under the stand-in HOME."""
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    report, probes, out = _run_outside_tests(tmp_path, module, home=real_home, tmpdir=tmp_path / "tmpdir",
                                             browser_cleanup=False)
    written = sorted(str(p.relative_to(real_home)) for p in real_home.rglob("*"))
    assert written == [], f"the run wrote into the real home: {written}"
    if module in _PUT_BACK:
        assert report["restored"] and report["checks"] == _EACH_PIN_CHECKED_TWICE, out
        assert probes == [("first", "true"), ("last", "true")], out


@pytest.mark.parametrize("module", _PUT_BACK)
def test_exit_hooks_of_a_run_outside_the_hermetic_conftest_stay_in_the_home_it_started_with(tmp_path, module):
    """Exit hooks run (browser cleanup included) in a process whose HOME is a tmp dir from the start:
    they see that HOME, not a pin, and the run and its hooks write nothing outside it."""
    home = tmp_path / "home"
    home.mkdir()
    there_before = set(tmp_path.rglob("*"))  # this test's own hermetic home, from the conftest
    report, probes, out = _run_outside_tests(tmp_path, module, home=home, tmpdir=home / "tmp",
                                             browser_cleanup=True)
    outside_home = sorted(str(p.relative_to(tmp_path)) for p in set(tmp_path.rglob("*")) - there_before
                          if not p.is_relative_to(home) and not p.is_relative_to(tmp_path / "outside-tests"))
    assert outside_home == [], f"the run wrote outside its HOME: {outside_home}"
    assert report["restored"] and report["checks"] == _EACH_PIN_CHECKED_TWICE and report["blocked"] == [], out
    assert probes == [("first", "true"), ("last", "true")], out


_LOAD_AND_RESOLVE = r'''
import importlib.util, subprocess, sys
from pathlib import Path
from hermes_constants import get_hermes_home_override, set_hermes_home_override
set_hermes_home_override(sys.argv[2])
spec = importlib.util.spec_from_file_location('loaded_peer_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from hermes_cli.config import get_config_path
path = get_config_path().resolve()
assert path.is_relative_to(module._PINNED_ROOT), (path, module._PINNED_ROOT)
override = Path(get_hermes_home_override()).resolve()
assert override.is_relative_to(module._PINNED_ROOT), ("override", override, module._PINNED_ROOT)
child = subprocess.run([sys.executable, '-c', 'from hermes_cli.config import get_config_path; '
                        'print(get_config_path().resolve())'], capture_output=True, text=True, check=True)
assert Path(child.stdout.strip()).is_relative_to(module._PINNED_ROOT), ("subprocess", child.stdout, child.stderr)
'''


# These two modules still keep their import-time pin after they load; the modules that put it back
# are checked by test_test_module_pins_its_imports_and_tests_and_restores_a_cached_profile_scope.
@pytest.mark.parametrize("module", ["tests/agent/test_canonical_peer_tools.py",
                                    "tests/gateway/test_canonical_peer_tool_grant.py"])
def test_test_module_pins_the_resolver_even_with_a_cached_profile_scope(tmp_path, module):
    """An override already bound to a (stand-in) live profile before the module loads is replaced:
    the override, the real resolver and a subprocess it spawns all name the module's tmp root."""
    old_home = tmp_path / "stand-in-live-profile"
    old_home.mkdir()
    (old_home / "config.yaml").write_text("canary: unchanged\n", encoding="utf-8")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "start-home"),
           "HERMES_HOME": str(tmp_path / "start-hermes-home"), "TMPDIR": str(tmp_path),
           "PYTHONPATH": str(_REPO)}
    result = subprocess.run([sys.executable, "-c", _LOAD_AND_RESOLVE, str(_REPO / module), str(old_home)],
                            cwd=_REPO, env=env, capture_output=True, text=True, timeout=60)
    assert (old_home / "config.yaml").read_text(encoding="utf-8") == "canary: unchanged\n"
    assert result.returncode == 0, result.stdout + result.stderr


# Every module-level value the load imported that names the caller's HOME, HERMES_HOME or override
# would be an import-time path the pin missed; one that names the module's own test home would be
# one that later modules' tests write through into that home.
_LOAD_PIN_AND_RESTORE = r'''
import importlib.util, os, subprocess, sys
from pathlib import Path
from hermes_constants import get_hermes_home_override, set_hermes_home_override
set_hermes_home_override(sys.argv[2])
callers = (os.environ['HOME'], os.environ['HERMES_HOME'], sys.argv[2])
def state():
    return os.environ.get('HOME'), os.environ.get('HERMES_HOME'), get_hermes_home_override()
def names(value, home, depth=0):
    if isinstance(value, (str, Path)):
        return str(value) == home or str(value).startswith(home + os.sep)
    if depth < 2 and isinstance(value, (list, tuple, set, frozenset)):
        return any(names(v, home, depth + 1) for v in value)
    if depth < 2 and isinstance(value, dict):
        return any(names(k, home, depth + 1) or names(v, home, depth + 1) for k, v in list(value.items()))
    return False
before, already = state(), set(sys.modules)
spec = importlib.util.spec_from_file_location('loaded_pinned_test', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
imported = [(name, key, value) for name, mod in list(sys.modules.items())
            if name not in already and name != spec.name and mod is not None
            for key, value in list(vars(mod).items()) if not key.startswith('__')]
missed = [f'{name}.{key}' for name, key, value in imported if any(names(value, home) for home in callers)]
assert not missed, ("import-time values name the caller's home", missed)
own = [f'{name}.{key}' for name, key, value in imported if names(value, str(module._PINNED_ROOT))]
assert not own, ("import-time values name the module's test home", own)
assert state() == before, ("left behind by the import", state(), before)
with module.pinned_hermes_home(module._PINNED_ROOT):  # what the module fixture holds during its tests
    from hermes_cli.config import get_config_path
    path = get_config_path().resolve()
    assert path.is_relative_to(module._PINNED_ROOT), (path, module._PINNED_ROOT)
    override = Path(get_hermes_home_override()).resolve()
    assert override.is_relative_to(module._PINNED_ROOT), ("override", override, module._PINNED_ROOT)
    child = subprocess.run([sys.executable, '-c', 'from hermes_cli.config import get_config_path; '
                            'print(get_config_path().resolve())'], capture_output=True, text=True, check=True)
    assert Path(child.stdout.strip()).is_relative_to(module._PINNED_ROOT), ("subprocess", child.stdout, child.stderr)
assert state() == before, ("left behind by the test pin", state(), before)
'''


@pytest.mark.parametrize("module", _PUT_BACK)
def test_test_module_pins_its_imports_and_tests_and_restores_a_cached_profile_scope(tmp_path, module):
    """With an override already bound to a (stand-in) live profile, the module's Hermes imports and,
    under its fixture's pin, the resolver, the override and a subprocess all name the module's tmp
    root; once the import and the pin are over, the caller's HOME, HERMES_HOME and override are back."""
    old_home = tmp_path / "stand-in-live-profile"
    old_home.mkdir()
    (old_home / "config.yaml").write_text("canary: unchanged\n", encoding="utf-8")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "start-home"),
           "HERMES_HOME": str(tmp_path / "start-hermes-home"), "TMPDIR": str(tmp_path),
           "PYTHONPATH": str(_REPO)}
    result = subprocess.run([sys.executable, "-c", _LOAD_PIN_AND_RESTORE, str(_REPO / module), str(old_home)],
                            cwd=_REPO, env=env, capture_output=True, text=True, timeout=60)
    assert (old_home / "config.yaml").read_text(encoding="utf-8") == "canary: unchanged\n"
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("module", ["tests/agent/test_canonical_peer_tools.py",
                                    "tests/gateway/test_canonical_peer_tool_grant.py",
                                    "tests/gateway/test_canonical_record_cas.py"])
def test_test_module_loads_without_the_posix_account_database(tmp_path, module):
    """No unconditional ``pwd``: the module still loads (and pins) where it does not exist."""
    code = ("import importlib.util, sys; sys.modules['pwd'] = None; "
            "spec = importlib.util.spec_from_file_location('loaded_peer_test', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; "
            "spec.loader.exec_module(module); print(module._PINNED_ROOT)")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path),
           "PYTHONPATH": str(_REPO)}
    result = subprocess.run([sys.executable, "-c", code, str(_REPO / module)], cwd=_REPO, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
