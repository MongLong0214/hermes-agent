"""No code in the canonical peer tool tests can touch the user's real HOME or HERMES_HOME.

Regression for an incident: run outside ``tests/`` (so without the hermetic conftest), the peer
tool test helpers resolved ``HERMES_HOME`` to the real home and replaced its ``config.yaml``. The
module pins HOME, HERMES_HOME and the context-local home override in a module-scoped fixture, so
the property holds in two halves: importing the module does no HOME-dependent work, and the pin is
active before every test body runs and gone again once the module is done. Each check runs in a
subprocess whose HOME, HERMES_HOME and home override name stand-in "real" homes."""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MODULES = {"tests/agent/test_canonical_peer_tools.py": "owner_turns_ignore_the_peer_policy"}


@pytest.mark.parametrize("module, selector", list(_MODULES.items()))
def test_a_run_outside_the_hermetic_conftest_writes_nothing_in_the_real_home(tmp_path, module, selector):
    real_home, outside = tmp_path / "real-home", tmp_path / "outside-tests"
    real_home.mkdir()
    outside.mkdir()
    copy = outside / Path(module).name  # outside tests/: no conftest, as in the incident
    shutil.copyfile(_REPO / module, copy)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(real_home), "PYTHONPATH": str(_REPO),
           "TMPDIR": str(tmp_path / "tmpdir"), "LANG": "C.UTF-8", "TZ": "UTC"}
    (tmp_path / "tmpdir").mkdir()
    run = subprocess.run([sys.executable, "-B", "-m", "pytest", str(copy), "-q", "-p", "no:cacheprovider",
                          "-k", selector], cwd=_REPO, env=env, capture_output=True, text=True, check=False, timeout=240)
    written = sorted(str(p.relative_to(real_home)) for p in real_home.rglob("*"))
    assert written == [], f"the run wrote into the real home: {written}"
    assert run.returncode == 0 and re.search(r"\b1 passed\b", run.stdout), run.stdout[-2000:] + run.stderr[-2000:]


def _stand_in_homes(tmp_path):
    """(HOME, HERMES_HOME, a live profile already bound as the override), each with a canary config."""
    homes = tmp_path / "real-home", tmp_path / "real-hermes-home", tmp_path / "live-profile"
    for home in homes:
        home.mkdir()
        (home / "config.yaml").write_text("canary: unchanged\n", encoding="utf-8")
    return homes


def _untouched(*homes):
    return all(sorted(p.name for p in home.iterdir()) == ["config.yaml"]
               and (home / "config.yaml").read_text(encoding="utf-8") == "canary: unchanged\n" for home in homes)


# Loads the module the way a collector does, auditing every filesystem and process event its import
# raises; a path event under HOME, HERMES_HOME or the bound profile, or any spawn, is HOME-dependent work.
_IMPORT_UNDER_AUDIT = r'''
import importlib.util, json, os, sys
from hermes_constants import get_hermes_home_override, set_hermes_home_override
module_path, profile = sys.argv[1], sys.argv[2]
set_hermes_home_override(profile)
roots = [os.path.realpath(p) for p in (os.environ["HOME"], os.environ["HERMES_HOME"], profile)]
state = lambda: [os.environ.get("HOME"), os.environ.get("HERMES_HOME"), get_hermes_home_override()]
before, touched, busy = state(), [], []
PATH_EVENTS = {"open", "os.mkdir", "os.listdir", "os.scandir", "os.remove", "os.rename", "os.rmdir", "os.chmod",
               "os.utime", "os.truncate", "shutil.rmtree", "shutil.copyfile", "sqlite3.connect", "tempfile.mkdtemp"}

def audit(event, args):
    if busy:
        return
    busy.append(1)
    try:
        if event in ("subprocess.Popen", "os.system", "os.posix_spawn"):
            touched.append([event, repr(args)[:200]])
        elif event in PATH_EVENTS and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = os.path.realpath(os.fsdecode(args[0]))
            if any(path == root or path.startswith(root + os.sep) for root in roots):
                touched.append([event, path])
    finally:
        busy.clear()

sys.addaudithook(audit)
spec = importlib.util.spec_from_file_location("loaded_peer_test", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(json.dumps({"touched": touched, "before": before, "after": state()}))
'''


@pytest.mark.parametrize("module", list(_MODULES))
def test_importing_the_module_does_no_home_dependent_work(tmp_path, module):
    """Import opens, creates and spawns nothing under HOME, HERMES_HOME or an already bound profile
    scope, and leaves all three as they were: no state exists before the pin does."""
    home, hermes_home, profile = _stand_in_homes(tmp_path)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "HERMES_HOME": str(hermes_home),
           "TMPDIR": str(tmp_path), "PYTHONPATH": str(_REPO)}
    result = subprocess.run([sys.executable, "-B", "-c", _IMPORT_UNDER_AUDIT, str(_REPO / module), str(profile)],
                            cwd=_REPO, env=env, capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    seen = json.loads(result.stdout.strip().splitlines()[-1])
    assert seen["touched"] == [], seen["touched"]
    assert seen["after"] == seen["before"] == [str(home), str(hermes_home), str(profile)], seen
    assert _untouched(home, hermes_home, profile)


# Records what a test would resolve, in-process and in a child it spawns, at two moments: "setup",
# just before the first function-scoped fixture of the test (only module-level pins are in force),
# and "body", just before the test body runs.
_PIN_PROBE = r'''
import json, os, subprocess, sys
import pytest
from hermes_constants import get_hermes_home_override, set_hermes_home_override

_set_up = set()


def pytest_configure(config):
    set_hermes_home_override(os.environ["PIN_PROBE_PROFILE"])  # a profile scope bound before collection


def _record(item, moment):
    from hermes_cli.config import get_config_path
    child = subprocess.run([sys.executable, "-B", "-c", "from hermes_cli.config import get_config_path; "
                            "print(get_config_path().resolve())"], capture_output=True, text=True, check=True)
    seen = {"test": item.name, "moment": moment, "home": os.environ.get("HOME"),
            "hermes_home": os.environ.get("HERMES_HOME"), "override": get_hermes_home_override(),
            "config": str(get_config_path().resolve()), "child_config": child.stdout.strip()}
    with open(os.environ["PIN_PROBE_OUT"], "a", encoding="utf-8") as out:
        out.write(json.dumps(seen) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    if fixturedef.scope == "function" and request.node.nodeid not in _set_up:
        _set_up.add(request.node.nodeid)
        _record(request.node, "setup")
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    _record(item, "body")
    yield
'''


@pytest.mark.parametrize("module, selector", list(_MODULES.items()))
def test_the_pin_holds_for_every_test_body_and_is_gone_after_the_module(tmp_path, module, selector):
    """Outside the hermetic conftest, with a live profile already bound: a test that runs before the
    module sees the caller's own homes; each test of the module sees only the module's tmp home before
    any of its function-scoped code runs, and nothing of the caller's in its body (nor does a child it
    spawns); a test that runs after the module sees the caller's homes again, and the tmp home is gone."""
    home, hermes_home, profile = _stand_in_homes(tmp_path)
    outside, plugin_dir, records = tmp_path / "outside-tests", tmp_path / "plugin", tmp_path / "seen.jsonl"
    outside.mkdir()
    plugin_dir.mkdir()
    (plugin_dir / "pin_probe.py").write_text(_PIN_PROBE, encoding="utf-8")
    first, last = outside / "test_pin_probe_first.py", outside / "test_pin_probe_last.py"
    first.write_text("def test_pin_probe_before():\n    pass\n", encoding="utf-8")
    last.write_text("def test_pin_probe_after():\n    pass\n", encoding="utf-8")
    copy = outside / Path(module).name
    shutil.copyfile(_REPO / module, copy)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "HERMES_HOME": str(hermes_home),
           "PYTHONPATH": os.pathsep.join((str(_REPO), str(plugin_dir))), "TMPDIR": str(tmp_path),
           "LANG": "C.UTF-8", "TZ": "UTC", "PIN_PROBE_PROFILE": str(profile), "PIN_PROBE_OUT": str(records)}
    run = subprocess.run([sys.executable, "-B", "-m", "pytest", str(first), str(copy), str(last), "-q",
                          "-p", "no:cacheprovider", "-p", "pin_probe", "-k", f"{selector} or pin_probe"],
                         cwd=_REPO, env=env, capture_output=True, text=True, check=False, timeout=240)
    assert run.returncode == 0 and re.search(r"\b3 passed\b", run.stdout), run.stdout[-2000:] + run.stderr[-2000:]
    seen = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    assert seen[0]["test"] == "test_pin_probe_before" and seen[-1]["test"] == "test_pin_probe_after", seen
    caller = {"home": str(home), "hermes_home": str(hermes_home), "override": str(profile),
              "config": str((profile / "config.yaml").resolve()),
              "child_config": str((hermes_home / "config.yaml").resolve())}
    for outside_the_module in (seen[0], seen[-1]):
        assert {k: outside_the_module[k] for k in caller} == caller, outside_the_module
    stand_ins = (home, hermes_home, profile)
    setups, bodies = ([s for s in seen[1:-1] if s["moment"] == moment] for moment in ("setup", "body"))
    assert setups and len(setups) == len(bodies), seen
    for setup in setups:
        root = Path(setup["home"])
        assert not any(root.is_relative_to(h) or h.is_relative_to(root) for h in stand_ins), setup
        assert setup["hermes_home"] == setup["override"] == str(root / ".hermes"), setup
        assert Path(setup["config"]).is_relative_to(root) and Path(setup["child_config"]).is_relative_to(root), setup
        assert not root.exists(), f"the module's tmp home outlived the module: {root}"
    for body in bodies:
        resolved = [Path(body[k]) for k in caller]
        assert not any(p.is_relative_to(h) for p in resolved for h in stand_ins), body
    assert _untouched(home, hermes_home, profile)


@pytest.mark.parametrize("module", list(_MODULES))
def test_test_module_loads_without_the_posix_account_database(tmp_path, module):
    """No unconditional ``pwd``: the module still loads where it does not exist."""
    code = ("import importlib.util, sys; sys.modules['pwd'] = None; "
            "spec = importlib.util.spec_from_file_location('loaded_peer_test', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; "
            "spec.loader.exec_module(module); print(module._LIVE_CONFIG)")
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "home"), "TMPDIR": str(tmp_path),
           "PYTHONPATH": str(_REPO)}
    result = subprocess.run([sys.executable, "-B", "-c", code, str(_REPO / module)], cwd=_REPO, env=env,
                            capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
