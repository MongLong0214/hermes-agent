"""The canonical peer tool tests can never write the user's real Hermes config.

Regression for an incident: run outside ``tests/`` (so without the hermetic conftest), the peer
tool test helpers resolved ``HERMES_HOME`` to the real home and replaced its ``config.yaml``. This
reproduces that exact run in a subprocess whose HOME is a stand-in "real home" and requires that
nothing at all is written under it."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("module, selector", [
    ("tests/agent/test_canonical_peer_tools.py", "owner_turns_ignore_the_peer_policy"),
    ("tests/gateway/test_canonical_peer_tool_grant.py", "sends_its_real_session"),
    ("tests/gateway/test_canonical_record_cas.py", "control_a_normal_record"),
])
def test_a_run_outside_the_hermetic_conftest_writes_nothing_in_the_real_home(tmp_path, module, selector):
    real_home, outside = tmp_path / "real-home", tmp_path / "outside-tests"
    real_home.mkdir()
    outside.mkdir()
    copy = outside / Path(module).name  # outside tests/: no conftest, as in the incident
    shutil.copyfile(_REPO / module, copy)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(real_home), "PYTHONPATH": str(_REPO),
           "TMPDIR": str(tmp_path / "tmpdir"), "LANG": "C.UTF-8", "TZ": "UTC"}
    (tmp_path / "tmpdir").mkdir()
    run = subprocess.run([sys.executable, "-m", "pytest", str(copy), "-q", "-p", "no:cacheprovider",
                          "-k", selector], cwd=_REPO, env=env, capture_output=True, text=True, timeout=240)
    written = sorted(str(p.relative_to(real_home)) for p in real_home.rglob("*"))
    assert written == [], f"the run wrote into the real home: {written}"
    assert run.returncode == 0 and re.search(r"\b1 passed\b", run.stdout), run.stdout[-2000:] + run.stderr[-2000:]


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


@pytest.mark.parametrize("module", ["tests/agent/test_canonical_peer_tools.py",
                                    "tests/gateway/test_canonical_peer_tool_grant.py",
                                    "tests/gateway/test_canonical_record_cas.py"])
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
