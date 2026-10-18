"""Collecting or running a module that pins its tmp Hermes home leaves no home behind (#90 review).

The modules below pinned HOME, HERMES_HOME and the home override at import, and put back only the
override, after their own tests. Every other module in the same process then ran its tests under
another module's home (test_canonical_surface_ingress.py, collected first: 36 failures). Their pins
now come from tests/hermes_home_pin.py: one while they import Hermes, one owned by a module fixture
while their tests run, both put back in ``finally``.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import hermes_home_pin

_REPO = Path(__file__).resolve().parents[1]
# Each pinned module, and a module it imports inside its import-time pin.
_PINNED_MODULES = {"tests/gateway/test_canonical_record_cas.py": "gateway.config",
                   "tests/gateway/test_canonical_text_cap_ingress.py": "gateway.config",
                   "tests/bridge/test_shim_text_cap_notice.py": "gateway.config",
                   "tests/agent/test_canonical_peer_tools.py": "tools.tool_search",
                   "tests/gateway/test_canonical_peer_tool_grant.py": "gateway.config"}
_CHILD = "HERMES_HOME_PIN_CHECK_CHILD"


def _startup_home():
    """HOME as the interpreter started: ``posix.environ`` (``nt.environ``) is the startup copy that
    assignments to ``os.environ`` never update. The hermetic conftest does not redirect HOME."""
    environ = __import__(os.name).environ
    return environ.get(b"HOME" if os.name == "posix" else "HOME")


def _collected(module):
    path = _REPO / module
    files = (getattr(m, "__file__", None) for m in list(sys.modules.values()))
    return any(f and f.endswith(path.name) and Path(f).resolve() == path for f in files)


# Read while this module is collected, i.e. after every module collected before it.
_STATE_AT_COLLECTION = hermes_home_pin.home_state()
_COLLECTED_BEFORE = tuple(m for m in _PINNED_MODULES if _collected(m))
_IS_CHILD = os.environ.get(_CHILD) == "1"


def _home_caches():
    import hermes_constants

    return (dict(hermes_constants._HOME_KEY_CACHE), hermes_constants._default_hermes_root_memo,
            hermes_constants._PINNED_PROCESS_HERMES_HOME, hermes_constants._profile_fallback_warned)


def test_the_modules_collected_before_this_one_left_home_env_and_override_unchanged():
    if not _COLLECTED_BEFORE:
        assert not _IS_CHILD, "the pinned modules were not collected ahead of this check"
        # Collected without them (e.g. the per-file runner): collect them ahead of this test in a
        # fresh process. Their own tests are deselected; only their import-time pins run.
        this = str(Path(__file__).resolve().relative_to(_REPO))
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *_PINNED_MODULES,
                              this, "-k", "modules_collected_before_this_one"], env={**os.environ, _CHILD: "1"},
                             cwd=_REPO, capture_output=True, text=True, timeout=240)
        assert run.returncode == 0 and re.search(r"\b1 passed\b", run.stdout), run.stdout[-3000:] + run.stderr
        return
    from tests.conftest import HERMES_HOME_AT_CONFTEST_IMPORT

    startup_home = _startup_home()
    if isinstance(startup_home, bytes):
        startup_home = os.fsdecode(startup_home)
    # As before any test module was imported: the startup HOME, the conftest's HERMES_HOME, no override.
    assert _STATE_AT_COLLECTION == (startup_home, HERMES_HOME_AT_CONFTEST_IMPORT, None), _COLLECTED_BEFORE
    # And after their tests ran (their module fixtures are torn down by now): HOME and the override
    # are back. HERMES_HOME is this test's own, from the conftest.
    from hermes_constants import get_hermes_home_override

    assert (os.environ.get("HOME"), get_hermes_home_override()) == (startup_home, None)
    # Every pin that has exited so far, each import-time pin included, checked what it put back.
    assert min(hermes_home_pin.RESTORE_CHECKS.values()) >= len(hermes_home_pin.IMPORT_PINNED) >= len(_COLLECTED_BEFORE)


def test_a_pin_puts_back_env_override_and_home_caches_even_when_its_block_raises(tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "caller-home"))
    (tmp_path / "pinned" / ".hermes").mkdir(parents=True)  # an existing home is what the key cache keeps
    caller = hermes_constants.set_hermes_home_override(tmp_path / "caller-override")
    try:
        before, caches, checks = hermes_home_pin.home_state(), _home_caches(), dict(hermes_home_pin.RESTORE_CHECKS)
        with pytest.raises(RuntimeError, match="inside the pin"):
            with hermes_home_pin.pinned_hermes_home(tmp_path / "pinned") as pinned:
                assert hermes_constants.get_hermes_home() == pinned == tmp_path / "pinned" / ".hermes"
                assert os.environ["HOME"] == str(tmp_path / "pinned")
                hermes_constants.hermes_home_key()  # the caches remember the pinned home ...
                hermes_constants.get_default_hermes_root()
                hermes_constants.pin_process_hermes_home(pinned)
                assert _home_caches() != caches
                raise RuntimeError("inside the pin")
        assert hermes_home_pin.home_state() == before  # ... and forget it, HOME unset again
        assert _home_caches() == caches
        assert hermes_home_pin.RESTORE_CHECKS == {part: count + 1 for part, count in checks.items()}
    finally:
        hermes_constants.reset_hermes_home_override(caller)


@pytest.mark.parametrize("module", list(_PINNED_MODULES))
@pytest.mark.parametrize("import_fails", [False, True], ids=["imports", "an-import-raises"])
def test_loading_a_pinned_module_leaves_env_override_and_home_caches_as_they_were(
        module, import_fails, tmp_path, monkeypatch):
    import hermes_constants

    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    if import_fails:  # raised inside the module's pinned import block
        monkeypatch.setitem(sys.modules, _PINNED_MODULES[module], None)
    caller = hermes_constants.set_hermes_home_override(tmp_path / "caller-override")
    try:
        before, caches, checks = hermes_home_pin.home_state(), _home_caches(), dict(hermes_home_pin.RESTORE_CHECKS)
        name = f"pinned_module_under_test_{len(hermes_home_pin.IMPORT_PINNED)}"
        spec = importlib.util.spec_from_file_location(name, _REPO / module)
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, loaded)
        if import_fails:
            with pytest.raises(ImportError):
                spec.loader.exec_module(loaded)
        else:
            spec.loader.exec_module(loaded)
        assert hermes_home_pin.IMPORT_PINNED[-1] == name  # its import-time pin did run
        assert hermes_home_pin.home_state() == before
        assert _home_caches() == caches
        assert hermes_home_pin.RESTORE_CHECKS == {part: count + 1 for part, count in checks.items()}
    finally:
        hermes_constants.reset_hermes_home_override(caller)
