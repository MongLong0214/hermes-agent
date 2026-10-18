"""Collecting or running a module that pins its tmp Hermes home leaves no home behind (#90 review).

The modules below pinned HOME, HERMES_HOME and the home override at import, and put back only the
override, after their own tests. Every other module in the same process then ran its tests under
another module's home (test_canonical_surface_ingress.py, collected first: 36 failures). Their pins
now come from tests/hermes_home_pin.py: one while they import Hermes, one owned by a module fixture
while their tests run, both put back in ``finally``.
"""

import importlib.util
import json
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


# A fresh interpreter forks twice: once after making its import home, once after an import-time pin
# ran (it imported gateway.run under it). Each child exits normally, so it runs every exit hook it
# inherited. Each side reports what it saw; the parent runs its own exit hooks when the script ends.
_FORK_WITNESS = r'''
import json, os, sys
from pathlib import Path
from tests import hermes_home_pin as pin


def forked(child):
    """``child()``'s report from a forked process that then exits normally, and its exit status."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            report = child()
        except BaseException as exc:
            report = {"error": repr(exc)}
        os.write(write_fd, json.dumps(report).encode())
        os.close(write_fd)
        sys.exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        report = json.loads(pipe.read())
    return report, os.waitpid(pid, 0)[1]


parent = pin.import_home()
def child_before_import_pins():
    home = pin.import_home()
    return {"home": str(home), "is_dir": home.is_dir(), "again": str(pin.import_home())}
child, status = forked(child_before_import_pins)
print("FORK-BEFORE-IMPORT-PINS " + json.dumps({
    "parent": str(parent), "child": child, "status": status, "parent_home_survives": parent.is_dir(),
    "parent_returns": str(pin.import_home()), "child_home_left": Path(child.get("home", parent)).exists()}),
    flush=True)

with pin.pinned_hermes_home(pin.import_home(), importer="fork_witness") as import_hermes_home:
    import gateway.run  # its _hermes_home: the import home, from now on
module_root = pin.tmp_root("hermes-fork-witness-")  # as a pinned module's own test home
(import_hermes_home / ".restart_notify.json").write_text("{}")  # only the import home has it
def child_after_import_pins():
    import hermes_constants

    report = {"run_home": str(gateway.run._hermes_home), "home_now": str(hermes_constants.get_hermes_home()),
              "reads_the_import_home_marker": gateway.run._restart_notification_pending()}
    def pin_its_own_root():
        with pin.pinned_hermes_home(module_root):
            pass
    for name, use in (("import_home", pin.import_home), ("pinned_hermes_home", pin_its_own_root)):
        try:
            use()
            report[name] = "allowed"
        except Exception as exc:
            report[name] = type(exc).__name__
    return report
child, status = forked(child_after_import_pins)
print("FORK-AFTER-IMPORT-PINS " + json.dumps({
    "parent": str(parent), "module_root": str(module_root), "child": child, "status": status,
    "parent_home_survives": parent.is_dir(), "module_root_survives": module_root.is_dir()}), flush=True)
'''


def test_a_forked_child_gets_its_own_import_home_and_never_removes_its_parents(tmp_path):
    """#90 review (S15-ISO-R1-01): a forked child got its parent's import home and, on a normal exit,
    removed it under the parent, whose import home then named nothing. Now each process makes and
    removes only its own tmp homes, and a child forked after an import-time pin is refused: what its
    inherited Hermes modules read goes to the parent's import home."""
    home, hermes_home, tmpdir = (tmp_path / name for name in ("home", "hermes", "tmp"))
    for path in (home, hermes_home, tmpdir):
        path.mkdir()
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "HERMES_HOME": str(hermes_home),
           "TMPDIR": str(tmpdir), "PYTHONPATH": str(_REPO), "LANG": "C.UTF-8", "TZ": "UTC"}
    run = subprocess.run([sys.executable, "-c", _FORK_WITNESS], cwd=_REPO, env=env, capture_output=True,
                         text=True, timeout=240)
    out = run.stdout[-3000:] + run.stderr[-3000:]
    reports = {line.split(" ", 1)[0]: json.loads(line.split(" ", 1)[1])
               for line in run.stdout.splitlines() if line.startswith("FORK-")}
    before = reports.get("FORK-BEFORE-IMPORT-PINS")
    assert before is not None, out
    parent, child = before["parent"], before["child"]
    # The parent and the child each have an import home of their own ...
    assert child.get("home") not in (None, parent) and child["is_dir"] and child["again"] == child["home"], out
    # ... the parent's survives the child's exit, and the child removed its own at that exit.
    assert before["status"] == 0 and before["parent_home_survives"] and before["parent_returns"] == parent, out
    assert not before["child_home_left"], out

    after = reports.get("FORK-AFTER-IMPORT-PINS")
    assert run.returncode == 0 and after is not None, out
    child = after["child"]
    # Where the child's reads through an inherited import-time constant go: the parent's import home,
    # while the home it resolves now is the one it started with ...
    assert Path(child["run_home"]).is_relative_to(parent) and child["reads_the_import_home_marker"], out
    assert child["home_now"] == str(hermes_home), out
    # ... so a pin in that child is refused, and nothing it inherited removed the parent's homes.
    assert child["import_home"] == child["pinned_hermes_home"] == "ForkedPinError", out
    assert after["status"] == 0 and after["parent_home_survives"] and after["module_root_survives"], out
    # The parent removed its own at its exit; no tmp home is left behind.
    assert not Path(parent).exists() and not Path(after["module_root"]).exists(), out
    assert not [path.name for path in tmpdir.iterdir() if path.name.startswith("hermes-")], out
