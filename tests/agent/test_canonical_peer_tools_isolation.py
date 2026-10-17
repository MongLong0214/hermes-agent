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
