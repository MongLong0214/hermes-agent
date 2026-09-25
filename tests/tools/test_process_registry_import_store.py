"""Importing the agent never opens state.db.

The module-level ``process_registry`` is built whenever ``model_tools`` is imported, read-only
commands (``hermes doctor``) included. Restoring the async-delegation ledger in its constructor
migrated a fork-lineage store in place to this build's generation, after which the fork build
refused its own store.
"""

import os
import subprocess
import sys
from pathlib import Path

from tests.hermes_state.fork_store_fixture import build_store, file_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_importing_the_agent_leaves_a_fork_store_untouched(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    db_path = build_store(tmp_path / "hermes_home" / "state.db", "fork29")
    before = file_fingerprint(db_path)

    subprocess.run(
        [sys.executable, "-c", "import run_agent"],
        cwd=REPO_ROOT, env=dict(os.environ, HOME=str(home), HERMES_HOME=str(db_path.parent)),
        stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=180)

    assert file_fingerprint(db_path) == before
