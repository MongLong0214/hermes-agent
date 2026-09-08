"""Fork portability contracts for reusable CI workflow runners."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_UPSTREAM_OWNER = "NousResearch"
_LARGE_RUNNER = "ubuntu-latest-32-core"
_STANDARD_RUNNER = "ubuntu-latest"
_EXPECTED_RUNS_ON = (
    "${{ github.repository_owner == 'NousResearch' && "
    "'ubuntu-latest-32-core' || 'ubuntu-latest' }}"
)
_WORKFLOW_JOBS = {
    ".github/workflows/js-tests.yml": "check",
    ".github/workflows/rust-tests.yml": "bootstrap-installer",
}


def _runner_for_owner(owner: str) -> str:
    """Evaluate the closed owner policy encoded by ``_EXPECTED_RUNS_ON``."""
    return _LARGE_RUNNER if owner == _UPSTREAM_OWNER else _STANDARD_RUNNER


def test_reusable_workflow_runners_are_identical_and_fork_portable():
    runs_on = {
        workflow: yaml.safe_load((_REPO / workflow).read_text(encoding="utf-8"))["jobs"][job]["runs-on"]
        for workflow, job in _WORKFLOW_JOBS.items()
    }

    assert set(runs_on.values()) == {_EXPECTED_RUNS_ON}
    assert _runner_for_owner(_UPSTREAM_OWNER) == _LARGE_RUNNER
    assert _runner_for_owner("MongLong0214") == _STANDARD_RUNNER
