"""CLI contract for the local target-bind preflight producer."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from hermes_state import SessionDB


_REPO_ROOT = Path(__file__).parents[2]


def _lineage_root_digest(lineage_root: str) -> str:
    payload = b"hermes.target-bind:lineage-root\0" + lineage_root.encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _profile_home(tmp_path: Path) -> Path:
    home = tmp_path / "hermes" / "profiles" / "preflight"
    home.mkdir(parents=True)
    return home


def _request(*, session_id: str, lineage_root: str, **overrides: object) -> dict:
    request = {
        "domain": "hermes.target-bind",
        "version": 1,
        "session_id": session_id,
        "expected_lineage_root_digest": _lineage_root_digest(lineage_root),
        "actor_id": "acp-actor-7",
        "binding_generation": 4,
        "executor_runtime_identity": "executor-runtime-9",
    }
    request.update(overrides)
    return request


def _run_preflight(
    profile_home: Path,
    payload: str,
    argv_prefix: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    controller_home = profile_home.parent.parent / "controller-home"
    controller_home.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(profile_home),
            "HOME": str(controller_home),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            *argv_prefix,
            "target",
            "bind",
            "--json",
        ],
        cwd=_REPO_ROOT,
        input=payload,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=False,
        env=env,
    )


def _encoded(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _assert_closed_error(result: subprocess.CompletedProcess[str], profile_home: Path) -> None:
    assert result.returncode != 0
    error = json.loads(result.stdout)
    assert set(error) == {"error"}
    assert error["error"] in {
        "target_bind_preflight_invalid",
        "target_bind_preflight_conflict",
        "target_bind_preflight_unavailable",
    }
    assert result.stdout == _encoded(error)
    output = result.stdout + result.stderr
    assert str(profile_home) not in output
    assert "Traceback" not in output


def test_target_bind_preflight_cli_returns_a_closed_receipt_and_exact_replay(tmp_path):
    profile_home = _profile_home(tmp_path)
    with SessionDB(profile_home / "state.db") as db:
        db.create_session("lineage-root", source="cli-test")
        db.create_session(
            "lineage-tip", source="cli-test", parent_session_id="lineage-root"
        )

    request = _request(session_id="lineage-tip", lineage_root="lineage-root")
    first = _run_preflight(profile_home, _encoded(request))

    expected = {
        "domain": "hermes.target-bind",
        "version": 1,
        "actor_id": "acp-actor-7",
        "binding_generation": 4,
        "executor_runtime_identity": "executor-runtime-9",
        "requested_session_id": "lineage-tip",
        "lineage_root_digest": _lineage_root_digest("lineage-root"),
    }
    canonical = json.dumps(
        expected, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected["receipt_digest"] = "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert first.returncode == 0, first.stderr
    assert first.stdout == _encoded(expected)
    assert json.loads(first.stdout) == expected
    assert set(json.loads(first.stdout)) == set(expected)

    replay = _run_preflight(profile_home, _encoded(request))
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout == first.stdout

    with SessionDB(profile_home / "state.db") as db:
        assert len(db.list_meta_prefix("target_bind_receipt:")) == 1
        db.create_session("other-root", source="cli-test")
        db.create_session(
            "other-tip", source="cli-test", parent_session_id="other-root"
        )

    conflict = _run_preflight(
        profile_home,
        _encoded(_request(session_id="other-tip", lineage_root="other-root")),
    )
    _assert_closed_error(conflict, profile_home)
    assert json.loads(conflict.stdout) == {
        "error": "target_bind_preflight_conflict"
    }
    with SessionDB(profile_home / "state.db") as db:
        assert len(db.list_meta_prefix("target_bind_receipt:")) == 1


def test_target_bind_preflight_rejects_bad_root_and_closed_request_schema(tmp_path):
    profile_home = _profile_home(tmp_path)
    with SessionDB(profile_home / "state.db") as db:
        db.create_session("lineage-root", source="cli-test")
        db.create_session(
            "lineage-tip", source="cli-test", parent_session_id="lineage-root"
        )

    bad_requests = [
        _request(
            session_id="lineage-tip",
            lineage_root="wrong-root",
        ),
        _request(
            session_id="lineage-tip",
            lineage_root="lineage-root",
            unexpected="closed-schema-violation",
        ),
        _request(
            session_id="lineage-tip",
            lineage_root="lineage-root",
            binding_generation=True,
        ),
        _request(
            session_id="lineage-tip\0not-allowed",
            lineage_root="lineage-root",
        ),
        _request(
            session_id="lineage-tip",
            lineage_root="lineage-root",
            expected_lineage_root_digest="sha256:" + "A" * 64,
        ),
    ]
    for request in bad_requests:
        result = _run_preflight(profile_home, _encoded(request))
        _assert_closed_error(result, profile_home)
        assert json.loads(result.stdout) == {"error": "target_bind_preflight_invalid"}
        with SessionDB(profile_home / "state.db") as db:
            assert db.list_meta_prefix("target_bind_receipt:") == []


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"domain":"hermes.target-bind"}{"version":1}',
        "[]",
    ],
)
def test_target_bind_preflight_rejects_malformed_or_trailing_json(tmp_path, payload):
    profile_home = _profile_home(tmp_path)
    result = _run_preflight(profile_home, payload)
    _assert_closed_error(result, profile_home)
    assert json.loads(result.stdout) == {"error": "target_bind_preflight_invalid"}


@pytest.mark.parametrize(
    "argv_prefix",
    [
        (),
        ("--safe-mode",),
        ("--reasoning", "high"),
        ("--safe-mode", "--reasoning", "high"),
        ("--resume", "sentinel-session"),
        ("-r", "sentinel-session"),
        ("-c", "sentinel-session"),
        ("--continue", "sentinel-session"),
    ],
)
@pytest.mark.parametrize("valid_request", [True, False])
def test_target_bind_preflight_bypasses_external_sources_and_plugin_discovery(
    tmp_path, valid_request, argv_prefix
):
    profile_home = _profile_home(tmp_path)
    with SessionDB(profile_home / "state.db") as db:
        db.create_session("lineage-root", source="cli-test")
        db.create_session(
            "lineage-tip", source="cli-test", parent_session_id="lineage-root"
        )

    secret_marker = tmp_path / "external-secret-ran"
    secret_helper = tmp_path / "external-secret-helper"
    secret_helper.write_text(
        "#!/bin/sh\n"
        f"printf ran > {secret_marker!s}\n"
        "printf 'SENTINEL_API_KEY=not-a-real-credential\\n'\n",
        encoding="utf-8",
    )
    secret_helper.chmod(0o700)

    plugin_marker = tmp_path / "plugin-discovery-ran"
    plugin = profile_home / "plugins" / "target-bind-sentinel"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: target-bind-sentinel\nversion: 0.1.0\ndescription: fixture\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        "from pathlib import Path\n"
        "import sys\n"
        f"Path({str(plugin_marker)!r}).write_text('ran', encoding='utf-8')\n"
        "sys.stdout.write('plugin-discovery-sentinel')\n"
        "def register(ctx):\n"
        "    pass\n",
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        "secrets:\n"
        "  command:\n"
        "    enabled: true\n"
        f"    command: {json.dumps(str(secret_helper))}\n"
        "plugins:\n"
        "  enabled:\n"
        "    - target-bind-sentinel\n",
        encoding="utf-8",
    )

    payload = (
        _encoded(_request(session_id="lineage-tip", lineage_root="lineage-root"))
        if valid_request
        else "{"
    )
    result = _run_preflight(profile_home, payload, argv_prefix)

    assert (secret_marker.exists(), plugin_marker.exists()) == (False, False)
    output = json.loads(result.stdout)
    assert result.stdout == _encoded(output)
    if valid_request:
        assert result.returncode == 0, result.stderr
        assert output["requested_session_id"] == "lineage-tip"
    else:
        _assert_closed_error(result, profile_home)
        assert output == {"error": "target_bind_preflight_invalid"}


def test_target_bind_preflight_refuses_missing_cyclic_and_unavailable_storage(tmp_path):
    profile_home = _profile_home(tmp_path)
    with SessionDB(profile_home / "state.db") as db:
        db.create_session("cycle", source="cli-test")
        db._execute_write(
            lambda conn: conn.execute(
                "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
                ("cycle", "cycle"),
            )
        )

    for request in (
        _request(session_id="missing", lineage_root="missing"),
        _request(session_id="cycle", lineage_root="cycle"),
    ):
        result = _run_preflight(profile_home, _encoded(request))
        _assert_closed_error(result, profile_home)
        assert json.loads(result.stdout) == {"error": "target_bind_preflight_invalid"}
        with SessionDB(profile_home / "state.db") as db:
            assert db.list_meta_prefix("target_bind_receipt:") == []

    unavailable_home = _profile_home(tmp_path / "unavailable")
    (unavailable_home / "state.db").mkdir()
    unavailable = _run_preflight(
        unavailable_home,
        _encoded(_request(session_id="missing", lineage_root="missing")),
    )
    _assert_closed_error(unavailable, unavailable_home)
    assert json.loads(unavailable.stdout) == {
        "error": "target_bind_preflight_unavailable"
    }
