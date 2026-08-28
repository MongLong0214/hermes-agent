"""``_sync_bot_capabilities`` must keep a profile Bot Chat bound to its own db.

Same defect class and same fix as ``_reset_session_agent``'s profile-db bug
(see ``tests/tui_gateway/test_reset_session_agent_profile_db.py``), reached
through a different trigger: a Bot Chat's capability surface changing
(Settings -> Capabilities, skill install, MCP toggle) rebuilds its agent via
``_make_agent`` at turn start WITHOUT forwarding ``session_db``, so a
profile-scoped Bot Chat silently rebinds to the shared launch db the moment
its capability fingerprint changes.

This drives the real ``_sync_bot_capabilities`` — only ``run_agent.AIAgent``
and runtime-resolution config are mocked, never the db-selection code path —
and proves the write-path consequence with raw sqlite3 against both files.
"""

from __future__ import annotations

import importlib
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

SESSION_ID = "sid-bot-caps-profile"
SESSION_KEY = "tui-bot-caps-profile-1"


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None


@pytest.fixture()
def launch_db(server, hermes_home):
    db = SessionDB(db_path=hermes_home / "state.db")
    server._db = db
    return db


@pytest.fixture()
def profile_db(tmp_path):
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    return profile_home, SessionDB(db_path=profile_home / "state.db")


class _BotAgent:
    """Stand-in for a live Bot Chat agent already bound to its own db."""

    def __init__(self, session_db):
        self._session_db = session_db
        self._owns_session_db = True
        self._session_title_hint = "Bot Chat"
        self.model = "old-model"
        self.provider = "anthropic"
        self.reasoning_config = None
        self.service_tier = None


def _register(server, *, profile_home, agent):
    session = {
        "session_key": SESSION_KEY,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "image_counter": 0,
        "cols": 120,
        "profile_home": str(profile_home),
        "show_reasoning": False,
        "tool_progress_mode": "all",
        # Establish a baseline different from the fingerprint the test
        # forces, so the change is detected and the rebuild actually runs.
        "bot_caps_seen": "old-fingerprint",
    }
    server._sessions[SESSION_ID] = session
    return session


def _raw_rows(db_path: Path, session_key: str) -> list[tuple]:
    with sqlite3.connect(str(db_path)) as con:
        return con.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_key,),
        ).fetchall()


def test_sync_bot_capabilities_keeps_the_profile_db_binding(
    server, launch_db, profile_db, monkeypatch
):
    profile_home, pdb = profile_db
    pdb.create_session(SESSION_KEY, source="tui")
    pdb.append_message(SESSION_KEY, "user", "pre-rebuild message")

    old_agent = _BotAgent(pdb)
    session = _register(server, profile_home=profile_home, agent=old_agent)

    monkeypatch.setattr(
        "tools.bot_mode_probe.capability_fingerprint",
        lambda home=None: "new-fingerprint",
    )

    fake_cfg = {"agent": {"system_prompt": ""}, "model": {"default": "unused"}}
    fake_runtime = {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key": "sk-test",
        "api_mode": "anthropic_messages",
        "command": None,
        "args": None,
        "credential_pool": None,
    }

    built_kwargs: dict = {}

    class _NewFakeAgent:
        def __init__(self, **kwargs):
            built_kwargs.update(kwargs)
            self.model = kwargs.get("model")
            self.provider = kwargs.get("provider")
            self.reasoning_config = kwargs.get("reasoning_config")
            self.service_tier = kwargs.get("service_tier")
            self._session_db = kwargs.get("session_db")
            self._owns_session_db = False

    with (
        patch("tui_gateway.server._load_cfg", return_value=fake_cfg),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch("tui_gateway.server._load_enabled_toolsets", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=fake_runtime,
        ),
        patch("run_agent.AIAgent", _NewFakeAgent),
    ):
        server._sync_bot_capabilities(SESSION_ID, session)

    assert session["agent"] is not old_agent, "capability sync did not rebuild the agent"
    new_agent = session["agent"]
    assert built_kwargs.get("session_db") is pdb, (
        "_sync_bot_capabilities rebuilt the agent bound to "
        f"{built_kwargs.get('session_db')!r} instead of the profile db {pdb!r} "
        "-- a capability/skill/MCP change silently rebinds a profile Bot Chat "
        "to the wrong database."
    )
    # The old agent no longer claims ownership of a handle the new agent is
    # now using — a real transfer, not a second claim (see BLOCKER 2 in
    # _reset_session_agent's own history).
    assert old_agent._owns_session_db is False

    # Prove the consequence: write through the rebuilt agent's own binding
    # and check with a raw sqlite3 connection against BOTH files.
    new_agent._session_db.append_message(SESSION_KEY, "user", "post-rebuild message")

    profile_rows = _raw_rows(profile_home / "state.db", SESSION_KEY)
    launch_rows = _raw_rows(Path(launch_db.db_path), SESSION_KEY)

    assert profile_rows == [
        ("user", "pre-rebuild message"),
        ("user", "post-rebuild message"),
    ], profile_rows
    assert launch_rows == [], (
        "post-rebuild message leaked into the launch db under the profile "
        f"Bot Chat's session id: {launch_rows!r}"
    )
