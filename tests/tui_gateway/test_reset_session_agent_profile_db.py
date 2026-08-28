"""``_reset_session_agent`` must keep a profile session bound to its own db.

App-global remote mode gives a session its own profile (``profile_home`` on
the session dict, see ``session.create``), and that profile keeps its own
``state.db``.  ``/tools enable|disable`` (``tui_gateway/methods_tools.py``
``tools.configure``) rebuilds the session's agent via
``_reset_session_agent`` -> ``_make_agent`` to pick up the new toolset. That
rebuild forwards no ``session_db`` and no ``profile_home``, so
``_make_agent``'s ``session_db=session_db if session_db is not None else
_get_db()`` falls through to the shared LAUNCH handle every time — silently
rebinding a profile session's agent to the wrong database. Turns before the
reset are in the profile db; turns after are in the launch db, under the
SAME session_id, and ``session.history`` (which reads through the
profile-aware ``_session_db(session)``) never sees the post-reset rows again.

This test builds a session exactly like a real profile session would be (an
agent already bound to a real, seeded profile ``state.db``), drives the real
``_reset_session_agent``, and proves both the binding and the write-path
consequence with raw sqlite3 against both database files.
"""

from __future__ import annotations

import importlib
import sqlite3
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

SESSION_ID = "sid-reset-profile"
SESSION_KEY = "tui-reset-profile-1"


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
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
    """The launch profile's state.db, wired in as the ``_get_db()`` handle."""
    db = SessionDB(db_path=hermes_home / "state.db")
    server._db = db
    return db


@pytest.fixture()
def profile_db(tmp_path):
    """A second, non-launch profile's state.db — a real file, real handle."""
    profile_home = tmp_path / "profiles" / "work"
    profile_home.mkdir(parents=True)
    return profile_home, SessionDB(db_path=profile_home / "state.db")


class _FakeAgent:
    """Stand-in for the pre-existing agent a profile session already has.

    Mirrors what a correctly-created profile session's agent carries: its own
    dedicated ``_session_db`` handle, owned by it.
    """

    def __init__(self, session_db):
        self._session_db = session_db
        self._owns_session_db = True
        self.model = "old-model"
        self.provider = "anthropic"
        self.reasoning_config = None
        self.service_tier = None


def _register(server, *, profile_home, old_agent):
    session = {
        "session_key": SESSION_KEY,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": old_agent,
        "attached_images": [],
        "image_counter": 0,
        "cols": 120,
        "profile_home": str(profile_home),
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }
    server._sessions[SESSION_ID] = session
    return session


def _raw_rows(db_path: Path, session_key: str) -> list[tuple]:
    with sqlite3.connect(str(db_path)) as con:
        return con.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_key,),
        ).fetchall()


def test_reset_session_agent_keeps_the_profile_db_binding(
    server, launch_db, profile_db, monkeypatch
):
    profile_home, pdb = profile_db
    pdb.create_session(SESSION_KEY, source="tui")
    pdb.append_message(SESSION_KEY, "user", "pre-reset message")

    old_agent = _FakeAgent(pdb)
    session = _register(server, profile_home=profile_home, old_agent=old_agent)

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
        server._reset_session_agent(SESSION_ID, session)

    new_agent = session["agent"]
    assert built_kwargs.get("session_db") is pdb, (
        "_reset_session_agent rebuilt the agent bound to "
        f"{built_kwargs.get('session_db')!r} instead of the profile db {pdb!r} "
        "-- writes after /tools enable|disable land in the wrong database."
    )

    # Prove the consequence: write through the rebuilt agent's own binding
    # and check with a raw sqlite3 connection against BOTH files.
    new_agent._session_db.append_message(SESSION_KEY, "user", "post-reset message")

    profile_rows = _raw_rows(profile_home / "state.db", SESSION_KEY)
    launch_rows = _raw_rows(Path(launch_db.db_path), SESSION_KEY)

    assert profile_rows == [
        ("user", "pre-reset message"),
        ("user", "post-reset message"),
    ], profile_rows
    assert launch_rows == [], (
        "post-reset message leaked into the launch db under the profile "
        f"session's id: {launch_rows!r}"
    )
