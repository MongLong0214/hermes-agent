"""``_reset_session_agent`` must not race a session whose FIRST agent build
is still pending.

``session.create`` registers a fresh profile session with ``agent: None`` and
an unset ``agent_ready`` Event (``_deferred_session_record``), then schedules
the real build ~50ms later on a background thread (``_start_agent_build`` /
its inner ``_build()``, which IS profile-aware — it opens the profile's own
``state.db`` and hands it to the agent it builds).

``tools.configure`` (``/tools enable|disable``) does not wait for that build
before calling ``_reset_session_agent``. Before this fix, ``_reset_session_agent``
read ``session["agent"]`` (None), so ``getattr(None, "_session_db", None)`` was
None, and the rebuild fell straight through to ``_make_agent``'s
``_get_db()`` default — the exact data-loss bug, reproduced through a
different trigger (racing the session's very first build instead of a later
one).

The fix waits for that single in-flight build to land (bounded) rather than
racing a second, independent one, then reuses ITS profile-bound agent. A
build that never completes in time must raise rather than silently falling
back to the launch db.
"""

from __future__ import annotations

import importlib
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB

SESSION_ID = "sid-reset-pending-build"
SESSION_KEY = "tui-reset-pending-build-1"


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


def _register_pending(server, *, profile_home):
    """Mirror ``_deferred_session_record``'s shape: no agent yet."""
    session = {
        "session_key": SESSION_KEY,
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": None,
        "agent_ready": threading.Event(),
        "agent_error": None,
        "attached_images": [],
        "image_counter": 0,
        "cols": 120,
        "profile_home": str(profile_home),
        "show_reasoning": False,
        "tool_progress_mode": "all",
    }
    server._sessions[SESSION_ID] = session
    return session


def test_reset_raises_instead_of_falling_back_to_launch_db_when_build_never_lands(
    server, launch_db, profile_db, monkeypatch
):
    """A build that never completes must fail loudly, not silently rebind."""
    profile_home, _pdb = profile_db
    session = _register_pending(server, profile_home=profile_home)
    # Shrink the wait so the test doesn't actually sit for 30s.
    monkeypatch.setattr(server, "_RESET_SESSION_AGENT_BUILD_WAIT_S", 0.05)

    with pytest.raises(RuntimeError, match="agent build timed out"):
        server._reset_session_agent(SESSION_ID, session)

    # No agent was fabricated, and nothing was ever bound to the launch db
    # under this profile session's id.
    assert session["agent"] is None


def test_reset_picks_up_the_profile_db_once_the_pending_build_lands(
    server, launch_db, profile_db, monkeypatch
):
    """The wait must observe a build that finishes mid-wait and reuse ITS db."""
    profile_home, pdb = profile_db
    pdb.create_session(SESSION_KEY, source="tui")
    session = _register_pending(server, profile_home=profile_home)
    monkeypatch.setattr(server, "_RESET_SESSION_AGENT_BUILD_WAIT_S", 5.0)

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

    class _PendingBuildAgent:
        """What _build() would actually hand off: a real, profile-bound,
        dedicated handle it owns."""

        def __init__(self, session_db):
            self._session_db = session_db
            self._owns_session_db = True
            self.model = "already-built-model"
            self.provider = "anthropic"
            self.reasoning_config = None
            self.service_tier = None

    def _land_the_pending_build():
        # Simulate _start_agent_build's _build(): sets session["agent"] THEN
        # signals agent_ready, exactly the order server.py's real _build()
        # uses (current["agent"] = agent precedes ready.set() in its finally).
        time.sleep(0.05)
        session["agent"] = _PendingBuildAgent(pdb)
        session["agent_ready"].set()

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

    builder = threading.Thread(target=_land_the_pending_build, daemon=True)
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
        builder.start()
        server._reset_session_agent(SESSION_ID, session)
    builder.join(timeout=5)

    assert built_kwargs.get("session_db") is pdb, (
        "_reset_session_agent did not reuse the profile db from the build "
        f"that landed mid-wait; got {built_kwargs.get('session_db')!r} instead."
    )
    assert session["agent"]._session_db is pdb
