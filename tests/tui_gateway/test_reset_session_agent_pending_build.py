"""``_reset_session_agent`` must not race — or block behind — a session whose
FIRST agent build is still pending.

``session.create`` registers a fresh profile session with ``agent: None`` and
an unset ``agent_ready`` Event (``_deferred_session_record``), then schedules
the real build ~50ms later on a background thread (``_start_agent_build`` /
its inner ``_build()``, which IS profile-aware — it opens the profile's own
``state.db`` and hands it to the agent it builds, and reads toolset/MCP
config fresh via ``hermes_cli.config.load_config()``, which
``tools.configure``'s ``save_config()`` call already invalidated).

``tools.configure`` (``/tools enable|disable``) does not wait for that build
before calling ``_reset_session_agent``. There is nothing for
``_reset_session_agent`` to do in this window: the pending build will pick up
the just-saved config on its own once it lands, with the correct db. Two
earlier, REJECTED shapes for this function both reproduced or worsened the
bug this ticket exists to fix:

  - doing nothing special: ``getattr(None, "_session_db", None)`` is None, so
    the rebuild fell straight through to ``_make_agent``'s ``_get_db()``
    default — the original data-loss bug, reproduced through a race instead
    of a later trigger.
  - waiting (bounded) for the pending build to land: on the standalone stdio
    TUI, ``tools.configure`` is not in ``_LONG_HANDLERS`` and runs on the
    single stdin/dispatch thread, so the wait could freeze ``/interrupt``,
    status requests, and approval responses for its whole duration — worse
    than the bug being fixed.

The fix instead treats "agent not built yet" as "nothing to reset": it
returns immediately without touching the session (in particular, WITHOUT
popping the per-session runtime pins — see the docstring in
``_reset_session_agent`` for why those are the session's initial build
inputs here, not conversation-boundary residue). A build that already ran
and failed (agent stayed None, ``agent_error`` set, event already fired) is a
distinct, permanent case: retrying it here would mean resolving ``session_db``
blind all over again, so it raises instead of guessing.
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


def _register_pending(server, *, profile_home, **extra):
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
        "cwd": "/tmp",
        "profile_home": str(profile_home),
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "model_override": {"model": "claude-opus-4-6"},
        "create_reasoning_override": {"enabled": True, "effort": "high"},
        "create_service_tier_override": "priority",
        **extra,
    }
    server._sessions[SESSION_ID] = session
    return session


def test_reset_is_a_non_blocking_noop_while_the_first_build_is_pending(
    server, launch_db, profile_db
):
    """No wait, no rebuild, no pin-clearing — and it must return promptly."""
    profile_home, _pdb = profile_db
    session = _register_pending(server, profile_home=profile_home)

    started = time.monotonic()
    info = server._reset_session_agent(SESSION_ID, session)
    elapsed = time.monotonic() - started

    # Non-blocking: nowhere near the old 30s wait bound.
    assert elapsed < 2.0, f"_reset_session_agent blocked for {elapsed:.2f}s"

    # No rebuild happened — the pending build still owns this session's agent.
    assert session["agent"] is None
    assert info is not None

    # The session's initial-build runtime pins are untouched: they are not
    # conversation-boundary residue on a session that has never had a first
    # conversation, they are this session's actual first-build inputs.
    assert session["model_override"] == {"model": "claude-opus-4-6"}
    assert session["create_reasoning_override"] == {"enabled": True, "effort": "high"}
    assert session["create_service_tier_override"] == "priority"


def test_reset_raises_for_a_build_that_already_failed_instead_of_guessing_at_db(
    server, launch_db, profile_db
):
    """A build that already ran and failed must fail loudly, not rebuild blind."""
    profile_home, _pdb = profile_db
    session = _register_pending(server, profile_home=profile_home)
    # Simulate _build()'s except branch: agent stays None, agent_error is
    # set, and ready IS signalled (the build thread finished, just badly).
    session["agent_error"] = "failed to open session db for profile 'work': boom"
    session["agent_ready"].set()

    with pytest.raises(RuntimeError, match="failed to open session db"):
        server._reset_session_agent(SESSION_ID, session)

    assert session["agent"] is None
