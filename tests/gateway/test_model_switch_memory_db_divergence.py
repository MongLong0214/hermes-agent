"""/model switches the LIVE agent, the DB refuses, and nobody is told.

WHAT THIS FILE IS ABOUT, AND WHO INTRODUCED IT
    Not an old hole. This one arrived with the fence itself.

    ``gateway/slash_commands.py`` ``_handle_model_command`` commits a switch in
    this order, at both of its commit sites (the inline-keyboard picker
    callback and the typed-text ``_finish_switch``):

        1. ``cached_entry[0].switch_model(...)`` — mutates the LIVE cached
           agent in place. This is the feature: /model is meant to switch a
           running agent.
        2. ``await _sess_db.update_session_model(...)``
        3. the whole of (2) sits inside ``except Exception as exc:`` →
           ``logger.debug("Failed to persist model switch to DB: %s")``

    ``SessionTurnLeaseLostError`` is a ``RuntimeError``
    (``hermes_state.py`` — ``class SessionTurnLeaseLostError(RuntimeError)``),
    so the refusal the previous slice installed lands in that ``except`` and is
    swallowed at DEBUG.

    Net effect while a turn is in flight: **the in-memory agent is switched,
    the DB is not, and the user is told "Model switched to X".** Before the
    guard this was a silent non-fence; after it, a silent refusal. The reply
    the user reads is a success message in both cases, and the two halves of
    the system now disagree about which model the conversation is on.

WHY "MID-TURN" IS THE ORDINARY CASE HERE AND NOT AN EDGE
    ``gateway/run.py`` takes the per-session asyncio turn lease around the
    message-turn region only; the slash-command path does not pass through it,
    and the handlers read ``self._running_agents`` / ``is_running`` precisely
    because a command can arrive while a turn is running. The DB-level turn
    lease that decides admission is taken in ``run_agent.py``
    (``acquire_session_turn_lease``) for the duration of the turn — so the
    conversation is genuinely owned, by THIS process, at the moment /model
    lands.

    That is why both halves below are needed and why one of them is not
    enough:

    OWNED BY US       the turn running in this process holds the grant. The
                      switch is legitimate and must LAND — in memory and in
                      the DB. Closing this with a rollback alone would make
                      /model stop working mid-turn, which is worse than the
                      bug.
    OWNED BY ANOTHER  no grant here authorises the write. The refusal is
                      correct; what is not correct is switching the agent
                      anyway. Presenting a grant cannot help — there is none.

WHAT THE PINS ASSERT
    The DIVERGENCE itself, as two values compared to each other: the model the
    cached agent is now on, and the ``model`` column of the session row. Not
    "an exception was logged" — a refusal that logs and diverges anyway is the
    defect with a log line attached. Each pin also reads the reply the user
    gets, because "the two agree" is satisfiable by a switch that silently
    does nothing, and that is a different bug rather than a fix.

HOW THE FOREIGN OWNER IS BUILT
    By inserting the lease row for a PID that exists and is not us
    (``os.getppid()``), the same construction
    ``tests/state/test_turn_lease_liveness`` uses for
    ``test_a_foreign_owner_is_not_freed_by_its_deadline``. Acquiring in-process
    would not do: acquisition registers the grant in this process's live-grant
    registry, so ``current_turn_grant`` would hand it straight back and the
    conversation would not be foreign at all.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import threading
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: This file, so a mutated extract can run the same checks it defines.
_SELF = pathlib.Path(__file__).resolve().relative_to(REPO_ROOT)

_OLD_MODEL = "old/model-before"
_NEW_MODEL = "new/model-after"


class _CachedAgent:
    """The in-memory half of the divergence, with the real swap contract.

    ``switch_model`` is called by production with exactly these keywords, and
    the real implementation (``agent.agent_runtime_helpers.switch_model``)
    leaves the agent carrying the new identity on these five attributes. A
    rollback has to be able to read the OLD identity off the same attributes,
    so they are here rather than a bare ``model`` string.
    """

    def __init__(self) -> None:
        self.model = _OLD_MODEL
        self.provider = "openrouter"
        self.api_key = "sk-old"
        self.base_url = "https://old.example/api/v1"
        self.api_mode = "chat_completions"
        self.switches: list = []

    def switch_model(
        self, new_model, new_provider, api_key="", base_url="", api_mode=""
    ):
        self.switches.append(new_model)
        self.model = new_model
        self.provider = new_provider
        self.api_key = api_key
        self.base_url = base_url
        self.api_mode = api_mode


def _install_stubs(home):
    """Point the switch pipeline at a temp HERMES_HOME; return an undo callable.

    Plain attribute swaps rather than ``monkeypatch`` so the same function runs
    under pytest and inside the mutation harness's bare subprocess, which has
    no fixtures.
    """
    import agent.models_dev as models_dev
    import gateway.run as gateway_run
    import hermes_cli.config
    import hermes_cli.model_switch as model_switch
    import hermes_constants
    from hermes_cli.model_switch import ModelSwitchResult

    saved = (
        model_switch.switch_model,
        models_dev.fetch_models_dev,
        hermes_constants.get_hermes_home,
        hermes_cli.config.get_hermes_home,
        gateway_run._hermes_home,
    )

    model_switch.switch_model = lambda **kw: ModelSwitchResult(
        success=True,
        new_model=_NEW_MODEL,
        target_provider="openrouter",
        provider_changed=False,
        api_key="sk-new",
        base_url="https://new.example/api/v1",
        api_mode="chat_completions",
        provider_label="OpenRouter",
    )
    models_dev.fetch_models_dev = lambda: {}
    hermes_constants.get_hermes_home = lambda: home
    hermes_cli.config.get_hermes_home = lambda: home
    gateway_run._hermes_home = home

    def _undo():
        (
            model_switch.switch_model,
            models_dev.fetch_models_dev,
            hermes_constants.get_hermes_home,
            hermes_cli.config.get_hermes_home,
            gateway_run._hermes_home,
        ) = saved

    return _undo


def _foreign_owner(db, session_id: str) -> str:
    """A live owner that is NOT this process, written straight into the row.

    ``os.getppid()`` is a PID that exists and is not ours, so liveness cannot
    resolve it as dead and the lease stays held; the row is written through
    ``_execute_write`` so the generation trigger sees a registered connection.
    Nothing registers a grant in this process, which is the whole point —
    ``current_turn_grant`` must answer ``None`` for this conversation.
    """
    pid = os.getppid()
    holder = f"pid={pid}:turn=another-process:platform=test"
    now = time.time()

    def _do(conn):
        conn.execute(
            "INSERT INTO session_turn_leases (conversation_id, holder, "
            "acquired_at, expires_at, epoch, owner_pid, owner_pid_start) "
            "VALUES (?, ?, ?, ?, 1, ?, NULL)",
            (session_id, holder, now, now + 600.0, pid),
        )

    db._execute_write(_do)
    return holder


def _drive_model_command(tmpdir, *, owner: str):
    """Run the real ``/model`` command once; return what memory and the DB say.

    *owner* is ``"us"`` (the turn running in this process holds the DB grant,
    the ordinary mid-turn case) or ``"another-process"``.
    """
    from gateway.config import Platform
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.run import GatewayRunner
    from gateway.session import SessionSource, build_session_key
    from hermes_state import AsyncSessionDB, SessionDB
    import yaml

    tmpdir = pathlib.Path(tmpdir)
    home = tmpdir / "hermes-home"
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump(
            {"model": {"default": _OLD_MODEL, "provider": "openrouter"}}
        ),
        encoding="utf-8",
    )
    undo = _install_stubs(home)

    db = SessionDB(db_path=tmpdir / "state.db")
    try:
        db.create_session("sess-1", "test")
        db.append_message("sess-1", "user", "the turn in flight")
        db.update_session_model("sess-1", _OLD_MODEL)

        if owner == "us":
            grant = db.try_acquire_session_turn_lease(
                "sess-1",
                f"pid={os.getpid()}:turn=live:platform=test",
                ttl_seconds=600,
            )
            assert grant, "could not take the lease this process is meant to hold"
            assert db.current_turn_grant("sess-1") is grant, (
                "the in-process registry does not know about the grant, so the "
                "commit site has nothing to present and this half measures "
                "nothing"
            )
        else:
            _foreign_owner(db, "sess-1")
            assert db.current_turn_grant("sess-1") is None, (
                "this process claims a grant on a conversation it was supposed "
                "to be a bystander on; the foreign-owner half is not foreign"
            )

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="u1",
            chat_id="c1",
            user_name="tester",
            chat_type="dm",
        )
        session_key = build_session_key(source)
        agent = _CachedAgent()

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        runner._voice_mode = {}
        runner._session_model_overrides = {}
        runner._pending_one_turn_model_restores = {}
        runner._pending_model_notes = {}
        runner._running_agents = {}
        runner._agent_cache = {session_key: (agent, 0)}
        runner._agent_cache_lock = threading.Lock()
        runner._session_db = AsyncSessionDB(db)

        class _Entry:
            session_id = "sess-1"
            was_auto_reset = False

        class _Store:
            _store = None

            async def get_or_create_session(self, _source):
                return _Entry()

            async def set_model_override(self, *a, **k):
                return None

        runner.session_store = None
        runner._async_session_store = _Store()

        event = MessageEvent(
            text=f"/model {_NEW_MODEL}",
            message_type=MessageType.TEXT,
            source=source,
        )
        reply = asyncio.run(runner._handle_model_command(event))
        row = db.get_session("sess-1")
        return {
            "reply": reply or "",
            "agent_model": agent.model,
            "db_model": None if row is None else row["model"],
            "switches": list(agent.switches),
        }
    finally:
        undo()
        db.close()


# ---------------------------------------------------------------------------
# The properties.
# ---------------------------------------------------------------------------

def check_a_refused_model_switch_leaves_memory_and_the_db_agreeing(
    tmpdir,
) -> None:
    """Another process owns the turn. The switch must not half-happen.

    The DB write is correctly refused — there is no grant here that authorises
    changing the route out from under somebody else's running turn. What the
    fence cannot decide on its own is what happens to the agent that was
    already switched one statement earlier. Two states, compared to each other:
    if they disagree, the next turn in THIS process runs on a model the session
    row has never heard of.
    """
    outcome = _drive_model_command(tmpdir, owner="another-process")

    assert outcome["agent_model"] == outcome["db_model"], (
        f"the live agent and the session row disagree about the model after a "
        f"refused persist: agent={outcome['agent_model']!r} "
        f"db={outcome['db_model']!r}. The in-place swap already happened and "
        f"the refusal was swallowed at DEBUG, so the next turn in this process "
        f"runs on a route the session row does not carry.\n"
        f"reply to the user: {outcome['reply']!r}"
    )
    assert outcome["db_model"] == _OLD_MODEL, (
        f"a bystander's /model rewrote the route of a conversation another "
        f"process is mid-turn on: {outcome['db_model']!r}"
    )
    assert _NEW_MODEL not in outcome["reply"] or "❌" in outcome["reply"], (
        f"the user was told the switch succeeded while nothing was persisted: "
        f"{outcome['reply']!r}"
    )


def check_the_owning_turns_own_model_switch_lands_in_both(tmpdir) -> None:
    """THIS process owns the turn. /model is the feature; it must work.

    The counterpart, and the reason the pin above cannot be satisfied by
    refusing everybody. ``current_turn_grant`` returns the grant the running
    turn holds, and a commit site that presents it is admitted — so a /model
    typed while the gateway is mid-turn switches both halves, which is what
    the command has always done.
    """
    outcome = _drive_model_command(tmpdir, owner="us")

    assert outcome["agent_model"] == outcome["db_model"], (
        f"the live agent and the session row disagree after the OWNER's own "
        f"switch: agent={outcome['agent_model']!r} db={outcome['db_model']!r}\n"
        f"reply to the user: {outcome['reply']!r}"
    )
    assert outcome["db_model"] == _NEW_MODEL, (
        f"the turn's own /model was refused or rolled back: "
        f"db={outcome['db_model']!r} agent={outcome['agent_model']!r}. The "
        f"conversation is owned by the turn running in THIS process, so the "
        f"commit site has a grant to present.\n"
        f"reply to the user: {outcome['reply']!r}"
    )
    assert _NEW_MODEL in outcome["reply"], (
        f"the switch landed but the user was not told: {outcome['reply']!r}"
    )


PINS = {
    "check_a_refused_model_switch_leaves_memory_and_the_db_agreeing":
        check_a_refused_model_switch_leaves_memory_and_the_db_agreeing,
    "check_the_owning_turns_own_model_switch_lands_in_both":
        check_the_owning_turns_own_model_switch_lands_in_both,
}


@pytest.mark.parametrize("name", sorted(PINS), ids=sorted(PINS))
def test_model_switch_divergence_property(name, tmp_path):
    """The pin. Each property, asserted against the tree under test."""
    PINS[name](tmp_path)
