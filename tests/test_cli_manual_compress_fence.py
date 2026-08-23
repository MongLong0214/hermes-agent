"""Blocker (g) — a turn that cannot hold the fence must fail BEFORE dispatching.

``/compress`` calls the provider, gets a summary back, and only then rotates the
session — and the rotation is where the turn fence is checked. So when another
process owns the conversation, the order of events is:

    summarise (a real provider call, real tokens, real latency)
    → publish_compression_child → SessionTurnLeaseLostError

The user pays for a summary that has nowhere to go. Worse, the failure arrives
at the end of a long operation, which is exactly where a caller is most likely
to have a fallback that writes something else instead.

WHY THE ASSERTION IS A COUNT AND NOT AN ABSENCE OF AN ERROR
    "It raised" and "it did no work" are different claims, and only the second
    one is the property. A path that dispatches, spends the tokens, and then
    raises passes every test written as ``pytest.raises`` — the frozen candidate
    passed 108 of 108 with this defect live. So the agent double here counts its
    ``_compress_context`` invocations and the assertion is ``== 0``.

THE OTHER DIRECTION, WHICH IS THE EASY WAY TO GET THIS WRONG
    Acquiring before dispatch deadlocks the moment the caller is already inside
    a leased turn: the thing it would wait for is itself. So the fence has to
    REUSE this process's existing grant when there is one
    (``SessionDB.current_turn_grant``), and only acquire when there is not.
    ``test_a_compress_inside_a_turn_this_process_owns_still_runs`` is that half,
    and without it the fix is a deadlock rather than a fence.
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import pytest

from cli import HermesCLI
from hermes_state import SessionDB


class CountingAgent:
    """Records every dispatch. The count is the assertion."""

    def __init__(self, session_id="s"):
        self.compression_enabled = True
        self._cached_system_prompt = "SYS"
        self.session_id = session_id
        self.provider_calls = 0
        self.flush_calls = []
        self.context_compressor = None

    def _flush_messages_to_session_db(self, messages, _session_id=None):
        self.flush_calls.append((list(messages), _session_id))

    def _compress_context(self, messages, system_message, **kwargs):
        self.provider_calls += 1
        return (
            [{"role": "user", "content": "[CONTEXT SUMMARY]: compacted"}],
            "new system prompt",
        )


def _cli_with(db, session_id="s"):
    cli = HermesCLI.__new__(HermesCLI)
    cli.conversation_history = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
    ]
    cli.agent = CountingAgent(session_id=session_id)
    cli.session_id = session_id
    cli._pending_title = None
    cli._session_db = db
    cli._busy_command = lambda _message, **_kwargs: nullcontext()
    return cli


def _seeded_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s", source="test")
    db.append_message("s", "user", "one")
    db.append_message("s", "assistant", "two")
    return db


def _foreign_owner(db, conversation_id="s"):
    """A lease row owned by a live process that is not this one.

    ``os.getppid()`` exists and is not us, so ``_turn_lease_owner_is_dead``
    answers 'not dead' — the same shape as a second Hermes mid-turn. Seeded
    through the row rather than a second interpreter because what is under test
    is the CLI's ordering, not cross-process exclusion (which
    tests/state/test_turn_lease_process_barriers.py owns).
    """
    foreign_pid = os.getppid()

    def _do(conn):
        conn.execute(
            "INSERT INTO session_turn_leases (conversation_id, holder, "
            "acquired_at, expires_at, epoch, owner_pid, owner_pid_start) "
            "VALUES (?, ?, 0, 0, 3, ?, NULL)",
            (conversation_id, f"pid={foreign_pid}:turn=foreign:platform=test",
             foreign_pid),
        )

    db._execute_write(_do)


def test_manual_compress_dispatches_nothing_when_the_fence_cannot_be_held(
    tmp_path, capsys
):
    """Zero provider calls. Not 'it raised' — zero."""
    db = _seeded_db(tmp_path)
    _foreign_owner(db)
    cli = _cli_with(db)

    cli._manual_compress("/compress")

    assert cli.agent.provider_calls == 0, (
        f"/compress dispatched {cli.agent.provider_calls} provider call(s) "
        f"while another process owned the conversation. The summary it paid "
        f"for cannot be published — the rotation is fenced — so the work is "
        f"spent to produce a value that is thrown away."
    )
    assert cli.agent.flush_calls == [], (
        "nothing may be persisted on a path that never held the fence"
    )
    out = capsys.readouterr().out
    assert "compress" in out.lower() or "lease" in out.lower() or out.strip(), (
        "the user was told nothing at all about why /compress did nothing"
    )
    db.close()


def test_manual_compress_leaves_the_history_alone_when_it_refuses(tmp_path):
    """A refusal must not be indistinguishable from a compression.

    ``conversation_history`` is the CLI's input and its retry material. A path
    that refuses after replacing it has destroyed the thing the user would
    retry with.
    """
    db = _seeded_db(tmp_path)
    _foreign_owner(db)
    cli = _cli_with(db)
    before = [dict(m) for m in cli.conversation_history]

    cli._manual_compress("/compress")

    assert cli.conversation_history == before, (
        "the refusing path rewrote the conversation history it never got to "
        "compress"
    )
    assert cli.session_id == "s", "the refusing path rotated the session id"
    db.close()


def test_a_compress_inside_a_turn_this_process_owns_still_runs(tmp_path):
    """The fence must not deadlock against a lease this process already holds.

    Acquiring here would wait out the full budget and then refuse, on every
    real in-turn invocation, because the holder it is waiting for is itself.
    """
    db = _seeded_db(tmp_path)
    grant = db.try_acquire_session_turn_lease(
        "s", f"pid={os.getpid()}:turn=live:platform=test", ttl_seconds=300
    )
    assert grant is not None
    cli = _cli_with(db)

    cli._manual_compress("/compress")

    assert cli.agent.provider_calls == 1, (
        "a /compress issued inside a turn THIS process owns was refused or "
        "deadlocked against its own grant"
    )
    db.close()


def test_manual_compress_still_runs_when_nothing_owns_the_conversation(tmp_path):
    """The ordinary case, so the fence cannot pass by refusing everything."""
    db = _seeded_db(tmp_path)
    cli = _cli_with(db)

    cli._manual_compress("/compress")

    assert cli.agent.provider_calls == 1, (
        "/compress refused to run on an unowned conversation"
    )
    db.close()


@pytest.mark.parametrize("attr", ["_session_db"])
def test_manual_compress_without_a_session_db_still_runs(tmp_path, attr):
    """No store, no fence to hold, and no reason to refuse the user's command.

    An embedded/scaffold CLI with no durable session must not lose /compress:
    with nothing persisted there is no transcript for a second writer to
    corrupt.
    """
    cli = _cli_with(None)
    setattr(cli, attr, None)

    cli._manual_compress("/compress")

    assert cli.agent.provider_calls == 1
