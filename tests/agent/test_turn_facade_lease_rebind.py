"""A turn that waited for the session lease and resumed onto the latest session id must also rebind
the session context, or its MCP calls keep naming the session it was rotated off."""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent import turn_facade_lease


class _LeaseDB:
    """A session store whose lease is held by another process until it compresses the session."""

    def __init__(self, *, waits, latest):
        self.waits = waits
        self.latest = latest
        self.resolved = []

    def get_session(self, session_id):
        return {"id": session_id}

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        if self.waits:
            kwargs["on_wait"](0.0)
        return True

    def resolve_resume_session_id(self, session_id):
        self.resolved.append(session_id)
        return self.latest

    def get_messages_as_conversation(self, session_id, **kwargs):
        return [{"role": "user", "content": f"rows of {session_id}"}]

    def release_session_turn_lease(self, session_id, holder):
        pass


class _Agent:
    _persist_disabled = False
    _interrupt_requested = False

    def __init__(self, db, session_id):
        self._session_db = db
        self.session_id = session_id

    def _emit_status(self, _text):
        pass


@pytest.fixture(autouse=True)
def _no_lease_threads(monkeypatch):
    monkeypatch.setattr(turn_facade_lease.DurableTurnLease, "build_threads", lambda self: None)
    # The rebind also writes the process env (tools read it as a fallback); restore it afterwards.
    monkeypatch.setenv("HERMES_SESSION_ID", "unrelated")


def _admit_then_provenance(db, session_id="parent"):
    """Run one admission on its own turn thread, bound to *session_id* like a real turn, and return
    the session id an MCP call made after admission would carry."""
    from gateway.session_context import set_current_session_id
    from tools.mcp_call_provenance import build_call_provenance

    agent = _Agent(db, session_id)
    task_context = {"platform": "cli", "session_id": session_id}

    def _turn():
        set_current_session_id(session_id)
        admission = turn_facade_lease.admit_durable_turn_lease(
            agent, session_id=session_id, relay_turn_id="t1", task_context=task_context,
            conversation_history=[],
        )
        try:
            return build_call_provenance()["session_id"]
        finally:
            admission.lease.release()

    with ThreadPoolExecutor(max_workers=1) as pool:
        sent = pool.submit(contextvars.copy_context().run, _turn).result()
    return agent, task_context, sent


def test_after_a_resume_rebind_mcp_calls_carry_the_latest_session_id():
    agent, task_context, sent = _admit_then_provenance(_LeaseDB(waits=True, latest="compressed-tip"))
    assert agent.session_id == task_context["session_id"] == "compressed-tip"
    assert sent == "compressed-tip"


@pytest.mark.parametrize("db", [_LeaseDB(waits=False, latest="compressed-tip"), _LeaseDB(waits=True, latest=None)],
                         ids=["no-wait", "no-newer-id"])
def test_without_a_rebind_mcp_calls_keep_the_admitted_session_id(db):
    agent, task_context, sent = _admit_then_provenance(db)
    assert agent.session_id == task_context["session_id"] == "parent"
    assert sent == "parent"
