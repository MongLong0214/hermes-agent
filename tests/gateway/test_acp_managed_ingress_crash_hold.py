"""A crash-left marker that startup cannot account for is held, never resumed.

Startup hands every marker it holds (an admitted /acp turn, or one whose identity could not be read)
to ``recover_interrupted_turns`` as ``hold_tokens``. One whose identity is unreadable, whose managed
cleanup failed, or whose chain follow-up could not be accounted for stays exactly as it is for the
next start instead of being promoted to a resume.
"""

from __future__ import annotations

import asyncio
import time

from gateway import acp_managed_ingress as ingress
from tests.gateway.test_acp_managed_ingress import _ledger_rows, _marked_turn, gw as gw_fixture

gw = gw_fixture


def _recover(gw):
    recovered = asyncio.run(gw.runner._recover_unclean_sessions())
    return recovered, [tuple(r) for r in _ledger_rows()]


def test_an_unreadable_marker_store_never_resumes_the_bound_session(gw, monkeypatch):
    _, store = _marked_turn(gw)
    monkeypatch.setattr(type(gw.db), "get_meta", lambda self, key: (_ for _ in ()).throw(OSError("disk")))
    assert _recover(gw) == ((0, 0), [])
    assert not store._entries[gw.entry.session_key].resume_pending


def test_a_failed_managed_marker_cleanup_never_promotes_a_resume(gw, monkeypatch):
    _, store = _marked_turn(gw)

    async def unavailable(*args):
        raise OSError("temporary failure persisting marker cleanup")

    monkeypatch.setattr(gw.runner.async_session_store, "clear_turn_active", unavailable)
    assert _recover(gw) == ((0, 0), [])
    assert not store._entries[gw.entry.session_key].resume_pending


def test_an_unaccounted_reply_keeps_the_marker_for_the_next_start(gw, monkeypatch):
    admission, store = _marked_turn(gw, answer=True)
    token = store._entries[gw.entry.session_key].active_turn_token
    assert ingress.mark_followup_start(admission, token, time.time())
    store.append_to_transcript(gw.entry.session_id, {"role": "user", "content": "ordinary follow-up"})
    store.append_to_transcript(gw.entry.session_id, {"role": "assistant", "content": "ordinary answer"})
    real = gw.runner.async_session_store.load_transcript

    async def unreadable(*a, **k):
        raise OSError("disk")

    monkeypatch.setattr(gw.runner.async_session_store, "load_transcript", unreadable)
    assert _recover(gw) == ((0, 0), [])
    assert store._entries[gw.entry.session_key].active_turn_token
    monkeypatch.setattr(gw.runner.async_session_store, "load_transcript", real)
    assert _recover(gw) == ((0, 1), [("ordinary answer", None)])
