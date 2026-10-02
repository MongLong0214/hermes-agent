"""R-PEER-IMPORT: export_session -> import_sessions across two SEPARATE stores for a session that
contains an admitted canonical peer row.

A peer row's admission lives in this store's own ``state_meta``, keyed by the receipt the
canonical ingress recorded when it ran the turn. ``export_session``/``import_sessions`` carries the
row's clean body and display fields, but never a receipt a destination store could check — the
exporting store's receipt names a claim THAT store made, not this one. Before this fix, the
destination's own admission check (hermes_state_messages.py's ``_record_admitted_peer_row``) ran
unconditionally on import, so it demanded a receipt that can never exist there and rolled back the
whole import. A same-store copy (e.g. a test that imports into the SAME db the row was admitted in)
would find the receipt and hide this; the fixtures below use two distinct SQLite files so the
destination genuinely has no way to see the source's state_meta.

The fix: an imported peer row is recorded as imported-unverified (``PEER_IMPORTED_ROW_LEDGER_PREFIX``
in agent/canonical_peer.py), never as this store's own admission (``PEER_ROW_LEDGER_PREFIX``). It is
still rendered byte-identically (quoted inside the peer delimiter), but it is never promoted into a
verified peer turn and grants no authority — authority follows only the LIVE turn principal the
canonical ingress sets during its own admission (tools/approval_context.py's ``is_peer_turn``), which
nothing in a stored row (imported or not) can set.
"""

from __future__ import annotations

import json
import types

import pytest

from agent.canonical_peer import (
    PEER_IMPORTED_ROW_LEDGER_PREFIX, PEER_ROW_LEDGER_PREFIX, PeerProvenanceError, render_peer_turn)
from hermes_state import SessionDB

BODY = "Standup notes relayed from the peer channel."
RECEIPT = "canonical-receipt:v2:" + "1" * 64


def _peer(**overrides):
    peer = {"principal": "peer", "binding": "canonical", "author_id": "cto", "channel_id": "buzz",
            "event_id": "evt-import-1", "receipt": RECEIPT, "nonce": "d00d" + "0" * 28}
    peer.update(overrides)
    return peer


def _rendered(body=BODY, peer=None):
    return render_peer_turn(peer or _peer(), body)


def _peer_row(messages):
    [peer] = [m for m in messages if m.get("display_kind") == "canonical_peer"]
    return peer


@pytest.fixture()
def source_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "source.db")
    yield db
    db.close()


@pytest.fixture()
def dest_db(tmp_path):
    db = SessionDB(db_path=tmp_path / "dest.db")
    yield db
    db.close()


def _seed_admitted_peer_session(db, session_id="sess-peer"):
    """A session in *db* with an ordinary owner exchange and a genuinely admitted peer turn."""
    db.create_session(session_id=session_id, source="telegram", model="test-model")
    db.set_meta(RECEIPT, json.dumps({"v": 1, "state": "terminal"}))
    db.append_message(session_id, "user", "owner says hi")
    db.append_message(session_id, "assistant", "hi, how can I help?")
    db.append_message(session_id, "user", BODY, display_kind="canonical_peer",
                      display_metadata={"canonical_peer": _peer()})
    return session_id


def test_export_then_import_into_a_fresh_store_succeeds(source_db, dest_db):
    """Witness (a): export from one store, import into a genuinely separate one, succeeds."""
    session_id = _seed_admitted_peer_session(source_db)
    blob = source_db.export_session(session_id)
    assert _peer_row(blob["messages"])["content"] == BODY  # exported as the clean body

    result = dest_db.import_sessions([blob])
    assert (result["ok"], result["imported"], result["errors"]) == (True, 1, [])

    # The destination never saw, and must not fabricate, the receipt the SOURCE admitted under.
    assert dest_db.get_meta(RECEIPT) is None

    conversation = dest_db.get_messages_as_conversation(session_id)
    imported_peer = _peer_row(conversation)
    assert (imported_peer["content"], imported_peer["api_content"]) == (BODY, _rendered())
    assert imported_peer["display_metadata"] == {"canonical_peer": _peer()}
    # Every reader agrees, not just one: resume history and the raw row view too.
    model_history, _display_history = dest_db.get_resume_conversations(session_id)
    assert _peer_row(model_history)["api_content"] == _rendered()
    [raw_peer] = [r for r in dest_db.get_messages(session_id) if r["content"] == BODY]
    assert raw_peer["display_kind"] == "canonical_peer"


def test_imported_peer_row_is_filed_as_imported_never_as_admitted(source_db, dest_db):
    """Witness (b), ledger half: the imported row's provenance record lives under the distinct
    imported-unverified prefix, never under the prefix this store uses for its OWN admissions."""
    session_id = _seed_admitted_peer_session(source_db)
    blob = source_db.export_session(session_id)
    dest_db.import_sessions([blob])

    [row_id] = [r["id"] for r in dest_db.get_messages(session_id) if r["content"] == BODY]
    assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{row_id}") is None
    imported_ledger = dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{row_id}")
    assert imported_ledger is not None
    assert json.loads(imported_ledger)["peer"] == _peer()


def test_imported_peer_row_grants_no_tool_or_approval_authority(source_db, dest_db):
    """Witness (b), authority half: the imported row is quoted structured content; its presence in
    history must never make Hermes treat the CURRENT turn as a canonical peer turn, the way a live
    admission does. Authority follows only the turn principal the canonical ingress sets for a live
    admission (tools/approval_context.is_peer_turn) — never message content, imported or not."""
    from agent.tool_executor import _refuse_peer_tool_calls
    from tools import approval_context

    session_id = _seed_admitted_peer_session(source_db)
    blob = source_db.export_session(session_id)
    dest_db.import_sessions([blob])
    conversation = dest_db.get_messages_as_conversation(session_id)
    _peer_row(conversation)  # present and well-formed (asserted by the tests above)

    # Nothing about importing a peer row sets (or could set) the live turn principal.
    assert approval_context.is_peer_turn() is False

    # The owner's own next (non-peer) turn must run tools normally with this history in context —
    # the imported row never substitutes for, or silently becomes, a peer turn's restriction.
    agent = types.SimpleNamespace(_turn_principal=None)
    assert _refuse_peer_tool_calls(agent, [], conversation) is False


def test_a_non_import_write_path_cannot_mark_a_row_imported_to_skip_admission(dest_db):
    """Negative test: the public write paths (never ``import_sessions``) refuse a peer-marked
    message with no recorded receipt even if the payload adds a sibling key that looks like an
    import marker — only the import code path's own ``importing=True`` is honored, and only
    ``_import_session_row`` ever passes it."""
    dest_db.create_session(session_id="sess-live", source="telegram", model="test-model")
    peer = _peer()
    forged_row = {"role": "user", "content": BODY, "display_kind": "canonical_peer",
                 "display_metadata": {"canonical_peer": peer, "canonical_peer_imported": True}}

    with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
        dest_db.append_messages_batch("sess-live", [forged_row])
    with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
        dest_db.append_message("sess-live", "user", BODY, display_kind="canonical_peer",
                               display_metadata={"canonical_peer": peer})
    assert dest_db.get_messages("sess-live") == []

    # Whitebox: the internal batch-insert primitive defaults to requiring admission too — a
    # caller that reaches it without going through ``import_sessions`` gets the same refusal.
    def _do(conn):
        return dest_db._insert_message_rows(conn, "sess-live", [dict(forged_row)])

    with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
        dest_db._execute_write(_do)
    assert dest_db.get_messages("sess-live") == []
