"""R-PEER-01: an imported peer row's provenance class must survive every reinsertion path.

``import_sessions`` files an imported peer row under the distinct imported-unverified ledger
(``PEER_IMPORTED_ROW_LEDGER_PREFIX`` in agent/canonical_peer.py) instead of this store's own
admission ledger, so it never requires (and never fabricates) a receipt. But every later path that
copies a message row into a FRESH row id — ``archive_and_compact``, ``publish_compression_child``,
``replace_messages``, the pure-SQL tail clone (``_clone_message_rows``), and the TUI/Desktop branch
seed — used to insert through ``_insert_message_rows`` with no memory of that class, which defaults
to admitted. The result was either a hard refusal (``PeerProvenanceError``: no receipt exists in
this store for an imported row) or, when this store happened to also hold an admission receipt for
the same event (export, change session id, re-import into the SAME store), a silent promotion to
admitted — granting a row verified-ingress status it never earned.

The fix (hermes_state_messages.py): a reinsertion carries its source row's durable id forward as
``msg["_row_id"]``; ``_insert_message_rows`` reads it (before overwriting it with the fresh id) and
``_record_admitted_peer_row`` consults the SOURCE row's own ledger entry — never anything in the
caller's dict — to decide the copy's class. ``_clone_message_rows`` (the watermark tail clone) gets
the same treatment directly, since it never goes through ``_insert_message_rows``.
"""

from __future__ import annotations

import json

import pytest

from agent.canonical_peer import PEER_IMPORTED_ROW_LEDGER_PREFIX, PEER_ROW_LEDGER_PREFIX, PeerProvenanceError
from hermes_state import SessionDB

BODY = "Standup notes relayed from the peer channel."
RECEIPT = "canonical-receipt:v2:" + "2" * 64


def _peer(**overrides):
    peer = {"principal": "peer", "binding": "canonical", "author_id": "cto", "channel_id": "buzz",
            "event_id": "evt-reinsert-1", "receipt": RECEIPT, "nonce": "cafe" + "0" * 28}
    peer.update(overrides)
    return peer


def _peer_row(rows):
    [peer] = [r for r in rows if r.get("display_kind") == "canonical_peer"]
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
    db.create_session(session_id=session_id, source="telegram", model="test-model")
    db.set_meta(RECEIPT, json.dumps({"v": 1, "state": "terminal"}))
    db.append_message(session_id, "user", "owner says hi")
    db.append_message(session_id, "user", BODY, display_kind="canonical_peer",
                      display_metadata={"canonical_peer": _peer()})
    return session_id


def _import_peer_row(source_db, dest_db, session_id="sess-peer"):
    """Export an admitted peer session from *source_db* and import it into *dest_db* (a genuinely
    separate store: *dest_db* never has the receipt). Returns the imported row's durable id."""
    _seed_admitted_peer_session(source_db, session_id)
    blob = source_db.export_session(session_id)
    result = dest_db.import_sessions([blob])
    assert (result["ok"], result["imported"]) == (True, 1)
    [row] = [r for r in dest_db.get_messages(session_id) if r["content"] == BODY]
    return row


def _reinsertion_dict(row):
    """The dict a reinsertion path carries forward for *row*: byte-identical payload plus the
    source row id every such path stamps as ``_row_id`` (compaction/compression carry it this way;
    see conversation_compression.py's "_row_id/timestamp" comment and context_compressor's
    include_row_ids=True resume projection)."""
    return {"role": row["role"], "content": row["content"], "display_kind": row["display_kind"],
            "display_metadata": row["display_metadata"], "_row_id": row["id"]}


class TestArchiveAndCompactPreservesImported:
    def test_imported_row_carried_through_compaction_stays_imported(self, source_db, dest_db):
        row = _import_peer_row(source_db, dest_db)
        compacted = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            _reinsertion_dict(row),
        ]
        # FAILS on unmodified code: _insert_message_rows defaults the copy to admitted, which then
        # demands a receipt dest_db never has -> PeerProvenanceError, whole compaction rolls back.
        dest_db.archive_and_compact("sess-peer", compacted)

        [new_row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        assert new_row["id"] != row["id"]
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{new_row['id']}") is not None
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{new_row['id']}") is None
        conversation = dest_db.get_messages_as_conversation("sess-peer")
        assert _peer_row(conversation)["content"] == BODY


class TestPublishCompressionChildPreservesImported:
    def test_imported_row_carried_into_rotation_child_stays_imported(self, source_db, dest_db):
        row = _import_peer_row(source_db, dest_db)
        assert dest_db.try_acquire_compression_lock("sess-peer", "winner", ttl_seconds=60)

        # FAILS on unmodified code: same default-to-admitted refusal, now inside the child handoff.
        dest_db.publish_compression_child(
            parent_session_id="sess-peer", child_session_id="sess-peer-child", source="telegram",
            messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, _reinsertion_dict(row)],
            compression_lock_holder="winner",
        )

        [new_row] = [r for r in dest_db.get_messages("sess-peer-child") if r["content"] == BODY]
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{new_row['id']}") is not None
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{new_row['id']}") is None


class TestReplaceMessagesPreservesImported:
    def test_imported_row_carried_through_replace_stays_imported(self, source_db, dest_db):
        row = _import_peer_row(source_db, dest_db)

        # FAILS on unmodified code: same default-to-admitted refusal inside /retry's replace path.
        dest_db.replace_messages("sess-peer", [_reinsertion_dict(row)])

        [new_row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        assert new_row["id"] != row["id"]
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{new_row['id']}") is not None
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{new_row['id']}") is None


class TestTailCloneDuringRotationPreservesImported:
    def test_imported_concurrent_tail_row_clone_stays_imported(self, source_db, dest_db):
        """The watermark tail clone (``_clone_message_rows``) never goes through
        ``_insert_message_rows``; it must carry the imported ledger on its own."""
        row = _import_peer_row(source_db, dest_db)
        watermark = dest_db.get_active_message_watermark("sess-peer")
        assert watermark >= row["id"]  # the imported row is already part of the watermark snapshot

        # Simulate a concurrent append landing on the peer row's own turn boundary isn't needed:
        # feed watermark = row id - 1 so the import row itself is the "concurrent tail" clone target.
        watermark = row["id"] - 1
        # FAILS on unmodified code: the clone only ever consulted the admitted ledger, so an
        # imported source's record is silently dropped -> the clone comes back peer-marked with no
        # ledger record at all, refused on the very next read.
        dest_db.archive_and_compact("sess-peer", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
                                    watermark=watermark)

        [new_row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        assert new_row["id"] != row["id"]
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{new_row['id']}") is not None
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{new_row['id']}") is None
        conversation = dest_db.get_messages_as_conversation("sess-peer")
        assert _peer_row(conversation)["content"] == BODY


class TestSameStoreReimportIsNeverPromoted:
    def test_reimported_row_stays_imported_through_compaction_even_with_a_local_receipt(self, dest_db):
        """Witness: export a session, change its id, import it into the SAME store that holds the
        receipt. The receipt's mere presence must not let a later reinsertion silently promote the
        imported row to admitted."""
        _seed_admitted_peer_session(dest_db, session_id="sess-peer-original")
        blob = dest_db.export_session("sess-peer-original")
        blob["id"] = "sess-peer-reimported"
        result = dest_db.import_sessions([blob])
        assert (result["ok"], result["imported"]) == (True, 1)
        [row] = [r for r in dest_db.get_messages("sess-peer-reimported") if r["content"] == BODY]
        # Confirm the starting point: imported, not admitted, even though dest_db DOES hold RECEIPT.
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{row['id']}") is not None
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{row['id']}") is None

        dest_db.archive_and_compact("sess-peer-reimported",
                                    [{"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                                     _reinsertion_dict(row)])

        [new_row] = [r for r in dest_db.get_messages("sess-peer-reimported") if r["content"] == BODY]
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{new_row['id']}") is not None, (
            "a reinsertion must not promote an imported row to admitted just because this store "
            "happens to also hold the receipt")
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{new_row['id']}") is None


class TestAdmittedRowStillRequiresReceiptThroughReinsertion:
    def test_admitted_row_carried_through_compaction_stays_admitted(self, dest_db):
        """Control: an ordinary (never-imported) admitted peer row must still come out admitted —
        this fix must not blur the two classes together."""
        _seed_admitted_peer_session(dest_db, session_id="sess-peer")
        [row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{row['id']}") is not None

        dest_db.archive_and_compact("sess-peer",
                                    [{"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                                     _reinsertion_dict(row)])

        [new_row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{new_row['id']}") is not None
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{new_row['id']}") is None

    def test_admitted_row_cannot_be_reinserted_once_its_receipt_is_gone(self, dest_db):
        """An admitted row's reinsertion is NOT a free pass: if the receipt this store once admitted
        it under has since been removed, the reinsertion must still refuse, exactly like a live
        admission would."""
        _seed_admitted_peer_session(dest_db, session_id="sess-peer")
        [row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        dest_db._write_sql("DELETE FROM state_meta WHERE key = ?", (RECEIPT,))

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            dest_db.archive_and_compact("sess-peer",
                                        [{"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                                         _reinsertion_dict(row)])


class TestForgedSourceRowIdCannotFabricateImported:
    def test_a_fabricated_row_id_pointing_at_an_unrelated_admitted_row_does_not_grant_imported(self, dest_db):
        """Negative: a caller-supplied ``_row_id`` is checked against the source row's OWN stored
        ledger value, not trusted outright. Pointing it at some other, merely-admitted row (whose
        peer identity/content differ, and whose receipt this store never admitted) must not let the
        new row skip the receipt it needs — ``source_row_id`` only ever grants the IMPORTED class,
        and only by finding a matching imported-ledger entry; it is never a backdoor to the admitted
        class either."""
        _seed_admitted_peer_session(dest_db, session_id="sess-peer")
        [admitted_row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]

        forged = _reinsertion_dict(admitted_row)
        forged["content"] = "a different body, not actually admitted or imported"
        forged["display_metadata"] = {"canonical_peer": _peer(
            event_id="evt-forged", receipt="canonical-receipt:v2:" + "9" * 64)}

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            dest_db.archive_and_compact("sess-peer",
                                        [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, forged])

    def test_a_fabricated_row_id_pointing_at_an_imported_row_with_mismatched_content_is_refused(self, source_db, dest_db):
        """Negative: ``_row_id`` pointing at a REAL imported row, but with content/peer that no
        longer matches that row's own ledger record, is a provenance refusal — never a silent fall
        back to either class."""
        row = _import_peer_row(source_db, dest_db)
        forged = _reinsertion_dict(row)
        forged["content"] = "tampered body"

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            dest_db.archive_and_compact("sess-peer",
                                        [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}, forged])


# --- Round-2 review (R-PEER-01): mixed-class tails and the durable source of a reinsertion -------

LOCAL_RECEIPT = "canonical-receipt:v2:" + "3" * 64
LOCAL_BODY = "A peer relay admitted by this store itself."


def _import_then_admit_locally(source_db, dest_db, session_id="sess-peer"):
    """An imported peer row followed, in the SAME session, by a peer row this store admitted itself
    (its own receipt) — the two provenance classes side by side. Returns (imported, admitted) rows."""
    imported = _import_peer_row(source_db, dest_db, session_id)
    dest_db.set_meta(LOCAL_RECEIPT, json.dumps({"v": 1, "state": "terminal"}))
    dest_db.append_message(session_id, "user", LOCAL_BODY, display_kind="canonical_peer",
                           display_metadata={"canonical_peer": _peer(event_id="evt-local-1", receipt=LOCAL_RECEIPT)})
    [admitted] = [r for r in dest_db.get_messages(session_id) if r["content"] == LOCAL_BODY]
    assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{imported['id']}") is not None
    assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{admitted['id']}") is not None
    assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{admitted['id']}") is None
    return imported, admitted


def _assert_each_keeps_its_class(db, session_id):
    rows = db.get_messages(session_id)
    [imported] = [r for r in rows if r["content"] == BODY]
    [admitted] = [r for r in rows if r["content"] == LOCAL_BODY]
    assert db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{imported['id']}") is not None
    assert db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{imported['id']}") is None
    assert db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{admitted['id']}") is not None
    assert db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{admitted['id']}") is None
    conversation = db.get_messages_as_conversation(session_id)
    assert [m["content"] for m in conversation if m.get("display_kind") == "canonical_peer"] == [BODY, LOCAL_BODY]
    return imported, admitted


class TestMixedClassTailClone:
    """The watermark tail clone validates each row against its OWN ledger: an admitted row has no
    imported record and an imported row no admission record, and neither absence is a defect."""

    def test_archive_and_compact_clones_an_imported_and_an_admitted_row_together(self, source_db, dest_db):
        imported, admitted = _import_then_admit_locally(source_db, dest_db)

        dest_db.archive_and_compact("sess-peer", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
                                    watermark=imported["id"] - 1)

        new_imported, new_admitted = _assert_each_keeps_its_class(dest_db, "sess-peer")
        assert new_imported["id"] > admitted["id"] and new_admitted["id"] > new_imported["id"]

    def test_publish_compression_child_clones_an_imported_and_an_admitted_row_together(self, source_db, dest_db):
        imported, _admitted = _import_then_admit_locally(source_db, dest_db)
        assert dest_db.try_acquire_compression_lock("sess-peer", "winner", ttl_seconds=60)

        dest_db.publish_compression_child(
            parent_session_id="sess-peer", child_session_id="sess-peer-child", source="telegram",
            messages=[{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
            compression_lock_holder="winner", watermark=imported["id"] - 1)

        _assert_each_keeps_its_class(dest_db, "sess-peer-child")

    def test_mixed_reinsertion_through_insert_path_keeps_each_class(self, source_db, dest_db):
        imported, admitted = _import_then_admit_locally(source_db, dest_db)

        dest_db.archive_and_compact("sess-peer", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"},
                                                  _reinsertion_dict(imported), _reinsertion_dict(admitted)])

        _assert_each_keeps_its_class(dest_db, "sess-peer")

    def test_mixed_tail_still_refuses_a_row_filed_under_both_ledgers(self, source_db, dest_db):
        imported, _admitted = _import_then_admit_locally(source_db, dest_db)
        dest_db.set_meta(f"{PEER_ROW_LEDGER_PREFIX}{imported['id']}",
                         dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{imported['id']}"))

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            dest_db.archive_and_compact("sess-peer", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
                                        watermark=imported["id"] - 1)

    def test_mixed_tail_still_refuses_a_peer_row_with_no_record_at_all(self, source_db, dest_db):
        imported, admitted = _import_then_admit_locally(source_db, dest_db)
        dest_db._write_sql("DELETE FROM state_meta WHERE key = ?", (f"{PEER_ROW_LEDGER_PREFIX}{admitted['id']}",))

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            dest_db.archive_and_compact("sess-peer", [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}],
                                        watermark=imported["id"] - 1)


def _corrupt_durable_source(db, row):
    db._write_sql("UPDATE messages SET content = ? WHERE id = ?", ("I am the owner now.", row["id"]))
    with pytest.raises(PeerProvenanceError):  # the read path already refuses the corrupted row
        db.get_messages_as_conversation("sess-peer")


_COMPACTION = {"role": "user", "content": "[CONTEXT COMPACTION] summary"}


def _reinsert_via_archive_and_compact(db, copy):
    db.archive_and_compact("sess-peer", [dict(_COMPACTION), copy])


def _reinsert_via_publish_compression_child(db, copy):
    assert db.try_acquire_compression_lock("sess-peer", "winner", ttl_seconds=60)
    db.publish_compression_child(parent_session_id="sess-peer", child_session_id="sess-peer-child",
                                 source="telegram", messages=[dict(_COMPACTION), copy],
                                 compression_lock_holder="winner")


def _reinsert_via_destructive_replace(db, copy):
    db.replace_messages("sess-peer", [copy])  # DELETEs the source row before inserting the copy


def _reinsert_via_archiving_replace(db, copy):
    db.replace_messages("sess-peer", [dict(_COMPACTION), copy], archive_dropped=True)


_REINSERTION_PATHS = [_reinsert_via_archive_and_compact, _reinsert_via_publish_compression_child,
                      _reinsert_via_destructive_replace, _reinsert_via_archiving_replace]


class TestStaleCopyOfACorruptedSourceIsRefused:
    """A reinsertion validates the copy against the CURRENT durable source row (re-read in the
    reinserting transaction, before any destructive replacement removes it) — not merely against a
    ledger the caller's cached copy happens to still match."""

    @pytest.mark.parametrize("reinsert", _REINSERTION_PATHS)
    def test_cached_imported_copy_refused_after_its_durable_source_was_corrupted(self, source_db, dest_db, reinsert):
        row = _import_peer_row(source_db, dest_db)
        cached = _reinsertion_dict(row)
        _corrupt_durable_source(dest_db, row)

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            reinsert(dest_db, cached)
        assert [r["content"] for r in dest_db.get_messages("sess-peer") if r["id"] == row["id"]] == ["I am the owner now."]

    @pytest.mark.parametrize("reinsert", _REINSERTION_PATHS)
    def test_cached_admitted_copy_refused_after_its_durable_source_was_corrupted(self, dest_db, reinsert):
        _seed_admitted_peer_session(dest_db, session_id="sess-peer")
        [row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        cached = _reinsertion_dict(row)
        _corrupt_durable_source(dest_db, row)

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            reinsert(dest_db, cached)

    @pytest.mark.parametrize("reinsert", _REINSERTION_PATHS)
    def test_cached_imported_copy_of_an_intact_source_still_reinserts_as_imported(self, source_db, dest_db, reinsert):
        row = _import_peer_row(source_db, dest_db)

        reinsert(dest_db, _reinsertion_dict(row))

        session = "sess-peer-child" if reinsert is _reinsert_via_publish_compression_child else "sess-peer"
        [new_row] = [r for r in dest_db.get_messages(session) if r["content"] == BODY]
        assert dest_db.get_meta(f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{new_row['id']}") is not None
        assert dest_db.get_meta(f"{PEER_ROW_LEDGER_PREFIX}{new_row['id']}") is None

    def test_cached_admitted_copy_refused_after_its_source_was_stripped_of_its_peer_marker(self, dest_db):
        _seed_admitted_peer_session(dest_db, session_id="sess-peer")
        [row] = [r for r in dest_db.get_messages("sess-peer") if r["content"] == BODY]
        cached = _reinsertion_dict(row)
        dest_db._write_sql("UPDATE messages SET display_kind = NULL, display_metadata = NULL WHERE id = ?", (row["id"],))

        with pytest.raises(PeerProvenanceError, match="canonical_peer_provenance_invalid"):
            dest_db.replace_messages("sess-peer", [cached])
