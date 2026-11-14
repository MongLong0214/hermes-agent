"""Canonical peer rows in the session store, and the open user tail. Split out of ``hermes_state_messages``
(FILE_LINES ratchet) and bound onto SessionDB through ``SessionMessagesMixin``; built on its row decoders
and _read_ctx/_read_all primitives.

A peer row keeps its clean body as ``content`` and its provenance in ``display_metadata``. Its admission
lives in ``state_meta`` (``PEER_ROW_LEDGER_PREFIX<id>``), written in the transaction that inserts the row,
beside the receipt the canonical ingress recorded. A row copied from another store is filed under
``PEER_IMPORTED_ROW_LEDGER_PREFIX`` instead: quoted the same way, never this store's own verified ingress.
Every load re-checks a peer row against its ledger and hands the model only its quoted rendering.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.canonical_peer import (
    PEER_DISPLAY_KIND, PEER_IMPORTED_ROW_LEDGER_PREFIX, PEER_METADATA_KEY, PEER_ROW_LEDGER_PREFIX,
    PEER_SIDECAR_VERIFIED_KEY, PeerProvenanceError, admitted_peer, content_digest as peer_content_digest,
    ledger_value as peer_ledger_value, parse_ledger_value as parse_peer_ledger_value, peer_metadata,
    peer_wire_text, sidecar_digest as peer_sidecar_digest)
from agent.message_metadata import lift_data_use
from hermes_state_common import _placeholders, _sql_json_extract
from hermes_state_peer_migrations import LEGACY_PEER_SIDECAR_MARKER

_PARAM_INDEX: Optional[tuple[int, ...]] = None


def _param_index() -> tuple[int, ...]:
    """Positions of role, content, api_content, display_kind and display_metadata in an insert's params."""
    global _PARAM_INDEX
    if _PARAM_INDEX is None:
        from hermes_state_messages import _MESSAGE_WRITE_COLUMNS  # that module binds this mixin
        _PARAM_INDEX = tuple(_MESSAGE_WRITE_COLUMNS.index(column)
                             for column in ("role", "content", "api_content", "display_kind", "display_metadata"))
    return _PARAM_INDEX


def _source_row_id(msg: dict[str, Any]) -> Optional[int]:
    """The durable row id a reinserted *msg* is a copy of (``None`` for a brand-new message)."""
    row_id = msg.get("_row_id")
    return row_id if isinstance(row_id, int) and not isinstance(row_id, bool) else None


class SessionPeerProvenanceMixin:
    """Peer row admission, verification and quoting; the open user tail's answered state."""

    def _peer_reinsertion_sources(self, conn, messages: list[dict[str, Any]], importing: bool,
                                  sources: Optional[dict[int, tuple]]) -> dict[int, tuple]:
        """The validated sources :meth:`_insert_message_rows` checks copies against: the caller's, none
        for an import (another store's ids), else read now (:meth:`_reinsertion_sources`)."""
        if sources is not None:
            return sources
        return {} if importing else self._reinsertion_sources(conn, messages)

    def _peer_ledgers_of(self, rows) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        """``(admitted, imported)`` ledger records of the fetched *rows*, for :meth:`_require_peer_row`."""
        return (self._peer_ledger_rows(rows, PEER_ROW_LEDGER_PREFIX),
                self._peer_ledger_rows(rows, PEER_IMPORTED_ROW_LEDGER_PREFIX))

    def _peer_clone_sources(self, conn, tail_ids: list[int]) -> tuple[dict[int, tuple], int]:
        """Validated classes of the peer rows among *tail_ids* and the id the clones will follow."""
        classes = self._validated_peer_ledger_classes(conn, tail_ids)
        return classes, int(conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0])

    @staticmethod
    def _carry_peer_ledgers(conn, tail_ids: list[int], peer_classes: dict[int, tuple], before: int) -> None:
        """File each cloned peer row's record under its fresh id, in the source's own class. A clone whose
        body no longer matches raises, so the caller's whole publication rolls back rather than leaving a
        peer-marked clone with no record."""
        if not peer_classes:
            return
        # The clones took ids above ``before`` in source-id order (ORDER BY id, one writer).
        clones = conn.execute("SELECT id, content FROM messages WHERE id > ? ORDER BY id", (before,)).fetchall()
        if len(clones) != len(tail_ids):
            raise PeerProvenanceError()
        for source_id, clone in zip(sorted(tail_ids), clones):
            if source_id not in peer_classes:
                continue
            prefix, value = peer_classes[source_id]
            if parse_peer_ledger_value(value)["content_sha256"] != peer_content_digest(clone["content"]):
                raise PeerProvenanceError()
            conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)", (f"{prefix}{int(clone['id'])}", value))

    @classmethod
    def _is_peer_marked(cls, msg: dict[str, Any]) -> bool:
        """The marker test :meth:`_record_admitted_peer_row` applies to the encoded row."""
        meta_json = cls._encode_display_metadata(msg.get("display_metadata"))
        return msg.get("display_kind") == PEER_DISPLAY_KIND or (
            isinstance(meta_json, str) and PEER_METADATA_KEY in meta_json)

    @classmethod
    def _record_admitted_peer_row(cls, conn, row_id: Optional[int], params: tuple, *,
                                  importing: bool = False, source: Optional[tuple] = None) -> None:
        """Record, in the inserting transaction, that row ``row_id`` is an admitted (or imported) peer row.

        A peer-marked row is admitted only while the receipt its metadata names exists in this store;
        anything else raises and the whole insert rolls back. ``importing`` (``import_sessions`` only)
        files it as imported-unverified and never requires or fabricates a receipt. ``source`` is the
        ``(ledger prefix, stored value)`` of the durable row this one copies, as
        :meth:`_reinsertion_sources` validated it: the copy must match it and keeps its class, so an
        imported row stays imported and an admitted one still needs its receipt."""
        role_at, content_at, api_content_at, kind_at, meta_at = _param_index()
        kind, meta_json = params[kind_at], params[meta_at]
        if kind != PEER_DISPLAY_KIND and not (isinstance(meta_json, str) and PEER_METADATA_KEY in meta_json):
            return
        peer = peer_metadata({"role": params[role_at], "display_kind": kind,
                              "display_metadata": cls._decode_display_metadata(meta_json)})
        if peer is None:
            return
        if row_id is None:
            raise PeerProvenanceError()
        source_prefix = None
        if not importing and source is not None:
            source_prefix, source_value = source
            record = parse_peer_ledger_value(source_value)
            if record["peer"] != peer or record["content_sha256"] != peer_content_digest(params[content_at]):
                raise PeerProvenanceError()
        if importing or source_prefix == PEER_IMPORTED_ROW_LEDGER_PREFIX:
            conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                         (f"{PEER_IMPORTED_ROW_LEDGER_PREFIX}{row_id}", peer_ledger_value(peer, params[content_at])))
            return
        if conn.execute("SELECT 1 FROM state_meta WHERE key = ?", (peer["receipt"],)).fetchone() is None:
            raise PeerProvenanceError()
        # A sidecar present at insert time (a replayed or re-inserted turn) is bound now.
        conn.execute("INSERT INTO state_meta (key, value) VALUES (?, ?)",
                     (f"{PEER_ROW_LEDGER_PREFIX}{row_id}",
                      peer_ledger_value(peer, params[content_at], api_content=params[api_content_at])))
        # This store now writes digests itself: the one-time legacy upgrade (which trusts a digest-less
        # sidecar by its prefix) must never run on a row admitted from here on.
        conn.execute("INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, '1')", (LEGACY_PEER_SIDECAR_MARKER,))

    def _reinsertion_sources(self, conn, messages: list[dict[str, Any]]) -> dict[int, tuple]:
        """``{source row id: (ledger prefix, stored value)}`` for the peer-marked *messages* that copy an
        existing durable row, read and validated from the CURRENT rows in the caller's transaction (so it
        runs before anything deletes or rewrites them). A source whose provenance fails raises; a source
        that is no peer row is absent, and its copy is held to the brand-new-insert bar."""
        ids = [row_id for msg in messages
               if (row_id := _source_row_id(msg)) is not None and self._is_peer_marked(msg)]
        return self._validated_peer_ledger_classes(conn, list(dict.fromkeys(ids))) if ids else {}

    @staticmethod
    def _peer_admission_records(conn, values: dict[int, Any]) -> dict[int, dict[str, Any]]:
        """Parsed admission records (row id -> stored value), each with ``receipt_present`` looked up."""
        records = {row_id: parse_peer_ledger_value(value) for row_id, value in values.items()}
        receipts = sorted({record["peer"]["receipt"] for record in records.values()})
        present = {row[0] for row in conn.execute(
            f"SELECT key FROM state_meta WHERE key IN ({_placeholders(receipts)})", tuple(receipts))} if receipts else set()
        for record in records.values():
            record["receipt_present"] = record["peer"]["receipt"] in present
        return records

    def _peer_ledger_rows(self, rows, prefix: str) -> dict[int, Any]:
        """Stored ledger values under *prefix* for the rows among ``rows``: one prefix range scan."""
        row_ids = {row["id"] for row in rows if row["id"] is not None}
        if not row_ids:
            return {}
        values: dict[int, Any] = {}
        with self._read_ctx() as conn:
            for key, value in conn.execute("SELECT key, value FROM state_meta WHERE key >= ? AND key < ?",
                                           (prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1))):
                suffix = key[len(prefix):]
                if suffix.isascii() and suffix.isdigit() and int(suffix) in row_ids:
                    values[int(suffix)] = value
            return self._peer_admission_records(conn, values) if prefix == PEER_ROW_LEDGER_PREFIX else {
                row_id: parse_peer_ledger_value(value) for row_id, value in values.items()}

    @staticmethod
    def _validated_imported_peer(msg: dict[str, Any], record: Optional[dict[str, Any]], raw_content: Any):
        """Like :func:`agent.canonical_peer.admitted_peer` for an imported row: never checks (or claims)
        a receipt, which only the store that admitted the event ever held."""
        peer = peer_metadata(msg)
        if peer is None and record is None:
            return None
        if (peer is None or record is None or record["peer"] != peer
                or record["content_sha256"] != peer_content_digest(raw_content)):
            raise PeerProvenanceError()
        return peer

    @staticmethod
    def _quote_peer(msg: dict[str, Any]) -> None:
        """Replace ``api_content`` with the peer's wire text; a body carrying its own nonce is a refusal."""
        try:
            msg["api_content"] = peer_wire_text(msg)
        except PeerProvenanceError:
            raise
        except ValueError as exc:
            raise PeerProvenanceError() from exc

    @classmethod
    def _require_peer_row(cls, row, msg: dict[str, Any], admitted: dict[int, dict[str, Any]],
                          imported: dict[int, dict[str, Any]]) -> None:
        """Verify a peer-marked loaded row and quote it. An imported row (an imported record and no
        admitted one) never trusts a stored sidecar. An admitted row keeps its stored sidecar only while
        the digest its ledger bound still matches it; otherwise the fresh rendering is sent."""
        if row["id"] in imported and row["id"] not in admitted:
            cls._validated_imported_peer(msg, imported[row["id"]], row["content"])
        else:
            record = admitted.get(row["id"])
            if admitted_peer(msg, record, row["content"]) is None:
                return
            sidecar, bound = msg.get("api_content"), record.get("sidecar_sha256")
            if isinstance(sidecar, str) and sidecar and bound and peer_sidecar_digest(sidecar) == bound:
                msg[PEER_SIDECAR_VERIFIED_KEY] = True
            else:
                msg.pop("api_content", None)
        cls._quote_peer(msg)

    def _validated_peer_ledger_classes(self, conn, row_ids: list[int]) -> dict[int, tuple]:
        """``{row id: (ledger prefix, stored value)}`` for the peer rows among ``row_ids``, each validated
        against its OWN class's ledger in the caller's transaction (imported: no receipt; admitted:
        receipt present). Raises for a peer row in neither ledger, in both, or a record whose row is gone
        or does not match it."""
        ledgers = {prefix: {int(key[len(prefix):]): value for key, value in conn.execute(
            f"SELECT key, value FROM state_meta WHERE key IN ({_placeholders(row_ids)})",
            tuple(f"{prefix}{row_id}" for row_id in row_ids))}
            for prefix in (PEER_ROW_LEDGER_PREFIX, PEER_IMPORTED_ROW_LEDGER_PREFIX)}
        admitted_values, imported_values = ledgers[PEER_ROW_LEDGER_PREFIX], ledgers[PEER_IMPORTED_ROW_LEDGER_PREFIX]
        if set(admitted_values) & set(imported_values):
            raise PeerProvenanceError()
        admitted = self._peer_admission_records(conn, admitted_values)
        imported = {row_id: parse_peer_ledger_value(value) for row_id, value in imported_values.items()}
        classes: dict[int, tuple] = {}
        for row in conn.execute("SELECT id, role, content, display_kind, display_metadata FROM messages "
                                f"WHERE id IN ({_placeholders(row_ids)})", tuple(row_ids)):
            row_id = int(row["id"])
            msg: dict[str, Any] = {"role": row["role"], "display_kind": row["display_kind"]}
            if row["display_metadata"] and (decoded := self._decode_display_metadata(row["display_metadata"])) is not None:
                msg["display_metadata"] = decoded
            if row_id in imported:
                self._validated_imported_peer(msg, imported[row_id], row["content"])
                classes[row_id] = (PEER_IMPORTED_ROW_LEDGER_PREFIX, imported_values[row_id])
            elif admitted_peer(msg, admitted.get(row_id), row["content"]) is not None:
                classes[row_id] = (PEER_ROW_LEDGER_PREFIX, admitted_values[row_id])
        if (set(admitted_values) | set(imported_values)) - set(classes):  # a record whose row is gone
            raise PeerProvenanceError()
        return classes

    def _loaded_row_view(self, row, admitted: dict, imported: dict) -> tuple[Any, Optional[dict[str, Any]], bool]:
        """``(content, display_metadata, peer_marked)`` of a fetched row. A peer body stays byte-exact (it
        reaches a model only quoted); every other body is seen through the loader's lens."""
        meta = self._decode_display_metadata(row["display_metadata"]) if row["display_metadata"] else None
        peer_marked = (row["display_kind"] == PEER_DISPLAY_KIND or row["id"] in admitted or row["id"] in imported
                       or (meta is not None and PEER_METADATA_KEY in meta))
        content = self._decode_content(row["content"])
        return (content if peer_marked else self._loaded_view_content(row["role"], content)), meta, peer_marked

    def _restore_display_fields(self, row, msg: dict[str, Any], meta: Optional[dict[str, Any]], peer_marked: bool,
                                admitted: dict, imported: dict) -> None:
        """Sidecar and display fields of a loaded *msg*; a peer row is then checked against its ledger and quoted."""
        msg.update((col, row[col]) for col in ("api_content", "display_kind") if row[col])
        if meta is not None:
            msg["display_metadata"] = meta
            lift_data_use(msg)
        if peer_marked:
            self._require_peer_row(row, msg, admitted, imported)

    def latest_conversation_role(self, session_id: str) -> Optional[str]:
        """Role of the newest active user/assistant/tool row, or ``None``. ``session_meta`` /
        ``system`` rows are transcript bookkeeping stripped before the model sees history, so
        they must not hide an open user tail from the failed-turn boundary check."""
        row = self._read_one(
            "SELECT role FROM messages WHERE session_id = ? AND active = 1 "
            "AND role NOT IN ('session_meta', 'system') ORDER BY id DESC LIMIT 1", (session_id,))
        return row[0] if row else None

    def open_user_tail_is_delivery_only(self, session_id: str) -> bool:
        """True when the open user tail (active user rows after the newest assistant/tool row) holds only
        recorded delegation deliveries, which ``append_delegation_delivery`` writes between client turns to
        be read with the next one. Such rows are no principal's message, so the history is not unanswered."""
        # ``hidden`` also marks ordinary off-screen user prompts: only a row that carries the delivery's own
        # delegation_id, as append_delegation_delivery writes it, counts.
        rows = self._read_all(
            f"SELECT role, display_kind, {_sql_json_extract('display_metadata', '$.delegation_id')} "
            "FROM messages WHERE session_id = ? AND active = 1 "
            "AND role NOT IN ('session_meta', 'system') AND id > coalesce((SELECT max(id) FROM messages "
            "WHERE session_id = ? AND active = 1 AND role IN ('assistant', 'tool')), 0) ORDER BY id",
            (session_id, session_id))
        return bool(rows) and all(
            role == "user" and kind in ("async_delegation_complete", "hidden")
            and isinstance(delegation_id, str) and bool(delegation_id.strip())
            for role, kind, delegation_id in rows)
