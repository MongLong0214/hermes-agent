"""One-time trust upgrade for locally admitted peer context written before sidecar digests."""

from agent.canonical_peer import (
    PEER_ROW_LEDGER_PREFIX, parse_ledger_value, peer_rendering, rebind_sidecar,
)
from hermes_state_fence import _in_write_transaction


LEGACY_PEER_SIDECAR_MARKER = "canonical-peer-sidecar-legacy:v1"


def migrate_legacy_peer_sidecars(store, cursor) -> None:
    """Bind legacy sidecars once, together with the marker under the writer lock.

    Limit: in the local single-user threat model, same-account edits to a legacy
    sidecar BEFORE this upgrade cannot be distinguished from Hermes' own context.
    Only this migration trusts the old exact-rendering prefix rule; loaders still
    require a digest. Imports never carried this store's admission and stay untouched.
    """
    def already_migrated():
        return cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = ?", (LEGACY_PEER_SIDECAR_MARKER,),
        ).fetchone() is not None

    if already_migrated():
        return

    def migrate():
        # Another opener may have finished while we waited for the write lock.
        if already_migrated():
            return
        ledgers = cursor.execute(
            "SELECT key, value FROM state_meta WHERE key >= ? AND key < ?",
            (PEER_ROW_LEDGER_PREFIX,
             PEER_ROW_LEDGER_PREFIX[:-1] + chr(ord(PEER_ROW_LEDGER_PREFIX[-1]) + 1)),
        ).fetchall()
        for key, value in ledgers:
            suffix = key[len(PEER_ROW_LEDGER_PREFIX):]
            if not suffix.isascii() or not suffix.isdigit():
                continue
            row = cursor.execute(
                "SELECT role, content, api_content, display_kind, display_metadata FROM messages WHERE id = ?",
                (suffix,),
            ).fetchone()
            if row is None or not isinstance(row["api_content"], str) or not row["api_content"]:
                continue
            try:
                record = parse_ledger_value(value)
                if record["sidecar_sha256"] is not None:
                    continue
                record["receipt_present"] = cursor.execute(
                    "SELECT 1 FROM state_meta WHERE key = ?", (record["peer"]["receipt"],),
                ).fetchone() is not None
                msg = {"role": row["role"], "display_kind": row["display_kind"],
                       "display_metadata": store._decode_display_metadata(row["display_metadata"])}
                peer = store._validated_peer(msg, record, row["content"])
                rendering = peer_rendering(peer, store._decode_content(row["content"]))
            except ValueError:  # Invalid provenance remains unbound; its loader still refuses it.
                continue
            if row["api_content"].startswith(rendering):
                cursor.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                               (rebind_sidecar(record, row["api_content"]), key))
        # Fresh stores also get the marker, before any later digest-less row can exist.
        cursor.execute("INSERT INTO state_meta (key, value) VALUES (?, '1')",
                       (LEGACY_PEER_SIDECAR_MARKER,))

    _in_write_transaction(cursor, migrate)
