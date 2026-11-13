"""One-time trust upgrade for locally admitted peer context written before sidecar digests."""

from agent.canonical_peer import (
    PEER_ROW_LEDGER_PREFIX, admitted_peer, parse_ledger_value, peer_rendering, rebind_sidecar,
)


LEGACY_PEER_SIDECAR_MARKER = "canonical-peer-sidecar-legacy:v1"
# The admitted-row ledgers sort in [prefix, prefix-with-its-last-char-bumped): one index range scan.
# The imported-row prefix sorts outside it.
_LEDGER_RANGE = (PEER_ROW_LEDGER_PREFIX,
                 PEER_ROW_LEDGER_PREFIX[:-1] + chr(ord(PEER_ROW_LEDGER_PREFIX[-1]) + 1))


def _in_write_transaction(cursor, body):
    # The writer connection is autocommit: BEGIN IMMEDIATE takes the write lock for the whole body.
    if cursor.connection.in_transaction:
        return body()
    cursor.execute("BEGIN IMMEDIATE")
    try:
        result = body()
    except BaseException:
        cursor.execute("ROLLBACK")
        raise
    cursor.execute("COMMIT")
    return result


def migrate_legacy_peer_sidecars(store, cursor) -> None:
    """Bind legacy sidecars once, together with the marker under the writer lock.

    Limit: in the local single-user threat model, same-account edits to a legacy
    sidecar BEFORE this upgrade cannot be distinguished from Hermes' own context.
    Only this migration trusts the old exact-rendering prefix rule; loaders still
    require a digest. Imports never carried this store's admission and stay untouched.

    A store with no admitted peer ledger is only read, never written: there is nothing to
    upgrade, and the writer that admits its first peer row records the marker in that row's
    transaction, so no digest-less row written after this build can ever reach the prefix rule.
    """
    def already_migrated():
        return cursor.execute(
            "SELECT 1 FROM state_meta WHERE key = ?", (LEGACY_PEER_SIDECAR_MARKER,),
        ).fetchone() is not None

    if already_migrated():
        return
    if cursor.execute("SELECT 1 FROM state_meta WHERE key >= ? AND key < ? LIMIT 1",
                      _LEDGER_RANGE).fetchone() is None:
        return

    def migrate():
        # Another opener may have finished while we waited for the write lock.
        if already_migrated():
            return
        ledgers = cursor.execute(
            "SELECT key, value FROM state_meta WHERE key >= ? AND key < ?", _LEDGER_RANGE,
        ).fetchall()
        for key, value in ledgers:
            suffix = key[len(PEER_ROW_LEDGER_PREFIX):]
            if not suffix.isascii() or not suffix.isdigit():
                continue
            row = cursor.execute(
                "SELECT role, content, api_content, display_kind, display_metadata FROM messages WHERE id = ?",
                (int(suffix),),
            ).fetchone()
            if row is None or not isinstance(row[2], str) or not row[2]:
                continue
            role, content, api_content, display_kind, display_metadata = row
            try:
                record = parse_ledger_value(value)
                if record["sidecar_sha256"] is not None:
                    continue
                record["receipt_present"] = cursor.execute(
                    "SELECT 1 FROM state_meta WHERE key = ?", (record["peer"]["receipt"],),
                ).fetchone() is not None
                msg = {"role": role, "display_kind": display_kind,
                       "display_metadata": store._decode_display_metadata(display_metadata)}
                peer = admitted_peer(msg, record, content)
                if peer is None:
                    continue
                rendering = peer_rendering(peer, store._decode_content(content))
            except (ValueError, RecursionError):  # Invalid provenance remains unbound; its loader refuses it.
                continue
            if api_content.startswith(rendering):
                cursor.execute("UPDATE state_meta SET value = ? WHERE key = ?",
                               (rebind_sidecar(record, api_content), key))
        cursor.execute("INSERT INTO state_meta (key, value) VALUES (?, '1')",
                       (LEGACY_PEER_SIDECAR_MARKER,))

    _in_write_transaction(cursor, migrate)
