"""A store whose lineage this build does not own is refused by every state.db opener, and the
refused open leaves the file exactly as it was: bytes, mtime, journal mode, no new sidecars,
sqlite_master and the schema cookie. The refusal must happen before SCHEMA_SQL, reconcile or a
journal-mode switch, because DDL is never fenced."""

from __future__ import annotations

import pytest

from tests.hermes_state.fork_store_fixture import REFUSED_KINDS, build_store, file_fingerprint, isolate_home


def _sessiondb_writer(db):
    from hermes_state import SessionDB

    SessionDB(db_path=db).close()


def _sessiondb_read_only(db):
    from hermes_state import SessionDB

    SessionDB(db_path=db, read_only=True).close()


def _async_delegation(db):
    from tools import async_delegation

    async_delegation._connect().close()


def _delivery_ledger(db):
    from gateway import delivery_ledger

    delivery_ledger._connect().close()


OPENERS = {
    "sessiondb_writer": _sessiondb_writer,
    "sessiondb_read_only": _sessiondb_read_only,
    "async_delegation": _async_delegation,
    "delivery_ledger": _delivery_ledger,
}


@pytest.mark.parametrize("kind", sorted(REFUSED_KINDS))
@pytest.mark.parametrize("opener", sorted(OPENERS))
def test_refused_store_is_left_untouched(tmp_path, monkeypatch, kind, opener):
    hermes_home = isolate_home(tmp_path, monkeypatch)
    db = build_store(hermes_home / "state.db", kind)
    before = file_fingerprint(db)

    with pytest.raises(Exception) as refused:
        OPENERS[opener](db)

    assert file_fingerprint(db) == before
    assert type(refused.value).__name__ == "IncompatibleSchemaError", refused.value
    assert refused.value.cause == REFUSED_KINDS[kind]


# Unfenced refused kinds: a raw writer without the probe would commit into them.
@pytest.mark.parametrize("kind", ["two_row", "text_scalar", "upstream31"])
def test_a2a_forward_does_not_write_a_refused_store(tmp_path, monkeypatch, kind):
    isolate_home(tmp_path, monkeypatch)
    from plugins.platforms.a2a import adapter

    db = build_store(tmp_path / "peer" / "state.db", kind)
    monkeypatch.setattr(adapter, "_profile_home", lambda profile: str(db.parent))
    before = file_fingerprint(db)

    result = adapter._state_db("peer", "UPDATE sessions SET title = ? WHERE id = ?", ("forwarded", "r-alpha"),
                               "refused", commit=True)

    assert result == ""
    assert file_fingerprint(db) == before
