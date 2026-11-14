"""Target-bind receipts in ``state_meta``: the lineage-root fence, replay, conflict and the
caller-side validation a later writer runs inside its own transaction. The receipt insert carries the
turn fence itself (``state_meta`` is not a governed table), and receipts a fork 1030 store already holds
survive its migration to 1033 byte for byte.

Fixtures are synthesized here; none of them is a copy of a real store."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

import hermes_state
import hermes_state_fence as fence
from hermes_state import SessionDB
from hermes_state_target_bind import (
    TargetBindReceiptConflictError,
    TargetBindReceiptFenceError,
    _insert_fenced_receipt,
)
from tests.hermes_state.test_state_fence_migration import fork_1030_store

_LINEAGE_DOMAIN = b"hermes.target-bind:lineage-root\0"
_PUBLIC_FIELDS = (
    "domain",
    "version",
    "actor_id",
    "binding_generation",
    "executor_runtime_identity",
    "requested_session_id",
    "lineage_root_digest",
)
_EVIDENCE_FIELDS = ("schema", *_PUBLIC_FIELDS, "receipt_digest")


def _lineage_digest(root_id: str) -> str:
    return "sha256:" + hashlib.sha256(_LINEAGE_DOMAIN + root_id.encode("utf-8")).hexdigest()


def _canonical(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_digest(payload: dict) -> str:
    return "sha256:" + hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    for session_id, parent in (("root", None), ("mid-1", "root"), ("mid-2", "mid-1"), ("tip", "mid-2")):
        store.create_session(session_id, source="cli", parent_session_id=parent)
    store.create_session("other", source="cli")
    yield store
    store.close()


def _receipt_rows(db):
    return [tuple(row) for row in db._conn.execute(
        "SELECT key, value FROM state_meta WHERE key GLOB 'target_bind_receipt:*' ORDER BY key"
    ).fetchall()]


def _raw_receipt_rows(path) -> list:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT key, value FROM state_meta WHERE key GLOB 'target_bind_receipt:*' ORDER BY key"
        ).fetchall()
    finally:
        conn.close()


def _bind(db, session_id="tip", **overrides):
    kwargs = {
        "actor_id": "actor:a",
        "binding_generation": 1,
        "executor_runtime_identity": "runtime:a",
        **overrides,
    }
    return db.prepare_target_bind_receipt(session_id, **kwargs)


def _evidence(record: dict) -> dict:
    return {field: record[field] for field in _EVIDENCE_FIELDS}


def test_bind_commits_to_the_resolved_root_and_replays(db):
    first = _bind(db, expected_lineage_root_digest=_lineage_digest("root"))
    replay = _bind(db, expected_lineage_root_digest=_lineage_digest("root"))

    assert replay == first
    assert first["lineage_root_digest"] == _lineage_digest("root")
    assert first["requested_session_id"] == "tip"
    assert first["receipt_digest"] == _canonical_digest({field: first[field] for field in _PUBLIC_FIELDS})
    rows = _receipt_rows(db)
    assert len(rows) == 1
    # The durable record keeps the private root so a later validation can re-derive it.
    assert json.loads(rows[0][1]) == first
    assert first["lineage_root_id"] == "root"


def test_omitted_expectation_accepts_the_server_resolved_root(db):
    assert _bind(db)["lineage_root_digest"] == _lineage_digest("root")


@pytest.mark.parametrize(
    ("session_id", "expected"),
    [
        ("tip", _lineage_digest("mid-1")),
        ("tip", _lineage_digest("other")),
        ("missing", _lineage_digest("missing")),
    ],
    ids=["ancestor-not-root", "foreign-root", "unknown-session"],
)
def test_expectation_fence_refuses_without_writing(db, session_id, expected):
    with pytest.raises(TargetBindReceiptFenceError):
        _bind(db, session_id, expected_lineage_root_digest=expected)
    assert _receipt_rows(db) == []


@pytest.mark.parametrize("session_id", ["missing", "tip-typo"])
def test_binding_to_a_session_that_does_not_exist_is_refused(db, session_id):
    # No expectation: nothing but the store's own session rows can refuse it.
    with pytest.raises(TargetBindReceiptFenceError, match="session is unavailable"):
        _bind(db, session_id)
    assert _receipt_rows(db) == []


def test_binding_to_a_session_whose_parent_does_not_exist_is_refused(db):
    db._conn.execute("PRAGMA foreign_keys=OFF")
    db._conn.execute("UPDATE sessions SET parent_session_id = 'gone' WHERE id = 'root'")
    db._conn.commit()

    with pytest.raises(TargetBindReceiptFenceError, match="session is unavailable"):
        _bind(db)
    assert _receipt_rows(db) == []


def test_cyclic_lineage_is_fenced(db):
    db._conn.execute("UPDATE sessions SET parent_session_id = 'tip' WHERE id = 'root'")
    db._conn.commit()

    with pytest.raises(TargetBindReceiptFenceError):
        _bind(db)
    assert _receipt_rows(db) == []


def test_unbounded_lineage_is_fenced(db):
    parent = "tip"
    for depth in range(200):
        db.create_session(f"deep-{depth}", source="cli", parent_session_id=parent)
        parent = f"deep-{depth}"

    with pytest.raises(TargetBindReceiptFenceError):
        _bind(db, parent)
    assert _receipt_rows(db) == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"actor_id": ""},
        {"actor_id": " actor:a"},
        {"actor_id": "actor:\x7f"},
        {"executor_runtime_identity": "x" * 513},
        {"binding_generation": True},
        {"binding_generation": -1},
        {"binding_generation": 0},
        # The controller holds the generation as a JS number; past 2**53 - 1 it cannot reproduce the digest.
        {"binding_generation": 2**53},
        {"expected_lineage_root_digest": "sha256:" + "A" * 64},
    ],
    ids=[
        "empty",
        "padded",
        "control",
        "oversized",
        "bool-generation",
        "negative-generation",
        "zero-generation",
        "unsafe-generation",
        "non-canonical-digest",
    ],
)
def test_non_canonical_input_is_rejected_before_any_write(db, overrides):
    with pytest.raises(ValueError):
        _bind(db, **overrides)
    assert _receipt_rows(db) == []


def test_same_identity_on_another_session_conflicts_and_keeps_the_first(db):
    first = _bind(db)
    rows = _receipt_rows(db)

    with pytest.raises(TargetBindReceiptConflictError):
        _bind(db, "other")
    # A sibling in the SAME lineage is still another session: the receipt names the session it bound.
    with pytest.raises(TargetBindReceiptConflictError):
        _bind(db, "mid-2")
    assert _receipt_rows(db) == rows
    assert _bind(db) == first


def test_a_new_generation_is_a_new_receipt(db):
    first = _bind(db)
    second = _bind(db, "other", binding_generation=2)

    assert second["receipt_digest"] != first["receipt_digest"]
    assert len(_receipt_rows(db)) == 2


def test_rebind_with_a_consumed_generation_is_refused_and_only_a_new_generation_rebinds(db):
    first = _bind(db)
    rows = _receipt_rows(db)

    # The controller moves the binding to "other" but sends the generation "tip" already consumed: the
    # stored record at that identity is compared and loses.
    with pytest.raises(TargetBindReceiptConflictError, match="already bound"):
        _bind(db, "other", binding_generation=1)
    assert _receipt_rows(db) == rows

    moved = _bind(db, "other", binding_generation=2)
    with pytest.raises(TargetBindReceiptConflictError):
        _bind(db, "other", binding_generation=1)
    assert _bind(db) == first and _bind(db, "other", binding_generation=2) == moved
    assert len(_receipt_rows(db)) == 2


def test_rebind_after_the_lineage_moved_is_refused_and_keeps_the_stored_receipt(db):
    _bind(db)
    rows = _receipt_rows(db)
    # "mid-2" becomes a root: the same identity on the same session now derives other evidence.
    db._conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE id = 'mid-2'")
    db._conn.commit()

    with pytest.raises(TargetBindReceiptConflictError):
        _bind(db)
    assert _receipt_rows(db) == rows


def test_validation_accepts_only_the_exact_durable_evidence(db):
    evidence = _evidence(_bind(db))

    with db._read_ctx() as conn:
        validated = db._validate_target_bind_receipt_on_conn(conn, "tip", dict(evidence))
    assert validated == evidence
    assert "lineage_root_id" not in validated and "binding_identity" not in validated


_TAMPERS = {
    "extra-private-field": lambda e: {**e, "lineage_root_id": "root"},
    "missing-schema": lambda e: {k: v for k, v in e.items() if k != "schema"},
    "other-schema": lambda e: {**e, "schema": "hermes.other"},
    "other-receipt-digest": lambda e: {**e, "receipt_digest": "sha256:" + "0" * 64},
    "other-lineage-digest": lambda e: {**e, "lineage_root_digest": _lineage_digest("other")},
    "unbound-actor": lambda e: {**e, "actor_id": "actor:b"},
    "bool-generation": lambda e: {**e, "binding_generation": True},
    # ``True == 1`` and ``1.0 == 1``: an equality check alone would accept both as version 1.
    "bool-version": lambda e: {**e, "version": True},
    "float-version": lambda e: {**e, "version": 1.0},
}


@pytest.mark.parametrize("tamper", list(_TAMPERS.values()), ids=list(_TAMPERS))
def test_validation_rejects_tampered_evidence(db, tamper):
    evidence = _evidence(_bind(db))

    with db._read_ctx() as conn, pytest.raises(TargetBindReceiptFenceError):
        db._validate_target_bind_receipt_on_conn(conn, "tip", tamper(evidence))


def test_validation_rejects_evidence_for_another_session(db):
    evidence = _evidence(_bind(db))

    with db._read_ctx() as conn, pytest.raises(TargetBindReceiptFenceError):
        db._validate_target_bind_receipt_on_conn(conn, "other", evidence)


def test_validation_rejects_a_receipt_whose_lineage_moved(db):
    evidence = _evidence(_bind(db))
    # "mid-2" becomes a root, so "tip" now resolves to a lineage the receipt did not commit to.
    db._conn.execute("UPDATE sessions SET parent_session_id = NULL WHERE id = 'mid-2'")
    db._conn.commit()

    with db._read_ctx() as conn, pytest.raises(TargetBindReceiptFenceError):
        db._validate_target_bind_receipt_on_conn(conn, "tip", evidence)


# ── The receipt insert carries the turn fence ──


def _bare(path=":memory:"):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("CREATE TABLE IF NOT EXISTS state_meta (key TEXT PRIMARY KEY, value TEXT)")
    return conn


def test_the_receipt_insert_needs_this_builds_fence_udf():
    conn = _bare()
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function: hermes_turn_fence_generation"):
            _insert_fenced_receipt(conn, "target_bind_receipt:x", "{}")
        conn.create_function(fence.TURN_FENCE_FUNCTION, 0, lambda: fence.FORK_STORED_SCHEMA_VERSION)
        with pytest.raises(sqlite3.IntegrityError, match="state DB generation incompatible"):
            _insert_fenced_receipt(conn, "target_bind_receipt:x", "{}")
        # ``1033.0 = 1033`` in SQL: only the integer itself is this build's generation, as in the trigger.
        for wrong_type in (float(fence.TURN_FENCE_GENERATION), str(fence.TURN_FENCE_GENERATION)):
            conn.create_function(fence.TURN_FENCE_FUNCTION, 0, lambda value=wrong_type: value)
            with pytest.raises(sqlite3.IntegrityError, match="state DB generation incompatible"):
                _insert_fenced_receipt(conn, "target_bind_receipt:x", "{}")
        assert conn.execute("SELECT COUNT(*) FROM state_meta").fetchone()[0] == 0

        fence.register_turn_fence_generation(conn)
        _insert_fenced_receipt(conn, "target_bind_receipt:x", "{}")
        assert conn.execute("SELECT key, value FROM state_meta").fetchall() == [("target_bind_receipt:x", "{}")]
    finally:
        conn.close()


@pytest.fixture(params=["fresh", "migrated-fork"])
def store_path(request, tmp_path):
    path = tmp_path / "home" / "state.db"
    if request.param == "fresh":
        path.parent.mkdir(parents=True)
        seed = SessionDB(db_path=path)
        seed.close()
    else:
        fork_1030_store(path)
        SessionDB(db_path=path).close()
        assert fence.probe_store_lineage(path).settled
    seed = SessionDB(db_path=path)
    try:
        seed.create_session("root", source="cli")
        seed.create_session("tip", source="cli", parent_session_id="root")
    finally:
        seed.close()
    return path


def _open_with_fence_udf(monkeypatch, path, udf):
    def register(conn):
        if udf is not None:
            conn.create_function(fence.TURN_FENCE_FUNCTION, 0, udf, deterministic=True)
    monkeypatch.setattr(hermes_state, "register_turn_fence_generation", register)
    return SessionDB(db_path=path)


def test_a_writer_connection_without_the_fence_udf_cannot_write_a_receipt(monkeypatch, store_path):
    db = _open_with_fence_udf(monkeypatch, store_path, None)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such function: hermes_turn_fence_generation"):
            _bind(db)
    finally:
        db.close()
    assert _raw_receipt_rows(store_path) == []


def test_a_writer_connection_of_another_fence_generation_cannot_write_a_receipt(monkeypatch, store_path):
    db = _open_with_fence_udf(monkeypatch, store_path, lambda: fence.FORK_STORED_SCHEMA_VERSION)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="state DB generation incompatible") as refused:
            _bind(db)
    finally:
        db.close()
    assert _raw_receipt_rows(store_path) == []
    # The same refusal a governed table's fence raises, so doctor and repair read it as one.
    assert fence.fence_refusal_verdict(refused.value, db_path=store_path) is not None


def test_this_builds_writer_writes_a_receipt_on_either_store(store_path):
    db = SessionDB(db_path=store_path)
    try:
        receipt = _bind(db)
    finally:
        db.close()
    assert [key for key, _ in _raw_receipt_rows(store_path)] == [
        "target_bind_receipt:" + _canonical_digest({
            "domain": "hermes.target-bind", "version": 1, "actor_id": "actor:a", "binding_generation": 1,
            "executor_runtime_identity": "runtime:a",
        }).removeprefix("sha256:")
    ]
    assert receipt["lineage_root_digest"] == _lineage_digest("root")


# ── Receipts a fork 1030 store already holds survive the migration to 1033 ──


def _fork_receipt_row(*, session_id, root_id, actor_id, generation, runtime) -> tuple[str, str]:
    """The fork's state_meta row, derived here from the fork's documented layout (not this port)."""
    identity = {
        "domain": "hermes.target-bind",
        "version": 1,
        "actor_id": actor_id,
        "binding_generation": generation,
        "executor_runtime_identity": runtime,
    }
    identity_digest = _canonical_digest(identity)
    public = {**identity, "requested_session_id": session_id, "lineage_root_digest": _lineage_digest(root_id)}
    record = {
        "schema": "hermes.target-bind-receipt",
        **public,
        "lineage_root_id": root_id,
        "binding_identity": identity_digest,
        "receipt_digest": _canonical_digest(public),
    }
    return "target_bind_receipt:" + identity_digest.removeprefix("sha256:"), _canonical(record)


_FORK_RECEIPTS = (
    {"session_id": "s1", "root_id": "s1", "actor_id": "actor:café-\U0001f98a", "generation": 7,
     "runtime": "runtime:a"},
    {"session_id": "c1", "root_id": "s1", "actor_id": "actor:b", "generation": 2**53 - 1,
     "runtime": 'rt:"quoted"\\path'},
)


@pytest.fixture
def fork_store_with_receipts(tmp_path):
    path = tmp_path / "home" / "state.db"
    fork_1030_store(path)
    rows = sorted(_fork_receipt_row(**spec) for spec in _FORK_RECEIPTS)
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        # state_meta is not governed, so the fork's own writer needs no fence UDF for these rows either.
        conn.executemany("INSERT INTO state_meta (key, value) VALUES (?, ?)", rows)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    lineage = fence.probe_store_lineage(path)
    assert (lineage.lineage, lineage.stored, lineage.migratable) == ("fenced", 1030, True)
    return path, rows


def test_fork_receipts_survive_the_1033_migration_unchanged_and_validate(fork_store_with_receipts):
    path, rows = fork_store_with_receipts

    db = SessionDB(db_path=path)
    try:
        assert db._conn.execute("SELECT version FROM schema_version").fetchone()[0] == fence.STORED_SCHEMA_VERSION
        assert _receipt_rows(db) == rows
        for spec, (_key, value) in zip(sorted(_FORK_RECEIPTS, key=lambda s: _fork_receipt_row(**s)), rows):
            stored = json.loads(value)
            with db._read_ctx() as conn:
                validated = db._validate_target_bind_receipt_on_conn(conn, spec["session_id"], _evidence(stored))
            assert validated == _evidence(stored)
            # A replay of the fork's identity returns the fork's record and writes nothing.
            replay = _bind(
                db, spec["session_id"], actor_id=spec["actor_id"], binding_generation=spec["generation"],
                executor_runtime_identity=spec["runtime"],
                expected_lineage_root_digest=_lineage_digest(spec["root_id"]),
            )
            assert replay == stored
        with pytest.raises(TargetBindReceiptConflictError):
            _bind(db, "c1", actor_id="actor:café-\U0001f98a", binding_generation=7,
                  executor_runtime_identity="runtime:a")
        assert _receipt_rows(db) == rows
    finally:
        db.close()
    assert fence.probe_store_lineage(path).settled
    assert _raw_receipt_rows(path) == rows
