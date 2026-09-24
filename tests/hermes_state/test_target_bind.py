"""Target-bind receipts in ``state_meta``: the lineage-root fence, replay, conflict and the
caller-side validation a later writer runs inside its own transaction."""

from __future__ import annotations

import hashlib
import json

import pytest

from hermes_state import SessionDB
from hermes_state_target_bind import TargetBindReceiptConflictError, TargetBindReceiptFenceError

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


def _canonical_digest(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.fixture
def db(tmp_path):
    store = SessionDB(tmp_path / "state.db")
    for session_id, parent in (("root", None), ("mid-1", "root"), ("mid-2", "mid-1"), ("tip", "mid-2")):
        store.create_session(session_id, source="cli", parent_session_id=parent)
    store.create_session("other", source="cli")
    yield store
    store.close()


def _receipt_rows(db):
    return db._conn.execute(
        "SELECT key, value FROM state_meta WHERE key GLOB 'target_bind_receipt:*'"
    ).fetchall()


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
