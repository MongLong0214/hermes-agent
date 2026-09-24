"""Repair and ``doctor --fix`` never operate on a store this build does not own. Such a store is
healthy for the build that wrote it; the repair ladder (FTS rebuild, dedupe, drop-and-rebuild
with VACUUM) would strip its fences or rewrite it, and even the forensic backup and the attempt
ledger misreport it as damage. The verdict is reported, and nothing is written."""

from __future__ import annotations

import contextlib
import io
import sqlite3

import pytest

from tests.hermes_state.fork_store_fixture import build_store, file_fingerprint, isolate_home

# fork29: accepted lineage whose fences carry another build's generation; stored1031: refused lineage.
KINDS = ("fork29", "stored1031")


def _shape(db):
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        return {
            "fts": sorted(r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE sql LIKE 'CREATE VIRTUAL TABLE%'")),
            "triggers": sorted(r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")),
            "page_count": conn.execute("PRAGMA page_count").fetchone()[0],
        }
    finally:
        conn.close()


def _repair_artifacts(db):
    return sorted(p.name for p in db.parent.iterdir()
                  if ".malformed-backup" in p.name or p.name.endswith(".repair-attempts.json") or "repair-scratch" in p.name)


@pytest.mark.parametrize("kind", KINDS)
def test_repair_refuses_a_store_this_build_does_not_own(tmp_path, monkeypatch, kind):
    db = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", kind)
    from hermes_state_repair import _db_opens_cleanly, repair_state_db_schema
    shape, before = _shape(db), file_fingerprint(db)

    verdict = _db_opens_cleanly(db)
    report = repair_state_db_schema(db)

    assert str(verdict).startswith("schema_incompatible: "), verdict
    assert not report["repaired"] and report["backup_path"] is None
    assert _repair_artifacts(db) == []
    assert _shape(db) == shape
    assert file_fingerprint(db) == before


@pytest.mark.parametrize("kind", KINDS)
def test_doctor_fix_reports_incompatibility_instead_of_repairing(tmp_path, monkeypatch, kind):
    db = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", kind)
    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_health
    shape = _shape(db)

    finding = Finding()
    with contextlib.redirect_stdout(io.StringIO()):
        _state_db_health(finding, True, db, "~/x")

    assert finding.fixed == 0
    assert any("incompatible" in issue for issue in finding.manual_issues + finding.issues), finding
    assert _repair_artifacts(db) == []
    assert _shape(db) == shape
