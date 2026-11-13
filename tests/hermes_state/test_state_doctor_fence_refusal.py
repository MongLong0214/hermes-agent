"""``hermes doctor`` and the offline state.db tools on a store this build does not write as it stands.

The live fork's 1030 store is migrated by this build's next writable SessionDB open. Until then its
fences refuse the doctor's rolled-back probe write, and that refusal used to read as FTS damage:
``doctor --fix`` took a forensic backup, ran every repair strategy on a scratch copy, promoted
nothing and used up a repair attempt. Doctor now decodes the lineage first and reports any store
this build refuses as a check that could not run (``HealthCheckUnavailable``), with the remedy.
A fence's refusal of the probe itself (a probe connection without this build's fence generation)
is classified the same way, while real FTS damage in a migrated 1033 store is still damage.
``sessions set-journal-mode`` and the declarative ``sessions repair`` refuse such stores unchanged.

Fixtures are synthesized here; none of them is a copy of a real store.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sqlite3

import pytest

import hermes_state_fence as fence
import hermes_state_repair as repair
from hermes_state import SessionDB
from tests.hermes_state.test_state_fence_migration import _fence_sql, _rows, _sha, fork_1030_store
from tests.hermes_state.test_state_lineage_admission import (
    _AUTHORITIES, REFUSED, _upstream_store, assert_untouched, make_store, snapshot,
)

# Shapes this build refuses without migrating them: the fork's 1030 before its migration, a fenced
# stamp it never migrates, an unfenced fork store, a newer build's and an unreadable stamp.
_OFFLINE_REFUSED = {
    "fenced_1030": lambda p: make_store(p, journal="wal"),
    "fenced_1031": lambda p: make_store(p, versions=(1031,), fences=1031, journal="wal"),
    "fork_authorities_at_30": lambda p: _upstream_store(p, 30, _AUTHORITIES),
    "fenced_1034": lambda p: make_store(p, versions=(1034,), fences=None, journal="wal"),
    "two_version_rows": lambda p: make_store(p, versions=(30, 31), fences=None, journal="wal"),
}


@pytest.fixture
def fork_store(tmp_path):
    path = tmp_path / "home" / "state.db"
    fork_1030_store(path)
    return path


@pytest.fixture
def migrated(fork_store):
    SessionDB(db_path=fork_store).close()
    assert fence.probe_store_lineage(fork_store).settled
    return fork_store


def _artifacts(path) -> list:
    """Everything beside the store except the WAL pair a read-only open of a WAL store may leave."""
    return sorted(p.name for p in path.parent.iterdir() if not p.name.endswith(("-wal", "-shm")))


def _fences_and_rows(path):
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        return _fence_sql(conn), _rows(conn)
    finally:
        conn.close()


def _spy_repair(monkeypatch) -> list:
    calls = []
    for name in ("repair_state_db_schema", "_backup_db_file", "_copy_database_snapshot",
                 "_run_repair_strategies", "_record_repair_outcome"):
        real = getattr(repair, name)

        def spy(*args, _name=name, _real=real, **kwargs):
            calls.append(_name)
            return _real(*args, **kwargs)

        monkeypatch.setattr(repair, name, spy)
    return calls


def _doctor(monkeypatch, path, *, should_fix):
    from hermes_cli import doctor, doctor_state

    monkeypatch.setattr(doctor, "HERMES_HOME", path.parent)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        finding = doctor_state._check_state_db(should_fix)
    return finding, out.getvalue()


# ── doctor on the fork's 1030 store ──

@pytest.mark.parametrize("held", [False, True], ids=["in_place", "snapshot"])
def test_doctor_write_health_on_a_1030_store_is_not_damage(fork_store, monkeypatch, held):
    from hermes_cli.doctor_state import _write_health_reason

    before = _sha(fork_store)
    # A store a live writer holds is probed on a read-only snapshot copy; the copy is still a 1030 store.
    monkeypatch.setattr(repair, "_live_writer_holds_db", lambda db_path: held)

    reason = _write_health_reason(fork_store, should_fix=False)

    assert isinstance(reason, repair.HealthCheckUnavailable)
    assert "migrates it" in reason and "1030" in reason and "fts5" not in reason
    assert _sha(fork_store) == before
    assert fence.probe_store_lineage(fork_store).migratable


def test_doctor_fix_on_a_1030_store_takes_no_backup_scratch_or_repair_attempt(fork_store, monkeypatch):
    calls = _spy_repair(monkeypatch)
    before, listing = _sha(fork_store), _artifacts(fork_store)

    finding, out = _doctor(monkeypatch, fork_store, should_fix=True)

    assert calls == []
    assert finding.fixed == 0 and finding.manual_issues == []
    assert not [issue for issue in finding.issues if "FTS" in issue or "state.db" in issue]
    assert "write-health probe could not run" in out and "migrates it" in out
    assert "FTS index may be corrupt" not in out
    assert _sha(fork_store) == before and _artifacts(fork_store) == listing
    assert fence.probe_store_lineage(fork_store).migratable


def test_schema_repair_refuses_a_1030_store_before_any_backup_lock_or_attempt(tmp_path):
    db = make_store(tmp_path / "store" / "state.db")
    before = snapshot(db)

    report = repair.repair_state_db_schema(db)

    assert (report["repaired"], report["strategy"], report["backup_path"]) == (False, None, None)
    assert "migrates it" in report["error"]
    assert_untouched(db, before)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_the_probe_reports_every_refused_shape_as_unavailable_without_touching_it(tmp_path, name):
    build, _cause, _lineage = REFUSED[name]
    db = build(tmp_path / "store" / "state.db")
    before = snapshot(db)

    reason = repair._db_opens_cleanly(db)

    assert isinstance(reason, repair.HealthCheckUnavailable)
    assert "no repair was attempted" in reason
    assert_untouched(db, before)


# ── a migrated 1033 store ──

def _corrupt_fts_shadow(path) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute("UPDATE messages_fts_data SET block = X'BADC0FFEE0DDF00D'")  # not governed: no fence
    finally:
        conn.close()


def test_real_fts_damage_on_a_migrated_store_is_still_damage_and_doctor_fix_repairs_it(migrated, monkeypatch):
    fences, rows = _fences_and_rows(migrated)
    _corrupt_fts_shadow(migrated)

    reason = repair._db_opens_cleanly(migrated)
    assert reason is not None and not isinstance(reason, repair.HealthCheckUnavailable)
    assert "fts5" in reason and "turn fences" not in reason

    finding, out = _doctor(monkeypatch, migrated, should_fix=False)
    assert "FTS index may be corrupt" in out
    assert any("FTS write corruption" in issue for issue in finding.issues)

    finding, _ = _doctor(monkeypatch, migrated, should_fix=True)
    assert finding.fixed == 1
    assert repair._db_opens_cleanly(migrated) is None
    assert _fences_and_rows(migrated) == (fences, rows)
    assert fence.probe_store_lineage(migrated).settled


def _unfenced_probe(monkeypatch):
    monkeypatch.setattr(repair, "register_turn_fence_generation", lambda conn: None)


def _other_generation_probe(monkeypatch):
    monkeypatch.setattr(fence, "_turn_fence_generation", lambda: fence.FORK_STORED_SCHEMA_VERSION)


@pytest.mark.parametrize("probe", [_unfenced_probe, _other_generation_probe], ids=["no_udf", "other_generation"])
def test_a_1033_store_refusing_the_probe_write_is_not_damage(migrated, monkeypatch, probe):
    calls = _spy_repair(monkeypatch)
    probe(monkeypatch)
    before, listing = _sha(migrated), _artifacts(migrated)

    reason = repair._db_opens_cleanly(migrated)
    assert isinstance(reason, repair.HealthCheckUnavailable)
    assert "turn fences refused the write probe" in reason and "fts5" not in reason

    finding, out = _doctor(monkeypatch, migrated, should_fix=True)
    assert calls == [] and finding.fixed == 0
    assert "write-health probe could not run" in out
    assert _sha(migrated) == before and _artifacts(migrated) == listing


def _write_error(path, *, generation=None) -> sqlite3.DatabaseError:
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        if generation is not None:
            conn.create_function(fence.TURN_FENCE_FUNCTION, 0, lambda: generation)
        with pytest.raises(sqlite3.DatabaseError) as caught:
            conn.execute("INSERT INTO sessions (id, source, started_at) VALUES ('x', 'cli', 1.0)")
        return caught.value
    finally:
        conn.close()


def test_fence_refusal_verdict_is_typed_and_confirmed_by_the_store(migrated, tmp_path):
    aborted = _write_error(migrated, generation=fence.FORK_STORED_SCHEMA_VERSION)
    missing = _write_error(migrated)
    assert "generation incompatible" in str(aborted) and fence.TURN_FENCE_FUNCTION in str(missing)

    assert fence.fence_refusal_verdict(aborted) is not None
    assert fence.fence_refusal_verdict(missing, db_path=migrated) is not None
    # A missing function is a fence refusal only on a store that carries fences to call it.
    assert fence.fence_refusal_verdict(missing) is None
    upstream = _upstream_store(tmp_path / "upstream" / "state.db", 33)
    assert fence.fence_refusal_verdict(missing, db_path=upstream) is None
    # Through the exception chain, never through prose that quotes the phrase.
    try:
        raise RuntimeError("probe failed") from aborted
    except RuntimeError as chained:
        assert fence.fence_refusal_verdict(chained) is not None
    assert fence.fence_refusal_verdict(RuntimeError(str(aborted))) is None
    assert fence.fence_refusal_verdict(sqlite3.OperationalError("database disk image is malformed")) is None


# ── set-journal-mode and the declarative repair ──

def _journal_args(mode, db):
    return argparse.Namespace(sessions_action="set-journal-mode", mode=mode, db=str(db), force=False)


@pytest.mark.parametrize("name", sorted(_OFFLINE_REFUSED))
def test_set_journal_mode_refuses_a_store_this_build_does_not_open_unchanged(tmp_path, name, capsys):
    from hermes_cli.sessions_cmd import cmd_sessions

    db = _OFFLINE_REFUSED[name](tmp_path / "store" / "state.db")
    assert db.read_bytes()[18:20] == b"\x02\x02"
    before = snapshot(db)

    assert cmd_sessions(_journal_args("delete", db)) == 1

    out = capsys.readouterr().out
    assert "Refusing to change the journal mode" in out and "nothing was changed" in out
    assert_untouched(db, before)


def test_set_journal_mode_still_converts_a_migrated_store(migrated, capsys):
    from hermes_cli.sessions_cmd import cmd_sessions

    fences, rows = _fences_and_rows(migrated)

    assert cmd_sessions(_journal_args("delete", migrated)) == 0

    assert migrated.read_bytes()[18:20] == b"\x01\x01"
    assert _fences_and_rows(migrated) == (fences, rows)
    assert "wal → delete" in capsys.readouterr().out


@pytest.mark.parametrize("name", sorted(_OFFLINE_REFUSED))
def test_declarative_repair_refuses_a_store_this_build_does_not_open_unchanged(tmp_path, name):
    from hermes_state_reconcile import repair_declarative_schema

    db = _OFFLINE_REFUSED[name](tmp_path / "store" / "state.db")
    before = snapshot(db)

    report = repair_declarative_schema(db)

    assert (report["repaired"], report["strategy"], report["backup_path"]) == (False, None, None)
    assert "nothing was changed" in report["error"]
    assert_untouched(db, before)
