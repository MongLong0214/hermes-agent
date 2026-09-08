"""Runtime FTS-corruption self-heal on the SessionDB write path (#65637 class).

A corrupted FTS5 shadow table (``messages_fts_data``) makes every message
write raise ``sqlite3.DatabaseError: database disk image is malformed``
through the FTS sync triggers, while the canonical ``messages`` rows stay
intact. Before this fix the gateway swallowed the failure at debug level and
the in-memory session advanced while disk silently fell behind — surfacing
later as "Persisted transcript lagged live cached history" amnesia.

The fix: ``_execute_write`` first attempts a one-shot in-place FTS rebuild.
If corruption persists, it records a durable stale marker, detaches the FTS
sync triggers, and retries the canonical write. Search degrades to ``LIKE``
until a later open atomically rebuilds the index and restores the triggers.
"""

import os
import sqlite3
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import (
    FTS_STALE_KEY,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    SessionDB,
    _FTS_TRIGGERS,
)


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _corrupt_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _corrupt_trigram_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_trigram_data "
        "SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _message_contents(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute("SELECT content FROM messages ORDER BY id").fetchall()
    raw.close()
    return [r[0] for r in rows]


def _meta_value(db_path, key):
    raw = sqlite3.connect(str(db_path))
    row = raw.execute(
        "SELECT value FROM state_meta WHERE key = ?", (key,)
    ).fetchone()
    raw.close()
    return None if row is None else row[0]


def _base_fts_triggers(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' "
        f"AND name IN ({','.join('?' for _ in _FTS_TRIGGERS)})",
        _FTS_TRIGGERS,
    ).fetchall()
    raw.close()
    return {row[0] for row in rows}


class _FtsCommitErrorConnection(sqlite3.Connection):
    """One-shot commit seam for FTS-shaped and generic corruption faults."""

    fail_next_fts_commit: str | None
    injected_commit_error: BaseException | None
    fts_commit_transaction_states: list[bool]

    def commit(self):
        self.commit_call_count = getattr(self, "commit_call_count", 0) + 1
        mode = getattr(self, "fail_next_fts_commit", None)
        injected_error = getattr(self, "injected_commit_error", None)
        if mode is None and injected_error is None:
            return super().commit()
        self.fail_next_fts_commit = None
        self.injected_commit_error = None
        states = getattr(self, "fts_commit_transaction_states", None)
        if states is None:
            states = []
            self.fts_commit_transaction_states = states
        if injected_error is not None:
            states.append(self.in_transaction)
            raise injected_error
        if mode == "after_real_commit":
            super().commit()
            states.append(self.in_transaction)
            raise sqlite3.DatabaseError(
                'fts5: corrupt structure record for table "messages_fts"'
            )
        states.append(self.in_transaction)
        if mode == "before_generic_malformed":
            raise sqlite3.DatabaseError("database disk image is malformed")
        raise sqlite3.DatabaseError(
            'fts5: corrupt structure record for table "messages_fts"'
        )

    def rollback(self):
        self.rollback_call_count = getattr(self, "rollback_call_count", 0) + 1
        return super().rollback()


def _db_with_fts_commit_error_seam(tmp_path, monkeypatch, *, db_path=None):
    original_connect = hermes_state._connect_tracked_db

    def connect_with_fts_commit_error(*args, **kwargs):
        kwargs["factory"] = _FtsCommitErrorConnection
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(
        hermes_state, "_connect_tracked_db", connect_with_fts_commit_error
    )
    return SessionDB(db_path=db_path or tmp_path / "state.db")


class TestRuntimeFtsRebuild:
    def test_foreign_holder_detection_includes_deleted_wal(
        self, db, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"

        class FakePsutil:
            @staticmethod
            def process_iter(_attrs):
                return iter(
                    (
                        SimpleNamespace(
                            info={
                                "pid": 111,
                                "open_files": [SimpleNamespace(path=str(db_path))],
                            }
                        ),
                        SimpleNamespace(
                            info={
                                "pid": 222,
                                "open_files": [
                                    SimpleNamespace(path=f"{db_path}-wal (deleted)")
                                ],
                            }
                        ),
                        SimpleNamespace(
                            info={
                                "pid": 333,
                                "open_files": [SimpleNamespace(path=str(tmp_path / "other.db"))],
                            }
                        ),
                    )
                )

        monkeypatch.setattr(hermes_state, "psutil", FakePsutil)
        monkeypatch.setattr(hermes_state, "_IS_WINDOWS", False)
        monkeypatch.setattr(hermes_state.os, "getpid", lambda: 111)
        # Force the macOS/psutil path even on Linux test runners
        monkeypatch.setattr(hermes_state.sys, "platform", "darwin")

        assert db._foreign_state_db_holders() == [
            (222, f"{db_path}-wal (deleted)")
        ]

    def test_foreign_holder_detection_proc_readlink_deleted_wal(
        self, db, tmp_path, monkeypatch
    ):
        """Linux /proc/<pid>/fd readlinks preserve '(deleted)' suffix.

        psutil.open_files() drops these entries (isfile_strict stats the
        literal path and fails).  The /proc path catches the split-brain
        holder that psutil silently misses.
        """
        db_path = tmp_path / "state.db"
        db_path_wal = str(db_path) + "-wal"

        # Build a fake /proc with two PIDs: self (111) and foreign (222).
        proc_root = tmp_path / "proc"
        for pid in (111, 222, 333):
            fd_dir = proc_root / str(pid) / "fd"
            fd_dir.mkdir(parents=True)
        # PID 222 holds the deleted WAL sidecar
        os.symlink(db_path_wal + " (deleted)", str(proc_root / "222" / "fd" / "3"))
        # PID 111 (self) holds the db — should be excluded
        os.symlink(str(db_path), str(proc_root / "111" / "fd" / "3"))
        # PID 333 holds an unrelated file
        other = tmp_path / "other.db"
        other.touch()
        os.symlink(str(other), str(proc_root / "333" / "fd" / "3"))

        monkeypatch.setattr(hermes_state, "_IS_WINDOWS", False)
        monkeypatch.setattr(hermes_state.os, "getpid", lambda: 111)
        monkeypatch.setattr(hermes_state.sys, "platform", "linux")
        real_listdir = os.listdir
        def _listdir(path):
            if isinstance(path, str):
                path = path.replace("/proc", str(proc_root))
            return real_listdir(path)
        monkeypatch.setattr(hermes_state.os, "listdir", _listdir)
        real_readlink = os.readlink
        def _readlink(path):
            path = path.replace("/proc", str(proc_root))
            return real_readlink(path)
        monkeypatch.setattr(hermes_state.os, "readlink", _readlink)

        holders = db._foreign_state_db_holders()
        assert holders == [(222, db_path_wal + " (deleted)")]

    def test_foreign_holder_uninspectable_process_cmdline_fallback(
        self, db, tmp_path, monkeypatch
    ):
        """A process whose fd table is unreadable (different user) is still
        flagged when /proc/<pid>/cmdline identifies it as a Hermes process."""
        db_path = tmp_path / "state.db"

        proc_root = tmp_path / "proc"
        for pid in (111, 222):
            (proc_root / str(pid) / "fd").mkdir(parents=True)
        # PID 222's fd dir is unreadable (PermissionError)
        os.chmod(proc_root / "222" / "fd", 0o000)
        # PID 222's cmdline is world-readable and looks like Hermes
        cmdline_path = proc_root / "222" / "cmdline"
        cmdline_path.write_bytes(b"python3\x00hermes_cli.main\x00chat\x00")

        monkeypatch.setattr(hermes_state, "_IS_WINDOWS", False)
        monkeypatch.setattr(hermes_state.os, "getpid", lambda: 111)
        monkeypatch.setattr(hermes_state.sys, "platform", "linux")
        real_listdir = os.listdir
        def _listdir(path):
            if isinstance(path, str):
                path = path.replace("/proc", str(proc_root))
            return real_listdir(path)
        monkeypatch.setattr(hermes_state.os, "listdir", _listdir)
        # _read_proc_cmdline opens /proc/<pid>/cmdline directly; redirect
        # it to our fake proc tree.
        def _fake_cmdline(pid):
            fake_path = str(proc_root / str(pid) / "cmdline")
            try:
                with open(fake_path, "rb") as f:
                    raw = f.read()
                if not raw:
                    return None
                return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
            except OSError:
                return None
        monkeypatch.setattr(hermes_state, "_read_proc_cmdline", _fake_cmdline)

        holders = db._foreign_state_db_holders()
        # Should include PID 222 with the cmdline info
        assert len(holders) == 1
        assert holders[0][0] == 222
        assert "hermes_cli.main" in holders[0][1]

        # Cleanup
        os.chmod(proc_root / "222" / "fd", 0o755)

    def test_corruption_error_classification_covers_both_sqlite_messages(self):
        """SQLite's message for a corrupt FTS index varies by version: older
        builds raise the generic malformed-image error, newer builds raise an
        FTS5-specific one. Both must trigger the self-heal."""
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("database disk image is malformed")
        )
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError(
                'fts5: corrupt structure record for table "messages_fts"'
            )
        )
        assert not SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("no such table: nothing_fts_related")
        )

    def test_append_self_heals_after_fts_corruption(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "hello world")

        _corrupt_fts(tmp_path / "state.db")

        # Before the fix this raised DatabaseError and the row was lost.
        msg_id = db.append_message("s1", "user", "healed append")
        assert msg_id is not None
        assert _message_contents(tmp_path / "state.db") == [
            "hello world",
            "healed append",
        ]

    def test_search_works_after_self_heal(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "before corruption")
        _corrupt_fts(tmp_path / "state.db")
        db.append_message("s1", "user", "searchable needle text")

        raw = sqlite3.connect(str(tmp_path / "state.db"))
        hits = raw.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'needle'"
        ).fetchall()
        raw.close()
        assert len(hits) == 1

    def test_search_messages_self_heals_after_fts_corruption(self, db, tmp_path):
        """A read-only session that only SEARCHES (no write after corruption)
        must self-heal too. The MATCH read raises the corruption class
        (DatabaseError / 'fts5: corrupt structure record'), NOT the
        OperationalError that search_messages caught — so before this fix the
        search crashed until a write or restart rebuilt the index.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "a searchable needle here")

        _corrupt_fts(tmp_path / "state.db")
        # Injected via a raw connection, so no write on THIS instance has
        # consumed the one-shot rebuild yet.
        assert db._fts_runtime_rebuild_attempted is False

        results = db.search_messages("needle")

        assert db._fts_runtime_rebuild_attempted is True  # the search rebuilt it
        assert results  # non-empty: the rebuilt index matched the query
        assert any("needle" in (r.get("snippet") or "") for r in results)

    def test_trigram_search_self_heals_after_fts_corruption(self, db, tmp_path):
        """The CJK/trigram MATCH branch has the same read-corruption exposure
        as the main FTS5 branch: it caught only OperationalError (query
        syntax), so a corrupt trigram shadow table raised DatabaseError
        straight out of search_messages. It must self-heal via the shared
        one-shot rebuild and answer from the rebuilt trigram index.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        if not db._trigram_available:
            pytest.skip("trigram tokenizer unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "关于大别山项目的进展报告")

        _corrupt_trigram_fts(tmp_path / "state.db")
        assert db._fts_runtime_rebuild_attempted is False

        # >=3 CJK chars per token → routed to the trigram branch.
        results = db.search_messages("大别山项目")

        assert db._fts_runtime_rebuild_attempted is True  # search rebuilt it
        assert results
        # The rebuilt trigram index answered (trigram snippets use >>> <<<),
        # i.e. we did not silently degrade to the LIKE fallback.
        assert any(">>>" in (r.get("snippet") or "") for r in results)


    def test_second_corruption_fails_open_and_rebuilds_on_reopen(
        self, db, tmp_path
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        db.append_message("s1", "user", "first heal")  # consumes the one shot
        assert db._fts_runtime_rebuild_attempted is True

        # A second corruption must not strand the canonical transcript. The
        # derived indexes are detached and marked stale instead of looping.
        _corrupt_fts(db_path)
        db.append_message("s1", "user", "second corruption")
        assert _message_contents(db_path) == [
            "seed",
            "first heal",
            "second corruption",
        ]
        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

        # Search remains available from canonical rows while FTS is stale.
        results = db.search_messages("second corruption")
        assert results
        assert any("second corruption" in row["snippet"] for row in results)

        # A later open atomically rebuilds all canonical rows before triggers
        # return, then clears the durable breadcrumb.
        db.close()
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert _base_fts_triggers(db_path) == set(_FTS_TRIGGERS)
            results = reopened.search_messages("second corruption")
            assert results
        finally:
            reopened.close()

    def test_failed_in_place_rebuild_fails_open(self, db, tmp_path, monkeypatch):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)

        def _failed_rebuild():
            raise sqlite3.DatabaseError("rebuild could not read corrupt FTS")

        monkeypatch.setattr(db, "rebuild_fts", _failed_rebuild)
        db.append_message("s1", "user", "canonical survives")

        assert _message_contents(db_path)[-1] == "canonical survives"
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

    def test_foreign_holder_skips_runtime_rebuild_and_fails_open(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)

        monkeypatch.setattr(
            db,
            "_foreign_state_db_holders",
            lambda: [(4242, str(db_path) + "-wal")],
            raising=False,
        )

        db.append_message("s1", "user", "canonical survives foreign holder")

        assert _message_contents(db_path)[-1] == "canonical survives foreign holder"
        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

    def test_stale_search_preserves_not_semantics(self, db, tmp_path, monkeypatch):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "python language guide")
        db.append_message("s1", "user", "python java interoperability")
        _corrupt_fts(db_path)

        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(
                sqlite3.DatabaseError("rebuild could not read corrupt FTS")
            ),
        )
        db.append_message("s1", "user", "canonical write survives")
        assert db._fts_stale is True

        results = db.search_messages("python NOT java")
        snippets = [row["snippet"] for row in results]
        assert any("python language guide" in snippet for snippet in snippets)
        assert all("java" not in snippet for snippet in snippets)

    def test_existing_peer_observes_fail_open_marker(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        peer = SessionDB(db_path=db_path)
        try:
            _corrupt_fts(db_path)

            def _failed_rebuild():
                raise sqlite3.DatabaseError("rebuild failed")

            monkeypatch.setattr(db, "rebuild_fts", _failed_rebuild)
            db.append_message("s1", "user", "visible through canonical search")

            assert peer._fts_stale is False
            results = peer.search_messages("canonical search")
            assert peer._fts_stale is True
            assert results
        finally:
            peer.close()

    def test_failed_startup_rebuild_keeps_fts_detached(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("still corrupt")),
        )
        db.append_message("s1", "user", "before restart")
        db.close()

        monkeypatch.setattr(
            SessionDB,
            "_recover_stale_fts",
            lambda self, cursor, legacy: False,
        )
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
            assert _base_fts_triggers(db_path) == set()
            reopened.append_message("s1", "user", "after failed recovery")
            assert _message_contents(db_path)[-1] == "after failed recovery"
            assert reopened.search_messages("failed recovery")
        finally:
            reopened.close()

    def test_foreign_holder_defers_startup_stale_rebuild(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("still corrupt")),
        )
        db.append_message("s1", "user", "before restart")
        db.close()

        monkeypatch.setattr(
            SessionDB,
            "_foreign_state_db_holders",
            lambda self: [(4242, str(db_path) + "-wal")],
            raising=False,
        )
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
            assert _base_fts_triggers(db_path) == set()
            reopened.append_message("s1", "user", "after deferred recovery")
            assert _message_contents(db_path)[-1] == "after deferred recovery"
        finally:
            reopened.close()

    @pytest.mark.parametrize(
        "table_name",
        ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"),
    )
    def test_commit_time_fts_error_recovers_only_after_definite_rollback(
        self, tmp_path, monkeypatch, table_name
    ):
        """An active commit-time FTS failure may replay only after rollback."""
        db = _db_with_fts_commit_error_seam(tmp_path, monkeypatch)
        try:
            db.create_session("s1", source="test")
            conn = db._conn
            assert isinstance(conn, _FtsCommitErrorConnection)
            conn.fts_commit_transaction_states = []
            conn.injected_commit_error = sqlite3.DatabaseError(
                f'fts5: corrupt structure record for table "{table_name}"'
            )
            rebuild_transaction_states = []
            fail_open_calls = []

            def rebuilt_after_rollback(exc):
                assert SessionDB._is_fts_write_corruption_error(exc)
                rebuild_transaction_states.append(db._conn.in_transaction)
                return True

            monkeypatch.setattr(db, "_try_runtime_fts_rebuild", rebuilt_after_rollback)
            monkeypatch.setattr(
                db,
                "_enter_fts_fail_open",
                lambda exc: fail_open_calls.append(exc) or False,
            )

            assert db.append_message("s1", "user", "rollback-gated canonical write")
            assert conn.fts_commit_transaction_states == [True]
            assert rebuild_transaction_states == [False]
            assert fail_open_calls == []
            assert _message_contents(tmp_path / "state.db") == [
                "rollback-gated canonical write"
            ]
        finally:
            db.close()

    def test_commit_time_generic_malformed_image_rolls_back_without_fts_recovery(
        self, tmp_path, monkeypatch
    ):
        """A generic corrupt-image commit error is never FTS recovery evidence."""
        db = _db_with_fts_commit_error_seam(tmp_path, monkeypatch)
        try:
            db.create_session("s1", source="test")
            conn = db._conn
            assert isinstance(conn, _FtsCommitErrorConnection)
            conn.fts_commit_transaction_states = []
            conn.commit_call_count = 0
            conn.rollback_call_count = 0
            conn.fail_next_fts_commit = "before_generic_malformed"
            rebuild_calls = []
            fail_open_calls = []
            monkeypatch.setattr(
                db,
                "_try_runtime_fts_rebuild",
                lambda exc: rebuild_calls.append(exc) or True,
            )
            monkeypatch.setattr(
                db,
                "_enter_fts_fail_open",
                lambda exc: fail_open_calls.append(exc) or True,
            )

            with pytest.raises(
                sqlite3.DatabaseError, match="database disk image is malformed"
            ):
                db.append_message("s1", "user", "must not replay generic corruption")

            assert conn.fts_commit_transaction_states == [True]
            assert conn.rollback_call_count == 1
            assert conn.in_transaction is False
            assert rebuild_calls == []
            assert fail_open_calls == []
            assert conn.commit_call_count == 1
            assert _message_contents(tmp_path / "state.db") == []
        finally:
            db.close()

    @pytest.mark.parametrize(
        ("case_id", "message", "outer_unrelated"),
        (
            (
                "uppercase-fts5",
                'FTS5: corrupt structure record for table "messages_fts"',
                False,
            ),
            (
                "arbitrary-prefix",
                'prefix fts5: corrupt structure record for table "messages_fts"',
                False,
            ),
            (
                "arbitrary-suffix",
                'fts5: corrupt structure record for table "messages_fts" suffix',
                False,
            ),
            (
                "leading-whitespace",
                ' fts5: corrupt structure record for table "messages_fts"',
                False,
            ),
            (
                "trailing-whitespace",
                'fts5: corrupt structure record for table "messages_fts" ',
                False,
            ),
            ("bare-fts5", "fts5: corrupt structure record", False),
            (
                "foreign-table",
                'fts5: corrupt structure record for table "foreign_fts"',
                False,
            ),
            ("non-corruption-fts5", "fts5: syntax error", False),
            (
                "nested-cause-outer-unrelated",
                'fts5: corrupt structure record for table "messages_fts"',
                True,
            ),
            ("generic-malformed-image", "database disk image is malformed", False),
        ),
    )
    def test_commit_time_fts_text_spoofs_never_recover_or_replay(
        self, tmp_path, monkeypatch, case_id, message, outer_unrelated
    ):
        """Only a direct, exact supported-table commit signature may recover."""
        db = _db_with_fts_commit_error_seam(tmp_path, monkeypatch)
        try:
            db.create_session("s1", source="test")
            conn = db._conn
            assert isinstance(conn, _FtsCommitErrorConnection)
            conn.fts_commit_transaction_states = []
            conn.commit_call_count = 0
            conn.rollback_call_count = 0
            if outer_unrelated:
                original_error = RuntimeError("outer unrelated commit error")
                original_error.__cause__ = sqlite3.DatabaseError(message)
            else:
                original_error = sqlite3.DatabaseError(message)
            conn.injected_commit_error = original_error
            rebuild_calls = []
            fail_open_calls = []
            monkeypatch.setattr(
                db,
                "_try_runtime_fts_rebuild",
                lambda exc: rebuild_calls.append(exc) or True,
            )
            monkeypatch.setattr(
                db,
                "_enter_fts_fail_open",
                lambda exc: fail_open_calls.append(exc) or True,
            )

            with pytest.raises(type(original_error)) as raised:
                db.append_message("s1", "user", f"spoofed {case_id}")

            assert raised.value is original_error
            assert conn.fts_commit_transaction_states == [True]
            assert conn.rollback_call_count == 1
            assert conn.in_transaction is False
            assert rebuild_calls == []
            assert fail_open_calls == []
            assert conn.commit_call_count == 1
            assert _message_contents(tmp_path / "state.db") == []
        finally:
            db.close()

    def test_fts_shaped_error_after_real_commit_never_replays_canonical_row(
        self, tmp_path, monkeypatch
    ):
        """A commit ambiguity is terminal even when its error looks like FTS."""
        db = _db_with_fts_commit_error_seam(tmp_path, monkeypatch)
        try:
            db.create_session("s1", source="test")
            conn = db._conn
            assert isinstance(conn, _FtsCommitErrorConnection)
            conn.fts_commit_transaction_states = []
            conn.fail_next_fts_commit = "after_real_commit"
            rebuild_calls = []
            fail_open_calls = []
            monkeypatch.setattr(
                db,
                "_try_runtime_fts_rebuild",
                lambda exc: rebuild_calls.append(exc) or False,
            )
            monkeypatch.setattr(
                db,
                "_enter_fts_fail_open",
                lambda exc: fail_open_calls.append(exc) or False,
            )

            with pytest.raises(sqlite3.DatabaseError, match="fts5: corrupt"):
                db.append_message("s1", "user", "committed exactly once")

            assert conn.fts_commit_transaction_states == [False]
            assert rebuild_calls == []
            assert fail_open_calls == []
            assert _message_contents(tmp_path / "state.db") == ["committed exactly once"]
        finally:
            db.close()

    def test_legacy_inline_fts_fails_open_and_recovers(self, tmp_path, monkeypatch):
        db_path = tmp_path / "legacy-state.db"
        raw = sqlite3.connect(str(db_path))
        raw.executescript(SCHEMA_SQL)
        raw.execute(
            "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
        )
        try:
            raw.executescript(LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL)
        except sqlite3.OperationalError as exc:
            raw.close()
            pytest.skip(f"required FTS tokenizer unavailable: {exc}")
        assert raw.execute(
            "SELECT version, typeof(version) FROM schema_version"
        ).fetchall() == [(SCHEMA_VERSION, "integer")]
        raw.commit()
        raw.close()

        legacy = _db_with_fts_commit_error_seam(
            tmp_path, monkeypatch, db_path=db_path
        )
        try:
            assert legacy._db_has_legacy_inline_fts(legacy._conn.cursor())
            assert isinstance(legacy._conn, _FtsCommitErrorConnection)
            legacy.create_session("s1", source="test")
            legacy.append_message("s1", "user", "legacy seed")
            _corrupt_fts(db_path)
            legacy._conn.fts_commit_transaction_states = []
            legacy._conn.fail_next_fts_commit = "before_real_commit"
            monkeypatch.setattr(
                legacy,
                "rebuild_fts",
                lambda: (_ for _ in ()).throw(
                    sqlite3.DatabaseError("legacy rebuild failed")
                ),
            )
            legacy.append_message("s1", "user", "legacy canonical survives")
            assert legacy._conn.fts_commit_transaction_states == [True]
            assert _message_contents(db_path)[-1] == "legacy canonical survives"
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        finally:
            legacy.close()

        recovered = SessionDB(db_path=db_path)
        try:
            assert recovered._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert recovered.search_messages("canonical survives")
        finally:
            recovered.close()
