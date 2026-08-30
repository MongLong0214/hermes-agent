"""Behavioral coverage for #68545's centralized journal-mode setting."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

import hermes_state


def _write_config(monkeypatch: pytest.MonkeyPatch, tmp_path, config: object) -> None:
    home = tmp_path / "hermes-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump(config),
        encoding="utf-8",
    )


def _configure_mode(monkeypatch: pytest.MonkeyPatch, tmp_path, mode: object) -> None:
    _write_config(monkeypatch, tmp_path, {"database": {"journal_mode": mode}})


def _disable_vulnerable_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_state.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: False,
    )


def test_database_journal_mode_has_a_canonical_default():
    from hermes_cli.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["database"]["journal_mode"] == "wal"


def test_resolve_journal_mode_uses_real_database_config(monkeypatch, tmp_path):
    from hermes_state import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, "DELETE")
    assert resolve_journal_mode() == "delete"


def test_new_nonsecret_hermes_env_override_is_not_exposed(monkeypatch, tmp_path):
    from hermes_state import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, "wal")
    monkeypatch.setenv("HERMES_JOURNAL_MODE", "delete")
    assert resolve_journal_mode() == "wal"


@pytest.mark.parametrize("value", ["bogus", "truncate", None, 42, {"bad": "shape"}])
def test_invalid_config_value_falls_back_to_wal(monkeypatch, tmp_path, value):
    from hermes_state import resolve_journal_mode

    _configure_mode(monkeypatch, tmp_path, value)
    assert resolve_journal_mode() == "wal"


@pytest.mark.parametrize("database", [[], "delete", 42, None])
def test_malformed_database_section_falls_back_to_wal(
    monkeypatch, tmp_path, database
):
    from hermes_state import resolve_journal_mode

    _write_config(monkeypatch, tmp_path, {"database": database})
    assert resolve_journal_mode() == "wal"


def test_apply_wal_with_fallback_honors_delete_config(monkeypatch, tmp_path):
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    conn = sqlite3.connect(tmp_path / "configured.db")
    try:
        assert apply_wal_with_fallback(conn, db_label="configured.db") == "delete"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "delete"
    finally:
        conn.close()


def test_apply_wal_with_fallback_defaults_to_wal(monkeypatch, tmp_path):
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "wal")
    _disable_vulnerable_gate(monkeypatch)
    conn = sqlite3.connect(tmp_path / "default.db")
    try:
        assert apply_wal_with_fallback(conn, db_label="default.db") == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_configured_delete_validates_vulnerable_sqlite_result(monkeypatch, tmp_path):
    """The safety gate must not report DELETE when SQLite returns MEMORY."""
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    monkeypatch.setattr(
        "hermes_state.is_sqlite_wal_reset_vulnerable",
        lambda **kwargs: True,
    )
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(sqlite3.OperationalError, match="configured.*delete"):
            apply_wal_with_fallback(conn, db_label="memory-configured.db")
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "memory"
    finally:
        conn.close()


def test_configured_delete_never_live_downgrades_existing_wal(monkeypatch, tmp_path):
    from hermes_state import apply_wal_with_fallback

    _configure_mode(monkeypatch, tmp_path, "delete")
    db_path = tmp_path / "existing-wal.db"
    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone()[0].lower() == "wal"
        monkeypatch.setattr(
            "hermes_state.is_sqlite_wal_reset_vulnerable",
            lambda **kwargs: True,
        )
        assert apply_wal_with_fallback(conn, db_label="existing-wal.db") == "wal"
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_real_db_openers_honor_configured_delete(monkeypatch, tmp_path):
    """All helper-routed file-backed openers must behaviorally use DELETE."""
    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)

    from agent import verification_evidence
    from cron import executions
    from gateway import delivery_ledger
    from gateway.platforms.api_server import ResponseStore
    from hermes_cli import kanban_db, projects_db
    from hermes_state import SessionDB
    from plugins.memory.holographic.store import MemoryStore
    from plugins.platforms.discord.recovery import DiscordRecoveryStore
    from tools import async_delegation

    observed: dict[str, str] = {}

    for name, connect in (
        ("async_delegation", async_delegation._connect),
        ("delivery_ledger", delivery_ledger._connect),
        ("verification_evidence", verification_evidence._connect),
    ):
        conn = connect()
        try:
            observed[name] = conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
        finally:
            conn.close()

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cron_conn = executions._connect()
    try:
        executions._initialize_schema(cron_conn)
        observed["cron_executions"] = cron_conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        cron_conn.close()

    discord = DiscordRecoveryStore(hermes_home=tmp_path)
    observed["discord_recovery"] = discord.call(
        lambda conn: conn.execute("PRAGMA journal_mode").fetchone()[0].lower()
    )

    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        observed["session_db"] = session_db._conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        session_db.close()

    kanban_conn = kanban_db.connect(db_path=tmp_path / "kanban.db")
    try:
        observed["kanban"] = kanban_conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        kanban_conn.close()

    projects_conn = projects_db.connect(db_path=tmp_path / "projects.db")
    try:
        observed["projects"] = projects_conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        projects_conn.close()

    holographic = MemoryStore(db_path=tmp_path / "memory_store.db")
    try:
        observed["holographic"] = holographic._conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        holographic.close()

    response_store = ResponseStore(db_path=str(tmp_path / "response_store.db"))
    try:
        observed["response_store"] = response_store._conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0].lower()
    finally:
        response_store.close()

    assert observed == {
        "async_delegation": "delete",
        "delivery_ledger": "delete",
        "verification_evidence": "delete",
        "cron_executions": "delete",
        "discord_recovery": "delete",
        "session_db": "delete",
        "kanban": "delete",
        "projects": "delete",
        "holographic": "delete",
        "response_store": "delete",
    }


def _install_rebuild_claim(db_path: Path, value: str | None) -> None:
    """Install exactly one durable rebuild row without retaining a writer."""
    with sqlite3.connect(str(db_path), isolation_level=None) as conn:
        conn.execute(
            "DELETE FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        )
        conn.execute(
            "INSERT INTO state_meta(key, value) VALUES (?, ?)",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY, value),
        )


def _install_malformed_rebuild_metadata(db_path: Path) -> None:
    with sqlite3.connect(str(db_path), isolation_level=None) as conn:
        conn.execute("DROP TABLE state_meta")
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, broken TEXT)")


@pytest.mark.parametrize("claim_state", ("foreign", "null", "malformed"))
def test_sessiondb_journal_initializer_rechecks_authority_at_mode_switch(
    monkeypatch, tmp_path, claim_state
):
    """A claim inserted after init's first check blocks the actual mode switch."""
    from hermes_state import SessionDB

    db_path = tmp_path / "state.db"
    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    bootstrap = SessionDB(db_path=db_path)
    bootstrap.close()
    _configure_mode(monkeypatch, tmp_path, "wal")

    statements: list[str] = []
    real_connect = hermes_state._connect_tracked_db

    def traced_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    real_apply = hermes_state.apply_wal_with_fallback
    injected = False

    def inject_claim_then_apply(conn, *args, **kwargs):
        nonlocal injected
        if not injected and kwargs.get("db_label") == "state.db":
            if claim_state == "foreign":
                _install_rebuild_claim(db_path, "foreign-owner")
            elif claim_state == "null":
                _install_rebuild_claim(db_path, None)
            else:
                _install_malformed_rebuild_metadata(db_path)
            injected = True
        return real_apply(conn, *args, **kwargs)

    monkeypatch.setattr(hermes_state, "_connect_tracked_db", traced_connect)
    monkeypatch.setattr(hermes_state, "apply_wal_with_fallback", inject_claim_then_apply)

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        SessionDB(db_path=db_path)

    assert injected is True
    normalized = [" ".join(sql.upper().split()) for sql in statements]
    assert "PRAGMA JOURNAL_MODE=WAL" not in normalized
    assert not any(
        sql.startswith(("CREATE ", "ALTER ", "INSERT ", "UPDATE ", "DELETE "))
        for sql in normalized
    )


def test_mode_switch_holds_exclusion_after_expected_local_claim_proof(
    monkeypatch, tmp_path
):
    """The raw mode switch leaves no commit window after its authority read."""
    from hermes_state import apply_wal_with_fallback

    db_path = tmp_path / "state.db"
    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    bootstrap = hermes_state.SessionDB(db_path=db_path)
    bootstrap.close()
    _configure_mode(monkeypatch, tmp_path, "wal")
    local_marker = "expected-local-owner"
    _install_rebuild_claim(db_path, local_marker)

    statements: list[str] = []
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.set_trace_callback(statements.append)
    takeover_error = None
    try:
        def assert_then_attempt_takeover() -> None:
            nonlocal takeover_error
            hermes_state._assert_offline_rebuild_write_authority(
                conn, local_marker=local_marker
            )
            try:
                _install_rebuild_claim(db_path, "replacement-owner")
            except sqlite3.OperationalError as exc:
                takeover_error = exc

        assert (
            apply_wal_with_fallback(
                conn,
                db_label="state.db",
                before_journal_mode_change=assert_then_attempt_takeover,
            )
            == "wal"
        )
    finally:
        conn.close()

    assert isinstance(takeover_error, sqlite3.OperationalError)
    assert "PRAGMA JOURNAL_MODE=WAL" in [
        " ".join(sql.upper().split()) for sql in statements
    ]
    with sqlite3.connect(str(db_path), isolation_level=None) as writer:
        assert writer.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
        ).fetchone() == (local_marker,)


@pytest.mark.parametrize("opener_name", ("delivery_ledger", "async_delegation"))
@pytest.mark.parametrize("claim_state", ("foreign", "null", "malformed"))
def test_public_state_db_openers_refuse_unprovable_mode_change(
    monkeypatch, tmp_path, opener_name, claim_state
):
    """The real auxiliary state.db openers fence their setting PRAGMA and DDL."""
    from gateway import delivery_ledger
    from tools import async_delegation

    _configure_mode(monkeypatch, tmp_path, "delete")
    _disable_vulnerable_gate(monkeypatch)
    db_path = tmp_path / "hermes-home" / "state.db"
    bootstrap = hermes_state.SessionDB(db_path=db_path)
    bootstrap.close()

    opener = {
        "delivery_ledger": delivery_ledger,
        "async_delegation": async_delegation,
    }[opener_name]
    statements: list[str] = []
    real_initialize = opener._initialize_schema

    def traced_initialize(conn):
        conn.set_trace_callback(statements.append)
        return real_initialize(conn)

    monkeypatch.setattr(opener, "_initialize_schema", traced_initialize)

    def install_claim() -> None:
        if claim_state == "foreign":
            _install_rebuild_claim(db_path, "foreign-owner")
        elif claim_state == "null":
            _install_rebuild_claim(db_path, None)
        else:
            _install_malformed_rebuild_metadata(db_path)

    if opener_name == "async_delegation":
        real_session_db = hermes_state.SessionDB

        def bootstrap_then_install_claim(*args, **kwargs):
            db = real_session_db(*args, **kwargs)
            install_claim()
            return db

        monkeypatch.setattr(hermes_state, "SessionDB", bootstrap_then_install_claim)
    else:
        install_claim()

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        opener._connect()

    normalized = [" ".join(sql.upper().split()) for sql in statements]
    assert not any(sql.startswith("PRAGMA JOURNAL_MODE=") for sql in normalized)
    assert not any(sql.startswith(("CREATE ", "ALTER ", "INSERT ", "UPDATE ", "DELETE ")) for sql in normalized)


def _prepare_existing_wal_state_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    """Build an ordinary state.db whose on-disk header is already WAL."""
    _configure_mode(monkeypatch, tmp_path, "wal")
    _disable_vulnerable_gate(monkeypatch)
    db_path = tmp_path / "hermes-home" / "state.db"
    bootstrap = hermes_state.SessionDB(db_path=db_path)
    bootstrap.close()
    with sqlite3.connect(str(db_path), isolation_level=None) as conn:
        assert conn.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    return db_path


def _install_auxiliary_claim(db_path: Path, claim_state: str) -> None:
    if claim_state == "foreign":
        _install_rebuild_claim(db_path, "foreign-existing-wal-owner")
    elif claim_state == "null":
        _install_rebuild_claim(db_path, None)
    else:
        _install_malformed_rebuild_metadata(db_path)


def _assert_auxiliary_claim_preserved(db_path: Path, claim_state: str) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        if claim_state == "foreign":
            assert conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
            ).fetchone() == ("foreign-existing-wal-owner",)
        elif claim_state == "null":
            assert conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                (hermes_state._OFFLINE_REBUILD_EPOCH_KEY,),
            ).fetchone() == (None,)
        else:
            assert conn.execute("PRAGMA table_info(state_meta)").fetchall() == [
                (0, "key", "TEXT", 0, None, 1),
                (1, "broken", "TEXT", 0, None, 0),
            ]


def _take_over_after_authority_proof(
    monkeypatch: pytest.MonkeyPatch,
    db_path: Path,
    *,
    proof_number: int,
) -> dict[str, int | bool]:
    """Attempt a foreign claim immediately after one successful proof."""
    real_assert = hermes_state._assert_offline_rebuild_write_authority
    result: dict[str, int | bool] = {"proofs": 0, "installed": False}

    def assert_then_take_over(conn: sqlite3.Connection, local_marker: str | None) -> None:
        real_assert(conn, local_marker)
        result["proofs"] = int(result["proofs"]) + 1
        if result["proofs"] != proof_number:
            return
        try:
            with sqlite3.connect(
                str(db_path), timeout=0.0, isolation_level=None
            ) as writer:
                writer.execute(
                    "INSERT INTO state_meta(key, value) VALUES (?, ?)",
                    (
                        hermes_state._OFFLINE_REBUILD_EPOCH_KEY,
                        "foreign-auxiliary-schema-owner",
                    ),
                )
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            raise hermes_state.SessionTurnLeaseLostError(
                "foreign auxiliary-schema takeover blocked by the DDL transaction"
            ) from exc
        result["installed"] = True

    monkeypatch.setattr(
        hermes_state,
        "_assert_offline_rebuild_write_authority",
        assert_then_take_over,
    )
    return result


def _legacy_async_delegations_without_origin_session_id(
    conn: sqlite3.Connection,
) -> None:
    conn.execute("DROP TABLE async_delegations")
    conn.execute(
        """CREATE TABLE async_delegations (
            delegation_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL,
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            state TEXT NOT NULL,
            dispatched_at REAL NOT NULL,
            completed_at REAL,
            updated_at REAL NOT NULL,
            event_json TEXT,
            result_json TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            task_json TEXT,
            delivery_claim TEXT,
            delivery_claimed_at REAL
        )"""
    )


def test_delivery_ledger_schema_create_refuses_takeover_after_no_owner_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing-WAL schema setup cannot create the ledger after a takeover."""
    from gateway import delivery_ledger

    db_path = _prepare_existing_wal_state_db(monkeypatch, tmp_path)
    statements: list[str] = []
    takeover = _take_over_after_authority_proof(
        monkeypatch, db_path, proof_number=1
    )
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.set_trace_callback(statements.append)
    refusal = None
    try:
        delivery_ledger._initialize_schema(conn)
    except hermes_state.SessionTurnLeaseLostError as exc:
        refusal = exc
    finally:
        conn.close()

    normalized = [" ".join(sql.upper().split()) for sql in statements]
    assert type(refusal) is hermes_state.SessionTurnLeaseLostError
    assert not any(sql.startswith("CREATE TABLE") for sql in normalized)
    assert takeover["proofs"] == 1


def test_async_schema_create_refuses_takeover_after_no_owner_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing-WAL schema setup cannot create async state after a takeover."""
    from tools import async_delegation

    db_path = _prepare_existing_wal_state_db(monkeypatch, tmp_path)
    statements: list[str] = []
    takeover = _take_over_after_authority_proof(
        monkeypatch, db_path, proof_number=1
    )
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.set_trace_callback(statements.append)
    refusal = None
    try:
        async_delegation._initialize_schema(conn)
    except hermes_state.SessionTurnLeaseLostError as exc:
        refusal = exc
    finally:
        conn.close()

    normalized = [" ".join(sql.upper().split()) for sql in statements]
    assert type(refusal) is hermes_state.SessionTurnLeaseLostError
    assert not any(sql.startswith("CREATE TABLE") for sql in normalized)
    assert takeover["proofs"] >= 1


def test_async_schema_alter_refuses_takeover_after_no_owner_proof(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Existing-WAL migration cannot add a column after a takeover."""
    from tools import async_delegation

    db_path = _prepare_existing_wal_state_db(monkeypatch, tmp_path)
    statements: list[str] = []
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        _legacy_async_delegations_without_origin_session_id(conn)
    finally:
        conn.close()
    takeover = _take_over_after_authority_proof(
        monkeypatch, db_path, proof_number=2
    )
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.set_trace_callback(statements.append)
    refusal = None
    try:
        async_delegation._initialize_schema(conn)
    except hermes_state.SessionTurnLeaseLostError as exc:
        refusal = exc
    finally:
        conn.close()

    normalized = [" ".join(sql.upper().split()) for sql in statements]
    assert type(refusal) is hermes_state.SessionTurnLeaseLostError
    assert not any(sql.startswith("ALTER TABLE") for sql in normalized)
    assert takeover["proofs"] >= 2


@pytest.mark.parametrize("opener_name", ("delivery_ledger", "async_delegation"))
@pytest.mark.parametrize("claim_state", ("foreign", "null", "malformed"))
def test_existing_wal_auxiliary_initializers_refuse_claim_before_schema_dml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener_name: str,
    claim_state: str,
) -> None:
    """An already-WAL state.db still fences auxiliary DDL on its own connection."""
    from gateway import delivery_ledger
    from tools import async_delegation

    db_path = _prepare_existing_wal_state_db(monkeypatch, tmp_path)
    opener = {
        "delivery_ledger": delivery_ledger,
        "async_delegation": async_delegation,
    }[opener_name]
    statements: list[str] = []
    real_initialize = opener._initialize_schema

    def trace_and_initialize(conn: sqlite3.Connection) -> None:
        conn.set_trace_callback(statements.append)
        real_initialize(conn)

    monkeypatch.setattr(opener, "_initialize_schema", trace_and_initialize)
    if opener_name == "async_delegation":
        real_session_db = hermes_state.SessionDB

        def bootstrap_then_take_over(*args, **kwargs):
            bootstrap = real_session_db(*args, **kwargs)
            _install_auxiliary_claim(db_path, claim_state)
            return bootstrap

        monkeypatch.setattr(hermes_state, "SessionDB", bootstrap_then_take_over)
    else:
        _install_auxiliary_claim(db_path, claim_state)

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        opener._connect()

    normalized = [" ".join(sql.upper().split()) for sql in statements]
    assert not any(
        sql.startswith(("CREATE ", "ALTER ", "INSERT ", "UPDATE ", "DELETE "))
        for sql in normalized
    )
    _assert_auxiliary_claim_preserved(db_path, claim_state)


@pytest.mark.parametrize("opener_name", ("delivery_ledger", "async_delegation"))
@pytest.mark.parametrize("claim_state", ("foreign", "null", "malformed"))
def test_existing_wal_auxiliary_writes_refuse_post_connect_takeover(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    opener_name: str,
    claim_state: str,
) -> None:
    """A claim appearing after schema setup blocks the real durable public write."""
    from gateway import delivery_ledger
    from tools import async_delegation

    db_path = _prepare_existing_wal_state_db(monkeypatch, tmp_path)
    opener = {
        "delivery_ledger": delivery_ledger,
        "async_delegation": async_delegation,
    }[opener_name]
    statements: list[str] = []
    real_connect = opener._connect

    def connect_then_takeover() -> sqlite3.Connection:
        conn = real_connect()
        conn.set_trace_callback(statements.append)
        _install_auxiliary_claim(db_path, claim_state)
        return conn

    monkeypatch.setattr(opener, "_connect", connect_then_takeover)

    with pytest.raises(hermes_state.SessionTurnLeaseLostError):
        if opener_name == "delivery_ledger":
            delivery_ledger.record_obligation(
                obligation_id="existing-wal-takeover",
                session_key="test-session",
                platform="test",
                chat_id="chat",
                thread_id=None,
                content="fenced",
            )
        else:
            async_delegation._persist_dispatch(
                {
                    "delegation_id": "existing-wal-takeover",
                    "session_key": "test-session",
                    "dispatched_at": 1.0,
                }
            )

    normalized = [" ".join(sql.upper().split()) for sql in statements]
    assert "BEGIN IMMEDIATE" in normalized
    assert not any(
        sql.startswith(("CREATE ", "ALTER ", "INSERT ", "UPDATE ", "DELETE "))
        for sql in normalized
    )
    _assert_auxiliary_claim_preserved(db_path, claim_state)


@pytest.mark.parametrize("opener_name", ("delivery_ledger", "async_delegation"))
def test_existing_wal_auxiliary_writes_still_succeed_without_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, opener_name: str
) -> None:
    """The same existing-WAL public paths retain their ordinary no-marker behavior."""
    from gateway import delivery_ledger
    from tools import async_delegation

    _prepare_existing_wal_state_db(monkeypatch, tmp_path)

    if opener_name == "delivery_ledger":
        delivery_ledger.record_obligation(
            obligation_id="existing-wal-no-claim",
            session_key="test-session",
            platform="test",
            chat_id="chat",
            thread_id=None,
            content="ordinary",
        )
        conn = delivery_ledger._connect()
        try:
            assert conn.execute(
                "SELECT state FROM delivery_obligations WHERE obligation_id = ?",
                ("existing-wal-no-claim",),
            ).fetchone() == ("pending",)
        finally:
            conn.close()
    else:
        async_delegation._persist_dispatch(
            {
                "delegation_id": "existing-wal-no-claim",
                "session_key": "test-session",
                "dispatched_at": 1.0,
            }
        )
        conn = async_delegation._connect()
        try:
            assert conn.execute(
                "SELECT state FROM async_delegations WHERE delegation_id = ?",
                ("existing-wal-no-claim",),
            ).fetchone() == ("running",)
        finally:
            conn.close()
