"""Tests for the messages_fts_cjk CJK-bigram index (salvaged from PR #65544).

Builds the loadable tokenizer from native/fts5_cjk/fts5_cjk.c on the fly;
skips when no C toolchain / extension loading is available.
"""

import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from hermes_state import FTS_CJK_STALE_KEY, SessionDB

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "native" / "fts5_cjk" / "fts5_cjk.c"
VENDOR = REPO / "native" / "fts5_cjk" / "vendor"


@pytest.fixture(scope="session")
def cjk_so(tmp_path_factory):
    if shutil.which("gcc") is None or not SRC.exists():
        pytest.skip("no C toolchain / tokenizer source")
    out = tmp_path_factory.mktemp("fts5cjk") / "libfts5_cjk.so"
    try:
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", f"-I{VENDOR}", str(SRC),
             "-o", str(out)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.skip(f"tokenizer build failed: {e.stderr[:200]}")
    # Loadability probe (extension loading may be disabled in this build).
    probe = sqlite3.connect(":memory:")
    try:
        probe.enable_load_extension(True)
        probe.load_extension(str(out))
    except Exception as e:
        pytest.skip(f"extension loading unavailable: {e}")
    finally:
        probe.close()
    return out


@pytest.fixture()
def db(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d = SessionDB(db_path=tmp_path / "state.db")
    assert d._fts_cjk_loaded, "tokenizer must load on the writer connection"
    assert d._fts_cjk_available, "fresh DB must be born with the cjk index"
    d.create_session(session_id="s1", source="cli", model="m")
    d.append_message("s1", role="user", content="웅기가 shared default 프로필을 요청했다")
    d.append_message("s1", role="assistant", content="일본 MCP 후보 우선순위 정리했습니다")
    d.append_message("s1", role="user", content="graphiti daemon looks healthy")
    d.append_message("s1", role="tool", content="일본 tool output blob", tool_name="terminal")
    yield d
    d.close()


def test_two_char_korean_hits_cjk_index(db):
    rows = db.search_messages("웅기", limit=10)
    assert rows and "웅기" in rows[0]["snippet"]
    rows = db.search_messages("일본", limit=10)
    assert rows


def test_mixed_and_ascii_queries(db):
    assert db.search_messages("graphiti", limit=10)
    assert db.search_messages('"shared default" AND 웅기', limit=10)
    assert db.search_messages("우선순위", limit=10)




def test_lone_single_cjk_char_routes_like(db):
    # 1-char CJK terms keep LIKE substring semantics (bigram index only
    # holds unigrams for isolated chars). "가" appears inside 웅기가.
    assert db._describe_search_path("가") == "like_scan"
    rows = db.search_messages("가", limit=10)
    assert rows, "LIKE fallback must still find substring matches"








def test_config_toggle_disables_cjk(cjk_so, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    monkeypatch.setenv("HERMES_CJK_FTS", "0")
    d = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert not d._fts_cjk_loaded
        assert not d._fts_cjk_available
        # No cjk objects created at all.
        with d._lock:
            row = d._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'messages_fts_cjk'"
            ).fetchone()
        assert row is None
    finally:
        d.close()






def test_existing_v23_db_gains_cjk_via_optimize(cjk_so, tmp_path, monkeypatch):
    """A v23 DB created BEFORE the extension existed: next capable open
    creates the index with backfill markers; optimize-storage backfills."""
    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(tmp_path / "absent.so"))
    db_path = tmp_path / "state.db"
    d1 = SessionDB(db_path=db_path)
    d1.create_session(session_id="s1", source="cli", model="m")
    for i in range(10):
        d1.append_message("s1", role="user", content=f"기존 메시지 {i}")
    d1.close()

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    d2 = SessionDB(db_path=db_path)
    assert d2._fts_cjk_loaded
    # Backfill pending — index not served yet, old rows not indexed.
    assert not d2._fts_cjk_available
    st = d2.fts_cjk_rebuild_status()
    assert st is not None and st["pending"]
    assert d2.fts_optimize_available()
    # NEW rows are indexed live by the id-gated triggers even mid-backfill.
    d2.append_message("s1", role="user", content="새로운 메시지")
    # Search answers via legacy routes meanwhile.
    assert d2.search_messages("기존", limit=10)

    result = d2.optimize_fts_storage(vacuum=False)
    assert result["ok"]
    assert d2._fts_cjk_available
    assert d2.fts_cjk_rebuild_status() is None
    assert d2._describe_search_path("기존") == "fts_cjk"
    rows = d2.search_messages("기존", limit=20)
    assert len(rows) == 10
    assert d2.search_messages("새로운", limit=10)
    d2.close()


def test_legacy_v22_optimize_lands_on_cjk(cjk_so, tmp_path, monkeypatch):
    """A legacy inline-FTS (pre-v23) DB optimized on a tokenizer-capable
    host comes out with BOTH the v23 external-content layout AND a complete
    cjk index in the same run."""
    import time as _time

    monkeypatch.setenv("HERMES_FTS5_CJK_SO", str(cjk_so))
    db_path = tmp_path / "state.db"

    # Hand-build the pre-v28 v22 persisted surfaces: no authority objects,
    # inline FTS tables, and the v22 schema marker. SessionDB then performs
    # the current migration against this actual legacy input.
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE schema_version (
            version INTEGER NOT NULL
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            user_id TEXT,
            session_key TEXT,
            chat_id TEXT,
            chat_type TEXT,
            thread_id TEXT,
            display_name TEXT,
            origin_json TEXT,
            expiry_finalized INTEGER DEFAULT 0,
            model TEXT,
            model_config TEXT,
            system_prompt TEXT,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            ended_at REAL,
            end_reason TEXT,
            message_count INTEGER DEFAULT 0,
            tool_call_count INTEGER DEFAULT 0,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0,
            cwd TEXT,
            git_branch TEXT,
            git_repo_root TEXT,
            billing_provider TEXT,
            billing_base_url TEXT,
            billing_mode TEXT,
            estimated_cost_usd REAL,
            actual_cost_usd REAL,
            cost_status TEXT,
            cost_source TEXT,
            pricing_version TEXT,
            title TEXT,
            api_call_count INTEGER DEFAULT 0,
            handoff_state TEXT,
            handoff_platform TEXT,
            handoff_error TEXT,
            compression_failure_cooldown_until REAL,
            compression_failure_error TEXT,
            compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
            profile_name TEXT,
            rewind_count INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            role TEXT NOT NULL,
            content TEXT,
            tool_call_id TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            effect_disposition TEXT,
            timestamp REAL NOT NULL,
            token_count INTEGER,
            finish_reason TEXT,
            reasoning TEXT,
            reasoning_content TEXT,
            reasoning_details TEXT,
            codex_reasoning_items TEXT,
            codex_message_items TEXT,
            platform_message_id TEXT,
            observed INTEGER DEFAULT 0,
            active INTEGER NOT NULL DEFAULT 1,
            compacted INTEGER NOT NULL DEFAULT 0,
            api_content TEXT
        );
        CREATE TABLE state_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE VIRTUAL TABLE messages_fts USING fts5(content);
        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
        END;
        CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts WHERE rowid = old.id;
            INSERT INTO messages_fts(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');
        CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
            INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
        CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
            DELETE FROM messages_fts_trigram WHERE rowid = old.id;
        END;
        CREATE TRIGGER messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
            DELETE FROM messages_fts_trigram WHERE rowid = old.id;
            INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                new.id,
                COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
            );
        END;
    """)
    conn.execute("INSERT INTO schema_version (version) VALUES (22)")
    conn.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)",
        (_time.time(),),
    )
    for role, content in (
        ("user", "레거시 일본 메시지"),
        ("assistant", "legacy english reply"),
        ("tool", "레거시 tool output"),
    ):
        conn.execute(
            "INSERT INTO messages (session_id, timestamp, role, content) "
            "VALUES ('s1', ?, ?, ?)",
            (_time.time(), role, content),
        )
    conn.commit()
    trigger_names = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'messages'"
        )
    }
    expected_trigger_names = {
        "messages_fts_insert",
        "messages_fts_delete",
        "messages_fts_update",
        "messages_fts_trigram_insert",
        "messages_fts_trigram_delete",
        "messages_fts_trigram_update",
    }
    assert trigger_names == expected_trigger_names, (
        f"pre-startup raw v22 trigger names: {sorted(trigger_names)!r}"
    )
    conn.close()

    d = SessionDB(db_path=db_path)
    try:
        assert d.fts_optimize_available()
        # Legacy DB: cjk index deliberately not created at open (the legacy
        # branch of _init_schema doesn't touch v23 surfaces).
        assert not d._fts_cjk_available

        result = d.optimize_fts_storage(vacuum=False)
        assert result["ok"]
        assert d._fts_cjk_available
        assert d.fts_cjk_rebuild_status() is None
        assert d._describe_search_path("일본") == "fts_cjk"
        assert d.search_messages("일본", limit=10)
        assert d.search_messages("legacy english", limit=10)
        with d._lock:
            idx = d._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_cjk"
            ).fetchone()[0]
            non_tool = d._conn.execute(
                "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
            ).fetchone()[0]
        assert idx == non_tool
    finally:
        d.close()


def test_pure_latin_embedded_in_cjk_recovered_via_cjk_index(db):
    """#54242 residual: a pure-Latin query for a token embedded in CJK text
    (no whitespace) misses on unicode61; with the cjk index available the
    zero-result fallback recovers it as an exact ranked token match."""
    db.append_message("s1", role="user", content="修改youer服务端的계획")
    rows = db.search_messages("youer", limit=10)
    assert rows and "youer" in rows[0]["snippet"]
    # Short tokens (<3 chars, no trigram) are also recoverable via cjk.
    db.append_message("s1", role="user", content="에러코드ab확인")
    rows = db.search_messages("ab", limit=10)
    assert rows


def test_fresh_db_index_counts_exclude_tool_rows(db):
    with db._lock:
        idx = db._conn.execute(
            "SELECT COUNT(*) FROM messages_fts_cjk"
        ).fetchone()[0]
        non_tool = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
        ).fetchone()[0]
    assert idx == non_tool


def test_integrity_after_lifecycle(db):
    db.append_message("s1", role="user", content="무결성 검사")
    with db._lock:
        db._conn.execute(
            "INSERT INTO messages_fts_cjk(messages_fts_cjk) "
            "VALUES('integrity-check')"
        )
