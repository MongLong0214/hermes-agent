-- Fork generation-29 state.db DDL, generated from commit 6c9d8d8074870237d434b21fbff00bd9627d27f1
-- by scripts/state_fence/gen_fork_fixture.py with SQLite 3.53.1.
-- sqlite_master.sql in rowid order, minus sqlite_% and FTS shadow tables.
CREATE TABLE schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
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
    system_prompt_hash TEXT,
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
    git_metadata_generation INTEGER NOT NULL DEFAULT 0,
    session_generation INTEGER NOT NULL DEFAULT 0 CHECK (session_generation >= 0),
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL,
    last_activity_description TEXT,
    last_activity_provenance TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0,
    last_read_at REAL,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
    FOREIGN KEY (system_prompt_hash) REFERENCES system_prompts(hash)
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
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT
);
CREATE TABLE session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    api_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
);
CREATE TABLE state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE session_process_authorities (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    session_generation INTEGER NOT NULL CHECK (session_generation >= 1),
    state_db_id TEXT NOT NULL CHECK (
        length(state_db_id) = 64 AND state_db_id NOT GLOB '*[^0-9a-f]*'
    ),
    state_family TEXT NOT NULL CHECK (state_family = 'sessiondb-v1'),
    authority_token TEXT NOT NULL CHECK (
        length(authority_token) = 64 AND authority_token NOT GLOB '*[^0-9a-f]*'
    ),
    status TEXT NOT NULL CHECK (status IN ('ISSUED', 'CLOSED', 'REVOKED')),
    issued_at REAL NOT NULL,
    terminal_at REAL,
    PRIMARY KEY (session_id, session_generation)
);
CREATE TABLE session_process_authority_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    session_generation INTEGER NOT NULL CHECK (session_generation >= 1),
    state_db_id TEXT NOT NULL CHECK (
        length(state_db_id) = 64 AND state_db_id NOT GLOB '*[^0-9a-f]*'
    ),
    state_family TEXT NOT NULL CHECK (state_family = 'sessiondb-v1'),
    event_type TEXT NOT NULL CHECK (event_type IN (
        'SESSION_ISSUED', 'SESSION_CLOSED', 'SESSION_REVOKED',
        'PROCESS_RESERVATION', 'PROCESS_BOUND', 'PROCESS_TERMINAL',
        'PROCESS_ABORTED'
    )),
    reservation_id TEXT,
    occurred_at REAL NOT NULL
);
CREATE TABLE session_process_reservations (
    reservation_id TEXT PRIMARY KEY,
    reservation_token_sha256 TEXT NOT NULL CHECK (
        length(reservation_token_sha256) = 64
        AND reservation_token_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    session_id TEXT NOT NULL,
    session_generation INTEGER NOT NULL CHECK (session_generation >= 1),
    state_db_id TEXT NOT NULL CHECK (
        length(state_db_id) = 64 AND state_db_id NOT GLOB '*[^0-9a-f]*'
    ),
    state_family TEXT NOT NULL CHECK (state_family = 'sessiondb-v1'),
    status TEXT NOT NULL CHECK (status IN ('RESERVED', 'BOUND', 'TERMINAL', 'ABORTED')),
    reserved_at REAL NOT NULL,
    expires_at REAL NOT NULL CHECK (expires_at > reserved_at),
    consumed_at REAL
);
CREATE TRIGGER session_process_authority_identity_immutable_update
BEFORE UPDATE ON state_meta
WHEN OLD.key IN ('session_process_state_db_id', 'session_process_state_family')
BEGIN
    SELECT RAISE(ABORT, 'session process authority identity is immutable');
END;
CREATE TRIGGER session_process_authority_identity_immutable_delete
BEFORE DELETE ON state_meta
WHEN OLD.key IN ('session_process_state_db_id', 'session_process_state_family')
BEGIN
    SELECT RAISE(ABORT, 'session process authority identity is immutable');
END;
CREATE TRIGGER session_process_authority_generation_insert_guard
BEFORE INSERT ON sessions
WHEN NEW.session_generation <> 0
BEGIN
    SELECT RAISE(ABORT, 'session authority generation must start at zero');
END;
CREATE TRIGGER session_process_authority_generation_monotonic_guard
BEFORE UPDATE OF session_generation ON sessions
WHEN NEW.session_generation <> OLD.session_generation + 1
BEGIN
    SELECT RAISE(ABORT, 'session authority generation must be monotonic');
END;
CREATE TRIGGER session_process_authority_sessions_issue
AFTER INSERT ON sessions
BEGIN
    UPDATE sessions
    SET session_generation = CASE
        WHEN session_generation < 1 THEN 1 ELSE session_generation END
    WHERE id = NEW.id;
    INSERT INTO session_process_authorities (
        session_id, session_generation, state_db_id, state_family,
        authority_token, status, issued_at
    )
    SELECT s.id, s.session_generation,
           (SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'),
           (SELECT value FROM state_meta WHERE key = 'session_process_state_family'),
           lower(hex(randomblob(32))), 'ISSUED', s.started_at
    FROM sessions AS s WHERE s.id = NEW.id;
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family,
        event_type, occurred_at
    )
    SELECT session_id, session_generation, state_db_id, state_family,
           'SESSION_ISSUED', issued_at
    FROM session_process_authorities
    WHERE session_id = NEW.id
      AND session_generation = (SELECT session_generation FROM sessions WHERE id = NEW.id);
END;
CREATE TRIGGER session_process_authority_sessions_close
AFTER UPDATE OF ended_at ON sessions
WHEN OLD.ended_at IS NULL AND NEW.ended_at IS NOT NULL
BEGIN
    UPDATE session_process_authorities
    SET status = CASE WHEN NEW.end_reason = 'authority_revoked'
                      THEN 'REVOKED' ELSE 'CLOSED' END,
        terminal_at = NEW.ended_at
    WHERE session_id = NEW.id
      AND session_generation = NEW.session_generation
      AND status = 'ISSUED';
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family,
        event_type, occurred_at
    )
    SELECT session_id, session_generation, state_db_id, state_family,
           CASE WHEN NEW.end_reason = 'authority_revoked'
                THEN 'SESSION_REVOKED' ELSE 'SESSION_CLOSED' END,
           NEW.ended_at
    FROM session_process_authorities
    WHERE session_id = NEW.id
      AND session_generation = NEW.session_generation
      AND status = CASE WHEN NEW.end_reason = 'authority_revoked'
                        THEN 'REVOKED' ELSE 'CLOSED' END;
END;
CREATE TRIGGER session_process_authority_sessions_reopen
AFTER UPDATE OF ended_at ON sessions
WHEN OLD.ended_at IS NOT NULL AND NEW.ended_at IS NULL
BEGIN
    UPDATE sessions SET session_generation = session_generation + 1 WHERE id = NEW.id;
    INSERT INTO session_process_authorities (
        session_id, session_generation, state_db_id, state_family,
        authority_token, status, issued_at
    )
    SELECT s.id, s.session_generation,
           (SELECT value FROM state_meta WHERE key = 'session_process_state_db_id'),
           (SELECT value FROM state_meta WHERE key = 'session_process_state_family'),
           lower(hex(randomblob(32))), 'ISSUED', strftime('%s', 'now')
    FROM sessions AS s WHERE s.id = NEW.id;
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family,
        event_type, occurred_at
    )
    SELECT session_id, session_generation, state_db_id, state_family,
           'SESSION_ISSUED', issued_at
    FROM session_process_authorities
    WHERE session_id = NEW.id
      AND session_generation = (SELECT session_generation FROM sessions WHERE id = NEW.id);
END;
CREATE TRIGGER session_process_reservation_issue
AFTER INSERT ON session_process_reservations
BEGIN
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family,
        event_type, reservation_id, occurred_at
    ) VALUES (
        NEW.session_id, NEW.session_generation, NEW.state_db_id, NEW.state_family,
        'PROCESS_RESERVATION', NEW.reservation_id, NEW.reserved_at
    );
END;
CREATE TRIGGER session_process_reservation_transition_guard
BEFORE UPDATE OF status ON session_process_reservations
WHEN NOT (
    (OLD.status = 'RESERVED' AND NEW.status IN ('BOUND', 'ABORTED'))
    OR (OLD.status = 'BOUND' AND NEW.status IN ('TERMINAL', 'ABORTED'))
)
BEGIN
    SELECT RAISE(ABORT, 'invalid session process reservation transition');
END;
CREATE TRIGGER session_process_reservation_transition_event
AFTER UPDATE OF status ON session_process_reservations
WHEN OLD.status <> NEW.status
BEGIN
    INSERT INTO session_process_authority_events (
        session_id, session_generation, state_db_id, state_family,
        event_type, reservation_id, occurred_at
    ) VALUES (
        NEW.session_id, NEW.session_generation, NEW.state_db_id, NEW.state_family,
        CASE NEW.status WHEN 'BOUND' THEN 'PROCESS_BOUND'
                        WHEN 'TERMINAL' THEN 'PROCESS_TERMINAL'
                        ELSE 'PROCESS_ABORTED' END,
        NEW.reservation_id,
        COALESCE(NEW.consumed_at, strftime('%s', 'now'))
    );
END;
CREATE TRIGGER session_process_authority_events_append_only_update
BEFORE UPDATE ON session_process_authority_events
BEGIN
    SELECT RAISE(ABORT, 'session process authority events are append-only');
END;
CREATE TRIGGER session_process_authority_events_append_only_delete
BEFORE DELETE ON session_process_authority_events
BEGIN
    SELECT RAISE(ABORT, 'session process authority events are append-only');
END;
CREATE TABLE gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);
CREATE TABLE gateway_hygiene_state (
    session_key TEXT PRIMARY KEY,
    failure_streak INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE session_turn_leases (
    conversation_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE TABLE turn_receipts (
    turn_request_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    binding_digest TEXT NOT NULL CHECK (binding_digest <> ''),
    receipt_identity_json TEXT,
    receipt_identity_digest TEXT,
    target_bind_receipt_json TEXT,
    target_bind_receipt_digest TEXT,
    status TEXT NOT NULL,
    claim_token TEXT,
    terminal_message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    response_digest TEXT,
    abort_receipt_id TEXT,
    abort_evidence_digest TEXT,
    abort_reason_code TEXT,
    created_at REAL NOT NULL,
    claimed_at REAL,
    completed_at REAL,
    aborted_at REAL
);
CREATE TABLE async_delegations (
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
, origin_session_id TEXT);
CREATE INDEX idx_sessions_source ON sessions(source);
CREATE INDEX idx_sessions_source_id ON sessions(source, id);
CREATE INDEX idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX idx_messages_session_id ON messages(session_id, id);
CREATE INDEX idx_messages_assistant_calls_by_session
    ON messages(session_id)
    WHERE role = 'assistant' AND tool_calls IS NOT NULL;
CREATE INDEX idx_compression_locks_expires ON compression_locks(expires_at);
CREATE INDEX idx_session_turn_leases_expires ON session_turn_leases(expires_at);
CREATE INDEX idx_session_model_usage_session ON session_model_usage(session_id);
CREATE INDEX idx_session_model_usage_model ON session_model_usage(model);
CREATE INDEX idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);
CREATE INDEX idx_session_process_authorities_current
    ON session_process_authorities(session_id, session_generation DESC);
CREATE INDEX idx_session_process_reservations_current
    ON session_process_reservations(session_id, session_generation, status, expires_at);
CREATE TRIGGER turn_fence_messages_insert BEFORE INSERT ON messages BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_messages_update BEFORE UPDATE ON messages BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_messages_delete BEFORE DELETE ON messages BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_sessions_insert BEFORE INSERT ON sessions BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_sessions_update BEFORE UPDATE ON sessions BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_sessions_delete BEFORE DELETE ON sessions BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_system_prompts_insert BEFORE INSERT ON system_prompts BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_system_prompts_update BEFORE UPDATE ON system_prompts BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_system_prompts_delete BEFORE DELETE ON system_prompts BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_model_usage_insert BEFORE INSERT ON session_model_usage BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_model_usage_update BEFORE UPDATE ON session_model_usage BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_model_usage_delete BEFORE DELETE ON session_model_usage BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_turn_leases_insert BEFORE INSERT ON session_turn_leases BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_turn_leases_update BEFORE UPDATE ON session_turn_leases BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_turn_leases_delete BEFORE DELETE ON session_turn_leases BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_compression_locks_insert BEFORE INSERT ON compression_locks BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_compression_locks_update BEFORE UPDATE ON compression_locks BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_compression_locks_delete BEFORE DELETE ON compression_locks BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_gateway_routing_insert BEFORE INSERT ON gateway_routing BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_gateway_routing_update BEFORE UPDATE ON gateway_routing BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_gateway_routing_delete BEFORE DELETE ON gateway_routing BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_async_delegations_insert BEFORE INSERT ON async_delegations BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_async_delegations_update BEFORE UPDATE ON async_delegations BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_async_delegations_delete BEFORE DELETE ON async_delegations BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_process_authorities_insert BEFORE INSERT ON session_process_authorities BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_process_authorities_update BEFORE UPDATE ON session_process_authorities BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_process_authorities_delete BEFORE DELETE ON session_process_authorities BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_process_reservations_insert BEFORE INSERT ON session_process_reservations BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_process_reservations_update BEFORE UPDATE ON session_process_reservations BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE TRIGGER turn_fence_session_process_reservations_delete BEFORE DELETE ON session_process_reservations BEGIN SELECT CASE WHEN typeof(hermes_turn_fence_generation()) != 'integer' OR hermes_turn_fence_generation() != 29 THEN RAISE(ABORT, 'state DB generation incompatible') END; END;
CREATE INDEX idx_messages_platform_msg_id ON messages(session_id, platform_message_id) WHERE platform_message_id IS NOT NULL;
CREATE INDEX idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX idx_messages_active_null
    ON messages(active) WHERE active IS NULL;
CREATE INDEX idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
CREATE INDEX idx_sessions_system_prompt_hash
    ON sessions(system_prompt_hash);
CREATE INDEX idx_turn_receipts_session_status
    ON turn_receipts(session_id, status, created_at);
CREATE INDEX idx_turn_receipts_completed_prune
    ON turn_receipts(status, completed_at);
CREATE UNIQUE INDEX idx_sessions_title_unique ON sessions(title) WHERE title IS NOT NULL;
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);
CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages
WHEN (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;
CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages
WHEN (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;
CREATE TRIGGER messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;
CREATE VIEW messages_fts_trigram_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';
CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_trigram_src',
    content_rowid='id',
    tokenize='trigram'
);
CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;
CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;
CREATE TRIGGER messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
