"""An upstream SCHEMA_VERSION bump moves the fenced lineage forward.

A store fenced by the previous build of this line is an older generation of the same lineage:
the next build migrates it to its own generation instead of refusing it as foreign. A stamp
above the new generation still refuses, untouched.
"""

from __future__ import annotations

import pytest

from hermes_state_errors import SCHEMA_CAUSE_BUILD_TOO_OLD, IncompatibleSchemaError
from tests.hermes_state.fence_store_probe import expected_fences, fence_triggers, read_ro, stored
from tests.hermes_state.fork_store_fixture import (
    CURRENT_STAMP, _connect_as, build_store, file_fingerprint, isolate_home, refence,
)


def _bump_schema_version(monkeypatch) -> int:
    """Patch the bindings the fence code reads at call time, as the next upstream release would set them."""
    import hermes_state_common
    import hermes_state_fence

    bumped = hermes_state_common.SCHEMA_VERSION + 1
    stored_version = hermes_state_fence.FENCE_LINEAGE_BASE + bumped
    monkeypatch.setattr(hermes_state_common, "SCHEMA_VERSION", bumped)
    monkeypatch.setattr(hermes_state_fence, "SCHEMA_VERSION", bumped)
    monkeypatch.setattr(hermes_state_fence, "STORED_SCHEMA_VERSION", stored_version)
    monkeypatch.setattr(hermes_state_fence, "TURN_FENCE_GENERATION", stored_version)
    return stored_version


def test_a_store_fenced_by_the_previous_generation_migrates_forward(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fenced1030")
    assert stored(db_path) == CURRENT_STAMP
    bumped_stored = _bump_schema_version(monkeypatch)

    db = SessionDB(db_path=db_path)
    try:
        db.create_session("after-bump", "cli")
        db.append_message("after-bump", "user", "written at the bumped generation")
    finally:
        db.close()

    assert stored(db_path) == bumped_stored
    assert fence_triggers(db_path) == expected_fences(db_path, bumped_stored)
    assert read_ro(db_path, "SELECT COUNT(*) FROM messages WHERE session_id = 'after-bump'")[0][0] == 1


# The stamp alone must refuse: fences at this build's generation do not vouch for a newer stamp.
@pytest.mark.parametrize("fence_offset", [1, 0], ids=["future-fences", "current-fences"])
def test_a_stamp_above_the_bumped_generation_still_refuses_untouched(tmp_path, monkeypatch, fence_offset):
    from hermes_state import SessionDB

    db_path = build_store(isolate_home(tmp_path, monkeypatch) / "state.db", "fenced1030")
    bumped_stored = _bump_schema_version(monkeypatch)
    conn = _connect_as(db_path, None)
    try:
        refence(conn, bumped_stored + fence_offset)
        conn.execute("UPDATE schema_version SET version = ?", (bumped_stored + 1,))
    finally:
        conn.close()
    before = file_fingerprint(db_path)

    with pytest.raises(IncompatibleSchemaError) as refused:
        SessionDB(db_path=db_path)

    assert refused.value.cause == SCHEMA_CAUSE_BUILD_TOO_OLD
    assert file_fingerprint(db_path) == before
