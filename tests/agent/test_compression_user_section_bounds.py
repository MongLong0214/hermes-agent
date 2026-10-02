"""L2-3: the lean handoff's user-message section must not certify a bounded selection as complete and verbatim.

The section keeps at most ~24,000 chars of the compacted region's real user messages (newest first) and cuts
each at 4,000, yet it told the next model that EVERY real user message was quoted verbatim. An older
constraint could be missing from the handoff while the handoff asserted completeness.

Checked against the text ``_augment_summary_lean`` appends, which both the LLM summary and the deterministic
fallback go through.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor

SID = "SESSION-L2-3"


def _section(turns):
    with patch("agent.context_compressor.get_model_context_length", return_value=272_000):
        c = ContextCompressor(model="main-model", quiet_mode=True, tail_mode="lean")
    c._session_id = SID
    text = c._augment_summary_lean("summary body", turns)
    start = text.index("## User Messages")
    end = text.find("\n## ", start + 1)
    return text[start:] if end == -1 else text[start:end]


def _user(content):
    return {"role": "user", "content": content}


def test_total_budget_states_how_many_older_messages_were_omitted():
    turns = []
    for i in range(12):  # 12 x 3,000 chars: the 24,000-char budget holds the newest 8
        turns.append(_user(f"constraint {i:02d}: " + "x" * 2_985))
        turns.append({"role": "assistant", "content": "ok"})

    section = _section(turns)

    assert "constraint 00" not in section  # the oldest really is left out...
    assert "every real user message" not in section.lower()  # ...so completeness must not be claimed
    assert "verbatim" not in section.lower()
    assert "8 of 12" in section and "4 older" in section
    assert f"session_search(query='<keywords>', session_id='{SID}')" in section


def test_per_message_limit_is_not_called_verbatim():
    section = _section([_user("first line\n" + "y" * 5_000), {"role": "assistant", "content": "ok"}])

    assert "…[truncated]" in section
    assert "verbatim" not in section.lower()
    assert "1 truncated" in section
    assert "session_search" in section


def test_complete_short_selection_keeps_the_verbatim_guarantee():
    section = _section([_user("use tabs"), {"role": "assistant", "content": "ok"}, _user("no new deps")])

    assert "verbatim" in section.lower()
    assert "omitted" not in section and "truncated" not in section
    assert section.index("no new deps") < section.index("use tabs")  # newest first
