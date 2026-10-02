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


def test_an_echoed_previous_user_section_is_replaced_not_trusted():
    """R1-2: an iterative round hands the model its own previous summary (already carrying this method's
    '## User Messages' section) as the "PREVIOUS SUMMARY" to preserve, and the model can echo that heading
    straight through with its old, short content underneath. Trusting the heading's mere presence would
    then skip rebuilding the bounded-selection disclosure for the turns actually being compacted THIS
    round, letting a stale "quoted verbatim" claim ride forward indefinitely."""
    with patch("agent.context_compressor.get_model_context_length", return_value=272_000):
        c = ContextCompressor(model="main-model", quiet_mode=True, tail_mode="lean")
    c._session_id = SID

    # Round 1: a short, genuinely-complete prior section.
    prior_turns = [_user("use tabs"), {"role": "assistant", "content": "ok"}]
    carried_forward = c._augment_summary_lean("summary body", prior_turns)
    assert "verbatim" in carried_forward.lower()

    # Round 2: the model echoed round 1's heading/content back verbatim (exactly what "PRESERVE all
    # existing information that is still relevant" invites), but THIS round's turns are the
    # bounded-selection case — the 24,000-char budget only holds the newest 8 of 12.
    turns = []
    for i in range(12):
        turns.append(_user(f"constraint {i:02d}: " + "x" * 2_985))
        turns.append({"role": "assistant", "content": "ok"})

    rebuilt = c._augment_summary_lean(carried_forward, turns)
    assert rebuilt.count("## User Messages") == 1  # replaced, not duplicated alongside the stale one
    start = rebuilt.index("## User Messages")
    end = rebuilt.find("\n## ", start + 1)
    section = rebuilt[start:] if end == -1 else rebuilt[start:end]

    assert "constraint 00" not in section  # the oldest really is left out...
    assert "use tabs" not in section  # ...and round 1's stale content does not ride forward
    assert "every real user message" not in section.lower()
    assert "8 of 12" in section and "4 older" in section
