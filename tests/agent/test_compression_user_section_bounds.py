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


def test_two_echoed_prior_user_sections_both_become_historical_with_no_stale_verbatim_claim():
    """R1-2: ``_replace_lean_section``'s single ``str.find`` only ever removed the FIRST occurrence of
    the heading. A reply that echoes the previous summary can carry the heading back more than once
    (e.g. the model quotes its own prior turn verbatim inside a longer response, or nests the historical
    section the fix below introduces alongside a stray duplicate of the current one). Every occurrence
    must be scoped as historical — not just the first — and the certified "quoted verbatim" claim must
    never survive attached to content that is no longer this round's."""
    with patch("agent.context_compressor.get_model_context_length", return_value=272_000):
        c = ContextCompressor(model="main-model", quiet_mode=True, tail_mode="lean")
    c._session_id = SID

    round1 = c._augment_summary_lean("summary body", [_user("use tabs"), {"role": "assistant", "content": "ok"}])
    # Simulate the model echoing the SAME current-heading section back twice in its reply (two sibling
    # occurrences of "## User Messages (newest first)" in the text handed to round 2).
    start = round1.index("## User Messages")
    echoed_twice = round1 + "\n\n" + round1[start:]

    turns = []
    for i in range(12):
        turns.append(_user(f"constraint {i:02d}: " + "x" * 2_985))
        turns.append({"role": "assistant", "content": "ok"})
    rebuilt = c._augment_summary_lean(echoed_twice, turns)

    assert rebuilt.count("## User Messages (newest first)") == 1  # exactly one CURRENT section
    cur_start = rebuilt.index("## User Messages (newest first)")
    cur_end = rebuilt.find("\n## ", cur_start + 1)
    current_section = rebuilt[cur_start:] if cur_end == -1 else rebuilt[cur_start:cur_end]
    assert "every real user message" not in current_section.lower()
    assert "verbatim" not in current_section.lower()  # no stale certification over historical text
    assert "use tabs" not in current_section  # historical content does not leak into the current claim

    assert "Earlier User Messages" in rebuilt
    hist_start = rebuilt.index("## Earlier User Messages")
    hist_end = rebuilt.find("\n## ", hist_start + 1)
    historical_section = rebuilt[hist_start:] if hist_end == -1 else rebuilt[hist_start:hist_end]
    assert historical_section.count("use tabs") == 2  # BOTH echoed occurrences preserved, not dropped
    assert "quoted verbatim" not in historical_section.lower()


def test_constraint_only_carried_in_an_old_user_section_survives_remediation():
    """R3-1: ``_replace_lean_section`` discarded the old section's contents outright and rebuilt from the
    current turns alone, so a standing constraint ("Always use tabs for indentation in this repository")
    preserved by the model ONLY in a previous round's user section was deleted the moment a later round's
    turns ("Add comments to the parser.") no longer repeated it. The constraint must survive, carried
    forward as historical content, rather than vanish with the stale certification that is correctly
    removed."""
    with patch("agent.context_compressor.get_model_context_length", return_value=272_000):
        c = ContextCompressor(model="main-model", quiet_mode=True, tail_mode="lean")
    c._session_id = SID

    round1 = c._augment_summary_lean(
        "summary body", [_user("Always use tabs for indentation in this repository")]
    )
    assert "Always use tabs for indentation" in round1

    round2 = c._augment_summary_lean(round1, [_user("Add comments to the parser."), {"role": "assistant", "content": "ok"}])

    assert "Always use tabs for indentation in this repository" in round2  # preserved, not thrown away
    assert "Add comments to the parser." in round2
    # The certification note must not be re-applied to the carried-forward constraint.
    cur_start = round2.index("## User Messages (newest first)")
    cur_end = round2.find("\n## ", cur_start + 1)
    current_section = round2[cur_start:] if cur_end == -1 else round2[cur_start:cur_end]
    assert "Always use tabs for indentation" not in current_section
