"""The lean summary's user-messages section says what it dropped and what it did not re-check.

``ContextCompressor._augment_summary_lean`` quotes the compacted region's real user messages. When the
char budget omits or truncates some, the note counts them and names the ``session_search`` path instead
of claiming every message is quoted verbatim. An iterative round's summary can echo the previous round's
section back, once or more: every echoed copy is removed, its quotes are carried forward under a label
saying they were not re-verified, and this round's quotes are always added.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import (
    _LEAN_USER_MESSAGE_MAX_CHARS,
    _LEAN_USER_MESSAGES_BUDGET_CHARS,
    _LEAN_USER_MESSAGES_HEADING,
    SUMMARY_PREFIX,
    ContextCompressor,
)

VERBATIM = "(Every real user message from the compacted region, quoted verbatim."
BOUNDED = "(A bounded selection, not every user message:"
CARRIED = "### Earlier user messages (from a previous summary, not re-verified)"


@pytest.fixture
def compressor():
    c = ContextCompressor(model="test/model", threshold_percent=0.85, protect_first_n=2, protect_last_n=2,
                          quiet_mode=True, tail_mode="lean")
    c._session_id = "sess-1"
    return c


def _users(*texts):
    return [m for text in texts for m in ({"role": "user", "content": text}, {"role": "assistant", "content": "ok"})]


def _section(summary):
    start = summary.index(_LEAN_USER_MESSAGES_HEADING)
    end = summary.find("\n## ", start + 1)
    return summary[start:end if end != -1 else len(summary)]


# ── the honesty note ─────────────────────────────────────────────────────────────────────────────────


def test_omitted_and_truncated_messages_are_counted_with_a_session_search_hint(compressor):
    long_ask = "keep the API stable " * 400  # over the per-message cap: truncated
    fillers = [f"filler {i} " + "x" * (_LEAN_USER_MESSAGE_MAX_CHARS - 20) for i in range(8)]
    turns = _users("the oldest ask", *fillers, long_ask)  # newest last
    assert len(long_ask) > _LEAN_USER_MESSAGE_MAX_CHARS
    assert sum(map(len, fillers)) + _LEAN_USER_MESSAGE_MAX_CHARS > _LEAN_USER_MESSAGES_BUDGET_CHARS

    section = _section(compressor._augment_summary_lean("## Goal\nship it", turns))

    assert VERBATIM not in section
    assert BOUNDED in section
    quoted = section.count("\n> ")
    omitted = 10 - quoted
    assert omitted >= 2 and "the oldest ask" not in section  # reach: the budget omitted the oldest
    assert f"{quoted} of 10 real user messages" in section
    assert f"{omitted} older message(s) omitted" in section and "1 truncated" in section
    assert "session_search(query='<keywords>', session_id='sess-1')" in section


def test_without_a_session_id_the_hint_names_session_search_alone(compressor):
    compressor._session_id = ""
    section = _section(compressor._augment_summary_lean("## Goal\nship it", _users("q" * 30_000)))

    assert BOUNDED in section and "0 older message(s) omitted, 1 truncated" in section
    assert "recover omitted or truncated messages with session_search before" in section


def test_control_a_complete_selection_keeps_the_verbatim_note(compressor):
    section = _section(compressor._augment_summary_lean("## Goal\nship it", _users("first ask", "second ask")))

    assert VERBATIM in section and BOUNDED not in section
    assert section.index("> second ask") < section.index("> first ask")  # newest first


# ── echoed sections ──────────────────────────────────────────────────────────────────────────────────


def _echo(*quotes, note=VERBATIM + " These are the user's actual words and override any paraphrase of them above.)"):
    return "\n\n" + _LEAN_USER_MESSAGES_HEADING + "\n" + "\n\n".join(f"> {q}" for q in quotes) + "\n" + note


def test_a_carried_forward_section_is_relabelled_and_this_rounds_quotes_are_added(compressor):
    summary = "## Goal\nship it" + _echo("always use tabs") + "\n\n## Next Steps\nrun tests"

    out = compressor._augment_summary_lean(summary, _users("fix the login bug"))

    assert out.count(_LEAN_USER_MESSAGES_HEADING) == 1
    section = _section(out)
    current, _, carried = section.partition(CARRIED)
    assert "> fix the login bug" in current and VERBATIM in current
    assert "always use tabs" not in current  # never re-certified as this round's words
    assert "> always use tabs" in carried and VERBATIM not in carried
    assert "## Next Steps\nrun tests" in out  # the rest of the summary is kept


def test_every_echoed_section_is_extracted_not_only_the_first(compressor):
    summary = ("## Goal\nship it" + _echo("always use tabs") + "\n\n## Progress\nhalf done"
               + _echo("always use tabs", "never push to main", note=BOUNDED + " 2 of 3 ...)"))

    out = compressor._augment_summary_lean(summary, _users("fix the login bug"))

    assert out.count(_LEAN_USER_MESSAGES_HEADING) == 1 and out.count(CARRIED) == 1
    assert "## Progress\nhalf done" in out
    carried = _section(out).partition(CARRIED)[2]
    assert carried.count("> always use tabs") == 1 and "> never push to main" in carried
    assert out.count(VERBATIM) == 1 and BOUNDED not in out  # only this round's note certifies anything


def test_an_iterative_round_trip_carries_the_previous_round_once(compressor):
    first = compressor._augment_summary_lean("## Goal\nship it", _users("always use tabs"))
    second = compressor._augment_summary_lean(first, _users("fix the login bug"))
    third = compressor._augment_summary_lean(second, _users("add a test"))

    assert third.count(_LEAN_USER_MESSAGES_HEADING) == 1 and third.count(CARRIED) == 1
    current, _, carried = _section(third).partition(CARRIED)
    assert "> add a test" in current and "fix the login bug" not in current
    assert carried.index("> fix the login bug") < carried.index("> always use tabs")  # newest first


def test_a_heading_quoted_inside_another_section_is_not_a_section(compressor):
    goal = f"## Goal\nthe user pasted:\n> {_LEAN_USER_MESSAGES_HEADING}\n> stale\nthen asked to ship it"

    out = compressor._augment_summary_lean(goal + _echo("always use tabs"), _users("next"))

    assert out.startswith(goal)  # the quoted heading and the rest of its section are untouched
    assert out.count(f"\n{_LEAN_USER_MESSAGES_HEADING}\n") == 1


def test_carried_quotes_are_bounded(compressor):
    old = [f"standing rule {i} " + "r" * 3_000 for i in range(20)]
    out = compressor._augment_summary_lean("## Goal\nship it" + _echo(*old), _users("next"))

    carried = _section(out).partition(CARRIED)[2]
    assert len(carried) < _LEAN_USER_MESSAGES_BUDGET_CHARS + 1_000
    assert "> standing rule 0 " in carried and "standing rule 19 " not in carried  # newest kept first
    kept = carried.count("> standing rule ")
    assert f"{kept} of 20 carried quote(s) kept" in carried


def test_the_reader_display_drops_the_carried_quotes_with_the_section(compressor):
    from agent.compaction_display import compaction_summary

    out = compressor._augment_summary_lean("## Goal\nship it" + _echo("always use tabs"), _users("fix it"))
    shown = compaction_summary({"role": "user", "_compressed_summary": True, "content": SUMMARY_PREFIX + "\n" + out})

    assert shown is not None and "always use tabs" not in shown["summary"] and "fix it" not in shown["summary"]
