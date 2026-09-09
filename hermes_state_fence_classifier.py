"""Conservative ownership proof for persisted generation-fence declarations.

A function call makes SQL a candidate, never permission to drop it. Only a
complete shipped declaration (apart from its name and lexical trivia) is owned.
"""

import re
import sqlite3

from hermes_state_common import TURN_FENCE_GENERATION, turn_fence_trigger_definitions

# Keep strings opaque and discard comments before recognizing calls. Quoted
# identifiers remain identifiers; do not lowercase string literal contents.
_TOKEN = re.compile(
    r"--[^\n]*(?:\n|$)|/\*.*?(?:\*/|$)|"
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|"
    r"[\w$]+|!=|<>|\S",
    re.DOTALL,
)
_FUNCTION = "hermes_turn_fence_generation"


def _tokens(sql):
    result = []
    for match in _TOKEN.finditer(sql):
        text = match.group()
        if text.startswith(("--", "/*")):
            continue
        if text[0] == "'":
            result.append(("string", text[1:-1].replace("''", "'")))
        elif text[0] in ('"', "`", "["):
            quote = text[0]
            value = text[1:-1]
            if quote != "[":
                value = value.replace(quote * 2, quote)
            result.append(("word", value.lower()))
        else:
            result.append(("word", text.lower()))
    return tuple(result)


def owned_turn_fence_triggers(cursor):
    """Return proven owned rows, refusing unknown calls/name collisions first."""
    definitions = turn_fence_trigger_definitions()
    names = {name.lower() for name, _sql in definitions}
    known = set()
    for name, sql in definitions:
        for generation in {27, 28, TURN_FENCE_GENERATION}:
            # Authority tables were first governed in v28.
            if generation == 27 and name.startswith("turn_fence_session_process_"):
                continue
            tokens = _tokens(
                sql.replace(f"!= {TURN_FENCE_GENERATION} ", f"!= {generation} ")
            )
            known.add(tokens[:2] + tokens[3:])
    owned = {}
    for name, sql in cursor.execute(
        "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
    ).fetchall():
        tokens = _tokens(sql or "")
        calls = any(
            token[1].lower() == _FUNCTION and following == ("word", "(")
            for token, following in zip(tokens, tokens[1:])
        )
        if calls or name.lower() in names:
            # sqlite_master omits IF NOT EXISTS; require the complete remaining
            # table, operation, condition, body and generation, not a prefix.
            if len(tokens) < 3 or tokens[:2] + tokens[3:] not in known:
                raise sqlite3.DatabaseError(
                    "TURN_FENCE_MIGRATION_REFUSED: unowned trigger declaration"
                )
            owned[name] = sql
    return owned
