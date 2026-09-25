"""Conservative ownership proof for persisted turn-fence trigger declarations.

A call to the fence function or a ``turn_fence_`` name makes a trigger a
candidate, never proof of ownership. Only a declaration that is token-exact to
the builder output (name included, lexical trivia excluded) for a generation
this build recognises is owned; anything else refuses the store untouched.
"""

from __future__ import annotations

import re

from hermes_state_errors import (
    SCHEMA_CAUSE_BUILD_TOO_OLD, SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, IncompatibleSchemaError,
)

# Strings stay opaque and comments are dropped before calls are recognised.
# Quoted identifiers remain identifiers; string literal contents keep their case.
_TOKEN = re.compile(
    r"--[^\n]*(?:\n|$)|/\*.*?(?:\*/|$)|"
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]|"
    r"[\w$]+|!=|<>|\S",
    re.DOTALL,
)
_LITERAL_SLOT = 0


def _tokens(sql: str) -> tuple:
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


def _templates() -> dict:
    """{canonical trigger name: (template tokens, index of the generation literal)}."""
    from hermes_state_fence import ALL_GOVERNED_TABLES, FENCE_OPERATIONS, turn_fence_trigger_name, turn_fence_trigger_sql

    templates = {}
    for table in ALL_GOVERNED_TABLES:
        for operation in FENCE_OPERATIONS:
            tokens = _tokens(turn_fence_trigger_sql(table, operation, generation=_LITERAL_SLOT))
            slot = tokens.index(("word", str(_LITERAL_SLOT)))
            templates[turn_fence_trigger_name(table, operation)] = (tokens, slot)
    return templates


def _declared_literal(tokens: tuple, template: tuple, slot: int):
    if len(tokens) != len(template):
        return None
    for index, (token, expected) in enumerate(zip(tokens, template)):
        if index == slot:
            if token[0] != "word" or not token[1].isdigit():
                return None
        elif token != expected:
            return None
    return int(tokens[slot][1])


def owned_turn_fence_literals(cursor) -> dict:
    """Return ``{trigger name: generation literal}`` for every owned fence trigger.

    Raises :class:`IncompatibleSchemaError` for any candidate that is not owned:
    BUILD_TOO_OLD when it is an exact fence for a generation above this build's,
    FENCE_GENERATION_MISMATCH otherwise (unknown body, name collision, or a
    generation this build never wrote)."""
    from hermes_state_fence import (
        FENCE_LINEAGE_BASE, FORK_BASE_UPSTREAM_GATE, FORK_LEGACY_GENERATIONS, SESSION_PROCESS_GOVERNED_TABLES,
        STORED_SCHEMA_VERSION, TURN_FENCE_FUNCTION, TURN_FENCE_GENERATION, schema_text,
    )

    templates = _templates()
    authority_names = {
        name for name in templates
        if any(name.startswith(f"turn_fence_{table}_") for table in SESSION_PROCESS_GOVERNED_TABLES)
    }
    owned = {}
    for raw_name, raw_sql in cursor.execute(
            "SELECT CAST(name AS BLOB), CAST(sql AS BLOB) FROM sqlite_master WHERE type = 'trigger'").fetchall():
        name, tokens = schema_text(raw_name), _tokens(schema_text(raw_sql))
        lowered = name.lower()
        calls = any(
            token == ("word", TURN_FENCE_FUNCTION) and following == ("word", "(")
            for token, following in zip(tokens, tokens[1:])
        )
        if not (calls or lowered.startswith("turn_fence_")):
            continue
        template = templates.get(lowered)
        literal = _declared_literal(tokens, *template) if template else None
        if literal is None:
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=TURN_FENCE_GENERATION,
                actual_generation=None, detail=f"unowned trigger declaration {name}",
            )
        if literal > STORED_SCHEMA_VERSION:
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_BUILD_TOO_OLD, expected_generation=TURN_FENCE_GENERATION,
                actual_generation=literal, detail=f"trigger {name}",
            )
        # The fork first governed the authority tables at generation 28.
        legacy_ok = literal in FORK_LEGACY_GENERATIONS and not (literal == 27 and lowered in authority_names)
        # An earlier fenced build of this line (before an upstream SCHEMA_VERSION bump): migrated forward.
        older_fenced = FENCE_LINEAGE_BASE + FORK_BASE_UPSTREAM_GATE < literal < STORED_SCHEMA_VERSION
        if literal != STORED_SCHEMA_VERSION and not (legacy_ok or older_fenced):
            raise IncompatibleSchemaError(
                cause=SCHEMA_CAUSE_FENCE_GENERATION_MISMATCH, expected_generation=TURN_FENCE_GENERATION,
                actual_generation=literal, detail=f"trigger {name}",
            )
        owned[name] = literal
    return owned
