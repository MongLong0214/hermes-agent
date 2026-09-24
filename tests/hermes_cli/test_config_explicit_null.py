"""An explicit YAML ``null`` the user wrote survives every ``save_config`` write.

Regression for migration-drill finding B3: ``_strip_default_values`` used ``None`` both as a
value and as its "stripped" marker, so a save (every migration step persists through
``save_config``) deleted explicit nulls. Where null and absent mean different things, the
config silently changed meaning: the closed canonical-binding schema rejected the binding,
and a null written to disable a feature brought the default back. The converse also holds:
a migration that retires a legacy key removes it by presence, whatever its value.
"""

import pytest
import yaml

from gateway.config import load_gateway_config
from hermes_cli.config import (
    DEFAULT_CONFIG,
    get_config_path,
    load_config,
    migrate_config,
    read_raw_config,
    save_config,
    validate_config_structure,
)
from hermes_cli.doctor_config import collect_deprecated_config_keys


def _write_config(document: dict) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _nested(path: tuple, value) -> dict:
    document = value
    for key in reversed(path):
        document = {key: document}
    return document


def _at(config: dict, path: tuple):
    for key in path:
        config = config[key]
    return config


def _present(config, path: tuple) -> bool:
    for key in path:
        if not isinstance(config, dict) or key not in config:
            return False
        config = config[key]
    return True


def test_migration_keeps_canonical_binding_with_no_thread_loadable():
    # 33 is the version the live install was on when the drill ran.
    _write_config({
        "_config_version": 33,
        "canonical_surface_bindings": {
            "ceo": {
                "session_key": "agent:main:telegram:dm:chat-placeholder",
                "session_id": "session-placeholder",
                "telegram": {
                    "chat_id": "chat-placeholder",
                    "chat_type": "dm",
                    "user_id": "user-placeholder",
                    "thread_id": None,
                },
                "buzz": {"author_ids": ["author-placeholder"], "channel_ids": ["channel-placeholder"]},
            },
        },
    })
    before = load_gateway_config().canonical_surface_bindings

    migrate_config(interactive=False, quiet=True)

    assert read_raw_config()["_config_version"] == DEFAULT_CONFIG["_config_version"]
    after = load_gateway_config().canonical_surface_bindings
    assert after == before
    assert after["ceo"].telegram_thread_id is None


# Keys whose default is not null but where the user's explicit null has its own meaning
# (off / disabled / ratio-only).
@pytest.mark.parametrize("path", [
    ("runtime", "nofile_soft_limit"),
    ("max_live_sessions",),
    ("prompt_caching", "cache_ttl"),
    ("compression", "threshold_tokens"),
], ids=".".join)
def test_round_trip_save_keeps_null_that_overrides_a_default(path):
    assert _at(DEFAULT_CONFIG, path) is not None
    _write_config(_nested(path, None))
    before = load_config()
    assert _at(before, path) is None

    save_config(read_raw_config())

    after = load_config()
    assert _at(after, path) is None
    assert after == before


# (version the step migrates from, legacy key path). Each step retires the key; written as
# null it carries nothing to fold, so only its presence decides the removal.
@pytest.mark.parametrize("version, path", [
    (16, ("compression", "summary_base_url")),
    (15, ("display", "tool_progress_overrides")),
    (44, ("base_url",)),  # root model key, relocated by the save-time model canon
], ids=lambda v: ".".join(v) if isinstance(v, tuple) else f"v{v}")
def test_migration_removes_legacy_key_written_as_null(version, path):
    _write_config({"_config_version": version, **_nested(path, None)})

    migrate_config(interactive=False, quiet=True)

    raw = read_raw_config()
    assert raw["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert not _present(raw, path)
    assert collect_deprecated_config_keys(raw) == []
    assert [i.message for i in validate_config_structure(raw) if path[-1] in i.message] == []
