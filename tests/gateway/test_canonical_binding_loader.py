"""Closed canonical bindings remain profile-local through the real YAML loader."""

import json

import pytest
import yaml

from gateway.config import GatewayConfig, load_gateway_config
from gateway import config_loader


def _binding(name: str) -> dict:
    return {
        name: {
            "session_key": f"agent:main:telegram:dm:{name}",
            "session_id": f"synthetic-{name}-predecessor",
            "telegram": {
                "chat_id": f"chat-{name}",
                "chat_type": "dm",
                "user_id": f"user-{name}",
                "thread_id": None,
            },
            "buzz": {
                "author_ids": [f"author-{name}"],
                "channel_ids": [f"channel-{name}"],
            },
        }
    }


def test_loader_uses_closed_schema_top_level_precedence_and_keeps_profiles_isolated(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    for home, document in (
        (home_a, {"canonical_surface_bindings": _binding("alpha"), "gateway": {"canonical_surface_bindings": _binding("wrong")}}),
        (home_b, {"gateway": {"canonical_surface_bindings": _binding("beta")}}),
    ):
        home.mkdir()
        (home / "gateway.json").write_text(json.dumps({"canonical_surface_bindings": _binding("legacy")}), encoding="utf-8")
        (home / "config.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    first_a = load_gateway_config().canonical_surface_bindings
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    loaded_b = load_gateway_config().canonical_surface_bindings
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    second_a = load_gateway_config().canonical_surface_bindings

    assert list(first_a) == ["alpha"]
    assert list(loaded_b) == ["beta"]
    assert second_a == first_a
    assert first_a["alpha"].session_key == "agent:main:telegram:dm:alpha"

    with pytest.raises(ValueError, match="schema"):
        GatewayConfig.from_dict({"canonical_surface_bindings": {"bad": {"session_key": "x"}}})


def test_valid_yaml_missing_canonical_binding_clears_legacy_authority_but_preserves_legacy_settings(tmp_path, monkeypatch):
    home = tmp_path / "missing-canonical-home"
    home.mkdir()
    (home / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _binding("legacy"), "session_store_max_age_days": 17}),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(yaml.safe_dump({"gateway": {"loop_watchdog": False}}), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_gateway_config()

    assert config.canonical_surface_bindings == {}
    assert config.session_store_max_age_days == 17
    assert config.loop_watchdog is False


def test_managed_overlay_canonical_binding_replaces_legacy_authority(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    home, managed = tmp_path / "managed-home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    (home / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _binding("legacy"), "session_store_max_age_days": 17}),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(yaml.safe_dump({"gateway": {"loop_watchdog": False}}), encoding="utf-8")
    (managed / "config.yaml").write_text(yaml.safe_dump({"canonical_surface_bindings": _binding("managed")}), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    try:
        config = load_gateway_config()
    finally:
        managed_scope.invalidate_managed_cache()

    assert list(config.canonical_surface_bindings) == ["managed"]
    assert config.session_store_max_age_days == 17
    assert config.loop_watchdog is False


@pytest.mark.parametrize(
    ("user_document", "managed_document", "expected"),
    (
        (
            {"canonical_surface_bindings": _binding("user")},
            {"gateway": {"canonical_surface_bindings": _binding("managed")}},
            _binding("managed"),
        ),
        (
            {"gateway": {"canonical_surface_bindings": _binding("user")}},
            {"canonical_surface_bindings": _binding("managed")},
            _binding("managed"),
        ),
        (
            {"canonical_surface_bindings": _binding("user")},
            {"gateway": {"canonical_surface_bindings": {}}},
            {},
        ),
        (
            {"gateway": {"canonical_surface_bindings": _binding("user")}},
            {"canonical_surface_bindings": {}},
            {},
        ),
    ),
)
def test_managed_canonical_bindings_override_user_regardless_of_spelling(
    tmp_path, monkeypatch, user_document, managed_document, expected
):
    from hermes_cli import managed_scope

    home, managed = tmp_path / "managed-cross-spelling-home", tmp_path / "managed-cross-spelling"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump(user_document), encoding="utf-8")
    (managed / "config.yaml").write_text(yaml.safe_dump(managed_document), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    try:
        config = load_gateway_config()
    finally:
        managed_scope.invalidate_managed_cache()

    assert config.to_dict()["canonical_surface_bindings"] == expected


def test_malformed_yaml_fails_closed_for_canonical_bindings_but_keeps_legacy_config(tmp_path, monkeypatch):
    home = tmp_path / "malformed-home"
    home.mkdir()
    (home / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _binding("legacy"), "session_store_max_age_days": 17}),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text("gateway: [not valid", encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_gateway_config()

    assert config.canonical_surface_bindings == {}
    assert config.session_store_max_age_days == 17


def test_unreadable_yaml_does_not_allow_a_second_read_to_restore_canonical_bindings(tmp_path, monkeypatch):
    home = tmp_path / "unreadable-home"
    home.mkdir()
    (home / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _binding("legacy"), "session_store_max_age_days": 17}),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(yaml.safe_dump({"canonical_surface_bindings": _binding("yaml")}), encoding="utf-8")

    def unreadable_yaml(*_args, **_kwargs):
        raise PermissionError("synthetic unreadable config.yaml")

    monkeypatch.setattr(config_loader, "load_yaml_layer", unreadable_yaml)
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_gateway_config()

    assert config.canonical_surface_bindings == {}
    assert config.session_store_max_age_days == 17


def test_loader_uses_its_first_authoritative_read_for_canonical_bindings(tmp_path, monkeypatch):
    home = tmp_path / "single-read-home"
    home.mkdir()
    (home / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _binding("legacy"), "session_store_max_age_days": 17}),
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(yaml.safe_dump({"canonical_surface_bindings": _binding("yaml")}), encoding="utf-8")

    real_load_yaml_layer = config_loader.load_yaml_layer

    def load_then_make_second_parse_fail(*args, **kwargs):
        yaml_data = real_load_yaml_layer(*args, **kwargs)

        def malformed_second_read(_source):
            raise yaml.YAMLError("synthetic malformed second read")

        monkeypatch.setattr(yaml, "safe_load", malformed_second_read)
        return yaml_data

    monkeypatch.setattr(config_loader, "load_yaml_layer", load_then_make_second_parse_fail)
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_gateway_config()

    assert list(config.canonical_surface_bindings) == ["yaml"]
    assert config.session_store_max_age_days == 17


def test_managed_canonical_bindings_expand_environment_values_before_loading(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    home, managed = tmp_path / "managed-expanded-home", tmp_path / "managed-expanded"
    home.mkdir()
    managed.mkdir()
    managed_binding = _binding("managed")
    managed_binding["managed"]["session_id"] = "${CANONICAL_SYNTHETIC_ID}"
    (home / "config.yaml").write_text(yaml.safe_dump({"canonical_surface_bindings": _binding("user")}), encoding="utf-8")
    (managed / "config.yaml").write_text(yaml.safe_dump({"canonical_surface_bindings": managed_binding}), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("CANONICAL_SYNTHETIC_ID", "expanded-id")
    managed_scope.invalidate_managed_cache()
    try:
        config = load_gateway_config()
    finally:
        managed_scope.invalidate_managed_cache()

    assert config.canonical_surface_bindings["managed"].session_id == "expanded-id"


def test_managed_canonical_bindings_use_one_snapshot_for_overlay_and_authority(tmp_path, monkeypatch):
    from hermes_cli import managed_scope

    home = tmp_path / "managed-one-snapshot-home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"canonical_surface_bindings": _binding("user")}), encoding="utf-8")
    snapshots = iter((
        {"canonical_surface_bindings": _binding("first"), "quick_commands": {"snapshot": "A"}},
        {"gateway": {"canonical_surface_bindings": _binding("second")}, "quick_commands": {"snapshot": "B"}},
    ))
    calls = 0

    def alternating_snapshot():
        nonlocal calls
        calls += 1
        return next(snapshots)

    monkeypatch.setattr(managed_scope, "load_managed_config", alternating_snapshot)
    yaml_cfg = config_loader.read_yaml_layers(home)

    assert calls == 1
    assert list(yaml_cfg["canonical_surface_bindings"]) == ["first"]
    assert yaml_cfg["quick_commands"] == {"snapshot": "A"}


def test_managed_overlay_failure_cannot_restore_user_canonical_bindings(tmp_path, monkeypatch):
    from hermes_cli import config as cli_config
    from hermes_cli import managed_scope

    home, managed = tmp_path / "managed-fail-closed-home", tmp_path / "managed-fail-closed"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "canonical_surface_bindings": _binding("top-user"),
            "gateway": {"canonical_surface_bindings": _binding("nested-user"), "loop_watchdog": False},
        }),
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(yaml.safe_dump({"canonical_surface_bindings": _binding("managed")}), encoding="utf-8")

    def synthetic_overlay_error(*_args, **_kwargs):
        raise RuntimeError("synthetic managed overlay failure")

    monkeypatch.setattr(cli_config, "_normalize_root_model_keys", synthetic_overlay_error)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    try:
        config = load_gateway_config()
    finally:
        managed_scope.invalidate_managed_cache()

    assert config.canonical_surface_bindings == {}
    assert config.loop_watchdog is False


@pytest.mark.parametrize(
    "managed_document",
    (
        {"canonical_surface_bindings": {}},
        {"gateway": {"canonical_surface_bindings": {}}},
    ),
)
def test_managed_empty_canonical_binding_removes_both_user_spellings_but_keeps_siblings(
    tmp_path, monkeypatch, managed_document
):
    from hermes_cli import managed_scope

    home, managed = tmp_path / "managed-both-spellings-home", tmp_path / "managed-both-spellings"
    home.mkdir()
    managed.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "canonical_surface_bindings": _binding("top-user"),
            "gateway": {"canonical_surface_bindings": _binding("nested-user"), "loop_watchdog": False},
        }),
        encoding="utf-8",
    )
    (managed / "config.yaml").write_text(yaml.safe_dump(managed_document), encoding="utf-8")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    try:
        config = load_gateway_config()
    finally:
        managed_scope.invalidate_managed_cache()

    assert config.canonical_surface_bindings == {}
    assert config.loop_watchdog is False
