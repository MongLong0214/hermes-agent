"""Closed canonical bindings come only from the effective YAML layer, and fail closed."""

import json

import pytest

import hermes_yaml as yaml
from gateway import config_loader
from gateway.config import GatewayConfig, load_gateway_config


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


@pytest.fixture
def managed_home(tmp_path, monkeypatch):
    """(home, managed dir) with HERMES_HOME / HERMES_MANAGED_DIR pointed at them, cache dropped."""
    from hermes_cli import managed_scope

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    yield home, managed
    managed_scope.invalidate_managed_cache()


def _write(path, document) -> None:
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


# --- counterexample 1: the managed overlay owns the bindings it declares -------------------------


@pytest.mark.parametrize(
    ("user_document", "managed_document", "expected"),
    (
        ({"canonical_surface_bindings": _binding("user")},
         {"gateway": {"canonical_surface_bindings": _binding("managed")}}, ["managed"]),
        ({"gateway": {"canonical_surface_bindings": _binding("user")}},
         {"canonical_surface_bindings": _binding("managed")}, ["managed"]),
        ({"canonical_surface_bindings": _binding("user")}, {"gateway": {"canonical_surface_bindings": {}}}, []),
        ({"gateway": {"canonical_surface_bindings": _binding("user")}}, {"canonical_surface_bindings": {}}, []),
        ({"canonical_surface_bindings": _binding("user")}, {"canonical_surface_bindings": None}, []),
        ({"canonical_surface_bindings": _binding("user"), "gateway": {"loop_watchdog": False}},
         {"canonical_surface_bindings": _binding("managed")}, ["managed"]),
    ),
    ids=["top-user-nested-managed", "nested-user-top-managed", "nested-empty", "top-empty", "null",
         "top-user-gateway-without-bindings"],
)
def test_user_binding_is_ignored_when_managed_overlay_declares_bindings(
    managed_home, user_document, managed_document, expected
):
    home, managed = managed_home
    _write(home / "config.yaml", user_document)
    _write(managed / "config.yaml", managed_document)

    bindings = load_gateway_config().canonical_surface_bindings

    assert list(bindings) == expected


def test_same_named_managed_binding_replaces_the_user_entry_instead_of_leaf_merging(managed_home):
    home, managed = managed_home
    user = _binding("shared")
    user["shared"]["buzz"]["pairs"] = [{"author_id": "author-shared", "channel_id": "channel-shared"}]
    _write(home / "config.yaml", {"canonical_surface_bindings": user})
    _write(managed / "config.yaml", {"canonical_surface_bindings": _binding("shared")})

    assert load_gateway_config().canonical_surface_bindings["shared"].pairs == ()


@pytest.mark.parametrize("managed_document", (
    {"canonical_surface_bindings": {}},
    {"gateway": {"canonical_surface_bindings": {}}},
))
def test_managed_empty_binding_removes_both_user_spellings_but_keeps_siblings(managed_home, managed_document):
    home, managed = managed_home
    _write(home / "config.yaml", {
        "canonical_surface_bindings": _binding("top-user"),
        "gateway": {"canonical_surface_bindings": _binding("nested-user"), "loop_watchdog": False},
    })
    _write(managed / "config.yaml", managed_document)

    config = load_gateway_config()

    assert config.canonical_surface_bindings == {}
    assert config.loop_watchdog is False


def test_managed_overlay_failure_cannot_restore_user_bindings(managed_home, monkeypatch):
    from hermes_cli import config as cli_config

    home, managed = managed_home
    _write(home / "config.yaml", {
        "canonical_surface_bindings": _binding("top-user"),
        "gateway": {"canonical_surface_bindings": _binding("nested-user"), "loop_watchdog": False},
    })
    _write(managed / "config.yaml", {"canonical_surface_bindings": _binding("managed")})

    def synthetic_overlay_error(*_args, **_kwargs):
        raise RuntimeError("synthetic managed overlay failure")

    monkeypatch.setattr(cli_config, "_normalize_root_model_keys", synthetic_overlay_error)
    config = load_gateway_config()

    assert config.canonical_surface_bindings == {}
    assert config.loop_watchdog is False


def _binding_names(yaml_cfg: dict) -> dict:
    return {
        "top": sorted(yaml_cfg.get("canonical_surface_bindings") or {}),
        "gateway": sorted((yaml_cfg.get("gateway") or {}).get("canonical_surface_bindings") or {}),
    }


@pytest.mark.parametrize(
    ("first_snapshot", "expected_names"),
    (
        ({"canonical_surface_bindings": _binding("first"), "quick_commands": {"snapshot": "A"}},
         {"top": ["first"], "gateway": []}),
        ({"gateway": {"canonical_surface_bindings": _binding("first")}, "quick_commands": {"snapshot": "A"}},
         {"top": [], "gateway": ["first"]}),
        ({"quick_commands": {"snapshot": "A"}},
         {"top": ["top-user"], "gateway": ["nested-user"]}),
    ),
    ids=["snapshot-declares-top", "snapshot-declares-nested", "snapshot-declares-none"],
)
def test_managed_canonical_bindings_use_one_snapshot_for_overlay_and_authority(
    managed_home, monkeypatch, first_snapshot, expected_names
):
    """The managed snapshot that decides whether the user's bindings count is the one the overlay
    applies: a second read of a rewritten file can neither add its bindings nor its other values."""
    from hermes_cli import managed_scope

    home, _managed = managed_home
    _write(home / "config.yaml", {
        "canonical_surface_bindings": _binding("top-user"),
        "gateway": {"canonical_surface_bindings": _binding("nested-user"), "loop_watchdog": False},
    })
    snapshots = iter((
        first_snapshot,
        {"canonical_surface_bindings": _binding("second"),
         "gateway": {"canonical_surface_bindings": _binding("second")}, "quick_commands": {"snapshot": "B"}},
    ))
    calls = 0

    def rewritten_after_first_read():
        nonlocal calls
        calls += 1
        return next(snapshots)

    monkeypatch.setattr(managed_scope, "load_managed_config", rewritten_after_first_read)

    yaml_cfg = config_loader.read_yaml_layers(home)

    assert calls == 1
    assert _binding_names(yaml_cfg) == expected_names
    assert yaml_cfg["quick_commands"] == {"snapshot": "A"}
    assert yaml_cfg["gateway"]["loop_watchdog"] is False


def test_a_non_mapping_user_layer_is_not_reshaped_by_a_managed_binding_declaration(managed_home):
    home, managed = managed_home
    (home / "config.yaml").write_text("- one\n- two\n", encoding="utf-8")
    _write(managed / "config.yaml", {"canonical_surface_bindings": {}})

    assert config_loader.read_yaml_layers(home) == ["one", "two"]


def test_managed_bindings_replace_legacy_gateway_json_authority(managed_home):
    home, managed = managed_home
    (home / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _binding("legacy"), "session_store_max_age_days": 17}),
        encoding="utf-8",
    )
    _write(home / "config.yaml", {"gateway": {"loop_watchdog": False}})
    _write(managed / "config.yaml", {"canonical_surface_bindings": _binding("managed")})

    config = load_gateway_config()

    assert list(config.canonical_surface_bindings) == ["managed"]
    assert config.session_store_max_age_days == 17
    assert config.loop_watchdog is False


def test_managed_bindings_expand_environment_values_before_parsing(managed_home, monkeypatch):
    home, managed = managed_home
    managed_binding = _binding("managed")
    managed_binding["managed"]["session_id"] = "${CANONICAL_SYNTHETIC_ID}"
    _write(home / "config.yaml", {"canonical_surface_bindings": _binding("user")})
    _write(managed / "config.yaml", {"canonical_surface_bindings": managed_binding})
    monkeypatch.setenv("CANONICAL_SYNTHETIC_ID", "expanded-id")

    assert load_gateway_config().canonical_surface_bindings["managed"].session_id == "expanded-id"


# --- legacy gateway.json is never a binding authority -------------------------------------------


@pytest.mark.parametrize("legacy", (
    {"canonical_surface_bindings": _binding("legacy")},
    {"gateway": {"canonical_surface_bindings": _binding("legacy")}},
), ids=["top", "nested"])
def test_valid_yaml_without_bindings_clears_legacy_authority_but_keeps_legacy_settings(tmp_path, monkeypatch, legacy):
    home = tmp_path / "home"
    home.mkdir()
    (home / "gateway.json").write_text(json.dumps({**legacy, "session_store_max_age_days": 17}), encoding="utf-8")
    _write(home / "config.yaml", {"gateway": {"loop_watchdog": False}})
    monkeypatch.setenv("HERMES_HOME", str(home))

    config = load_gateway_config()

    assert config.canonical_surface_bindings == {}
    assert config.session_store_max_age_days == 17
    assert config.loop_watchdog is False


def test_malformed_yaml_fails_closed_for_bindings_but_keeps_legacy_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
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


def test_unreadable_yaml_does_not_let_a_second_read_restore_bindings(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _binding("legacy"), "session_store_max_age_days": 17}),
        encoding="utf-8",
    )
    _write(home / "config.yaml", {"canonical_surface_bindings": _binding("yaml")})

    def unreadable_yaml(*_args, **_kwargs):
        raise PermissionError("synthetic unreadable config.yaml")

    monkeypatch.setattr(config_loader, "load_yaml_layer", unreadable_yaml)
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = load_gateway_config()

    assert config.canonical_surface_bindings == {}
    assert config.session_store_max_age_days == 17


def test_loader_takes_bindings_from_its_first_and_only_read(tmp_path, monkeypatch):
    from hermes_cli import config_backend

    home = tmp_path / "home"
    home.mkdir()
    (home / "gateway.json").write_text(json.dumps({"canonical_surface_bindings": _binding("legacy")}), encoding="utf-8")
    _write(home / "config.yaml", {"canonical_surface_bindings": _binding("yaml")})
    real_load_yaml_layer = config_loader.load_yaml_layer

    def load_then_fail_any_second_read(*args, **kwargs):
        yaml_data = real_load_yaml_layer(*args, **kwargs)

        def second_read(*_args, **_kwargs):
            raise AssertionError("canonical bindings must not re-read config.yaml")

        monkeypatch.setattr(config_backend, "read_config_doc", second_read)
        return yaml_data

    monkeypatch.setattr(config_loader, "load_yaml_layer", load_then_fail_any_second_read)
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert list(load_gateway_config().canonical_surface_bindings) == ["yaml"]


def test_loader_uses_top_level_precedence_and_keeps_profiles_isolated(tmp_path, monkeypatch):
    home_a, home_b = tmp_path / "home-a", tmp_path / "home-b"
    for home, document in (
        (home_a, {"canonical_surface_bindings": _binding("alpha"),
                  "gateway": {"canonical_surface_bindings": _binding("wrong")}}),
        (home_b, {"gateway": {"canonical_surface_bindings": _binding("beta")}}),
    ):
        home.mkdir()
        (home / "gateway.json").write_text(json.dumps({"canonical_surface_bindings": _binding("legacy")}), encoding="utf-8")
        _write(home / "config.yaml", document)

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


# --- counterexample 2: a malformed schema yields no bindings, never a partial set ----------------


def _malformed(mutate) -> dict:
    document = _binding("bad")
    mutate(document["bad"])
    return document


@pytest.mark.parametrize(
    "malformed",
    (
        {"bad": {"session_key": "x"}},
        _malformed(lambda b: b.update(extra="field")),
        _malformed(lambda b: b["telegram"].update(chat_type="supergroup")),
        _malformed(lambda b: b["telegram"].pop("thread_id")),
        _malformed(lambda b: b.update(session_id="   ")),
        _malformed(lambda b: b.update(session_key="k" * 513)),
        _malformed(lambda b: b["buzz"].update(author_ids=[])),
        _malformed(lambda b: b["buzz"].update(author_ids=["a", "a"])),
        _malformed(lambda b: b["buzz"].update(author_ids=["a", "b"])),
        _malformed(lambda b: b["buzz"].update(pairs=[{"author_id": "author-bad", "channel_id": "other"}])),
        _malformed(lambda b: b["buzz"].update(pairs=[])),
        {"bad": ["not", "a", "mapping"]},
        ["not", "a", "mapping"],
    ),
    ids=["missing-keys", "extra-key", "chat-type", "telegram-key", "blank", "too-long", "empty-authors",
         "duplicate-authors", "unpaired-authors", "pairs-not-covering", "empty-pairs", "entry-not-mapping",
         "root-not-mapping"],
)
def test_malformed_binding_schema_fails_closed_with_no_partial_bindings(malformed):
    document = {**_binding("good"), **malformed} if isinstance(malformed, dict) else malformed

    with pytest.raises(ValueError, match="canonical"):
        GatewayConfig.from_dict({"canonical_surface_bindings": document})


def test_valid_paired_binding_parses_and_round_trips():
    document = _binding("paired")
    document["paired"]["buzz"] = {
        "author_ids": ["a1", "a2"],
        "channel_ids": ["c1", "c2"],
        "pairs": [{"author_id": "a1", "channel_id": "c1"}, {"author_id": "a2", "channel_id": "c2"}],
    }

    config = GatewayConfig.from_dict({"gateway": {"canonical_surface_bindings": document}})
    binding = config.canonical_surface_bindings["paired"]

    assert binding.pairs == (("a1", "c1"), ("a2", "c2"))
    assert binding.admits("a1", "c1") and not binding.admits("a1", "c2")
    assert binding.roster is None
    assert config.to_dict()["canonical_surface_bindings"] == document
    assert GatewayConfig.from_dict(config.to_dict()).canonical_surface_bindings == config.canonical_surface_bindings


# --- counterexample 4: with no bindings the config is the target's ------------------------------


def test_no_bindings_leave_the_yaml_layer_and_config_equal_to_target(managed_home):
    from hermes_cli import managed_scope
    from hermes_cli.config import _expand_env_vars
    from hermes_cli.config_backend import read_config_doc

    home, managed = managed_home
    (home / "gateway.json").write_text(json.dumps({"session_store_max_age_days": 17}), encoding="utf-8")
    _write(home / "config.yaml", {"gateway": {"loop_watchdog": False}, "quick_commands": {"u": {"type": "exec"}}})
    _write(managed / "config.yaml", {"quick_commands": {"m": {"type": "exec"}}, "gateway": {"loop_watchdog": True}})

    target_layer = managed_scope.apply_managed_overlay(_expand_env_vars(read_config_doc(home / "config.yaml")))
    assert config_loader.read_yaml_layers(home) == target_layer

    config = load_gateway_config()
    assert config.canonical_surface_bindings == {}
    assert "canonical_surface_bindings" not in config.to_dict()
    assert GatewayConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    assert GatewayConfig().canonical_surface_bindings == {}


def test_user_bindings_without_a_managed_declaration_leave_the_yaml_layer_equal_to_target(managed_home):
    from hermes_cli import managed_scope
    from hermes_cli.config import _expand_env_vars
    from hermes_cli.config_backend import read_config_doc

    home, managed = managed_home
    _write(home / "config.yaml", {
        "canonical_surface_bindings": _binding("top-user"),
        "gateway": {"canonical_surface_bindings": _binding("nested-user"), "loop_watchdog": False},
    })
    _write(managed / "config.yaml", {"gateway": {"loop_watchdog": True}})

    target_layer = managed_scope.apply_managed_overlay(_expand_env_vars(read_config_doc(home / "config.yaml")))
    assert config_loader.read_yaml_layers(home) == target_layer
    assert list(load_gateway_config().canonical_surface_bindings) == ["top-user"]
