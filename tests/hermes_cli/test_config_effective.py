"""Invariants for ``hermes_cli.config_effective.load_user_config_effective`` — the one loader every
defaults-free config reader (gateway runtime, TUI gateway, cron, ``hermes send`` bridge, doctor,
bootstrap modules) goes through."""
import textwrap

import pytest
import yaml


@pytest.fixture
def homes(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setenv("FIXTURE_USER_KEY", "user-secret")
    monkeypatch.setenv("FIXTURE_MANAGED_URL", "https://managed.example")
    _reset_caches()
    return home, managed


def _reset_caches():
    import hermes_cli.config as cfg
    from hermes_cli import config_effective, managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()
    config_effective._LAST_GOOD_USER_RAW.clear()
    managed_scope.invalidate_managed_cache()


def _write(path, body):
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    _reset_caches()


USER_YAML = """
    model:
      name: user/model
      api_key: ${FIXTURE_USER_KEY}
    provider: custom
    display:
      skin: user-skin
    """
MANAGED_YAML = """
    model:
      base_url: ${FIXTURE_MANAGED_URL}
    display:
      skin: managed-skin
    """


def test_effective_is_user_plus_managed_plus_env_with_no_defaults(homes):
    """Contract as a fixture: given user config.yaml X, managed overlay Y and env Z, the effective
    dict is exactly this literal — ``${VAR}`` expanded on both layers, managed keys winning,
    root ``provider`` migrated under ``model``, and no DEFAULT_CONFIG key introduced (a missing
    key stays missing). Per-message gateway reads (and the system prompt built from them) are
    pinned by this shape, not by re-running the implementation's primitives."""
    from hermes_cli.config_effective import load_user_config_effective

    home, managed = homes
    _write(home / "config.yaml", USER_YAML)
    _write(managed / "config.yaml", MANAGED_YAML)

    effective = load_user_config_effective(home / "config.yaml")

    assert effective == {
        "model": {
            "default": "user/model",
            "provider": "custom",
            "api_key": "user-secret",
            "base_url": "https://managed.example",
        },
        "display": {"skin": "managed-skin"},
    }


def test_broken_yaml_serves_last_good_and_fail_closed_raises(homes):
    """A torn mid-edit write must not silently drop user overrides: the fail-open path serves the last
    successfully parsed user file through the same pipeline; ``fail_closed`` surfaces the error to
    callers that keep their own last-good state."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    _write(home / "config.yaml", USER_YAML)
    good = load_user_config_effective(home / "config.yaml")

    (home / "config.yaml").write_text("model: [unterminated", encoding="utf-8")
    _reset_caches_keep_last_good()

    assert load_user_config_effective(home / "config.yaml") == good
    with pytest.raises(yaml.YAMLError):  # the type _refresh_fallback_model's own last-good path keys on
        load_user_config_effective(home / "config.yaml", fail_closed=True)


def test_good_backup_is_written_only_for_the_active_home(homes, tmp_path):
    """Reading ANOTHER profile's config (doctor, TUI cwd lookup) is a read: it must not create
    ``backups/config/`` inside that profile. The active home keeps the last-good copy."""
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    other = tmp_path / "other-profile"
    other.mkdir()
    _write(home / "config.yaml", USER_YAML)
    _write(other / "config.yaml", USER_YAML)

    load_user_config_effective(other / "config.yaml")
    load_user_config_effective(home / "config.yaml")

    assert not (other / "backups").exists()
    assert list((home / "backups" / "config").glob("config.yaml.good.*"))


RESTRICTED_YAML = """
    agent:
      reasoning_effort: medium
    display:
      background_process_notifications: 'off'
    """


def test_non_mapping_root_serves_last_good_and_fail_closed_raises(homes):
    """L7-1 regression: an edit whose root parses as valid YAML but is not a mapping (e.g. a bare
    list — ``- stray``) must be rejected the same way broken YAML is, not silently coerced to
    ``{}``. The live-reload path previously recorded that empty dict as last-good and overwrote
    the ``good`` backup with it, so the NEXT gateway turn lost restricted settings like
    ``display.background_process_notifications: off`` (reverts to the ``concise`` default) and
    ``agent.reasoning_effort`` (reverts to the hardcoded default) even though the broken edit was
    never applied."""
    from hermes_cli.config_backups import load_newest_good_backup
    from hermes_cli.config_effective import load_user_config_effective

    home, _ = homes
    _write(home / "config.yaml", RESTRICTED_YAML)
    good = load_user_config_effective(home / "config.yaml")
    assert good == {
        "agent": {"reasoning_effort": "medium"},
        "display": {"background_process_notifications": "off"},
    }

    (home / "config.yaml").write_text("- stray\n", encoding="utf-8")
    _reset_caches_keep_last_good()

    # The next gateway turn must still see the restricted settings, not {}.
    assert load_user_config_effective(home / "config.yaml") == good
    # The "good" backup must still hold the prior mapping — not get overwritten with the list.
    assert load_newest_good_backup(home / "config.yaml") == good

    with pytest.raises(TypeError):
        load_user_config_effective(home / "config.yaml", fail_closed=True)


GATEWAY_RESTRICTED_YAML = RESTRICTED_YAML + """
    platform_toolsets:
      telegram: [web]
    """


@pytest.mark.parametrize("bad_root", ["[]\n", "false\n", "0\n", "''\n", "- stray\n"])
def test_non_mapping_root_survives_shared_cache_readers(homes, bad_root):
    """L7-1 regression, with every cache kept live: the raw reader (``read_raw_config()`` behind
    ``gateway_help_lines()`` / ``/help``) and the defaults loader (``load_config()``) parse the
    same file and publish into the shared raw cache / the ``good`` backup. A falsy non-mapping
    root (``[]``, ``false``, ``0``, ``''``) used to be coerced to ``{}`` there and then served to
    the gateway's effective reader as a successful parse — dropping the restricted settings and
    widening the Telegram toolsets to the default bundle (terminal, file, code execution)."""
    from hermes_cli.commands import gateway_help_lines
    from hermes_cli.config import load_config
    from hermes_cli.config_backups import load_newest_good_backup
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.tools_config import _get_platform_tools

    home, _ = homes
    path = home / "config.yaml"
    _write(path, GATEWAY_RESTRICTED_YAML)
    good = load_user_config_effective(path)
    assert good["display"]["background_process_notifications"] == "off"
    good_tools = _get_platform_tools(good, "telegram")
    assert "terminal" not in good_tools
    assert load_config()["display"]["background_process_notifications"] == "off"

    path.write_text(bad_root, encoding="utf-8")  # no cache reset: the bypass needs live caches
    gateway_help_lines()  # raw reader runs first, as on a gateway /help
    assert load_config()["display"]["background_process_notifications"] == "off"

    served = load_user_config_effective(path)
    assert served == good
    assert _get_platform_tools(served, "telegram") == good_tools
    assert load_newest_good_backup(path) == good
    with pytest.raises(TypeError):
        load_user_config_effective(path, fail_closed=True)


def _reset_caches_keep_last_good():
    import hermes_cli.config as cfg
    from hermes_cli import config_effective

    cfg._RAW_CONFIG_CACHE.clear()
    config_effective._EFFECTIVE_CACHE.clear()
