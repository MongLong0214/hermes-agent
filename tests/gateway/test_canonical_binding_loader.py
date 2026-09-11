"""Canonical binding admission must survive the real YAML loader."""

import json

import pytest
import yaml

from gateway.config import GatewayConfig, load_gateway_config


def _bindings(name):
    return {
        name: {
            "session_key": "agent:main:telegram:dm:10001",
            "session_id": "synthetic-existing-session",
            "telegram": {
                "chat_id": "10001",
                "chat_type": "dm",
                "user_id": "10001",
                "thread_id": None,
            },
            "buzz": {"author_ids": ["synthetic-author"], "channel_ids": ["synthetic-channel"]},
        }
    }


@pytest.mark.parametrize(
    "document",
    [
        {"canonical_surface_bindings": _bindings("top")},
        {"gateway": {"canonical_surface_bindings": _bindings("nested")}},
        {"canonical_surface_bindings": _bindings("top"), "gateway": {"canonical_surface_bindings": _bindings("nested")}},
        {"canonical_surface_bindings": {}, "gateway": {"canonical_surface_bindings": _bindings("nested")}},
        {"canonical_surface_bindings": None, "gateway": {"canonical_surface_bindings": _bindings("nested")}},
    ],
    ids=["top", "nested", "top-wins", "empty-top-wins", "null-top-wins"],
)
def test_yaml_canonical_bindings_follow_from_dict_precedence(tmp_path, monkeypatch, document):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway.json").write_text(
        json.dumps({"canonical_surface_bindings": _bindings("legacy")}), encoding="utf-8"
    )
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")

    expected = GatewayConfig.from_dict(document).canonical_surface_bindings
    actual = load_gateway_config().canonical_surface_bindings

    assert actual == expected
