"""Explicit null at the config backend seam.

An explicit ``null`` root and an empty file both parse to ``None``. Only the empty file is a
first run; ``read_config_doc(..., explicit_null=True)`` tells the two apart. A null or ``false``
value under a key is a setting in its own right and must survive a write.
"""

import pytest

from hermes_cli import config_backend as cb


@pytest.mark.parametrize("data", [
    b"null\n", b"~\n", b"\xef\xbb\xbfnull\n", b"%YAML 1.2\n--- !!null\n", b"--- ~\n",
    b"null # disabled\n", b"&a null\n", b'!!null ""\n', b"!!null\n",
], ids=["null", "tilde", "bom-null", "yaml12-tagged", "marker-tilde", "commented", "anchored",
        "tagged-empty", "bare-tag"])
def test_explicit_null_root_is_exposed(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_bytes(data)

    assert cb.read_config_doc(path, explicit_null=True) is cb.EXPLICIT_NULL
    assert cb.read_config_doc(path) is None  # the default read is unchanged for every other caller


@pytest.mark.parametrize("data", [
    b"", b"\n", b"# only a comment\n", b"\xef\xbb\xbf", b"---\n", b"---\n...\n", b"%YAML 1.1\n---\n",
    b"---\n# just a comment\n",
], ids=["empty", "newline", "comment", "bom-only", "marker", "markers", "directive", "marker-comment"])
def test_empty_document_is_not_explicit_null(tmp_path, data):
    path = tmp_path / "config.yaml"
    path.write_bytes(data)

    assert cb.read_config_doc(path, explicit_null=True) is None


def test_write_changes_keeps_explicit_null_and_false(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("# keep me\nkeep: 1\nmodel:\n  api_key: old\n", encoding="utf-8")

    cb.write_changes(path, cb.Changes(set={"model.api_key": None, "flag": False, "new.leaf": None}))

    assert cb.read_config_doc(path) == {
        "keep": 1, "model": {"api_key": None}, "flag": False, "new": {"leaf": None}}
    assert "# keep me" in path.read_text(encoding="utf-8")


def test_null_and_false_survive_a_document_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: null\nb: false\nc: 1\n", encoding="utf-8")

    doc = cb.read_config_doc(path)
    doc["c"] = 2
    cb.write_config_document(path, doc)

    assert cb.read_config_doc(path) == {"a": None, "b": False, "c": 2}


def test_write_config_keys_none_still_removes(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb: 2\n", encoding="utf-8")

    cb.write_config_keys(path, {"a": None})

    assert cb.read_config_doc(path) == {"b": 2}
