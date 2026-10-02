"""config.yaml backups: one dir, deduped, bounded — never a pile of siblings in HERMES_HOME."""
from pathlib import Path

from hermes_cli.config_backups import backup_config, list_config_backups


def test_repeat_backups_dedupe_and_rotate(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    stamps = iter(f"2026010100000{i}" for i in range(10))
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: next(stamps))

    first = backup_config(cfg, "pre-setup", keep=2)
    assert first is not None and first.parent == tmp_path / "backups" / "config"
    # Same bytes again → no new file (the hermes-setup-three-times case).
    assert backup_config(cfg, "pre-setup", keep=2) is None
    for i in range(3):
        cfg.write_text(f"model: {i}\n")
        backup_config(cfg, "pre-setup", keep=2)
    kept = list_config_backups(cfg, "pre-setup")
    assert len(kept) == 2 and kept[0].read_text() == "model: 2\n"
    # Nothing left beside config.yaml in the home root.
    assert [p.name for p in tmp_path.iterdir() if p.is_file()] == ["config.yaml"]


def test_legacy_siblings_move_but_user_named_copies_stay(tmp_path: Path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("model: a\n")
    for name in ("config.yaml.bak.1778718391", "config.yaml.corrupt.20260729-093706.bak",
                 "config.yaml.bak-pre-migrate-xai-20260515-120000"):
        (tmp_path / name).write_text("old")
    (tmp_path / "config.yaml.bak-my-note").write_text("mine")

    backup_config(cfg, "corrupt")

    root = tmp_path / "backups" / "config"
    assert (root / "config.yaml.bak.1778718391").exists()
    assert (root / "config.yaml.corrupt.20260729-093706.bak").exists()
    assert (tmp_path / "config.yaml.bak-my-note").read_text() == "mine"
    assert not list(tmp_path.glob("config.yaml.bak.*")) and not list(tmp_path.glob("config.yaml.corrupt.*"))


def test_same_stamp_backups_recover_in_creation_order_even_after_pruning(tmp_path: Path, monkeypatch):
    """REMEDIATION-1: with the clock frozen (every call lands on the same second-level stamp) and
    retention pruning the oldest copies as newer ones are made, list_config_backups' lexical sort
    must still return the TRUE newest content first -- not a stale copy whose name happens to
    collide, after pruning, with the single second this test freezes on."""
    from hermes_cli.config_backups import backup_config, list_config_backups

    cfg = tmp_path / "config.yaml"
    frozen = "20261002-000000"
    monkeypatch.setattr("hermes_cli.config_backups.time.strftime", lambda _fmt: frozen)

    for i in range(10):
        cfg.write_text(f"model: {i}\n")
        assert backup_config(cfg, "good", keep=3) is not None

    kept = list_config_backups(cfg, "good")
    assert len(kept) == 3
    assert kept[0].read_text() == "model: 9\n", "the newest backup must sort first, even reused names"
    assert [p.read_text() for p in kept] == ["model: 9\n", "model: 8\n", "model: 7\n"]
