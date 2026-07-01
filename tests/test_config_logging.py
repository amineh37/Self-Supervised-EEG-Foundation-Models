from __future__ import annotations

import argparse

import pytest

from stelar_utils import apply_yaml_defaults, build_loggers, load_yaml_flat
from stelar_utils.loggers import add_logging_args


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_yaml_flatten_merges_sections(tmp_path):
    cfg = _write(tmp_path, """
seed: 7
model:
  preset: v2_aug
  tag: tiny
data:
  batch_size: 8
""")
    flat = load_yaml_flat(cfg)
    assert flat == {"seed": 7, "preset": "v2_aug", "tag": "tiny", "batch_size": 8}


def test_yaml_applies_as_defaults_and_cli_overrides(tmp_path):
    cfg = _write(tmp_path, "model:\n  preset: v2_aug\n  batch_size: 8\n")
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="v0_baseline")
    p.add_argument("--batch_size", type=int, default=64)
    apply_yaml_defaults(p, cfg)

    # YAML wins over built-in defaults...
    args = p.parse_args([])
    assert args.preset == "v2_aug"
    assert args.batch_size == 8

    # ...but an explicit CLI flag still wins over YAML.
    args = p.parse_args(["--preset", "vA_combo"])
    assert args.preset == "vA_combo"
    assert args.batch_size == 8


def test_unknown_yaml_keys_are_ignored(tmp_path, capsys):
    cfg = _write(tmp_path, "totally_unknown_key: 123\nseed: 1\n")
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    accepted = apply_yaml_defaults(p, cfg)
    assert accepted == {"seed": 1}
    assert "totally_unknown_key" in capsys.readouterr().out


@pytest.mark.parametrize("choice,expected", [("csv", 1), ("none", 0)])
def test_build_loggers_file_choices(tmp_path, choice, expected):
    loggers = build_loggers(logger=choice, save_root=tmp_path, name="t")
    assert len(loggers) == expected


@pytest.mark.parametrize("choice,expected", [("both", 2), ("tensorboard", 1)])
def test_build_loggers_tensorboard_choices(tmp_path, choice, expected):
    pytest.importorskip("tensorboard")
    loggers = build_loggers(logger=choice, save_root=tmp_path, name="t")
    assert len(loggers) == expected


def test_add_logging_args_defaults():
    p = argparse.ArgumentParser()
    add_logging_args(p)
    args = p.parse_args([])
    assert args.logger == "both"
    assert args.wandb is False
    assert args.wandb_project == "stelar"


def test_wandb_offline_logger(tmp_path):
    wandb = pytest.importorskip("wandb")
    loggers = build_loggers(
        logger="csv", save_root=tmp_path, name="t",
        use_wandb=True, wandb_offline=True, wandb_project="stelar-test",
    )
    try:
        # CSV + W&B
        assert len(loggers) == 2
        from pytorch_lightning.loggers import WandbLogger
        assert any(isinstance(lg, WandbLogger) for lg in loggers)
    finally:
        wandb.finish()
