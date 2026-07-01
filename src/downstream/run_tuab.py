#!/usr/bin/env python3
"""
TUAB-only downstream runner (4-fold subject CV for val, fixed TUAB test)

Updates requested:
- Keep TUAB only
- Use ONE seed for everything (global, CV split, dataset split)
- Add grad clip value (Lightning gradient_clip_val)
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import KFold

# -----------------------------------------------------------------------------
# Config + local imports
# -----------------------------------------------------------------------------
import importlib
import sys

CFG = importlib.import_module("finetune_configs")

# add pretraining/ to sys.path (relative to this file)
sys.path.append(str(Path(__file__).resolve().parents[1] / "pretraining"))

from modules import build_module  # noqa: E402
from utils import default_num_workers, get_checkpoint_path, parse_devices  # noqa: E402
from channels import CHANNELS_BY_DATASET, DEFAULT_EEGPT58, MONTAGE_BY_DATASET  # noqa: E402

# Optional YAML config + W&B logger layer (additive; off by default).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stelar_utils import add_logging_args, apply_yaml_defaults  # noqa: E402
from stelar_utils.loggers import make_wandb_logger  # noqa: E402

# Populated from CLI/YAML in parse_args(); consumed by make_loggers().
_WANDB_OPTS: Dict[str, Any] = {}

# Optional direct modules
try:
    from modules.cbramod_ft import LitCBraMod  # noqa: E402
except Exception:
    LitCBraMod = None  # type: ignore

try:
    from modules.labram_ft import LitLaBraM  # noqa: E402
except Exception:
    LitLaBraM = None  # type: ignore

try:
    from argparse import BooleanOptionalAction  # py>=3.9
except Exception:  # pragma: no cover
    BooleanOptionalAction = None  # type: ignore[misc]

# =============================================================================
# Small helpers
# =============================================================================
def _add_bool_flag(p: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if BooleanOptionalAction is not None:
        p.add_argument(f"--{name}", action=BooleanOptionalAction, default=default, help=help_text)
    else:
        p.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
        p.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable: {help_text}")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    header_needed = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if header_needed:
            w.writeheader()
        w.writerows(rows)
    print("[SUMMARY] wrote:", str(path))


def make_loggers(base_dir: Path, version: str = "run"):
    base_dir = Path(base_dir)
    loggers: list = []
    if _WANDB_OPTS.get("logger", "both") != "wandb":
        loggers.append(pl_loggers.TensorBoardLogger(
            save_dir=str(base_dir), name="tb", version=version, default_hp_metric=False
        ))
        if _WANDB_OPTS.get("logger", "both") in ("both", "csv"):
            loggers.append(pl_loggers.CSVLogger(save_dir=str(base_dir), name="csv", version=version))
    if _WANDB_OPTS.get("use_wandb") or _WANDB_OPTS.get("logger") == "wandb":
        wl = make_wandb_logger(
            save_root=base_dir,
            name=_WANDB_OPTS.get("wandb_name") or f"{base_dir.name}-{version}",
            offline=bool(_WANDB_OPTS.get("wandb_offline")),
            project=_WANDB_OPTS.get("wandb_project", "stelar"),
            entity=_WANDB_OPTS.get("wandb_entity"),
            tags=_WANDB_OPTS.get("wandb_tags"),
            notes=_WANDB_OPTS.get("wandb_notes"),
        )
        if wl is not None:
            loggers.append(wl)
    return loggers


def build_trainer(
    *,
    devices: List[int],
    max_epochs: int,
    loggers,
    precision: str,
    ckpt_dir: Path,
    monitor_metric: Optional[str],
    grad_clip: float,
) -> Tuple[pl.Trainer, ModelCheckpoint]:
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if monitor_metric:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            save_top_k=1,
            monitor=monitor_metric,
            mode="max",
            filename=f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
            save_last=True,
        )
    else:
        ckpt_cb = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            save_top_k=0,
            save_last=True,
            filename="last",
        )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    accelerator = "gpu" if (devices and torch.cuda.is_available()) else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices if devices else None,
        precision=precision,
        max_epochs=max_epochs,
        logger=loggers,
        callbacks=[lr_monitor, ckpt_cb],
        enable_checkpointing=True,
        enable_progress_bar=True,
        gradient_clip_val=float(grad_clip) if float(grad_clip) > 0 else 0.0,
        strategy="auto",
    )
    return trainer, ckpt_cb



def resolve_pretrain_checkpoint(args) -> Tuple[str, str]:
    # --- LaBraM expects explicit .pth passed via --pretrain_ckpt ---
    if getattr(args, "scheme", "").lower() == "labram":
        if not args.pretrain_ckpt:
            raise ValueError("For --scheme labram, pass --pretrain_ckpt pointing to a LaBraM .pth file.")
        ck = Path(args.pretrain_ckpt)
        if not ck.is_file():
            raise FileNotFoundError(f"--pretrain_ckpt not found (LaBraM expects .pth): {ck}")
        return str(ck), f"LaBraM_{ck.stem}"

    # 1) exact .ckpt
    if args.pretrain_ckpt:
        ck = Path(args.pretrain_ckpt)
        if not ck.is_file():
            raise FileNotFoundError(f"--pretrain_ckpt not found: {ck}")
        try:
            run_tag = ck.parent.parent.parent.name.replace("_tb", "")
        except Exception:
            run_tag = "custom"
        return str(ck), run_tag

    # 2) exact version dir (.../_tb/version_X)
    if args.pretrain_dir:
        ver_dir = Path(args.pretrain_dir)
        if not ver_dir.is_dir():
            raise FileNotFoundError(f"--pretrain_dir is not a directory: {ver_dir}")
        run_tag = ver_dir.parent.name.replace("_tb", "")
        ckpt_path, _ = get_checkpoint_path(
            str(ver_dir),
            which="best" if args.epoch_select == "best" else "last",
        )
        m = re.search(r"(?:^|/)version_(\d+)(?:/|$)", str(ver_dir))
        if m:
            args._pretrain_version = int(m.group(1))
        return ckpt_path, run_tag

    # 3) discover under roots using stable prefix + flags (EEGPT-style)
    stable_prefix = f"EEGPT_{args.tag}_{args.variant}_{args.preprocess}_{args.encoder}_{args.transformer}_"

    if args.pretrain_run_name:
        run_name = args.pretrain_run_name
    else:
        flags_code = args.pretrain_flags_code or getattr(CFG, "pretrain_flags_code", None)
        if not flags_code and hasattr(CFG, "build_pretrain_flags_code"):
            flags_code = CFG.build_pretrain_flags_code(transformer_name=args.transformer)
        run_name = f"{stable_prefix}{flags_code}_tb" if flags_code else None

    roots: List[Path] = []
    if args.pretrain_search_root:
        roots.append(Path(args.pretrain_search_root))
    if hasattr(CFG, "default_pretrain_search_roots"):
        roots.extend(CFG.default_pretrain_search_roots(args.encoder, args.transformer, args.pretrain_max_lr))
    else:
        roots.append(Path(f"../pretrain/{args.encoder}_{args.transformer}/logs/{args.pretrain_max_lr}"))
        roots.append(Path(getattr(CFG, "PRETRAIN_LOG_ROOT", "./logs")) / f"{args.pretrain_max_lr}")

    candidates: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        if run_name:
            p = root / run_name
            if p.is_dir():
                candidates.append(p)
        else:
            for p in root.glob(f"{stable_prefix}*_tb"):
                if p.is_dir():
                    candidates.append(p)

    if not candidates:
        searched = [str(r) for r in roots]
        raise FileNotFoundError(
            "No pretrain runs found.\n"
            f"  stable_prefix: {stable_prefix}\n"
            f"  run_name:      {run_name or '<wildcard>'}\n"
            f"  searched in:   {searched}"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen_run = candidates[0]
    run_tag = chosen_run.name.replace("_tb", "")

    version_dir: Optional[Path] = None
    if getattr(args, "version", None) is not None and int(args.version) >= 0:
        vd = chosen_run / f"version_{int(args.version)}"
        if vd.is_dir():
            version_dir = vd

    if version_dir is None:
        versions = [p for p in chosen_run.glob("version_*") if p.is_dir()]
        if not versions:
            raise FileNotFoundError(f"No version_* dirs under {chosen_run}")
        versions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        version_dir = versions[0]

    ckpt_path, _ = get_checkpoint_path(
        str(version_dir), which="best" if args.epoch_select == "best" else "last"
    )

    m = re.search(r"(?:^|/)version_(\d+)(?:/|$)", str(version_dir))
    if m:
        args._pretrain_version = int(m.group(1))

    return ckpt_path, run_tag

def downstream_log_base(args, model_type: str, extra: str = "") -> Path:
    ver = getattr(args, "_pretrain_version", args.version)
    parts = [
        f"{args.epoch_select}_v{ver}_{args.pretrain_max_lr}_{model_type}_{args.head_type}",
        "TUAB",
        f"seed_{args.seed}",
        "frozen" if args.freeze_encoder else None,
        extra or None,
    ]
    parts = [p for p in parts if p]
    return Path(args.log_root).joinpath(*parts)

# ----------------- Encoder-config capture helpers -----------------

def _collect_encoder_build_flags(args, scheme: str) -> Dict[str, Any]:
    flags: Dict[str, Any] = dict(
        scheme=scheme,
        transformer=args.transformer,
        rope_time=getattr(args, "rope_time", None),
        rope_chan=getattr(args, "rope_chan", None),
        rope_dim_frac=getattr(args, "rope_dim_frac", None),
        rope_interp_time=getattr(args, "rope_interp_time", None),
        rope_interp_chan=getattr(args, "rope_interp_chan", None),
        use_dynamic_pe=getattr(args, "use_dynamic_pe", None),
        use_channel_embed=getattr(args, "use_channel_embed", None),
        load_pretrained=bool(getattr(args, "load_pretrained", True)),
        init_scheme=str(getattr(args, "init_scheme", "")) if not getattr(args, "load_pretrained", True) else "",
        freeze_encoder=bool(getattr(args, "freeze_encoder", False)),
        ft_top_k=int(getattr(args, "ft_top_k", 0) or 0),
    )
    if scheme in {"stelar_patch_adapt", "stelar_patch_adapt_tuev"}:
        flags.update(
            dict(
                fe_use_chan_conv=bool(getattr(args, "use_chan_conv", False)),
                fe_chan_max_norm=float(getattr(args, "chan_max_norm", 1.0)),
                fe_use_time_chan_conv=bool(getattr(args, "use_time_chan_conv", False)),
                fe_conv_kernel_time=int(getattr(args, "conv_kernel_time", 55)),
                fe_conv_dropout=float(getattr(args, "conv_dropout", 0.8)),
            )
        )

    return {k: v for k, v in flags.items() if v is not None}


def _stash_encoder_logging_payload(args, models_cfg: Dict[str, Any], scheme: str) -> None:
    enc_cfg = models_cfg.get("encoder", {}) if isinstance(models_cfg, dict) else {}
    flags = _collect_encoder_build_flags(args, scheme)
    setattr(args, "_encoder_cfg_for_log", enc_cfg)
    setattr(args, "_encoder_flags_for_log", flags)


def _encoder_csv_fields(args) -> Dict[str, Any]:
    enc = getattr(args, "_encoder_cfg_for_log", {}) or {}
    flg = getattr(args, "_encoder_flags_for_log", {}) or {}
    csv_bits: Dict[str, Any] = {}

    for key in ("embed_dim", "embed_num", "depth", "num_heads", "use_criss_cross", "use_channel_embed", "use_dynamic_pe"):
        if key in enc:
            csv_bits[f"enc_{key}"] = enc[key]

    for key in ("rope_time", "rope_chan", "rope_dim_frac", "rope_interp_time", "rope_interp_chan"):
        if key in flg:
            csv_bits[f"enc_{key}"] = flg[key]

    for key in ("load_pretrained", "init_scheme", "freeze_encoder", "ft_top_k"):
        if key in flg:
            csv_bits[key] = flg[key]

    for key in (
        "fe_use_chan_conv",
        "fe_chan_max_norm",
        "fe_use_time_chan_conv",
        "fe_conv_kernel_time",
        "fe_conv_dropout",
    ):
        if key in flg:
            csv_bits[key] = flg[key]

    return csv_bits


def _log_encoder_everywhere(base_dir: Path, loggers, args) -> None:
    enc = getattr(args, "_encoder_cfg_for_log", {}) or {}
    flg = getattr(args, "_encoder_flags_for_log", {}) or {}
    payload = dict(
        encoder_config=enc,
        encoder_flags=flg,
        tag=getattr(args, "tag", None),
        variant=getattr(args, "variant", None),
        preprocess=getattr(args, "preprocess", None),
        encoder_name=getattr(args, "encoder", None),
        transformer_name=getattr(args, "transformer", None),
        pretrain_max_lr=getattr(args, "pretrain_max_lr", None),
        pretrain_version=getattr(args, "_pretrain_version", getattr(args, "version", None)),
        seed=int(getattr(args, "seed", 0)),
    )
    _save_json(Path(base_dir) / "encoder_config.json", payload)

    tb_hparams = {}
    tb_hparams.update(_flatten("enc.", enc))
    tb_hparams.update(_flatten("flag.", flg))
    _log_hparams_to_tb(loggers, tb_hparams)


def make_result_row(
    *,
    args,
    seed: int,
    test_metrics: dict,
    ckpt_cb: ModelCheckpoint,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    use_names = getattr(args, "use_channel_names", None)
    model_target_chans = len(use_names) if use_names else int(args.target_chans)

    row: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "TUAB",
        "seed": int(seed),
        "dataset_in_chans": int(args.in_chans),
        "model_target_chans": int(model_target_chans),
        "best_valid_score": float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else float("nan"),
        "test_balanced_accuracy": float(test_metrics.get("test_balanced_accuracy", float("nan"))),
        "test_acc": float(test_metrics.get("test_accuracy", float("nan"))),
        "test_roc_auc": float(test_metrics.get("test_roc_auc", float("nan"))),
        "test_pr_auc": float(test_metrics.get("test_pr_auc", float("nan"))),
        "test_f1": float(test_metrics.get("test_f1", float("nan"))),
        "scheme": str(args.scheme),
        "tag": str(args.tag),
        "version_arg": int(args.version),
        "pretrain_version": int(getattr(args, "_pretrain_version", args.version)),
        "frozen": int(bool(args.freeze_encoder)),
        "best_ckpt": ckpt_cb.best_model_path or "",
        "grad_clip": float(getattr(args, "grad_clip", 0.0)),
    }
    if extra:
        row.update(extra)
    return row


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("TUAB-only downstream runner (4-fold subject CV)")

    # Required-ish
    p.add_argument("--scheme", type=str, required=True, help="Module key in modules registry (e.g., stelar, eegpt_full_token, ...)")

    # Data
    p.add_argument("--tuab_path", type=str, required=True, help="Root of TUAB downstream dataset folder")

    # 4-fold control
    p.add_argument("--fold", type=int, default=-1, help="Run a single fold [0..3], or -1 for all folds")
    

    # Split controller
    p.add_argument("--split", type=str, default="loso",
                   choices=["loso", "fixed", "eegpt", "cbramod", "5fold", "4fold"],
                   help="For 'fixed', pass --train_subject_ids/--val_subject_ids/--test_subject_ids or --split_spec.")
    p.add_argument("--split_spec", type=str, default=None)
    p.add_argument("--train_subject_ids", type=str, default=None)
    p.add_argument("--val_subject_ids", type=str, default=None)
    p.add_argument("--test_subject_ids", type=str, default=None)

    # Training
    p.add_argument("--batch_size", type=int, default=getattr(CFG, "BATCH_SIZE", 64))
    p.add_argument("--epochs", type=int, default=getattr(CFG, "MAX_EPOCHS", 30))

    # ONE seed for everything (global + CV + dataset split)
    p.add_argument("--seed", type=int, default=getattr(CFG, "SEED", 7))

    p.add_argument("--precision", type=str, default="32")
    p.add_argument("--grad_clip", type=float, default=0.0, help="Lightning gradient_clip_val (0 disables)")

    # Optim (passed through to module)
    p.add_argument("--max_lr_head", type=float, default=getattr(CFG, "MAX_LR_HEAD", 1e-3))
    p.add_argument("--max_lr_enc", type=float, default=getattr(CFG, "MAX_LR_ENC", 1e-5))
    p.add_argument("--wd_head", type=float, default=getattr(CFG, "WD_HEAD", 0.0))
    p.add_argument("--wd_enc", type=float, default=getattr(CFG, "WD_ENC", 0.0))
    p.add_argument("--scheduler", type=str, default=getattr(CFG, "SCHEDULER", "onecycle"), choices=["onecycle", "cosine", "none"])
    p.add_argument("--warmup_pct", type=float, default=getattr(CFG, "WARMUP_PCT", 0.0))
    p.add_argument("--freeze_encoder", action="store_true", default=False)

    # Model I/O
    p.add_argument("--in_chans", type=int, default=getattr(CFG, "IN_CHANS", 19))
    p.add_argument("--target_chans", type=int, default=getattr(CFG, "TARGET_CHANS", 58))
    p.add_argument("--seq_len", type=int, default=getattr(CFG, "SEQ_LEN", 1024))
    p.add_argument("--patch", type=int, default=getattr(CFG, "PATCH", 64))
    p.add_argument("--dropout", type=float, default=getattr(CFG, "DROPOUT", 0.5))
    p.add_argument("--scale_div", type=float, default=100.0)
    p.add_argument("--head_type", type=str, default=getattr(CFG, "HEAD_TYPE", "avg"),
                   choices=["avg", "per_time_attn", "flatten", "eegpt_probe", "all_patch_reps"])

    # Devices/workers
    p.add_argument("--devices", default=",".join(map(str, getattr(CFG, "devices", []))))
    p.add_argument("--num_workers", type=int, default=None)

    # Pretrain ckpt selection (minimal)
    p.add_argument("--epoch_select", type=str, default=getattr(CFG, "epoch_select", "last"), choices=["best", "last"])
    p.add_argument("--version", type=int, default=getattr(CFG, "version", 0))
    p.add_argument("--pretrain_max_lr", type=float, default=getattr(CFG, "pretrain_max_lr", 5e-4))
    p.add_argument("--pretrain_dir", type=str, default=None, help="Exact .../_tb/version_X directory")
    p.add_argument("--pretrain_ckpt", type=str, default=None, help="Exact .ckpt path")


    # Model fingerprint
    p.add_argument("--tag", type=str, default=CFG.tag)
    p.add_argument("--variant", type=str, default=CFG.variant)
    p.add_argument("--preprocess", type=str, default=CFG.preprocess)
    p.add_argument("--encoder", type=str, default=CFG.encoder)
    p.add_argument("--transformer", type=str, default=CFG.transformer)

    # RoPE / ACPE / channel-embedding toggles (STELAR-style)
    _add_bool_flag(p, "use_dynamic_pe", CFG.USE_DYNAMIC_PE, "Enable dynamic positional encoding (ACPE).")
    _add_bool_flag(p, "rope_time", CFG.ROPE_TIME_DEFAULT, "Enable RoPE along time axis (criss-cross).")
    _add_bool_flag(p, "rope_chan", CFG.ROPE_CHAN_DEFAULT, "Enable RoPE along channel axis (criss-cross).")
    p.add_argument("--rope_dim_frac", type=float, default=CFG.ROPE_DIM_FRAC_DEFAULT)
    p.add_argument("--rope_interp_time", type=float, default=CFG.ROPE_INTERP_TIME_DEFAULT)
    p.add_argument("--rope_interp_chan", type=float, default=CFG.ROPE_INTERP_CHAN_DEFAULT)
    _add_bool_flag(p, "use_channel_embed", True, "Use channel embedding with channel IDs inside the encoder.")

    # Channels / montage
    p.add_argument("--dataset_channel_names", type=str, default=None, help="Comma-separated list or JSON list of channel names.")
    p.add_argument("--dataset_channel_set", type=str, default=None, choices=sorted(CHANNELS_BY_DATASET.keys()))
    p.add_argument("--model_channel_set", type=str, default=None, choices=sorted(MONTAGE_BY_DATASET.keys()))


    p.add_argument("--pool", type=str, default=getattr(CFG, "POOL", "mean"), choices=["mean", "max", "meanmax"])
    p.add_argument("--attn_temperature", type=float, default=None)
    p.add_argument("--attn_project_kv", dest="attn_project_kv", action="store_true")
    p.add_argument("--no-attn_project_kv", dest="attn_project_kv", action="store_false")
    p.set_defaults(attn_project_kv=True)

    # LaBraM-specific
    p.add_argument("--labram_model", type=str, default="labram_base_patch200_200")
    p.add_argument("--labram_handle_remainder", type=str, default="crop", choices=["crop", "pad"])
    p.add_argument("--labram_pad_value", type=float, default=0.0)

    # STELAR front-end toggles
    _add_bool_flag(p, "use_chan_conv", True, "1x1 channel-mixing Conv1d before encoder (STELAR).")
    _add_bool_flag(p, "use_time_chan_conv", False, "EEGNet-style temporal depthwise Conv2d (STELAR TUEV).")
    p.add_argument("--chan_max_norm", type=float, default=1.0)
    p.add_argument("--conv_kernel_time", type=int, default=55)
    p.add_argument("--conv_dropout", type=float, default=0.8)


    # Logging / resume / skip
    p.add_argument("--log_root", type=str, default="./logs")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--skip_completed_folds", action="store_true", default=True)
    p.add_argument("--no_skip_completed_folds", dest="skip_completed_folds", action="store_false")

    # ---- Optional YAML config + W&B logging (additive) ----
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML config; CLI flags override its values.")
    add_logging_args(p)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()
    if pre_args.config:
        apply_yaml_defaults(p, pre_args.config)

    args = p.parse_args()

    _WANDB_OPTS.update(
        logger=getattr(args, "logger", "both"),
        use_wandb=bool(getattr(args, "wandb", False)),
        wandb_offline=bool(getattr(args, "wandb_offline", False)),
        wandb_project=getattr(args, "wandb_project", "stelar"),
        wandb_entity=getattr(args, "wandb_entity", None),
        wandb_name=getattr(args, "wandb_name", None),
        wandb_tags=getattr(args, "wandb_tags", ""),
        wandb_notes=getattr(args, "wandb_notes", None),
    )

    # ONE global seed (also seeds dataloader workers)
    pl.seed_everything(int(args.seed), workers=True)

    # Parse devices/workers
    args.devices = parse_devices(args.devices)
    if args.num_workers is None:
        args.num_workers = default_num_workers(args.devices)

    # Channel naming logic (TUAB bookkeeping + adapters)
    if args.dataset_channel_set:
        ds_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
    else:
        ds_names = CHANNELS_BY_DATASET.get("TUAB")
    args.dataset_channel_names = ds_names
    
    # if ds_names:
    #     args.in_chans = len(ds_names)

    if args.model_channel_set:
        args.use_channel_names = MONTAGE_BY_DATASET.get(args.model_channel_set) or list(DEFAULT_EEGPT58)
    else:
        args.use_channel_names = MONTAGE_BY_DATASET.get("TUAB", list(DEFAULT_EEGPT58))
    args.target_chans = len(args.use_channel_names)

    return args


# =============================================================================
# TUAB runner (4-fold subject CV)
# =============================================================================

# =============================================================================
# Engine builder
# =============================================================================

def build_engine(args, *, ckpt_path: Optional[str], steps_per_epoch: int, num_classes: int):
    scheme_lower = args.scheme.lower()

    # LaBraM
    if scheme_lower == "labram":
        if LitLaBraM is None:
            raise ImportError("modules.labram_ft.LitLaBraM not found. Ensure it's importable.")
        _stash_encoder_logging_payload(args, {"encoder": {"timm_model": str(args.labram_model)}}, "labram")
        return LitLaBraM(
            in_chans=int(args.in_chans),
            seq_len=int(args.seq_len),
            patch_size=int(args.patch),
            handle_remainder=str(args.labram_handle_remainder),
            pad_value=float(args.labram_pad_value),
            timm_model_name=str(args.labram_model),
            timm_kwargs=None,
            load_path=str(ckpt_path) if ckpt_path else None,
            student_prefix="student.",
            load_pretrained=bool(getattr(args, "load_pretrained", True)),
            init_scheme=str(getattr(args, "init_scheme", "xavier_uniform")),
            head_type=str(args.head_type),
            num_classes=int(num_classes),
            dropout=float(args.dropout),
            freeze_encoder=bool(args.freeze_encoder),
            lr_head=float(args.max_lr_head),
            lr_enc=float(args.max_lr_enc),
            wd_head=float(args.wd_head),
            wd_enc=float(args.wd_enc),
            max_epochs=int(args.epochs),
            steps_per_epoch=int(steps_per_epoch),
            scheduler=str(args.scheduler),
            warmup_pct=float(args.warmup_pct),
        )

    # CBraMod
    if scheme_lower in {"cbramod", "cbramod_tuev"}:
        if LitCBraMod is None:
            raise ImportError("modules.cbramod_ft.LitCBraMod not found. Make sure it’s in PYTHONPATH.")
        _stash_encoder_logging_payload(
            args,
            {"encoder": {"patch_size": int(args.patch), "seq_len_patches": args.seq_len // max(1, int(args.patch))}},
            "cbramod",
        )
        return LitCBraMod(
            in_chans=int(args.in_chans),
            target_chans=int(args.target_chans),
            patch_size=int(args.patch),
            patch_stride=int(args.patch),
            seq_len=args.seq_len // max(1, int(args.patch)),
            load_path=ckpt_path or "",
            replace_proj_out_with_identity=True,
            head_type=str(args.head_type),
            num_classes=int(num_classes),
            hidden=128,
            dropout=float(args.dropout),
            time_pool=str(args.pool),
            project_kv=bool(args.attn_project_kv),
            temperature=args.attn_temperature,
            freeze_encoder=bool(args.freeze_encoder),
            lr_head=float(args.max_lr_head),
            lr_enc=float(args.max_lr_enc),
            wd_head=float(args.wd_head),
            wd_enc=float(args.wd_enc),
            max_epochs=int(args.epochs),
            steps_per_epoch=int(steps_per_epoch),
            scheduler=str(args.scheduler),
            warmup_pct=float(args.warmup_pct),
        )

    # EEGPT / STELAR via registry
    extra_kwargs: Dict[str, Any] = {}

    if scheme_lower in {"eegpt", "eegpt_tuev"}:
        ModuleClass = build_module(args.scheme)
        models_cfg = CFG.get_config_eegpt(**CFG.EEGPT_MODELS_CONFIGS[args.tag])

    elif scheme_lower in {"stelar_patch_adapt", "stelar_patch_adapt_tuev"}:
        ModuleClass = build_module(args.scheme)
        models_cfg = CFG.get_config(args.tag)

        if args.transformer == "crisscross":
            models_cfg["encoder"].update(
                dict(
                    use_criss_cross=True,
                    rope_time=bool(args.rope_time),
                    rope_chan=bool(args.rope_chan),
                    rope_dim_frac=float(args.rope_dim_frac),
                    rope_interp_time=float(args.rope_interp_time),
                    rope_interp_chan=float(args.rope_interp_chan),
                )
            )
        elif args.transformer == "standard":
            models_cfg["encoder"].update(
                dict(
                    use_criss_cross=False,
                )
            )
        models_cfg["encoder"]["use_dynamic_pe"] = bool(getattr(args, "use_dynamic_pe", CFG.USE_DYNAMIC_PE))
        models_cfg["encoder"]["use_channel_embed"] = bool(getattr(args, "use_channel_embed", True))

        extra_kwargs = dict(
            use_chan_conv=bool(getattr(args, "use_chan_conv", True)),
            chan_max_norm=float(getattr(args, "chan_max_norm", 1.0)),
            load_pretrained=bool(getattr(args, "load_pretrained", True)),
            init_scheme=str(getattr(args, "init_scheme", "xavier_uniform")),
        )
        if scheme_lower == "stelar_patch_adapt_tuev":
            extra_kwargs.update(
                use_time_chan_conv=bool(getattr(args, "use_time_chan_conv", False)),
                conv_kernel_time=int(getattr(args, "conv_kernel_time", 55)),
                conv_dropout=float(getattr(args, "conv_dropout", 0.8)),
            )
    else:
        raise NotImplementedError(f"Unknown or unsupported scheme: {args.scheme}")

    _stash_encoder_logging_payload(args, models_cfg, args.scheme)

    print(
        f"[INFO] in_chans={args.in_chans}, target_chans={args.target_chans}, "
        f"seq_len={args.seq_len}, patch={args.patch}"
    )
    return ModuleClass(
        models_cfg=models_cfg,
        load_path=str(ckpt_path) if ckpt_path else None,
        num_classes=num_classes,
        steps_per_epoch=steps_per_epoch,
        max_epochs=args.epochs,
        lr_head=args.max_lr_head,
        lr_enc=args.max_lr_enc,
        wd_head=args.wd_head,
        wd_enc=args.wd_enc,
        scheduler=args.scheduler,
        warmup_pct=args.warmup_pct,
        in_chans=args.in_chans,
        target_chans=args.target_chans,
        seq_len=args.seq_len,
        patch=args.patch,
        head_type=args.head_type,
        dropout=args.dropout,
        channel_names=args.dataset_channel_names,
        use_channel_names=args.use_channel_names,
        freeze_encoder=args.freeze_encoder,
        ft_top_k=getattr(args, "ft_top_k", 0),
        **extra_kwargs,
    )


def _ckpt_global_step(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return int(ckpt.get("global_step", -1))


def run_tuab_4fold(args):
    try:
        from tuab_dataset import LoadTrainValDataset, LoadTest
        from tuab_dataset import _scan_tuab
    except Exception:
        from downstream.datasets.tuab_dataset import LoadTrainValDataset, LoadTest
        from downstream.datasets.tuab_dataset import _scan_tuab

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base(args, model_type, extra="4fold")

    files, labels, subj_ids = _scan_tuab(args.tuab_path, "train")
    subj_tokens = sorted(set(subj_ids))

    kf = KFold(n_splits=4)
    fold_subjects = [[] for _ in range(4)]
    for fold_idx, (_, test_idx) in enumerate(kf.split(subj_tokens)):
        for idx in test_idx:
            fold_subjects[fold_idx].append(subj_tokens[idx])

    rows: List[Dict[str, Any]] = []
    metric_keys: Optional[List[str]] = None
    metric_accum: Dict[str, float] = {}

    class _Params:
        datasets_dir: str = args.tuab_path
        batch_size: int = args.batch_size
        num_workers: int = args.num_workers
        pin_memory: bool = True
        drop_last: bool = False
        shuffle_Test: bool = False
        scale_div: float = args.scale_div
        transform = None
    def _parse_monitor_from_name(p, monitor: str) -> Optional[float]:
        # tries to parse "...monitor=0.8123..." from checkpoint filename
        import re
        s = p.name
        m = re.search(rf"{re.escape(monitor)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _pick_best_ckpt_from_dir(ckpt_dir: Path, monitor: Optional[str]) -> Optional[str]:
        if not ckpt_dir.exists():
            return None
        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            return None

        last_ckpt = ckpt_dir / "last.ckpt"
        non_last = [p for p in ckpts if p.name != "last.ckpt"]

        # If we have a monitor, try to pick the max value from filenames
        if monitor and non_last:
            scored = []
            for p in non_last:
                v = _parse_monitor_from_name(p, monitor)
                if v is not None:
                    scored.append((v, p))
            if scored:
                scored.sort(key=lambda t: t[0], reverse=True)
                return str(scored[0][1])

        # Fallbacks
        if last_ckpt.is_file():
            return str(last_ckpt)

        # Otherwise: newest by mtime
        non_last = non_last if non_last else ckpts
        non_last.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(non_last[0])
    test_loader = LoadTest(_Params()).get_data_loader()

    def _run_one_fold(fold_idx: int, val_subjects: List[str]):
        class _ParamsTV:
            datasets_dir: str = args.tuab_path
            batch_size: int = args.batch_size
            split_mode: str = "subject_random"
            val_subject_ids = val_subjects
            seed: int = int(args.seed)  # single seed
            num_workers: int = args.num_workers
            pin_memory: bool = True
            drop_last: bool = False
            shuffle_train: bool = True
            scale_div: float = args.scale_div
            transform = None

        loaders = LoadTrainValDataset(_ParamsTV()).get_data_loader()
        train_loader = loaders["train"]
        val_loader = loaders.get("val", None)

        steps = math.ceil(len(train_loader))
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=2)

        fold_dir = base_dir / f"fold{fold_idx}"
        ckpt_dir = fold_dir / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        done_flag = fold_dir / ".done"
        best_ckpt_txt = fold_dir / "best_ckpt.txt"
        test_metrics_json = fold_dir / "test_metrics.json"

        monitor_metric = "valid_balanced_accuracy" if val_loader is not None else None

        loggers = make_loggers(fold_dir, version="run")
        trainer, ckpt_cb = build_trainer(
            devices=args.devices,
            max_epochs=int(args.epochs),
            loggers=loggers,
            precision=str(args.precision),
            ckpt_dir=ckpt_dir,
            monitor_metric="valid_balanced_accuracy" if val_loader is not None else None,
            grad_clip=float(args.grad_clip),
        )
        # ----------------------------
        # Otherwise: FIT then TEST best ckpt
        # ----------------------------
        resume_ckpt = None
        resume_gs = -1

        if bool(getattr(args, "resume", True)):
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.is_file():
                resume_ckpt = str(last_ckpt)
            else:
                ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpts:
                    resume_ckpt = str(ckpts[0])

            # Compute "planned" total steps for OneCycle (per-rank)
            # Your module uses steps_per_epoch=len(train_loader) and epochs=args.epochs
            planned_total_steps = int(len(train_loader) * int(args.epochs))

            # If resuming, check if ckpt already finished => skip fit and go straight to test
            if resume_ckpt is not None:
                try:
                    resume_gs = _ckpt_global_step(resume_ckpt)
                except Exception as e:
                    print(f"[TUAB][Fold {fold_idx}] Could not read global_step from {resume_ckpt}: {e}")
                    resume_gs = -1

                if resume_gs >= planned_total_steps - 1:
                    print(
                        f"[TUAB][Fold {fold_idx}] Resume ckpt already finished "
                        f"(global_step={resume_gs}, planned_total_steps={planned_total_steps}). "
                        f"Skipping fit and testing."
                    )

                    # Mark fold done (so requeues don't loop)
                    try:
                        done_flag.touch()
                    except Exception:
                        pass

                    # Pick best checkpoint and test
                    best_path = _pick_best_ckpt_from_dir(ckpt_dir, monitor_metric)
                    if not best_path:
                        print(f"[TUAB][Fold {fold_idx}] No ckpt found to test in {ckpt_dir}.")
                        return

                    test_metrics = trainer.test(model=model, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]

                    # persist
                    try:
                        test_metrics_json.write_text(json.dumps(test_metrics, indent=2))
                    except Exception:
                        pass

                    row = make_result_row(
                        seed=args.seed,
                        args=args,
                        test_metrics=test_metrics,
                        ckpt_cb=ckpt_cb,
                        extra={
                            "strategy": "4fold",
                            "fold_number": fold_idx,
                            "fold_val_subject_ids": ",".join(map(str, val_subjects)),
                            "best_ckpt": best_path,
                            "skipped_fit": True,
                            "resume_global_step": resume_gs,
                            "planned_total_steps": planned_total_steps,
                        },
                    )
                    rows.append(row)

                    nonlocal metric_keys, metric_accum
                    if metric_keys is None:
                        metric_keys = list(test_metrics.keys())
                        metric_accum = {k: 0.0 for k in metric_keys}
                    for k in metric_keys:
                        metric_accum[k] += float(test_metrics[k])

                    return

            # Normal training path
            trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)


        # Choose best ckpt path robustly
        best_path = ckpt_cb.best_model_path or None
        if not best_path:
            # fallback: parse from dir (or last)
            best_path = _pick_best_ckpt_from_dir(ckpt_dir, monitor_metric)

        # Save best ckpt path for future "done" re-tests
        if best_path:
            try:
                best_ckpt_txt.write_text(str(best_path))
            except Exception:
                pass

        test_metrics = trainer.test(
            model=None, ckpt_path=best_path, dataloaders=test_loader, verbose=False
        )[0]

        # Persist metrics (optional)
        try:
            test_metrics_json.write_text(json.dumps(test_metrics, indent=2))
        except Exception:
            pass

        row = make_result_row(
            seed=args.seed,
            args=args,
            test_metrics=test_metrics,
            ckpt_cb=ckpt_cb,
            extra={
                "strategy": "4fold",
                "fold_number": fold_idx,
                "fold_val_subject_ids": ",".join(map(str, val_subjects)),
                "best_ckpt": best_path or (ckpt_cb.best_model_path or ""),
                "skipped_fit": False,
            },
        )
        rows.append(row)

        metric_keys, metric_accum
        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys:
            metric_accum[k] += float(test_metrics[k])

        try:
            done_flag.touch()
        except Exception:
            pass

    # Run folds
    if args.fold != -1:
        print(f"\n[TUAB] === 4-fold {args.fold+1}/4 ===")
        _run_one_fold(args.fold, fold_subjects[args.fold])
    else:
        for fold_idx, val_subjects in enumerate(fold_subjects):
            print(f"\n[TUAB] === 4-fold {fold_idx+1}/4 ===")
            _run_one_fold(fold_idx, val_subjects)

    # Mean over actually-evaluated folds (more robust than hard /4)
    if metric_keys:
        n_eval = 0
        for r in rows:
            if str(r.get("fold_number", "")) not in ("MEAN",):
                n_eval += 1
        n_eval = max(n_eval, 1)

        avg_row = {"dataset": "TUAB", "strategy": "4fold", "fold_number": "MEAN", "fold_val_subject_ids": "MEAN"}
        for k in metric_keys:
            avg_row[f"test/{k}"] = metric_accum[k] / float(n_eval)
        rows.append(avg_row)

    out_csv = str((base_dir / "csv" / "summary.csv"))
    out_csv = Path(out_csv)
    write_csv(out_csv, rows)
    print(f"\n[TUAB] 4-fold finished. Summary at: {out_csv}")

def main():
    args = parse_args()
    run_tuab_4fold(args)


if __name__ == "__main__":
    main()
