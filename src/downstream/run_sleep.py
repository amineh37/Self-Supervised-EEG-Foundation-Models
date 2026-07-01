#!/usr/bin/env python3
"""
Downstream evaluation runner — SLEEPEDF only (5-fold)

Kept from the original:
- pretrain checkpoint resolution (explicit ckpt/dir or auto-discovery)
- scheme-based engine construction (registry + optional LaBraM/CBraMod)
- logging (TB + CSV) + best-ckpt selection
- per-fold resume + ".done" sentinel skipping
"""

from __future__ import annotations

import argparse
import csv
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

CFG = importlib.import_module("finetune_configs")

# add pretraining/ to sys.path (relative to this file)
sys.path.append(str(Path(__file__).resolve().parents[1] / "pretraining"))

from modules import build_module  # noqa: E402
from utils import default_num_workers, get_checkpoint_path, parse_devices  # noqa: E402
from channels import CHANNELS_BY_DATASET, DEFAULT_EEGPT58, MONTAGE_BY_DATASET  # noqa: E402

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
# Utilities
# =============================================================================

def _add_bool_flag(p: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if BooleanOptionalAction is not None:
        p.add_argument(f"--{name}", action=BooleanOptionalAction, default=default, help=help_text)
    else:
        p.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
        p.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable: {help_text}")


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    header_needed = not out.exists()
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if header_needed:
            w.writeheader()
        w.writerows(rows)
    print("[SUMMARY] wrote:", str(out))


# =============================================================================
# Checkpoint discovery
# =============================================================================

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


def downstream_log_base(dataset: str, args, model_type: str, extra: str = "") -> Path:
    ver = getattr(args, "_pretrain_version", args.version)
    head_name = args.head_type
    use_val = float(getattr(args, "val_ratio", 0.0) or 0.0) > 0.0
    parts = [
        f"{args.epoch_select}_v{ver}_{args.pretrain_max_lr}_{model_type}_{head_name}",
        dataset,
        f"seed_{args.seed}",
        "frozen" if getattr(args, "freeze_encoder", False) else None,
        "val" if use_val else "noval",
        extra or None,
    ]
    parts = [p for p in parts if p]
    return Path(args.log_root).joinpath(*parts)


def make_loggers(base_dir: Path, version: str):
    base_dir = Path(base_dir)
    tb = pl_loggers.TensorBoardLogger(
        save_dir=str(base_dir), name="tb", version=version, default_hp_metric=False
    )
    csv = pl_loggers.CSVLogger(save_dir=str(base_dir), name="csv", version=version)
    return [tb, csv]


def build_trainer_with_callbacks(
    args,
    devices: List[int],
    max_epochs: int,
    loggers,
    precision: str = "32",
    monitor_metric: Optional[str] = "valid_cohen_kappa",
    ckpt_dir: Optional[Path] = None,
) -> Tuple[pl.Trainer, ModelCheckpoint]:

    if monitor_metric:
        ckpt_kwargs: Dict[str, Any] = dict(
            save_top_k=1,
            monitor=monitor_metric,
            mode="max",
            filename=f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        )
    else:
        ckpt_kwargs = dict(save_top_k=0, save_last=True, filename="last")

    if ckpt_dir is not None:
        ckpt_kwargs["dirpath"] = str(ckpt_dir)

    ckpt_cb = ModelCheckpoint(**ckpt_kwargs)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    accelerator = "gpu" if (devices and torch.cuda.is_available()) else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices if devices else None,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[lr_monitor, ckpt_cb],
        enable_checkpointing=True,
        logger=loggers,
        strategy="auto",
        enable_progress_bar=True,
        gradient_clip_val=float(getattr(args, "grad_clip", 0.0)),
    )
    return trainer, ckpt_cb


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
    dataset: str,
    seed: int,
    args,
    test_metrics: dict,
    ckpt_cb: ModelCheckpoint,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    use_names = getattr(args, "use_channel_names", None)
    model_target_chans = (
        (len(use_names) if use_names else 0)
        or int(getattr(args, "target_chans", 0) or 0)
        or int(getattr(args, "in_chans", 0) or 0)
    )
    montage_preview = (
        ",".join(use_names[:10]) + ("..." if len(use_names) > 10 else "")
        if use_names else ""
    )

    row: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset,
        "seed": seed,
        "dataset_in_chans": args.in_chans,
        "model_target_chans": model_target_chans,
        "model_montage": montage_preview,
        "best_valid_cohen_kappa": (
            float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else float("nan")
        ),
        "test_balanced_accuracy": float(test_metrics.get("test_balanced_accuracy", float("nan"))),
        "test_acc": float(test_metrics.get("test_acc", float("nan"))),
        "test_f1_macro": float(test_metrics.get("test_f1_macro", float("nan"))),
        "test_f1_weighted": float(test_metrics.get("test_f1_weighted", float("nan"))),
        "test_cohen_kappa": float(test_metrics.get("test_cohen_kappa", float("nan"))),
        "scheme": args.scheme,
        "tag": args.tag,
        "version_arg": args.version,
        "pretrain_version": getattr(args, "_pretrain_version", args.version),
        "frozen": int(bool(getattr(args, "freeze_encoder", False))),
    }
    if extra:
        row.update(extra)
    return row


# =============================================================================
# CLI (SLEEPEDF only)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("Downstream evaluation runner (SLEEPEDF only)")

    # Task / scheme
    p.add_argument("--task", type=str, default="SLEEPEDF", choices=["SLEEPEDF"])
    p.add_argument("--scheme", type=str, required=True,
                   help="Module key in modules registry (e.g., stelar, eegpt_full_token, ...), or labram/cbramod.")

    # Split controller (SleepEDF = 5-fold)
    p.add_argument("--split", type=str, default="5fold", choices=["5fold"])
    p.add_argument("--fold", type=int, default=-1, help="Run a single fold (0-4). -1 runs all folds.")

    # Generic overrides
    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=CFG.MAX_EPOCHS)
    p.add_argument("--seed", type=int, default=getattr(CFG, "SEED", 7))

    # Optim & schedule
    p.add_argument("--max_lr_head", type=float, default=CFG.MAX_LR_HEAD)
    p.add_argument("--max_lr_enc", type=float, default=CFG.MAX_LR_ENC)
    p.add_argument("--wd_head", type=float, default=CFG.WD_HEAD)
    p.add_argument("--wd_enc", type=float, default=CFG.WD_ENC)
    p.add_argument("--scheduler", type=str, default=CFG.SCHEDULER, choices=["onecycle", "cosine", "none"])
    p.add_argument("--warmup_pct", type=float, default=CFG.WARMUP_PCT)
    p.add_argument("--freeze_encoder", action="store_true", default=False)

    # Validation & dataset toggles
    p.add_argument("--val_ratio", type=float, default=0.0)
    p.add_argument("--scale_div", type=float, default=100.0)

    # Pretrain ckpt selection
    p.add_argument("--epoch_select", type=str, default=getattr(CFG, "epoch_select", "last"), choices=["best", "last"])
    p.add_argument("--version", type=int, default=getattr(CFG, "version", 0))
    p.add_argument("--pretrain_max_lr", type=float, default=getattr(CFG, "pretrain_max_lr", 5e-4))
    p.add_argument("--pretrain_dir", type=str, default=None)
    p.add_argument("--pretrain_ckpt", type=str, default=None)
    p.add_argument("--pretrain_search_root", type=str, default=None)
    p.add_argument("--pretrain_run_name", type=str, default=None)
    p.add_argument("--pretrain_flags_code", type=str, default=getattr(CFG, "pretrain_flags_code", None))

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


    # Frontend / head options
    p.add_argument("--in_chans", type=int, default=getattr(CFG, "IN_CHANS", 2))
    p.add_argument("--target_chans", type=int, default=getattr(CFG, "TARGET_CHANS", 58))
    p.add_argument("--seq_len", type=int, default=getattr(CFG, "SEQ_LEN", 1024))
    p.add_argument("--patch", type=int, default=getattr(CFG, "PATCH", 64))
    p.add_argument("--dropout", type=float, default=getattr(CFG, "DROPOUT", 0.5))
    p.add_argument("--head_type", type=str, default=getattr(CFG, "HEAD_TYPE", "avg"),
                   choices=["avg", "per_time_attn", "flatten", "eegpt_probe", "all_patch_reps"])
    p.add_argument("--pool", type=str, default=getattr(CFG, "POOL", "mean"),
                   choices=["mean", "max", "meanmax"])
    p.add_argument("--attn_temperature", type=float, default=None)
    p.add_argument("--attn_project_kv", dest="attn_project_kv", action="store_true")
    p.add_argument("--no_attn_project_kv", dest="attn_project_kv", action="store_false")
    p.set_defaults(attn_project_kv=True)

    # Dataset channels / montage
    p.add_argument("--dataset_channel_set", type=str, default=None, choices=sorted(CHANNELS_BY_DATASET.keys()))
    p.add_argument("--model_channel_set", type=str, default=None, choices=sorted(MONTAGE_BY_DATASET.keys()))
    p.add_argument("--dataset_channel_names", type=str, default=None)  # kept for compatibility

    # Devices / workers / precision
    p.add_argument("--devices", default=",".join(map(str, getattr(CFG, "devices", []))))
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--precision", type=str, default="32")

    # Data roots
    p.add_argument("--sleepedf_path", type=str, default="/home/infres/vnguyen-23/data/EEGPT_data/downstream/sleep_edf_all_data/TrainFold")

    # Log root
    p.add_argument("--log_root", type=str, default="./logs")

    # LaBraM-specific
    p.add_argument("--labram_model", type=str, default="labram_base_patch200_200")
    p.add_argument("--labram_handle_remainder", type=str, default="crop", choices=["crop", "pad"])
    p.add_argument("--labram_pad_value", type=float, default=0.0)

    # Resumption controls
    _add_bool_flag(p, "resume", True, "Resume training from last.ckpt when available.")
    _add_bool_flag(p, "skip_completed_folds", True, "Skip folds that already have a .done sentinel.")

    p.add_argument("--grad_clip", type=float, default=0.0,
                help="Gradient clipping value (0 disables).")

    args = p.parse_args()
    
    pl.seed_everything(int(args.seed), workers=True)
    # Devices & workers
    args.devices = parse_devices(args.devices)
    if args.num_workers is None:
        args.num_workers = default_num_workers(args.devices)

    # Montage/channel logic: skip for LaBraM (it doesn't need channel adapters)
    if args.scheme.lower() in {"cbramod"}:
        pass
    elif args.scheme.lower() == "labram":
        if args.dataset_channel_set:
            args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
            if args.dataset_channel_names:
                args.in_chans = len(args.dataset_channel_names)
        args.use_channel_names = None
        args.target_chans = int(args.in_chans)
    else:
        # DATASET channels (compute in_chans)
        if args.dataset_channel_set:
            args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
        else:
            args.dataset_channel_names = CHANNELS_BY_DATASET.get("SLEEPEDF")
        if args.dataset_channel_names:
            args.in_chans = len(args.dataset_channel_names)

        # MODEL montage -> names (drives target_chans)
        if args.model_channel_set:
            args.use_channel_names = MONTAGE_BY_DATASET.get(args.model_channel_set) or list(DEFAULT_EEGPT58)
        else:
            args.use_channel_names = MONTAGE_BY_DATASET.get("SLEEPEDF", list(DEFAULT_EEGPT58))
        args.target_chans = len(args.use_channel_names)

    return args


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
    print (models_cfg)

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


# =============================================================================
# SLEEPEDF (5-fold)
# =============================================================================
def run_sleepedf_5fold(args):
    try:
        from sleepedf_dataset import LoadDataset
        from sleepedf_dataset import _scan_sleepedf_auto
    except Exception:
        from downstream.datasets.sleepedf_dataset import LoadDataset
        from downstream.datasets.sleepedf_dataset import _scan_sleepedf_auto

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("SLEEPEDF", args, model_type, extra="5fold")

    files, labels, subj_ids = _scan_sleepedf_auto(args.sleepedf_path)
    subj_tokens = sorted(set(subj_ids))
    n_subj = len(subj_tokens)
    print(f"[SLEEPEDF] Discovered {n_subj} subjects: {subj_tokens}")

    kf = KFold(n_splits=5)
    fold_subjects = [[] for _ in range(5)]
    for fold_idx, (_, test_idx) in enumerate(kf.split(subj_tokens)):
        for idx in test_idx:
            fold_subjects[fold_idx].append(subj_tokens[idx])

    rows: List[Dict[str, Any]] = []
    metric_keys: Optional[List[str]] = None
    metric_accum: Dict[str, float] = {}

    # ---- Helpers to recover best ckpt when skipping ----
    def _parse_monitor_from_name(p: Path, monitor: str) -> Optional[float]:
        import re
        m = re.search(rf"{re.escape(monitor)}=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", p.name)
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

        # If we have a monitor, prefer the max parsed from filename
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
        non_last = non_last if non_last else ckpts
        non_last.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(non_last[0])

    def _run_one_fold(fold_idx: int, test_subjects: List[str]):
        nonlocal metric_keys, metric_accum  # MUST be first in the function

        class _Params:
            datasets_dir: str = args.sleepedf_path
            batch_size: int = args.batch_size
            split_mode: str = "subject_random"
            test_subject_ids = test_subjects
            val_ratio: float = args.val_ratio
            seed: int = int(args.seed)  # single seed
            num_workers: int = args.num_workers
            pin_memory: bool = True
            drop_last: bool = False
            shuffle_train: bool = True
            scale_div: float = args.scale_div
            transform = None

        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader = loaders["train"]
        val_loader = loaders.get("val", None)
        test_loader = loaders["test"]
        steps = math.ceil(len(train_loader))

        fold_dir = base_dir / f"fold{fold_idx}"
        ckpt_dir = fold_dir / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        done_flag = fold_dir / ".done"
        best_ckpt_txt = fold_dir / "best_ckpt.txt"
        test_metrics_json = fold_dir / "test_metrics.json"

        monitor_metric = "valid_cohen_kappa" if val_loader is not None else None

        loggers = make_loggers(fold_dir, version="run")

        trainer, ckpt_cb = build_trainer_with_callbacks(
            args,
            args.devices,
            args.epochs,
            loggers,
            args.precision,
            monitor_metric=monitor_metric,
            ckpt_dir=ckpt_dir,
        )

        # Build model (needed for testing even if skipping fit)
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=5)

        # ----------------------------
        # DONE: skip fit but still test best ckpt
        # ----------------------------
        if bool(getattr(args, "skip_completed_folds", True)) and done_flag.exists():
            best_path = None
            if best_ckpt_txt.is_file():
                try:
                    best_path = best_ckpt_txt.read_text().strip() or None
                except Exception:
                    best_path = None
            if not best_path:
                best_path = _pick_best_ckpt_from_dir(ckpt_dir, monitor_metric)

            if not best_path:
                print(f"[SLEEP][Fold {fold_idx}] .done exists but no ckpt found in {ckpt_dir}. Skipping.")
                return

            print(f"[SLEEP][Fold {fold_idx}] .done exists — testing best ckpt: {best_path}")
            test_metrics = trainer.test(model=model, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]

            try:
                test_metrics_json.write_text(json.dumps(test_metrics, indent=2))
            except Exception:
                pass

            row = make_result_row(
                dataset="SLEEPEDF",
                seed=args.seed,
                args=args,
                test_metrics=test_metrics,
                ckpt_cb=ckpt_cb,
                extra={
                    "strategy": "5fold",
                    "fold_number": fold_idx,
                    "fold_test_subject_ids": ",".join(map(str, test_subjects)),
                    "best_ckpt": best_path or "",
                    "skipped_fit": True,
                },
            )
            rows.append(row)

            if metric_keys is None:
                metric_keys = list(test_metrics.keys())
                metric_accum = {k: 0.0 for k in metric_keys}
            for k in metric_keys:
                metric_accum[k] += float(test_metrics[k])

            return

        # ----------------------------
        # FIT then test best ckpt
        # ----------------------------
        resume_ckpt = None
        if bool(getattr(args, "resume", True)):
            last_ckpt = ckpt_dir / "last.ckpt"
            if last_ckpt.is_file():
                resume_ckpt = str(last_ckpt)
            else:
                ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                if ckpts:
                    resume_ckpt = str(ckpts[0])

        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

        best_path = ckpt_cb.best_model_path or None
        if not best_path:
            best_path = _pick_best_ckpt_from_dir(ckpt_dir, monitor_metric)

        # Persist best ckpt for future skip+test runs
        if best_path:
            try:
                best_ckpt_txt.write_text(str(best_path))
            except Exception:
                pass

        test_metrics = trainer.test(model=None, ckpt_path=best_path, dataloaders=test_loader, verbose=False)[0]

        try:
            test_metrics_json.write_text(json.dumps(test_metrics, indent=2))
        except Exception:
            pass

        row = make_result_row(
            dataset="SLEEPEDF",
            seed=args.seed,
            args=args,
            test_metrics=test_metrics,
            ckpt_cb=ckpt_cb,
            extra={
                "strategy": "5fold",
                "fold_number": fold_idx,
                "fold_test_subject_ids": ",".join(map(str, test_subjects)),
                "best_ckpt": best_path or (ckpt_cb.best_model_path or ""),
                "skipped_fit": False,
            },
        )
        rows.append(row)

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
        print(f"\n[SLEEPEDF] === 5-fold {args.fold+1}/5 — test subjects: {fold_subjects[args.fold]} ===")
        _run_one_fold(args.fold, fold_subjects[args.fold])
    else:
        for fold_idx, test_subjects in enumerate(fold_subjects):
            print(f"\n[SLEEPEDF] === 5-fold {fold_idx+1}/5 — test subjects: {test_subjects} ===")
            _run_one_fold(fold_idx, test_subjects)

    # Mean over evaluated folds (more robust than hard /5)
    if metric_keys:
        n_eval = 0
        for r in rows:
            if str(r.get("fold_number", "")) != "MEAN":
                n_eval += 1
        n_eval = max(n_eval, 1)

        avg_row = {"dataset": "SLEEPEDF", "strategy": "5fold", "fold_number": "MEAN", "fold_test_subject_ids": "MEAN"}
        for k in metric_keys:
            avg_row[f"test/{k}"] = metric_accum[k] / float(n_eval)
        rows.append(avg_row)

    out_csv = str((base_dir / "csv" / "summary.csv"))
    write_csv(out_csv, rows)
    print(f"\n[SLEEPEDF] 5-fold finished. Summary at: {out_csv}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    if args.task != "SLEEPEDF" or args.split != "5fold":
        raise ValueError("This runner only supports: --task SLEEPEDF --split 5fold")
    run_sleepedf_5fold(args)


if __name__ == "__main__":
    main()
