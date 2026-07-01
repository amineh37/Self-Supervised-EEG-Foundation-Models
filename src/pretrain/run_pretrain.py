#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_cb, loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from argparse import BooleanOptionalAction  # py>=3.9
except Exception:
    BooleanOptionalAction = None  # type: ignore

# Env hygiene
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.multiprocessing.set_sharing_strategy("file_system")

# Project imports
from configs import (
    FILE_EXTENSIONS,
    NUM_WORKERS,
    PIN_LOG_VERSION as CFG_PIN,
    PRED_CC,
    PRED_KV_MODE_DEFAULT as CFG_PRED_KV_MODE,
    RESUME as CFG_RESUME,
    ROPE_CHAN_DEFAULT,
    ROPE_DIM_FRAC_DEFAULT,
    ROPE_INTERP_CHAN_DEFAULT,
    ROPE_INTERP_TIME_DEFAULT,
    ROPE_TIME_DEFAULT,
    TRAIN_ROOT,
    VALID_ROOT,
    devices as CFG_DEVICES,
    encoder,
    get_config,
    get_dataloaders,
    global_batch_size as CFG_BS,
    mask_ratio as CFG_MASK_RATIO,
    max_epochs as CFG_MAX_EPOCHS,
    pretrain_max_lr as CFG_LR,
    preprocess,
    seed_torch,
    steps_per_epoch_from_loader,
    tag as CFG_TAG,
    transformer as CFG_TRANSFORMER,
    variant,
)

# Optional dynamic PE default
try:
    from configs import USE_DYNAMIC_PE as CFG_USE_DYNAMIC_PE
except Exception:
    CFG_USE_DYNAMIC_PE = False

# Optional default for channel embedding
try:
    from configs import USE_CHANNEL_EMBED as CFG_USE_CHANNEL_EMBED
except Exception:
    CFG_USE_CHANNEL_EMBED = True

from configs import (
    USE_LN as CFG_USE_LN,
    USE_LOSS_AM as CFG_USE_LOSS_AM,
    USE_LOSS_R as CFG_USE_LOSS_R,
    USE_SKIP as CFG_USE_SKIP,
)

from pretrain_engine_w_mtpe_gb import LitSTELAR  # new engine (MTPE_GB inside)
from utils import get_checkpoint_path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _flatten_dict(d, prefix: str = "") -> dict:
    flat: dict = {}
    for k, v in (d.items() if isinstance(d, dict) else []):
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            if isinstance(v, (int, float, bool)) or v is None:
                flat[key] = v
            else:
                s = str(v)
                flat[key] = s if len(s) <= 200 else s[:197] + "..."
    return flat


def _safe_len(x):
    try:
        return len(x)
    except Exception:
        return None


def _count_params(mod: torch.nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in mod.parameters() if p.requires_grad)
    total = sum(p.numel() for p in mod.parameters())
    return trainable, total


def _maybe(v, default=None):
    try:
        return v
    except Exception:
        return default


def _to_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _add_bool_flag(p: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if BooleanOptionalAction is not None:
        p.add_argument(f"--{name}", action=BooleanOptionalAction, default=default, help=help_text)
    else:
        p.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
        p.add_argument(f"--no_{name}", dest=name, action="store_false", help=f"Disable: {help_text}")


def _warn_ignored_flag(name: str, reason: str) -> None:
    print(f"[WARN] Ignoring '{name}': {reason}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("STELAR pretraining (new engine: MTPE_GB + masked-align + recon)")

    # Training/model
    p.add_argument("--tag", default=CFG_TAG)
    p.add_argument("--pretrain_max_lr", type=float, default=CFG_LR)
    p.add_argument("--batch_size", type=int, default=CFG_BS)
    p.add_argument("--max_epochs", type=int, default=CFG_MAX_EPOCHS)
    p.add_argument(
        "--devices",
        type=str,
        default=",".join(map(str, CFG_DEVICES)),
        help="Comma-separated GPU ids, e.g. '0' or '0,1'. Empty -> CPU.",
    )
    _add_bool_flag(p, "resume", CFG_RESUME, "Resume from last checkpoint if available.")
    p.add_argument("--version", type=int, default=0)
    p.add_argument(
        "--pin_log_version",
        type=int,
        default=CFG_PIN,
        help="Pinned version (0,1,2...). Use -1 to auto-create.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log_root", type=str, default="./logs")

    # Transformer family
    p.add_argument(
        "--transformer",
        choices=["standard", "crisscross"],
        default=CFG_TRANSFORMER,
        help="'crisscross' enables dual time×channel attention.",
    )

    # Data roots
    p.add_argument("--train_root", type=str, default=str(TRAIN_ROOT))
    p.add_argument("--valid_root", type=str, default=str(VALID_ROOT))
    p.add_argument("--exts", type=str, default=",".join(FILE_EXTENSIONS))
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)

    # Input mode & shapes
    p.add_argument("--input_mode", choices=["raw", "pretoken"], default="raw")
    p.add_argument("--pretoken_layout", choices=["CND", "NCD"], default="CND")
    p.add_argument("--montage", choices=["", "58", "19"], default="")
    p.add_argument("--img_c", type=int, default=None)
    p.add_argument("--img_t", type=int, default=None)
    p.add_argument("--img_n", type=int, default=None)
    p.add_argument("--patch_size", type=int, default=None)
    p.add_argument("--mask_ratio", type=float, default=CFG_MASK_RATIO)

    # Channels override
    p.add_argument("--channels", type=str, default="", help="Comma-separated channel list.")

    # Feature switches
    _add_bool_flag(p, "use_loss_r", CFG_USE_LOSS_R, "Enable reconstruction loss.")
    _add_bool_flag(p, "use_align_mask", CFG_USE_LOSS_AM, "Enable masked-token alignment loss.")
    _add_bool_flag(p, "use_ln", CFG_USE_LN, "LayerNorm on recon targets (y) before recon loss.")
    _add_bool_flag(p, "use_skip", CFG_USE_SKIP, "Use skip combine for recon input.")
    _add_bool_flag(p, "use_dynamic_pe", CFG_USE_DYNAMIC_PE, "Enable dynamic positional encoding.")
    _add_bool_flag(p, "use_channel_embed", CFG_USE_CHANNEL_EMBED, "Use channel IDs + channel embedding.")

    # Criss-Cross / RoPE
    _add_bool_flag(p, "rope_time", ROPE_TIME_DEFAULT, "RoPE along time axis.")
    _add_bool_flag(p, "rope_chan", ROPE_CHAN_DEFAULT, "RoPE along channel axis.")
    p.add_argument("--rope_dim_frac", type=float, default=ROPE_DIM_FRAC_DEFAULT)
    p.add_argument("--rope_interp_time", type=float, default=ROPE_INTERP_TIME_DEFAULT)
    p.add_argument("--rope_interp_chan", type=float, default=ROPE_INTERP_CHAN_DEFAULT)
    _add_bool_flag(p, "pred_cc", PRED_CC, "Use crisscross in predictor.")

    # Predictor KV policy
    p.add_argument(
        "--pred_kv_mode",
        choices=["global_vis_kv", "all"],
        default=CFG_PRED_KV_MODE,
        help="Predictor KV attention policy.",
    )

    # Backend / LMDB
    p.add_argument("--data_backend", choices=["files", "lmdb"], default="files")
    p.add_argument("--train_lmdb", type=str, default=None)
    p.add_argument("--valid_lmdb", type=str, default=None)

    # Validation
    p.add_argument("--no_valid", action="store_true", help="Disable validation.")
    p.add_argument("--val_fraction", type=float, default=0.0)

    # Loss weights: now *initial priors* for MTPE_GB
    p.add_argument("--lambda_recon", type=float, default=1.0, help="MTPE_GB init prior for recon.")
    p.add_argument("--lambda_align_mask", type=float, default=1.0, help="MTPE_GB init prior for masked-align.")

    # MTPE_GB knobs
    p.add_argument("--mtpe_gb_ema_c", type=float, default=0.5, help="EMA smoothing for MTPE_GB weights (0..1).")
    p.add_argument("--mtpe_gb_eps", type=float, default=1e-8, help="Numerical epsilon for MTPE_GB.")
    p.add_argument("--mtpe_gb_clamp", type=float, default=1e6, help="Clamp for MTPE_GB urgency ratio.")

    # Step-wise validation & logging cadence
    p.add_argument("--val_check_interval", type=float, default=1.0)
    p.add_argument("--log_every_n_steps", type=int, default=100)
    _add_bool_flag(p, "epoch_only_logging", True, "Log only epoch-level metrics (LR logged per epoch).")

    # Local smoke-test / debugging (CPU-friendly)
    p.add_argument("--fast_dev_run", action="store_true",
                   help="PL fast_dev_run: 1 train+val batch then stop (no logging/ckpt). For smoke tests.")
    p.add_argument("--limit_train_batches", type=int, default=0,
                   help="If >0, cap train batches per epoch (smoke test on a single LMDB).")
    p.add_argument("--limit_val_batches", type=int, default=0,
                   help="If >0, cap val batches per epoch (smoke test).")

    return p


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    args = build_parser().parse_args()
    print(f"[args] {args}")

    # warn if legacy is set
    if bool(getattr(args, "use_align_vis", False)):
        _warn_ignored_flag("--use_align_vis", "new engine does not implement visible alignment.")
    if getattr(args, "lambda_align_vis", 1.0) not in (0.0, 1.0):
        _warn_ignored_flag("--lambda_align_vis", "new engine does not use visible alignment weight.")

    seed_torch(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    # Devices / accelerator
    devices: List[int] = [int(x) for x in args.devices.split(",") if x.strip() != ""] if args.devices else []
    accelerator = "gpu" if (torch.cuda.is_available() and len(devices) > 0) else "cpu"

    # Dataloaders
    exts = [e if e.startswith(".") else f".{e}" for e in args.exts.split(",") if e.strip()]
    train_loader, valid_loader = get_dataloaders(
        train_root=Path(args.train_root),
        valid_root=Path(args.valid_root),
        extensions=exts,
        global_bs=args.batch_size,
        num_workers=args.num_workers,
        devs=devices,
        input_mode=args.input_mode,
        pretoken_layout=args.pretoken_layout,
        data_backend=args.data_backend,
        train_lmdb=Path(args.train_lmdb) if args.train_lmdb else None,
        valid_lmdb=None if args.no_valid else (Path(args.valid_lmdb) if args.valid_lmdb else None),
        val_fraction=(0.0 if args.no_valid else args.val_fraction),
        seed=args.seed,
    )
    have_val = valid_loader is not None
    print(f"Train loader: {len(train_loader)}, Val loader: {_safe_len(valid_loader)}")

    steps_per_epoch = steps_per_epoch_from_loader(train_loader)
    if steps_per_epoch <= 0:
        raise RuntimeError("Computed steps_per_epoch <= 0. Check dataset and batch size.")
    print(f"[info] steps_per_epoch = {steps_per_epoch}")

    # Infer shapes from a sample
    batch = next(iter(train_loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch

    if args.input_mode == "pretoken":
        _, c_auto, n_auto, d_auto = x.shape
        c_final = args.img_c or int(c_auto)
        n_final = args.img_n or int(n_auto)
        img_size: Tuple[int, int] = (c_final, n_final)
        patch_size: int = args.patch_size or int(d_auto)
        if args.patch_size is None:
            print(f"[infer] pretoken: inferred D={patch_size} from data")
    else:
        _, c_auto, t_auto = x.shape
        c_final = args.img_c or int(c_auto)
        t_final = args.img_t or int(t_auto)
        img_size = (c_final, t_final)
        patch_size = args.patch_size or 64
        if args.patch_size is None:
            print(f"[infer] raw: using default patch_size={patch_size}")

    # Model config
    models_cfg = get_config(tag_name=args.tag)

    # Criss-Cross flags
    if args.transformer == "crisscross":
        cc_pack = {
            "use_criss_cross": True,
            "rope_time": bool(args.rope_time),
            "rope_chan": bool(args.rope_chan),
            "rope_dim_frac": float(args.rope_dim_frac),
            "rope_interp_time": float(args.rope_interp_time),
            "rope_interp_chan": float(args.rope_interp_chan),
        }
        models_cfg["encoder"].update(cc_pack)
        if args.pred_cc:
            models_cfg["predictor"].update(cc_pack)

    # Dynamic PE + channel embed
    models_cfg["encoder"].update({"use_dynamic_pe": bool(args.use_dynamic_pe)})
    models_cfg["encoder"].update({"use_channel_embed": bool(args.use_channel_embed)})

    # Predictor KV policy
    models_cfg.setdefault("predictor", {})
    models_cfg["predictor"]["kv_mode"] = args.pred_kv_mode

    # Some naming codes
    im_code = "PT" if args.input_mode == "pretoken" else "RW"
    mon_code = f"M{args.montage or 'XX'}"
    cc_enc = (args.transformer == "crisscross")
    cc_pred = bool(args.pred_cc)
    if cc_enc and cc_pred:
        cc_code = "CCep"
    elif cc_enc:
        cc_code = "CCe"
    elif cc_pred:
        cc_code = "CCp"
    else:
        cc_code = "STD"
    kv_code = "ALL" if args.pred_kv_mode == "all" else "GV"
    pe_code = "".join(
        [
            "T" if args.rope_time else "t",
            "C" if args.rope_chan else "c",
            "E" if args.use_channel_embed else "e",
            "A" if args.use_dynamic_pe else "a",
        ]
    )
    loss_code = "".join(
        [
            "R" if args.use_loss_r else "r",
            "M" if args.use_align_mask else "m",
        ]
    )
    flags_code = f"{loss_code}-N{int(bool(args.use_ln))}-S{int(bool(args.use_skip))}-{im_code}-{mon_code}-{cc_code}-KV{kv_code}-PE{pe_code}-MTPEGB"

    var_str = variant if isinstance(variant, str) else str(variant)
    pre_str = preprocess if isinstance(preprocess, str) else str(preprocess)
    enc_str = encoder if isinstance(encoder, str) else str(encoder)
    model_type_base = f"STELAR_{args.tag}_{var_str}_{pre_str}_{enc_str}_{args.transformer}_{flags_code}"
    print(f"[model] base_type: {model_type_base}")

    # Channels override
    channels_override: Optional[List[str]] = None
    if args.channels.strip():
        channels_override = [c.strip() for c in args.channels.split(",") if c.strip()]

    # Engine (new)
    model = LitSTELAR(
        models_configs=models_cfg,
        img_size=img_size,
        patch_size=patch_size,
        input_mode=args.input_mode,
        mask_ratio=args.mask_ratio,
        montage=args.montage if not channels_override else None,
        channels=channels_override,
        steps_per_epoch=steps_per_epoch,
        max_epochs=args.max_epochs,
        pretrain_max_lr=args.pretrain_max_lr,
        ema_start_end=(0.996, 1.0),
        ref_wd=1e-2,
        final_wd=1e-2,
        USE_LOSS_R=args.use_loss_r,
        USE_LOSS_A=bool(args.use_align_mask),  # alignment enabled only if masked-align enabled
        USE_ALIGN_VIS=False,                   # force-off
        USE_ALIGN_MASK=args.use_align_mask,
        USE_LN=args.use_ln,
        USE_SKIP=args.use_skip,
        # MTPE_GB init priors + knobs
        lambda_recon=float(args.lambda_recon),
        lambda_align_mask=float(args.lambda_align_mask),
        mtpe_gb_ema_c=float(args.mtpe_gb_ema_c),
        mtpe_gb_eps=float(args.mtpe_gb_eps),
        mtpe_gb_clamp=float(args.mtpe_gb_clamp),
    )

    # Reconstructor-aware run tag
    rec_class = model.reconstructor.__class__.__name__
    if "Transformer" in rec_class:
        rec_code = "TREC"
    elif "Light" in rec_class:
        rec_code = "LREC"
    else:
        rec_code = rec_class.upper()
    model_type = f"{model_type_base}_{rec_code}"
    print(f"[model] reconstructor: {rec_class} -> {rec_code}")
    print(f"[model] full model_type: {model_type}")

    # Loggers
    pin = None if args.pin_log_version in (-1, None) else args.pin_log_version
    csv_root = Path(args.log_root) / f"{args.pretrain_max_lr}" / f"{model_type}_csv"
    tb_root = Path(args.log_root) / f"{args.pretrain_max_lr}" / f"{model_type}_tb"
    if pin is None:
        csv_logger = pl_loggers.CSVLogger(save_dir=str(csv_root), name="", version=None)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tb_root), name="", version=None)
    else:
        csv_logger = pl_loggers.CSVLogger(save_dir=str(csv_root), name="", version=pin)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir=str(tb_root), name="", version=pin)

    ckpt_dir = Path(tb_logger.log_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor_metric = "valid_loss" if have_val else "train_loss"
    best_name = "val" if have_val else "train"

    ckpt_best = None
    if have_val:
        ckpt_best = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename=(f"best-{best_name}-epoch{{epoch:03d}}-" + "{" + f"{monitor_metric}" + ":.6f}"),
            monitor=monitor_metric,
            mode="min",
            save_top_k=3,
            save_last=True,
        )

    ckpt_last = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=("last-epoch{epoch:03d}" + "{" + f"{monitor_metric}" + ":.6f}"),
        save_top_k=1,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )

    # Resume
    ckpt_path_for_fit: Optional[str] = None
    if args.resume:
        try:
            ckpt_path, ep = get_checkpoint_path(str(ckpt_dir), which="last")
        except Exception as e:
            print(f"[resume] no checkpoint found in {ckpt_dir}: {e}")
        else:
            ckpt_path_for_fit = ckpt_path
            print(f"[resume] from {ckpt_path} at epoch {ep}")

    strategy = "ddp_find_unused_parameters_true" if (accelerator == "gpu" and len(devices) > 1) else "auto"

    epoch_only = bool(getattr(args, "epoch_only_logging", True))
    lr_logging_interval = "epoch" if epoch_only else "step"
    if epoch_only:
        log_every_n_steps = max(steps_per_epoch + 1, 10**9)
        val_check_interval = 1.0
    else:
        log_every_n_steps = args.log_every_n_steps
        val_check_interval = args.val_check_interval

    callbacks = [pl_cb.LearningRateMonitor(logging_interval=lr_logging_interval), ckpt_last]
    if ckpt_best is not None:
        callbacks.insert(1, ckpt_best)

    # Optional caps for local smoke-testing on a single LMDB.
    extra_trainer_kwargs: dict = {}
    if getattr(args, "fast_dev_run", False):
        extra_trainer_kwargs["fast_dev_run"] = True
    if getattr(args, "limit_train_batches", 0) and args.limit_train_batches > 0:
        extra_trainer_kwargs["limit_train_batches"] = int(args.limit_train_batches)
    if getattr(args, "limit_val_batches", 0) and args.limit_val_batches > 0:
        extra_trainer_kwargs["limit_val_batches"] = int(args.limit_val_batches)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=(devices if accelerator == "gpu" else "auto"),
        strategy=strategy,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        val_check_interval=val_check_interval,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        **extra_trainer_kwargs,
    )

    # Train
    t0 = time.time()
    if valid_loader is None:
        trainer.fit(model, train_loader, ckpt_path=ckpt_path_for_fit)
    else:
        trainer.fit(model, train_loader, valid_loader, ckpt_path=ckpt_path_for_fit)
    elapsed_hours = (time.time() - t0) / 3600.0

    # ---------------- richer hyperparam snapshot ---------------- #
    C, N = _maybe(model.encoder.num_patches, (args.img_c or 0, args.img_n or 0))
    P = _maybe(model.reconstructor.patch_size, args.patch_size)

    enc_tr, enc_all = _count_params(model.encoder)
    pred_tr, pred_all = _count_params(model.predictor)
    rec_tr, rec_all = _count_params(model.reconstructor)

    env_info = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_0": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "cudnn": _maybe(torch.backends.cudnn.version(), None),
        "platform": platform.platform(),
    }

    train_ds_len = _maybe(_safe_len(getattr(train_loader, "dataset", None)))
    valid_ds_len = _maybe(_safe_len(getattr(valid_loader, "dataset", None))) if valid_loader is not None else None

    total_steps = int(args.max_epochs * steps_per_epoch)
    warmup_steps = max(500, int(0.1 * total_steps))

    hparams_full = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_type": model_type,
        "log_root": args.log_root,
        "ckpt_dir": str(ckpt_dir),
        "backend": args.data_backend,
        "train_root": args.train_root,
        "valid_root": (None if args.no_valid else args.valid_root),
        "train_lmdb": args.train_lmdb,
        "valid_lmdb": args.valid_lmdb,
        "train_loader_len": _safe_len(train_loader),
        "valid_loader_len": _safe_len(valid_loader) if valid_loader is not None else None,
        "train_dataset_len": train_ds_len,
        "valid_dataset_len": valid_ds_len,
        "input_mode": args.input_mode,
        "montage": args.montage or "default",
        "img_size": {"C": C, "N": N, "T": (_maybe(args.img_t))},
        "patch_size": P,
        "mask_ratio": args.mask_ratio,
        "transformer": args.transformer,
        "predictor_crisscross": bool(args.pred_cc),
        "pred_kv_mode": args.pred_kv_mode,
        "use_dynamic_pe": bool(args.use_dynamic_pe),
        "use_channel_embed": bool(args.use_channel_embed),
        # legacy flags (logged but ignored)
        "legacy_use_align_vis": bool(getattr(args, "use_align_vis", False)),
        "legacy_lambda_align_vis": float(getattr(args, "lambda_align_vis", 1.0)),
        # actual losses toggles
        "losses": {
            "use_loss_r": bool(args.use_loss_r),
            "use_align_mask": bool(args.use_align_mask),
            "use_ln": bool(args.use_ln),
            "use_skip": bool(args.use_skip),
        },
        "mtpe_gb": {
            "lambda_recon_init": float(args.lambda_recon),
            "lambda_align_mask_init": float(args.lambda_align_mask),
            "ema_c": float(args.mtpe_gb_ema_c),
            "eps": float(args.mtpe_gb_eps),
            "clamp": float(args.mtpe_gb_clamp),
        },
        "epoch_only_logging": epoch_only,
        "max_epochs": args.max_epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "pretrain_max_lr": args.pretrain_max_lr,
        "warmup_steps": warmup_steps,
        "ema_start": _maybe(model.ema_start, None),
        "ema_end": _maybe(model.ema_end, None),
        "ref_wd": _maybe(model.ref_wd, None),
        "final_wd": _maybe(model.final_wd, None),
        "grad_clip_val": 1.0,
        "params": {
            "encoder": {"trainable": enc_tr, "total": enc_all},
            "predictor": {"trainable": pred_tr, "total": pred_all},
            "reconstructor": {"trainable": rec_tr, "total": rec_all},
            "total_trainable": enc_tr + pred_tr + rec_tr,
            "total_all": enc_all + pred_all + rec_all,
        },
        "rope": {
            "time": bool(args.rope_time),
            "chan": bool(args.rope_chan),
            "dim_frac": args.rope_dim_frac,
            "interp_time": args.rope_interp_time,
            "interp_chan": args.rope_interp_chan,
        },
        "env": env_info,
        "models_cfg": models_cfg,
    }

    run_dir = Path(tb_logger.log_dir)
    manifest_path = run_dir / "hparams_full.json"
    manifest_path.write_text(json.dumps(hparams_full, indent=2))
    with open(run_dir / "hparams_flat.txt", "w") as f:
        for k, v in sorted(_flatten_dict(hparams_full).items()):
            f.write(f"{k}: {v}\n")

    # preview schedules to CSV (LR & WD)
    sched_csv = run_dir / "schedules_preview.csv"
    with open(sched_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "lr", "wd"])
        start_factor = 1e-3
        ref_wd = _maybe(model.ref_wd, 1e-2)
        final_wd = _maybe(model.final_wd, 1e-2)
        for t in range(total_steps + 1):
            if t <= warmup_steps:
                frac = t / max(1, warmup_steps)
                lr = args.pretrain_max_lr * (start_factor + (1.0 - start_factor) * frac)
            else:
                tt = t - warmup_steps
                T = max(1, total_steps - warmup_steps)
                lr = 0.5 * args.pretrain_max_lr * (1.0 + math.cos(math.pi * tt / T))
            wd = final_wd + 0.5 * (ref_wd - final_wd) * (1.0 + math.cos(math.pi * t / max(1, total_steps)))
            w.writerow([t, lr, wd])

    tb_hparams = {
        "lr_peak": float(args.pretrain_max_lr),
        "mask_ratio": float(args.mask_ratio),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "params_total_million": round((enc_all + pred_all + rec_all) / 1e6, 3),
        "enc_params_m": round(enc_all / 1e6, 3),
        "pred_params_m": round(pred_all / 1e6, 3),
        "rec_params_m": round(rec_all / 1e6, 3),
        "mtpe_gb_c": float(args.mtpe_gb_ema_c),
        "w_init_recon": float(args.lambda_recon),
        "w_init_align_mask": float(args.lambda_align_mask),
        "legacy_use_align_vis": int(bool(getattr(args, "use_align_vis", False))),
        "epoch_only_logging": int(epoch_only),
        "use_channel_embed": int(bool(args.use_channel_embed)),
    }
    tb_logger.log_hyperparams(tb_hparams)

    time_file = Path(csv_logger.log_dir) / "pretraining_time.txt"
    with open(time_file, "w") as f:
        f.write(f"{elapsed_hours:.4f} hours\n")

    print(f"[hparams] wrote {manifest_path} and {sched_csv}")
    print(f"[logs] TB:  {tb_logger.log_dir}")
    print(f"[logs] CSV: {csv_logger.log_dir}")
    print(f"[ckpts]    {ckpt_dir}")
    print(f"[time]     {elapsed_hours:.4f} h -> {time_file}")


if __name__ == "__main__":
    main()
