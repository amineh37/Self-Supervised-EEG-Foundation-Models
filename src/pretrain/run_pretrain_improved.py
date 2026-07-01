#!/usr/bin/env python3
"""
STELAR-improved pretraining driver.

Reuses the baseline CLI (run_pretrain.build_parser) for all data/model/optim
args, adds the improvement controls, and trains LitSTELARImproved.

Examples
--------
# faithful PDF baseline (== run_pretrain.py)
python run_pretrain_improved.py --preset v0_baseline \
  --input_mode pretoken --data_backend lmdb \
  --train_lmdb $DATA/STELAR/data/pretrain/train_90pct \
  --valid_lmdb $DATA/STELAR/data/pretrain/val_10pct \
  --tag tiny --mask_ratio 0.5 --max_epochs 30

# #1 only (attention-guided masking) -- diagnostic
python run_pretrain_improved.py --preset v1_attnmask ...

# #2 only (augmentations + asymmetric views) -- Option B / diagnostic
python run_pretrain_improved.py --preset v2_aug ...

# Option A (combo) -- tune tau on top of fixed #2
python run_pretrain_improved.py --preset vA_combo --attn_tau_start 1.0 --attn_tau_min 0.1 ...

# any version + VICReg anti-collapse (only if embedding std collapses)
python run_pretrain_improved.py --preset vA_combo --use_vicreg ...
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks as pl_cb
from pytorch_lightning.callbacks import ModelCheckpoint

try:
    from argparse import BooleanOptionalAction
except Exception:  # pragma: no cover
    BooleanOptionalAction = None  # type: ignore

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.multiprocessing.set_sharing_strategy("file_system")

from configs import (
    get_config,
    get_dataloaders,
    seed_torch,
    steps_per_epoch_from_loader,
)
from run_pretrain import build_parser  # reuse all shared args
from pretrain_engine_improved import LitSTELARImproved
from versions import get_version, list_versions

# Repo-root `stelar_utils` package (optional YAML config + W&B logger layer).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stelar_utils import add_logging_args, apply_yaml_defaults, build_loggers
from stelar_utils.loggers import primary_log_dir


# --------------------------------------------------------------------------- #
def add_improved_args(p):
    g = p.add_argument_group("improvements")
    g.add_argument("--preset", type=str, default="v0_baseline",
                   help=f"Named version. One of: {list_versions()}")
    if BooleanOptionalAction is not None:
        g.add_argument("--use_aug", action=BooleanOptionalAction, default=None,
                       help="Override preset: enable #2 augmentation / asymmetric views.")
        g.add_argument("--use_attn_mask", action=BooleanOptionalAction, default=None,
                       help="Override preset: enable #1 attention-guided masking.")
        g.add_argument("--use_vicreg", action=BooleanOptionalAction, default=None,
                       help="Enable VICReg anti-collapse (3rd MTPE-GB task).")
    else:  # pragma: no cover
        g.add_argument("--use_aug", dest="use_aug", action="store_true", default=None)
        g.add_argument("--use_attn_mask", dest="use_attn_mask", action="store_true", default=None)
        g.add_argument("--use_vicreg", dest="use_vicreg", action="store_true", default=None)
    # #1 knobs
    g.add_argument("--attn_tau_start", type=float, default=None)
    g.add_argument("--attn_tau_min", type=float, default=None)
    g.add_argument("--attn_tau_epochs", type=int, default=None)
    g.add_argument("--attn_imp_floor", type=float, default=None)
    # VICReg knobs
    g.add_argument("--vicreg_gamma", type=float, default=None)
    g.add_argument("--lambda_vicreg", type=float, default=None)
    # #2 student-aug overrides (None -> module STRONG defaults)
    g.add_argument("--student_channel_dropout", type=float, default=None)
    g.add_argument("--student_noise_std", type=float, default=None)
    g.add_argument("--student_amp_lo", type=float, default=None)
    g.add_argument("--student_amp_hi", type=float, default=None)
    return p


def resolve_improved_kwargs(args) -> dict:
    """Merge preset defaults with explicit CLI overrides (CLI wins when not None)."""
    kw = get_version(args.preset)

    for name in ("use_aug", "use_attn_mask", "use_vicreg",
                 "attn_tau_start", "attn_tau_min", "attn_tau_epochs", "attn_imp_floor",
                 "vicreg_gamma", "lambda_vicreg"):
        v = getattr(args, name, None)
        if v is not None:
            kw[name] = v

    # student-aug overrides -> build a student_aug dict from STRONG defaults
    from improvements.augment import STRONG_AUG
    overrides = {}
    if args.student_channel_dropout is not None:
        overrides["channel_dropout_p"] = args.student_channel_dropout
    if args.student_noise_std is not None:
        overrides["noise_std"] = args.student_noise_std
    if args.student_amp_lo is not None:
        overrides["amp_scale_lo"] = args.student_amp_lo
    if args.student_amp_hi is not None:
        overrides["amp_scale_hi"] = args.student_amp_hi
    if overrides:
        kw["student_aug"] = {**STRONG_AUG, **overrides}
    return kw


# --------------------------------------------------------------------------- #
def main() -> None:
    # Optional YAML config: pre-scan for --config, then apply it as parser
    # defaults so explicit CLI flags still override it (CLI > YAML > defaults).
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = add_improved_args(build_parser())
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config; CLI flags override its values.")
    add_logging_args(parser)
    if pre_args.config:
        apply_yaml_defaults(parser, pre_args.config)

    args = parser.parse_args()
    improved_kwargs = resolve_improved_kwargs(args)
    print(f"[preset] {args.preset} -> {improved_kwargs}")

    seed_torch(args.seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    devices: List[int] = [int(x) for x in args.devices.split(",") if x.strip() != ""] if args.devices else []
    accelerator = "gpu" if (torch.cuda.is_available() and len(devices) > 0) else "cpu"

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
    steps_per_epoch = steps_per_epoch_from_loader(train_loader)
    if steps_per_epoch <= 0:
        raise RuntimeError("steps_per_epoch <= 0. Check dataset / batch size.")
    print(f"[info] steps_per_epoch={steps_per_epoch} | val={have_val}")

    # infer shapes
    batch = next(iter(train_loader))
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    if args.input_mode == "pretoken":
        _, c_auto, n_auto, d_auto = x.shape
        img_size: Tuple[int, int] = (args.img_c or int(c_auto), args.img_n or int(n_auto))
        patch_size = args.patch_size or int(d_auto)
    else:
        _, c_auto, t_auto = x.shape
        img_size = (args.img_c or int(c_auto), args.img_t or int(t_auto))
        patch_size = args.patch_size or 64

    # model config (same as baseline runner)
    models_cfg = get_config(tag_name=args.tag)
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
    models_cfg["encoder"].update({"use_dynamic_pe": bool(args.use_dynamic_pe)})
    models_cfg["encoder"].update({"use_channel_embed": bool(args.use_channel_embed)})
    models_cfg.setdefault("predictor", {})
    models_cfg["predictor"]["kv_mode"] = args.pred_kv_mode

    channels_override: Optional[List[str]] = None
    if args.channels.strip():
        channels_override = [c.strip() for c in args.channels.split(",") if c.strip()]

    model = LitSTELARImproved(
        # ---- base (faithful PDF) ----
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
        USE_LOSS_A=bool(args.use_align_mask),
        USE_ALIGN_VIS=False,
        USE_ALIGN_MASK=args.use_align_mask,
        USE_LN=args.use_ln,
        USE_SKIP=args.use_skip,
        lambda_recon=float(args.lambda_recon),
        lambda_align_mask=float(args.lambda_align_mask),
        mtpe_gb_ema_c=float(args.mtpe_gb_ema_c),
        mtpe_gb_eps=float(args.mtpe_gb_eps),
        mtpe_gb_clamp=float(args.mtpe_gb_clamp),
        # ---- improvements (from preset + CLI) ----
        **improved_kwargs,
    )

    # logging / checkpoints
    model_type = f"STELAR_improved_{args.preset}_{args.tag}_{args.transformer}"
    log_save_root = Path(args.log_root) / f"{args.pretrain_max_lr}"
    loggers = build_loggers(
        logger=args.logger,
        save_root=log_save_root,
        name=model_type,
        use_wandb=args.wandb,
        wandb_offline=args.wandb_offline,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_name=args.wandb_name,
        wandb_tags=args.wandb_tags,
        wandb_notes=args.wandb_notes,
        wandb_config=vars(args),
    )

    ckpt_dir = primary_log_dir(loggers, fallback=log_save_root / model_type) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    monitor = "valid_loss" if have_val else "train_loss"

    callbacks = [pl_cb.LearningRateMonitor(logging_interval="epoch")]
    if have_val:
        callbacks.append(ModelCheckpoint(
            dirpath=str(ckpt_dir), monitor=monitor, mode="min", save_top_k=3, save_last=True,
            filename=f"best-{{epoch:03d}}-{{{monitor}:.6f}}",
        ))
    callbacks.append(ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="last-{epoch:03d}", every_n_epochs=1,
        save_on_train_epoch_end=True, save_top_k=1,
    ))

    strategy = "ddp_find_unused_parameters_true" if (accelerator == "gpu" and len(devices) > 1) else "auto"

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
        callbacks=callbacks,
        logger=(loggers if loggers else False),
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        **extra_trainer_kwargs,
    )

    if valid_loader is None:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, valid_loader)

    print(f"[done] ckpts -> {ckpt_dir}")


if __name__ == "__main__":
    main()
