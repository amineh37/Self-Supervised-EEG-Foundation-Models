#!/usr/bin/env python3
"""
TUEV downstream runner (single split or subject-wise K-fold)

TUEV-only runner with explicit checkpoint selection (no auto discovery).
Seed policy: a SINGLE seed (--seed) is used everywhere (PL/Torch/NumPy splits).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

import importlib

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

# Optional modules
try:
    from modules.cbramod_ft import LitCBraMod  # noqa: E402
except Exception:
    LitCBraMod = None  # type: ignore

try:
    from modules.labram_ft import LitLaBraM  # noqa: E402
except Exception:
    LitLaBraM = None  # type: ignore

try:
    from modules.csbrain_ft import LitCSBrain  # noqa: E402
except Exception:
    LitCSBrain = None  # type: ignore

try:
    from argparse import BooleanOptionalAction  # py>=3.9
except Exception:  # pragma: no cover
    BooleanOptionalAction = None  # type: ignore[misc]


# =============================================================================
# Small utilities
# =============================================================================

def _add_bool_flag(p: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    if BooleanOptionalAction is not None:
        p.add_argument(f"--{name}", action=BooleanOptionalAction, default=default, help=help_text)
    else:
        p.add_argument(f"--{name}", dest=name, action="store_true", default=default, help=help_text)
        p.add_argument(f"--no-{name}", dest=name, action="store_false", help=f"Disable: {help_text}")


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    file_exists = out.exists()
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            w.writeheader()
        w.writerows(rows)
    print("[SUMMARY] wrote:", str(out))


def _flatten(prefix: str, d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(f"{key}.", v))
        else:
            out[key] = v
    return out


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("[CONFIG] wrote:", str(path))


def _log_hparams_to_tb(loggers, hparams: Dict[str, Any]) -> None:
    for lg in (loggers or []):
        try:
            if isinstance(lg, pl_loggers.TensorBoardLogger):
                lg.log_hyperparams(hparams)
                break
        except Exception:
            pass


def _family_from_scheme(s: str) -> str:
    s = (s or "").lower()
    if s.startswith("stelar"):
        return "STELAR"
    if s.startswith("eegpt"):
        return "EEGPT"
    if s == "labram":
        return "LaBraM"
    if s.startswith("cbramod"):
        return "CBraMod"
    return (s or "model").upper()


def _discover_resume_ckpt(ckpt_dir: Path) -> Optional[str]:
    if not ckpt_dir.is_dir():
        return None
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        return str(last_ckpt)
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(ckpts[0]) if ckpts else None


def _best_or_last_from_cb(ckpt_cb: ModelCheckpoint) -> Optional[str]:
    return ckpt_cb.best_model_path or getattr(ckpt_cb, "last_model_path", None)


# =============================================================================
# Checkpoint resolution (NO discovery)
# =============================================================================

def resolve_pretrain_checkpoint(args) -> Tuple[Optional[str], str]:
    family = _family_from_scheme(getattr(args, "scheme", ""))

    if not getattr(args, "load_pretrained", True):
        run_tag = f"{family}_{args.tag}_{args.variant}_{args.preprocess}_{args.encoder}_{args.transformer}_randinit"
        return None, run_tag

    scheme_lower = getattr(args, "scheme", "").lower()

    if scheme_lower == "labram":
        ck = Path(str(getattr(args, "pretrain_ckpt", "")))
        if not ck.is_file():
            raise ValueError("For --scheme labram with pretrained, pass --pretrain_ckpt pointing to a LaBraM .pth file.")
        return str(ck), f"LaBraM_{ck.stem}"

    if getattr(args, "pretrain_ckpt", None):
        ck = Path(str(args.pretrain_ckpt))
        if not ck.is_file():
            raise FileNotFoundError(f"--pretrain_ckpt not found: {ck}")
        try:
            run_tag = ck.parent.parent.parent.name.replace("_tb", "")
        except Exception:
            run_tag = f"{family}_custom"
        return str(ck), run_tag

    if getattr(args, "pretrain_dir", None):
        ver_dir = Path(str(args.pretrain_dir))
        if not ver_dir.is_dir():
            raise FileNotFoundError(f"--pretrain_dir is not a directory: {ver_dir}")
        run_tag = ver_dir.parent.name.replace("_tb", "")
        ckpt_path, _ = get_checkpoint_path(str(ver_dir), which="best" if args.epoch_select == "best" else "last")

        m = re.search(r"(?:^|/)version_(\d+)(?:/|$)", str(ver_dir))
        if m:
            args._pretrain_version = int(m.group(1))

        print("--- Checkpoint selection details ---")
        print(f"Resolved pretrained ckpt from --pretrain_dir: {ckpt_path}")
        return ckpt_path, run_tag

    raise ValueError(
        "Checkpoint discovery is disabled.\n"
        "Either pass --pretrain_ckpt (exact .ckpt/.pth) or --pretrain_dir (.../_tb/version_X),\n"
        "or run with --no-load_pretrained for random initialization."
    )


# =============================================================================
# Logging helpers
# =============================================================================

def downstream_log_base(args, model_type: str, extra: str = "") -> Path:
    ver = getattr(args, "_pretrain_version", args.version)
    head_name = args.head_type
    use_val = float(getattr(args, "val_ratio", 0.0) or 0.0) > 0.0
    family = _family_from_scheme(args.scheme)
    scheme = (args.scheme or "").lower()

    enc_tag: Optional[str] = None
    if scheme.startswith("stelar"):
        pe_bits: List[str] = []
        use_ce = bool(getattr(args, "use_channel_embed", True))
        pe_bits.append("chid" if use_ce else "nochid")
        if bool(getattr(args, "use_dynamic_pe", False)):
            pe_bits.append("acpe")
        if bool(getattr(args, "rope_time", False)):
            pe_bits.append("ropeT")
        if bool(getattr(args, "rope_chan", False)):
            pe_bits.append("ropeC")
        enc_tag = "_".join(pe_bits) if pe_bits else None

    if not getattr(args, "load_pretrained", True):
        run_tag = f"{family}_{args.tag}_{args.variant}_{args.preprocess}_{args.encoder}_{args.transformer}_randinit"
        parts = [run_tag, "TUEV", enc_tag, "randinit", "val" if use_val else "noval", extra or None]
        return Path(args.log_root).joinpath(*[p for p in parts if p])

    topk_tag = None
    if not getattr(args, "freeze_encoder", False):
        k = int(getattr(args, "ft_top_k", 0) or 0)
        if k > 0:
            topk_tag = f"topk{k}"

    parts = [
        f"{args.epoch_select}_v{ver}_{args.pretrain_max_lr}_{model_type}_{head_name}",
        "TUEV",
        f"seed_{args.seed}",
        enc_tag,
        "pretrained",
        "frozen" if getattr(args, "freeze_encoder", False) else None,
        topk_tag,
        "val" if use_val else "noval",
        extra or None,
    ]
    return Path(args.log_root).joinpath(*[p for p in parts if p])


def make_loggers(base_dir: Path, version: str, *, enable_csv: bool = True):
    base_dir = Path(base_dir)
    loggers: list = []
    if _WANDB_OPTS.get("logger", "both") != "wandb":
        loggers.append(pl_loggers.TensorBoardLogger(save_dir=str(base_dir), name="tb", version=version, default_hp_metric=False))
        if enable_csv and _WANDB_OPTS.get("logger", "both") in ("both", "csv"):
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


def build_trainer_with_callbacks(
    devices: List[int],
    max_epochs: int,
    loggers,
    precision: str = "32",
    monitor_metric: Optional[str] = "valid_cohen_kappa",
    ckpt_dir: Optional[Path] = None,
    *,
    grad_clip: float = 0.0,
) -> Tuple[pl.Trainer, ModelCheckpoint]:
    if monitor_metric:
        ckpt_kwargs: Dict[str, Any] = dict(
            save_top_k=1,
            monitor=monitor_metric,
            mode="max",
            filename=f"best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
            save_last=True,
        )
    else:
        ckpt_kwargs = dict(save_top_k=0, save_last=True, filename="last")

    if ckpt_dir is not None:
        ckpt_kwargs["dirpath"] = str(ckpt_dir)

    ckpt_cb = ModelCheckpoint(**ckpt_kwargs)
    accelerator = "gpu" if (devices and torch.cuda.is_available()) else "cpu"

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices if devices else None,
        precision=precision,
        max_epochs=max_epochs,
        callbacks=[ckpt_cb],
        enable_checkpointing=True,
        logger=loggers,
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        gradient_clip_val=float(grad_clip) if float(grad_clip or 0.0) > 0 else 0.0,
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
        freeze_encoder=bool(getattr(args, "freeze_encoder", False)),
        ft_top_k=int(getattr(args, "ft_top_k", 0) or 0),
    )
    if scheme in {"stelar_patch_adapt", "stelar_patch_adapt_tuev"}:
        flags.update(
            dict(
                fe_use_raw_channel_map=bool(getattr(args, "use_raw_channel_map", False)),
                fe_raw_map_missing=str(getattr(args, "raw_map_missing", "error")),
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

    for key in ("load_pretrained", "freeze_encoder", "ft_top_k"):
        if key in flg:
            csv_bits[key] = flg[key]

    for key in (
        "fe_use_raw_channel_map",
        "fe_raw_map_missing",
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
    seed: int,
    args,
    test_metrics: dict,
    ckpt_cb: ModelCheckpoint,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    use_names = getattr(args, "use_channel_names", None)
    model_target_chans = (
        len(use_names)
        if use_names
        else int(getattr(args, "target_chans", 0) or 0)
        or int(getattr(args, "in_chans", 0) or 0)
    )
    montage_preview = (
        ",".join(use_names[:10]) + ("..." if use_names and len(use_names) > 10 else "")
        if use_names
        else ""
    )

    row: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": "TUEV",
        "seed": seed,
        "dataset_in_chans": args.in_chans,
        "model_target_chans": model_target_chans,
        "model_montage": montage_preview,
        "best_valid_cohen_kappa": (
            float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else float("nan")
        ),
        "test_balanced_accuracy": float(test_metrics.get("test_balanced_accuracy", float("nan"))),
        "test_accuracy": float(test_metrics.get("test_accuracy", test_metrics.get("test_acc", float("nan")))),
        "test_f1_macro": float(test_metrics.get("test_f1_macro", float("nan"))),
        "test_f1_weighted": float(test_metrics.get("test_f1_weighted", float("nan"))),
        "test_cohen_kappa": float(test_metrics.get("test_cohen_kappa", float("nan"))),
        "scheme": args.scheme,
        "tag": args.tag,
        "version_arg": args.version,
        "pretrain_version": getattr(args, "_pretrain_version", args.version),
        "frozen": int(bool(getattr(args, "freeze_encoder", False))),
        "ft_top_k": int(getattr(args, "ft_top_k", 0) or 0),
        "load_pretrained": int(bool(getattr(args, "load_pretrained", True))),
        "grad_clip": float(getattr(args, "grad_clip", 0.0)),
    }

    row.update(_encoder_csv_fields(args))
    if extra:
        row.update(extra)
    return row


# =============================================================================
# Front-end normalization (STELAR)
# =============================================================================

def _normalize_frontend_flags(args) -> None:
    if not hasattr(args, "use_chan_conv"):
        return

    args.use_raw_channel_map = bool(getattr(args, "use_raw_channel_map", False))
    args.use_chan_conv = bool(getattr(args, "use_chan_conv", True))
    args.use_time_chan_conv = bool(getattr(args, "use_time_chan_conv", False))

    n = int(args.use_raw_channel_map) + int(args.use_chan_conv) + int(args.use_time_chan_conv)
    if n <= 1:
        return

    print(
        "[WARN] Multiple STELAR front-ends were enabled. "
        "Priority: use_raw_channel_map > use_time_chan_conv > use_chan_conv."
    )
    if args.use_raw_channel_map:
        args.use_time_chan_conv = False
        args.use_chan_conv = False
    elif args.use_time_chan_conv:
        args.use_chan_conv = False


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
            use_raw_channel_map=bool(getattr(args, "use_raw_channel_map", False)),
            raw_map_missing=str(getattr(args, "raw_map_missing", "error")),
            use_chan_conv=bool(getattr(args, "use_chan_conv", True)),
            chan_max_norm=float(getattr(args, "chan_max_norm", 1.0)),
            load_pretrained=bool(getattr(args, "load_pretrained", True)),
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


# =============================================================================
# TUEV runners
# =============================================================================

def run_tuev(args) -> None:
    try:
        from tuev_dataset import LoadDataset
    except Exception:
        from downstream.datasets.tuev_dataset import LoadDataset

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)

    if getattr(args, "val_group", None) is None:
        if args.split == "eegpt":
            args.val_group = "subject_random"
        elif args.split == "cbramod":
            args.val_group = "subject_sorted"
        else:
            args.val_group = "class"

    class _Params:
        datasets_dir: str = args.tuev_path
        batch_size: int = args.batch_size
        num_workers: int = args.num_workers
        pin_memory: bool = True
        drop_last: bool = False
        shuffle_train: bool = True
        val_group: str = str(getattr(args, "val_group", "class"))
        subject_regex: str = str(getattr(args, "subject_regex", r"^(?P<sid>[^_]+)"))
        val_ratio: float = float(args.val_ratio)
        val_seed: int = int(args.seed)
        stratified_val: bool = bool(getattr(args, "stratified_val", True))
        reshape = (args.target_chans, None, None)
        scale_div: float = float(getattr(args, "scale_div", 1.0))
        transform = None

    loaders = LoadDataset(_Params()).get_data_loader()
    train_loader = loaders["train"]
    val_loader = loaders.get("val", None)
    test_loader = loaders["test"]

    steps = len(train_loader)
    model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=int(args.num_classes))

    base_dir = downstream_log_base(args, model_type)
    loggers = make_loggers(base_dir, version="run", enable_csv=bool(getattr(args, "log_csv", True)))
    _log_encoder_everywhere(base_dir, loggers, args)

    monitor = "valid_cohen_kappa" if val_loader is not None else None
    trainer, ckpt_cb = build_trainer_with_callbacks(
        args.devices,
        args.epochs,
        loggers,
        args.precision,
        monitor_metric=monitor,
        ckpt_dir=base_dir / "ckpts",
        grad_clip=float(getattr(args, "grad_clip", 0.0)),
    )

    resume_ckpt = _discover_resume_ckpt(base_dir / "ckpts") if bool(getattr(args, "resume", False)) else None
    print(f"[TUEV] resume={resume_ckpt or 'None'}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    best_path = _best_or_last_from_cb(ckpt_cb)
    test_metrics = trainer.test(
        model if best_path is None else None,
        ckpt_path=best_path,
        dataloaders=test_loader,
        verbose=False,
    )[0]

    row = make_result_row(
        seed=int(args.seed),
        args=args,
        test_metrics=test_metrics,
        ckpt_cb=ckpt_cb,
        extra={"best_ckpt": ckpt_cb.best_model_path or ""},
    )
    write_csv(str(base_dir / "csv" / "summary.csv"), [row])
    print(f"[TUEV] Done. Summary at: {base_dir / 'csv' / 'summary.csv'}")


def run_tuev_kfold(args):
    from collections import defaultdict, Counter

    try:
        from tuev_dataset import CustomDataset, _list_pickles, _read_labels_for_files
    except Exception:
        from downstream.datasets.tuev_dataset import (
            CustomDataset,
            _list_pickles,
            _read_labels_for_files,
        )

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)

    root = Path(args.tuev_path)
    train_dir = root / "train"
    eval_dir = root / "eval"
    if not train_dir.is_dir():
        if (root / "processed_train").is_dir():
            train_dir = root / "processed_train"
        else:
            raise FileNotFoundError(f"Missing train dir under {root}")
    if not eval_dir.is_dir():
        if (root / "processed_eval").is_dir():
            eval_dir = root / "processed_eval"
        elif (root / "processed_test").is_dir():
            eval_dir = root / "processed_test"
        else:
            raise FileNotFoundError(f"Missing eval/test dir under {root}")

    train_files = _list_pickles(train_dir)
    if len(train_files) == 0:
        raise RuntimeError(f"No .pkl files found in {train_dir}")
    labels = _read_labels_for_files(train_dir, train_files)

    subj_pat = re.compile(getattr(args, "subject_regex", r"^(?P<sid>[^_]+)"))

    def get_sid(p: Path) -> str:
        m = subj_pat.search(p.stem)
        if m and ("sid" in m.groupdict()):
            return str(m.group("sid"))
        if m:
            return m.group(0)
        try:
            rel = p.relative_to(train_dir)
            parts = rel.parts
            if len(parts) >= 2:
                return parts[0]
        except Exception:
            pass
        return p.stem.split("_", 1)[0].split("-", 1)[0]

    subj_to_indices: dict[str, List[int]] = defaultdict(list)
    for i, fp in enumerate(train_files):
        sid = get_sid(fp)
        subj_to_indices[sid].append(i)

    subjects = list(subj_to_indices.keys())
    print(f"[TUEV K-fold] Found {len(subjects)} subjects across {len(train_files)} windows.")

    subj_hist: dict[str, Counter] = {}
    for sid, idxs in subj_to_indices.items():
        c = Counter(labels[i] for i in idxs)
        subj_hist[sid] = c

    K = int(args.kfold)
    if K < 2:
        raise ValueError("--kfold must be >= 2 for K-fold CV")

    rng = np.random.RandomState(int(args.seed))
    subjects_sorted = sorted(subjects, key=lambda s: sum(subj_hist[s].values()), reverse=True)
    rng.shuffle(subjects_sorted)

    fold_subjs: List[set[str]] = [set() for _ in range(K)]
    fold_hist: List[Counter] = [Counter() for _ in range(K)]

    def fold_cost(fh: Counter) -> float:
        return float(sum(v * v for v in fh.values()))

    for sid in subjects_sorted:
        subj_c = subj_hist[sid]
        best_f, best_cost = None, None
        for f in range(K):
            tmp = fold_hist[f].copy()
            tmp.update(subj_c)
            c = fold_cost(tmp)
            if (best_cost is None) or (c < best_cost):
                best_cost, best_f = c, f
        fold_subjs[best_f].add(sid)
        fold_hist[best_f].update(subj_c)

    for f in range(K):
        n_files = sum(len(subj_to_indices[s]) for s in fold_subjs[f])
        print(
            f"[TUEV K-fold] Fold {f}: subjects={len(fold_subjs[f])}, "
            f"windows={n_files}, hist={dict(fold_hist[f])}"
        )

    test_set = CustomDataset(eval_dir, _list_pickles(eval_dir), scale_div=args.scale_div)

    def _make_loader(ds, shuffle):
        return torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=False,
            drop_last=False,
            collate_fn=ds.collate,
        )

    def _find_best_or_last_ckpt(ck_dir: Path) -> Optional[str]:
        if not ck_dir.is_dir():
            return None
        ckpts = sorted(ck_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not ckpts:
            return None
        for p in ckpts:
            name = p.name.lower()
            if "best" in name:
                return str(p)
        return str(ckpts[0])

    base_dir = downstream_log_base(args, model_type, extra=f"K{K}fold_SUBJ")
    fold_rows: List[Dict[str, Any]] = []
    metric_keys: Optional[List[str]] = None
    metric_accum: Dict[str, float] = {}
    n_eval = 0

    for f in range(K):
        val_subj = fold_subjs[f]
        train_subj = set().union(*[fold_subjs[j] for j in range(K) if j != f])

        val_indices = sorted(i for s in val_subj for i in subj_to_indices[s])
        train_indices = sorted(i for s in train_subj for i in subj_to_indices[s])

        tr_files = [train_files[i] for i in train_indices]
        va_files = [train_files[i] for i in val_indices]

        train_set = CustomDataset(train_dir, tr_files, scale_div=args.scale_div)
        val_set = CustomDataset(train_dir, va_files, scale_div=args.scale_div)
        train_loader = _make_loader(train_set, shuffle=True)
        val_loader = _make_loader(val_set, shuffle=False)
        test_loader = _make_loader(test_set, shuffle=False)

        steps = len(train_loader)

        fold_dir = base_dir / f"fold{f}"
        ck_dir = fold_dir / "ckpts"
        ck_dir.mkdir(parents=True, exist_ok=True)

        loggers = make_loggers(fold_dir, version="run", enable_csv=getattr(args, "log_csv", True))
        _log_encoder_everywhere(fold_dir, loggers, args)

        done_flag = fold_dir / ".done"
        skip_done = bool(getattr(args, "skip_completed_folds", True)) and done_flag.exists()

        existing_ckpt = _find_best_or_last_ckpt(ck_dir) if skip_done else None
        if skip_done and existing_ckpt is None:
            print(f"[TUEV][Fold {f}] .done exists but no ckpt found in {ck_dir} — will re-train.")

        monitor = "valid_cohen_kappa" if (val_loader is not None) else None
        trainer, ckpt_cb = build_trainer_with_callbacks(
            args.devices,
            args.epochs,
            loggers,
            args.precision,
            monitor_metric=monitor,
            ckpt_dir=ck_dir,
            grad_clip=float(getattr(args, "grad_clip", 0.0)),
        )

        model = build_engine(
            args,
            ckpt_path=ckpt_path,
            steps_per_epoch=steps,
            num_classes=getattr(args, "num_classes", 6),
        )

        if skip_done and existing_ckpt is not None:
            print(f"[TUEV][Fold {f}] .done exists — skipping TRAIN, running TEST from: {existing_ckpt}")
            test_metrics = trainer.test(
                model,
                ckpt_path=existing_ckpt,
                dataloaders=test_loader,
                verbose=False,
            )[0]
            best_path = existing_ckpt
        else:
            resume_ckpt = _discover_resume_ckpt(ck_dir) if getattr(args, "resume", False) else None
            print(
                f"[TUEV][Fold {f+1}/{K}] train_subj={len(train_subj)}  val_subj={len(val_subj)}  "
                f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}  resume={resume_ckpt or 'None'}"
            )
            trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

            best_path = _best_or_last_from_cb(ckpt_cb) or _find_best_or_last_ckpt(ck_dir)
            test_metrics = trainer.test(
                model,
                ckpt_path=best_path,
                dataloaders=test_loader,
                verbose=False,
            )[0]

            try:
                done_flag.touch()
            except Exception:
                pass

        row = make_result_row(
            seed=args.seed,
            args=args,
            test_metrics=test_metrics,
            ckpt_cb=ckpt_cb,
            extra={
                "kfold": K,
                "fold": f,
                "fold_train_subjects": len(train_subj),
                "fold_val_subjects": len(val_subj),
                "best_ckpt": (ckpt_cb.best_model_path or "") if (ckpt_cb is not None) else "",
                "tested_ckpt": best_path or "",
                "skipped_train": bool(skip_done and existing_ckpt is not None),
            },
        )
        fold_rows.append(row)

        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys:
            metric_accum[k] += float(test_metrics[k])
        n_eval += 1

    if metric_keys and n_eval > 0:
        mean_row = {"dataset": "TUEV", "kfold": K, "fold": "MEAN", "n_eval": n_eval}
        for k in metric_keys:
            mean_row[k] = metric_accum[k] / float(n_eval)
        fold_rows.append(mean_row)

    out_csv = str((base_dir / "csv" / "summary.csv"))
    write_csv(out_csv, fold_rows)
    print(f"[TUEV] Subject-wise K-fold finished. Summary at: {out_csv}")


# =============================================================================
# CLI (TUEV-only)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("TUEV downstream evaluation runner")

    p.add_argument("--tuev_path", type=str, required=True, help="Path to processed TUEV folder.")
    p.add_argument(
        "--scheme",
        type=str,
        required=True,
        help="Module key in registry (e.g., eegpt_tuev, stelar_patch_adapt_tuev, cbramod_tuev, labram).",
    )
    p.add_argument("--split", type=str, default="default", choices=["default", "eegpt", "cbramod"])

    # K-fold controls
    p.add_argument("--kfold", type=int, default=1, help="If >1: subject-wise K-fold on TUEV train/ with fixed eval/.")

    # Val construction controls (for LoadDataset)
    p.add_argument("--val_ratio", type=float, default=0.0)
    p.add_argument("--val_group", type=str, default=None, help="Override: 'class', 'subject_random', 'subject_sorted'.")
    p.add_argument("--subject_regex", type=str, default=r"^(?P<sid>[^_]+)", help="Extract subject id from filename stem.")
    _add_bool_flag(p, "stratified_val", True, "Use stratified split when building val from train.")

    # Basic training args
    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=CFG.MAX_EPOCHS)

    # SINGLE SEED
    p.add_argument("--seed", type=int, default=CFG.SEED)

    p.add_argument("--num_classes", type=int, default=6)

    # Optim & schedule
    p.add_argument("--max_lr_head", type=float, default=CFG.MAX_LR_HEAD)
    p.add_argument("--max_lr_enc", type=float, default=CFG.MAX_LR_ENC)
    p.add_argument("--wd_head", type=float, default=CFG.WD_HEAD)
    p.add_argument("--wd_enc", type=float, default=CFG.WD_ENC)
    p.add_argument("--scheduler", type=str, default=CFG.SCHEDULER, choices=["onecycle", "cosine", "none"])
    p.add_argument("--warmup_pct", type=float, default=CFG.WARMUP_PCT)

    # Gradient clipping
    p.add_argument("--grad_clip", type=float, default=1.0, help="Global grad clipping value. 0 disables.")

    # Train policy
    p.add_argument("--freeze_encoder", action="store_true", default=False)
    p.add_argument("--ft_top_k", type=int, default=0)

    # Pretrain ckpt selection / toggle
    p.add_argument("--epoch_select", type=str, default=getattr(CFG, "epoch_select", "last"), choices=["best", "last"])
    p.add_argument("--version", type=int, default=getattr(CFG, "version", 0))
    p.add_argument("--pretrain_max_lr", type=float, default=getattr(CFG, "pretrain_max_lr", 5e-4))
    p.add_argument("--pretrain_dir", type=str, default=None, help="Exact .../_tb/version_X directory.")
    p.add_argument("--pretrain_ckpt", type=str, default=None, help="Exact .ckpt/.pth path (LaBraM requires .pth).")
    _add_bool_flag(p, "load_pretrained", True, "Load encoder weights; set --no-load_pretrained for random init.")

    # Model fingerprint
    p.add_argument("--tag", type=str, default=CFG.tag)
    p.add_argument("--variant", type=str, default=CFG.variant)
    p.add_argument("--preprocess", type=str, default=CFG.preprocess)
    p.add_argument("--encoder", type=str, default=CFG.encoder)
    p.add_argument("--transformer", type=str, default=CFG.transformer)

    # RoPE / ACPE / channel-embedding toggles
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

    p.add_argument("--in_chans", type=int, default=getattr(CFG, "IN_CHANS", 22))
    p.add_argument("--target_chans", type=int, default=getattr(CFG, "TARGET_CHANS", 19))
    p.add_argument("--seq_len", type=int, default=getattr(CFG, "SEQ_LEN", 1024))
    p.add_argument("--patch", type=int, default=getattr(CFG, "PATCH", 64))
    p.add_argument("--dropout", type=float, default=getattr(CFG, "DROPOUT", 0.5))
    p.add_argument("--head_type", type=str, default=getattr(CFG, "HEAD_TYPE", "avg"),
                   choices=["avg", "per_time_attn", "flatten", "eegpt_probe", "all_patch_reps"])
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
    _add_bool_flag(p, "use_raw_channel_map", False, "Directly select/reorder raw channels by name before encoder.")
    p.add_argument("--raw_map_missing", type=str, default="error", choices=["error", "zero"])
    _add_bool_flag(p, "use_chan_conv", True, "1x1 channel-mixing Conv1d before encoder (STELAR).")
    _add_bool_flag(p, "use_time_chan_conv", False, "EEGNet-style temporal depthwise Conv2d (STELAR TUEV).")
    p.add_argument("--chan_max_norm", type=float, default=1.0)
    p.add_argument("--conv_kernel_time", type=int, default=55)
    p.add_argument("--conv_dropout", type=float, default=0.8)

    # Devices / workers / precision / resume
    p.add_argument("--devices", default=",".join(map(str, getattr(CFG, "devices", []))))
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--precision", type=str, default="32")
    _add_bool_flag(p, "resume", False, "Resume training from last.ckpt when available.")
    _add_bool_flag(p, "skip_completed_folds", True, "Skip folds that already have a .done sentinel.")
    p.add_argument("--scale_div", type=float, default=1.0)

    # Logging
    p.add_argument("--log_root", type=str, default="./logs")
    _add_bool_flag(p, "log_csv", True, "Also write CSV metrics alongside TensorBoard.")

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

    # Devices & workers
    args.devices = parse_devices(args.devices)
    if args.num_workers is None:
        args.num_workers = default_num_workers(args.devices)

    # Parse dataset_channel_names
    if isinstance(args.dataset_channel_names, str) and args.dataset_channel_names:
        s = args.dataset_channel_names.strip()
        try:
            if s.startswith("["):
                lst = json.loads(s)
            else:
                lst = [t.strip() for t in s.split(",") if t.strip()]
            if lst:
                args.dataset_channel_names = lst
                args.in_chans = len(lst)
        except Exception:
            pass

    # Montage/channel logic
    scheme = args.scheme.lower()
    if scheme == "labram":
        if getattr(args, "dataset_channel_set", None):
            args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
            if args.dataset_channel_names:
                args.in_chans = len(args.dataset_channel_names)
        args.use_channel_names = None
        args.target_chans = int(args.in_chans)
    elif scheme in {"cbramod", "cbramod_tuev"}:
        args.use_channel_names = None
    else:
        if getattr(args, "dataset_channel_set", None):
            args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
            if args.dataset_channel_names:
                args.in_chans = len(args.dataset_channel_names)

        if getattr(args, "model_channel_set", None):
            args.use_channel_names = MONTAGE_BY_DATASET.get(args.model_channel_set) or list(DEFAULT_EEGPT58)
        else:
            args.use_channel_names = MONTAGE_BY_DATASET.get("TUEV", list(DEFAULT_EEGPT58))
        args.target_chans = len(args.use_channel_names)

    _normalize_frontend_flags(args)
    return args


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    pl.seed_everything(int(args.seed), workers=True)

    if int(getattr(args, "kfold", 1)) > 1:
        run_tuev_kfold(args)
    else:
        run_tuev(args)


if __name__ == "__main__":
    main()