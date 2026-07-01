#!/usr/bin/env python3
# FACED-only downstream evaluation runner (keeps encoder-config logging, NO auto-ckpt discovery)
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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

# Optional YAML config + W&B logger layer (additive; off by default).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stelar_utils import add_logging_args, apply_yaml_defaults  # noqa: E402
from stelar_utils.loggers import make_wandb_logger  # noqa: E402

# Populated from CLI/YAML in parse_args(); consumed by make_loggers().
_WANDB_OPTS: Dict[str, Any] = {}

# --- optional: CBraMod Lightning module ---
try:
    from modules.cbramod_ft import LitCBraMod  # noqa: E402
except Exception:
    LitCBraMod = None  # type: ignore

# --- optional: LaBraM Lightning module ---
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


def _save_encoder_config_json(base_dir: Path, payload: Dict[str, Any]) -> None:
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "encoder_config.json"
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print("[CONFIG] wrote:", str(path))


def _log_encoder_hparams_to_tb(loggers, hparams: Dict[str, Any]) -> None:
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
    """Find a reasonable ckpt to resume from (last or newest)."""
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


def downstream_log_base(dataset: str, args, model_type: str, extra: str = "") -> Path:
    ver = getattr(args, "_pretrain_version", args.version)
    head_name = args.head_type
    family = _family_from_scheme(args.scheme)

    # Encoder fingerprint tag (only for STELAR-family)
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
        parts = [run_tag, dataset, enc_tag, "randinit", "frozen" if args.freeze_encoder else None, extra or None]
        return Path(args.log_root).joinpath(*[p for p in parts if p])

    topk_tag = None
    if not getattr(args, "freeze_encoder", False):
        k = int(getattr(args, "ft_top_k", 0) or 0)
        if k > 0:
            topk_tag = f"topk{k}"

    parts = [
        f"{args.epoch_select}_v{ver}_{args.pretrain_max_lr}_{model_type}_{head_name}",
        dataset,
        f"seed_{args.seed}",
        enc_tag,
        "pretrained",
        "frozen" if args.freeze_encoder else None,
        topk_tag,
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
    monitor_metric: Optional[str] = "valid_accuracy",
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
        strategy="auto",
        enable_progress_bar=True,
        gradient_clip_val=float(grad_clip) if (grad_clip and grad_clip > 0) else 0.0,
        gradient_clip_algorithm="norm",
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
                fe_use_chan_conv=bool(getattr(args, "use_chan_conv", True)),
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

    for key in ("fe_use_chan_conv", "fe_chan_max_norm", "fe_use_time_chan_conv", "fe_conv_kernel_time", "fe_conv_dropout"):
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
    )
    _save_encoder_config_json(base_dir, payload)

    tb_hparams = {}
    tb_hparams.update(_flatten("enc.", enc))
    tb_hparams.update(_flatten("flag.", flg))
    _log_encoder_hparams_to_tb(loggers, tb_hparams)


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
        "dataset": dataset,
        "seed": seed,
        "dataset_in_chans": args.in_chans,
        "model_target_chans": model_target_chans,
        "model_montage": montage_preview,
        "best_valid": float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else float("nan"),
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
        "init_scheme": getattr(args, "init_scheme", ""),
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

    args.use_chan_conv = bool(getattr(args, "use_chan_conv", True))
    args.use_time_chan_conv = bool(getattr(args, "use_time_chan_conv", False))

    if args.use_chan_conv and args.use_time_chan_conv:
        print(
            "[WARN] Both --use_chan_conv and --use_time_chan_conv were set; "
            "disabling use_chan_conv and keeping use_time_chan_conv=True."
        )
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

        # Override encoder flags according to CLI
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
            models_cfg["encoder"].update(dict(use_criss_cross=False))

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
                use_chan_conv=bool(getattr(args, "use_chan_conv", False)),
                use_time_chan_conv=bool(getattr(args, "use_time_chan_conv", True)),
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
        num_classes=int(num_classes),
        steps_per_epoch=int(steps_per_epoch),
        max_epochs=int(args.epochs),
        lr_head=float(args.max_lr_head),
        lr_enc=float(args.max_lr_enc),
        wd_head=float(args.wd_head),
        wd_enc=float(args.wd_enc),
        scheduler=str(args.scheduler),
        warmup_pct=float(args.warmup_pct),
        in_chans=int(args.in_chans),
        target_chans=int(args.target_chans),
        seq_len=int(args.seq_len),
        patch=int(args.patch),
        head_type=str(args.head_type),
        dropout=float(args.dropout),
        channel_names=args.dataset_channel_names,
        use_channel_names=args.use_channel_names,
        freeze_encoder=bool(args.freeze_encoder),
        ft_top_k=int(getattr(args, "ft_top_k", 0) or 0),
        **extra_kwargs,
    )


# =============================================================================
# FACED runner (4-fold)
# =============================================================================

def run_faced_4fold(args):
    import math
    from pathlib import Path
    from typing import Any, Dict, List, Optional

    try:
        from downstream.datasets.faced_dataset import LoadDataset
    except ImportError:
        from faced_dataset import LoadDataset

    def _best_ckpt_in_dir(ckpt_dir: Path) -> Optional[str]:
        """
        Prefer a 'best*.ckpt' (newest), else 'last.ckpt', else newest *.ckpt.
        Return None if nothing found.
        """
        if not ckpt_dir.exists():
            return None

        ckpts = sorted(ckpt_dir.glob("*.ckpt"))
        if not ckpts:
            return None

        best = [p for p in ckpts if "best" in p.name.lower()]
        if best:
            best.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(best[0])

        last = ckpt_dir / "last.ckpt"
        if last.is_file():
            return str(last)

        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(ckpts[0])

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base("FACED", args, model_type, extra="4fold")
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "csv").mkdir(parents=True, exist_ok=True)

    folds = list(range(4))
    if int(getattr(args, "fold", -1)) != -1:
        folds = [int(args.fold)]
        print(f"[FACED] Running specific fold: {folds[0]}")
    else:
        print("[FACED] Running all 4 folds sequentially.")

    rows: List[Dict[str, Any]] = []
    metric_keys: Optional[List[str]] = None
    metric_accum: Dict[str, float] = {}
    n_eval = 0  # number of folds actually evaluated in THIS run

    for f_idx in folds:
        print(f"\n[FACED] === 4-fold {f_idx+1}/4 ===")

        class _Params:
            datasets_dir = args.faced_path
            batch_size = args.batch_size
            num_workers = args.num_workers
            pin_memory = True
            drop_last = False
            shuffle_train = True
            scale_div = args.scale_div
            fold_idx = f_idx

        loaders = LoadDataset(_Params()).get_data_loader()
        train_loader = loaders["train"]
        val_loader = loaders.get("val", None)
        test_loader = loaders["test"]

        steps = math.ceil(len(train_loader))
        model = build_engine(
            args,
            ckpt_path=ckpt_path,
            steps_per_epoch=steps,
            num_classes=int(getattr(args, "num_classes", 9)),
        )

        fold_dir = base_dir / f"fold{f_idx}"
        ckpt_dir = fold_dir / "ckpts"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        loggers = make_loggers(
            fold_dir,
            version="run",
            enable_csv=bool(getattr(args, "log_csv", True)),
        )
        _log_encoder_everywhere(fold_dir, loggers, args)

        trainer, ckpt_cb = build_trainer_with_callbacks(
            args.devices,
            args.epochs,
            loggers,
            args.precision,
            monitor_metric="valid_accuracy" if val_loader is not None else None,
            ckpt_dir=ckpt_dir,
            grad_clip=float(getattr(args, "grad_clip", 0.0)),
        )

        done_flag = fold_dir / ".done"
        skip_completed = bool(getattr(args, "skip_completed_folds", True))

        # DONE means: do NOT train; just evaluate best ckpt and include in NEW summary
        if skip_completed and done_flag.exists():
            best_path = _best_ckpt_in_dir(ckpt_dir)
            if best_path is None:
                print(f"[FACED][Fold {f_idx}] .done exists but no ckpt found in {ckpt_dir} — skipping fold.")
                continue

            print(f"[FACED][Fold {f_idx}] DONE: evaluating ckpt={best_path}")
            test_metrics = trainer.test(
                model,  # Lightning loads from ckpt_path
                ckpt_path=best_path,
                dataloaders=test_loader,
                verbose=False,
            )[0]

            row = make_result_row(
                dataset="FACED",
                seed=int(args.seed),
                args=args,
                test_metrics=test_metrics,
                ckpt_cb=ckpt_cb,
                extra={
                    "strategy": "4fold",
                    "fold_number": f_idx,
                    "best_ckpt": best_path,
                },
            )
            rows.append(row)

        else:
            # NOT done: resume if possible, else start fresh, then evaluate best and mark done
            resume_ckpt = None
            if bool(getattr(args, "resume", True)):
                # your previous behavior: prefer last.ckpt else newest
                last_ckpt = ckpt_dir / "last.ckpt"
                if last_ckpt.is_file():
                    resume_ckpt = str(last_ckpt)
                else:
                    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if ckpts:
                        resume_ckpt = str(ckpts[0])

            print(f"[FACED][Fold {f_idx}] NOT done: training (resume={resume_ckpt or 'None'})")
            trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

            # evaluate best/last
            best_path = ckpt_cb.best_model_path or _best_ckpt_in_dir(ckpt_dir)
            test_metrics = trainer.test(
                model if best_path is None else None,
                ckpt_path=best_path,
                dataloaders=test_loader,
                verbose=False,
            )[0]

            row = make_result_row(
                dataset="FACED",
                seed=int(args.seed),
                args=args,
                test_metrics=test_metrics,
                ckpt_cb=ckpt_cb,
                extra={
                    "strategy": "4fold",
                    "fold_number": f_idx,
                    "best_ckpt": best_path or "",
                },
            )
            rows.append(row)

            # mark done AFTER successful eval
            try:
                done_flag.touch()
            except Exception:
                pass

        # accumulate mean over folds actually evaluated in THIS run
        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys:
            metric_accum[k] += float(test_metrics.get(k, 0.0))
        n_eval += 1

    if metric_keys and n_eval > 1:
        avg_row: Dict[str, Any] = {"dataset": "FACED", "strategy": "4fold", "fold_number": "MEAN"}
        for k in metric_keys:
            avg_row[f"test/{k}"] = metric_accum[k] / float(n_eval)
        rows.append(avg_row)

    out_csv = str(base_dir / "csv" / "summary.csv")
    write_csv(out_csv, rows)
    print(f"\n[FACED] 4-fold finished. Summary at: {out_csv}")



# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("FACED downstream evaluation runner (4-fold)")

    # FACED only
    p.add_argument("--faced_path", type=str, required=True, help="Path to FACED processed dataset root.")
    p.add_argument("--fold", type=int, default=-1, help="Fold index in {0,1,2,3}. -1 runs all folds.")
    p.add_argument("--num_classes", type=int, default=9, help="FACED num classes (default: 9).")
    p.add_argument("--monitor_metric", type=str, default="valid_accuracy", help="Metric to monitor for best ckpt.")

    # Scheme
    p.add_argument(
        "--scheme",
        type=str,
        required=True,
        help="Module key in modules registry (e.g., eegpt, stelar_patch_adapt, stelar_patch_adapt_tuev, cbramod, cbramod_tuev, labram)",
    )

    # Generic overrides
    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=CFG.MAX_EPOCHS)
    p.add_argument("--seed", type=int, default=CFG.SEED)

    # Optim & schedule
    p.add_argument("--max_lr_head", type=float, default=CFG.MAX_LR_HEAD)
    p.add_argument("--max_lr_enc", type=float, default=CFG.MAX_LR_ENC)
    p.add_argument("--wd_head", type=float, default=CFG.WD_HEAD)
    p.add_argument("--wd_enc", type=float, default=CFG.WD_ENC)
    p.add_argument("--scheduler", type=str, default=CFG.SCHEDULER, choices=["onecycle", "cosine", "none"])
    p.add_argument("--warmup_pct", type=float, default=CFG.WARMUP_PCT)

    # Train policy
    p.add_argument("--freeze_encoder", action="store_true", default=False)
    p.add_argument("--ft_top_k", type=int, default=0)

    # Dataset toggles
    p.add_argument("--scale_div", type=float, default=1.0)

    # Pretrain ckpt selection / toggle (NO discovery)
    p.add_argument("--epoch_select", type=str, default=getattr(CFG, "epoch_select", "last"), choices=["best", "last"])
    p.add_argument("--version", type=int, default=getattr(CFG, "version", 0))
    p.add_argument("--pretrain_max_lr", type=float, default=getattr(CFG, "pretrain_max_lr", 5e-4))
    p.add_argument("--pretrain_dir", type=str, default=None, help="Exact .../_tb/version_X directory (used directly; no discovery).")
    p.add_argument("--pretrain_ckpt", type=str, default=None, help="Exact .ckpt/.pth path (LaBraM requires .pth).")
    _add_bool_flag(p, "load_pretrained", True, "Load encoder weights; set --no-load_pretrained for random init.")
    p.add_argument("--init_scheme", type=str, default="xavier_uniform", choices=["xavier_uniform", "xavier_normal", "kaiming_normal"])

    # Model fingerprint
    p.add_argument("--tag", type=str, default=CFG.tag)
    p.add_argument("--variant", type=str, default=CFG.variant)
    p.add_argument("--preprocess", type=str, default=CFG.preprocess)
    p.add_argument("--encoder", type=str, default=CFG.encoder)
    p.add_argument("--transformer", type=str, default=CFG.transformer)
    p.add_argument("--loss_switch", type=str, default=getattr(CFG, "loss_switch", "default"))

    # RoPE / ACPE / channel-embedding toggles (STELAR-style)
    _add_bool_flag(p, "use_dynamic_pe", CFG.USE_DYNAMIC_PE, "Enable dynamic positional encoding (ACPE).")
    _add_bool_flag(p, "rope_time", CFG.ROPE_TIME_DEFAULT, "Enable RoPE along time axis.")
    _add_bool_flag(p, "rope_chan", CFG.ROPE_CHAN_DEFAULT, "Enable RoPE along channel axis.")
    p.add_argument("--rope_dim_frac", type=float, default=CFG.ROPE_DIM_FRAC_DEFAULT)
    p.add_argument("--rope_interp_time", type=float, default=CFG.ROPE_INTERP_TIME_DEFAULT)
    p.add_argument("--rope_interp_chan", type=float, default=CFG.ROPE_INTERP_CHAN_DEFAULT)
    _add_bool_flag(p, "use_channel_embed", True, "Use channel embedding with channel IDs inside the encoder.")

    # Frontend / head options
    p.add_argument("--dataset_channel_names", type=str, default=None, help="Comma-separated list or JSON list of channel names.")
    p.add_argument("--dataset_channel_set", type=str, default=None, choices=sorted(CHANNELS_BY_DATASET.keys()))
    p.add_argument("--model_channel_set", type=str, default=None, choices=sorted(MONTAGE_BY_DATASET.keys()))

    p.add_argument("--in_chans", type=int, default=getattr(CFG, "IN_CHANS", 22))
    p.add_argument("--target_chans", type=int, default=getattr(CFG, "TARGET_CHANS", 19))
    p.add_argument("--seq_len", type=int, default=getattr(CFG, "SEQ_LEN", 1024))
    p.add_argument("--patch", type=int, default=getattr(CFG, "PATCH", 64))
    p.add_argument("--dropout", type=float, default=getattr(CFG, "DROPOUT", 0.5))
    p.add_argument(
        "--head_type",
        type=str,
        default=getattr(CFG, "HEAD_TYPE", "avg"),
        choices=["avg", "per_time_attn", "flatten", "eegpt_probe", "all_patch_reps"],
    )
    p.add_argument("--pool", type=str, default=getattr(CFG, "POOL", "mean"), choices=["mean", "max", "meanmax"])
    p.add_argument("--attn_temperature", type=float, default=None)
    p.add_argument("--attn_project_kv", dest="attn_project_kv", action="store_true")
    p.add_argument("--no-attn_project_kv", dest="attn_project_kv", action="store_false")
    p.set_defaults(attn_project_kv=True)

    # ---- LaBraM-specific ----
    p.add_argument("--labram_model", type=str, default="labram_base_patch200_200")
    p.add_argument("--labram_handle_remainder", type=str, default="crop", choices=["crop", "pad"])
    p.add_argument("--labram_pad_value", type=float, default=0.0)

    # ---- STELAR front-end toggles ----
    _add_bool_flag(p, "use_chan_conv", True, "1x1 channel-mixing Conv1d before encoder (STELAR).")
    _add_bool_flag(p, "use_time_chan_conv", False, "EEGNet-style temporal depthwise Conv2d before encoder (STELAR TUEV).")
    p.add_argument("--chan_max_norm", type=float, default=1.0)
    p.add_argument("--conv_kernel_time", type=int, default=55)
    p.add_argument("--conv_dropout", type=float, default=0.8)

    # ---- Devices / workers / precision / resume ----
    p.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clip value (global norm). 0 disables.")
    p.add_argument("--devices", default=",".join(map(str, getattr(CFG, "devices", []))))
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--precision", type=str, default="32")
    _add_bool_flag(p, "resume", False, "Resume training from last.ckpt when available.")
    _add_bool_flag(p, "skip_completed_folds", True, "Skip folds that already have a .done sentinel.")

    # ---- Logging ----
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

    # Montage/channel logic (kept generic)
    scheme = args.scheme.lower()
    if scheme in {"cbramod", "cbramod_tuev"}:
        pass
    elif scheme == "labram":
        if getattr(args, "dataset_channel_set", None):
            args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
            if args.dataset_channel_names:
                args.in_chans = len(args.dataset_channel_names)
        args.use_channel_names = None
        args.target_chans = int(args.in_chans)
    else:
        if getattr(args, "dataset_channel_set", None):
            args.dataset_channel_names = CHANNELS_BY_DATASET.get(args.dataset_channel_set)
            if args.dataset_channel_names:
                args.in_chans = len(args.dataset_channel_names)
        elif getattr(args, "dataset_channel_names", None):
            args.in_chans = len(args.dataset_channel_names)

        if getattr(args, "model_channel_set", None):
            args.use_channel_names = MONTAGE_BY_DATASET.get(args.model_channel_set) or list(DEFAULT_EEGPT58)
        else:
            args.use_channel_names = list(DEFAULT_EEGPT58)
        args.target_chans = len(args.use_channel_names)

    # Model config per scheme: stash embed info for flatten heads if needed
    if scheme in {"eegpt", "eegpt_tuev"}:
        _models_cfg = CFG.get_config_eegpt(**CFG.EEGPT_MODELS_CONFIGS[args.tag])
        args._enc_embed_dim = int(_models_cfg["encoder"]["embed_dim"])
        args._enc_embed_slots = int(_models_cfg["encoder"].get("embed_num", 1))
    elif scheme in {"stelar_patch_adapt", "stelar_patch_adapt_tuev"}:
        _models_cfg = CFG.get_config(args.tag)
        args._enc_embed_dim = int(_models_cfg["encoder"]["embed_dim"])
        args._enc_embed_slots = int(_models_cfg["encoder"].get("embed_num", 1))

    # Normalize front-end flags once
    _normalize_frontend_flags(args)

    return args


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    pl.seed_everything(int(args.seed), workers=True)
    run_faced_4fold(args)


if __name__ == "__main__":
    main()
