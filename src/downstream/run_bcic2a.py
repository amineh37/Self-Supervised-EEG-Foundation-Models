#!/usr/bin/env python3
"""
BCIC2a downstream runner (BCIC2a-only)

Seed policy: a SINGLE seed (--seed) is used everywhere (PL/Torch/NumPy + dataset splits).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
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
# Split parsing (fixed split)
# =============================================================================

def _parse_id_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        if "-" in t:
            a, b = t.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(t))
    seen: set[int] = set()
    final: List[int] = []
    for v in out:
        if v not in seen:
            final.append(v)
            seen.add(v)
    return final


def _parse_split_spec(spec: Optional[str]) -> Dict[str, Optional[List[int]]]:
    """
    Fixed-split spec parser. Supports env var EEGPT_SPLIT_SPEC.
    No seed here; seed comes ONLY from args.seed.
    """
    spec = (spec or os.getenv("EEGPT_SPLIT_SPEC", "")).strip()
    if not spec:
        return {"train": None, "val": None, "test": None}
    parts = [p.strip() for p in spec.split(";") if p.strip()]
    out: Dict[str, Optional[List[int]]] = {"train": None, "val": None, "test": None}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        key = k.strip().lower()
        if key in out:
            out[key] = _parse_id_list(v.strip())
    return out


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
            raise ValueError(
                "For --scheme labram with pretrained, pass --pretrain_ckpt pointing to a LaBraM .pth file."
            )
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
        ckpt_path, _ = get_checkpoint_path(
            str(ver_dir),
            which="best" if args.epoch_select == "best" else "last",
        )

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
# Logging helpers (encoder config)
# =============================================================================

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
    if scheme in {"stelar_patch_adapt", "stelar_dann", "stelar_patch_adapt_tuev", "stelar_summary_patch_adapt"}:
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
    out: Dict[str, Any] = {}

    for key in ("embed_dim", "embed_num", "depth", "num_heads", "use_criss_cross", "use_channel_embed", "use_dynamic_pe"):
        if key in enc:
            out[f"enc_{key}"] = enc[key]

    for key in ("rope_time", "rope_chan", "rope_dim_frac", "rope_interp_time", "rope_interp_chan"):
        if key in flg:
            out[f"enc_{key}"] = flg[key]

    for key in ("load_pretrained", "init_scheme", "freeze_encoder", "ft_top_k"):
        if key in flg:
            out[key] = flg[key]

    for key in ("fe_use_chan_conv", "fe_chan_max_norm", "fe_use_time_chan_conv", "fe_conv_kernel_time", "fe_conv_dropout"):
        if key in flg:
            out[key] = flg[key]

    return out


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
        # SINGLE seed recorded:
        seed=int(getattr(args, "seed", 0)),
    )
    _save_json(Path(base_dir) / "encoder_config.json", payload)

    tb_hparams = {}
    tb_hparams.update(_flatten("enc.", enc))
    tb_hparams.update(_flatten("flag.", flg))
    _log_hparams_to_tb(loggers, tb_hparams)


def make_loggers(base_dir: Path, version: str, *, enable_csv: bool = True):
    base_dir = Path(base_dir)
    loggers: list = []
    if _WANDB_OPTS.get("logger", "both") != "wandb":
        loggers.append(
            pl_loggers.TensorBoardLogger(save_dir=str(base_dir), name="tb", version=version, default_hp_metric=False)
        )
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
        strategy="auto",
        enable_progress_bar=True,
        gradient_clip_val=float(grad_clip) if (grad_clip and grad_clip > 0) else 0.0,
    )
    return trainer, ckpt_cb


def downstream_log_base(args, model_type: str, extra: str = "") -> Path:
    dataset = "BCIC2a"
    ver = getattr(args, "_pretrain_version", args.version)
    head_name = args.head_type
    use_val = float(getattr(args, "val_ratio", 0.0) or 0.0) > 0.0

    scheme = (args.scheme or "").lower()
    family = _family_from_scheme(args.scheme)

    enc_tag: Optional[str] = None
    if scheme.startswith("stelar"):
        pe_bits: List[str] = []
        pe_bits.append("chid" if bool(getattr(args, "use_channel_embed", True)) else "nochid")
        if bool(getattr(args, "use_dynamic_pe", False)):
            pe_bits.append("acpe")
        if bool(getattr(args, "rope_time", False)):
            pe_bits.append("ropeT")
        if bool(getattr(args, "rope_chan", False)):
            pe_bits.append("ropeC")
        enc_tag = "_".join(pe_bits) if pe_bits else None

    if not getattr(args, "load_pretrained", True):
        run_tag = f"{family}_{args.tag}_{args.variant}_{args.preprocess}_{args.encoder}_{args.transformer}_randinit"
        parts = [
            run_tag, dataset, enc_tag, "randinit",
            "frozen" if getattr(args, "freeze_encoder", False) else None,
            "val" if use_val else "noval",
            extra or None,
        ]
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
        "frozen" if getattr(args, "freeze_encoder", False) else None,
        topk_tag,
        "val" if use_val else "noval",
        extra or None,
    ]
    return Path(args.log_root).joinpath(*[p for p in parts if p])


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
        "dataset": "BCIC2a",
        "seed": seed,
        "dataset_in_chans": args.in_chans,
        "model_target_chans": model_target_chans,
        "model_montage": montage_preview,
        "best_valid_cohen_kappa": float(ckpt_cb.best_model_score.item()) if ckpt_cb.best_model_score else float("nan"),
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
# Engine builder
# =============================================================================

def build_engine(args, *, ckpt_path: Optional[str], steps_per_epoch: int, num_classes: int, num_subjects: int = 0):
    scheme_lower = args.scheme.lower()

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

    if scheme_lower.startswith("cbramod"):
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

    extra_kwargs: Dict[str, Any] = {}

    if scheme_lower.startswith("eegpt"):
        ModuleClass = build_module(args.scheme)
        models_cfg = CFG.get_config_eegpt(**CFG.EEGPT_MODELS_CONFIGS[args.tag])

    elif scheme_lower in {"stelar_patch_adapt", "stelar_dann", "stelar_patch_adapt_tuev", "stelar_summary_patch_adapt"}:
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
        if scheme_lower == "stelar_dann":
            extra_kwargs.update(
                num_subjects=max(int(num_subjects), 1),
                lambda_adv=float(getattr(args, "lambda_adv", 1.0)),
                grl_alpha=float(getattr(args, "grl_alpha", 1.0)),
                schedule_alpha=bool(getattr(args, "schedule_alpha", True)),
                grl_gamma=float(getattr(args, "grl_gamma", 10.0)),
                subj_disc_hidden=int(getattr(args, "subj_disc_hidden", 64)),
                subj_disc_dropout=float(getattr(args, "subj_disc_dropout", 0.3)),
            )
    else:
        raise NotImplementedError(f"Unknown or unsupported scheme: {args.scheme}")

    _stash_encoder_logging_payload(args, models_cfg, args.scheme)

    print(f"[INFO] in_chans={args.in_chans}, target_chans={args.target_chans}, seq_len={args.seq_len}, patch={args.patch}")
    print(f"INFO]---------------------------------------------------------------")
    print(f"MODEL CONFIG")
    print(models_cfg)
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
# BCIC2a dataset helpers + runners
# =============================================================================

def _bcic2a_discover_subjects(db_path: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """Reads LMDB __keys__ and infers subject tokens from key prefixes."""
    import lmdb

    env = lmdb.open(db_path, readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin(write=False) as txn:
            raw = pickle.loads(txn.get(b"__keys__"))
    finally:
        env.close()

    if isinstance(raw, dict):
        keys, seen = [], set()
        for m in ("train", "val", "test"):
            for k in raw.get(m, []):
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
        for m, lst in raw.items():
            if m in {"train", "val", "test"}:
                continue
            for k in lst:
                if k not in seen:
                    keys.append(k)
                    seen.add(k)
    else:
        keys = list(raw)

    def _tok(key: str) -> str:
        return key.split("-", 1)[0][:3]

    toks: List[str] = []
    by: Dict[str, List[str]] = {}
    seen = set()
    for k in keys:
        s = _tok(k)
        if s not in seen:
            toks.append(s)
            seen.add(s)
        by.setdefault(s, []).append(k)

    print("[BCIC2a] subject index mapping:")
    for i, t in enumerate(toks):
        print(f"  {i}: {t}")
    return toks, by


def run_bcic2a_loso(args) -> None:
    try:
        from bcic2a_dataset import LoadDataset
    except Exception:
        from downstream.datasets.bcic2a_dataset import LoadDataset

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base(args, model_type, extra="LOSO")

    subj_tokens, _ = _bcic2a_discover_subjects(args.bcic2a_path)
    n_subj = len(subj_tokens)

    rows: List[Dict[str, Any]] = []
    metric_keys: Optional[List[str]] = None
    metric_accum: Dict[str, float] = {}

    for sid, tok in enumerate(subj_tokens):
        print(f"\n[BCIC2a] === LOSO fold {sid}/{n_subj-1} — test subject: {tok} ===")

        class _Params:
            datasets_dir: str = args.bcic2a_path
            batch_size: int = args.batch_size
            split_mode: str = "mixed"
            test_subject_ids = [sid]
            val_ratio: float = float(args.val_ratio)
            stratified_val: bool = bool(args.stratified_val)
            # SINGLE seed everywhere:
            seed: int = int(args.seed)
            num_workers: int = int(args.num_workers)
            pin_memory: bool = True
            drop_last: bool = False
            shuffle_train: bool = True
            scale_div: float = float(args.scale_div)
            # DANN needs subject ids in the train batch (3-tuple).
            return_train_subjects: bool = (str(getattr(args, "scheme", "")).lower() == "stelar_dann")

        ds_loader = LoadDataset(_Params())
        loaders = ds_loader.get_data_loader()
        train_loader = loaders["train"]
        val_loader = loaders.get("val", None)
        test_loader = loaders["test"]
        num_subjects = int(getattr(ds_loader, "num_train_subjects", 0))

        steps = len(train_loader)
        model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps,
                             num_classes=4, num_subjects=num_subjects)

        fold_dir = base_dir / f"s{sid:02d}_{tok}"
        ck_dir = fold_dir / "ckpts"
        ck_dir.mkdir(parents=True, exist_ok=True)

        loggers = make_loggers(fold_dir, version="run", enable_csv=bool(getattr(args, "log_csv", True)))
        _log_encoder_everywhere(fold_dir, loggers, args)

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

        resume_ckpt = _discover_resume_ckpt(ck_dir) if bool(getattr(args, "resume", False)) else None
        trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

        best_path = _best_or_last_from_cb(ckpt_cb)
        test_metrics = trainer.test(
            model if best_path is None else None,
            ckpt_path=best_path,
            dataloaders=test_loader,
            verbose=False,
        )[0]

        rows.append(
            make_result_row(
                seed=int(args.seed),
                args=args,
                test_metrics=test_metrics,
                ckpt_cb=ckpt_cb,
                extra={
                    "strategy": "loso",
                    "fold_test_subject_id": sid,
                    "fold_test_subject_token": tok,
                    "best_ckpt": ckpt_cb.best_model_path or "",
                },
            )
        )

        if metric_keys is None:
            metric_keys = list(test_metrics.keys())
            metric_accum = {k: 0.0 for k in metric_keys}
        for k in metric_keys:
            metric_accum[k] += float(test_metrics[k])

    if metric_keys:
        avg_row: Dict[str, Any] = {
            "dataset": "BCIC2a",
            "strategy": "loso",
            "fold_test_subject_id": "MEAN",
            "fold_test_subject_token": "MEAN",
            "seed": int(args.seed),
        }
        for k in metric_keys:
            avg_row[k] = metric_accum[k] / float(n_subj)
        rows.append(avg_row)

    out_csv = str(base_dir / "csv" / "summary.csv")
    write_csv(out_csv, rows)
    print(f"\n[BCIC2a] LOSO finished. Summary at: {out_csv}")


def run_bcic2a_fixed(args) -> None:
    try:
        from bcic2a_dataset import LoadDataset
    except Exception:
        from downstream.datasets.bcic2a_dataset import LoadDataset

    ckpt_path, model_type = resolve_pretrain_checkpoint(args)
    base_dir = downstream_log_base(args, model_type, extra="Fixed")

    _ = _bcic2a_discover_subjects(args.bcic2a_path)

    spec = args._split_spec
    train_ids = spec.get("train")
    val_ids = spec.get("val")
    test_ids = spec.get("test")

    if not (train_ids and test_ids):
        raise ValueError(
            "For --split=fixed specify train/test (and optionally val) subject IDs via "
            "--split_spec 'train=...;val=...;test=...' or --train_subject_ids/--val_subject_ids/--test_subject_ids."
        )

    class _Params:
        datasets_dir: str = args.bcic2a_path
        batch_size: int = args.batch_size
        split_mode: str = "subject_explicit"
        train_subject_ids = train_ids
        val_subject_ids = val_ids if val_ids is not None else []
        test_subject_ids = test_ids
        num_workers: int = args.num_workers
        pin_memory: bool = True
        drop_last: bool = False
        shuffle_train: bool = True
        scale_div: float = args.scale_div
        # SINGLE seed everywhere:
        seed: int = int(args.seed)

    loaders = LoadDataset(_Params()).get_data_loader()
    train_loader = loaders["train"]
    val_loader = loaders.get("val", None)
    test_loader = loaders["test"]

    steps = len(train_loader)
    model = build_engine(args, ckpt_path=ckpt_path, steps_per_epoch=steps, num_classes=4)

    ck_dir = base_dir / "ckpts"
    ck_dir.mkdir(parents=True, exist_ok=True)
    loggers = make_loggers(base_dir, version="run", enable_csv=bool(getattr(args, "log_csv", True)))
    _log_encoder_everywhere(base_dir, loggers, args)

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

    resume_ckpt = _discover_resume_ckpt(ck_dir) if bool(getattr(args, "resume", False)) else None
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
        extra={
            "strategy": "fixed_subject",
            "train_subject_ids": train_ids,
            "val_subject_ids": val_ids,
            "test_subject_ids": test_ids,
            "best_ckpt": ckpt_cb.best_model_path or "",
        },
    )
    write_csv(str(base_dir / "csv" / "summary.csv"), [row])
    print(f"[BCIC2a] Fixed split finished. Summary at: {base_dir / 'csv' / 'summary.csv'}")


# =============================================================================
# CLI (BCIC2a-only)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser("BCIC2a downstream runner")

    p.add_argument("--bcic2a_path", type=str, required=True, help="Path to BCIC2a LMDB root.")
    p.add_argument("--split", type=str, default="loso", choices=["loso", "fixed"])
    p.add_argument("--split_spec", type=str, default=None, help="e.g. 'train=0-5;val=6;test=7-8'")
    p.add_argument("--train_subject_ids", type=str, default=None)
    p.add_argument("--val_subject_ids", type=str, default=None)
    p.add_argument("--test_subject_ids", type=str, default=None)

    p.add_argument("--scheme", type=str, required=True)

    p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=CFG.MAX_EPOCHS)

    # SINGLE SEED (used for everything)
    p.add_argument("--seed", type=int, default=CFG.SEED)

    p.add_argument("--val_ratio", type=float, default=0.0)
    _add_bool_flag(p, "stratified_val", True, "Use stratified val split when building val from train.")
    p.add_argument("--scale_div", type=float, default=1.0)

    # Optim & schedule
    p.add_argument("--max_lr_head", type=float, default=CFG.MAX_LR_HEAD)
    p.add_argument("--max_lr_enc", type=float, default=CFG.MAX_LR_ENC)
    p.add_argument("--wd_head", type=float, default=CFG.WD_HEAD)
    p.add_argument("--wd_enc", type=float, default=CFG.WD_ENC)
    p.add_argument("--scheduler", type=str, default=CFG.SCHEDULER, choices=["onecycle", "cosine", "none"])
    p.add_argument("--warmup_pct", type=float, default=CFG.WARMUP_PCT)

    p.add_argument("--freeze_encoder", action="store_true", default=False)
    p.add_argument("--ft_top_k", type=int, default=0)

    p.add_argument("--epoch_select", type=str, default=getattr(CFG, "epoch_select", "last"), choices=["best", "last"])
    p.add_argument("--version", type=int, default=getattr(CFG, "version", 0))
    p.add_argument("--pretrain_max_lr", type=float, default=getattr(CFG, "pretrain_max_lr", 5e-4))
    p.add_argument("--pretrain_dir", type=str, default=None, help="Exact .../_tb/version_X directory.")
    p.add_argument("--pretrain_ckpt", type=str, default=None, help="Exact .ckpt/.pth path (LaBraM requires .pth).")
    _add_bool_flag(p, "load_pretrained", True, "Load encoder weights; set --no-load_pretrained for random init.")
    p.add_argument("--init_scheme", type=str, default="xavier_uniform",
                   choices=["xavier_uniform", "xavier_normal", "kaiming_normal"])

    p.add_argument("--tag", type=str, default=CFG.tag)
    p.add_argument("--variant", type=str, default=CFG.variant)
    p.add_argument("--preprocess", type=str, default=CFG.preprocess)
    p.add_argument("--encoder", type=str, default=CFG.encoder)
    p.add_argument("--transformer", type=str, default=CFG.transformer)

    _add_bool_flag(p, "use_dynamic_pe", CFG.USE_DYNAMIC_PE, "Enable dynamic positional encoding (ACPE).")
    _add_bool_flag(p, "rope_time", CFG.ROPE_TIME_DEFAULT, "Enable RoPE along time axis (criss-cross).")
    _add_bool_flag(p, "rope_chan", CFG.ROPE_CHAN_DEFAULT, "Enable RoPE along channel axis (criss-cross).")
    p.add_argument("--rope_dim_frac", type=float, default=CFG.ROPE_DIM_FRAC_DEFAULT)
    p.add_argument("--rope_interp_time", type=float, default=CFG.ROPE_INTERP_TIME_DEFAULT)
    p.add_argument("--rope_interp_chan", type=float, default=CFG.ROPE_INTERP_CHAN_DEFAULT)
    _add_bool_flag(p, "use_channel_embed", True, "Use channel embedding with channel IDs inside the encoder.")

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

    p.add_argument("--labram_model", type=str, default="labram_base_patch200_200")
    p.add_argument("--labram_handle_remainder", type=str, default="crop", choices=["crop", "pad"])
    p.add_argument("--labram_pad_value", type=float, default=0.0)

    _add_bool_flag(p, "use_chan_conv", True, "1x1 channel-mixing Conv1d before encoder (STELAR).")
    _add_bool_flag(p, "use_time_chan_conv", False, "EEGNet-style temporal depthwise Conv2d before encoder (STELAR TUEV).")
    p.add_argument("--chan_max_norm", type=float, default=1.0)
    p.add_argument("--conv_kernel_time", type=int, default=55)
    p.add_argument("--conv_dropout", type=float, default=0.8)

    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (0 disables).")

    p.add_argument("--devices", default=",".join(map(str, getattr(CFG, "devices", []))))
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--precision", type=str, default="32")
    _add_bool_flag(p, "resume", False, "Resume training from last.ckpt when available.")

    p.add_argument("--log_root", type=str, default="./logs")
    _add_bool_flag(p, "log_csv", True, "Also write CSV metrics alongside TensorBoard.")

    # ---- DANN (domain-adversarial; --scheme stelar_dann) ----
    g = p.add_argument_group("DANN")
    g.add_argument("--lambda_adv", type=float, default=1.0, help="Weight of the adversarial subject loss.")
    g.add_argument("--grl_alpha", type=float, default=1.0, help="Fixed GRL alpha (when --no-schedule_alpha).")
    _add_bool_flag(p, "schedule_alpha", True, "Use the DANN progressive GRL alpha schedule.")
    g.add_argument("--grl_gamma", type=float, default=10.0, help="Steepness of the GRL sigmoid schedule.")
    g.add_argument("--subj_disc_hidden", type=int, default=64, help="Subject discriminator hidden dim.")
    g.add_argument("--subj_disc_dropout", type=float, default=0.3, help="Subject discriminator dropout.")

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

    args.devices = parse_devices(args.devices)
    if args.num_workers is None:
        args.num_workers = default_num_workers(args.devices)

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

    scheme = args.scheme.lower()
    if scheme.startswith("cbramod"):
        args.use_channel_names = None
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
        if getattr(args, "model_channel_set", None):
            args.use_channel_names = MONTAGE_BY_DATASET.get(args.model_channel_set) or list(DEFAULT_EEGPT58)
        else:
            args.use_channel_names = MONTAGE_BY_DATASET.get("BCIC2a", list(DEFAULT_EEGPT58))
        args.target_chans = len(args.use_channel_names)

    args._split_spec = (
        _parse_split_spec(args.split_spec)
        if args.split_spec
        else {
            "train": _parse_id_list(args.train_subject_ids),
            "val": _parse_id_list(args.val_subject_ids),
            "test": _parse_id_list(args.test_subject_ids),
        }
    )

    _normalize_frontend_flags(args)
    return args


def main() -> None:
    args = parse_args()

    # SINGLE seed drives everything (torch, numpy, python, dataloader workers).
    pl.seed_everything(int(args.seed), workers=True)

    if args.split == "loso":
        run_bcic2a_loso(args)
    elif args.split == "fixed":
        run_bcic2a_fixed(args)
    else:
        raise ValueError(f"Unknown split: {args.split}")


if __name__ == "__main__":
    main()
