# configs.py (STELAR)
from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
import lmdb

# =========================
# Dataloader safety knobs (override via env)
# =========================


def _env_flag(name: str, default: str) -> str:
    """Return raw env var string or default if unset."""
    v = os.getenv(name)
    return v if v is not None else default


STELAR_NUM_WORKERS = os.getenv("STELAR_NUM_WORKERS")  # allow explicit override

CPU_COUNT = os.cpu_count() or 2
# Start conservative: 2 workers max by default; never exceed 1/8 of CPUs.
DEFAULT_WORKERS = max(0, min(2, CPU_COUNT // 8))
NUM_WORKERS: int = int(STELAR_NUM_WORKERS) if STELAR_NUM_WORKERS else DEFAULT_WORKERS

# pin_memory default OFF; allow override by env
PIN_MEMORY: bool = _env_flag("STELAR_PIN_MEMORY", "0") not in ("0", "false", "False", "no", "NO")
# modest prefetch; 1 prevents a pile-up of pinned batches
PREFETCH_FACTOR: int = int(os.getenv("STELAR_PREFETCH_FACTOR", "1"))
# disable persistent workers by default; enable later if stable
PERSISTENT_WORKERS: bool = _env_flag("STELAR_PERSISTENT_WORKERS", "0") not in (
    "0",
    "false",
    "False",
    "no",
    "NO",
)

# =========================
# Repro & tiny helpers
# =========================

SEED: int = 7


def seed_torch(seed: int = 1029) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# =========================
# GPU / batch helpers
# =========================


def _parse_devices_from_env() -> List[int]:
    env = os.getenv("STELAR_DEVICES")
    if env:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    n = torch.cuda.device_count()
    if n and n > 0:
        return list(range(n))
    return []


def world_size(devs: List[int]) -> int:
    return max(1, len(devs))


def per_gpu_batch_size(global_bs: int, devs: List[int]) -> int:
    ws = world_size(devs)
    return max(1, global_bs // ws)


# =========================
# Data loading
# =========================

WINDOW_SECONDS = 4
FS = 256
WINDOW_LENGTH = WINDOW_SECONDS * FS  # 4s @ 256Hz


def load_fn(
    path_str: str,
    *,
    input_mode: str = "raw",
    window_length: int = WINDOW_LENGTH,
    pretoken_layout: str = "CND",
) -> torch.Tensor:
    """
    raw mode:
        returns x: [C, T]; randomly crop to `window_length` if T > window_length.
    pretoken mode:
        returns x: [C, N, D]. Supports files saved as [C,N,D] or [N,C,D] (via `pretoken_layout`).
        No cropping applied.
    """
    x = torch.load(path_str)
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Loaded object is not a torch.Tensor: {path_str}")

    if input_mode == "pretoken":
        if x.ndim != 3:
            raise ValueError(
                f"Pretoken file should be 3D [C,N,D] or [N,C,D], got {tuple(x.shape)} from {path_str}"
            )
        if pretoken_layout.upper() == "CND":
            cnd = x
        elif pretoken_layout.upper() == "NCD":
            # [N,C,D] -> [C,N,D]
            cnd = x.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(f"Unknown pretoken_layout={pretoken_layout!r}")
        return cnd.to(torch.float32)

    # raw
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor (C,T) in raw mode, got {tuple(x.shape)} from {path_str}")
    C, T = x.shape
    if T > window_length:
        start = random.randint(0, T - window_length)
        x = x[:, start : start + window_length]
    return x.to(torch.float32)


# =========================
# Training hyperparameters
# =========================

max_epochs: int = 200
pretrain_max_lr: float = 1e-4
global_batch_size: int = 64
devices: List[int] = _parse_devices_from_env()
mask_ratio: float = 0.5

# =========================
# Training feature switches (defaults)
# =========================
USE_LOSS_R: bool = True
USE_LOSS_AM: bool = True
USE_LN: bool = True
USE_SKIP: bool = False

# Channel embedding via channel IDs (default ON; runner can disable)
USE_CHANNEL_EMBED: bool = True

# Optional feature: ACPE / dynamic PE (default OFF because we already use channel IDs)
USE_DYNAMIC_PE: bool = False

# =========================
# Data roots & loaders
# =========================

DATA_ROOT = Path(
    os.getenv("STELAR_DATA_ROOT", "/projects/EEG-foundation-model/STELAR_pretrain/merged")
)
TRAIN_ROOT = DATA_ROOT / "TrainFolder"
VALID_ROOT = DATA_ROOT / "ValidFolder"
FILE_EXTENSIONS: List[str] = [".pt", ".pth", ".edf"]  # keep as-is if your tree expects EDF/pt/pth mixed


def _gather_files(root: Path, exts: List[str]) -> List[Path]:
    exts = [e.lower() for e in exts]
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


class TensorFileDataset(Dataset):
    def __init__(self, root: Path, extensions: List[str], *, input_mode: str = "raw", pretoken_layout: str = "CND"):
        if not root.exists():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        self.paths = _gather_files(root, extensions)
        if not self.paths:
            raise RuntimeError(
                f"No files with extensions {extensions} found under {root}. "
                f"Example expected path: {root}/**/*{extensions[0]}"
            )
        self.input_mode = input_mode
        self.pretoken_layout = pretoken_layout

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return load_fn(
            str(self.paths[idx]),
            input_mode=self.input_mode,
            window_length=WINDOW_LENGTH,
            pretoken_layout=self.pretoken_layout,
        )


# CBraMod-style LMDB
class PretrainingLMDBDataset(Dataset):
    """
    LMDB -> pickled arrays/tensors.
    Expected 3D samples either [C,N,D] or [N,C,D] (use pretoken_layout).
    """

    def __init__(self, dataset_dir: Path, *, pretoken_layout: str = "CND"):
        self.env = lmdb.open(
            str(dataset_dir), readonly=True, lock=False, readahead=True, meminit=False
        )
        if self.env is None:
            raise FileNotFoundError(f"Could not open LMDB at: {dataset_dir}")
        self.pretoken_layout = pretoken_layout.upper()
        with self.env.begin(write=False) as txn:
            raw = txn.get(b"__keys__")
            if raw is None:
                raise RuntimeError(f"LMDB at {dataset_dir} missing __keys__")
            self.keys = pickle.loads(raw)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> torch.Tensor:
        key = self.keys[idx]
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode())
            if buf is None:
                raise KeyError(f"Missing LMDB key: {key}")
            x = pickle.loads(buf)
        x = torch.as_tensor(x)
        if x.ndim != 3:
            raise ValueError(f"LMDB sample must be 3D, got {tuple(x.shape)} for key={key}")
        if self.pretoken_layout == "NCD":
            x = x.permute(1, 0, 2).contiguous()
        elif self.pretoken_layout != "CND":
            raise ValueError(f"Unknown pretoken_layout={self.pretoken_layout!r}")
        return x.to(torch.float32)


# ------------------------- Normalization ------------------------- #


class _MapDataset(torch.utils.data.Dataset):
    """Wrap any Dataset to apply a transform to the first item (the tensor)."""

    def __init__(self, base_ds: torch.utils.data.Dataset, transform: Callable[[torch.Tensor], torch.Tensor]):
        self.base_ds = base_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_ds)

    def __getitem__(self, idx: int):
        item = self.base_ds[idx]
        if isinstance(item, (tuple, list)):
            x = item[0]
            x = self.transform(x)
            return (x, *item[1:])
        else:
            return self.transform(item)


def _build_normalizer(
    *,
    mode: str,  # "raw" ([C,T]) or "pretoken" ([C,N,D]) at item level
    kind: str = "scale",  # "none" | "zscore" | "robust" | "scale"
    eps: float = 1e-6,
    clip: Optional[float] = None,
    fixed_scale: Optional[float] = None,  # used when kind="scale"
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable that normalizes a single sample tensor."""
    kind = kind.lower().strip()
    assert mode in ("raw", "pretoken"), f"mode must be raw|pretoken, got {mode}"
    assert kind in ("none", "zscore", "robust", "scale"), f"unknown kind={kind}"

    def _zscore_last(x: torch.Tensor) -> torch.Tensor:
        # raw: [C,T] → zscore over T; pretoken: [C,N,D] → zscore over D
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True)
        x = (x - m) / (s + eps)
        return x

    def _robust_last(x: torch.Tensor) -> torch.Tensor:
        # median/MAD over last dim
        med = x.median(dim=-1, keepdim=True).values
        mad = (x - med).abs().median(dim=-1, keepdim=True).values
        x = (x - med) / (1.4826 * mad + eps)
        return x

    def _scale(x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 if fixed_scale is None else float(fixed_scale)
        return x * scale

    def _noop(x: torch.Tensor) -> torch.Tensor:
        return x

    fn = {"none": _noop, "zscore": _zscore_last, "robust": _robust_last, "scale": _scale}[kind]

    def _apply(x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            return x
        if x.ndim not in (2, 3):
            return x
        y = fn(x)
        if clip is not None:
            y = y.clamp_(min=-float(clip), max=float(clip))
        return y

    return _apply


def get_dataloaders(
    train_root: Path = TRAIN_ROOT,
    valid_root: Path = VALID_ROOT,
    extensions: List[str] = FILE_EXTENSIONS,
    global_bs: int = global_batch_size,
    num_workers: int = NUM_WORKERS,
    devs: List[int] = devices,
    *,
    input_mode: str = "raw",  # "raw" or "pretoken"
    pretoken_layout: str = "CND",
    data_backend: str = "files",  # "files" or "lmdb"
    train_lmdb: Optional[Path] = None,
    valid_lmdb: Optional[Path] = None,
    val_fraction: float = 0.0,  # 0 => no split; >0 => split from train
    seed: int = SEED,
    # ---- Normalization controls (per-sample, per-channel) ----
    normalize: str = "scale",  # "none" | "zscore" | "robust" | "scale"
    norm_clip: Optional[float] = None,  # e.g., 5.0 after zscore/robust
    norm_eps: float = 1e-6,
    norm_fixed_scale: Optional[float] = 0.01,  # used if normalize == "scale"
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Returns (train_loader, valid_loader_or_None).
    Normalization is applied per item: raw=[C,T] (over T), pretoken=[C,N,D] (over D).
    """
    bs = per_gpu_batch_size(global_bs, devs)

    # pick dataset(s)
    if data_backend.lower() == "lmdb":
        if input_mode != "pretoken":
            raise ValueError("LMDB backend expects input_mode='pretoken'.")
        if train_lmdb is None:
            raise ValueError("Provide train_lmdb for data_backend=lmdb.")
        train_ds = PretrainingLMDBDataset(Path(train_lmdb), pretoken_layout=pretoken_layout)
        if valid_lmdb is not None:
            valid_ds = PretrainingLMDBDataset(Path(valid_lmdb), pretoken_layout=pretoken_layout)
            do_split = False
        else:
            valid_ds = None
            do_split = val_fraction > 0.0
    else:
        train_ds = TensorFileDataset(train_root, extensions, input_mode=input_mode, pretoken_layout=pretoken_layout)
        if valid_root.exists():
            try:
                valid_ds = TensorFileDataset(valid_root, extensions, input_mode=input_mode, pretoken_layout=pretoken_layout)
                do_split = False
            except RuntimeError:
                valid_ds = None
                do_split = val_fraction > 0.0
        else:
            valid_ds = None
            do_split = val_fraction > 0.0

    # optional split from train
    if do_split:
        n_total = len(train_ds)
        n_val = max(1, int(round(n_total * float(val_fraction))))
        n_train = max(1, n_total - n_val)
        g = torch.Generator().manual_seed(int(seed))
        train_ds, valid_ds = torch.utils.data.random_split(train_ds, [n_train, n_val], generator=g)

    # wrap datasets with normalizer (if enabled)
    nm = _build_normalizer(
        mode=("pretoken" if input_mode == "pretoken" else "raw"),
        kind=normalize,
        eps=norm_eps,
        clip=norm_clip,
        fixed_scale=norm_fixed_scale,
    )
    if normalize.lower() != "none":
        train_ds = _MapDataset(train_ds, nm)
        if valid_ds is not None:
            valid_ds = _MapDataset(valid_ds, nm)

    # loaders
    dl_common = dict(
        batch_size=bs,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS if num_workers > 0 else False),
        drop_last=False,
    )
    if num_workers > 0:
        dl_common["prefetch_factor"] = PREFETCH_FACTOR

    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **dl_common)
    valid_loader = (
        torch.utils.data.DataLoader(valid_ds, shuffle=False, **dl_common)
        if valid_ds is not None
        else None
    )
    return train_loader, valid_loader


# =========================
# Steps per epoch
# =========================


def steps_per_epoch_from_loader(train_loader: torch.utils.data.DataLoader) -> int:
    try:
        return len(train_loader)
    except TypeError:
        pass
    ds_len = len(getattr(train_loader, "dataset", [])) or 0
    bs = getattr(train_loader, "batch_size", None) or per_gpu_batch_size(
        global_batch_size, devices
    )
    return math.ceil(ds_len / bs) if ds_len and bs else 0


def compute_steps_per_epoch(
    train_root: Path = TRAIN_ROOT,
    extensions: List[str] = FILE_EXTENSIONS,
    global_bs: int = global_batch_size,
    devs: List[int] = devices,
) -> int:
    train_ds = TensorFileDataset(train_root, extensions)
    bs = per_gpu_batch_size(global_bs, devs)
    return math.ceil(len(train_ds) / bs)


# =========================
# Model selection & logging
# =========================

tag: str = "tiny"
variant: str = "D"
preprocess: str = "rep"
encoder: str = "full_token"
transformer: str = "crisscross"  # CLI can override
model_type = f"STELAR_{tag}"
log_root = Path("./logs") / f"{pretrain_max_lr}"
# model_type = f"STELAR_{tag}_{variant}_{preprocess}_{encoder}_{transformer}"

# Resume/run options
RESUME: bool = False
version: int = 0


def make_base_dir(encoder: str, transformer: str, lr: float, model_type: str, version: int) -> Path:
    return (
        Path("../pretrain")
        / f"{encoder}_{transformer}"
        / "logs"
        / f"{lr}"
        / f"{model_type}_tb"
        / f"version_{version}"
    )


PIN_LOG_VERSION: Optional[int] = None  # None => timestamped runs

# =========================
# Predictor defaults
# =========================

PRED_KV_MODE_DEFAULT: str = "global_vis_kv"
PRED_CC: bool = False

# =========================
# Criss-Cross / RoPE defaults (shared)
# =========================
ROPE_TIME_DEFAULT: bool = True
ROPE_CHAN_DEFAULT: bool = False  # because we have channel ids
ROPE_DIM_FRAC_DEFAULT: float = 1.0
ROPE_INTERP_TIME_DEFAULT: float = 1.0
ROPE_INTERP_CHAN_DEFAULT: float = 1.0

# =========================
# Fixed loss weights (global)
# =========================
LAMBDA_RECON: float = 0.5  # reconstruction weight
LAMBDA_ALIGN_VIS: float = 0.0  # visible alignment weight
LAMBDA_ALIGN_MASK: float = 0.5  # masked alignment weight

LOSS_WEIGHTS: Dict[str, float] = {
    "lambda_recon": LAMBDA_RECON,
    "lambda_align_vis": LAMBDA_ALIGN_VIS,
    "lambda_align_mask": LAMBDA_ALIGN_MASK,
}

# =========================
# Model configs (per tag)
# =========================
# NOTE:
# - predictor configs here are *predictor-side* widths; LitSTELAR passes them through
#   to TransformerPredictor.
# - reconstructor configs control the reconstructor head (light or transformer).
#   All other reconstructor hyperparams are hard-coded in the engine.

MODELS_CONFIGS: Dict[str, Dict] = {
    "tiny": {
        "encoder": {
            "embed_dim": 64,
            "num_heads": 2,
            "depth": 2,
        },
        "predictor": {
            "embed_dim": 64,
            "predictor_embed_dim": 64,
            "num_heads": 2,
            "depth": 1,
        },
        "reconstructor": {
            "type": "mlp",  # "mlp" or "transformer"
            "hidden_ratio": 0.0,
            "act": "gelu",
            "reconstructor_embed_dim": 64,
            "depth": 1,
            "num_heads": 2,
        },
    },
    # "little": {
    #     "encoder": {
    #         "embed_dim": 128,
    #         "num_heads": 2,
    #         "depth": 8,
    #     },
    #     "predictor": {
    #         "embed_dim": 128,
    #         "predictor_embed_dim": 64,
    #         "num_heads": 2,
    #         "depth": 1,
    #     },
    #     "reconstructor": {
    #         "type": "transformer",
    #         "hidden_ratio": 0.0,
    #         "act": "gelu",
    #         "reconstructor_embed_dim": 64,
    #         "depth": 1,
    #         "num_heads": 2,
    #     },
    # },
    "little": {
        "encoder": {
            "embed_dim": 128,
            "num_heads": 4,
            "depth": 8,
        },
        "predictor": {
            "embed_dim": 128,
            "predictor_embed_dim": 128,
            "num_heads": 4,
            "depth": 4,
        },
        "reconstructor": {
            "type": "mlp",
            "hidden_ratio": 0.0,
            "act": "gelu",
            "reconstructor_embed_dim": 128,
            "depth": 4,
            "num_heads": 4,
        },
    },
    "base": {
        "encoder": {
            "embed_dim": 256,
            "num_heads": 4,
            "depth": 8,
        },
        "predictor": {
            "embed_dim": 256,
            "predictor_embed_dim": 256,
            "num_heads": 4,
            "depth": 4,
        },
        "reconstructor": {
            "type": "mlp",
            "hidden_ratio": 0.0,
            "act": "gelu",
            "reconstructor_embed_dim": 256,
            "depth": 4,
            "num_heads": 4,
        },
    },
    "large": {
        "encoder": {
            "embed_dim": 512,
            "num_heads": 8,
            "depth": 8,
        },
        "predictor": {
            "embed_dim": 512,
            "predictor_embed_dim": 256,
            "num_heads": 8,
            "depth": 1,
        },
        "reconstructor": {
            "type": "transformer",
            "hidden_ratio": 0.0,
            "act": "gelu",
            "reconstructor_embed_dim": 256,
            "depth": 1,
            "num_heads": 8,
        },
    },
    "huge": {
        "encoder": {
            "embed_dim": 512,
            "num_heads": 8,
            "depth": 12,
        },
        "predictor": {
            "embed_dim": 512,
            "predictor_embed_dim": 256,
            "num_heads": 8,
            "depth": 2,
        },
        "reconstructor": {
            "type": "transformer",
            "hidden_ratio": 0.0,
            "act": "gelu",
            "reconstructor_embed_dim": 256,
            "depth": 1,
            "num_heads": 8,
        },
    },
}

def get_config(tag_name: str | None = None) -> Dict[str, Dict]:
    """
    Return encoder + predictor + reconstructor config for a given model tag.

    Uses MODELS_CONFIGS defined in this file only.
    """
    name = tag_name or tag
    cfg = MODELS_CONFIGS.get(name)
    if cfg is None:
        raise KeyError(f"Unknown tag '{name}'. Available: {list(MODELS_CONFIGS)}")
    # Ensure reconstructor key exists (defensive) even if a tag only defined encoder/predictor.
    recon_cfg = cfg.get("reconstructor", {"type": "light"})
    return {
        "encoder": cfg["encoder"],
        "predictor": cfg["predictor"],
        "reconstructor": recon_cfg,
    }


__all__ = [
    # data
    "WINDOW_SECONDS",
    "FS",
    "WINDOW_LENGTH",
    "load_fn",
    "DATA_ROOT",
    "TRAIN_ROOT",
    "VALID_ROOT",
    "FILE_EXTENSIONS",
    "NUM_WORKERS",
    "get_dataloaders",
    "compute_steps_per_epoch",
    "steps_per_epoch_from_loader",
    # train hparams
    "max_epochs",
    "pretrain_max_lr",
    "global_batch_size",
    "devices",
    "mask_ratio",
    # model/logging
    "tag",
    "variant",
    "preprocess",
    "encoder",
    "transformer",
    "log_root",
    "model_type",
    "RESUME",
    "version",
    "PIN_LOG_VERSION",
    "make_base_dir",
    # model cfgs
    "MODELS_CONFIGS",
    "get_config",
    # utils
    "SEED",
    "seed_torch",
    # gpu helpers
    "per_gpu_batch_size",
    "world_size",
    # training feature switches
    "USE_LOSS_R",
    "USE_LOSS_AM",
    "USE_LN",
    "USE_SKIP",
    "USE_CHANNEL_EMBED",
    "USE_DYNAMIC_PE",
    # CC/RoPE defaults
    "ROPE_TIME_DEFAULT",
    "ROPE_CHAN_DEFAULT",
    "ROPE_DIM_FRAC_DEFAULT",
    "ROPE_INTERP_TIME_DEFAULT",
    "ROPE_INTERP_CHAN_DEFAULT",
    # predictor defaults
    "PRED_KV_MODE_DEFAULT",
    "PRED_CC",
    # loss weight defaults
    "LAMBDA_RECON",
    "LAMBDA_ALIGN_VIS",
    "LAMBDA_ALIGN_MASK",
    "LOSS_WEIGHTS",
]
