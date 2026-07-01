# utils.py
# -*- coding: utf-8 -*-
"""
Lightweight, import-friendly helpers used across the codebase.

Contains:
- RNG & device helpers
- Small data transforms / augmentations
- Simple logging helpers
- Minimal EEG dataset wrapper
- Thin wrapper `get_checkpoint_path` (delegates to io_utils) to keep old imports working
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------- to_tensor helper --------------------------- #
def to_tensor(x, *, dtype: Optional[torch.dtype] = None,
              device: Optional[torch.device] = None, copy: bool = False) -> torch.Tensor:
    """
    Robust numpy -> torch converter.
    - Keeps numpy arrays contiguous (copies only if needed or copy=True).
    - Preserves dtype unless `dtype` is specified.
    - Moves to `device` if given.
    """
    if isinstance(x, torch.Tensor):
        if dtype is not None and x.dtype is not dtype:
            x = x.to(dtype)
        if device is not None:
            x = x.to(device)
        return x

    arr = np.asarray(x)
    if (copy or not arr.flags.c_contiguous):
        arr = np.ascontiguousarray(arr)

    t = torch.from_numpy(arr)
    if dtype is not None and t.dtype is not dtype:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t

# ============================ RNG & DEVICES ============================

def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU/CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_devices(devices_arg) -> List[int]:
    """
    Accepts '0,1', '0', 0, [0,1], [], None -> returns list of GPU indices (or []).
    """
    if devices_arg is None:
        return [0] if torch.cuda.is_available() else []
    if isinstance(devices_arg, (list, tuple)):
        return [int(d) for d in devices_arg]
    if isinstance(devices_arg, int):
        return [devices_arg]
    if isinstance(devices_arg, str):
        s = devices_arg.strip()
        if s == "":
            return [0] if torch.cuda.is_available() else []
        return [int(x) for x in s.split(",") if x.strip()]
    return [0] if torch.cuda.is_available() else []


def default_num_workers(devices: List[int]) -> int:
    """Heuristic: min(8 * max(1, len(devices)), 32, max(0, cpu-1))."""
    try:
        cpu = os.cpu_count() or 8
    except Exception:
        cpu = 8
    return min(8 * max(1, len(devices)), 32, max(0, cpu - 1))


def select_devices(num_device: int, gpus: Optional[Sequence[int]] = None) -> List[torch.device]:
    """
    Pick `num_device` CUDA devices from the provided list (or all).
    Only selects cards with >= ~12.7GB total memory by default.
    """
    if gpus is None:
        gpus = list(range(torch.cuda.device_count()))
    res: List[torch.device] = []
    last = 0
    min_total_mem = 25447170048 // 2  # ≈12.7GB; tweak as needed
    for _ in range(num_device):
        if not gpus:
            break
        dev = gpus[last % len(gpus)]
        last += 1
        tries = 0
        while torch.cuda.get_device_properties(dev).total_memory < min_total_mem and tries < len(gpus):
            dev = gpus[last % len(gpus)]
            last += 1
            tries += 1
        res.append(torch.device(f"cuda:{dev}"))
    return res


def select_free_gpu() -> Optional[torch.device]:
    """
    Randomly sample a GPU and return the first with >= ~12.7GB total memory.
    (Total memory is a coarse proxy; adjust to your environment.)
    """
    count = torch.cuda.device_count()
    if count == 0:
        return None
    min_total_mem = 25447170048 // 2
    for _ in range(count):
        i = random.randint(0, count - 1)
        props = torch.cuda.get_device_properties(i)
        if props.total_memory >= min_total_mem:
            return torch.device(f"cuda:{i}")
    return torch.device("cuda:0") if count > 0 else None


# ============================= DATASET ==============================

class eeg_dataset(Dataset):
    """Minimal dataset wrapper for (x, y[, subject_id]) tensors."""
    def __init__(self, feature: torch.Tensor, label: torch.Tensor,
                 subject_id: Optional[torch.Tensor] = None):
        super().__init__()
        self.x = feature
        self.y = label
        self.s = subject_id

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    # Few-shot helpers (kept for backward-compat)
    def get_num_class(self, num_class: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        buckets = [[] for _ in num_class]
        remaining = list(num_class)
        idxs = list(range(len(self.y)))
        import random as _rnd
        while sum(remaining) > 0:
            i = _rnd.choice(idxs)
            lab = int(self.y[i])
            if remaining[lab] > 0:
                remaining[lab] -= 1
                buckets[lab].append((self.x[i], self.y[i]))
        flat = [p for b in buckets for p in b]
        xs = torch.stack([t[0] for t in flat], dim=0)
        ys = torch.stack([t[1] for t in flat], dim=0)
        return xs, ys

    def get_num_subject(self, num_subject: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.s is not None, "Subject ids not provided to dataset."
        buckets = [[] for _ in num_subject]
        remaining = list(num_subject)
        idxs = list(range(len(self.y)))
        import random as _rnd
        while sum(remaining) > 0:
            i = _rnd.choice(idxs)
            sid = int(self.s[i])
            if remaining[sid] > 0:
                remaining[sid] -= 1
                buckets[sid].append((self.x[i], self.y[i]))
        flat = [p for b in buckets for p in b]
        xs = torch.stack([t[0] for t in flat], dim=0)
        ys = torch.stack([t[1] for t in flat], dim=0)
        return xs, ys


# =================== Compatibility: get_checkpoint_path ===================

def get_checkpoint_path(base_dir: str, which: str | int = "last",
                        metric: str = "val_loss", mode: str = "min"):
    """
    Compatibility wrapper so existing code can still:
        from utils import get_checkpoint_path
    while the actual implementation lives in io_utils.
    """
    from io_utils import get_checkpoint_path as _gcp  # local import avoids circulars
    return _gcp(base_dir, which=which, metric=metric, mode=mode)



def resolve_pretrain_checkpoint(args) -> Tuple[str, str]:
    """
    Return (ckpt_path, run_tag).
    Mirrors pretrain runner naming:
      RUN = f"EEGPT_{tag}_{variant}_{preprocess}_{encoder}_{transformer}_{flags_code}_tb"
      BASE = ../pretrain/{encoder}_{transformer}/logs/{pretrain_max_lr}/{RUN}/version_{X}
    Priority:
      1) --pretrain_ckpt
      2) --pretrain_dir + epoch_select
      3) search_root + (pretrain_run_name | flags_code | wildcard) + (version | newest) + epoch_select
    """
    import glob, os
    from pathlib import Path

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

    # 2) exact version dir (…/_tb/version_X)
    if args.pretrain_dir:
        ver_dir = Path(args.pretrain_dir)
        if not ver_dir.is_dir():
            raise FileNotFoundError(f"--pretrain_dir is not a directory: {ver_dir}")
        run_tag = ver_dir.parent.name.replace("_tb", "")
        ckpt_path, _ = get_checkpoint_path(str(ver_dir), which="best" if args.epoch_select == "best" else "last")
        return ckpt_path, run_tag

    # 3) discover under search root
    stable_prefix = f"EEGPT_{args.tag}_{args.variant}_{args.preprocess}_{args.encoder}_{args.transformer}_"
    search_root = Path(args.pretrain_search_root) if args.pretrain_search_root else Path(
        f"../pretrain/{args.encoder}_{args.transformer}/logs/{args.pretrain_max_lr}"
    )
    if not search_root.is_dir():
        raise FileNotFoundError(f"Search root not found: {search_root}")

    # decide the run folder(s)
    if args.pretrain_run_name:
        candidates = [search_root / args.pretrain_run_name]
    else:
        # if user provided a flags_code, build exact run name; else wildcard
        if args.pretrain_flags_code:
            run_name = f"{stable_prefix}{args.pretrain_flags_code}_tb"
            candidates = [search_root / run_name]
        else:
            pattern = f"{stable_prefix}*_tb"
            candidates = [Path(p) for p in glob.glob(str(search_root / pattern))]

    candidates = [c for c in candidates if c.is_dir()]
    if not candidates:
        present = [Path(p).name for p in glob.glob(str(search_root / "*"))]
        raise FileNotFoundError(
            f"No pretrain runs found under {search_root} matching {args.pretrain_run_name or args.pretrain_flags_code or '<auto *>'}.\n"
            f"Expected prefix: {stable_prefix} ; Present: {present}"
        )

    # pick newest run by mtime if multiple
    if len(candidates) > 1:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen_run = candidates[0]
    run_tag = chosen_run.name.replace("_tb", "")

    # choose version dir (explicit or newest)
    version_dir = None
    if getattr(args, "version", None) is not None and int(args.version) >= 0:
        vd = chosen_run / f"version_{int(args.version)}"
        if vd.is_dir():
            version_dir = vd
    if version_dir is None:
        versions = [Path(p) for p in glob.glob(str(chosen_run / "version_*")) if Path(p).is_dir()]
        if not versions:
            raise FileNotFoundError(f"No version_* dirs under {chosen_run}")
        versions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        version_dir = versions[0]

    ckpt_path, _ = get_checkpoint_path(
        str(version_dir),
        which="best" if args.epoch_select == "best" else "last"
    )
    return ckpt_path, run_tag

__all__ = [
    "to_tensor",
    # RNG & devices
    "set_seed", "parse_devices", "default_num_workers", "select_devices", "select_free_gpu",
    # dataset
    "eeg_dataset",
    # compat
    "get_checkpoint_path",
    "resolve_pretrain_checkpoint"
]
