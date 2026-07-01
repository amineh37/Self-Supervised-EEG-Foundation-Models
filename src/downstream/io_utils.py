# io_utils.py
# -*- coding: utf-8 -*-
"""
I/O helpers:
- Kaggle-ERN CSV readers
- Checkpoint discovery (Lightning-style)
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import torch

# Lazy import of normalize inside functions to avoid heavy deps at module import time
# from utils import min_max_normalize  # avoid top-level to prevent circulars


# ============================== Kaggle ERN ===============================

USE_CHANNELS_19 = [
    "FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4","P8","O1","O2"
]
CH_NAMES_KAGGLE_ERN = "Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(",")

def _map_channels_kaggle(names: Sequence[str]) -> List[int]:
    base = [x.lower().strip(".") for x in CH_NAMES_KAGGLE_ERN]
    idxs = []
    for ch in names:
        try:
            idxs.append(base.index(ch.lower()))
        except ValueError as e:
            raise KeyError(f"Channel '{ch}' not found in Kaggle-ERN header list.") from e
    return idxs


def read_csv_epochs(filename: str, tmin: float, tlen: float,
                    use_channels_names: Optional[Sequence[str]] = USE_CHANNELS_19,
                    data_max: Optional[float] = None,
                    data_min: Optional[float] = None) -> List[torch.Tensor]:
    """
    Read one Kaggle-ERN CSV file and extract per-feedback epochs.

    Returns a list of trials with shape [C, T] (possibly C+1 if scale row appended).
    """
    import pandas as pd
    import mne
    from utils import min_max_normalize  # lazy to avoid circular import
    mne.set_log_level("ERROR")

    sfreq = 200
    raw = pd.read_csv(filename)
    data = torch.tensor(raw.iloc[:, 1:-2].values, dtype=torch.float32)  # exclude time/EOG/Feedback
    feed = torch.tensor(raw["FeedBackEvent"].values)
    stim_pos = torch.nonzero(feed > 0, as_tuple=False).flatten()

    use_idxs: Optional[List[int]] = None
    if use_channels_names is not None:
        use_idxs = _map_channels_kaggle(use_channels_names)
        if data_max is not None and data_min is not None:
            use_idxs += [-1]  # add scale row

    trials: List[torch.Tensor] = []
    for pos in stim_pos.tolist():
        start = max(int(pos + sfreq * tmin), 0)
        end = min(start + int(sfreq * tlen), data.shape[0])
        trial = data[start:end, :].T.numpy()  # [C, T]

        info = mne.create_info(ch_names=[str(i) for i in range(trial.shape[0])], ch_types="eeg", sfreq=sfreq)
        raw_mne = mne.io.RawArray(trial, info)

        trial_t = torch.tensor(raw_mne.get_data(), dtype=torch.float32)  # [C, T]
        trial_t = min_max_normalize(trial_t, data_max=data_max, data_min=data_min)
        if use_idxs is not None:
            trial_t = trial_t[use_idxs]
        trials.append(trial_t)
    return trials


def read_kaggle_ern_train(path: str,
                          subjects: Sequence[int],
                          sessions: Sequence[int] = (1, 2, 3, 4, 5),
                          tmin: float = -0.7, tlen: float = 2.0,
                          data_max: Optional[float] = None, data_min: Optional[float] = None,
                          use_channels_names: Optional[Sequence[str]] = USE_CHANNELS_19):
    """Return list of (trial_tensor, label) for Kaggle-ERN Train split."""
    from tqdm import tqdm

    labels = []
    with open(os.path.join(path, "KaggleERN", "TrainLabels.csv"), "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 0:
                labels.append(row)
    labels = dict(labels)  # key: 'S02_Sess01_FB001' -> '1'

    data = []
    for i in tqdm(subjects, desc="KaggleERN-Train"):
        for j in sessions:
            if i > 9:
                if (i == 22 and j == 5):
                    print("Skipped known-bad file: Data_S22_Sess05.csv")
                    continue
                filename = os.path.join(path, "KaggleERN", "train", f"Data_S{i}_Sess0{j}.csv")
            else:
                filename = os.path.join(path, "KaggleERN", "train", f"Data_S0{i}_Sess0{j}.csv")

            trials = read_csv_epochs(filename, tmin=tmin, tlen=tlen,
                                     data_max=data_max, data_min=data_min,
                                     use_channels_names=use_channels_names)
            for fb, trial in enumerate(trials, 1):
                key = f"S{i:02d}_Sess{j:02d}_FB{fb:03d}"
                label = int(labels[key])
                data.append((trial, label))
    return data


def read_kaggle_ern_test(path: str,
                         subjects: Sequence[int],
                         sessions: Sequence[int] = (1, 2, 3, 4, 5),
                         tmin: float = -0.7, tlen: float = 2.0,
                         data_max: Optional[float] = None, data_min: Optional[float] = None,
                         use_channels_names: Optional[Sequence[str]] = USE_CHANNELS_19):
    """Return list of (trial_tensor, label) for Kaggle-ERN Test split (labels from true_labels.csv)."""
    from tqdm import tqdm
    import pandas as pd

    labels = pd.read_csv(os.path.join(path, "KaggleERN", "true_labels.csv"))["label"].tolist()
    label_it = iter(labels)

    data = []
    for i in tqdm(subjects, desc="KaggleERN-Test"):
        for j in sessions:
            filename = os.path.join(path, "KaggleERN", "test", f"Data_S{i:02d}_Sess{j:02d}.csv")
            trials = read_csv_epochs(filename, tmin=tmin, tlen=tlen,
                                     data_max=data_max, data_min=data_min,
                                     use_channels_names=use_channels_names)
            for trial in trials:
                data.append((trial, int(next(label_it))))
    return data


# ============================ Checkpoints ============================

_STEP_PAT  = re.compile(r"(?:^|[^0-9])step=([0-9]+)")
_EPOCH_PAT = re.compile(r"(?:^|[^0-9])epoch=([0-9]+)")

# Robust key=val parser and metric aliasing (handles 'validvalid_loss', etc.)
_NUM_RE = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
_KEYVAL_RE = re.compile(rf"([A-Za-z0-9_]+)=({_NUM_RE})")
_METRIC_ALIASES = {
    "val_loss":   ["val_loss", "valid_loss", "validvalid_loss"],
    "valid_loss": ["valid_loss", "val_loss", "validvalid_loss"],
}

def _exists_ok(p: Path) -> bool:
    try:
        return p.is_file() and p.exists()
    except OSError:
        return False

def _extract_int(name: str, pat: re.Pattern) -> int:
    m = pat.search(name)
    return int(m.group(1)) if m else -1

def _mtime_key(p: Path) -> int:
    try:
        return p.stat().st_mtime_ns
    except FileNotFoundError:
        return -1

def _sort_key(p: Path):
    # prefer later steps/epochs, then newest mtime
    step  = _extract_int(p.name, _STEP_PAT)
    epoch = _extract_int(p.name, _EPOCH_PAT)
    return (step, epoch, _mtime_key(p))

def _parse_metric_from_name(name: str, metric: str) -> Optional[float]:
    """
    Extract a float metric from 'key=value' segments in filename.
    Tries exact key, then aliases, then any '*loss' if metric contains 'loss'.
    """
    pairs = {k: float(v) for k, v in _KEYVAL_RE.findall(name)}
    keys = [metric] + _METRIC_ALIASES.get(metric, [])
    for k in keys:
        if k in pairs:
            return pairs[k]
    if "loss" in metric:
        for k, v in pairs.items():
            if k.endswith("loss"):
                return v
    return None

def _first_of_ints(*vals: int) -> int:
    for v in vals:
        if v is not None and v >= 0:
            return v
    return -1

def get_checkpoint_path(base_dir: str, which: str | int = "last",
                        metric: str = "val_loss", mode: str = "min") -> Tuple[str, int]:
    """
    Inspect a Lightning log dir and pick a checkpoint.

    Args:
      base_dir: Path to your Lightning version dir (parent of 'checkpoints').
      which: 'last' | 'best' | specific epoch number (int or 'epoch=XX').
      metric: metric key to parse from filename for 'best' fallback.
      mode: 'min' or 'max' when comparing metric values.

    Returns:
      (checkpoint_path, epoch_number_or_step_if_epoch_missing)
    """
    ckpt_dir = (Path(base_dir).expanduser().resolve() / "checkpoints")
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    ckpts = [p for p in ckpt_dir.glob("*.ckpt") if _exists_ok(p)]
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files in {ckpt_dir}")

    # ---- LAST ----
    if str(which) == "last":
        explicit = [p for p in ckpts if p.name.startswith("last") or "last" in p.name]
        chosen = max(explicit, key=_mtime_key) if explicit else max(ckpts, key=_mtime_key)
        ep = _extract_int(chosen.name, _EPOCH_PAT)
        st = _extract_int(chosen.name, _STEP_PAT)
        return str(chosen), _first_of_ints(ep, st)

    # ---- BEST ----
    if str(which) == "best":
        scored: List[Tuple[Path, float]] = []
        for p in ckpts:
            val = _parse_metric_from_name(p.name, metric)
            if val is not None:
                scored.append((p, val))
        if scored:
            if mode.lower() == "min":
                chosen = min(scored, key=lambda t: t[1])[0]
            else:
                chosen = max(scored, key=lambda t: t[1])[0]
            ep = _extract_int(chosen.name, _EPOCH_PAT)
            st = _extract_int(chosen.name, _STEP_PAT)
            return str(chosen), _first_of_ints(ep, st)

        # Fallback to names containing 'best'
        best_named = [p for p in ckpts if "best" in p.name]
        if best_named:
            chosen = max(best_named, key=_mtime_key)
            ep = _extract_int(chosen.name, _EPOCH_PAT)
            st = _extract_int(chosen.name, _STEP_PAT)
            return str(chosen), _first_of_ints(ep, st)

        # Final fallback: newest file
        chosen = max(ckpts, key=_mtime_key)
        ep = _extract_int(chosen.name, _EPOCH_PAT)
        st = _extract_int(chosen.name, _STEP_PAT)
        return str(chosen), _first_of_ints(ep, st)

    # ---- SPECIFIC EPOCH ----
    ep_req = int(str(which).replace("epoch=", ""))
    matches = [p for p in ckpts if f"epoch={ep_req}" in p.name]
    if not matches:
        raise FileNotFoundError(f"No ckpts for epoch {ep_req} in {ckpt_dir}")
    chosen = max(matches, key=_mtime_key)
    ep = _extract_int(chosen.name, _EPOCH_PAT)
    st = _extract_int(chosen.name, _STEP_PAT)
    return str(chosen), _first_of_ints(ep, st)


__all__ = [
    # Kaggle
    "USE_CHANNELS_19", "CH_NAMES_KAGGLE_ERN",
    "read_csv_epochs", "read_kaggle_ern_train", "read_kaggle_ern_test",
    # ckpts
    "get_checkpoint_path",
]
