"""
Dataset builders/splitters for common EEG setups:
- LOSO (leave-one-subject-out) split helpers (subject-wise only)
- SUBJECT_SPLIT (random or explicit subject ids)
- SESSION_SPLIT (train/val/test by session)

Notes:
- Returns raw NumPy arrays (X, Y, S) by default.
- Optional temporal interpolation to a desired sequence length (target_sample).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple, Dict
from collections import defaultdict

import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split

# --------------------- low-level subject/session I/O ---------------------

def _load_sessions_for_subject(
    subj: int,
    data_path: str,
    sessions: Sequence[str] = ("train", "test"),
    *,
    target_sample: int = -1,   # <= 0 disables interpolation
    use_avg: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ss = [], [], []
    for sname in sessions:
        mat = sio.loadmat(os.path.join(data_path, f"sub{subj}_{sname}/Data.mat"))
        X = mat["x_data"]                      # [N,C,T]
        Y = mat["y_data"].reshape(-1)          # [N]
        X = _interp_np(X, target_sample, use_avg)
        xs.append(X); ys.append(Y); ss.append(np.full(len(Y), subj, dtype=np.int64))
    X = np.concatenate(xs, 0) if xs else np.empty((0,))
    Y = np.concatenate(ys, 0) if ys else np.empty((0,), dtype=np.int64)
    S = np.concatenate(ss, 0) if ss else np.empty((0,), dtype=np.int64)
    return X, Y, S

__all__ = [
    "_load_sessions_for_subject",
]
