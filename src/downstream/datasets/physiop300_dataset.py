#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.util import to_tensor


# ------------------------------ Dataset ------------------------------ #

class PhysioP300Dataset(Dataset):
    """
    Dataset of .pt tensors saved as [C, T] (µV), filename includes '_sub{ID}.pt'.
    Returns (x/scale_div, y) where y ∈ {0,1}.
    """
    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        *,
        scale_div: float = 100.0,
        transform=None,
    ):
        super().__init__()
        assert len(files) == len(labels), "files and labels must align"
        self.files = list(files)
        self.labels = list(labels)
        self.scale_div = float(scale_div)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        x = torch.load(self.files[idx])  # [C, T], float32
        if self.transform is not None:
            x = self.transform(x)
        x = (x / self.scale_div).contiguous()
        y = int(self.labels[idx])
        return x.numpy(), y  # keep collate consistent with to_tensor

    @staticmethod
    def collate(batch):
        xs = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        ys = np.array([b[1] for b in batch], dtype=np.int64)
        return to_tensor(xs), to_tensor(ys).long()


# --------------------------- Helper utilities -------------------------- #

_SUB_RE = re.compile(r"_sub(\d+)\.pt$", re.IGNORECASE)

def _scan_physio_p300(root: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Scan PhysioNetP300/{0,1} for .pt files.

    Returns:
      files  : list of absolute file paths
      labels : 0/1 per file
      subj_id: subject number (int) parsed from filename suffix '_sub{ID}.pt'
    """
    files: List[str] = []
    labels: List[int] = []
    subj_ids: List[int] = []

    for lab_dir, lab in (("0", 0), ("1", 1)):
        d = os.path.join(root, lab_dir)
        if not os.path.isdir(d):
            # allow missing class dir (but warn)
            print(f"[WARN] Missing class dir: {d}")
            continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".pt"):
                continue
            fpath = os.path.join(d, fname)
            m = _SUB_RE.search(fname)
            if not m:
                raise ValueError(f"Cannot parse subject id from filename: {fname} (expected '*_sub{{ID}}.pt')")
            sid = int(m.group(1))
            files.append(fpath)
            labels.append(lab)
            subj_ids.append(sid)

    if not files:
        raise FileNotFoundError(f"No .pt files found under {root}/{{0,1}}")
    return files, labels, subj_ids


def _subset_by_subject(
    files: Sequence[str], labels: Sequence[int], subj_ids: Sequence[int], chosen: Sequence[int]
) -> Tuple[List[str], List[int]]:
    chosen_set = set(int(s) for s in chosen)
    f, y = [], []
    for fp, lb, sid in zip(files, labels, subj_ids):
        if sid in chosen_set:
            f.append(fp); y.append(lb)
    return f, y


def _subset_excluding_subjects(
    files: Sequence[str], labels: Sequence[int], subj_ids: Sequence[int], excluded: Sequence[int]
) -> Tuple[List[str], List[int]]:
    excl = set(int(s) for s in excluded)
    f, y = [], []
    for fp, lb, sid in zip(files, labels, subj_ids):
        if sid not in excl:
            f.append(fp); y.append(lb)
    return f, y


def _stratified_idx_split(labels: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Return (train_idx, val_idx) from per-class split on indices."""
    rng = random.Random(seed)
    by_cls: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    tr, va = [], []
    for idxs in by_cls.values():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            k = 1 if val_ratio >= 1.0 else 0
        else:
            k = max(1, int(round(n * val_ratio)))
            k = min(k, n - 1)
        va.extend(idxs[:k]); tr.extend(idxs[k:])
    rng.shuffle(tr); rng.shuffle(va)
    return tr, va


# ------------------------------- Loader -------------------------------- #

class LoadDataset:
    """
    Split modes:

    - subject_explicit:
        • Provide any of: train_subject_ids, val_subject_ids, test_subject_ids (subject **numbers**, e.g., 2,3,4).
        • Any subject not listed goes to train by default.
        • Sets must be disjoint.

    - mixed:
        • Provide test_subject_ids (int or list[int]) for subject-wise test.
        • Provide val_ratio in (0,1). Performs sample-wise val split within remaining subjects.
        • Optional stratified_val (default True).

    Required params:
      - datasets_dir: str (root with subfolders '0' and '1')
      - batch_size:   int

    Optional params (defaults):
      - split_mode: 'subject_explicit' | 'mixed' (default 'subject_explicit')
      - train_subject_ids, val_subject_ids, test_subject_ids
      - val_ratio: 0.1 (mixed)
      - stratified_val: True (mixed)
      - seed: 7
      - num_workers: 0, pin_memory: True, drop_last: False, shuffle_train: True, scale_div: 100.0
      - transform: optional callable applied to each tensor before scaling
    """

    def __init__(self, params):
        self.p = params
        self.root = params.datasets_dir
        self.batch_size = params.batch_size

        self.split_mode = getattr(params, "split_mode", "subject_explicit")
        self.seed = int(getattr(params, "seed", 7))

        # subject_explicit
        self.train_subject_ids = getattr(params, "train_subject_ids", None)
        self.val_subject_ids   = getattr(params, "val_subject_ids", None)
        self.test_subject_ids  = getattr(params, "test_subject_ids", None)

        # mixed
        self.val_ratio = float(getattr(params, "val_ratio", 0.1))
        self.stratified_val = bool(getattr(params, "stratified_val", True))

        # loader knobs
        self.num_workers = int(getattr(params, "num_workers", 0))
        self.pin_memory  = bool(getattr(params, "pin_memory", True))
        self.drop_last   = bool(getattr(params, "drop_last", False))
        self.shuffle_train = bool(getattr(params, "shuffle_train", True))
        self.scale_div = float(getattr(params, "scale_div", 100.0))
        self.transform = getattr(params, "transform", None)

    def _make_loader(self, ds: PhysioP300Dataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=PhysioP300Dataset.collate,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if shuffle else False,
        )

    def get_data_loader(self) -> Dict[str, DataLoader]:
        files, labels, subj_ids = _scan_physio_p300(self.root)
        uniq_subjects = sorted(set(subj_ids))
        print(f"[PhysioP300] discovered subjects: {uniq_subjects}")

        if self.split_mode == "subject_explicit":
            tr_set = set(int(s) for s in (self.train_subject_ids or []))
            va_set = set(int(s) for s in (self.val_subject_ids or []))
            te_set = set(int(s) for s in (self.test_subject_ids or []))

            # Any unlisted subject goes to train
            if not tr_set:
                tr_set = set(uniq_subjects) - va_set - te_set

            # Validate sets are disjoint and within discovered subjects
            if (tr_set & va_set) or (tr_set & te_set) or (va_set & te_set):
                raise ValueError("train/val/test subject sets must be disjoint.")
            missing = (tr_set | va_set | te_set) - set(uniq_subjects)
            if missing:
                raise ValueError(f"Subjects not found in dataset: {sorted(missing)}")

            tr_files, tr_labels = _subset_by_subject(files, labels, subj_ids, sorted(tr_set))
            va_files, va_labels = _subset_by_subject(files, labels, subj_ids, sorted(va_set))
            te_files, te_labels = _subset_by_subject(files, labels, subj_ids, sorted(te_set))

            print(f"[PhysioP300] train={len(tr_files)} | val={len(va_files)} | test={len(te_files)}")
            loaders: Dict[str, DataLoader] = {
                "train": self._make_loader(
                    PhysioP300Dataset(tr_files, tr_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=self.shuffle_train
                ),
                "test": self._make_loader(
                    PhysioP300Dataset(te_files, te_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
            }
            if va_files:
                loaders["val"] = self._make_loader(
                    PhysioP300Dataset(va_files, va_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                )
            return loaders

        elif self.split_mode == "mixed":
            # Require test subjects
            if self.test_subject_ids is None:
                raise ValueError("mixed split requires `test_subject_ids` (int or list of ints).")
            te_ids = self.test_subject_ids if isinstance(self.test_subject_ids, (list, tuple)) else [self.test_subject_ids]
            te_ids = sorted(int(s) for s in te_ids)
            missing = set(te_ids) - set(uniq_subjects)
            if missing:
                raise ValueError(f"Test subjects not found: {sorted(missing)}")

            te_files, te_labels = _subset_by_subject(files, labels, subj_ids, te_ids)
            pool_files, pool_labels = _subset_excluding_subjects(files, labels, subj_ids, te_ids)

            if not (0.0 < self.val_ratio < 1.0):
                # no val split; everything else to train
                print(f"[PhysioP300] train={len(pool_files)} | val=0 | test={len(te_files)}")
                return {
                    "train": self._make_loader(
                        PhysioP300Dataset(pool_files, pool_labels, scale_div=self.scale_div, transform=self.transform),
                        shuffle=self.shuffle_train
                    ),
                    "test": self._make_loader(
                        PhysioP300Dataset(te_files, te_labels, scale_div=self.scale_div, transform=self.transform),
                        shuffle=False
                    ),
                }

            # sample-wise split within pool
            if self.stratified_val:
                idx_tr, idx_va = _stratified_idx_split(pool_labels, self.val_ratio, self.seed)
            else:
                rng = random.Random(self.seed)
                idxs = list(range(len(pool_files))); rng.shuffle(idxs)
                k = max(1, int(round(len(pool_files) * self.val_ratio)))
                idx_va, idx_tr = idxs[:k], idxs[k:]

            tr_files = [pool_files[i] for i in idx_tr]
            tr_labels = [pool_labels[i] for i in idx_tr]
            va_files = [pool_files[i] for i in idx_va]
            va_labels = [pool_labels[i] for i in idx_va]

            print(f"[PhysioP300] train={len(tr_files)} | val={len(va_files)} | test={len(te_files)}")
            return {
                "train": self._make_loader(
                    PhysioP300Dataset(tr_files, tr_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=self.shuffle_train
                ),
                "val": self._make_loader(
                    PhysioP300Dataset(va_files, va_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
                "test": self._make_loader(
                    PhysioP300Dataset(te_files, te_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
            }

        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode!r}")
