#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils import to_tensor


# ------------------------------ Dataset ------------------------------ #

class SleepEDFWindowDataset(Dataset):
    """
    Dataset of .pt windows saved as [C, T] (already µV and z-scored per window).
    Returns (x/scale_div, y), where y ∈ {0,1,2,3,4}.
    """
    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        *,
        scale_div: float = 1.0,   # SleepEDF windows are z-scored; keep 1.0 by default
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

_SUB_RE = re.compile(r"^s(\d+)_", re.IGNORECASE)

def _scan_sleepedf_nosplit(root: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Scan <root>/{0..4}/*.pt  (no-split layout).
    Returns files, labels, subject_ids (parsed from '*_sub{ID}.pt').
    """
    files, labels, subj_ids = [], [], []
    found_any = False
    for cls in map(str, range(5)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        found_any = True
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith(".pt"):
                continue
            m = _SUB_RE.search(fname)
            if not m:
                raise ValueError(f"Cannot parse subject id from filename: {fname} (expected '*_sub{{ID}}.pt')")
            sid = int(m.group(1))
            files.append(os.path.join(cls_dir, fname))
            labels.append(int(cls))
            subj_ids.append(sid)
    if not found_any:
        raise FileNotFoundError(f"No class dirs 0..4 found under {root}")
    if not files:
        raise FileNotFoundError(f"No .pt files found under {root}/{{0..4}}")
    return files, labels, subj_ids


def _scan_sleepedf_fold(root: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Back-compat: scan <root>/{TrainFold,ValidFold,TestFold}/{0..4}/*.pt
    """
    folds = ["TrainFold", "ValidFold", "TestFold"]
    files, labels, subj_ids = [], [], []
    found_any = False
    for fold in folds:
        fold_dir = os.path.join(root, fold)
        if not os.path.isdir(fold_dir):
            continue
        for cls in map(str, range(5)):
            cls_dir = os.path.join(fold_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            found_any = True
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.endswith(".pt"):
                    continue
                m = _SUB_RE.search(fname)
                if not m:
                    raise ValueError(f"Cannot parse subject id from filename: {fname}")
                sid = int(m.group(1))
                files.append(os.path.join(cls_dir, fname))
                labels.append(int(cls))
                subj_ids.append(sid)
    if not found_any:
        raise FileNotFoundError(
            f"No SleepEDF data found under {root} in either no-split (0..4) or fold layout."
        )
    return files, labels, subj_ids


def _scan_sleepedf_auto(root: str) -> Tuple[List[str], List[int], List[int]]:
    """
    Prefer no-split layout; fall back to fold layout if needed.
    """
    if any(os.path.isdir(os.path.join(root, str(c))) for c in range(5)):
        return _scan_sleepedf_nosplit(root)
    return _scan_sleepedf_fold(root)


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
    Works with no-split layout: <root>/{0..4}/*.pt
    (Also auto-detects old TrainFold/ValidFold/TestFold layout.)

    Split modes:

    - subject_explicit:
        • Provide any of: train_subject_ids, val_subject_ids, test_subject_ids (subject numbers).
        • Any subject not listed goes to train by default.
        • Sets must be disjoint.

    - mixed:
        • Provide test_subject_ids (int or list[int]) for subject-wise test.
        • Provide val_ratio in (0,1): sample-wise val split within remaining subjects.
        • Optional stratified_val (default True).

    Required params:
      - datasets_dir: str (root path)
      - batch_size:   int

    Optional params:
      - split_mode: 'subject_explicit' | 'mixed' (default 'subject_explicit')
      - train_subject_ids, val_subject_ids, test_subject_ids
      - val_ratio: 0.1 (mixed)
      - stratified_val: True (mixed)
      - seed: 7
      - num_workers: 0, pin_memory: True, drop_last: False, shuffle_train: True
      - scale_div: 1.0 (windows already z-scored)
      - transform: callable on loaded tensor
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

        # mixed and subject_random
        self.val_ratio = float(getattr(params, "val_ratio", 0.1))
        self.stratified_val = bool(getattr(params, "stratified_val", True))

        # loader knobs
        self.num_workers = int(getattr(params, "num_workers", 0))
        self.pin_memory  = bool(getattr(params, "pin_memory", True))
        self.drop_last   = bool(getattr(params, "drop_last", False))
        self.shuffle_train = bool(getattr(params, "shuffle_train", True))
        self.scale_div = float(getattr(params, "scale_div", 1.0))
        self.transform = getattr(params, "transform", None)

    def _make_loader(self, ds: SleepEDFWindowDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=SleepEDFWindowDataset.collate,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if shuffle else False,
        )

    def get_data_loader(self) -> Dict[str, DataLoader]:
        files, labels, subj_ids = _scan_sleepedf_auto(self.root)
        uniq_subjects = sorted(set(subj_ids))
        print(f"[SleepEDF] discovered subjects: {uniq_subjects}")

        if self.split_mode == "subject_explicit":
            tr_set = set(int(s) for s in (self.train_subject_ids or []))
            va_set = set(int(s) for s in (self.val_subject_ids or []))
            te_set = set(int(s) for s in (self.test_subject_ids or []))

            if not tr_set:
                tr_set = set(uniq_subjects) - va_set - te_set

            if (tr_set & va_set) or (tr_set & te_set) or (va_set & te_set):
                raise ValueError("train/val/test subject sets must be disjoint.")
            missing = (tr_set | va_set | te_set) - set(uniq_subjects)
            if missing:
                raise ValueError(f"Subjects not found in dataset: {sorted(missing)}")

            tr_files, tr_labels = _subset_by_subject(files, labels, subj_ids, sorted(tr_set))
            va_files, va_labels = _subset_by_subject(files, labels, subj_ids, sorted(va_set))
            te_files, te_labels = _subset_by_subject(files, labels, subj_ids, sorted(te_set))

            print(f"[SleepEDF] train={len(tr_files)} | val={len(va_files)} | test={len(te_files)}")
            loaders: Dict[str, DataLoader] = {
                "train": self._make_loader(
                    SleepEDFWindowDataset(tr_files, tr_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=self.shuffle_train
                ),
                "test": self._make_loader(
                    SleepEDFWindowDataset(te_files, te_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
            }
            if va_files:
                loaders["val"] = self._make_loader(
                    SleepEDFWindowDataset(va_files, va_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                )
            return loaders

        elif self.split_mode == "mixed":
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
                print(f"[SleepEDF] train={len(pool_files)} | val=0 | test={len(te_files)}")
                return {
                    "train": self._make_loader(
                        SleepEDFWindowDataset(pool_files, pool_labels, scale_div=self.scale_div, transform=self.transform),
                        shuffle=self.shuffle_train
                    ),
                    "test": self._make_loader(
                        SleepEDFWindowDataset(te_files, te_labels, scale_div=self.scale_div, transform=self.transform),
                        shuffle=False
                    ),
                }

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

            print(f"[SleepEDF] train={len(tr_files)} | val={len(va_files)} | test={len(te_files)}")
            return {
                "train": self._make_loader(
                    SleepEDFWindowDataset(tr_files, tr_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=self.shuffle_train
                ),
                "val": self._make_loader(
                    SleepEDFWindowDataset(va_files, va_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
                "test": self._make_loader(
                    SleepEDFWindowDataset(te_files, te_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
            }
            
        elif self.split_mode == "subject_random":
            if self.test_subject_ids is None:
                raise ValueError("subject_random split requires `test_subject_ids` (int or list of ints).")
            te_ids = self.test_subject_ids if isinstance(self.test_subject_ids, (list, tuple)) else [self.test_subject_ids]
            te_ids = sorted(int(s) for s in te_ids)
            missing = set(te_ids) - set(uniq_subjects)
            if missing:
                raise ValueError(f"Test subjects not found: {sorted(missing)}")
            te_files, te_labels = _subset_by_subject(files, labels, subj_ids, te_ids)
            pool_subjects = sorted(set(subj_ids) - set(te_ids))
            if not pool_subjects:
                raise ValueError("No subjects remain for train/val after test split.")
            rng = random.Random(self.seed)
            rng.shuffle(pool_subjects)
            n_val = max(1, int(round(len(pool_subjects) * self.val_ratio)))
            va_subjects = pool_subjects[:n_val]
            tr_subjects = pool_subjects[n_val:]
            tr_files, tr_labels = _subset_by_subject(files, labels, subj_ids, tr_subjects)
            va_files, va_labels = _subset_by_subject(files, labels, subj_ids, va_subjects)
            print(f"[SleepEDF] train={len(tr_files)} (subjects {tr_subjects}) | val={len(va_files)} (subjects {va_subjects}) | test={len(te_files)} (subjects {te_ids})")
            loaders: Dict[str, DataLoader] = {
                "train": self._make_loader(
                    SleepEDFWindowDataset(tr_files, tr_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=self.shuffle_train
                ),
                "val": self._make_loader(
                    SleepEDFWindowDataset(va_files, va_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
                "test": self._make_loader(
                    SleepEDFWindowDataset(te_files, te_labels, scale_div=self.scale_div, transform=self.transform),
                    shuffle=False
                ),
            }
            return loaders

        else:
            raise ValueError(f"Unknown split_mode: {self.split_mode!r}")