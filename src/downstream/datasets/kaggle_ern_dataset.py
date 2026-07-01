#!/usr/bin/env python3
from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from downstream.utils import to_tensor 

# ------------------------------ Dataset ------------------------------ #

class CustomDataset(Dataset):
    """KaggleERN .npz dataset: {'data': np.ndarray[B,C,T], 'labels': np.ndarray[B]}. Scales by /10."""
    def __init__(self, npz_file: str, scale_div: float = 10.0):
        super().__init__()
        self.npz_file = npz_file
        self.scale_div = float(scale_div)
        data = np.load(npz_file, allow_pickle=True)
        self.data = [torch.tensor(d, dtype=torch.float32) / self.scale_div for d in data['data']]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in data['labels']]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx]  # [C, T]
        y = self.labels[idx]  # int
        return x, y

    @staticmethod
    def collate(batch):
        x = torch.stack([b[0] for b in batch], dim=0)  # [B, C, T]
        y = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        return x, y

# --------------------------- Small utilities -------------------------- #

def _normalize_ids(ids: Optional[list], n_subj: int) -> List[int]:
    """Accept 0- or 1-based indices. Return sorted unique 0-based."""
    if not ids:
        return []
    ids = list(dict.fromkeys(int(i) for i in ids))  # dedupe, keep order
    if all(1 <= i <= n_subj for i in ids) and not any(i == 0 for i in ids):
        ids = [i - 1 for i in ids]
    bad = [i for i in ids if i < 0 or i >= n_subj]
    if bad:
        raise ValueError(f"Subject ids out of range [0..{n_subj-1}]: {bad}")
    return sorted(ids)

# ------------------------------- Loader -------------------------------- #

class LoadDataset:
    """
    KaggleERN dataset loader for preprocessed .npz files with a fixed 5-fold split.

    Required params:
      - datasets_dir (path to processed .npz files), batch_size, fold (int, 1-5)

    Optional params (defaults):
      - seed: 7
      - num_workers: 0, pin_memory: True, drop_last: False, shuffle_train: True, scale_div: 10.0
    """

    def __init__(self, params):
        self.p = params
        self.db_path = params.datasets_dir
        self.batch_size = params.batch_size
        self.fold = getattr(params, "fold", None)
        if self.fold is None or not isinstance(self.fold, int) or self.fold < 1 or self.fold > 5:
            raise ValueError("fold must be an integer in [1, 5]")

        self.seed = int(getattr(params, "seed", 7))
        self.num_workers = int(getattr(params, "num_workers", 0))
        self.pin_memory = bool(getattr(params, "pin_memory", True))
        self.drop_last = bool(getattr(params, "drop_last", False))
        self.shuffle_train = bool(getattr(params, "shuffle_train", True))
        self.scale_div = float(getattr(params, "scale_div", 10.0))

    def get_data_loader(self) -> Dict[str, DataLoader]:
        train_file = f"{self.db_path}/fold{self.p.fold}_train.npz"
        valid_file = f"{self.db_path}/fold{self.p.fold}_valid.npz"
        test_file = f"{self.db_path}/test.npz"

        for f in [train_file, test_file]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"Missing required .npz file: {f}")
        if not os.path.exists(valid_file):
            print(f"[KaggleERN] Warning: Validation file {valid_file} not found, proceeding without validation.")

        loaders: Dict[str, DataLoader] = {
            "train": self._make_loader(CustomDataset(train_file, self.scale_div), shuffle=self.shuffle_train),
            "test": self._make_loader(CustomDataset(test_file, self.scale_div), shuffle=False),
        }
        if os.path.exists(valid_file):
            loaders["val"] = self._make_loader(CustomDataset(valid_file, self.scale_div), shuffle=False)

        print(f"[KaggleERN] fold={self.p.fold} | train={len(loaders['train'].dataset)} | "
              f"val={len(loaders['val'].dataset) if 'val' in loaders else 0} | "
              f"test={len(loaders['test'].dataset)}")
        return loaders

    def _make_loader(self, ds: CustomDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=CustomDataset.collate,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if shuffle else False,
        )