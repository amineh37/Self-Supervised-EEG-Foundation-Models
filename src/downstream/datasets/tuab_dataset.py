import os
import pickle
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
# from downstream import utils
from utils import to_tensor
# from scipy import signal  # keep if you want resampling

# ------------------------------ Dataset ------------------------------ #

class TUABWindowDataset(Dataset):
    """
    Dataset of .pkl windows saved as {'X': np.ndarray[C, T], 'y': int} (µV, not z-scored).
    Returns (x/scale_div, y), where y ∈ {0, 1}.
    """
    def __init__(
        self,
        files: Sequence[str],
        labels: Sequence[int],
        *,
        scale_div: float = 100.0,  # TUAB data in µV; scale to reasonable range
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
        with open(self.files[idx], "rb") as fh:
            pair = pickle.load(fh)
        x = torch.from_numpy(pair["X"].astype(np.float32))  # [C, T], float32
        if self.transform is not None:
            x = self.transform(x)
        x = (x / self.scale_div).contiguous()
        y = int(pair["y"])
        return x.numpy(), y  # keep collate consistent with to_tensor

    @staticmethod
    def collate(batch):
        xs = np.stack([b[0] for b in batch], axis=0).astype(np.float32)
        ys = np.array([b[1] for b in batch], dtype=np.int64)
        return to_tensor(xs), to_tensor(ys).long()

# --------------------------- Helper utilities -------------------------- #

_SUB_RE = re.compile(r"^(.*?)_s\d+_", re.IGNORECASE)


def _scan_tuab(root: str, split: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan <root>/split/{0,1}/*.pkl.
    Returns (files, labels, subject_prefixes) where subject_prefix is parsed
    from 'PREFIX_s###_' in the filename (e.g., 'aaaaagvr_s005_t000_67.pkl' -> 'aaaaagvr').
    """
    files: List[str] = []
    labels: List[int] = []
    subj_ids: List[str] = []
    found_any = False

    for cls in ("0", "1"):
        cls_dir = os.path.join(root, split, cls)
        # print("cls_dir: ",cls_dir )
        if not os.path.isdir(cls_dir):
            continue
        found_any = True

        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith(".pkl"):
                continue
            m = _SUB_RE.match(fname)  # anchored at start
            if not m:
                raise ValueError(
                    f"Cannot parse subject prefix from filename: {fname} "
                    f"(expected 'PREFIX_s###_*.pkl')"
                )
            sid = m.group(1)  # the PREFIX (string)
            files.append(os.path.join(cls_dir, fname))
            labels.append(int(cls))
            subj_ids.append(sid)

    if not found_any:
        raise FileNotFoundError(f"No class dirs 0 or 1 found under {root}")
    if not files:
        raise FileNotFoundError(f"No .pkl files found under {root}/{{0,1}}")

    return files, labels, subj_ids


def _subset_by_subject(
    files: Sequence[str], labels: Sequence[int], subj_ids: Sequence[str], chosen: Sequence[str]
) -> Tuple[List[str], List[int]]:
    chosen_set = set(s for s in chosen)
    f, y = [], []
    for fp, lb, sid in zip(files, labels, subj_ids):
        if sid in chosen_set:
            f.append(fp); y.append(lb)
    return f, y


def _subset_excluding_subjects(
    files: Sequence[str], labels: Sequence[int], subj_ids: Sequence[str], excluded: Sequence[str]
) -> Tuple[List[str], List[int]]:
    excl = set(s for s in excluded)
    f, y = [], []
    for fp, lb, sid in zip(files, labels, subj_ids):
        if sid not in excl:
            f.append(fp); y.append(lb)
    return f, y


# def _stratified_idx_split(labels: Sequence[int], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
#     """Return (train_idx, val_idx) from per-class split on indices."""
#     rng = random.Random(seed)
#     by_cls: Dict[int, List[int]] = {}
#     for i, y in enumerate(labels):
#         by_cls.setdefault(int(y), []).append(i)
#     tr, va = [], []
#     for idxs in by_cls.values():
#         rng.shuffle(idxs)
#         n = len(idxs)
#         if n == 1:
#             k = 1 if val_ratio >= 1.0 else 0
#         else:
#             k = max(1, int(round(n * val_ratio)))
#             k = min(k, n - 1)
#         va.extend(idxs[:k]); tr.extend(idxs[k:])
#     rng.shuffle(tr); rng.shuffle(va)
#     return tr, va


# ------------------------------ Loader ------------------------------ #

class LoadTest:
    """
    Loads TUAB EEG test data from <root>/eval/{0,1}/*.pkl.

    Required params:
      - datasets_dir: str (root path to eval folder)
      - batch_size: int

    Optional params:
      - num_workers: int (default 0)
      - pin_memory: bool (default True)
      - drop_last: bool (default False)
      - shuffle_test: bool (default False)
      - scale_div: float (default 100.0)
      - transform: callable on loaded tensor
    """

    def __init__(self, params):
        self.p = params
        self.root = params.datasets_dir  # Should point to /processed/eval
        self.batch_size = params.batch_size
        self.num_workers = int(getattr(params, "num_workers", 0))
        self.pin_memory = bool(getattr(params, "pin_memory", True))
        self.drop_last = bool(getattr(params, "drop_last", False))
        self.shuffle_test = bool(getattr(params, "shuffle_test", False))
        self.scale_div = float(getattr(params, "scale_div", 100.0))
        self.transform = getattr(params, "transform", None)

    def _make_loader(self, ds: TUABWindowDataset) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=TUABWindowDataset.collate,
            shuffle=self.shuffle_test,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def get_data_loader(self) -> Dict[str, DataLoader]:
        files, labels, subj_ids = _scan_tuab(self.root, "eval")
        uniq_subjects = sorted(set(subj_ids))
        # print(f"[TUAB Test] discovered subjects: {uniq_subjects}")
        # print(f"[TUAB Test] test={len(files)}")

        return self._make_loader(
                TUABWindowDataset(files, labels, scale_div=self.scale_div, transform=self.transform)
            )


# ------------------------------- LoadTrainValDataset ------------------------------- #

class LoadTrainValDataset:
    """
    Loads TUAB EEG train data from <root>/train/{0,1}/*.pkl and splits into train/val
    based on val_subject_ids.

    Split mode:
    - subject_random: Use val_subject_ids for validation; remaining subjects for train.

    Required params:
      - datasets_dir: str (root path to processed folder)
      - batch_size: int
      - val_subject_ids: list[int] (subjects for validation)

    Optional params:
      - split_mode: str (default 'subject_random')
      - seed: int (default 7)
      - num_workers: int (default 0)
      - pin_memory: bool (default True)
      - drop_last: bool (default False)
      - shuffle_train: bool (default True)
      - scale_div: float (default 100.0)
      - transform: callable on loaded tensor
    """

    def __init__(self, params):
        self.p = params
        self.root = params.datasets_dir  # Should point to /processed
        self.batch_size = params.batch_size
        self.split_mode = getattr(params, "split_mode", "subject_random")
        self.val_subject_ids = params.val_subject_ids
        self.seed = int(getattr(params, "seed", 7))
        self.num_workers = int(getattr(params, "num_workers", 0))
        self.pin_memory = bool(getattr(params, "pin_memory", True))
        self.drop_last = bool(getattr(params, "drop_last", False))
        self.shuffle_train = bool(getattr(params, "shuffle_train", True))
        self.scale_div = float(getattr(params, "scale_div", 100.0))
        self.transform = getattr(params, "transform", None)

    def _make_loader(self, ds: TUABWindowDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            collate_fn=TUABWindowDataset.collate,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last if shuffle else False,
        )

    def get_data_loader(self) -> Dict[str, DataLoader]:
        files, labels, subj_ids = _scan_tuab(self.root, "train")
        uniq_subjects = sorted(set(subj_ids))
        # print(f"[TUAB TrainVal] discovered subjects: {uniq_subjects}")

        if self.split_mode != "subject_random":
            raise ValueError(f"Only 'subject_random' split_mode is supported, got {self.split_mode!r}")

        va_ids = sorted(s for s in (self.val_subject_ids if isinstance(self.val_subject_ids, (list, tuple)) else [self.val_subject_ids]))
        missing = set(va_ids) - set(uniq_subjects)
        if missing:
            raise ValueError(f"Validation subjects not found: {sorted(missing)}")

        va_files, va_labels = _subset_by_subject(files, labels, subj_ids, va_ids)
        tr_files, tr_labels = _subset_excluding_subjects(files, labels, subj_ids, va_ids)
        tr_subjects = sorted(set(subj_ids) - set(va_ids))

        if not tr_files:
            raise ValueError("No subjects remain for train after validation split.")

        # print(f"[TUAB TrainVal] train={len(tr_files)} (subjects {tr_subjects}) | val={len(va_files)} (subjects {va_ids})")
        return {
            "train": self._make_loader(
                TUABWindowDataset(tr_files, tr_labels, scale_div=self.scale_div, transform=self.transform),
                shuffle=self.shuffle_train
            ),
            "val": self._make_loader(
                TUABWindowDataset(va_files, va_labels, scale_div=self.scale_div, transform=self.transform),
                shuffle=False
            ),
        }