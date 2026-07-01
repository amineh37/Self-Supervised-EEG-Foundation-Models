#!/usr/bin/env python3
from __future__ import annotations

import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset, DataLoader

from downstream.utils import to_tensor 


# ----------------------------- helpers ----------------------------- #

def _as_int_label(v) -> int:
    """Accept int, list/tuple, or numpy array; return python int."""
    if isinstance(v, np.ndarray):
        if v.size == 0:
            raise ValueError("Empty numpy label array")
        return int(v.flat[0])
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            raise ValueError("Empty list/tuple label")
        return int(v[0])
    return int(v)


def _list_pickles(d: Path) -> List[Path]:
    """List *.pkl files in a directory, sorted by name."""
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".pkl"])


# ------------------------- subject helpers ------------------------- #

def _subject_from_name(fname: str, subject_regex: Optional[str]) -> str:
    """
    Infer subject ID from filename.
    Default: prefix before first underscore (common for TUEV exports).
    Or pass a custom regex containing a named group (?P<sid>...).
    """
    stem = Path(fname).stem             # e.g., SUBJ001_sessionA_12
    stem = re.sub(r'_\d+$', '', stem)   # drop trailing _<eventidx> if present
    if subject_regex:
        m = re.search(subject_regex, stem)
        if not m or 'sid' not in m.groupdict():
            raise ValueError(f"subject_regex didn't match or lacks (?P<sid>): {subject_regex} on {stem}")
        return m.group('sid')
    return stem.split('_')[0]


def _subject_split_indices(
    files: Sequence[Path],
    val_ratio: float,
    seed: int,
    strategy: str,                 # 'random' or 'sorted'
    subject_regex: Optional[str],
) -> Tuple[List[int], List[int]]:
    """
    Return (train_idx, val_idx) by grouping on subject ID so all
    windows from a subject stay in the same fold.
    """
    buckets: Dict[str, List[int]] = {}
    for i, f in enumerate(files):
        sid = _subject_from_name(Path(f).name, subject_regex)
        buckets.setdefault(sid, []).append(i)

    sids = list(buckets.keys())
    if strategy == 'sorted':
        sids.sort()
    else:
        rng = random.Random(seed)
        rng.shuffle(sids)

    # number of subjects to place into val
    k = max(1, int(round(len(sids) * val_ratio)))
    k = min(k, max(1, len(sids) - 1))  # keep at least one subject for train
    val_sids = set(sids[:k])

    train_idx, val_idx = [], []
    for sid, idxs in buckets.items():
        (val_idx if sid in val_sids else train_idx).extend(idxs)

    rnd = random.Random(seed)
    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    return train_idx, val_idx


# ----------------------------- dataset ----------------------------- #

class CustomDataset(Dataset):
    """
    TUEV pickled sample loader.

    Each pickle has:
      - 'signal': np.ndarray [C, S] in µV
      - 'label' : scalar or 1-element container (1-based); converted to 0-based

    Default preprocessing:
      - divide by `scale_div` (100.0 by default)
    """

    def __init__(
        self,
        data_root: Path | str,
        files: Sequence[Path | str],
        *,
        scale_div: float = 100.0,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.files = [Path(f) for f in files]
        self.scale_div = float(scale_div)

    def __len__(self) -> int:
        return len(self.files)

    def _open_path(self, file: Path) -> Path:
        # If `file` is absolute, use it; otherwise resolve under `data_root`
        return file if file.is_absolute() else (self.data_root / file)

    def _load_item(self, file: Path) -> Tuple[np.ndarray, int]:
        fp = self._open_path(file)
        with open(fp, "rb") as f:
            d = pickle.load(f)

        # labels are 1-based in your preproc -> convert to 0-based
        y = _as_int_label(d["label"]) - 1

        x = np.asarray(d["signal"], dtype=np.float32)
        if self.scale_div != 1.0:
            # clip_uv = 300.0
            # x = np.clip(x, -clip_uv, clip_uv).astype(np.float32)
            x = x / self.scale_div

        # Expect shape [C, S]
        if x.ndim != 2:
            raise ValueError(f"Expected 'signal' to be 2D [C,S], got {x.shape} in {fp}")

        return x, y

    def __getitem__(self, idx: int):
        return self._load_item(self.files[idx])

    @staticmethod
    def collate(batch):
        xs = np.stack([b[0] for b in batch], axis=0).astype(np.float32)  # [B, C, S]
        ys = np.array([b[1] for b in batch], dtype=np.int64)
        return to_tensor(xs), to_tensor(ys).long()


# -------------------------- split utilities ------------------------- #

def _read_labels_for_files(folder: Path, files: List[Path]) -> List[int]:
    labels: List[int] = []
    for fname in files:
        fp = fname if fname.is_absolute() else (folder / fname)
        with open(fp, "rb") as f:
            d = pickle.load(f)
        labels.append(_as_int_label(d["label"]) - 1)
    return labels


def _stratified_split_indices(
    labels: Sequence[int],
    val_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """
    Return (train_idx, val_idx) using a simple stratified per-class split.
    Guarantees at least one train item per class when possible.
    """
    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_class.setdefault(int(y), []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    for _, idxs in by_class.items():
        rng.shuffle(idxs)
        n = len(idxs)
        if n == 1:
            if val_ratio >= 1.0:
                val_idx.extend(idxs)
            else:
                train_idx.extend(idxs)
            continue
        k = max(1, int(round(n * val_ratio)))
        k = min(k, n - 1)  # keep at least one for train
        val_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# ------------------------------ loader ------------------------------ #

class LoadDataset:
    """
    TUEV loader:
      - train from <datasets_dir>/train
      - test  from <datasets_dir>/eval  (use official eval set as test)
      - optional validation carved from train via val_ratio in [0,1)

    Required: params.datasets_dir (str|Path), params.batch_size (int)
    Optional (with defaults):
      - num_workers: int = 0
      - pin_memory: bool = True
      - drop_last: bool = False
      - val_ratio: float = 0.0
      - val_seed: int = 42
      - stratified_val: bool = True        # used when val_group == "class"
      - shuffle_train: bool = True
      - scale_div: float = 100.0
      - val_group: str = "class"           # "class", "subject_random", "subject_sorted"
      - subject_regex: str = r'^(?P<sid>[^_]+)'
    """

    def __init__(self, params):
        self.params = params
        self.datasets_dir = Path(params.datasets_dir)

        # Optional knobs
        self.num_workers: int = getattr(params, "num_workers", 0)
        self.pin_memory: bool = getattr(params, "pin_memory", True)
        self.drop_last: bool = getattr(params, "drop_last", False)
        self.val_ratio: float = float(getattr(params, "val_ratio", 0.0))
        self.val_seed: int = int(getattr(params, "val_seed", 42))
        self.stratified_val: bool = bool(getattr(params, "stratified_val", True))
        self.shuffle_train: bool = bool(getattr(params, "shuffle_train", True))
        self.scale_div: float = float(getattr(params, "scale_div", 100.0))

        # New knobs for subject-wise splits
        self.val_group: str = getattr(params, "val_group", "class")
        self.subject_regex: Optional[str] = getattr(params, "subject_regex", r'^(?P<sid>[^_]+)')

    def _make_loader(self, ds: CustomDataset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.params.batch_size,
            collate_fn=ds.collate,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(self.drop_last if shuffle else False),
        )

    def get_data_loader(self) -> Dict[str, DataLoader]:
        train_dir = self.datasets_dir / "train"
        eval_dir  = self.datasets_dir / "eval"   # treated as test

        if not train_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {train_dir}")
        if not eval_dir.is_dir():
            raise FileNotFoundError(f"Missing directory: {eval_dir}")

        train_files: List[Path] = _list_pickles(train_dir)
        test_files:  List[Path] = _list_pickles(eval_dir)

        # ---------- optional validation split from train ----------
        has_val = 0.0 < self.val_ratio < 1.0
        if has_val:
            if self.val_group == "subject_random":
                tr_idx, va_idx = _subject_split_indices(
                    train_files, self.val_ratio, self.val_seed,
                    strategy="random", subject_regex=self.subject_regex
                )
            elif self.val_group == "subject_sorted":
                tr_idx, va_idx = _subject_split_indices(
                    train_files, self.val_ratio, self.val_seed,
                    strategy="sorted", subject_regex=self.subject_regex
                )
            elif self.stratified_val:
                labels = _read_labels_for_files(train_dir, train_files)
                tr_idx, va_idx = _stratified_split_indices(labels, self.val_ratio, self.val_seed)
            else:
                idxs = list(range(len(train_files)))
                rnd = random.Random(self.val_seed)
                rnd.shuffle(idxs)
                k = int(round(len(idxs) * self.val_ratio))
                va_idx, tr_idx = idxs[:k], idxs[k:]

            train_sel = [train_files[i] for i in tr_idx]
            val_sel   = [train_files[i] for i in va_idx]
        else:
            train_sel = train_files
            val_sel   = []

        # ---------- datasets ----------
        train_set = CustomDataset(train_dir, train_sel, scale_div=self.scale_div)
        test_set  = CustomDataset(eval_dir,  test_files, scale_div=self.scale_div)

        val_set: Optional[CustomDataset] = None
        if len(val_sel) > 0:
            val_set = CustomDataset(train_dir, val_sel, scale_div=self.scale_div)

        # ---------- logs ----------
        n_train, n_val, n_test = len(train_set), (len(val_set) if val_set else 0), len(test_set)
        print(f"[TUEV] train={n_train} | val={n_val} | test={n_test} | total(train+eval)={len(train_files)+len(test_files)}")

        # ---------- dataloaders ----------
        loaders: Dict[str, DataLoader] = {
            "train": self._make_loader(train_set, shuffle=self.shuffle_train),
            "test":  self._make_loader(test_set,  shuffle=False),
        }
        if val_set is not None:
            loaders["val"] = self._make_loader(val_set, shuffle=False)

        return loaders
