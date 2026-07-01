# downstream/datasets/physiomi_kfold.py
import json
import pickle
import random
from typing import Dict, List, Tuple, Optional

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from downstream.utils import to_tensor


# ---------------------------- Core dataset ----------------------------
class CustomDataset(Dataset):
    """
    Reads (C, T) EEG samples and integer labels from LMDB.
    Initialized with a list of keys (filtered by subject-wise split).
    """
    def __init__(self, data_dir: str, keys: List[str], scale_div: float = 100.0):
        super().__init__()
        self.env = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False, subdir=True)
        self.keys = keys
        self.scale_div = float(scale_div)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        k = self.keys[idx]
        with self.env.begin(write=False) as txn:
            pair = pickle.loads(txn.get(k.encode()))
        x = pair["sample"] / self.scale_div  # (C, T)
        y = pair["label"]
        return x, y

    @staticmethod
    def collate(batch):
        xs = np.array([b[0] for b in batch])  # (B, C, T)
        ys = np.array([b[1] for b in batch])
        return to_tensor(xs).float(), to_tensor(ys).long()


# ---------------------------- Split helpers ----------------------------
def _extract_subject(key: str) -> str:
    # Keys like "S001R04-3" -> "S001"
    return key.split("R", 1)[0]


def _subjects_and_buckets(all_keys: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    subj2keys: Dict[str, List[str]] = {}
    for k in all_keys:
        s = _extract_subject(k)
        subj2keys.setdefault(s, []).append(k)
    subjects = sorted(subj2keys.keys())
    return subjects, subj2keys


def _make_5_folds(subjects: List[str], seed: int) -> List[List[str]]:
    """
    Deterministic shuffle, then split into 5 folds with sizes 22/22/22/22/21 for 109 subjects.
    Works for any N: as even as possible with remainder distributed to early folds.
    """
    rng = random.Random(seed)
    subs = subjects[:]
    rng.shuffle(subs)
    n = len(subs)
    base, rem = divmod(n, 5)
    sizes = [base + (1 if i < rem else 0) for i in range(5)]
    folds, i = [], 0
    for sz in sizes:
        folds.append(subs[i:i + sz])
        i += sz
    return folds


def _read_all_keys(lmdb_dir: str) -> List[str]:
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=True, meminit=False, subdir=True)
    try:
        with env.begin(write=False) as txn:
            keys_dict = pickle.loads(txn.get(b"__keys__"))
    finally:
        env.close()
    all_keys: List[str] = []
    for bucket in ("train", "eval", "val", "test"):
        if bucket in keys_dict:
            all_keys.extend(keys_dict[bucket])
    # dedup (in case of overlap)
    return sorted(set(all_keys))


def _folds_from_json(json_path: str) -> Tuple[List[List[str]], Optional[int]]:
    with open(json_path, "r") as f:
        spec = json.load(f)
    folds = [spec["folds"][str(i)] for i in range(5)]
    fixed_test_idx = spec.get("test_fold_index", None)
    return folds, fixed_test_idx


def build_subjectwise_split(
    data_dir: str,
    *,
    seed: int = 7,
    test_fold_index: int = 4,
    cv_fold_index: int = 0,
    folds_json: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns (train_keys, val_keys, test_keys) for ONE CV round.
    - test fold is fixed by test_fold_index
    - among the remaining 4 folds, cv_fold_index selects the validation fold
    - if folds_json is provided, use it (and override test_fold_index if present inside)
    """
    all_keys = _read_all_keys(data_dir)
    subjects, subj2keys = _subjects_and_buckets(all_keys)

    if folds_json:
        folds, fixed_idx = _folds_from_json(folds_json)
        if fixed_idx is not None:
            test_fold_index = int(fixed_idx)
    else:
        folds = _make_5_folds(subjects, seed=seed)

    assert 0 <= test_fold_index < 5, "test_fold_index must be in [0,4]"
    non_test_ids = [i for i in range(5) if i != test_fold_index]
    assert 0 <= cv_fold_index < len(non_test_ids), "cv_fold_index must be in [0,3] when using fixed test fold"

    val_fold_id = non_test_ids[cv_fold_index]
    test_subjects = set(folds[test_fold_index])
    val_subjects = set(folds[val_fold_id])
    train_subjects = set(s for fid in non_test_ids if fid != val_fold_id for s in folds[fid])

    def keys_for(subj_set: set[str]) -> List[str]:
        out: List[str] = []
        for s in subj_set:
            out.extend(subj2keys.get(s, []))
        return out

    train_keys = keys_for(train_subjects)
    val_keys = keys_for(val_subjects)
    test_keys = keys_for(test_subjects)

    print(f"[PhysioMI Split] subj | train={len(train_subjects)} val={len(val_subjects)} test={len(test_subjects)}")
    print(f"[PhysioMI Split] keys | train={len(train_keys)} val={len(val_keys)} test={len(test_keys)}")
    return train_keys, val_keys, test_keys


# ---------------------------- Public loader API ----------------------------
class LoadDataset:
    """
    Build DataLoaders for one CV round with FIXED test fold and rotating val fold.

    params must provide:
      - datasets_dir : str           (LMDB directory)
      - batch_size   : int
      - num_workers  : int = 0
      - seed         : int = 7
      - test_fold_index : int = 4    (0..4)
      - cv_fold_index   : int = 0    (which non-test fold is val; 0..3)
      - folds_json      : Optional[str] (precomputed fold file)
      - pin_memory      : bool = True
      - drop_last       : bool = False
      - scale_div       : float = 100.0
    """
    def __init__(self, params):
        self.p = params
        self.dir = params.datasets_dir

        train_keys, val_keys, test_keys = build_subjectwise_split(
            data_dir=self.dir,
            seed=getattr(self.p, "seed", 7),
            test_fold_index=getattr(self.p, "test_fold_index", 4),
            cv_fold_index=getattr(self.p, "cv_fold_index", 0),
            folds_json=getattr(self.p, "folds_json", None),
        )

        scale = float(getattr(self.p, "scale_div", 100.0))
        self.train_set = CustomDataset(self.dir, train_keys, scale_div=scale)
        self.val_set   = CustomDataset(self.dir, val_keys,   scale_div=scale)
        self.test_set  = CustomDataset(self.dir, test_keys,  scale_div=scale)

    def get_data_loader(self) -> Dict[str, DataLoader]:
        bs = int(self.p.batch_size)
        nw = int(getattr(self.p, "num_workers", 0))
        pin = bool(getattr(self.p, "pin_memory", True))
        drop_last = bool(getattr(self.p, "drop_last", False))

        def _worker_init(worker_id):
            seed = (torch.initial_seed() + worker_id) % (2**32)
            np.random.seed(seed % (2**32 - 1))
            random.seed(seed)

        return {
            "train": DataLoader(self.train_set, batch_size=bs, shuffle=True,
                                num_workers=nw, pin_memory=pin, drop_last=drop_last,
                                collate_fn=self.train_set.collate, worker_init_fn=_worker_init),
            "val":   DataLoader(self.val_set,   batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=pin, drop_last=False,
                                collate_fn=self.val_set.collate, worker_init_fn=_worker_init),
            "test":  DataLoader(self.test_set,  batch_size=bs, shuffle=False,
                                num_workers=nw, pin_memory=pin, drop_last=False,
                                collate_fn=self.test_set.collate, worker_init_fn=_worker_init),
        }
