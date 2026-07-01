#!/usr/bin/env python3
"""
BCICIV2a -> LMDB (no split), 4s chunks, keep shape (C, T), no patchify.

- Scans all *.mat files in --root_dir (default ./data/BCICIV2a_mat, or $BCIC2A_MAT_DIR).
- For each trial, takes a fixed 4-second slice starting at --t_start (default 2.0s).
- Per-channel centering, zero-phase band-pass (Butter), crop, optional resample to --resample_hz.
- Optional EMS (exponential moving standardization) per channel over time.
- Stores {'sample': np.ndarray[C,T], 'label': int} under key "<file>-<block>-<trial>".
- Writes flat key list under '__keys__' into the LMDB (default ./data/BCICIV2a.lmdb, or $BCIC2A_LMDB).
"""

from __future__ import annotations

import os
import pickle
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import lmdb
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
from scipy.signal import lfilter
# --- EMS import (optional) ---
try:
    from braindecode.preprocessing import exponential_moving_standardize as _ems
except Exception:
    _ems = None

# ----------------------------- Utils --------------------------------- #

def setup_logger(verbosity: int = 1) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=level)

def butter_bandpass(low_cut: float, high_cut: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a

# --------------------------- Core logic ------------------------------- #

def _unwrap_cell(x):
    """Iteratively unwrap 1x1 object arrays / MATLAB cell wrappers."""
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x

def _extract_triplet_from_block(block, *, n_chans: int):
    b = _unwrap_cell(block)

    # Case A: MATLAB struct (mat_struct) with common BCIC2a fields
    if hasattr(b, "__dict__") and not isinstance(b, (np.ndarray, list, tuple)):
        if all(hasattr(b, nm) for nm in ("X", "trial", "y")):
            raw, events, labels = b.X, b.trial, b.y
        else:
            raw_attr = next((getattr(b, nm, None) for nm in ("raw", "signal", "data", "X", "cnt", "eeg", "samples")), None)
            evt_attr = next((getattr(b, nm, None) for nm in ("trial", "events", "ev", "markers")), None)
            lab_attr = next((getattr(b, nm, None) for nm in ("y", "labels", "label", "class", "target")), None)
            if raw_attr is None or evt_attr is None or lab_attr is None:
                fields = sorted(k for k in b.__dict__.keys() if not k.startswith("_"))
                raise TypeError(f"Unrecognized MATLAB struct fields: {fields}")
            raw, events, labels = raw_attr, evt_attr, lab_attr

    # Case B: object array / tuple-of-3: [raw, events, labels]
    elif isinstance(b, np.ndarray) and b.dtype == object and b.size >= 3:
        flat = b.ravel()
        raw, events, labels = flat[0], flat[1], flat[2]
    elif isinstance(b, (list, tuple)) and len(b) >= 3:
        raw, events, labels = b[0], b[1], b[2]
    else:
        if isinstance(block, np.ndarray) and block.dtype == object:
            try:
                return _extract_triplet_from_block(block[0, 0], n_chans=n_chans)
            except Exception:
                pass
            try:
                return _extract_triplet_from_block(block.item(), n_chans=n_chans)
            except Exception:
                pass
        raise TypeError(f"Unsupported block type: {type(b)} with shape={getattr(b, 'shape', None)}")

    raw = np.asarray(raw)
    events = np.asarray(events).ravel()
    labels = np.asarray(labels).ravel()

    if raw.ndim != 2:
        raise ValueError(f"Expected 2D raw, got shape {raw.shape}")
    S, C = raw.shape
    if C == n_chans:
        pass
    elif S == n_chans:
        raw = raw.T
    return raw, events, labels

def process_mat_file(
    mat_path: Path,
    *,
    fs: int,
    n_chans: int,
    band_low: float,
    band_high: float,
    butter_order: int,
    t_start: float,
    window_sec: float,
    resample_hz: int,
    # --- EMS toggles ---
    apply_ems: bool = False,
    ems_factor_new: float = 0.001,
    ems_init_block_size: Optional[int] = None,
    ems_eps: float = 1e-4,
    key_prefix: Optional[str] = None,
) -> List[Tuple[str, bytes]]:
    """
    Returns list of (lmdb_key, pickled_value) for one .mat file.
    Each value is {'sample': np.ndarray[C,T], 'label': int}.
    """
    logging.info(f"Reading {mat_path.name}")
    data = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    if "data" not in data:
        raise KeyError(f"'data' key not found in {mat_path}")

    blocks = data["data"]
    if not isinstance(blocks, np.ndarray):
        raise ValueError(f"Unexpected 'data' container type: {type(blocks)}")

    if apply_ems and _ems is None:
        raise RuntimeError("apply_ems=True but 'braindecode' is not installed. Install it or disable EMS.")

    b, a = butter_bandpass(band_low, band_high, fs, order=butter_order)

    kvs: List[Tuple[str, bytes]] = []
    base_name = key_prefix if key_prefix is not None else mat_path.stem

    j_start = 3 if blocks.shape[-1] > 3 else 0
    for j in range(j_start, blocks.shape[-1]):
        cell = blocks.item(j) if blocks.dtype == object and blocks.ndim == 1 else blocks[(0, j)] if blocks.ndim == 2 else blocks[j]
        try:
            raw_full, events, labels = _extract_triplet_from_block(cell, n_chans=n_chans)
        except Exception as e_try:
            try:
                raw_full, events, labels = _extract_triplet_from_block(_unwrap_cell(cell)[0, 0], n_chans=n_chans)
            except Exception as e_fallback:
                raise TypeError(f"{mat_path.name} block {j}: failed to parse block. "
                                f"Primary error: {e_try}; Fallback error: {e_fallback}")

        raw_full = np.asarray(raw_full, dtype=np.float32)
        if raw_full.ndim != 2:
            logging.debug(f"Skip block {j}: raw ndim={raw_full.ndim}")
            continue
        if raw_full.shape[1] < n_chans and raw_full.shape[0] >= n_chans:
            raw_full = raw_full.T
        raw = raw_full[:, :n_chans]  # [S, C]
        S, C = raw.shape
        if C != n_chans:
            logging.warning(f"{mat_path.name} block {j}: expected {n_chans} chans, got {C}; slicing available.")

        ev = events.astype(int).tolist()
        ev.append(S)
        segs = list(zip(ev[:-1], ev[1:]))

        s0 = int(round(t_start * fs))
        win_len = int(round(window_sec * fs))
        s1 = s0 + win_len
        if s0 < 0 or win_len <= 0:
            raise ValueError(f"Bad crop: t_start={t_start}, window_sec={window_sec}, fs={fs}")

        for i, ((s, e), lab) in enumerate(zip(segs, labels)):
            x = raw[s:e].T  # -> [C, T_trial]
            x = x - x.mean(axis=0, keepdims=True)

            x = lfilter(b, a, x, axis=-1)

            if x.shape[-1] < s1:
                logging.debug(f"Skip short seg {mat_path.name} j={j} i={i}: T={x.shape[-1]} < {s1}")
                continue

            x = x[:, s0:s1]  # [C, 4*fs]

            # Optional resample
            if resample_hz > 0 and resample_hz != fs:
                target_len = int(round(window_sec * resample_hz))
                x = resample(x, target_len, axis=-1)

            # ----- EMS (optional) -----
            if apply_ems:
                x = _ems(
                    x.astype(np.float32, copy=False),
                    factor_new=float(ems_factor_new),
                    init_block_size=(None if ems_init_block_size in (None, 0) else int(ems_init_block_size)),
                    eps=float(ems_eps),
                ).astype(np.float32, copy=False)

            y = int(lab) - 1  # to 0-based
            key = f"{base_name}-{j}-{i}"
            kvs.append((key, pickle.dumps({"sample": x.astype(np.float32), "label": y}, protocol=pickle.HIGHEST_PROTOCOL)))

    logging.info(f"{mat_path.name}: produced {len(kvs)} samples")
    return kvs

def write_lmdb(kvs: List[Tuple[str, bytes]], out_path: Path, *, map_size: int, batch_size: int = 512) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_path), map_size=map_size, subdir=True, lock=True)
    try:
        with env.begin(write=True) as txn:
            pass  # ensure db exists

        count = 0
        txn = env.begin(write=True)
        try:
            for k, v in kvs:
                txn.put(k.encode("utf-8"), v)
                count += 1
                if count % batch_size == 0:
                    txn.commit()
                    txn = env.begin(write=True)
            txn.commit()
        finally:
            if txn is not None:
                try:
                    txn.abort()
                except lmdb.Error:
                    pass
    finally:
        env.close()

# ------------------------------ CLI ---------------------------------- #

def parse_args() -> argparse.Namespace:
    default_root = Path(os.getenv("BCIC2A_MAT_DIR", "/projects/EEG-foundation-model/EEG-CDCL_data/downstream/BCIC2A_mat"))
    default_out  = Path(os.getenv("BCIC2A_LMDB", "/projects/EEG-foundation-model/EEG-CDCL_data/downstream/BCIC2A/processed_stelar"))

    p = argparse.ArgumentParser(description="BCICIV2a -> LMDB (4s chunks, keep CxT, no patchify)")
    p.add_argument("--root_dir", type=Path, default=default_root,
                   help="Folder with *.mat files (default: ./data/BCICIV2a_mat or $BCIC2A_MAT_DIR)")
    p.add_argument("--out_db", type=Path, default=default_out,
                   help="Output LMDB directory (default: ./data/BCICIV2a.lmdb or $BCIC2A_LMDB)")
    p.add_argument("--map_size", type=int, default=1_500_000_000, help="LMDB map size (bytes)")

    # Signal params
    p.add_argument("--fs", type=int, default=250, help="Native sampling rate of source data (Hz)")
    p.add_argument("--n_chans", type=int, default=22)
    p.add_argument("--band_low", type=float, default=0.3)
    p.add_argument("--band_high", type=float, default=50.0)
    p.add_argument("--butter_order", type=int, default=5)

    # Windowing (fixed 4s)
    p.add_argument("--t_start", type=float, default=2.0, help="Start time within each trial (sec)")
    p.add_argument("--window_sec", type=float, default=4.0, help="Window length (sec) — leave at 4.0")

    # Optional resample target rate
    p.add_argument("--resample_hz", type=int, default=200, help="0=keep native fs; else resample 4s to this Hz")

    # EMS toggles
    p.add_argument("--apply_ems", action="store_true", help="Apply exponential moving standardization (per channel).")
    p.add_argument("--ems_factor_new", type=float, default=0.001, help="EMS factor_new (Braindecode default 0.001).")
    p.add_argument("--ems_init_block_size", type=int, default=0,
                   help="EMS init_block_size (0 -> None; Braindecode default None).")
    p.add_argument("--ems_eps", type=float, default=1e-4, help="EMS eps (Braindecode default 1e-4).")

    p.add_argument("--verbosity", type=int, default=1, help="0=warn, 1=info, 2=debug")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_logger(args.verbosity)

    # Discover files
    if not args.root_dir.exists():
        raise FileNotFoundError(f"root_dir not found: {args.root_dir} "
                                "(set --root_dir or $BCIC2A_MAT_DIR)")
    mat_files = sorted([Path(args.root_dir) / f for f in os.listdir(args.root_dir) if f.endswith(".mat")])
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {args.root_dir}")

    all_kvs: List[Tuple[str, bytes]] = []
    all_keys: List[str] = []

    for mat_path in mat_files:
        kvs = process_mat_file(
            mat_path,
            fs=args.fs,
            n_chans=args.n_chans,
            band_low=args.band_low,
            band_high=args.band_high,
            butter_order=args.butter_order,
            t_start=args.t_start,
            window_sec=args.window_sec,
            resample_hz=args.resample_hz,
            apply_ems=args.apply_ems,
            ems_factor_new=args.ems_factor_new,
            ems_init_block_size=(None if args.ems_init_block_size in (None, 0) else int(args.ems_init_block_size)),
            ems_eps=args.ems_eps,
        )
        all_kvs.extend(kvs)
        all_keys.extend([k for k, _ in kvs])

    logging.info(f"Writing {len(all_kvs)} samples to {args.out_db}")
    write_lmdb(all_kvs, args.out_db, map_size=args.map_size, batch_size=512)

    # Save flat key list
    env = lmdb.open(str(args.out_db), map_size=args.map_size, subdir=True, lock=True)
    try:
        with env.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(all_keys, protocol=pickle.HIGHEST_PROTOCOL))
    finally:
        env.close()

    logging.info("Done.")

if __name__ == "__main__":
    main()
