#!/usr/bin/env python3
"""
TUH EEG -> LMDB windowed dataset builder — RESUMABLE, PARALLEL

Adds cheap duration accounting (almost free):
- raw_sec (after resample)
- aligned_sec (after trim + window alignment)
- kept_sec (after amp filter + optional limit_windows)

Writes:
- __config__ (json)
- __keys__ (pickle list of keys)
- __seen_files__ (pickle list of file signatures)
- __meta_json__ (json summary: durations + counts)

NEW:
- --montages: choose 1, 2, or all 3 montages (default: all)
  Example:
    --montages 01_tcp_ar
    --montages 01_tcp_ar,02_tcp_le
    --montages all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import lmdb
import mne
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# -------------------------- Channel templates -------------------------- #

SELECTED_CHANNELS: Dict[str, List[str]] = {
    "01_tcp_ar": [
        "EEG FP1-REF","EEG FP2-REF","EEG F3-REF","EEG F4-REF","EEG C3-REF","EEG C4-REF","EEG P3-REF",
        "EEG P4-REF","EEG O1-REF","EEG O2-REF","EEG F7-REF","EEG F8-REF","EEG T3-REF","EEG T4-REF",
        "EEG T5-REF","EEG T6-REF","EEG FZ-REF","EEG CZ-REF","EEG PZ-REF",
    ],
    "02_tcp_le": [
        "EEG FP1-LE","EEG FP2-LE","EEG F3-LE","EEG F4-LE","EEG C3-LE","EEG C4-LE","EEG P3-LE",
        "EEG P4-LE","EEG O1-LE","EEG O2-LE","EEG F7-LE","EEG F8-LE","EEG T3-LE","EEG T4-LE",
        "EEG T5-LE","EEG T6-LE","EEG FZ-LE","EEG CZ-LE","EEG PZ-LE",
    ],
    "03_tcp_ar_a": [
        "EEG FP1-REF","EEG FP2-REF","EEG F3-REF","EEG F4-REF","EEG C3-REF","EEG C4-REF","EEG P3-REF",
        "EEG P4-REF","EEG O1-REF","EEG O2-REF","EEG F7-REF","EEG F8-REF","EEG T3-REF","EEG T4-REF",
        "EEG T5-REF","EEG T6-REF","EEG FZ-REF","EEG CZ-REF","EEG PZ-REF",
    ],
}

ALL_MONTAGES = tuple(SELECTED_CHANNELS.keys())
META_KEYS = {b"__keys__", b"__config__", b"__seen_files__", b"__meta_json__"}


# ------------------------------ Helpers -------------------------------- #
# -------------------------- Exclusion by SUBJECT prefix -------------------------- #

_SUBJECT_PREFIX_RE = re.compile(r"^([A-Za-z0-9]+)_")


def subject_prefix_from_name(name: str) -> Optional[str]:
    """
    Extract subject prefix from EDF filename (before first underscore).
    Examples:
      aaaaaaar_00000001.edf -> aaaaaaar
      aaaaajte_s001_t003.edf -> aaaaajte
      aaaaaaaa_s001_t000.edf -> aaaaaaaa
    """
    m = _SUBJECT_PREFIX_RE.match(name)
    return m.group(1) if m else None


def collect_subject_prefixes(root: Optional[Path]) -> Set[str]:
    """Collect subject prefixes from all .edf names under root."""
    if root is None:
        return set()
    if not root.exists():
        raise FileNotFoundError(f"Exclude root not found: {root}")
    subs: Set[str] = set()
    for p in root.rglob("*.edf"):
        sid = subject_prefix_from_name(p.name)
        if sid:
            subs.add(sid)
    return subs


def filter_files_by_excluded_subjects(files: List[Path], excluded_subjects: Set[str]) -> Tuple[List[Path], int]:
    """Remove files whose subject prefix is in excluded_subjects."""
    if not excluded_subjects:
        return files, 0

    kept: List[Path] = []
    removed = 0
    for fp in files:
        sid = subject_prefix_from_name(fp.name)
        if sid and sid in excluded_subjects:
            removed += 1
        else:
            kept.append(fp)
    return kept, removed


def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def iter_edf_files(root: Path):
    yield from sorted(root.rglob("*.edf"))


def detect_montage_key(file_path: Path) -> Optional[str]:
    s = str(file_path).lower()
    if "02_tcp_le" in s:
        return "02_tcp_le"
    if "01_tcp_ar" in s:
        return "01_tcp_ar"
    if "03_tcp_ar_a" in s:
        return "03_tcp_ar_a"
    return None


def get_subject_id(file_path: Path) -> Optional[str]:
    for part in file_path.parts:
        m = re.fullmatch(r"s\d{1,4}", part.lower())
        if m:
            return m.group(0)
    return None


def pick_ordered_channels(raw: mne.io.BaseRaw, montage_key: str) -> bool:
    want = SELECTED_CHANNELS[montage_key]
    have = set(raw.info["ch_names"])
    if any(ch not in have for ch in want):
        return False
    raw.pick_channels(want, ordered=True)
    return True


def file_signature(path: Path) -> str:
    st = path.stat()
    sig = f"{path.resolve()}::{st.st_size}::{int(st.st_mtime)}"
    return hashlib.sha1(sig.encode("utf-8")).hexdigest()


def init_meta() -> Dict[str, float]:
    return {
        "files_seen": 0.0,
        "raw_sec": 0.0,
        "aligned_sec": 0.0,
        "kept_sec": 0.0,
        "windows_total": 0.0,
        "windows_kept": 0.0,
        "windows_dropped": 0.0,
    }


def add_meta(acc: Dict[str, float], st: Dict[str, float]) -> None:
    acc["files_seen"] += 1.0
    acc["raw_sec"] += float(st.get("raw_sec", 0.0))
    acc["aligned_sec"] += float(st.get("aligned_sec", 0.0))
    acc["kept_sec"] += float(st.get("kept_sec", 0.0))
    acc["windows_total"] += float(st.get("windows_total", 0.0))
    acc["windows_kept"] += float(st.get("windows_kept", 0.0))
    acc["windows_dropped"] += float(st.get("windows_dropped", 0.0))


def parse_montages_arg(s: str) -> Tuple[str, ...]:
    """
    Accept:
      - "all"
      - "01_tcp_ar"
      - "01_tcp_ar,02_tcp_le"
    """
    s = (s or "").strip()
    if not s or s.lower() == "all":
        return ALL_MONTAGES

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return ALL_MONTAGES

    bad = [p for p in parts if p not in SELECTED_CHANNELS]
    if bad:
        raise SystemExit(
            f"Unknown montage(s): {bad}. Allowed: {list(SELECTED_CHANNELS.keys())} or 'all'."
        )
    # preserve user order, but make unique
    out: List[str] = []
    seen: Set[str] = set()
    for p in parts:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return tuple(out)


def filter_files_by_montages(files: List[Path], allowed: Set[str]) -> Tuple[List[Path], int]:
    """
    Fast path-based filter: if montage can be detected from path and is not in allowed, drop.
    If montage cannot be detected (unknown), keep (it will be validated later by channel pick).
    """
    if not allowed or allowed == set(ALL_MONTAGES):
        return files, 0
    kept: List[Path] = []
    dropped = 0
    for f in files:
        mk = detect_montage_key(f)
        if mk is not None and mk not in allowed:
            dropped += 1
            continue
        kept.append(f)
    return kept, dropped


# -------------------------- Split utilities ---------------------------- #

def stratified_subject_split(
    files: List[Path],
    train_frac: float,
    val_frac: float,
    seed: int,
    montages: Set[str],
) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)

    # only stratify across chosen montages (path-detected)
    by_m: Dict[str, List[Path]] = {k: [] for k in montages}
    for f in files:
        mk = detect_montage_key(f)
        if mk is not None and mk in by_m:
            by_m[mk].append(f)

    train_all: List[Path] = []
    val_all: List[Path] = []

    for mk, group in by_m.items():
        if not group:
            continue

        subj_to_files: Dict[str, List[Path]] = {}
        no_subj: List[Path] = []
        for f in group:
            sid = get_subject_id(f)
            if sid is None:
                no_subj.append(f)
            else:
                subj_to_files.setdefault(sid, []).append(f)

        subjects = list(subj_to_files.keys())
        rng.shuffle(subjects)
        rng.shuffle(no_subj)

        n_files = len(group)
        target_train = max(1, math.ceil(train_frac * n_files))
        target_val = max(1, math.ceil(val_frac * n_files)) if val_frac > 0 else 0

        pick_train: List[Path] = []
        pick_val: List[Path] = []

        for sid in subjects:
            if len(pick_train) < target_train:
                pick_train.extend(subj_to_files[sid])
            elif len(pick_val) < target_val:
                pick_val.extend(subj_to_files[sid])
            if len(pick_train) >= target_train and len(pick_val) >= target_val:
                break

        def fill(pool: List[Path], acc: List[Path], target: int, already: Set[Path]):
            for f in pool:
                if len(acc) >= target:
                    break
                if f not in already:
                    acc.append(f)

        already = set(pick_train) | set(pick_val)
        if len(pick_train) < target_train:
            fill(no_subj, pick_train, target_train, already)
            already = set(pick_train) | set(pick_val)
        if len(pick_val) < target_val:
            fill(no_subj, pick_val, target_val, already)
            already = set(pick_train) | set(pick_val)

        remain = [f for f in group if f not in already]
        rng.shuffle(remain)
        if len(pick_train) < target_train:
            need = target_train - len(pick_train)
            pick_train.extend(remain[:need])
            remain = remain[need:]
        if len(pick_val) < target_val:
            need = target_val - len(pick_val)
            pick_val.extend(remain[:need])

        inter = set(pick_train).intersection(pick_val)
        if inter:
            pick_val = [f for f in pick_val if f not in inter]

        train_all.extend(pick_train)
        val_all.extend(pick_val)

    train_set: List[Path] = []
    seen: Set[Path] = set()
    for f in train_all:
        if f not in seen:
            train_set.append(f)
            seen.add(f)
    val_set = [f for f in val_all if f not in seen]
    return train_set, val_set


# ----------------------------- Core step ------------------------------- #

def preprocess_and_segment(
    file_path: Path,
    *,
    montages_allowed: Set[str],
    fs: int,
    l_freq: float,
    h_freq: float,
    mains: Optional[int],
    window_sec: int,
    edge_trim_sec: int,
    min_len_sec: int,
    amp_max_uv: float,
    dtype: str,
    n_jobs: int,
    limit_windows: Optional[int] = None,
) -> Tuple[List[str], List[np.ndarray], Dict[str, float]]:
    """
    Returns (ids, samples, stats)
    stats fields are cheap (derived from shapes/counters):
      raw_sec, aligned_sec, kept_sec, windows_total, windows_kept, windows_dropped
    """
    stats = {
        "raw_sec": 0.0,
        "aligned_sec": 0.0,
        "kept_sec": 0.0,
        "windows_total": 0.0,
        "windows_kept": 0.0,
        "windows_dropped": 0.0,
    }

    try:
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose="ERROR")
    except Exception:
        return ([], [], stats)

    mkey = detect_montage_key(file_path)
    if mkey is None:
        return ([], [], stats)
    if mkey not in montages_allowed:
        return ([], [], stats)
    if not pick_ordered_channels(raw, mkey):
        return ([], [], stats)

    try:
        raw.resample(fs, npad="auto", n_jobs=n_jobs)
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, verbose="ERROR")
        nyq = fs / 2.0
        # NOTE: original code always notched at 60. Respect --mains.
        if mains is not None and mains > 0 and mains <= nyq:
            raw.notch_filter(freqs=[mains], n_jobs=n_jobs, verbose="ERROR")
    except Exception:
        return ([], [], stats)

    try:
        data = raw.get_data(units="uV")
    except TypeError:
        data = raw.get_data() * 1e6

    if dtype == "float32":
        data = data.astype(np.float32, copy=False)
    elif dtype == "float64":
        data = data.astype(np.float64, copy=False)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    C, T = data.shape
    stats["raw_sec"] = float(T) / float(fs)

    if T < min_len_sec * fs:
        return ([], [], stats)

    wlen = window_sec * fs
    edge = edge_trim_sec * fs
    leftover = T % wlen
    start = edge
    end = T - (leftover + edge)
    if end <= start:
        return ([], [], stats)

    data = data[:, start:end]
    n_win = data.shape[1] // wlen
    if n_win <= 0:
        return ([], [], stats)

    stats["aligned_sec"] = float(n_win * window_sec)
    stats["windows_total"] = float(n_win)

    data = data[:, : n_win * wlen].reshape(C, n_win, window_sec, fs).transpose(1, 0, 2, 3)

    mx = np.max(np.abs(data), axis=(1, 2, 3))
    keep_idx = np.nonzero(mx < amp_max_uv)[0]
    if keep_idx.size == 0:
        stats["windows_dropped"] = float(n_win)
        return ([], [], stats)

    if limit_windows is not None and keep_idx.size > limit_windows:
        keep_idx = keep_idx[:limit_windows]

    kept = int(keep_idx.size)
    stats["windows_kept"] = float(kept)
    stats["windows_dropped"] = float(n_win - kept)
    stats["kept_sec"] = float(kept * window_sec)

    ids = [str(i) for i in keep_idx.tolist()]
    samples = [data[i] for i in keep_idx.tolist()]
    return (ids, samples, stats)


# --------------------------- Compaction -------------------------------- #

def compact_lmdb_inplace(db_dir: Path, map_size_tb: float) -> None:
    db_dir = db_dir.resolve()
    if not db_dir.exists():
        raise FileNotFoundError(db_dir)

    with tempfile.TemporaryDirectory(dir=db_dir.parent) as tmpd:
        tmp_path = Path(tmpd) / "lmdb_compact"
        tmp_path.mkdir(parents=True, exist_ok=True)

        env_src = lmdb.open(str(db_dir), readonly=True, lock=False, subdir=True, readahead=False)
        src_info = env_src.info()
        map_size_bytes = max(int(map_size_tb * (1024 ** 4)), src_info.get("map_size", 0))

        env_dst = lmdb.open(
            str(tmp_path),
            map_size=map_size_bytes,
            subdir=True,
            lock=True,
            writemap=True,
            map_async=True,
            metasync=False,
            readahead=False,
            max_readers=2048,
        )

        with env_src.begin() as s_txn, env_dst.begin(write=True) as d_txn:
            for k, v in s_txn.cursor():
                d_txn.put(k, v)

        env_src.close()
        env_dst.close()

        swap_old = db_dir.with_name(db_dir.name + "_to_delete")
        db_dir.rename(swap_old)
        shutil.move(str(tmp_path), str(db_dir))
        shutil.rmtree(swap_old)
        print(f"[compact] Completed compaction at {db_dir}")


# ---------------------------- Parallel --------------------------------- #

def _producer_worker(shard_files, cfg, q, key_scheme: str, limit_windows: Optional[int]):
    mne.set_log_level("ERROR")
    allowed = set(cfg["montages"])  # tuple/list -> set

    for fp in shard_files:
        fsig = file_signature(fp)
        try:
            ids, arrays, st = preprocess_and_segment(
                fp,
                montages_allowed=allowed,
                fs=cfg["fs_hz"],
                l_freq=cfg["l_freq_hz"],
                h_freq=cfg["h_freq_hz"],
                mains=(None if cfg["mains_hz"] == 0 else cfg["mains_hz"]),
                window_sec=cfg["window_sec"],
                edge_trim_sec=cfg["edge_trim_sec"],
                min_len_sec=cfg["min_len_sec"],
                amp_max_uv=cfg["amp_max_uV"],
                dtype=cfg["dtype"],
                n_jobs=cfg["n_jobs"],
                limit_windows=limit_windows,
            )

            base = fp.stem
            fsig_short = fsig[:10]
            for wid, arr in zip(ids, arrays):
                key_str = f"{base}_{wid}" if key_scheme == "stem" else f"{base}_{fsig_short}_{wid}"
                payload = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
                q.put(("data", key_str, payload), block=True)

            q.put(("seen", fsig, st), block=True)
        except Exception:
            q.put(("seen", fsig, init_meta()), block=True)
            continue

    q.put(("done", None, None))


def _run_parallel(
    files,
    env,
    using_config,
    *,
    present_keys: Set[str],
    seen_files: Set[str],
    batch_writes: int,
    checkpoint_interval: int,
    seen_flush_every: int,
    key_scheme: str,
    workers: int,
    writer_queue_cap: int,
    limit_windows: Optional[int],
    dry_run: bool,
) -> Dict[str, float]:
    manager = mp.Manager()
    q = manager.Queue(maxsize=writer_queue_cap)

    shards = [[] for _ in range(workers)]
    for i, fp in enumerate(files):
        shards[i % workers].append(fp)

    procs = []
    for shard in shards:
        p = mp.Process(target=_producer_worker, args=(shard, using_config, q, key_scheme, limit_windows))
        p.start()
        procs.append(p)

    pending: List[Tuple[str, bytes]] = []
    written_cnt = 0
    skipped_cnt = 0
    since_last_ckpt = 0
    seen_local = set(seen_files)
    live = workers

    agg = init_meta()
    pbar = tqdm(total=len(files), desc="Writing", unit="file")
    seen_files_counted: Set[str] = set()
    sample_printed = 0

    try:
        while live > 0:
            typ, a, b = q.get()

            if typ == "data":
                key_str, payload = a, b
                if dry_run:
                    if sample_printed < 5:
                        arr = pickle.loads(payload)
                        print(f"[dry-run] {key_str}: shape={arr.shape}, dtype={arr.dtype}")
                        sample_printed += 1
                    continue

                if key_str in present_keys:
                    skipped_cnt += 1
                    continue

                pending.append((key_str, payload))
                if len(pending) >= batch_writes:
                    with env.begin(write=True) as txn:
                        for k, v in pending:
                            kb = k.encode("utf-8")
                            if txn.get(kb) is None:
                                txn.put(kb, v)
                                present_keys.add(k)
                                written_cnt += 1
                    pending.clear()
                    since_last_ckpt += batch_writes
                    if checkpoint_interval and since_last_ckpt >= checkpoint_interval:
                        with env.begin(write=True) as txn:
                            txn.put(b"__keys__", pickle.dumps(sorted(present_keys), protocol=pickle.HIGHEST_PROTOCOL))
                        since_last_ckpt = 0

            elif typ == "seen":
                fsig, st = a, (b or {})
                if fsig not in seen_local:
                    seen_local.add(fsig)
                    add_meta(agg, st)
                    if fsig not in seen_files_counted:
                        pbar.update(1)
                        seen_files_counted.add(fsig)

                    if not dry_run and seen_flush_every and (len(seen_local) % seen_flush_every == 0):
                        with env.begin(write=True) as txn:
                            txn.put(b"__seen_files__", pickle.dumps(sorted(seen_local), protocol=pickle.HIGHEST_PROTOCOL))

            elif typ == "done":
                live -= 1

        if not dry_run and pending:
            with env.begin(write=True) as txn:
                for k, v in pending:
                    kb = k.encode("utf-8")
                    if txn.get(kb) is None:
                        txn.put(kb, v)
                        present_keys.add(k)
                        written_cnt += 1

        if not dry_run:
            with env.begin(write=True) as txn:
                txn.put(b"__keys__", pickle.dumps(sorted(present_keys), protocol=pickle.HIGHEST_PROTOCOL))
                txn.put(b"__seen_files__", pickle.dumps(sorted(seen_local), protocol=pickle.HIGHEST_PROTOCOL))

    finally:
        pbar.close()
        for p in procs:
            p.join()

    if dry_run:
        print(f"[parallel][dry-run] files_seen={int(agg['files_seen'])} (no writes)")
    else:
        print(f"[parallel] written={written_cnt} skipped={skipped_cnt} files_seen={int(agg['files_seen'])}")

    return agg


# ------------------------------- Build one subset ---------------------- #

def build_subset(
    files: List[Path],
    out_lmdb: Optional[Path],
    using_config: Dict,
    *,
    args,
) -> None:
    env = None
    present_keys: Set[str] = set()
    existing_config = None
    seen_files: Set[str] = set()

    agg_run = init_meta()

    if not args.dry_run:
        assert out_lmdb is not None, "out_lmdb must be provided unless --dry-run"
        out_lmdb.mkdir(parents=True, exist_ok=True)
        map_size_bytes = int(args.map_size_tb * (1024 ** 4))
        env = lmdb.open(
            str(out_lmdb),
            map_size=map_size_bytes,
            subdir=True,
            lock=True,
            writemap=True,
            map_async=True,
            metasync=False,
            readahead=False,
            max_readers=2048,
        )

        with env.begin(write=False) as txn:
            blob = txn.get(b"__keys__")
            if blob is not None:
                try:
                    present_keys = set(pickle.loads(blob))
                except Exception:
                    for k, _ in txn.cursor():
                        if k not in META_KEYS:
                            present_keys.add(k.decode("utf-8"))

            cfg_blob = txn.get(b"__config__")
            if cfg_blob is not None:
                try:
                    existing_config = json.loads(cfg_blob.decode("utf-8"))
                except Exception:
                    existing_config = None

            s_blob = txn.get(b"__seen_files__")
            if s_blob is not None:
                try:
                    seen_files = set(pickle.loads(s_blob))
                except Exception:
                    seen_files = set()

        if existing_config is None or args.force_rewrite_config:
            with env.begin(write=True) as txn:
                txn.put(b"__config__", json.dumps(using_config, ensure_ascii=False).encode("utf-8"))
        else:
            if existing_config != using_config:
                raise RuntimeError(
                    f"Existing __config__ at {out_lmdb} differs from current args. "
                    f"Use --force-rewrite-config to overwrite."
                )

    print("========================================")
    print(f"[subset] output LMDB         : {out_lmdb if out_lmdb else '(dry-run)'}")
    print(f"[resume] present keys loaded : {len(present_keys)}")
    print(f"[resume] seen_files entries  : {len(seen_files)}")
    print("[mode] dry-run               :", args.dry_run)
    print("========================================")

    # resume filter
    if env is not None and args.resume and seen_files:
        files = [fp for fp in files if file_signature(fp) not in seen_files]

    if args.workers <= 1:
        pending: List[Tuple[str, bytes]] = []
        since_last_ckpt = 0
        files_processed_since_seen_flush = 0
        sample_printed = 0

        def flush_pending(force_checkpoint: bool = False) -> None:
            nonlocal pending, since_last_ckpt
            if env is None or not pending:
                pending.clear()
                return
            with env.begin(write=True) as txn:
                for k, v in pending:
                    kb = k.encode("utf-8")
                    if txn.get(kb) is None:
                        txn.put(kb, v)
                        present_keys.add(k)
                        since_last_ckpt += 1
            pending.clear()
            if env is not None and args.checkpoint_interval > 0:
                if since_last_ckpt >= args.checkpoint_interval or force_checkpoint:
                    with env.begin(write=True) as txn:
                        txn.put(b"__keys__", pickle.dumps(sorted(present_keys), protocol=pickle.HIGHEST_PROTOCOL))
                    since_last_ckpt = 0

        allowed = set(using_config["montages"])
        for fp in tqdm(files, desc="Processing EDFs", unit="file"):
            fsig = file_signature(fp)

            ids, arrays, st = preprocess_and_segment(
                file_path=fp,
                montages_allowed=allowed,
                fs=using_config["fs_hz"],
                l_freq=using_config["l_freq_hz"],
                h_freq=using_config["h_freq_hz"],
                mains=(None if using_config["mains_hz"] == 0 else using_config["mains_hz"]),
                window_sec=using_config["window_sec"],
                edge_trim_sec=using_config["edge_trim_sec"],
                min_len_sec=using_config["min_len_sec"],
                amp_max_uv=using_config["amp_max_uV"],
                dtype=using_config["dtype"],
                n_jobs=using_config["n_jobs"],
                limit_windows=args.limit_windows,
            )

            add_meta(agg_run, st)

            base = fp.stem
            fsig_short = fsig[:10]
            for wid, arr in zip(ids, arrays):
                key_str = f"{base}_{wid}" if args.key_scheme == "stem" else f"{base}_{fsig_short}_{wid}"

                if args.dry_run:
                    if sample_printed < 5:
                        print(f"[dry-run] {key_str}: shape={arr.shape}, dtype={arr.dtype}")
                        sample_printed += 1
                    continue

                if args.resume and key_str in present_keys:
                    continue

                payload = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
                pending.append((key_str, payload))
                if len(pending) >= args.batch_writes:
                    flush_pending(force_checkpoint=False)

            flush_pending(force_checkpoint=False)

            if env is not None:
                seen_files.add(fsig)
                files_processed_since_seen_flush += 1
                if args.seen_files_flush_every and files_processed_since_seen_flush >= args.seen_files_flush_every:
                    with env.begin(write=True) as txn:
                        txn.put(b"__seen_files__", pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL))
                    files_processed_since_seen_flush = 0

        flush_pending(force_checkpoint=True)

        if env is not None:
            with env.begin(write=True) as txn:
                txn.put(b"__keys__", pickle.dumps(sorted(present_keys), protocol=pickle.HIGHEST_PROTOCOL))
                txn.put(b"__seen_files__", pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL))

    else:
        assert env is not None or args.dry_run, "env should exist unless dry-run"
        agg_run = _run_parallel(
            files=files,
            env=env,
            using_config=using_config,
            present_keys=present_keys,
            seen_files=seen_files,
            batch_writes=args.batch_writes,
            checkpoint_interval=args.checkpoint_interval,
            seen_flush_every=args.seen_files_flush_every,
            key_scheme=args.key_scheme,
            workers=args.workers,
            writer_queue_cap=args.writer_queue_cap,
            limit_windows=args.limit_windows,
            dry_run=args.dry_run,
        )

    if env is not None and not args.dry_run:
        meta = {
            "config": using_config,
            "run_summary": {
                "files_processed": int(agg_run["files_seen"]),
                "raw_sec": float(agg_run["raw_sec"]),
                "aligned_sec": float(agg_run["aligned_sec"]),
                "kept_sec": float(agg_run["kept_sec"]),
                "windows_total": int(agg_run["windows_total"]),
                "windows_kept": int(agg_run["windows_kept"]),
                "windows_dropped": int(agg_run["windows_dropped"]),
            },
        }
        with env.begin(write=True) as txn:
            txn.put(b"__meta_json__", json.dumps(meta, ensure_ascii=False).encode("utf-8"))

        env.sync()
        env.close()

    print("========================================")
    if args.dry_run:
        print("DRY-RUN complete (no LMDB writes).")
    else:
        print(f"LMDB: {out_lmdb}")
        print(f"Run files processed : {int(agg_run['files_seen']):,}")
        print(f"Run kept duration   : {agg_run['kept_sec']/3600.0:,.2f} hours")
        print(f"Run kept windows    : {int(agg_run['windows_kept']):,}")
    print("========================================")


# ------------------------------- Main ---------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LMDB from TUH EEG EDF files (windowed).")
    p.add_argument("--in-root", type=Path, required=True, help="Root folder of TUH EDFs")

    # Single-output
    p.add_argument("--out-lmdb", type=Path, default=None, help="Output LMDB directory (single build mode)")

    # Split mode
    p.add_argument("--pretrain-split", action="store_true")
    p.add_argument("--train-out-lmdb", type=Path, default=None)
    p.add_argument("--val-out-lmdb", type=Path, default=None)
    p.add_argument("--train-fraction", type=float, default=0.10)
    p.add_argument("--val-fraction", type=float, default=0.01)

    p.add_argument("--map-size-tb", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1)

    # Processing
    p.add_argument("--fs", type=int, default=200)
    p.add_argument("--l-freq", type=float, default=0.3)
    p.add_argument("--h-freq", type=float, default=75.0)
    p.add_argument("--mains", type=int, default=60, choices=[0, 50, 60])
    p.add_argument("--window-sec", type=int, default=30)
    p.add_argument("--edge-trim-sec", type=int, default=60)
    p.add_argument("--min-len-sec", type=int, default=300)
    p.add_argument("--amp-max", type=float, default=100)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--n-jobs", type=int, default=4, help="Threads per MNE op (per process)")

    # Montage selection: 1, 2, or all 3
    p.add_argument(
        "--montages",
        type=str,
        default="all",
        help=(
            "Which TUH montages to include. "
            "Use 'all' or a comma-separated list like '01_tcp_ar' or '01_tcp_ar,02_tcp_le'."
        ),
    )

    # IO / resume
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--batch-writes", type=int, default=512)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--force-rewrite-config", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=20000)
    p.add_argument("--seen-files-flush-every", type=int, default=50)
    p.add_argument("--key-scheme", type=str, default="stem_sig", choices=["stem", "stem_sig"])

    # Small runs
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--limit-windows", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")

    # Parallel
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--writer-queue-cap", type=int, default=4096)

    # Compact
    p.add_argument("--compact", action="store_true")
    p.add_argument("--compact-db", type=Path, default=None)

    # Exclude TUAB/TUEV subjects from TUEG based on subject prefix (before first underscore)
    p.add_argument("--exclude-tuab-root", type=Path, default=None,
                   help="Root folder of TUAB EDFs (exclude matching subject prefixes).")
    p.add_argument("--exclude-tuev-root", type=Path, default=None,
                   help="Root folder of TUEV EDFs (exclude matching subject prefixes).")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    mne.set_log_level("ERROR")

    if args.compact:
        target = args.compact_db or args.out_lmdb
        if not target:
            raise SystemExit("Provide --out-lmdb or --compact-db with --compact.")
        compact_lmdb_inplace(target, args.map_size_tb)
        return

    if args.pretrain_split:
        if not (args.train_out_lmdb and args.val_out_lmdb):
            raise SystemExit("With --pretrain-split, provide --train-out-lmdb and --val-out-lmdb.")
    else:
        if not args.dry_run and not args.out_lmdb:
            raise SystemExit("Provide --out-lmdb (unless --dry-run).")

    setup_seed(args.seed)

    montages = parse_montages_arg(args.montages)
    montages_set = set(montages)
    print(f"[montages] enabled: {list(montages)}")

    files = list(iter_edf_files(args.in_root))
    if args.shuffle:
        random.shuffle(files)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    # fast filter by montage based on path detection
    files, dropped = filter_files_by_montages(files, montages_set)
    if dropped:
        print(f"[montages] dropped_by_path: {dropped:,} (kept={len(files):,})")

    # ---------------- Exclude TUAB/TUEV SUBJECTS (not filenames) ----------------
    excluded_subjects: Set[str] = set()
    if args.exclude_tuab_root is not None:
        print(f"[exclude] collecting TUAB subject prefixes from: {args.exclude_tuab_root}")
        excluded_subjects |= collect_subject_prefixes(args.exclude_tuab_root)
    if args.exclude_tuev_root is not None:
        print(f"[exclude] collecting TUEV subject prefixes from: {args.exclude_tuev_root}")
        excluded_subjects |= collect_subject_prefixes(args.exclude_tuev_root)

    if excluded_subjects:
        before = len(files)
        files, removed = filter_files_by_excluded_subjects(files, excluded_subjects)
        print(f"[exclude] removed {removed:,}/{before:,} TUEG files by subject match. remaining={len(files):,}")

    using_config = {
        "montages": list(montages),  # <- stored in LMDB config
        "fs_hz": args.fs,
        "l_freq_hz": args.l_freq,
        "h_freq_hz": args.h_freq,
        "mains_hz": (0 if args.mains == 0 else args.mains),
        "window_sec": args.window_sec,
        "edge_trim_sec": args.edge_trim_sec,
        "min_len_sec": args.min_len_sec,
        "amp_max_uV": args.amp_max,
        "units": "uV",
        "dtype": args.dtype,
        "n_jobs": args.n_jobs,
        "note": "Channel ordering is standardized per file per montage; montage inclusion is controlled by --montages.",
    }

    if not args.pretrain_split:
        build_subset(files=files, out_lmdb=args.out_lmdb, using_config=using_config, args=args)
        return

    train_files, val_files = stratified_subject_split(
        files=files,
        train_frac=args.train_fraction,
        val_frac=args.val_fraction,
        seed=args.seed,
        montages=montages_set,
    )

    def count_by_m(files_list: List[Path]) -> Dict[str, int]:
        c = {k: 0 for k in ALL_MONTAGES}
        c["unknown"] = 0
        for f in files_list:
            mk = detect_montage_key(f) or "unknown"
            c[mk] = c.get(mk, 0) + 1
        return c

    print("------------ Split summary -------------")
    print(f"All files         : {len(files)}")
    print(f"Train files (≈{args.train_fraction:.2%}) : {len(train_files)}  {count_by_m(train_files)}")
    print(f"Val   files (≈{args.val_fraction:.2%})   : {len(val_files)}    {count_by_m(val_files)}")

    inter = set(train_files).intersection(val_files)
    if inter:
        print(f"[warn] overlap detected ({len(inter)}). Removing from val.")
        val_files = [f for f in val_files if f not in inter]

    build_subset(files=train_files, out_lmdb=args.train_out_lmdb, using_config=using_config, args=args)
    build_subset(files=val_files, out_lmdb=args.val_out_lmdb, using_config=using_config, args=args)


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    main()
