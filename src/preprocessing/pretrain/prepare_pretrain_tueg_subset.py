#!/usr/bin/env python3
"""
TUH EEG -> LMDB windowed dataset builder — stable, resumable, parallel-safe

Adds duration logging (kept vs dropped files), per sub-dataset and per split:
- files_seen / files_kept / files_dropped
- orig duration (from EDF header)
- used duration (after edge trim + window alignment)
- kept duration (windows actually kept/written)
- drop_amp duration (windows rejected by amplitude filter)
- top drop reasons

- Deterministic file-level split: --fraction-train / --fraction-val / --split-seed
- Two output modes:
    (1) Two LMDBs:  --out-lmdb-train and --out-lmdb-val
    (2) One LMDB :  --out-lmdb (keys prefixed 'trn_'/'val_')
- Avoids nested parallelism (no loky warnings) by setting:
    * BLAS/OMP threads = 1
    * If --workers > 1 => force MNE --n-jobs=1
- Uses multiprocessing "spawn" for safety with NumPy/MKL/joblib
"""

from __future__ import annotations
import os

# ---- Thread caps to avoid oversubscription / loky issues ----
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import hashlib
import json
import pickle
import random
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict, Any
import gc

import lmdb
import mne
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# -------------------------- Channel templates --------------------------

SELECTED_CHANNELS = {
    "01_tcp_ar": [
        "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF", "EEG C4-REF", "EEG P3-REF",
        "EEG P4-REF", "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF",
        "EEG T5-REF", "EEG T6-REF", "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF",
    ],
    "02_tcp_le": [
        "EEG FP1-LE", "EEG FP2-LE", "EEG F3-LE", "EEG F4-LE", "EEG C3-LE", "EEG C4-LE", "EEG P3-LE",
        "EEG P4-LE", "EEG O1-LE", "EEG O2-LE", "EEG F7-LE", "EEG F8-LE", "EEG T3-LE", "EEG T4-LE",
        "EEG T5-LE", "EEG T6-LE", "EEG FZ-LE", "EEG CZ-LE", "EEG PZ-LE",
    ],
    "03_tcp_ar_a": [
        "EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF", "EEG C4-REF", "EEG P3-REF",
        "EEG P4-REF", "EEG O1-REF", "EEG O2-REF", "EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF",
        "EEG T5-REF", "EEG T6-REF", "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF",
    ],
}

# ------------------------------ Helpers --------------------------------


def setup_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def iter_edf_files_multi(parent_root: Path, dataset_names: List[str]):
    """Yield *.edf under specific sub-datasets inside parent_root."""
    for dname in dataset_names:
        droot = parent_root / dname
        if not droot.exists():
            print(f"[warn] dataset folder missing: {droot}")
            continue
        for p in sorted(droot.rglob("*.edf")):
            if p.is_file():
                yield p


def dataset_of_file(parent_root: Path, fp: Path, dataset_names: List[str]) -> str:
    """Infer which sub-dataset a file belongs to (first path component under parent_root)."""
    try:
        rel = fp.resolve().relative_to(parent_root.resolve())
        if rel.parts:
            d0 = rel.parts[0]
            return d0
    except Exception:
        pass

    s = str(fp)
    for d in dataset_names:
        if f"/{d}/" in s or f"\\{d}\\" in s:
            return d
    return "unknown"


def detect_montage_key(file_path: Path) -> Optional[str]:
    s = str(file_path).lower()
    if "02_tcp_le" in s:
        return "02_tcp_le"
    if "01_tcp_ar" in s:
        return "01_tcp_ar"
    if "03_tcp_ar_a" in s:
        return "03_tcp_ar_a"
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


def split_decision(path: Path, frac_train: float, frac_val: float, seed: int) -> str:
    """Deterministically assign a file to 'train', 'val', or 'skip' using SHA-1."""
    h = hashlib.sha1((str(path.resolve()) + f"|{seed}").encode("utf-8")).digest()
    u = int.from_bytes(h[:8], "big") / float(2**64)
    if u < frac_train:
        return "train"
    if u < frac_train + frac_val:
        return "val"
    return "skip"


# ------------------------ Duration logging helpers ----------------------


def _new_dur_bucket() -> Dict[str, Any]:
    return {
        "files_seen": 0,
        "files_kept": 0,
        "files_dropped": 0,
        "sec_orig": 0.0,      # EDF header duration
        "sec_used": 0.0,      # after trim + window alignment
        "sec_kept": 0.0,      # kept windows duration
        "sec_drop_amp": 0.0,  # windows dropped by amp filter only
        "win_total": 0,
        "win_kept": 0,
        "win_drop_amp": 0,
        "drop_reasons": {},   # reason -> count
    }


def _dur_add_reason(bucket: Dict[str, Any], reason: str) -> None:
    if not reason:
        return
    bucket["drop_reasons"][reason] = bucket["drop_reasons"].get(reason, 0) + 1


def _fmt_h(sec: float) -> str:
    return f"{sec/3600.0:.2f}h"


def print_duration_report(dur_stats: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
    """
    dur_stats[dataset][split] = bucket
    """
    print("========================================")
    print("[duration] per-dataset summary (hours)")
    total = _new_dur_bucket()

    for dset in sorted(dur_stats.keys()):
        for split in ("train", "val"):
            if split not in dur_stats[dset]:
                continue
            b = dur_stats[dset][split]

            for k in (
                "files_seen", "files_kept", "files_dropped",
                "sec_orig", "sec_used", "sec_kept", "sec_drop_amp",
                "win_total", "win_kept", "win_drop_amp",
            ):
                total[k] += b.get(k, 0)

            dr = b.get("drop_reasons", {}) or {}
            dr_str = ""
            if dr:
                top = sorted(dr.items(), key=lambda x: -x[1])[:5]
                dr_str = " | top_drop: " + ", ".join([f"{r}:{c}" for r, c in top])

            print(
                f"[duration] {dset:24s} {split:5s} | "
                f"files seen={b['files_seen']:6d} kept={b['files_kept']:6d} dropped={b['files_dropped']:6d} | "
                f"orig={_fmt_h(b['sec_orig'])} used={_fmt_h(b['sec_used'])} kept={_fmt_h(b['sec_kept'])} "
                f"drop_amp={_fmt_h(b['sec_drop_amp'])} | "
                f"win kept/total={b['win_kept']}/{b['win_total']}"
                f"{dr_str}"
            )

    print("----------------------------------------")
    print(
        f"[duration] TOTAL                    | "
        f"files seen={total['files_seen']:6d} kept={total['files_kept']:6d} dropped={total['files_dropped']:6d} | "
        f"orig={_fmt_h(total['sec_orig'])} used={_fmt_h(total['sec_used'])} kept={_fmt_h(total['sec_kept'])} "
        f"drop_amp={_fmt_h(total['sec_drop_amp'])} | "
        f"win kept/total={total['win_kept']}/{total['win_total']}"
    )
    print("========================================")


# ----------------------------- Core step -------------------------------


def preprocess_and_segment(
    file_path: Path,
    *,
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
    amp_filter: bool = True,
) -> Tuple[List[str], List[np.ndarray], Dict[str, Any]]:
    """
    Return (per_window_ids, samples, stats) for one file.
    Each sample is np.ndarray [C, window_sec, fs] in µV.

    stats includes:
      reason, dur_orig_sec, dur_used_sec, n_win_total, n_win_kept, n_win_drop_amp
    """
    raw = None
    stats: Dict[str, Any] = {
        "reason": "",
        "dur_orig_sec": 0.0,
        "dur_used_sec": 0.0,
        "n_win_total": 0,
        "n_win_kept": 0,
        "n_win_drop_amp": 0,
        "montage_key": None,
    }

    try:
        raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose="ERROR")

        # original recording duration from header
        try:
            stats["dur_orig_sec"] = float(raw.n_times) / float(raw.info["sfreq"])
        except Exception:
            stats["dur_orig_sec"] = 0.0

        mkey = detect_montage_key(file_path)
        stats["montage_key"] = mkey
        if mkey is None:
            stats["reason"] = "unknown_montage"
            return ([], [], stats)
        if not pick_ordered_channels(raw, mkey):
            stats["reason"] = "missing_channels"
            return ([], [], stats)

        # Resample / filter (+ optional notch at mains with harmonics)
        raw.resample(fs, npad="auto", n_jobs=n_jobs)
        raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs, verbose="ERROR")
        nyq = fs / 2.0
        if mains is not None and mains > 0 and mains <= nyq:
            freqs = [mains * k for k in range(1, int(nyq // mains) + 1)]
            raw.notch_filter(freqs=freqs, n_jobs=n_jobs, verbose="ERROR")

        # Data in µV
        try:
            data = raw.get_data(units="uV")
        except TypeError:
            data = raw.get_data() * 1e6

        # dtype normalization
        if dtype == "float32":
            data = data.astype(np.float32, copy=False)
        elif dtype == "float64":
            if data.dtype != np.float64:
                data = data.astype(np.float64, copy=False)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        C, T = data.shape
        if T < min_len_sec * fs:
            stats["reason"] = "too_short_after_resample"
            return ([], [], stats)

        # Trim edges & align to windows
        wlen = window_sec * fs
        edge = edge_trim_sec * fs
        leftover = T % wlen
        start = edge
        end = T - (leftover + edge)
        if end <= start:
            stats["reason"] = "trim_removed_all"
            return ([], [], stats)

        data = data[:, start:end]
        n_win = data.shape[1] // wlen
        stats["dur_used_sec"] = float(data.shape[1]) / float(fs)
        stats["n_win_total"] = int(n_win)

        if n_win == 0:
            stats["reason"] = "no_full_windows"
            return ([], [], stats)

        # Reshape to [N, C, window_sec, fs]
        data = data[:, : n_win * wlen].reshape(C, n_win, window_sec, fs).transpose(1, 0, 2, 3)

        # Decide which windows to keep
        if amp_filter:
            mx = np.max(np.abs(data), axis=(1, 2, 3))
            keep_idx = np.nonzero(mx < amp_max_uv)[0]
            stats["n_win_drop_amp"] = int(n_win - keep_idx.size)
            if keep_idx.size == 0:
                stats["reason"] = "all_windows_rejected_amp"
                return ([], [], stats)
        else:
            keep_idx = np.arange(data.shape[0])
            stats["n_win_drop_amp"] = 0

        # Optional cap per file (does NOT count as amp-drop)
        if limit_windows is not None and keep_idx.size > limit_windows:
            keep_idx = keep_idx[:limit_windows]

        stats["n_win_kept"] = int(keep_idx.size)
        if stats["n_win_kept"] == 0 and not stats["reason"]:
            stats["reason"] = "no_windows_kept"

        ids = [str(i) for i in keep_idx.tolist()]
        samples = [data[i] for i in keep_idx.tolist()]
        return (ids, samples, stats)

    except Exception:
        stats["reason"] = "exception"
        return ([], [], stats)
    finally:
        try:
            if raw is not None:
                raw.close()
        except Exception:
            pass
        del raw
        gc.collect()


# --------------------------- Compaction --------------------------------


def compact_lmdb_inplace(db_dir: Path, map_size_tb: float) -> None:
    """Copy all entries to a fresh env and swap directories."""
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
            cur = s_txn.cursor()
            for k, v in cur:
                d_txn.put(k, v)

        env_src.close()
        env_dst.close()

        swap_old = db_dir.with_name(db_dir.name + "_to_delete")
        db_dir.rename(swap_old)
        shutil.move(str(tmp_path), str(db_dir))
        shutil.rmtree(swap_old)
        print(f"[compact] Completed compaction at {db_dir}")


# ---------------------------- Parallel ---------------------------------


def _producer_worker(
    shard_files,
    cfg,
    q,
    key_scheme: str,
    limit_windows: Optional[int],
    split_assign: Dict[Path, str],
    single_db_prefix: bool,
    joblib_temp: Optional[Path],
    parent_root: Path,
    dataset_names: List[str],
):
    """Emit messages for windows + file-level stats."""
    mne.set_log_level("ERROR")
    if joblib_temp:
        os.environ["JOBLIB_TEMP_FOLDER"] = str(joblib_temp)

    for fp in shard_files:
        s = split_assign.get(fp, "skip")
        if s == "skip":
            q.put(("seen", file_signature(fp), None))
            continue

        fsig = file_signature(fp)
        dset = dataset_of_file(parent_root, fp, dataset_names)

        try:
            ids, arrays, st = preprocess_and_segment(
                fp,
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
                amp_filter=cfg.get("amp_filter", True),
            )

            # file-level duration stats
            n_total = int(st.get("n_win_total", 0) or 0)
            n_kept = int(st.get("n_win_kept", 0) or 0)
            n_drop_amp = int(st.get("n_win_drop_amp", 0) or 0)
            dur_orig = float(st.get("dur_orig_sec", 0.0) or 0.0)
            dur_used = float(st.get("dur_used_sec", 0.0) or 0.0)
            reason = str(st.get("reason", "") or "")
            q.put(("stat", s, dset, n_total, n_kept, n_drop_amp, dur_orig, dur_used, reason), block=True)

            base = fp.stem
            fsig_short = fsig[:10]
            for wid, arr in zip(ids, arrays):
                base_key = f"{base}_{wid}" if key_scheme == "stem" else f"{base}_{fsig_short}_{wid}"
                if single_db_prefix:
                    pref = "trn_" if s == "train" else "val_"
                    base_key = pref + base_key
                payload = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
                q.put(("data", s, base_key, payload), block=True)

            q.put(("seen", fsig, None), block=True)

        except Exception:
            # count as dropped (exception)
            q.put(("stat", s, dset, 0, 0, 0, 0.0, 0.0, "exception"), block=True)
            q.put(("seen", fsig, None), block=True)
        finally:
            gc.collect()

    q.put(("done", None, None))


def _run_parallel(
    files,
    env_train,
    env_val,
    using_config,
    *,
    present_keys_train: Set[str],
    present_keys_val: Set[str],
    seen_files: Set[str],
    batch_writes: int,
    checkpoint_interval: int,
    seen_flush_every: int,
    key_scheme: str,
    workers: int,
    writer_queue_cap: int,
    limit_windows: Optional[int],
    dry_run: bool,
    split_assign: Dict[Path, str],
    single_db_prefix: bool,
    joblib_temp: Optional[Path],
    parent_root: Path,
    dataset_names: List[str],
    window_sec: int,
):
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    q = manager.Queue(maxsize=writer_queue_cap)

    shards = [[] for _ in range(workers)]
    for i, fp in enumerate(files):
        shards[i % workers].append(fp)

    procs = []
    for shard in shards:
        p = ctx.Process(
            target=_producer_worker,
            args=(
                shard,
                using_config,
                q,
                key_scheme,
                limit_windows,
                split_assign,
                single_db_prefix,
                joblib_temp,
                parent_root,
                dataset_names,
            ),
            daemon=True,
        )
        p.start()
        procs.append(p)

    pending_trn: List[Tuple[str, bytes]] = []
    pending_val: List[Tuple[str, bytes]] = []
    written_trn = written_val = 0
    skipped_trn = skipped_val = 0
    since_last_ckpt_trn = since_last_ckpt_val = 0
    seen_local = set(seen_files)
    live = workers

    # duration aggregation
    dur_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def _get_bucket(dset: str, split: str) -> Dict[str, Any]:
        if dset not in dur_stats:
            dur_stats[dset] = {}
        if split not in dur_stats[dset]:
            dur_stats[dset][split] = _new_dur_bucket()
        return dur_stats[dset][split]

    pbar = tqdm(total=len(files), desc="Writing", unit="file")
    seen_files_counted = set()
    sample_printed = 0

    try:
        while live > 0:
            msg = q.get()
            if not isinstance(msg, (tuple, list)) or not msg:
                continue

            typ = msg[0]

            if typ == "data":
                try:
                    _, split, key_str, payload = msg
                except Exception:
                    continue

                if dry_run:
                    if sample_printed < 5:
                        arr = pickle.loads(payload)
                        print(f"[dry-run] {split}:{key_str} -> shape={arr.shape}, dtype={arr.dtype}")
                        sample_printed += 1
                    continue

                if split == "train":
                    if key_str in present_keys_train:
                        skipped_trn += 1
                    else:
                        pending_trn.append((key_str, payload))
                        if len(pending_trn) >= batch_writes:
                            with env_train.begin(write=True) as txn:
                                for k, v in pending_trn:
                                    if txn.get(k.encode("utf-8")) is None:
                                        txn.put(k.encode("utf-8"), v)
                                        present_keys_train.add(k)
                                        written_trn += 1
                            pending_trn.clear()
                            since_last_ckpt_trn += batch_writes
                            if checkpoint_interval and since_last_ckpt_trn >= checkpoint_interval:
                                with env_train.begin(write=True) as txn:
                                    txn.put(
                                        b"__keys__",
                                        pickle.dumps(sorted(present_keys_train), protocol=pickle.HIGHEST_PROTOCOL),
                                    )
                                since_last_ckpt_trn = 0
                else:  # val
                    if key_str in present_keys_val:
                        skipped_val += 1
                    else:
                        pending_val.append((key_str, payload))
                        if len(pending_val) >= batch_writes:
                            with env_val.begin(write=True) as txn:
                                for k, v in pending_val:
                                    if txn.get(k.encode("utf-8")) is None:
                                        txn.put(k.encode("utf-8"), v)
                                        present_keys_val.add(k)
                                        written_val += 1
                            pending_val.clear()
                            since_last_ckpt_val += batch_writes
                            if checkpoint_interval and since_last_ckpt_val >= checkpoint_interval:
                                with env_val.begin(write=True) as txn:
                                    txn.put(
                                        b"__keys__",
                                        pickle.dumps(sorted(present_keys_val), protocol=pickle.HIGHEST_PROTOCOL),
                                    )
                                since_last_ckpt_val = 0

            elif typ == "seen":
                try:
                    _, fsig, _ = msg
                except Exception:
                    continue
                if fsig not in seen_local:
                    seen_local.add(fsig)
                    if fsig not in seen_files_counted:
                        pbar.update(1)
                        seen_files_counted.add(fsig)
                    if not dry_run and seen_flush_every and (len(seen_local) % seen_flush_every == 0):
                        for env in (env_train, env_val):
                            with env.begin(write=True) as txn:
                                txn.put(
                                    b"__seen_files__",
                                    pickle.dumps(sorted(seen_local), protocol=pickle.HIGHEST_PROTOCOL),
                                )

            elif typ == "stat":
                # ("stat", split, dset, n_total, n_kept, n_drop_amp, dur_orig, dur_used, reason)
                try:
                    _, split, dset, n_total, n_kept, n_drop_amp, dur_orig, dur_used, reason = msg
                except Exception:
                    continue
                b = _get_bucket(str(dset), str(split))
                b["files_seen"] += 1
                b["sec_orig"] += float(dur_orig or 0.0)
                b["sec_used"] += float(dur_used or 0.0)
                b["win_total"] += int(n_total or 0)
                b["win_kept"] += int(n_kept or 0)
                b["win_drop_amp"] += int(n_drop_amp or 0)

                kept_file = int(n_kept or 0) > 0
                if kept_file:
                    b["files_kept"] += 1
                else:
                    b["files_dropped"] += 1
                    _dur_add_reason(b, str(reason or "dropped"))

                b["sec_kept"] += float(int(n_kept or 0) * window_sec)
                b["sec_drop_amp"] += float(int(n_drop_amp or 0) * window_sec)

            elif typ == "done":
                live -= 1
            else:
                continue

        # Flush leftovers
        if not dry_run:
            if pending_trn:
                with env_train.begin(write=True) as txn:
                    for k, v in pending_trn:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_train.add(k)
                            written_trn += 1
            if pending_val:
                with env_val.begin(write=True) as txn:
                    for k, v in pending_val:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_val.add(k)
                            written_val += 1
            for env, keys in ((env_train, present_keys_train), (env_val, present_keys_val)):
                with env.begin(write=True) as txn:
                    txn.put(b"__keys__", pickle.dumps(sorted(keys), protocol=pickle.HIGHEST_PROTOCOL))
                    txn.put(b"__seen_files__", pickle.dumps(sorted(seen_local), protocol=pickle.HIGHEST_PROTOCOL))

    finally:
        pbar.close()
        for p in procs:
            p.join(timeout=10)

    if dry_run:
        print(f"[parallel][dry-run] files_seen={len(seen_local)} (no writes)")
    else:
        print(
            f"[parallel] written(train)={written_trn} skipped(train)={skipped_trn} | "
            f"written(val)={written_val} skipped(val)={skipped_val} | files_seen={len(seen_local)}"
        )

    print_duration_report(dur_stats)


# ------------------------------- Main ----------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build LMDB from TUH EEG EDF files (windowed, global meta).")
    p.add_argument("--in-root", type=Path, required=True,
                   help="Parent folder containing TUH sub-datasets (e.g., tuh_eeg_*).")
    p.add_argument("--datasets", type=str,
                   default="tuh_eeg_artifact,tuh_eeg_epilepsy,tuh_eeg_seizure,tuh_eeg_slowing",
                   help="Comma-separated subfolders under --in-root to scan for EDFs.")
    p.add_argument("--allowed-montages", type=str, default="01_tcp_ar",
                   help="Comma-separated subset of {01_tcp_ar,02_tcp_le,03_tcp_ar_a} to include.")

    # Deterministic split (both produced in one run)
    p.add_argument("--fraction-train", type=float, default=0.10, help="Fraction of files assigned to train.")
    p.add_argument("--fraction-val", type=float, default=0.01, help="Fraction of files assigned to val.")
    p.add_argument("--split-seed", type=int, default=12345, help="Seed for deterministic file split.")

    # Output choices
    p.add_argument("--out-lmdb-train", type=Path, help="LMDB for TRAIN split.")
    p.add_argument("--out-lmdb-val", type=Path, help="LMDB for VAL split.")
    p.add_argument("--out-lmdb", type=Path,
                   help="Single LMDB output for BOTH splits (keys will be prefixed 'trn_'/'val_').")

    p.add_argument("--map-size-tb", type=float, default=0.5, help="LMDB map size in terabytes")
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
    p.add_argument(
        "--disable-amp-filter",
        action="store_true",
        help="If set, do NOT drop windows based on amp_max (keep all amplitudes).",
    )
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--n-jobs", type=int, default=1, help="Threads per MNE op (forced to 1 if --workers>1)")
    p.add_argument("--joblib-temp", type=Path, default=Path("/tmp/joblib"),
                   help="Folder for joblib temp files (helps on small /dev/shm).")

    # IO policy / resume
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--batch-writes", type=int, default=512)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--force-rewrite-config", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=20000)
    p.add_argument("--seen-files-flush-every", type=int, default=50)
    p.add_argument("--key-scheme", type=str, default="stem_sig", choices=["stem", "stem_sig"])

    # Small run / smoke test
    p.add_argument("--limit-files", type=int, default=None, help="Process only first N EDF files")
    p.add_argument("--limit-windows", type=int, default=None, help="Keep at most N windows per file")
    p.add_argument("--dry-run", action="store_true", help="Run without writing LMDB; print sample shapes")

    # Parallel
    p.add_argument("--workers", type=int, default=1, help="Producer processes (set >1 for parallel)")
    p.add_argument("--writer-queue-cap", type=int, default=2048)

    # Modes
    p.add_argument("--compact", action="store_true", help="Compact an existing LMDB in place and exit")
    p.add_argument("--compact-db", type=Path, help="LMDB directory to compact (defaults to outputs)")
    return p.parse_args()


def main() -> None:
    # Use spawn globally for stability (no fork with MKL/joblib)
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    mne.set_log_level("ERROR")

    # Ensure joblib temp (used if any internal parallelism appears)
    if args.joblib_temp:
        os.environ["JOBLIB_TEMP_FOLDER"] = str(args.joblib_temp)
        Path(args.joblib_temp).mkdir(parents=True, exist_ok=True)

    # Compact mode
    if args.compact:
        target = args.compact_db or args.out_lmdb or args.out_lmdb_train or args.out_lmdb_val
        if not target:
            raise SystemExit(
                "Provide one of --out-lmdb, --out-lmdb-train, --out-lmdb-val (or --compact-db) with --compact."
            )
        compact_lmdb_inplace(target, args.map_size_tb)
        return

    # Validate outputs
    if not (args.out_lmdb or (args.out_lmdb_train and args.out_lmdb_val)):
        raise SystemExit("Provide either --out-lmdb OR both --out-lmdb-train and --out-lmdb-val.")
    single_db_mode = args.out_lmdb is not None

    setup_seed(args.seed)

    in_root: Path = args.in_root
    dataset_names = [s.strip() for s in args.datasets.split(",") if s.strip()]
    files_all = list(iter_edf_files_multi(in_root, dataset_names))

    # Optional montage filter
    allowed_montages = set(s.strip() for s in args.allowed_montages.split(",") if s.strip())
    if allowed_montages:
        files_all = [fp for fp in files_all if (detect_montage_key(fp) in allowed_montages)]

    # Deterministic split assignment
    split_assign: Dict[Path, str] = {}
    train_files, val_files = [], []
    for fp in files_all:
        s = split_decision(fp, args.fraction_train, args.fraction_val, args.split_seed)
        split_assign[fp] = s
        if s == "train":
            train_files.append(fp)
        elif s == "val":
            val_files.append(fp)

    files = train_files + val_files  # both, in one pass
    if args.shuffle:
        random.shuffle(files)
    if args.limit_files is not None:
        files = files[: args.limit_files]

    print("========================================")
    print(f"[split] total files scanned : {len(files_all)}")
    print(f"[split] assigned -> train   : {len(train_files)} "
          f"({len(train_files)/max(1,len(files_all))*100:.2f}%)")
    print(f"[split] assigned -> val     : {len(val_files)} "
          f"({len(val_files)/max(1,len(files_all))*100:.2f}%)")
    print(f"[run]    files this pass    : {len(files)} (train+val)")
    print("========================================")

    # Open LMDBs
    map_size_bytes = int(args.map_size_tb * (1024 ** 4))
    if single_db_mode:
        args.out_lmdb.mkdir(parents=True, exist_ok=True)
        env_train = env_val = lmdb.open(
            str(args.out_lmdb),
            map_size=map_size_bytes, subdir=True, lock=True,
            writemap=True, map_async=True, metasync=False, readahead=False,
            max_readers=2048,
        )
        single_db_prefix = True
    else:
        args.out_lmdb_train.mkdir(parents=True, exist_ok=True)
        args.out_lmdb_val.mkdir(parents=True, exist_ok=True)
        env_train = lmdb.open(
            str(args.out_lmdb_train),
            map_size=map_size_bytes, subdir=True, lock=True,
            writemap=True, map_async=True, metasync=False, readahead=False,
            max_readers=2048,
        )
        env_val = lmdb.open(
            str(args.out_lmdb_val),
            map_size=map_size_bytes, subdir=True, lock=True,
            writemap=True, map_async=True, metasync=False, readahead=False,
            max_readers=2048,
        )
        single_db_prefix = False

    # Load resume meta
    def load_meta(env):
        present: Set[str] = set()
        existing_cfg = None
        seen: Set[str] = set()
        with env.begin(write=False) as txn:
            blob = txn.get(b"__keys__")
            if blob is not None:
                try:
                    present = set(pickle.loads(blob))
                except Exception:
                    cur = txn.cursor()
                    for k, _ in cur:
                        if k not in (b"__keys__", b"__config__", b"__seen_files__"):
                            present.add(k.decode("utf-8"))
            cfg_blob = txn.get(b"__config__")
            if cfg_blob is not None:
                try:
                    existing_cfg = json.loads(cfg_blob.decode("utf-8"))
                except Exception:
                    existing_cfg = None
            s_blob = txn.get(b"__seen_files__")
            if s_blob is not None:
                try:
                    seen = set(pickle.loads(s_blob))
                except Exception:
                    seen = set()
        return present, existing_cfg, seen

    present_keys_train, existing_config_tr, seen_files_tr = load_meta(env_train)
    present_keys_val, existing_config_vl, seen_files_vl = load_meta(env_val)
    seen_files = seen_files_tr.union(seen_files_vl)

    desired_config = {
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
        "note": "Channel ordering is standardized per file per montage.",
        "fraction_train": args.fraction_train,
        "fraction_val": args.fraction_val,
        "split_seed": args.split_seed,
        "single_db_mode": single_db_mode,
        "amp_filter": (not args.disable_amp_filter),
    }

    def ensure_config(env, existing_cfg):
        # Allow older DBs that don't have amp_filter key by treating it as True
        if existing_cfg is not None and "amp_filter" not in existing_cfg:
            existing_cfg = dict(existing_cfg)
            existing_cfg["amp_filter"] = True

        if existing_cfg is None or (existing_cfg != desired_config and args.force_rewrite_config):
            with env.begin(write=True) as txn:
                txn.put(b"__config__", json.dumps(desired_config, ensure_ascii=False).encode("utf-8"))
            return desired_config
        if existing_cfg != desired_config and not args.force_rewrite_config:
            raise RuntimeError("Existing __config__ differs. Use --force-rewrite-config to overwrite.")
        return existing_cfg

    using_config = ensure_config(env_train, existing_config_tr)
    ensure_config(env_val, existing_config_vl)

    print("========================================")
    print(f"[resume] keys(train) loaded : {len(present_keys_train)}")
    print(f"[resume] keys(val)   loaded : {len(present_keys_val)}")
    print(f"[resume] seen_files merged  : {len(seen_files)}")
    print(f"[config] amp_filter         : {using_config.get('amp_filter', True)}")
    print("========================================")

    # duration aggregation (single-process path)
    dur_stats: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def get_bucket(dset: str, split: str) -> Dict[str, Any]:
        if dset not in dur_stats:
            dur_stats[dset] = {}
        if split not in dur_stats[dset]:
            dur_stats[dset][split] = _new_dur_bucket()
        return dur_stats[dset][split]

    # Filter out already-seen files up-front (resume)
    if args.resume and seen_files:
        files = [fp for fp in files if file_signature(fp) not in seen_files]

    # IMPORTANT: avoid nested parallelism → force n_jobs=1 when workers>1
    if args.workers > 1 and using_config["n_jobs"] != 1:
        print(f"[info] workers={args.workers} > 1 → forcing MNE n_jobs=1 to avoid nested parallelism.")
        using_config["n_jobs"] = 1

    # Single-process or parallel
    if args.workers <= 1:
        pending_trn: List[Tuple[str, bytes]] = []
        pending_val: List[Tuple[str, bytes]] = []
        written_trn = written_val = 0
        since_last_ckpt_trn = since_last_ckpt_val = 0
        files_processed_since_seen_flush = 0
        sample_printed = 0

        def flush_pending(force_checkpoint: bool = False) -> None:
            nonlocal pending_trn, pending_val, written_trn, written_val
            nonlocal since_last_ckpt_trn, since_last_ckpt_val
            if pending_trn:
                with env_train.begin(write=True) as txn:
                    for k, v in pending_trn:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_train.add(k)
                            written_trn += 1
                            since_last_ckpt_trn += 1
                pending_trn.clear()
            if pending_val:
                with env_val.begin(write=True) as txn:
                    for k, v in pending_val:
                        if txn.get(k.encode("utf-8")) is None:
                            txn.put(k.encode("utf-8"), v)
                            present_keys_val.add(k)
                            written_val += 1
                            since_last_ckpt_val += 1
                pending_val.clear()
            if args.checkpoint_interval > 0 and force_checkpoint:
                with env_train.begin(write=True) as txn:
                    txn.put(b"__keys__", pickle.dumps(sorted(present_keys_train), protocol=pickle.HIGHEST_PROTOCOL))
                with env_val.begin(write=True) as txn:
                    txn.put(b"__keys__", pickle.dumps(sorted(present_keys_val), protocol=pickle.HIGHEST_PROTOCOL))
                since_last_ckpt_trn = since_last_ckpt_val = 0

        for fp in tqdm(files, desc="Processing EDFs (train+val)", unit="file"):
            s = split_assign.get(fp, "skip")
            if s == "skip":
                continue

            fsig = file_signature(fp)
            dset = dataset_of_file(in_root, fp, dataset_names)

            try:
                win_ids, arrays, st = preprocess_and_segment(
                    file_path=fp,
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
                    amp_filter=using_config.get("amp_filter", True),
                )
            except Exception:
                win_ids, arrays, st = [], [], {"reason": "exception", "dur_orig_sec": 0.0, "dur_used_sec": 0.0,
                                              "n_win_total": 0, "n_win_kept": 0, "n_win_drop_amp": 0}

            # duration stats per file
            b = get_bucket(dset, s)
            n_total = int(st.get("n_win_total", 0) or 0)
            n_kept = int(st.get("n_win_kept", 0) or 0)
            n_drop_amp = int(st.get("n_win_drop_amp", 0) or 0)
            dur_orig = float(st.get("dur_orig_sec", 0.0) or 0.0)
            dur_used = float(st.get("dur_used_sec", 0.0) or 0.0)
            reason = str(st.get("reason", "") or "")

            b["files_seen"] += 1
            b["sec_orig"] += dur_orig
            b["sec_used"] += dur_used
            b["win_total"] += n_total
            b["win_kept"] += n_kept
            b["win_drop_amp"] += n_drop_amp

            if n_kept > 0:
                b["files_kept"] += 1
            else:
                b["files_dropped"] += 1
                _dur_add_reason(b, reason or "dropped")

            b["sec_kept"] += float(n_kept * args.window_sec)
            b["sec_drop_amp"] += float(n_drop_amp * args.window_sec)

            base = fp.stem
            fsig_short = fsig[:10]
            for wid, arr in zip(win_ids, arrays):
                base_key = f"{base}_{wid}" if args.key_scheme == "stem" else f"{base}_{fsig_short}_{wid}"
                if single_db_mode:
                    pref = "trn_" if s == "train" else "val_"
                    base_key = pref + base_key
                payload = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)

                if args.dry_run:
                    if sample_printed < 5:
                        print(f"[dry-run] {s}:{base_key} -> shape={arr.shape}, dtype={arr.dtype}")
                        sample_printed += 1
                    continue

                if s == "train":
                    if not (args.resume and base_key in present_keys_train):
                        pending_trn.append((base_key, payload))
                else:
                    if not (args.resume and base_key in present_keys_val):
                        pending_val.append((base_key, payload))

                if len(pending_trn) + len(pending_val) >= args.batch_writes:
                    flush_pending(force_checkpoint=False)

            if not args.dry_run:
                seen_files.add(fsig)
                files_processed_since_seen_flush += 1
                if args.seen_files_flush_every and files_processed_since_seen_flush >= args.seen_files_flush_every:
                    for env in (env_train, env_val):
                        with env.begin(write=True) as txn:
                            txn.put(
                                b"__seen_files__",
                                pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL),
                            )
                    files_processed_since_seen_flush = 0

            gc.collect()

        flush_pending(force_checkpoint=True)

        # Final meta write
        with env_train.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(sorted(present_keys_train), protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(b"__seen_files__", pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL))
        with env_val.begin(write=True) as txn:
            txn.put(b"__keys__", pickle.dumps(sorted(present_keys_val), protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(b"__seen_files__", pickle.dumps(sorted(seen_files), protocol=pickle.HIGHEST_PROTOCOL))

        print_duration_report(dur_stats)

    else:
        _run_parallel(
            files=files,
            env_train=env_train,
            env_val=env_val,
            using_config=using_config,
            present_keys_train=present_keys_train,
            present_keys_val=present_keys_val,
            seen_files=seen_files,
            batch_writes=args.batch_writes,
            checkpoint_interval=args.checkpoint_interval,
            seen_flush_every=args.seen_files_flush_every,
            key_scheme=args.key_scheme,
            workers=args.workers,
            writer_queue_cap=args.writer_queue_cap,
            limit_windows=args.limit_windows,
            dry_run=args.dry_run,
            split_assign=split_assign,
            single_db_prefix=single_db_mode,  # prefix only if single DB
            joblib_temp=args.joblib_temp,
            parent_root=in_root,
            dataset_names=dataset_names,
            window_sec=args.window_sec,
        )

    # Close
    env_train.sync()
    env_train.close()
    if env_val is not env_train:
        env_val.sync()
        env_val.close()

    # --- summary ---
    print("========================================")
    if not args.dry_run:
        if single_db_mode:
            print(f"LMDB (single): {args.out_lmdb}  [keys prefixed 'trn_'/'val_']")
        else:
            print(f"LMDB train  : {args.out_lmdb_train}")
            print(f"LMDB val    : {args.out_lmdb_val}")
        print(f"Seen files  : {len(seen_files)}")
    else:
        print("DRY-RUN complete (no LMDB writes).")
    print("========================================")


if __name__ == "__main__":
    main()
