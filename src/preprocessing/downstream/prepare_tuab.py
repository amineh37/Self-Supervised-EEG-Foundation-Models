#!/usr/bin/env python3
"""
TUAB preprocess (no folder restructuring)

- Reads EDFs under:
    <root>/{train,eval}/{abnormal,normal}/01_tcp_ar
- Preprocess:
    • bandpass ~0.1–75 Hz
    • notch 50 Hz (EU mains)   # change to 60 if needed
    • resample 200 Hz
    • checks channel order against a standard list
- Windows: 10 s non-overlap => 2000 samples @ 200 Hz
- Writes pickles to:
    <root>/processed/{train,eval}/{abnormal,normal}/01_tcp_ar/<stem>_<idx>.pkl
  with {"X": float32 (n_channels, 2000), "y": int} where y: abnormal=1, normal=0
"""

import os
import pickle
from pathlib import Path
from multiprocessing import Pool
import numpy as np
import mne

# ---------- channel config ----------
drop_channels = [
    'PHOTIC-REF', 'IBI', 'BURSTS', 'SUPPR', 'EEG ROC-REF', 'EEG LOC-REF',
    'EEG EKG1-REF', 'EMG-REF', 'EEG C3P-REF', 'EEG C4P-REF', 'EEG SP1-REF',
    'EEG SP2-REF', 'EEG LUC-REF', 'EEG RLC-REF', 'EEG RESP1-REF', 'EEG RESP2-REF',
    'EEG EKG-REF', 'RESP ABDOMEN-REF', 'ECG EKG-REF', 'PULSE RATE',
    'EEG PG2-REF', 'EEG PG1-REF'
]
drop_channels.extend([f'EEG {i}-REF' for i in range(20, 129)])

chOrder_standard = [
    'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
    'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
    'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF',
    'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF'
]

# WIN_SAMPLES = 2000  # 10s @ 200 Hz
WIN_SAMPLES = 2560  # 10s @ 256 Hz

def process_one_edf(edf_path: Path, out_dir: Path, label: int):
    edf_path = Path(edf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

        # drop non-EEG channels if present
        to_drop = [ch for ch in drop_channels if ch in raw.ch_names]
        if to_drop:
            raw.drop_channels(to_drop)

        # reorder to standard if lengths match
        if len(raw.ch_names) == len(chOrder_standard):
            raw.reorder_channels(chOrder_standard)

        if raw.ch_names != chOrder_standard:
            raise RuntimeError("channel order is wrong!")

        # filters and resample
        raw.filter(l_freq=0.1, h_freq=75.0, verbose="ERROR")
        raw.notch_filter(50.0, verbose="ERROR")   # change to 60.0 for US mains
        # raw.resample(200, n_jobs="auto", verbose="ERROR")
        raw.resample(200, n_jobs=os.cpu_count(), verbose="ERROR")


        data = raw.get_data(units='uV')  # (n_channels, n_samples)
    except Exception as e:
        with open("tuab-process-error-files.txt", "a") as f:
            f.write(f"{edf_path}\t{repr(e)}\n")
        return

    n_full = data.shape[1] // WIN_SAMPLES
    stem = edf_path.stem
    for i in range(n_full):
        seg = data[:, i*WIN_SAMPLES:(i+1)*WIN_SAMPLES].astype(np.float32)
        
        dump_path = out_dir / f"{stem}_{i}.pkl"
        with open(dump_path, "wb") as fh:
            pickle.dump({"X": seg, "y": label}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def gather_tasks(root: Path, target: Path,channel_std: str):
    """
    Build (edf_path, out_dir, label) triplets mirroring the source tree.
    """
    tasks = []

    for split in ("train", "eval"):
        for cls, label in (("abnormal", 1), ("normal", 0)):
            in_dir  = root / split / cls / channel_std
            out_dir = target/split /str(label)
            if not in_dir.is_dir():
                continue
            for fn in os.listdir(in_dir):
                if fn.lower().endswith(".edf"):
                    tasks.append((in_dir / fn, out_dir, label))
    return tasks


if __name__ == "__main__":
    # ---- set your TUAB root here ----
    root = Path("/projects/EEG-foundation-model/tuh_eeg_abnormal/v3.0.1/edf")
    # target = Path("/projects/EEG-foundation-model/STELAR/data/downstream/tuab_stelar/processed")
    target = Path("/projects/EEG-foundation-model/STELAR/data/downstream/tuab_eegpt/processed")
    channel_std = "01_tcp_ar"

    os.makedirs(target, exist_ok=True)

    tasks = gather_tasks(root, target, channel_std)

    # for split in ("train", "eval"):
    #     for cls, label in (("abnormal", 1), ("normal", 0)):
    #         in_dir  = root / split / cls / channel_std
    #         out_dir = target/split
    #         if not in_dir.is_dir():
    #             continue
    #         for fn in os.listdir(in_dir):
    #             if fn.lower().endswith(".edf"):
    #                 tasks.append((in_dir / fn, out_dir, label))
                   

    # Parallel processing
    # Tune worker count for your node (24 was your previous choice)
    workers = 20
    with Pool(processes=workers) as pool:
        pool.starmap(process_one_edf, tasks)

    print(f"Done! Windows saved under: {target/'processed'} (same subfolders as source)")
