#!/usr/bin/env python3
"""
Sleep-EDF (PhysioNet) -> per-window .pt tensors (NO SPLIT)

- Resamples to 200 Hz
- Converts V -> µV
- Low-pass filter at 30 Hz
- 30s non-overlapping windows (6000 samples at 200 Hz)
- Channel-wise z-score per window
- Saves windows into dataset_root/<label>/*.pt (no Train/Valid/Test split)
"""

import os
from pathlib import Path
import numpy as np
import torch

from braindecode.datasets import SleepPhysionet
from braindecode.preprocessing import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events
from sklearn.preprocessing import scale as standard_scale

# ---------------- Config ---------------- #
n_jobs = 4

# Sampling & windowing
target_sfreq = 200          # <- resample to 200 Hz
window_size_s = 30
window_size_samples = window_size_s * target_sfreq  # 30s * 200 Hz = 6000

# Filtering
high_cut_hz = 30.0          # low-pass only (None, 30)
factor_v_to_uv = 1e6        # V -> µV

# Label mapping (merge N3/N4 per AASM)
mapping = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}

# Output root (NO SPLIT)
dataset_root = Path("/projects/EEG-foundation-model/STELAR/data/downstream/sleep_edf_stelar")
(dataset_root).mkdir(parents=True, exist_ok=True)
for lab in range(5):
    (dataset_root / str(lab)).mkdir(parents=True, exist_ok=True)

# Subjects 0..82 with missing: [39, 68, 69, 78, 79]
all_subjects = [s for s in range(0, 83) if s not in {39, 68, 69, 78, 79}]

# ---------------- Preprocessors ---------------- #
preprocessors_record = [
    Preprocessor(lambda data: data * factor_v_to_uv),         # V -> µV
    Preprocessor('filter', l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs),
    Preprocessor('resample', sfreq=target_sfreq, n_jobs=n_jobs),
]

# Channel-wise z-score on windows
preprocessors_window = [
    Preprocessor(standard_scale, channel_wise=True)
]

# ---------------- Processing loop (no split) ---------------- #
for sub in all_subjects:
    # Load subject (both recordings for the subject)
    dataset = SleepPhysionet(subject_ids=[sub], crop_wake_mins=30)  # crop long wake
    preprocess(dataset, preprocessors_record)
    
    print(dataset.description)

    # Create non-overlapping 30s windows aligned to events
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        preload=True,
        mapping=mapping,
    )

    # Normalize each window channel-wise
    preprocess(windows_dataset, preprocessors_window)

    # Save windows (label folders only)
    for i, (X, y, meta) in enumerate(windows_dataset):
        lbl = int(y)
        cls_dir = dataset_root / str(lbl)
        cls_dir.mkdir(parents=True, exist_ok=True)

        # rec_id = meta.get('recording', 'rec')
        # start  = meta.get('i_window_in_trial', i)
        save_path = cls_dir / f"s{sub}_{meta[1]}_{meta[2]}.pt"

        torch.save(torch.tensor(X, dtype=torch.float32), save_path)

print("Done. Wrote windows under", dataset_root)