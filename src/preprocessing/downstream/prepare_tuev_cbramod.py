#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import mne
import numpy as np
from tqdm import tqdm


# ------------------------- event windowing ------------------------- #

def build_events(signals: np.ndarray, times: np.ndarray, event_data: np.ndarray, fs: float = 200.0):
    """Slice 5s windows around each annotated event."""
    num_events = int(event_data.shape[0])
    num_ch, _ = signals.shape
    win_len = int(fs * 5)  # 5 seconds

    features = np.zeros((num_events, num_ch, win_len), dtype=np.float32)
    offending_channel = np.zeros((num_events, 1), dtype=np.int32)
    labels = np.zeros((num_events, 1), dtype=np.int32)

    offset = signals.shape[1]
    pad = np.concatenate([signals, signals, signals], axis=1)

    for i in range(num_events):
        chan = int(event_data[i, 0])
        start = int(np.where(times >= event_data[i, 1])[0][0])
        end   = int(np.where(times >= event_data[i, 2])[0][0])
        s = offset + start - 2 * int(fs)
        e = offset + end   + 2 * int(fs)
        features[i, :] = pad[:, s:e]
        offending_channel[i, 0] = chan
        labels[i, 0] = int(event_data[i, 3])
    return features, offending_channel, labels


# ------------------------- bipolar montage ------------------------- #

def convert_signals(signals: np.ndarray, raw: mne.io.BaseRaw) -> np.ndarray:
    """Build 16-channel bipolar montage."""
    ch2idx = {name: i for i, name in enumerate(raw.info["ch_names"])}
    def g(name: str) -> np.ndarray:
        return signals[ch2idx[name]]

    new_signals = np.vstack((
        g("EEG FP1-REF") - g("EEG F7-REF"),
        g("EEG F7-REF")  - g("EEG T3-REF"),
        g("EEG T3-REF")  - g("EEG T5-REF"),
        g("EEG T5-REF")  - g("EEG O1-REF"),
        g("EEG FP2-REF") - g("EEG F8-REF"),
        g("EEG F8-REF")  - g("EEG T4-REF"),
        g("EEG T4-REF")  - g("EEG T6-REF"),
        g("EEG T6-REF")  - g("EEG O2-REF"),
        g("EEG FP1-REF") - g("EEG F3-REF"),
        g("EEG F3-REF")  - g("EEG C3-REF"),
        g("EEG C3-REF")  - g("EEG P3-REF"),
        g("EEG P3-REF")  - g("EEG O1-REF"),
        g("EEG FP2-REF") - g("EEG F4-REF"),
        g("EEG F4-REF")  - g("EEG C4-REF"),
        g("EEG C4-REF")  - g("EEG P4-REF"),
        g("EEG P4-REF")  - g("EEG O2-REF"),
    )).astype(np.float32)
    return new_signals


# ------------------------- EDF reader ------------------------- #

def read_edf(path: Path, fs: float, l_freq: float, h_freq: float, notch: float):
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    raw.resample(fs, npad="auto")
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose="ERROR")
    if notch > 0:
        raw.notch_filter(notch, verbose="ERROR")

    _, times = raw[:]
    signals = raw.get_data(units="uV")

    rec_path = path.with_suffix(".rec")
    event_data = np.genfromtxt(str(rec_path), delimiter=",")
    if event_data.ndim == 1:
        event_data = np.expand_dims(event_data, axis=0)

    return signals, times, event_data, raw


# ------------------------- save helper ------------------------- #

def save_pickle(obj, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


# ------------------------- main routine ------------------------- #

def export_split(base_dir: Path, out_dir: Path, fs: float, l_freq: float, h_freq: float, notch: float):
    edf_files = [p for p in base_dir.rglob("*.edf")]
    edf_files.sort()

    for edf in tqdm(edf_files, desc=f"Processing {base_dir.name}"):
        try:
            signals, times, event_data, raw = read_edf(edf, fs, l_freq, h_freq, notch)
            signals = convert_signals(signals, raw)
            feats, off_ch, labels = build_events(signals, times, event_data, fs=fs)
        except Exception as e:
            print(f"[WARN] Skip {edf}: {e}")
            continue
        finally:
            try: raw.close()
            except Exception: pass

        stem = edf.stem
        for idx, (sig, oc, lab) in enumerate(zip(feats, off_ch, labels)):
            sample = {
                "signal": sig.astype(np.float32),
                "offending_channel": oc.astype(np.int32),
                "label": lab.astype(np.int32),
            }
            out_path = out_dir / f"{stem}-{idx}.pkl"
            save_pickle(sample, out_path)


# ------------------------- CLI ------------------------- #

def parse_args():
    p = argparse.ArgumentParser("TUEV Preprocessing (no train/val split)")
    p.add_argument("--root", type=str, required=True, help="Path to TUH EEG Events edf/ root")
    p.add_argument("--target", type=str, required=True, help="Output folder for processed pickles")
    p.add_argument("--fs", type=float, default=200.0, help="Resample frequency (Hz)")
    p.add_argument("--l_freq", type=float, default=0.3, help="Bandpass lower bound (Hz)")
    p.add_argument("--h_freq", type=float, default=75.0, help="Bandpass upper bound (Hz)")
    p.add_argument("--notch", type=float, default=60.0, help="Notch frequency (0=disable)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = Path(args.root)
    target = Path(args.target)

    train_in  = root / "train"
    eval_in   = root / "eval"

    train_out = target / "processed_train"
    eval_out  = target / "processed_eval"

    export_split(train_in, train_out, args.fs, args.l_freq, args.h_freq, args.notch)
    export_split(eval_in,  eval_out,  args.fs, args.l_freq, args.h_freq, args.notch)

    print("Done!")
