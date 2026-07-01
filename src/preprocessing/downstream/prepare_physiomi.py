import os
import pickle
import lmdb
import numpy as np
import mne

# -----------------------------
# Config
# -----------------------------
tasks = ['04', '06', '08', '10', '12', '14']   # motor imagery runs

root_dir = '/projects/EEG-foundation-model/PhysioNetMI/files'
out_db  = '/projects/EEG-foundation-model/STELAR/data/downstream/PhysioMI/processed_cbramod'

selected_channels = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
    'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
    'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
    'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
    'O1..', 'Oz..', 'O2..', 'Iz..'
]

sfreq_target = 200.0          # Hz sample to 256Hz for eegpt
epoch_len_s  = 4.0            # seconds  -> T = 800 samples at 200 Hz
hp_cut       = 0.3            # Hz
notch_hz     = 60             # US mains
map_size     = 5_000_000_000  # ~5 GB, bump if needed
commit_every = 256            # batch commits for speed

# -----------------------------
# Discover subjects and split
# -----------------------------
files = [
    f for f in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, f)) and f.startswith('S') and f[1:].isdigit()
]
files = sorted(files)

# Only train + eval; leave (train,val) split to the dataloader later
files_dict = {
    'train': files[:70],
    'eval' : files[90:109],   # combine old val+test (adjust if your corpus size differs)
}

print("Subject split:", files_dict)

# For index lists per split
dataset_keys = {'train': [], 'eval': []}

# -----------------------------
# Open LMDB
# -----------------------------
db = lmdb.open(out_db, map_size=map_size, subdir=True, lock=True)
txn = db.begin(write=True)
written_since_commit = 0

def put_sample(txn, key, value):
    global written_since_commit
    txn.put(key=key.encode(), value=pickle.dumps(value))
    return 1

try:
    for split_name, subject_list in files_dict.items():
        for subj in subject_list:
            for task in tasks:
                edf_path = os.path.join(root_dir, subj, f'{subj}R{task}.edf')
                if not os.path.isfile(edf_path):
                    print(f"[WARN] Missing file: {edf_path}")
                    continue

                # ---- Load & preprocess ----
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

                # channel subset + ordering
                raw.pick_channels(selected_channels, ordered=True)

                # interpolate if 'bads' already marked in the file
                if len(raw.info.get('bads', [])) > 0:
                    raw.interpolate_bads()

                # average reference
                raw.set_eeg_reference(ref_channels='average')

                # filter & notch
                raw.filter(l_freq=hp_cut, h_freq=None, verbose=False)
                raw.notch_filter(freqs=[notch_hz], verbose=False)

                # resample
                raw.resample(sfreq_target, npad="auto", verbose=False)

                # events / epoching: 4.0 s windows starting at event onset
                events_from_annot, event_dict = mne.events_from_annotations(raw, verbose=False)

                epochs = mne.Epochs(
                    raw,
                    events_from_annot,
                    event_dict,
                    tmin=0.0,
                    tmax=epoch_len_s - 1.0 / raw.info['sfreq'],
                    baseline=None,
                    preload=True,
                    verbose=False,
                )

                data = epochs.get_data(units='uV')     # (N, C, T)
                events = epochs.events[:, 2]           # (N,)
                # Ensure (C, T)
                # At 200 Hz and 4 s, T should be 800
                # Keep entire epoch -> no patchify
                # Skip "rest" if it is coded as 1, as in your original code
                for i, (sample, ev) in enumerate(zip(data, events)):
                    if ev == 1:
                        continue

                    # Label normalization to keep compact class IDs
                    if task in ['04', '08', '12']:
                        label = ev - 2
                    else:
                        label = ev

                    # sample: (C, T)
                    if sample.ndim != 2:
                        # (C, T) expected; if got (C, T) ok; (C,) or other -> skip
                        sample = np.asarray(sample)
                        if sample.ndim != 2:
                            print(f"[WARN] Unexpected sample shape {sample.shape} in {subj}R{task}-{i}, skipping.")
                            continue

                    key = f'{subj}R{task}-{i}'
                    sample = sample.astype(np.float32, copy=False)
                    data_dict = {'sample': sample, 'label': label}

                    written_since_commit += put_sample(txn, key, data_dict)
                    dataset_keys[split_name].append(key)

                    if written_since_commit >= commit_every:
                        txn.commit()
                        txn = db.begin(write=True)
                        written_since_commit = 0

    # write split index
    txn.put(key='__keys__'.encode(), value=pickle.dumps(dataset_keys))
    txn.commit()

finally:
    db.close()

print("Done. LMDB at:", out_db)
print({k: len(v) for k, v in dataset_keys.items()})
