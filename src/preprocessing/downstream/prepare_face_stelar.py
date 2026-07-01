import argparse
import os
import lmdb
import pickle
import numpy as np
import scipy.signal as signal
from sklearn.model_selection import KFold, train_test_split

# ----------------- ARGS ----------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="FACED Dataset -> LMDB with optional Z-Score & 4-Fold Split")

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the folder containing .pkl subject files",
    )
    parser.add_argument(
        "--out_db",
        type=str,
        required=True,
        help="Output path for the LMDB folder",
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=1099511627776 * 2,
        help="Maximum size of LMDB (default ~2TB)",
    )

    # Toggle z-score (default: ON, use --no_zscore to disable)
    parser.add_argument(
        "--no_zscore",
        action="store_true",
        help="Disable per-window channel-wise z-score normalization (default: z-score ON).",
    )

    # Sampling-rate / resampling controls
    parser.add_argument(
        "--target_fs",
        type=int,
        default=200,
        help="Target sampling rate (Hz) after resampling. Default: 200.",
    )
    parser.add_argument(
        "--trial_secs",
        type=int,
        default=30,
        help="Trial duration in seconds (FACED is typically 30s). Used with --target_fs. Default: 30.",
    )
    parser.add_argument(
        "--chunk_secs",
        type=int,
        default=10,
        help="Chunk duration in seconds (FACED code uses 3 chunks of 10s). Default: 10.",
    )

    return parser.parse_args()

# ----------------- HELPER ----------------- #
def zscore_2d(x: np.ndarray) -> np.ndarray:
    """Normalize (C, T) array channel-wise."""
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    return (x - mean) / (std + 1e-8)

# ----------------- MAIN ----------------- #
def main():
    args = parse_args()

    # Labels (Hardcoded as they are specific to FACED dataset structure)
    labels = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8])

    if args.trial_secs % args.chunk_secs != 0:
        raise ValueError(f"--trial_secs must be divisible by --chunk_secs (got {args.trial_secs} and {args.chunk_secs})")

    n_chunks = args.trial_secs // args.chunk_secs
    trial_len = args.trial_secs * args.target_fs                 # samples per 30s trial after resample
    seg_len = args.target_fs                                     # samples per 1s segment (we keep 30 segments)
    chunk_len = args.chunk_secs * args.target_fs                  # samples per chunk (10s)
    segs_per_chunk = args.chunk_secs                              # 10 segments of 1s per 10s chunk (kept by construction)

    if not os.path.exists(args.root_dir):
        raise FileNotFoundError(f"Root dir not found: {args.root_dir}")

    # 1. GET SUBJECT FILES
    files = sorted([f for f in os.listdir(args.root_dir) if not f.startswith(".")])
    file_to_keys_map = {f: [] for f in files}

    # 2. SETUP LMDB
    if not os.path.exists(args.out_db):
        os.makedirs(args.out_db)

    print(f"Opening LMDB at {args.out_db}...")
    db = lmdb.open(args.out_db, map_size=args.map_size)

    zscore_enabled = not args.no_zscore
    print(
        f"Processing {len(files)} subjects... "
        f"(z-score: {'ON' if zscore_enabled else 'OFF'}, target_fs={args.target_fs}Hz, "
        f"trial={args.trial_secs}s, chunk={args.chunk_secs}s)"
    )

    # 3. PROCESSING LOOP
    for file_idx, file in enumerate(files):
        try:
            file_path = os.path.join(args.root_dir, file)
            with open(file_path, "rb") as f:
                array = pickle.load(f)

            # Resample along time axis to target length = trial_secs * target_fs
            eeg = signal.resample(array, trial_len, axis=2)  # (28, 32, trial_len)

            # Reshape to (Trials, Channels, Segments(=trial_secs), TimePerSeg(=target_fs))
            # Keeping the same "30 segments" structure, but each segment has target_fs samples now.
            eeg_ = eeg.reshape(28, 32, args.trial_secs, seg_len)  # (28, 32, 30, target_fs)

            txn = db.begin(write=True)

            for i, (samples, label) in enumerate(zip(eeg_, labels)):
                # samples: (32, 30, target_fs)
                for j in range(n_chunks):
                    # Extract chunk (32, chunk_secs, target_fs) -> Flatten to (32, chunk_secs*target_fs)
                    chunk_3d = samples[:, segs_per_chunk * j : segs_per_chunk * (j + 1), :]
                    chunk_2d = chunk_3d.reshape(32, -1)  # (32, chunk_len)

                    # Optional Z-Score
                    if zscore_enabled:
                        sample_out = zscore_2d(chunk_2d).astype(np.float32)
                    else:
                        sample_out = chunk_2d.astype(np.float32)

                    # Save
                    sample_key = f"{file}-{i}-{j}"
                    data_dict = {"sample": sample_out, "label": int(label), "fs": int(args.target_fs)}

                    txn.put(
                        key=sample_key.encode(),
                        value=pickle.dumps(data_dict, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    file_to_keys_map[file].append(sample_key)

            txn.commit()
            if (file_idx + 1) % 10 == 0:
                print(f"Processed {file_idx + 1} / {len(files)} files")

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # 4. CREATE SPLITS
    print("\nGenerating Subject-Wise Train/Val/Test Splits...")
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    dataset_split = {}

    for fold_num, (train_val_idx, test_idx) in enumerate(kf.split(files)):
        test_files = [files[k] for k in test_idx]
        train_val_files = [files[k] for k in train_val_idx]

        # Split Train into Train (80%) and Val (20%)
        train_files, val_files = train_test_split(train_val_files, test_size=0.2, random_state=42)

        def get_keys(flist):
            k = []
            for f in flist:
                k.extend(file_to_keys_map[f])
            return k

        dataset_split[f"fold_{fold_num}"] = {
            "train": get_keys(train_files),
            "val": get_keys(val_files),
            "test": get_keys(test_files),
        }
        print(
            f"Fold {fold_num} | Train: {len(train_files)} subj | "
            f"Val: {len(val_files)} subj | Test: {len(test_files)} subj"
        )

    txn = db.begin(write=True)
    txn.put(key="__keys__".encode(), value=pickle.dumps(dataset_split, protocol=pickle.HIGHEST_PROTOCOL))
    txn.commit()
    db.close()
    print("Done.")

if __name__ == "__main__":
    main()
