#!/bin/bash
#SBATCH --job-name=prepare_pretrain_tuh_eeg_subsets
#SBATCH --output=logs/prepare_pretrain_tuh_eeg_subsets%j.out
#SBATCH --error=logs/prepare_pretrain_tuh_eeg_subsets%j.err
#SBATCH --partition=CPU               # Change to your partition (e.g., V100/A100/P100)
#SBATCH --time=5-00:00:00            # Max run time (DD-HH:MM:SS)

# ------------------ Environment ------------------
echo "[`date`] Loading environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegpt

# Increase number of open file handles
ulimit -n 8192

python prepare_pretrain_tueg_subset.py \
  --in-root /projects/EEG-foundation-model/tuh_eeg_seperated \
  --datasets tuh_eeg_artifact,tuh_eeg_epilepsy,tuh_eeg_seizure,tuh_eeg_slowing \
  --allowed-montages 01_tcp_ar \
  --out-lmdb-train /projects/EEG-foundation-model/STELAR/data/pretrain/tuh_eeg_subsets_small_mapsize/train_90pct \
  --out-lmdb-val   /projects/EEG-foundation-model/STELAR/data/pretrain/tuh_eeg_subsets_small_mapsize/val_10pct \
  --fraction-train 0.90 --fraction-val 0.10 --split-seed 12345 \
  --fs 200 --l-freq 0.3 --h-freq 75 --mains 60 \
  --window-sec 30 --edge-trim-sec 60 --min-len-sec 300 \
  --amp-max 100 --dtype float32 --workers 8 
  # --disable-amp-filter
