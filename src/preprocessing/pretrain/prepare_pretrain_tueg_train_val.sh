#!/bin/bash
#SBATCH --job-name=prepare_pretrain_tueg
#SBATCH --output=logs/prepare_pretrain_tueg_%j.out
#SBATCH --error=logs/prepare_pretrain_tueg_%j.err
#SBATCH --partition=CPU               # Change to your partition (e.g., V100/A100/P100)
#SBATCH --time=4-00:00:00           # Max run time (HH:MM:SS)

# ------------------ Environment ------------------
echo "[`date`] Loading environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegpt

# Increase number of open file handles
ulimit -n 8192

# ------------------ Run ------------------
echo "[`date`] Starting dataset preparation..."
python prepare_pretrain_tueg_train_val.py \
  --in-root /projects/EEG-foundation-model/tuh_eeg/v2.0.1/edf \
  --force-rewrite-config \
  --pretrain-split \
  --train-out-lmdb /projects/EEG-foundation-model/STELAR/data/pretrain/tueg_processed_full/train_90pct \
  --val-out-lmdb   /projects/EEG-foundation-model/STELAR/data/pretrain/tueg_processed_full/val_10pct \
  --train-fraction 0.90 \
  --val-fraction   0.10 \
  --workers 32 --n-jobs 1
echo "[`date`] Done."