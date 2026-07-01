#!/bin/bash
#SBATCH --job-name=prepare_faced
#SBATCH --output=logs/prepare_faced_%j.out
#SBATCH --error=logs/prepare_faced_%j.err
#SBATCH --partition=CPU
#SBATCH --time=1-00:00:00

# ------------------ Environment ------------------
echo "[`date`] Loading environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegpt

# Increase number of open file handles
ulimit -n 8192

# ------------------ Config ------------------
ROOT_DIR="/projects/EEG-foundation-model/faced/Processed_data"
# OUT_DB="/projects/EEG-foundation-model/STELAR/data/downstream/faced_stelar_no_zscore"
OUT_DB="/projects/EEG-foundation-model/STELAR/data/downstream/faced_eegpt"
TARGET_FS=200          # change (e.g., 128, 256)
USE_ZSCORE=0           # 1 = zscore ON, 0 = zscore OFF

# ------------------ Run ------------------
echo "[`date`] Starting dataset preparation..."
CMD=(python prepare_face_stelar.py --root_dir "$ROOT_DIR" --out_db "$OUT_DB" --target_fs "$TARGET_FS")

if [ "$USE_ZSCORE" -eq 0 ]; then
  CMD+=(--no_zscore)
fi

echo "[`date`] Running: ${CMD[@]}"
srun "${CMD[@]}"

echo "[`date`] Done."
