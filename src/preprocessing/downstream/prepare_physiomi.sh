#!/bin/bash
#SBATCH --job-name=prepare_tuab
#SBATCH --output=logs/prepare_physiomi_%j.out
#SBATCH --error=logs/prepare_physiomi_%j.err
#SBATCH --partition=CPU               # Change to your partition (e.g., V100/A100/P100)
#SBATCH --cpus-per-task=32         # Adjust based on need
#SBATCH --time=5-00:00:00           # Max run time (HH:MM:SS)

# ------------------ Environment ------------------
echo "[`date`] Loading environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegpt

# Increase number of open file handles
ulimit -n 8192

# ------------------ Run ------------------
echo "[`date`] Starting dataset preparation..."
srun python prepare_physiomi.py

echo "[`date`] Done."
