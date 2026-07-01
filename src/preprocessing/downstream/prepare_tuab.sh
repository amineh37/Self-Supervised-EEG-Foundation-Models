#!/bin/bash
#SBATCH --job-name=prepare_tuab
#SBATCH --output=logs/prepare_tuab%j.out
#SBATCH --error=logs/prepare_tuab%j.err
#SBATCH --partition=CPU
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate eegpt

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python prepare_tuab.py 
