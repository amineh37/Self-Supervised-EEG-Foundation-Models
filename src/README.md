# stelar-tsalm-2026
Official implementation of [Balanced Latent Semantics and Signal Fidelity for EEG representation learning](https://openreview.net/forum?id=ofwQZjoI4c) at 1st ICLR Workshop on Time Series in the Age of Large Models
## Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretraining](#pretraining)
- [Downstream Tasks](#downstream-tasks)
- [Project Structure](#project-structure)

## Installation

### Prerequisites

- Conda (Miniconda or Anaconda)
- GPU with CUDA 11.8+ (recommended)

### Setup Conda Environment

```bash
conda env create -f environment.yml
conda activate stelar
```

## Data Preparation

### Directory Structure

Set up your data directories:

```bash
export DATA_PATH=/path/to/your/data  # Set this to your data location
export PROJECT_ROOT=$(pwd)
```

### Preprocessing: Two Paths

This project supports two types of data preprocessing:

1. **Pretraining Data Preprocessing** - For preparing large-scale EEG data (e.g., TUEG) for self-supervised pretraining
2. **Downstream Task Preprocessing** - For preparing specific downstream task datasets (BCIC2a, FACED, TUAB, etc.)

#### Pretraining Data Preprocessing

Prepare large-scale EEG datasets for pretraining:

```bash
cd $PROJECT_ROOT/preprocessing/pretrain

# Example: Prepare TUEG dataset subsets for pretraining

python prepare_pretrain_tueg_subset.py \
  --in-root $DATA_PATH/tuh_eeg_seperated \
  --datasets tuh_eeg_artifact,tuh_eeg_epilepsy,tuh_eeg_seizure,tuh_eeg_slowing \
  --allowed-montages 01_tcp_ar \
  --out-lmdb-train $DATA_PATH/STELAR/data/pretrain/train_90pct \
  --out-lmdb-val   $DATA_PATH/STELAR/data/pretrain/val_10pct \
  --fraction-train 0.90 --fraction-val 0.10 --split-seed 12345 \
  --fs 200 --l-freq 0.3 --h-freq 75 --mains 60 \
  --window-sec 30 --edge-trim-sec 60 --min-len-sec 300 \
  --amp-max 100 --dtype float32 --workers 8 

# This creates preprocessed data in LMDB format for efficient loading
```

#### Downstream Task Preprocessing

Prepare datasets for downstream task fine-tuning:

```bash
cd $PROJECT_ROOT/preprocessing/downstream

# BCIC2a
python prepare_bcic2a_stelar.py 

# FACED
python prepare_face_stelar.py 

# TUAB
python prepare_tuab.py 

# TUEV
python prepare_tuev_cbramod.py 

# KaggleERN
python preprocessing_kaggle.py

# PhysioMI
python prepare_physiomi.py

# SleepEDF
python prepare_sleepedf.py
```

## Pretraining

### Quick Start

Run a simple pretraining experiment:

```bash
cd $PROJECT_ROOT/pretrain

# Configure environment variables (optional)
export LOG_ROOT=./logs
export PRETRAIN_DIR=$(pwd)

# Run pretraining (CPU/single GPU)
python run_pretrain.py \
  --input_mode pretoken \
  --data_backend lmdb \
  --tag tiny \
  --mask_ratio 0.5 \
  --max_epochs 30 
```

### Quick local test (single LMDB, split into train/val on the fly)

If you only have **one** LMDB (e.g. a small `val_10pct` slice) and just want to
verify the training script runs end-to-end, point `--train_lmdb` at it, **omit**
`--valid_lmdb`, and pass `--val_fraction` — `get_dataloaders`
(`pretrain/configs.py`) then splits that single LMDB into train/val at load time.
On a CPU-only machine set `--devices ""` and cap the batches with
`--limit_train_batches` / `--limit_val_batches`:

```bash
cd $PROJECT_ROOT/pretrain
python run_pretrain_improved.py --preset v0_baseline \
  --input_mode pretoken --data_backend lmdb \
  --train_lmdb ../val_10pct --val_fraction 0.1 \
  --devices "" --num_workers 2 --batch_size 8 \
  --montage 19 --tag tiny --transformer crisscross --mask_ratio 0.5 \
  --max_epochs 1 --limit_train_batches 5 --limit_val_batches 2 \
  --log_root ./logs/local_test

# or simply (defaults to ../val_10pct on CPU):
./run_pretrain_local.sh
```

### Using SLURM (HPC Clusters)

For cluster execution with SLURM:

```bash
cd $PROJECT_ROOT/pretrain
sbatch run_pretrain.sh
```

Edit `run_pretrain.sh` to customize:
- `--partition=YOUR_PARTITION`: Set your SLURM partition
- `--gres=gpu:N`: Number of GPUs
- `--time=HH:MM:SS`: Job time limit

### Pretraining Parameters

Key parameters:

- `--input_mode`: Data input format (`pretoken`, `raw`)
- `--tag`: Experiment tag (used in logging)
- `--mask_ratio`: Masking ratio (0.0-1.0)
- `--transformer`: Transformer type (`standard`, `crisscross`)
- `--max_epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--seed`: Random seed for reproducibility

See `pretrain/run_pretrain.py` for all available options.

## Downstream Tasks

### Supported Datasets

- BCIC2a: Motor imagery competition
- FACED: Facial expression recognition
- Kaggle ERP: Event-related potentials
- PhysioMI: PhysioNet imagination
- SleepEDF: Sleep staging
- TUAB: Brain activity abnormality
- TUEV: Brain activity variability

### Running Downstream Tasks

#### Option 1: Direct Python Script

```bash
cd $PROJECT_ROOT

# Set environment variables
export PROJECT_ROOT=$(pwd)
export DATA_PATH=/path/to/your/data
export LOG_ROOT=./logs
export PRETRAIN_DIR=/path/to/pretrain/checkpoints

# Run downstream task
python downstream/run_downstream.py \
  --task BCIC2a \
  --scheme stelar_patch_adapt \
  --split loso \
  --pretrain_ckpt checkpoints/model.pth \
  --bcic2a_path $DATA_PATH/BCIC2a_processed/stelar \
  --batch_size 64 \
  --epochs 100 \
  --seed 123 \
  --log_root $LOG_ROOT
```

#### Option 2: Using SLURM Scripts

Pre-configured scripts are available in `downstream/scripts/`:

```bash
# Example: STELAR on FACED
cd $PROJECT_ROOT

# Edit the script to set environment variables
export PROJECT_ROOT=$(pwd)
export DATA_PATH=/path/to/data
export LOG_ROOT=./logs

# Submit job
sbatch downstream/scripts/single_run_stelar_faced.sh

# Other available models/datasets:
# downstream/scripts/single_run_stelar_*.sh (STELAR)
# downstream/scripts/single_run_cbramod_*.sh (CBraMod)
# downstream/scripts/single_run_csbrain_*.sh (CSBrain)
# downstream/scripts/single_run_eegpt_*.sh (EEGPT)
# downstream/scripts/single_run_labram_*.sh (LaBraM)
```

### Downstream Task Parameters

Common parameters:

- `--task`: Dataset name (BCIC2a, FACED, TUAB, TUEV, KAGGLE, PHYSIOMI, SLEEPEDF)
- `--scheme`: Model backbone (stelar_patch_adapt, stelar, cbramod, csbrain, eegpt, labram)
- `--pretrain_ckpt`: Path to pretrained checkpoint
- `--split`: Evaluation split (loso, kfold, etc.)
- `--batch_size`: Batch size
- `--epochs`: Number of epochs
- `--seed`: Random seed

## Citation

If you use this code in your research, please cite the paper.
```bibtex
@inproceedings{
nguyen2026balanced,
title={Balanced Latent Semantics and Signal Fidelity for {EEG} representation learning},
author={Van-Chien Nguyen and Trung-Hieu Tran and Tuan-Kiet Doan and Quang Hung Pham and Ngoc-Son Vu and Duc Han LE and Huy Phan and Phi Le Nguyen and Nikola Simidjievski and Samuel Tardieu and Van-Tam Nguyen},
booktitle={1st ICLR Workshop on Time Series in the Age of Large Models},
year={2026},
url={https://openreview.net/forum?id=ofwQZjoI4c}
}
```
## License

See [LICENSE](LICENSE) for details.
