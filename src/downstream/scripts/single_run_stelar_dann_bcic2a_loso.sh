#!/bin/bash
# Domain-Adversarial (DANN) fine-tuning of STELAR on BCIC-2A (leave-one-subject-out).
# Run from the project root (architecture/). Equivalent to the YAML config at
# configs/downstream/dann_bcic2a_loso.yaml.
#
# Required:
#   export DATA_PATH=/path/to/BCIC2a_stelar          # LMDB from preprocessing/downstream
#   export PRETRAIN_CKPT=/path/to/stelar_base.ckpt   # STELAR base checkpoint
# Optional:
#   export LOG_ROOT=./logs
#   add  --wandb  (and `wandb login`) to log to Weights & Biases.
set -eo pipefail

: "${DATA_PATH:?set DATA_PATH to the BCIC2a LMDB}"
: "${PRETRAIN_CKPT:?set PRETRAIN_CKPT to the STELAR base checkpoint}"
: "${LOG_ROOT:=./logs}"

python src/downstream/run_bcic2a.py \
  --scheme stelar_dann \
  --tag base --variant D --preprocess rep --encoder full_token \
  --split loso \
  --val_ratio 0.1 \
  --bcic2a_path "${DATA_PATH}" \
  --pretrain_ckpt "${PRETRAIN_CKPT}" \
  --scale_div 100.0 \
  --devices 0 \
  --batch_size 64 \
  --epochs 100 \
  --head_type eegpt_probe \
  --dropout 0.5 \
  --seq_len 800 \
  --patch 50 \
  --transformer crisscross \
  --rope_time --no-rope_chan \
  --use_chan_conv \
  --chan_max_norm 1.0 \
  --lambda_adv 1.0 \
  --schedule_alpha \
  --grl_gamma 10.0 \
  --subj_disc_hidden 64 \
  --freeze_encoder \
  --grad_clip 1.0 \
  --seed 42 \
  --num_workers 4 \
  --log_root "${LOG_ROOT}"
