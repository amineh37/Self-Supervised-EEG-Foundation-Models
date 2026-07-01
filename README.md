# STELAR

Self-supervised EEG foundation model (**STELAR**) with two improvements
(attention-guided masking, asymmetric augmented views), a downstream
finetuning/linear-probing suite, and a domain-adversarial (DANN) extension.

See [**report**](report/report.pdf) !

**Project root is this directory — run everything from here.**

```
architecture/
├── src/
│   ├── pretrain/        # self-supervised pretraining (+ improvements #1, #2, VICReg)
│   ├── downstream/      # finetuning / linear probing (FACED, TUEV, TUAB, BCIC-2A, PhysioMI, KaggleERN)
│   │   └── modules/     #   incl. stelar_dann_ft.py (DANN: GRL + subject discriminator)
│   ├── preprocessing/   # TUEG + downstream dataset preparation
│   └── stelar_utils/    # shared YAML-config + W&B logger layer
├── configs/
│   ├── pretrain/        # v0_baseline / v2_aug / vA_combo / local_smoke
│   └── downstream/      # faced / tuev / tuab / physiomi / kaggle / dann_bcic2a_loso
├── tests/               # pytest: CPU smoke + config/logger/DANN unit tests
├── report/              # project report (report.pdf / report.tex)
└── ressources/          # reference papers
```

## Install

```bash
conda env create -f src/environment.yml && conda activate stelar
```

## Configuration: YAML + CLI

Every runner takes an **optional** `--config x.yaml`. A YAML just pre-fills the
runner's arguments; explicit CLI flags still win. Precedence:

```
CLI flag  >  YAML value  >  built-in default
```

So a config captures a full, reproducible experiment in one tracked file, and a
sweep is just one overridden flag:

```bash
python src/pretrain/run_pretrain_improved.py --config configs/pretrain/v2_aug.yaml          # full config
python src/pretrain/run_pretrain_improved.py --config configs/pretrain/v2_aug.yaml --seed 1 # override
```

Scripts are launched by path; Python puts the script's own directory on
`sys.path`, so in-package imports resolve wherever you launch from.

## Pretraining

| Preset (`configs/pretrain/`) | What |
|---|---|
| `v0_baseline` | faithful STELAR reference |
| `v1_attnmask` *(preset only)* | improvement #1 (attention-guided masking) |
| `v2_aug` | improvement #2 (asymmetric augmented views) |
| `vA_combo` | #1 + #2 |
| `local_smoke` | CPU smoke run on synthetic data (seconds) |

```bash
python src/pretrain/run_pretrain_improved.py --config configs/pretrain/v2_aug.yaml
# or a bare preset:
python src/pretrain/run_pretrain_improved.py --preset v1_attnmask --train_lmdb $DATA/val_10pct --val_fraction 0.1 ...
```

## Downstream

All downstream runners accept `--config` too. Reuse a pretrained backbone via
`--pretrain_ckpt`.

| Config (`configs/downstream/`) | Runner | Task |
|---|---|---|
| `faced.yaml` | `run_faced.py` | emotion (FACED) |
| `tuev.yaml` | `run_tuev.py` | event classification (TUEV, k-fold) |
| `tuab.yaml` | `run_tuab.py` | normal/abnormal (TUAB) |
| `physiomi.yaml` | `run_physiomi.py` | motor imagery (PhysioMI) |
| `kaggle.yaml` | `run_kaggle.py` | error-related negativity (KaggleERN) |
| `dann_bcic2a_loso.yaml` | `run_bcic2a.py` | **DANN** on BCIC-2A (leave-one-subject-out) |

```bash
python src/downstream/run_faced.py --config configs/downstream/faced.yaml
# adaptation-strategy variants = override one flag:
python src/downstream/run_faced.py --config configs/downstream/faced.yaml --freeze_encoder
python src/downstream/run_faced.py --config configs/downstream/faced.yaml --ft_top_k 2 --head_type avg

# Domain-adversarial training (subject-invariant representations):
python src/downstream/run_bcic2a.py --config configs/downstream/dann_bcic2a_loso.yaml
```

Edit the `${DATA}` / `${CKPT}` placeholders in each YAML before running.

## Weights & Biases

W&B is **off by default** (file loggers stay TensorBoard + CSV) and **offline by
default** when no `WANDB_API_KEY` is set, so a run never blocks on login. Enable
per run:

```bash
python src/downstream/run_tuab.py --config configs/downstream/tuab.yaml --wandb
wandb login          # to log online; sync offline runs later with `wandb sync`
```

Flags (on every runner): `--logger {both,tensorboard,csv,wandb}`, `--wandb`,
`--wandb_offline`, `--wandb_project`, `--wandb_entity`, `--wandb_tags`, ...

## Verify it runs (no GPU / no real data)

```bash
pip install pytest && pytest
```

Generates synthetic EEG and runs one capped pretraining epoch on CPU (presets
v0 + v2), plus config/logger and DANN unit tests.

## More

- `src/README_IMPROVED.md` — research details (presets, improvement layer, decision protocol)
- `src/RESULTS_PRETRAIN_VAL10PCT.md` — measured pretraining metrics
- `report/report.pdf` — full project report
