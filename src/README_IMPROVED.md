# STELAR-improved

This repo extends the **STELAR** EEG foundation model from
`new_stellar (1).pdf` (ICLR 2026 TSALM workshop — *"Balanced Latent Semantics
and Signal Fidelity"*). It keeps the **exact same layout** as
`stelar-tsalm-2026` (`preprocessing/ pretrain/ downstream/`) and adds an
**improvement layer** with several clearly-separated versions.

> **Honest status.** Code is syntax-checked (`py_compile`) and the pure-Python
> parts (version registry, τ curriculum) are runtime-checked. **It has NOT been
> trained** here: this environment has no `torch`/GPU/TUEG data. Treat the
> improvement results as *to be measured* — see the decision protocol below.

---

## 0. What is faithful to the PDF (the baseline, v0)

The PDF model is reproduced **verbatim** by the files copied from
`stelar-tsalm-2026` (the wired engine `pretrain/pretrain_engine_w_mtpe_gb.py`):

- Criss-Cross encoder + RoPE(time) + learnable channel embeddings
  (`pretrain/pretrain_modeling.py`)
- dual-branch time (conv) + frequency (FFT-MLP) patch embedder
- EMA **target** encoder `E_ξ`, online **student** `E_θ`, tiny **predictor**
  `P_φ`, linear **reconstructor** `R_ψ`
- dual-space objective **`L_total = w_lat·L_lat + w_rec·L_rec`** (2 tasks)
- **MTPE-GB** validation-driven gradient balancer
- 50% masking, 30s windows, 19 channels @ 200 Hz

`run_pretrain_improved.py --preset v0_baseline` ≡ the original `run_pretrain.py`.

---

## 1. Data: TUEG → ~500h (unchanged from stelar-tsalm)

Identical pipeline — reused verbatim in `preprocessing/`:

```bash
cd preprocessing/pretrain
python prepare_pretrain_tueg_subset.py \
  --in-root $DATA/tuh_eeg_seperated \
  --datasets tuh_eeg_artifact,tuh_eeg_epilepsy,tuh_eeg_seizure,tuh_eeg_slowing \
  --allowed-montages 01_tcp_ar \
  --out-lmdb-train $DATA/STELAR/data/pretrain/train_90pct \
  --out-lmdb-val   $DATA/STELAR/data/pretrain/val_10pct \
  --fraction-train 0.90 --fraction-val 0.10 --split-seed 12345 \
  --fs 200 --l-freq 0.3 --h-freq 75 --mains 60 \
  --window-sec 30 --edge-trim-sec 60 --min-len-sec 300 \
  --amp-max 100 --dtype float32 --workers 8
```

Cleaning steps (edge-trim, 19-channel select, 0.3–75 Hz + 60 Hz notch, 200 Hz
resample, 30s non-overlap windows, >100 µV rejection). Final `[-1,1]` scaling is
applied at load time (`pretrain/configs.py`, `norm_fixed_scale=0.01`).

---

## 2. Improvement layer (NEW) — `pretrain/improvements/`

Two axes from the design brief. **Neither adds a loss term** — they plug into the
existing `L_lat` (so MTPE-GB stays 2-task), unless the optional VICReg
anti-collapse term is switched on (then 3-task).

| File | What it changes |
|---|---|
| `improvements/augment.py` (**#2**) | *What the encoder sees.* Asymmetric views: teacher encodes a **weak/clean** view (stable targets), student encodes a **strong-augmented + masked** view. Raw-signal-space augmentations (so conv **and** FFT branches see them): channel-dropout (≤20%, zeroed in raw signal), amplitude scaling ~U(0.8,1.2)/channel, additive noise, temporal jitter, crop-resize. **Band-dropout OFF** by default. No time-reversal/sign-flip/channel-permutation. |
| `improvements/masking_attn.py` (**#1**) | *Where masking pressure applies.* Mask patches ∝ **teacher last-block attention** (semantic saliency, **not** variance), via Gumbel-top-k with temperature τ, detached gradient, cosine **τ curriculum** (high→low, non-zero floor). Attention extracted by non-invasive forward-hooks; falls back to uniform masking on any failure. |
| `improvements/vicreg.py` (optional) | VICReg **variance hinge** on student embeddings (anti-collapse) → enabled as MTPE-GB's 3rd task. Also always **logs** teacher/student embedding std for monitoring. |

Engine: `pretrain/pretrain_engine_improved.py` (`LitSTELARImproved`) subclasses the
faithful baseline and gates all of the above behind flags. Critical safeguards:
- #1 importance is computed on the **teacher (weak) view only**, never on the
  strong/corrupted student view (mitigates the #1×#2 friction).
- reconstruction & latent **targets are the CLEAN signal**; the student predicts
  them from the augmented+masked view (data2vec-like asymmetry).

---

## 3. Versions (`pretrain/versions.py`)

| Preset | #2 aug | #1 attn-mask | VICReg | Role |
|---|---|---|---|---|
| `v0_baseline` | ✗ | ✗ | ✗ | faithful PDF reference |
| `v1_attnmask` | ✗ | ✓ | ✗ | **#1 isolated** (diagnostic) |
| `v2_aug` | ✓ | ✗ | ✗ | **#2 isolated** = **Option B** (defensible paper) |
| `vA_combo` | ✓ | ✓ | ✗ | **Option A** (#1+#2, top leaderboard) |

`--use_vicreg` can be added to any preset (only if embedding std collapses).

### Run

```bash
cd pretrain
COMMON="--input_mode pretoken --data_backend lmdb \
  --train_lmdb $DATA/STELAR/data/pretrain/train_90pct \
  --valid_lmdb $DATA/STELAR/data/pretrain/val_10pct \
  --tag tiny --transformer crisscross --mask_ratio 0.5 --max_epochs 30 --seed 7"

python run_pretrain_improved.py --preset v0_baseline $COMMON
python run_pretrain_improved.py --preset v1_attnmask $COMMON
python run_pretrain_improved.py --preset v2_aug      $COMMON
python run_pretrain_improved.py --preset vA_combo    $COMMON --attn_tau_start 1.0 --attn_tau_min 0.1
# if teacher_emb_std / student_emb_std collapse in the logs:
python run_pretrain_improved.py --preset vA_combo $COMMON --use_vicreg
```

### Quick local test — single LMDB, split on the fly

To smoke-test the pipeline on **one** LMDB (e.g. the bundled `val_10pct`) with no
separate validation set, point `--train_lmdb` at it, **omit** `--valid_lmdb`, and
pass `--val_fraction`: `get_dataloaders` (`pretrain/configs.py`) then carves the
validation split out of that single LMDB at load time via
`torch.utils.data.random_split` (triggered when `valid_lmdb is None and
val_fraction > 0`). On a CPU-only machine set `--devices ""` and cap the batches
with `--limit_train_batches` / `--limit_val_batches` (or `--fast_dev_run`) so it
finishes in seconds:

```bash
cd pretrain
python run_pretrain_improved.py --preset v0_baseline \
  --input_mode pretoken --data_backend lmdb \
  --train_lmdb ../val_10pct --val_fraction 0.1 \
  --devices "" --num_workers 2 --batch_size 8 \
  --montage 19 --img_c 19 --img_n 30 --patch_size 200 \
  --tag tiny --transformer crisscross --mask_ratio 0.5 \
  --max_epochs 1 --limit_train_batches 5 --limit_val_batches 2 \
  --log_root ./logs/local_test

# equivalent one-liner (defaults to ../val_10pct on CPU):
./run_pretrain_local.sh
```

`--montage 19` matches the 19-channel TUEG preprocessing; the `(C=19, N=30,
D=200)` shapes are auto-inferred from the data if you drop the
`--img_*`/`--patch_size` overrides.

---

## 3b. YAML config & Weights & Biases (NEW — additive, opt-in)

> **Project layout.** This code lives in `architecture/src/`; the **project root
> is `architecture/`** and all commands are run from there (e.g.
> `python src/pretrain/run_pretrain_improved.py ...`). YAML configs live in
> `architecture/configs/`, tests in `architecture/tests/`. The older `cd
> pretrain` / `cd downstream` style still works too — Python adds the launched
> script's directory to `sys.path`, so imports resolve either way.

The runners are unchanged: every command above still works. On top of the
existing CLI there is now an **optional** YAML config layer and a **W&B-capable**
logger, both opt-in. Precedence is:

```
CLI flag  >  YAML value  >  built-in argparse default
```

so a `--config` only *pre-fills* defaults and any explicit flag still wins.

### Ready-made configs (`configs/pretrain/`)

| File | Reproduces |
|---|---|
| `v0_baseline.yaml` | faithful PDF baseline, 30 epochs on the 500h LMDB |
| `v2_aug.yaml` | Option B (#2 augmentations) |
| `vA_combo.yaml` | Option A (#1 + #2) |
| `local_smoke.yaml` | CPU smoke run on synthetic `.pt` tensors (seconds) |

```bash
# from the project root (architecture/)
# drive a run entirely from YAML
python src/pretrain/run_pretrain_improved.py --config configs/pretrain/v2_aug.yaml
# YAML + a CLI override (CLI wins)
python src/pretrain/run_pretrain_improved.py --config configs/pretrain/v2_aug.yaml --max_epochs 50
```

Section names in the YAML (`data:`, `model:`, `trainer:`, `improvements:`,
`logging:`) are cosmetic — leaf keys map 1-to-1 to the CLI flags.

### Weights & Biases

W&B is **off by default** (file loggers stay TensorBoard + CSV). Enable it with
flags or YAML; it is **offline by default** when no `WANDB_API_KEY` is set, so a
run never blocks on login:

```bash
python src/pretrain/run_pretrain_improved.py --config configs/pretrain/v2_aug.yaml --wandb            # online if logged in, else offline
python src/pretrain/run_pretrain_improved.py --config configs/pretrain/v2_aug.yaml --wandb --wandb_offline
wandb login                # to log online; sync offline runs later with `wandb sync`
```

The same `--logger {both,tensorboard,csv,wandb}` / `--wandb*` flags are also wired
into `src/downstream/run_downstream.py` (TB + CSV remain the default there too).

### Verify it runs (no GPU / no real data)

A clean-room smoke test lives in `tests/` — it generates synthetic pretoken
tensors and runs one capped epoch on CPU for both `v0_baseline` and `v2_aug`:

```bash
pip install pytest && pytest        # run from architecture/ ; or: python -m pytest -v
```

---

## 4. Downstream finetuning / linear probing (unchanged interface)

The improvements add **no encoder parameters**, so an improved checkpoint's
`target_encoder.*` is structurally identical to the baseline. The downstream
loader (`downstream/modules/stelar_patch_adapt_ft.py`) keys on `target_encoder.`
with `strict=False`, so improved checkpoints load directly:

```bash
python downstream/run_downstream.py \
  --task BCIC2a --scheme stelar_patch_adapt --split loso \
  --pretrain_ckpt <path-to-improved-checkpoint>.ckpt \
  --bcic2a_path $DATA/BCIC2a_processed/stelar \
  --batch_size 64 --epochs 100 --seed 123 --log_root ./logs
```

Same for the other datasets (FACED, KaggleERN, PhysioMI, SleepEDF, TUAB, TUEV)
via the scripts in `downstream/scripts/`.

---

## 5. Decision protocol — let the empirics decide (from the brief)

Additivity of #1+#2 is a **hypothesis**, not a guarantee. Before committing to
Option A or B:

1. Run **`v1_attnmask` alone** and **`v2_aug` alone** with the *same seeds/splits*.
2. If **#1-alone barely helps** → don't combine; ship **Option B** (`v2_aug`,
   pushed; produce the ablations: uniform-no-aug vs with-aug, asymmetric vs
   symmetric, embedding-std curve showing no collapse).
3. If **both clearly help alone** → additivity plausible → **Option A**
   (`vA_combo`), tuned **SEQUENTIALLY** (fix #2 at moderate defaults → search #1's
   τ/schedule → lightly retouch #2). **Never** tune the ~7 hyperparameters jointly.

| | Option A (`vA_combo`) | Option B (`v2_aug`) |
|---|---|---|
| Goal | max score | defensible paper |
| Extra loss term | 0 (VICReg only if collapse) | 0 (VICReg only if collapse) |
| MTPE-GB regime | 2-task (→3 if VICReg) | 2-task (→3 if VICReg) |
| Tuning cost | high, sequential mandatory | moderate |
| Main risk | collapse + #1×#2 friction | collapse (already instrumented) |
| Prerequisite to choose | #1-alone **and** #2-alone win | #2-alone wins |

Watch `teacher_emb_std` / `student_emb_std` in the logs; if they collapse, add
`--use_vicreg` (stress-tests the paper's acknowledged 2-task limitation).

---

## 6. Files added (everything else is the proven stelar-tsalm-2026 base)

```
pretrain/improvements/__init__.py
pretrain/improvements/augment.py          # #2
pretrain/improvements/masking_attn.py     # #1
pretrain/improvements/vicreg.py           # optional anti-collapse
pretrain/pretrain_engine_improved.py      # LitSTELARImproved (flags)
pretrain/versions.py                      # v0/v1/v2/vA registry
pretrain/run_pretrain_improved.py         # CLI driver (--preset / --config / --wandb)
README_IMPROVED.md                        # this file

# Engineering layer (NEW): YAML config + W&B logging + tests
stelar_utils/loggers.py                   # build_loggers / make_wandb_logger (offline-safe)
stelar_utils/yaml_config.py               # YAML -> argparse defaults (CLI overrides)
configs/pretrain/*.yaml                    # v0 / v2 / vA / local_smoke configs
tests/test_config_logging.py              # unit tests (YAML precedence, loggers)
tests/test_pretrain_smoke.py              # CPU end-to-end smoke (v0 + v2)
conftest.py, pytest.ini                    # test wiring
```

> The improvement results in `RESULTS_PRETRAIN_VAL10PCT.md` were produced by the
> runners above. The NEW engineering layer is **additive** (config/logging/tests
> only) and does not change any model / training / loss code, so it does not
> affect those results. Install with `conda env create -f environment.yml`.
