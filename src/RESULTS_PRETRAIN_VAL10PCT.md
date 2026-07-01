# STELAR-improved — Pretraining run on `val_10pct`

**Status:** ✅ complete — 3 presets × 30 epochs on CPU (`v0_baseline`, `v1_attnmask`, `v2_aug`)
**Date:** 2026-06-19 → 2026-06-20
**Run dir:** `pretrain/logs/pretrain_val10pct_full/`

---

## 1. Objective

Validate the STELAR-improved self-supervised pretraining pipeline end-to-end on a
real (if small) EEG corpus, and record all the metrics that matter for judging an
**EEG encoder pretrained with a data2vec / I-JEPA-style objective** (latent
alignment + masked reconstruction, weighted by MTPE-GB).

This is **not** a production checkpoint: it runs on CPU over a 10 % data slice. The
point is correctness, convergence behaviour, and **representation health
(no collapse)** — not leaderboard accuracy.

## 2. Hardware & environment

| Item | Value |
|---|---|
| Device | CPU only (no CUDA) |
| Cores / RAM | 16 vCPU / 15 GiB |
| Python | venv at `architecture/venv` |
| torch | 2.12.0 |
| pytorch_lightning | 2.6.5 |
| Precision | fp32 (`set_float32_matmul_precision("medium")`) |

## 3. Dataset

| Item | Value |
|---|---|
| Source | `stelar-improved/val_10pct/` (single LMDB, the 10 % TUEG validation slice) |
| Samples | 4 674 windows |
| Sample shape | `[C=19, N=30, D=200]` (19-ch montage, 30 patches, patch dim 200) |
| Preprocessing | TUEG → 19-ch `01_tcp_ar`, 0.3–75 Hz + 60 Hz notch, 200 Hz, 30 s windows, ±100 µV reject |
| Train/val split | **on the fly** — single LMDB carved 90/10 via `--val_fraction 0.1` (seed 7), `random_split` in `get_dataloaders` |
| Train / val sizes | 4 207 / 467 windows |
| Normalization | per-sample fixed scale ×0.01 (`norm_fixed_scale`) |

## 4. Model (`tag=tiny`) & objective

| Component | Config |
|---|---|
| Encoder | TransformerEncoder, embed_dim 64, depth 2, heads 2, **criss-cross** time×channel attention, RoPE-time, channel-embedding |
| Predictor | TransformerPredictor, embed_dim 64, depth 1, heads 2, criss-cross, KV mode `global_vis_kv` |
| Reconstructor | `MLPReconstructor`, embed_dim 64 |
| Target encoder | EMA copy of encoder (momentum 0.996 → 1.0) |
| Params | ~304 K total (191 K trainable + 112 K EMA target) |
| Objective | MTPE-GB weighted sum of **masked latent alignment** (`L_align`) + **masked reconstruction** (`L_recon`), mask ratio 0.5 |
| Preset | `v0_baseline` (faithful PDF baseline — no #1 attn-mask, no #2 aug, no VICReg) |

## 5. Training configuration

```bash
python run_pretrain_improved.py --preset v0_baseline \
  --input_mode pretoken --data_backend lmdb \
  --train_lmdb ../val_10pct --val_fraction 0.1 \
  --devices "" --num_workers 8 \
  --montage 19 --img_c 19 --img_n 30 --patch_size 200 \
  --tag tiny --transformer crisscross --mask_ratio 0.5 \
  --batch_size 64 --max_epochs 30 --seed 7 \
  --pretrain_max_lr 1e-4 \
  --log_root ./logs/pretrain_val10pct_full
```

| Hyperparam | Value |
|---|---|
| Optimizer | AdamW (engine default) |
| Peak LR | 1e-4, cosine decay with linear warmup (~10 % of steps) |
| Weight decay | cosine 1e-2 → 1e-2 |
| Batch size | 64 |
| Steps/epoch | 66 (train) / 8 (val) |
| Max epochs | 30 |
| Grad clip | 1.0 (norm) |
| EMA momentum | 0.996 → 1.0 |
| Seed | 7 |

## 6. Metrics glossary (why each matters for an EEG SSL encoder)

| Metric (CSV column) | Meaning | What "good" looks like |
|---|---|---|
| `train_loss` / `valid_loss` | total MTPE-GB-weighted objective | steadily ↓, val tracks train (no large gap) |
| `train_loss_recon` / `valid_loss_recon` | masked reconstruction MSE on raw patches | ↓ — encoder captures signal structure |
| `train_loss_align` / `valid_loss_align` | masked latent alignment (student vs EMA-teacher) | ↓ — predicted latents match target latents |
| `teacher_emb_std` | std of EMA-teacher embeddings | **stays > 0** — the anti-collapse canary |
| `student_emb_std` | std of student embeddings | **stays > 0**, not vanishing → no representation collapse |
| `w_recon` / `w_align_mask` | live MTPE-GB task weights | adapt smoothly, neither task starved |
| `mtpe_cur_*` / `mtpe_first_*` / `w_*_after` | MTPE-GB internal urgency / post-update weights | bounded, stable |
| `lr-AdamW` | learning rate | warmup ramp then cosine decay |
| epoch wall-time | throughput | — |
| best `valid_loss` / epoch | model-selection signal | — |

> **Collapse** is the dominant failure mode of latent SSL: the network maps every
> input to the same vector, driving the alignment loss to ~0 while the encoder
> learns nothing. `teacher_emb_std` / `student_emb_std` are the primary guardrails;
> a flat-positive (non-vanishing) std alongside a falling loss = healthy.

---

## 7. Results

The three presets were launched back-to-back with the **same config** (only
`--preset` changed), via:

```bash
for PRESET in v0_baseline v1_attnmask v2_aug; do
  python run_pretrain_improved.py --preset "$PRESET" \
    --input_mode pretoken --data_backend lmdb \
    --train_lmdb ../val_10pct --val_fraction 0.1 --devices "" --num_workers 8 \
    --montage 19 --img_c 19 --img_n 30 --patch_size 200 \
    --tag tiny --transformer crisscross --mask_ratio 0.5 \
    --batch_size 64 --max_epochs 30 --seed 7 --pretrain_max_lr 1e-4 \
    --log_root ./logs/pretrain_val10pct_full
done
```

Source of truth: `…/STELAR_improved_<preset>_tiny_crisscross_csv/version_*/metrics.csv`.

### 7.1 Comparison (validation metrics, 30 epochs)

| Metric | `v0_baseline` | `v1_attnmask` | `v2_aug` |
|---|---|---|---|
| Wall-time | ~73 min | ~75 min | ~81 min |
| **valid_loss — best (epoch)** | 0.7428 (ep5) | 0.7413 (ep4) | **0.7410** (ep4) |
| valid_loss — final (ep29) | 0.8116 | 0.8120 | 0.8114 |
| `valid_loss_recon` first → last | 1.2932 → 0.9961 | 1.2917 → 0.9961 | 1.2934 → 0.9961 |
| `valid_loss_align_mask` first → last | 0.9976 → 0.0416 | 0.9951 → **0.0384** | 0.9979 → 0.0395 |
| `student_emb_std` first → last | 0.894 → 0.243 | 0.893 → 0.236 | 0.883 → 0.228 |
| `teacher_emb_std` first → last (min) | 0.869 → 0.198 | 0.869 → 0.191 | 0.869 → 0.194 |
| `w_recon` / `w_align` (final, MTPE-GB) | 0.807 / 0.193 | 0.808 / 0.192 | 0.807 / 0.193 |
| `mask_tau` curriculum (#1 only) | — | 1.0 → 0.1 ✅ | — |
| Collapse | none | none | none |

### 7.2 Loss dynamics (all presets behave the same)

- **Alignment** (`L_align`) drops from ~1.0 to ~0.04 → the student matches the
  EMA-teacher latents almost perfectly. The core data2vec objective is met.
- **Reconstruction** (`L_recon`) plateaus at ~0.996 → the `tiny` MLP reconstructor
  cannot reconstruct raw EEG patches well at this scale (expected).
- **Total `valid_loss` rises after ep4–5 then plateaus (~0.81)** — *not* a
  regression: MTPE-GB shifts weight toward recon (`w_recon` 0.50 → 0.81) once align
  collapses, so the weighted sum goes up while both individual losses stay flat/↓.
- **No collapse**: `emb_std` contracts (~0.87 → ~0.20) but stays clearly > 0. The
  contraction is the usual baseline shrinkage; #1/#2 did **not** noticeably slow it.

### 7.3 Verdict

1. **Pipeline validated** — all three model scripts (`v0_baseline`, `v1_attnmask`,
   `v2_aug`) train end-to-end, converge cleanly, and show no collapse.
2. **#1 / #2 vs baseline: no measurable difference.** Best `valid_loss` differs by
   ~0.2 % (0.7428 / 0.7413 / 0.7410) — within noise. recon, align and emb_std are
   essentially identical across presets. **No winner at this scale.**
3. **Why:** `tiny` model + 10 % data + 30 CPU epochs is too small for #1/#2 to
   express any advantage — consistent with the README decision protocol ("if #1
   alone barely helps, don't combine").
4. **To actually compare the improvements:** scale up (`tag=little`/`base`,
   full `train_90pct`, more epochs, GPU) and judge on **downstream accuracy**, not
   pretraining loss.

---

## 8. Reproduce

```bash
cd stelar-improved/pretrain
# CPU full run (this experiment):
PY=../../venv/bin/python  # or your conda `stelar` python
$PY run_pretrain_improved.py --preset v0_baseline \
  --input_mode pretoken --data_backend lmdb \
  --train_lmdb ../val_10pct --val_fraction 0.1 \
  --devices "" --num_workers 8 \
  --montage 19 --img_c 19 --img_n 30 --patch_size 200 \
  --tag tiny --transformer crisscross --mask_ratio 0.5 \
  --batch_size 64 --max_epochs 30 --seed 7 \
  --pretrain_max_lr 1e-4 --log_root ./logs/pretrain_val10pct_full
```

## 9. Caveats

- CPU-only, 10 % data slice, `tiny` model → **smoke/convergence test**, not a usable
  foundation-model checkpoint.
- Validation is split from the same distribution as train (random 90/10 of one
  LMDB), so val measures fit, not transfer.
- For a real pretraining, use the full `train_90pct` + held-out `val_10pct` LMDBs on
  GPU (see `README_IMPROVED.md`).
