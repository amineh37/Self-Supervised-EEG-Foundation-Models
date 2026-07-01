"""
Minimal defaults for downstream fine-tuning (encoder-only).

- No dataset loaders, no pretrain selection, no predictor configs.
- Runner reads these constants as CLI defaults (override via flags).
- Architecture presets expose ONLY the encoder block needed by the modules.

This version assumes STELAR-style encoders use:
  - NO learned channel embedding (use_channel_embed=False)
  - ACPE / dynamic positional encoding (use_dynamic_pe=True)
"""

from __future__ import annotations
from typing import Dict, Optional, List

# ---------------------------
# Repro / run scale
# ---------------------------
SEED: int = 7
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 100

# Devices default (override via --devices)
devices: List[int] = [0]

# ---------------------------
# Optimization (head vs. encoder)
# ---------------------------
MAX_LR_HEAD: float = 4e-4
MAX_LR_ENC:  float = 1e-4
WD_HEAD: float = 0.01
WD_ENC:  float = 0.05

# Scheduler
SCHEDULER: str    = "onecycle"   # {"onecycle","cosine","none"}
WARMUP_PCT: float = 0.20

# ---------------------------
# Model fingerprint (used by runner to build the module)
# ---------------------------
tag: str         = "tiny"          # model size key (see MODELS_CONFIGS/EEGPT_MODELS_CONFIGS)
variant: str     = "D"             # optional freeform variant tag
preprocess: str  = "rep"           # optional freeform preprocessing tag
encoder: str     = "full_token"    # backend encoder name expected by your backbone
transformer: str = "crisscross"    # {"standard","crisscross"}

# ---------------------------
# Frontend / head defaults (downstream)
# ---------------------------
IN_CHANS: int       = 22
TARGET_CHANS: int   = 19
SEQ_LEN: int        = 1024
PATCH: int          = 64
DROPOUT: float      = 0.5

HEAD_TYPE: str      = "avg"        # {"avg","per_time_attn","flatten","eegpt_probe","all_patch_reps"}
POOL: str           = "mean"       # {"mean","max","meanmax"}

# ---------------------------
# Criss-Cross / RoPE / ACPE toggles (used by STELAR-style encoders)
# ---------------------------

# ACPE / dynamic positional encoding over (C,N) grid
USE_DYNAMIC_PE: bool       = False   # <-- ACPE ON for STELAR fine-tuning

# Whether to use learned channel embeddings inside the encoder
# For the "no channel embedding + ACPE" setup, keep this False.
USE_CHANNEL_EMBED: bool    = True

# RoPE / dual-axis attentions (if enabled in encoder)
ROPE_TIME_DEFAULT: bool        = True
ROPE_CHAN_DEFAULT: bool        = False
ROPE_DIM_FRAC_DEFAULT: float   = 1.0
ROPE_INTERP_TIME_DEFAULT: float = 1.0
ROPE_INTERP_CHAN_DEFAULT: float = 1.0

# ---------------------------
# EEGPT (multi-stage) presets → encoder-only dict by get_config_eegpt(...)
# ---------------------------

EEGPT_MODELS_CONFIGS: Dict[str, Dict] = {
    # embed_dim, embed_num (slots), depth per stage (we pass only stage-1 depth to encoder)
    "tiny1":  {"embed_dim": 64,  "embed_num": 1, "depth": [2, 2, 4], "num_heads": 4},
    "tiny2":  {"embed_dim": 64,  "embed_num": 4, "depth": [2, 2, 4], "num_heads": 4},
    "tiny3":  {"embed_dim": 64,  "embed_num": 4, "depth": [8, 8, 8], "num_heads": 4},
    "little": {"embed_dim": 128, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 4},
    "base1":  {"embed_dim": 256, "embed_num": 1, "depth": [6, 6, 6], "num_heads": 4},
    "base2":  {"embed_dim": 256, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 4},
    "base3":  {"embed_dim": 512, "embed_num": 1, "depth": [6, 6, 6], "num_heads": 8},
    "large":  {"embed_dim": 512, "embed_num": 4, "depth": [8, 8, 8], "num_heads": 8},
}

def get_config_eegpt(
    *,
    embed_dim: int = 512,
    embed_num: int = 4,
    depth: List[int] = (8, 8, 8),
    num_heads: int = 4,
) -> Dict[str, Dict]:
    """
    Return {"encoder": {...}} for EEGPT-style models.
    We expose a single-stage encoder dict expected by the Lightning modules.
    """
    return {
        "encoder": {
            "embed_dim": embed_dim,
            "embed_num": embed_num,
            "depth": int(depth[0] if isinstance(depth, (list, tuple)) else depth),
            "num_heads": num_heads,
        }
    }

# ============================================================
# STELAR-style presets (single encoder block)
# ============================================================

MODELS_CONFIGS: Dict[str, Dict] = {
    "tiny":   {"encoder": {"embed_dim":  64, "num_heads": 2, "depth": 2}},
    "little": {"encoder": {"embed_dim": 128, "num_heads": 4, "depth": 8}},
    "base":   {"encoder": {"embed_dim": 256, "num_heads": 4, "depth": 8}},
    "large":  {"encoder": {"embed_dim": 512, "num_heads": 8, "depth": 8}},
    "huge":   {"encoder": {"embed_dim": 512, "num_heads": 8, "depth": 12}},
}

def get_config(tag_name: Optional[str] = None) -> Dict[str, Dict]:
    """
    Return {"encoder": {...}} for STELAR-style models.

    This folds in:
      - ACPE/dynamic positional encoding (USE_DYNAMIC_PE)
      - Channel embedding toggle (USE_CHANNEL_EMBED)
      - Criss-Cross vs standard transformer (from global `transformer` flag)
      - RoPE defaults over time/channel axes.
    """
    name = tag_name or tag
    cfg = MODELS_CONFIGS.get(name)
    if cfg is None:
        raise KeyError(f"Unknown tag '{name}'. Available: {list(MODELS_CONFIGS)}")

    enc = dict(cfg["encoder"])  # shallow copy

    # use_criss_cross comes from global transformer choice
    use_criss_cross = (transformer.lower() == "crisscross")

    enc.update(
        use_dynamic_pe=USE_DYNAMIC_PE,
        use_channel_embed=USE_CHANNEL_EMBED,
        use_criss_cross=use_criss_cross,
        rope_time=ROPE_TIME_DEFAULT,
        rope_chan=ROPE_CHAN_DEFAULT,
        rope_dim_frac=ROPE_DIM_FRAC_DEFAULT,
        rope_interp_time=ROPE_INTERP_TIME_DEFAULT,
        rope_interp_chan=ROPE_INTERP_CHAN_DEFAULT,
    )

    return {"encoder": enc}

# ---------------------------
# PhysioMI (subject-wise K-fold) split controls
# ---------------------------

# Path to the LMDB produced by your preprocessor (samples (C,T), __keys__ present)
PHYSIOMI_PATH: str = "/path/to/data/downstream/PhysioMI/processed_average"

# Fixed outer test fold (0..4), sizes ≈ 22/22/22/22/21 for 109 subjects
PHYSIOMI_TEST_FOLD_INDEX: int = 4

# Inner CV fold to run:
#   - set to None  -> run all 4 inner rounds (each non-test fold is val once)
#   - set to 0..3  -> run only that single inner round
PHYSIOMI_CV_FOLD_INDEX: Optional[int] = None

# Optional JSON with {"folds": {"0":[...],...,"4":[...]}, "test_fold_index": int}
# Leave as None to build folds deterministically from SEED.
PHYSIOMI_FOLDS_JSON: Optional[str] = None
