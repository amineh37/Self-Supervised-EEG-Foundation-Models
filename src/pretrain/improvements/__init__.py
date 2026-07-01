"""
STELAR-improved: improvement modules layered on top of the faithful PDF baseline.

Two improvement axes from the design brief, each plugging into the existing
dual-space objective (L_total = w_lat * L_lat + w_rec * L_rec) WITHOUT adding a
new loss term (so MTPE-GB stays in 2-task regime), unless the optional VICReg
anti-collapse term is enabled (then 3-task regime).

- improvement #1: teacher-attention-guided masking            -> masking_attn.py
- improvement #2: EEG augmentations + asymmetric views         -> augment.py
- optional      : VICReg variance anti-collapse (3rd task)     -> vicreg.py

See ../versions.py for the named presets (v0_baseline, v1_attnmask, v2_aug,
vA_combo) and ../README_IMPROVED.md for the Option A / Option B rationale.
"""

from .augment import EEGAugment, WEAK_AUG, STRONG_AUG, build_augment
from .masking_attn import teacher_attention_importance, sample_mask_attn_guided
from .vicreg import vicreg_variance_loss, embedding_std

__all__ = [
    "EEGAugment",
    "WEAK_AUG",
    "STRONG_AUG",
    "build_augment",
    "teacher_attention_importance",
    "sample_mask_attn_guided",
    "vicreg_variance_loss",
    "embedding_std",
]
