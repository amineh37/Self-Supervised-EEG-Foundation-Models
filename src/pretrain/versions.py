"""
Named pretraining versions for STELAR-improved.

Each version is a set of *improvement* flags passed to LitSTELARImproved. The
backbone / objective / MTPE-GB are identical across versions -- only WHERE the
masking pressure is applied (#1) and WHAT the encoder sees (#2) change.

Diagnostic-first protocol (from the brief -- "let the empirics decide"):
  1. run v1_attnmask alone AND v2_aug alone (same seeds/splits).
  2. if v1-alone barely helps  -> do NOT combine; ship Option B (v2 pushed).
  3. if BOTH clearly help alone -> additivity plausible; Option A (vA_combo),
     tuned SEQUENTIALLY (fix #2, search #1's tau, then lightly retouch #2).

Mapping to the brief's options:
  * Option A (top leaderboard)   -> vA_combo
  * Option B (defensible paper)  -> v2_aug
  * #1 isolated diagnostic       -> v1_attnmask
  * faithful PDF reference       -> v0_baseline

VICReg (anti-collapse, 3rd MTPE-GB task) is OFF by default everywhere; enable it
only if the logged teacher/student embedding std collapses. Append "+vicreg" via
the --use_vicreg CLI flag (works on any version).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict

# Each entry holds ONLY improvement kwargs for LitSTELARImproved.
# (Base/model kwargs come from configs.py + CLI, as in the baseline runner.)
VERSIONS: Dict[str, Dict] = {
    # ---- faithful PDF baseline: no augmentation, random/block mask, 2-task ----
    "v0_baseline": dict(
        use_aug=False,
        use_attn_mask=False,
        use_vicreg=False,
    ),
    # ---- #1 only: teacher-attention-guided masking (Gumbel-top-k + tau curric) ----
    "v1_attnmask": dict(
        use_aug=False,
        use_attn_mask=True,
        attn_tau_start=1.0,
        attn_tau_min=0.1,
        attn_tau_epochs=15,
        attn_imp_floor=0.05,
        use_vicreg=False,
    ),
    # ---- #2 only (Option B): asymmetric augmented views, random mask ----
    "v2_aug": dict(
        use_aug=True,
        use_attn_mask=False,
        # teacher_aug=None -> WEAK (identity); student_aug=None -> STRONG menu
        use_vicreg=False,
    ),
    # ---- Option A: #1 + #2 combined (tune SEQUENTIALLY, not jointly) ----
    "vA_combo": dict(
        use_aug=True,
        use_attn_mask=True,
        attn_tau_start=1.0,
        attn_tau_min=0.1,
        attn_tau_epochs=15,
        attn_imp_floor=0.05,
        use_vicreg=False,
    ),
}

# Convenience aliases
VERSIONS["optionA"] = deepcopy(VERSIONS["vA_combo"])
VERSIONS["optionB"] = deepcopy(VERSIONS["v2_aug"])


def list_versions():
    return sorted(VERSIONS.keys())


def get_version(name: str) -> Dict:
    if name not in VERSIONS:
        raise KeyError(f"Unknown version '{name}'. Available: {list_versions()}")
    return deepcopy(VERSIONS[name])
