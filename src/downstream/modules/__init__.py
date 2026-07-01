# modules/__init__.py
from typing import Dict, Callable

from .stelar_patch_adapt_ft import LitSTELARPatchAdapt
from .stelar_dann_ft import LitSTELARDANN
from .stelar_patch_adapt_tuev_ft import LitSTELARPatchAdaptTUEV
from .stelar_summary_patch_adapt_ft import LitSTELARSummaryAdapt
from .eegpt_ft import LitEEGPT
from .eegpt_tuev_ft import LitEEGPTTUEV
from .cbramod_ft import LitCBraMod
from .heads import AvgPoolMLPHead, PerTimeAttnMLPHead, FlattenAllTokensMLPHead
from .base import LitBase

_REGISTRY: Dict[str, Callable] = {
    "stelar_patch_adapt": LitSTELARPatchAdapt,
    "stelar_dann": LitSTELARDANN,
    "stelar_patch_adapt_tuev": LitSTELARPatchAdaptTUEV,
    "stelar_summary_patch_adapt": LitSTELARSummaryAdapt,
    "eegpt": LitEEGPT,
    "eegpt_tuev": LitEEGPTTUEV,
    "cbramod": LitCBraMod,
}

def build_module(name: str):
    """Factory for LightningModules by string key."""
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"Unknown module '{name}'. Available: {list(_REGISTRY.keys())}") from e
