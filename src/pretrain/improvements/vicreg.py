"""
Optional VICReg-style anti-collapse term.

Stacking augmentation-invariance (#2) and hard masking (#1) increases pressure
toward degenerate (collapsed) solutions. The brief prescribes:
  * always LOG the mean per-dimension variance of the teacher embeddings;
  * if it collapses, add a cheap VICReg variance term -> MTPE-GB becomes 3-task
    (which usefully stress-tests the paper's acknowledged 2-task limitation).

Design choice: the teacher E_xi is an EMA copy with no gradient, so a loss on it
cannot train anything. We therefore apply the VICReg *variance* hinge to the
STUDENT online embeddings (z_enc, which carry gradient) -- the standard place to
prevent collapse -- while still LOGGING the teacher variance for monitoring.
Covariance/invariance VICReg terms are intentionally omitted: invariance is
already provided by L_lat, and the variance hinge alone is the cheap anti-collapse.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _flatten_tokens(z: torch.Tensor) -> torch.Tensor:
    """[B, N, C, D] or [B, T, D] or [B, D] -> [M, D]."""
    if z.dim() >= 2:
        return z.reshape(-1, z.shape[-1])
    raise ValueError(f"expected >=2D embedding, got {tuple(z.shape)}")


def embedding_std(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Mean over dims of per-dimension std of the embeddings (monitoring scalar)."""
    x = _flatten_tokens(z)
    std = torch.sqrt(x.var(dim=0) + eps)
    return std.mean()


def vicreg_variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    """
    VICReg variance hinge: encourage each embedding dimension to keep std >= gamma.
        L_var = mean_d relu(gamma - std_d)
    Applied to student embeddings to prevent representational collapse.
    """
    x = _flatten_tokens(z)
    std = torch.sqrt(x.var(dim=0) + eps)
    return F.relu(float(gamma) - std).mean()
