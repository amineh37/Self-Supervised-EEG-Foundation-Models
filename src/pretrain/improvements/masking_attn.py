"""
Improvement #1 -- masking guided by the teacher's attention.

Instead of uniform random masking, sample the patches to MASK proportionally to
their *semantic importance*, estimated from the attention RECEIVED in the last
block of the teacher (which already does a full forward to produce targets).

Why teacher attention and NOT raw variance:
  variance mixes semantics with subject idiosyncrasy; masking by variance would
  push the encoder to learn subject-specific signatures (bad for generalization).
  Attention received is a semantic-saliency proxy.

Mechanics:
  * combine the two Criss-Cross maps (spatial across channels, temporal across
    time), each min-max normalized, then sum -> importance [N, C]
  * gradient is detached (importance is a sampling prior, not a trained signal)
  * sample via Gumbel-top-k with temperature tau:
        perturbed = log(importance) + tau * Gumbel
        masked = top-k(perturbed)
    high tau -> noise dominates -> ~uniform random masking
    low  tau -> importance dominates -> mask the most salient patches
  * curriculum: tau decays (cosine) from tau_start to a NON-ZERO floor tau_min
    over ~15 epochs (handled by the engine via current_tau()).

Extraction is done with non-invasive forward-pre-hooks on the teacher's last
block, so pretrain_modeling.py is left unmodified. If anything is unexpected
(missing modules, shape surprise, NaNs) we fall back to UNIFORM importance,
which makes #1 degrade gracefully to plain random masking.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

try:  # apply_rotary_emb lives in the modeling module
    from pretrain_modeling import apply_rotary_emb
except Exception:  # pragma: no cover - allows isolated import / linting
    apply_rotary_emb = None  # type: ignore


# ----------------------------------------------------------------------------- #
def _norm01(t: torch.Tensor) -> torch.Tensor:
    """Min-max normalize each batch element (dim 0) to [0, 1]."""
    B = t.shape[0]
    flat = t.reshape(B, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    flat = (flat - mn) / (mx - mn + 1e-8)
    return flat.view_as(t)


def _enforce_not_fully_masked(mask: torch.Tensor) -> torch.Tensor:
    """Guarantee >=1 visible token per time-row and per channel-col."""
    N, C = mask.shape
    device = mask.device
    all_masked_t = mask.all(dim=1)
    if all_masked_t.any():
        for t in torch.nonzero(all_masked_t, as_tuple=False).flatten():
            c = torch.randint(0, C, (1,), device=device)
            mask[t, c] = False
    all_masked_c = mask.all(dim=0)
    if all_masked_c.any():
        for c in torch.nonzero(all_masked_c, as_tuple=False).flatten():
            t = torch.randint(0, N, (1,), device=device)
            mask[t, c] = False
    return mask


@torch.no_grad()
def teacher_attention_importance(
    encoder: torch.nn.Module,
    x: torch.Tensor,
    chan_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return importance map [N, C] (detached), higher = more attended by the teacher.
    Falls back to a uniform map on any failure.

    `encoder` is the (EMA) target encoder; `x` is the WEAK/clean teacher view
    [B, C, N, D] (pretoken) or [B, C, T] (raw). Importance is averaged over batch
    because masking in this codebase is shared across the batch.
    """
    device = x.device
    C, N = encoder.num_patches  # encoder stores (C, N)
    uniform = torch.ones(N, C, device=device)
    if apply_rotary_emb is None:
        return uniform

    # Locate the last block and its attention module(s)
    try:
        if getattr(encoder, "use_criss", False) and hasattr(encoder, "cc_stack"):
            blk = encoder.cc_stack.blocks[-1]
            mods = {"c": blk.attn_c, "t": blk.attn_t}
        elif hasattr(encoder, "blocks") and len(encoder.blocks) > 0:
            blk = encoder.blocks[-1]
            mods = {"full": blk.attn}
        else:
            return uniform
    except Exception:
        return uniform

    captured: dict = {}
    handles = []

    def make_hook(key: str):
        def hook(module, args, kwargs):
            xin = args[0] if args else kwargs.get("x")
            freqs = kwargs.get("freqs", None)
            if xin is not None:
                captured[key] = (xin.detach(), None if freqs is None else freqs.detach())
        return hook

    for key, m in mods.items():
        handles.append(m.register_forward_pre_hook(make_hook(key), with_kwargs=True))

    full_keep = torch.arange(N * C, device=device, dtype=torch.long)
    was_training = encoder.training
    encoder.eval()
    try:
        encoder(x, chan_ids, mask_x=full_keep)
    except Exception:
        for h in handles:
            h.remove()
        if was_training:
            encoder.train()
        return uniform
    for h in handles:
        h.remove()
    if was_training:
        encoder.train()

    def attn_received(attn, xin, freqs):
        # xin [Bt, T, Dh] -> [Bt, T] attention received per key token
        q = attn._shape_q(xin)
        k, _ = attn._shape_kv(xin)
        if getattr(attn, "use_rope", False) and freqs is not None:
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)  # [Bt,H,T,T]
        return scores.mean(dim=1).mean(dim=1)  # mean over heads, then queries -> [Bt, T]

    try:
        B = x.shape[0]
        if "c" in captured and "t" in captured:
            xc, fc = captured["c"]                       # [B*N, C, Dh]
            recv_c = attn_received(blk.attn_c, xc, fc).view(B, N, C)
            xt, ft = captured["t"]                       # [B*C, N, Dh]
            recv_t = attn_received(blk.attn_t, xt, ft).view(B, C, N).permute(0, 2, 1)  # [B,N,C]
            imp = (_norm01(recv_c) + _norm01(recv_t)).mean(dim=0)  # [N, C]
        elif "full" in captured:
            xf, ff = captured["full"]                    # [B, N*C, D]
            recv = attn_received(blk.attn, xf, ff)        # [B, N*C]
            imp = _norm01(recv).mean(dim=0).view(N, C)
        else:
            return uniform
    except Exception:
        return uniform

    imp = imp.detach()
    if not torch.isfinite(imp).all():
        return uniform
    return imp


def sample_mask_attn_guided(
    importance: torch.Tensor,
    mask_ratio: float,
    tau: float,
    *,
    imp_floor: float = 0.05,
    generator: Optional[torch.Generator] = None,
):
    """
    Gumbel-top-k masking. Returns (mask_x, mask_y) as LongTensors of flat indices
    (t*C + c), matching LitSTELAR._sample_masks conventions:
      mask_x = VISIBLE/kept indices, mask_y = MASKED indices.
    """
    device = importance.device
    N, C = importance.shape
    T = N * C
    k_mask = int(round(float(mask_ratio) * T))
    k_mask = max(1, min(T - 1, k_mask))

    logits = torch.log(importance.reshape(-1).clamp_min(float(imp_floor)))  # [T]
    u = torch.rand(T, device=device, generator=generator).clamp_(1e-8, 1.0 - 1e-8)
    gumbel = -torch.log(-torch.log(u))
    perturbed = logits + float(tau) * gumbel  # high tau -> ~uniform

    topk = torch.topk(perturbed, k_mask).indices
    mask = torch.zeros(N, C, dtype=torch.bool, device=device)
    mask.view(-1)[topk] = True
    mask = _enforce_not_fully_masked(mask)

    flat = torch.arange(T, device=device, dtype=torch.long).view(N, C)
    mask_y = flat[mask].reshape(-1)
    mask_x = flat[~mask].reshape(-1)
    return mask_x, mask_y


def cosine_tau(epoch: int, tau_start: float, tau_min: float, tau_epochs: int) -> float:
    """Cosine decay from tau_start to tau_min over tau_epochs, then stays at tau_min."""
    e = max(0, min(int(epoch), int(tau_epochs)))
    if tau_epochs <= 0:
        return float(tau_min)
    cos = 0.5 * (1.0 + math.cos(math.pi * e / float(tau_epochs)))
    return float(tau_min) + (float(tau_start) - float(tau_min)) * cos
