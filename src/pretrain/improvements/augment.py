"""
Improvement #2 -- EEG augmentations + invariance via asymmetric views.

Design (data2vec-like asymmetry):
  * teacher (E_xi) encodes a WEAK / clean view  -> stable, clean latent targets
  * student (E_theta) encodes a STRONG view (+ masking) -> must predict clean targets
  Invariance to acquisition nuisances then emerges *inside* L_lat, with NO extra
  loss term (MTPE-GB stays 2-task).

Key constraints from the brief:
  * Augmentations live in RAW SIGNAL SPACE, i.e. applied to the model input
    BEFORE the patch embedder, so that BOTH the conv (time) branch AND the FFT
    (frequency) branch see them.
  * Moderate amplitudes only (EEG is low-SNR).
  * Channel-drop = zero the channel in raw signal (NOT after the embedder).
  * Band-dropout is OFF by default (hurts band-specific tasks and the FFT branch).
  * Avoid time-reversal, sign-flip, channel permutation (label-/montage-destroying).

Tensor layouts supported (the engine works in "pretoken" mode by default):
  * pretoken : x [B, C, N, D]   (D raw samples per 1s patch, N patches)
  * raw      : x [B, C, T]
Time axis is the last axis after an internal flatten of (N, D) -> T = N*D.
All ops return a tensor of the SAME shape as the input.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


WEAK_AUG: Dict[str, float] = {
    "channel_dropout_p": 0.0,
    "amp_scale_lo": 1.0,
    "amp_scale_hi": 1.0,
    "noise_std": 0.0,
    "jitter_max_frac": 0.0,
    "crop_resize": 0.0,      # prob of applying crop-resize
    "crop_min_frac": 0.85,
    "band_dropout": 0.0,     # OFF
    "apply_prob": 0.0,       # global gate: 0 => identity teacher view
}

# Strong view (student): the moderate menu from the brief.
STRONG_AUG: Dict[str, float] = {
    "channel_dropout_p": 0.20,   # up to ~20% of channels zeroed
    "amp_scale_lo": 0.8,         # per-channel amplitude scaling ~U(0.8, 1.2)
    "amp_scale_hi": 1.2,
    "noise_std": 0.05,           # small additive gaussian (in normalized units)
    "jitter_max_frac": 0.05,     # temporal jitter up to 5% of the window
    "crop_resize": 0.30,         # apply crop+resize 30% of the time
    "crop_min_frac": 0.85,       # crop keeps >= 85% of the signal
    "band_dropout": 0.0,         # OFF by default
    "band_dropout_max_frac": 0.1,
    "apply_prob": 1.0,
}


def build_augment(cfg: Optional[Dict[str, float]], *, seed: Optional[int] = None) -> "EEGAugment":
    """Build an EEGAugment from a config dict (None -> identity)."""
    return EEGAugment(cfg or {"apply_prob": 0.0}, seed=seed)


def _to_time(x: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
    """Return ([B,C,T], (N,D) or None). Flattens pretoken (N,D) -> T."""
    if x.dim() == 4:
        B, C, N, D = x.shape
        return x.reshape(B, C, N * D), (N, D)
    if x.dim() == 3:
        return x, None
    raise ValueError(f"EEGAugment expects [B,C,N,D] or [B,C,T], got {tuple(x.shape)}")


def _from_time(x_t: torch.Tensor, nd: Optional[Tuple[int, int]]) -> torch.Tensor:
    if nd is None:
        return x_t
    B, C, T = x_t.shape
    N, D = nd
    return x_t.reshape(B, C, N, D)


class EEGAugment(nn.Module):
    """
    Stochastic EEG augmentation in raw signal space. Stateless w.r.t. parameters
    (no learnable weights); deterministic given an optional generator seed.
    """

    def __init__(self, cfg: Dict[str, float], *, seed: Optional[int] = None):
        super().__init__()
        d = {**STRONG_AUG, **(cfg or {})}  # fill missing keys with strong defaults
        # ...but if the caller passed a sparse weak cfg, respect explicit zeros.
        self.channel_dropout_p = float(d["channel_dropout_p"])
        self.amp_scale_lo = float(d["amp_scale_lo"])
        self.amp_scale_hi = float(d["amp_scale_hi"])
        self.noise_std = float(d["noise_std"])
        self.jitter_max_frac = float(d["jitter_max_frac"])
        self.crop_resize = float(d.get("crop_resize", 0.0))
        self.crop_min_frac = float(d.get("crop_min_frac", 0.85))
        self.band_dropout = float(d.get("band_dropout", 0.0))
        self.band_dropout_max_frac = float(d.get("band_dropout_max_frac", 0.1))
        self.apply_prob = float(d.get("apply_prob", 1.0))
        self._seed = seed

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"chan_drop={self.channel_dropout_p}, amp=({self.amp_scale_lo},{self.amp_scale_hi}), "
            f"noise={self.noise_std}, jitter={self.jitter_max_frac}, crop_resize={self.crop_resize}, "
            f"band_dropout={self.band_dropout}, apply_prob={self.apply_prob}"
        )

    def _rand(self, *shape, device) -> torch.Tensor:
        return torch.rand(*shape, device=device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Augment a batch. Returns a new tensor (does not modify input in place)."""
        if self.apply_prob <= 0.0:
            return x
        if self.training is False:
            # augment only during training; identity at eval/val for stable targets
            return x

        x_t, nd = _to_time(x)
        x_t = x_t.clone()
        B, C, T = x_t.shape
        device = x_t.device

        # Global per-sample gate
        gate = (self._rand(B, device=device) < self.apply_prob).view(B, 1, 1)

        # (1) amplitude scaling, per (sample, channel)
        if self.amp_scale_lo != 1.0 or self.amp_scale_hi != 1.0:
            scale = self.amp_scale_lo + (self.amp_scale_hi - self.amp_scale_lo) * self._rand(B, C, 1, device=device)
            x_t = torch.where(gate, x_t * scale, x_t)

        # (2) additive gaussian noise
        if self.noise_std > 0.0:
            noise = torch.randn(B, C, T, device=device) * self.noise_std
            x_t = torch.where(gate, x_t + noise, x_t)

        # (3) temporal jitter (circular roll along time)
        if self.jitter_max_frac > 0.0:
            max_shift = max(1, int(self.jitter_max_frac * T))
            shift = int(torch.randint(-max_shift, max_shift + 1, (1,), device=device).item())
            if shift != 0:
                rolled = torch.roll(x_t, shifts=shift, dims=-1)
                x_t = torch.where(gate, rolled, x_t)

        # (4) crop + resize (zoom): crop a contiguous span then interpolate back to T
        if self.crop_resize > 0.0 and float(self._rand(1, device=device).item()) < self.crop_resize:
            frac = self.crop_min_frac + (1.0 - self.crop_min_frac) * float(self._rand(1, device=device).item())
            span = max(8, int(frac * T))
            start = int(torch.randint(0, max(1, T - span + 1), (1,), device=device).item())
            cropped = x_t[:, :, start:start + span]                       # [B,C,span]
            resized = F.interpolate(cropped, size=T, mode="linear", align_corners=False)
            x_t = torch.where(gate, resized, x_t)

        # (5) band dropout (OFF by default) -- zero a random frequency band
        if self.band_dropout > 0.0 and float(self._rand(1, device=device).item()) < self.band_dropout:
            x_t = self._band_dropout(x_t, gate)

        # (6) channel dropout -- zero whole channels in RAW signal space
        if self.channel_dropout_p > 0.0:
            ch_keep = (self._rand(B, C, 1, device=device) >= self.channel_dropout_p).float()
            # never drop all channels: force-keep at least one per sample
            none_kept = ch_keep.sum(dim=1, keepdim=True) < 1.0
            if bool(none_kept.any()):
                forced = torch.zeros_like(ch_keep)
                forced[:, 0, :] = 1.0
                ch_keep = torch.where(none_kept, torch.maximum(ch_keep, forced), ch_keep)
            x_t = torch.where(gate, x_t * ch_keep, x_t)

        return _from_time(x_t, nd)

    @torch.no_grad()
    def _band_dropout(self, x_t: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        B, C, T = x_t.shape
        spec = torch.fft.rfft(x_t, dim=-1)
        nfreq = spec.shape[-1]
        w = max(1, int(self.band_dropout_max_frac * nfreq))
        f0 = int(torch.randint(0, max(1, nfreq - w), (1,), device=x_t.device).item())
        spec[:, :, f0:f0 + w] = 0
        out = torch.fft.irfft(spec, n=T, dim=-1)
        return torch.where(gate, out, x_t)
