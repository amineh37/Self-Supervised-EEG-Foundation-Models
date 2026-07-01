"""
stelar_dann_ft.py
=================
STELAR fine-tuning with Domain-Adversarial Neural Network (DANN).

Architecture
------------
                 ┌──────────────────────────────────────────────────┐
  EEG [B,C,T] ──► chan_conv ──► patching ──► proj_in ──► encoder z  │
                 │                           [B,N,C,D]              │
                 │          ┌──── mean(N,C) ──► z_pool [B,D]        │
                 │          │                     │                  │
                 │          │              GradientReversalLayer     │
                 │          │                     │                  │
                 │          │            subject_discriminator       │
                 │          │             → subject logits           │
                 │          │                                        │
                 │          └──── task_head ──► task logits          │
                 └──────────────────────────────────────────────────┘

Total loss = task_loss + λ_adv * adversarial_subject_loss

The GRL reverses gradients during backprop so the encoder learns
subject-INVARIANT representations while still solving the task.

Batch format
------------
  (x, y)              → task loss only  (graceful fallback)
  (x, y, subject_id)  → task + λ_adv * adversarial loss

GRL alpha schedule (from DANN paper)
-------------------------------------
  alpha(p) = 2 / (1 + exp(-gamma * p)) - 1,  p = epoch / max_epochs
  Starts at 0, smoothly reaches ≈1 at p=1.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stelar_patch_adapt_ft import LitSTELARPatchAdapt


# ============================================================
# 1. Gradient Reversal Layer
# ============================================================

class _GradReverseFn(torch.autograd.Function):
    """Custom autograd function that negates gradients during backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward pass: identity.
    Backward pass: multiply gradients by -alpha.

    The alpha attribute is mutable so it can be updated each epoch
    to follow the DANN progressive schedule.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradReverseFn.apply(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha:.4f}"


# ============================================================
# 2. DANN Module
# ============================================================

class LitSTELARDANN(LitSTELARPatchAdapt):
    """
    STELAR fine-tuning with Domain-Adversarial training (DANN).

    Inherits the full STELAR patch-adapt stack and adds:
      - A Gradient Reversal Layer (GRL)
      - A subject discriminator head
      - A progressive alpha schedule for GRL
      - A combined loss: task_loss + lambda_adv * adv_loss

    Parameters
    ----------
    num_subjects : int
        Number of subjects in the training set.  For BCIC2a LOSO
        this is (n_total_subjects - 1) per fold.
    grl_alpha : float
        Fixed GRL alpha (used only when schedule_alpha=False).
    lambda_adv : float
        Weight of the adversarial loss term.
    schedule_alpha : bool
        If True, use DANN-style progressive alpha schedule.
    grl_gamma : float
        Steepness of the sigmoid schedule (default 10 per DANN paper).
    subj_disc_hidden : int
        Hidden dimension of the subject discriminator MLP.
    subj_disc_dropout : float
        Dropout rate inside the subject discriminator.
    **kwargs
        All other keyword arguments forwarded to LitSTELARPatchAdapt.
    """

    def __init__(
        self,
        models_cfg: Dict[str, Any],
        load_path: Optional[str],
        *,
        # DANN-specific
        num_subjects: int = 9,
        grl_alpha: float = 1.0,
        lambda_adv: float = 1.0,
        schedule_alpha: bool = True,
        grl_gamma: float = 10.0,
        subj_disc_hidden: int = 64,
        subj_disc_dropout: float = 0.3,
        # pass-through to LitSTELARPatchAdapt
        **kwargs,
    ):
        super().__init__(models_cfg, load_path, **kwargs)

        self._num_subjects = int(num_subjects)
        self._lambda_adv = float(lambda_adv)
        self._schedule_alpha = bool(schedule_alpha)
        self._grl_gamma = float(grl_gamma)
        self._grl_alpha_init = float(grl_alpha)

        # GRL with mutable alpha (updated at each epoch)
        self.grl = GradientReversalLayer(alpha=grl_alpha)

        # Subject discriminator: Linear → ReLU → Dropout → Linear
        self.subject_discriminator = nn.Sequential(
            nn.Linear(self.D_enc, subj_disc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(subj_disc_dropout),
            nn.Linear(subj_disc_hidden, self._num_subjects),
        )

    # ----------------------------------------------------------
    # Alpha schedule
    # ----------------------------------------------------------

    def _current_alpha(self) -> float:
        """
        Returns the current GRL alpha.
        Progressive schedule: alpha grows from 0 → ~1 as training progresses.
        """
        if not self._schedule_alpha:
            return self._grl_alpha_init
        p = self.current_epoch / max(self._max_epochs, 1)
        return float(2.0 / (1.0 + math.exp(-self._grl_gamma * p)) - 1.0)

    # ----------------------------------------------------------
    # Core encoding (shared logic)
    # ----------------------------------------------------------

    def _encode_with_pool(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the full encoder stack.

        Returns
        -------
        x_raw      : original input tensor  [B, C, T]
        z_pooled   : encoder output mean-pooled over (N, C)  [B, D_enc]
        task_logits: task classification logits  [B, num_classes]
        """
        # Encoder train/eval mode consistent with fine-tune policy
        if any(p.requires_grad for p in self.target_encoder.parameters()):
            self.target_encoder.train()
        else:
            self.target_encoder.eval()

        xa = x  # [B, C, T]  or  [B, C, N, patch]

        # (1) Optional 1x1 channel-mixing conv over time axis
        if self.use_chan_conv and self.chan_conv is not None:
            if xa.dim() == 3:
                xa = self.chan_conv(xa)                         # [B, target_chans, T]
            elif xa.dim() == 4:
                B, C, N, D = xa.shape
                xa_flat = xa.reshape(B, C, N * D)
                xa_flat = self.chan_conv(xa_flat)
                xa = xa_flat.reshape(B, self.target_chans, N, D)

        # (2) Ensure token shape [B, C, N, patch]
        if xa.dim() == 3:
            B, C, T = xa.shape
            assert T % self.patch == 0, (
                f"Signal length T={T} must be divisible by patch size {self.patch}"
            )
            xa = xa.unfold(dimension=2, size=self.patch, step=self.patch).contiguous()
        elif xa.dim() == 4:
            B, C, N, P = xa.shape
            assert P == self.patch, (
                f"Expected last dim == patch={self.patch}, got {P}"
            )

        # (3) Local projection → encoder
        xa = self.proj_in(xa)               # [B, C, N, 200]
        B, C, N, D = xa.shape
        keep_all = torch.arange(N * C, device=xa.device, dtype=torch.long)

        z = self.target_encoder(
            xa,
            chan_ids=self.chans_id,
            mask_x=keep_all,
            mask_t=None,
        )  # [B, N, C, D_enc]

        # (4) Mean-pool over patch-time (N) and channel (C) dims → subject repr
        z_pooled = z.mean(dim=(1, 2))       # [B, D_enc]

        # (5) Task head
        task_logits = self.head(z)          # [B, num_classes]

        return x, z_pooled, task_logits

    # ----------------------------------------------------------
    # forward — API-compatible with parent
    # ----------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Returns (x_raw, task_logits) — same interface as LitSTELARPatchAdapt.
        Used during validation and test steps (inherited from parent).
        """
        x_raw, _, task_logits = self._encode_with_pool(x)
        return x_raw, task_logits

    # ----------------------------------------------------------
    # Training step — handles 2-tuple and 3-tuple batches
    # ----------------------------------------------------------

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, subject_ids = batch
            subject_ids = subject_ids.long()
        else:
            x, y = batch
            subject_ids = None

        y = y.long()

        x_raw, z_pooled, task_logits = self._encode_with_pool(x)

        # ── Task classification loss ──────────────────────────────
        task_loss = self._loss_fn(task_logits, y)

        # ── Adversarial subject-discrimination loss via GRL ───────
        adv_loss = torch.zeros(1, device=self.device, dtype=task_loss.dtype).squeeze()
        if subject_ids is not None:
            alpha = self._current_alpha()
            self.grl.alpha = alpha

            z_reversed = self.grl(z_pooled)                 # GRL: identity forward, negated grad backward
            subject_logits = self.subject_discriminator(z_reversed)
            adv_loss = F.cross_entropy(subject_logits, subject_ids)

            self.log("grl_alpha",      torch.tensor(alpha, device=self.device),
                     on_epoch=True, on_step=False, sync_dist=True)
            self.log("train_adv_loss", adv_loss,
                     on_epoch=True, on_step=False, sync_dist=True)

        # ── Combined loss ─────────────────────────────────────────
        total_loss = task_loss + self._lambda_adv * adv_loss

        pred = task_logits.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log("train_loss",      total_loss,     on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_task_loss", task_loss,      on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_accuracy",  accuracy,       on_epoch=True, on_step=False, sync_dist=True)
        self.log("data_avg",        x_raw.mean(),   on_epoch=True, on_step=False, sync_dist=True)
        self.log("data_std",        x_raw.std(),    on_epoch=True, on_step=False, sync_dist=True)

        return total_loss

    # ----------------------------------------------------------
    # Optimizer — includes subject_discriminator params
    # ----------------------------------------------------------

    def configure_optimizers(self):
        head_params: List[nn.Parameter] = []
        head_params += list(self.proj_in.parameters())
        head_params += list(self.head.parameters())
        head_params += list(self.subject_discriminator.parameters())
        if self.chan_conv is not None:
            head_params += list(self.chan_conv.parameters())

        enc_params = [p for p in self.target_encoder.parameters() if p.requires_grad]

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
        if enc_params:
            groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

        optimizer = torch.optim.AdamW(groups)

        if self._scheduler.lower() == "onecycle":
            max_lrs = [self._lr_head] + ([self._lr_enc] if enc_params else [])
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)
            if total_steps <= 0:
                total_steps = int(self._steps_per_epoch * self._max_epochs)
            total_steps = max(total_steps, int(self.global_step) + 1)

            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self._warmup_pct,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }

        return {"optimizer": optimizer}
