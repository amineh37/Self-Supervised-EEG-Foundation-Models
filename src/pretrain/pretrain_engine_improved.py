"""
STELAR-improved engine.

Subclasses the faithful PDF baseline (LitSTELAR from pretrain_engine_w_mtpe_gb)
and layers, behind explicit flags, the two improvement axes from the brief:

  #2  use_aug      : asymmetric augmented views (teacher = weak/clean view -> stable
                     latent + signal targets; student = strong view + masking).
                     Invariance emerges inside L_lat, NO new loss term.
  #1  use_attn_mask: teacher-attention-guided masking (Gumbel-top-k over the
                     teacher's last-block attention, cosine tau curriculum).
                     Replaces the random/block sampler, NO new loss term.

  optional use_vicreg: VICReg variance hinge on STUDENT embeddings to prevent
                     collapse. This (and only this) makes MTPE-GB 3-task.

Everything else (architecture, EMA teacher, predictor, reconstructor, MTPE-GB,
optimizer/schedule) is inherited unchanged from the baseline, so:
  * use_aug=False, use_attn_mask=False, use_vicreg=False  ==  v0 baseline (PDF).

IMPORTANT design notes (see README_IMPROVED.md):
  * #1 importance is computed on the TEACHER (weak) view only -- never on the
    strong/corrupted student view (the brief's mitigation against #1x#2 friction).
  * Reconstruction & latent targets come from the CLEAN teacher view; the student
    sees the augmented+masked view (data2vec-like asymmetry).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torchmetrics import MeanMetric

from pretrain_engine_w_mtpe_gb import (
    LitSTELAR,
    _get_x,
    _gather_tokens,
    _flat_to_bool_mask,
)
from improvements.augment import build_augment, WEAK_AUG, STRONG_AUG
from improvements.masking_attn import (
    teacher_attention_importance,
    sample_mask_attn_guided,
    cosine_tau,
)
from improvements.vicreg import vicreg_variance_loss, embedding_std


class LitSTELARImproved(LitSTELAR):
    def __init__(
        self,
        *,
        # ---- improvement #2: augmentation / asymmetric views ----
        use_aug: bool = False,
        teacher_aug: Optional[Dict[str, float]] = None,   # default WEAK (identity)
        student_aug: Optional[Dict[str, float]] = None,   # default STRONG menu
        # ---- improvement #1: attention-guided masking ----
        use_attn_mask: bool = False,
        attn_tau_start: float = 1.0,
        attn_tau_min: float = 0.1,
        attn_tau_epochs: int = 15,
        attn_imp_floor: float = 0.05,
        # ---- optional VICReg anti-collapse (3rd MTPE-GB task) ----
        use_vicreg: bool = False,
        vicreg_gamma: float = 1.0,
        lambda_vicreg: float = 1.0,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        # ---------------- improvement #2 ---------------- #
        self.use_aug = bool(use_aug)
        self.aug_teacher = build_augment(teacher_aug if teacher_aug is not None else WEAK_AUG)
        self.aug_student = build_augment(student_aug if student_aug is not None else STRONG_AUG)

        # ---------------- improvement #1 ---------------- #
        self.use_attn_mask = bool(use_attn_mask)
        self.attn_tau_start = float(attn_tau_start)
        self.attn_tau_min = float(attn_tau_min)
        self.attn_tau_epochs = int(attn_tau_epochs)
        self.attn_imp_floor = float(attn_imp_floor)
        self._last_tau = float(attn_tau_start)

        # ---------------- optional VICReg --------------- #
        self.use_vicreg = bool(use_vicreg)
        self.vicreg_gamma = float(vicreg_gamma)

        if self.use_vicreg:
            # promote MTPE-GB to 3 tasks
            self.mtl_losses = ["recon", "align_mask", "vicreg"]
            self.first_val = {k: 0.0 for k in self.mtl_losses}
            self.has_first_val = False
            init_w = {
                "recon": float(getattr(self, "loss_weights", {}).get("recon", 1.0)),
                "align_mask": float(getattr(self, "loss_weights", {}).get("align_mask", 1.0)),
                "vicreg": float(lambda_vicreg),
            }
            self.loss_weights = self._normalize_enabled_weights(init_w)

        # generic per-task running means (replace the base 2-task metrics)
        self._task_means_train = nn.ModuleDict(
            {k: MeanMetric(sync_on_compute=True) for k in self.mtl_losses}
        )
        self._task_means_val = nn.ModuleDict(
            {k: MeanMetric(sync_on_compute=True) for k in self.mtl_losses}
        )

        # monitors
        self._last_teacher_std = 0.0
        self._last_student_std = 0.0

    # ------------------------------------------------------------------ #
    # task-enabled mask (adds vicreg)
    # ------------------------------------------------------------------ #
    def _enabled_mask(self) -> Dict[str, bool]:
        return {
            "recon": bool(self.USE_LOSS_R),
            "align_mask": bool(self.USE_LOSS_A and self.USE_ALIGN_MASK),
            "vicreg": bool(getattr(self, "use_vicreg", False)),
        }

    # ------------------------------------------------------------------ #
    # tau curriculum for #1
    # ------------------------------------------------------------------ #
    def current_tau(self) -> float:
        return cosine_tau(
            int(self.current_epoch),
            self.attn_tau_start,
            self.attn_tau_min,
            self.attn_tau_epochs,
        )

    # ------------------------------------------------------------------ #
    # core: asymmetric views + masking choice + losses
    # ------------------------------------------------------------------ #
    def _compute_losses(self, x: torch.Tensor):  # type: ignore[override]
        device = x.device

        # ---- #2 asymmetric views (teacher clean, student strong) ----
        if self.use_aug:
            x_teacher = self.aug_teacher(x)
            x_student = self.aug_student(x)
        else:
            x_teacher = x
            x_student = x

        # ---- masking: #1 attention-guided OR baseline random/block ----
        if self.use_attn_mask:
            tau = self.current_tau()
            chan_ids = (
                self.chans_id.to(device)
                if (self.use_channel_embed and self.chans_id.numel() > 0)
                else None
            )
            importance = teacher_attention_importance(self.target_encoder, x_teacher, chan_ids)
            mask_x, mask_y = sample_mask_attn_guided(
                importance, self.mask_ratio, tau, imp_floor=self.attn_imp_floor
            )
            self._last_tau = float(tau)
        else:
            mask_x, mask_y = self._sample_masks(
                self.encoder.num_patches,
                mask_ratio=self.mask_ratio,
                device=device,
                mode=self.mask_mode,
                block_time=self.mask_block_time,
                block_chan=self.mask_block_chan,
            )

        # ---- teacher targets on CLEAN view; student on AUGMENTED view ----
        h_all, y_tgt = self.forward_target(x_teacher, mask_y)
        z_enc, z_pred, z_comb, r, masked_bool = self.forward_context(x_student, mask_x, mask_y)

        # ---- L_lat (masked latent prediction) ----
        loss_align = torch.zeros((), device=device)
        if self.USE_LOSS_A and self.USE_ALIGN_MASK:
            C, N = self.encoder.num_patches
            masked_y_bool = _flat_to_bool_mask(mask_y, C=C, N=N, device=device)
            h_m = _gather_tokens(h_all, masked_y_bool)
            z_m = _gather_tokens(z_pred, masked_y_bool)
            if h_m.numel() > 0:
                loss_align = self.loss_fn(h_m, z_m)

        # ---- L_rec (reconstruct CLEAN signal) ----
        loss_recon = self.loss_fn(y_tgt, r) if self.USE_LOSS_R else torch.zeros((), device=device)

        # ---- optional VICReg variance on STUDENT embeddings ----
        loss_vic = torch.zeros((), device=device)
        if self.use_vicreg:
            loss_vic = vicreg_variance_loss(z_enc, gamma=self.vicreg_gamma)

        # ---- monitors: embedding std (always logged) ----
        with torch.no_grad():
            try:
                self._last_teacher_std = float(embedding_std(h_all).item())
                self._last_student_std = float(embedding_std(z_enc).item())
            except Exception:
                pass

        # ---- MTPE-GB weighted sum (normalized over enabled tasks) ----
        w_rec = float(self.loss_weights.get("recon", 0.0))
        w_align = float(self.loss_weights.get("align_mask", 0.0))
        loss = w_rec * loss_recon + w_align * loss_align
        if self.use_vicreg:
            w_vic = float(self.loss_weights.get("vicreg", 0.0))
            loss = loss + w_vic * loss_vic

        raw = {"recon": loss_recon.detach(), "align_mask": loss_align.detach()}
        if self.use_vicreg:
            raw["vicreg"] = loss_vic.detach()
        return loss, raw

    # ------------------------------------------------------------------ #
    # Lightning hooks (generic over self.mtl_losses)
    # ------------------------------------------------------------------ #
    def on_train_epoch_start(self) -> None:
        for k in self.mtl_losses:
            self._task_means_train[k].reset()

    def on_validation_epoch_start(self) -> None:
        for k in self.mtl_losses:
            self._task_means_val[k].reset()

    def _log_common(self, prefix: str, loss, raw: Dict[str, torch.Tensor]) -> None:
        self.log(f"{prefix}_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        for k, v in raw.items():
            self.log(f"{prefix}_loss_{k}", v, on_step=False, on_epoch=True, sync_dist=True)
        for k in self.mtl_losses:
            self.log(f"w_{k}", float(self.loss_weights.get(k, 0.0)), on_step=False, on_epoch=True, sync_dist=True)
        self.log("teacher_emb_std", self._last_teacher_std, on_step=False, on_epoch=True, sync_dist=True)
        self.log("student_emb_std", self._last_student_std, on_step=False, on_epoch=True, sync_dist=True)
        if self.use_attn_mask:
            self.log("mask_tau", self._last_tau, on_step=False, on_epoch=True, sync_dist=True)

    def training_step(self, batch, batch_idx):
        x = _get_x(batch)
        loss, raw = self._compute_losses(x)
        for k, v in raw.items():
            self._task_means_train[k].update(v)
        self._log_common("train", loss, raw)
        return loss

    def validation_step(self, batch, batch_idx):
        x = _get_x(batch)
        loss, raw = self._compute_losses(x)
        for k, v in raw.items():
            self._task_means_val[k].update(v)
        self._log_common("valid", loss, raw)
        return loss

    def on_validation_epoch_end(self) -> None:
        current_val: Dict[str, float] = {}
        for k in self.mtl_losses:
            try:
                current_val[k] = float(self._task_means_val[k].compute().item())
            except Exception:
                current_val[k] = 0.0

        # MTPE-GB update (base method is generic over self.mtl_losses / first_val)
        self._mtpe_gb_update(current_val)

        for k in self.mtl_losses:
            self.log(f"mtpe_first_{k}", float(self.first_val.get(k, 0.0)), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"mtpe_cur_{k}", float(current_val[k]), on_epoch=True, on_step=False, sync_dist=True)
            self.log(f"w_{k}_after", float(self.loss_weights.get(k, 0.0)), on_epoch=True, on_step=False, sync_dist=True)
