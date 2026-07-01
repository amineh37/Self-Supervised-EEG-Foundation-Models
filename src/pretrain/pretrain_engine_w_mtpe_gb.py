from __future__ import annotations

import copy
from functools import partial
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torchmetrics import MeanMetric  # add

from utils import CosineWDSchedule, grad_logger
from pretrain_modeling import (
    TransformerEncoder,
    TransformerPredictor,
    MLPReconstructor,
    TransformerReconstructor,
    apply_mask,
)

# ---------------------------------------------------------------------
# TUH 19-channel order EXACTLY matching preprocessing SELECTED_CHANNELS["01_tcp_ar"]
# ---------------------------------------------------------------------
TUH_19_CHANNELS: List[str] = [
    "FP1", "FP2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4",
    "T5", "T6",
    "FZ", "CZ", "PZ",
]

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _gather_tokens(x_bNCD: torch.Tensor, mask_NC: torch.Tensor) -> torch.Tensor:
    """Gather tokens from x[B,N,C,D] where mask[N,C] is True -> [B, M, D]."""
    if mask_NC is None or mask_NC.numel() == 0 or mask_NC.sum() == 0:
        B, _, _, D = x_bNCD.shape
        return x_bNCD.new_empty((B, 0, D))
    B, N, C, D = x_bNCD.shape
    mask4 = mask_NC.unsqueeze(0).unsqueeze(-1).expand(B, N, C, D)
    return x_bNCD[mask4].view(B, -1, D)

def _flat_to_bool_mask(mask_flat: torch.Tensor, C: int, N: int, device) -> torch.Tensor:
    """Convert flat indices (t*C + c) into bool mask [N,C]."""
    masked = torch.zeros((N, C), dtype=torch.bool, device=device)
    if mask_flat is None or mask_flat.numel() == 0:
        return masked
    t = (mask_flat // C).long().to(device)
    c = (mask_flat % C).long().to(device)
    masked[t, c] = True
    return masked

def _full_keep_flat_indices(C: int, N: int, device) -> torch.Tensor:
    """Return [N,C] of flat indices (keep all tokens)."""
    return torch.stack(
        [torch.arange(i * C, (i + 1) * C, device=device, dtype=torch.long) for i in range(N)],
        dim=0,
    )

def _get_x(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch

def _ln_lastdim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """LN over last dim only (patch samples dim P), no learnable affine."""
    return F.layer_norm(x, (x.size(-1),), eps=eps)


# ---------------------------------------------------------------------
# Lightning engine (MTPE_GB for recon + masked-align)
# ---------------------------------------------------------------------
class LitSTELAR(pl.LightningModule):
    def __init__(
        self,
        *,
        models_configs: Dict,
        img_size: Tuple[int, int] = (19, 30),
        patch_size: int = 200,
        input_mode: str = "raw",
        montage: Optional[str] = None,
        channels: Optional[List[str]] = None,
        mask_ratio: float = 0.5,
        mask_mode: str = "square_block",  # "per_channel", "per_time", or "square_block"
        mask_block_time: int = 4,   # for "square_block" mode: height in time (N axis)
        mask_block_chan: int = 4,   # for "square_block" mode: width in channels (C axis)
        # model hyperparams
        norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        init_std: float = 0.02,
        # --- training schedules ---
        steps_per_epoch: int = 1000,
        max_epochs: int = 50,
        pretrain_max_lr: float = 1e-4,
        # weight decay schedule
        ref_wd: float = 1e-4,
        final_wd: float = 1e-4,
        # EMA momentum schedule
        ema_start_end: Tuple[float, float] = (0.996, 1.0),
        # feature switches
        USE_LOSS_R: bool = True,
        USE_LOSS_A: bool = True,
        USE_ALIGN_VIS: bool = True,   # kept for drop-in compatibility (ignored)
        USE_ALIGN_MASK: bool = True,
        USE_SKIP: bool = True,
        # LN only on target y (for recon loss)
        USE_LN: bool = True,
        # initial priors (used only as init for MTPE_GB)
        lambda_recon: float = 1.0,
        lambda_align_vis: float = 1.0,  # kept (ignored)
        lambda_align_mask: float = 1.0,
        # --- MTPE_GB knobs ---
        mtpe_gb_ema_c: float = 0.5,
        mtpe_gb_eps: float = 1e-8,
        mtpe_gb_clamp: float = 1e6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[])

        # ---- enforce TUH-19 grid ----
        C_img, N_img = int(img_size[0]), int(img_size[1])
        if C_img != 19:
            raise ValueError(f"[STELAR] Forced TUH-19. Got img_size[0]={C_img}. Set img_size=(19, N).")
        if len(TUH_19_CHANNELS) != 19:
            raise RuntimeError("TUH_19_CHANNELS must have exactly 19 items.")

        input_mode = str(input_mode).lower().strip()
        if input_mode not in ("raw", "pretoken"):
            raise ValueError(f"input_mode must be 'raw' or 'pretoken', got {input_mode!r}")
        self.input_mode = input_mode

        # flags
        self.USE_LOSS_A = bool(USE_LOSS_A)
        self.USE_LOSS_R = bool(USE_LOSS_R)
        self.USE_SKIP = bool(USE_SKIP)
        self.USE_ALIGN_MASK = bool(USE_ALIGN_MASK)
        self.USE_LN = bool(USE_LN)

        # ---- visible alignment removed (force-off) ----
        self.USE_ALIGN_VIS = False
        self.lambda_align_vis = 0.0

        self.norm_eps = float(norm_eps)

        # schedules / scalars
        assert steps_per_epoch > 0 and max_epochs > 0 and pretrain_max_lr > 0
        self.steps_per_epoch = int(steps_per_epoch)
        self.max_epochs = int(max_epochs)
        self.pretrain_max_lr = float(pretrain_max_lr)
        self.ref_wd = float(ref_wd)
        self.final_wd = float(final_wd)
        self.ema_start, self.ema_end = float(ema_start_end[0]), float(ema_start_end[1])
        self.mask_ratio = float(mask_ratio)
        self.mask_mode = str(mask_mode).lower().strip()
        self.mask_block_time = int(mask_block_time)
        self.mask_block_chan = int(mask_block_chan)
        # ----------------- models ----------------- #
        self.encoder = TransformerEncoder(
            img_size=[19, N_img],
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            init_std=init_std,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=norm_eps),
            **models_configs["encoder"],
        )

        self.use_channel_embed = bool(getattr(self.encoder, "use_channel_embed", True))
        if self.use_channel_embed:
            chans_id = self.encoder.prepare_chan_ids(TUH_19_CHANNELS)
        else:
            chans_id = torch.empty(0, dtype=torch.long)
        self.register_buffer("chans_id", chans_id, persistent=False)

        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = TransformerPredictor(
            num_patches=self.encoder.num_patches,
            init_std=init_std,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=norm_eps),
            **models_configs["predictor"],
        )

        # ----------------- Reconstructor head ----------------- #
        recon_cfg = dict(models_configs.get("reconstructor", {}))
        recon_type = recon_cfg.pop("type", "light").lower().strip()
        self.reconstructor_type = recon_type

        default_enc_dim = getattr(self.encoder, "embed_dim", models_configs["encoder"].get("embed_dim", 768))
        recon_embed_dim = int(recon_cfg.pop("reconstructor_embed_dim", default_enc_dim))
        use_internal_embed = bool(recon_cfg.pop("use_internal_embed", recon_embed_dim != default_enc_dim))

        if recon_type in ("mlp", "light"):  # "light" kept as a legacy alias
            self.reconstructor = MLPReconstructor(
                embed_dim=default_enc_dim,
                patch_size=patch_size,
                hidden_ratio=recon_cfg.get("hidden_ratio", 0.0),
                act=recon_cfg.get("act", "gelu"),
                recon_embed_dim=recon_embed_dim,
                use_internal_embed=use_internal_embed,
            )
        elif recon_type == "transformer":
            self.reconstructor = TransformerReconstructor(
                embed_dim=default_enc_dim,
                patch_size=patch_size,
                recon_embed_dim=recon_embed_dim,
                use_internal_embed=use_internal_embed,
                depth=recon_cfg.get("depth", 1),
                num_heads=recon_cfg.get("num_heads", 4),
                mlp_ratio=recon_cfg.get("mlp_ratio", 4.0),
                qkv_bias=recon_cfg.get("qkv_bias", True),
                drop=recon_cfg.get("drop", 0.0),
                attn_drop=recon_cfg.get("attn_drop", 0.0),
                drop_path=recon_cfg.get("drop_path", 0.0),
                norm_layer=partial(nn.LayerNorm, eps=norm_eps),
            )
        else:
            raise ValueError(f"Unknown reconstructor type={recon_type!r}; use 'mlp' or 'transformer'.")

        self.loss_fn = torch.nn.MSELoss()

        self.wd_scheduler = None
        self.momentum_scheduler = None

        # ----------------- MTPE_GB state ----------------- #
        self.mtpe_gb_ema_c = float(mtpe_gb_ema_c)
        self.mtpe_gb_eps = float(mtpe_gb_eps)
        self.mtpe_gb_clamp = float(mtpe_gb_clamp)

        # Only two components for this request
        self.mtl_losses = ["recon", "align_mask"]

        # initial weights (priors), then normalize over enabled losses
        init_w = {
            "recon": float(lambda_recon),
            "align_mask": float(lambda_align_mask),
        }
        self.loss_weights = self._normalize_enabled_weights(init_w)

        # baseline (first) val losses
        self.first_val = {"recon": 0.0, "align_mask": 0.0}
        self.has_first_val = False

        # distributed-safe running means
        self.train_recon_mean = MeanMetric(sync_on_compute=True)
        self.train_align_mean = MeanMetric(sync_on_compute=True)
        self.val_recon_mean = MeanMetric(sync_on_compute=True)
        self.val_align_mean = MeanMetric(sync_on_compute=True)

        self.encoder_cc = bool(getattr(self.encoder, "use_criss", False) or getattr(self.encoder, "use_criss_cross", False))
        self.predictor_cc = bool(getattr(self.predictor, "use_criss", False) or getattr(self.predictor, "use_criss_cross", False))

    # ------------------------------------------------------------------
    # helpers: enabled mask + normalize
    # ------------------------------------------------------------------
    def _enabled_mask(self) -> Dict[str, bool]:
        return {
            "recon": bool(self.USE_LOSS_R),
            "align_mask": bool(self.USE_LOSS_A and self.USE_ALIGN_MASK),
        }

    def _normalize_enabled_weights(self, w: Dict[str, float]) -> Dict[str, float]:
        enabled = self._enabled_mask()
        s = 0.0
        out = {}
        for k in self.mtl_losses:
            out[k] = float(w.get(k, 0.0)) if enabled[k] else 0.0
            s += out[k]
        if s <= 0:
            # fallback: uniform over enabled
            m = sum(1 for k in self.mtl_losses if enabled[k])
            if m == 0:
                return {k: 0.0 for k in self.mtl_losses}
            return {k: (1.0 / m if enabled[k] else 0.0) for k in self.mtl_losses}
        return {k: (out[k] / s) for k in self.mtl_losses}

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_masks(
        num_patches: Tuple[int, int],
        mask_ratio: float,
        device,
        mode: str = "per_channel",
        # --- new params for square blocks ---
        block_time: int = 4,   # height in time (N axis)
        block_chan: int = 4,   # width in channels (C axis)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        C, N = num_patches 
        mask = torch.zeros(N, C, dtype=torch.bool, device=device)

        def _enforce_not_fully_masked(mask: torch.Tensor) -> torch.Tensor:
            # ensure not fully masked in any time row
            all_masked_t = mask.all(dim=1)
            if all_masked_t.any():
                for t in torch.nonzero(all_masked_t, as_tuple=False).flatten():
                    c = torch.randint(0, C, (1,), device=device)
                    mask[t, c] = False

            # ensure not fully masked in any channel col
            all_masked_c = mask.all(dim=0)
            if all_masked_c.any():
                for c in torch.nonzero(all_masked_c, as_tuple=False).flatten():
                    t = torch.randint(0, N, (1,), device=device)
                    mask[t, c] = False
            return mask

        if mode == "per_channel":
            k = int(round(N * float(mask_ratio)))
            k = max(0, min(k, N - 1))
            for c in range(C):
                idx_t = torch.randperm(N, device=device)[:k]
                mask[idx_t, c] = True

        elif mode == "per_time":
            k = int(round(C * float(mask_ratio)))
            k = max(0, min(k, C - 1))
            for t in range(N):
                idx_c = torch.randperm(C, device=device)[:k]
                mask[t, idx_c] = True

        elif mode == "square_block":
            # Target number of masked tokens
            total_tokens = N * C
            target = int(round(total_tokens * float(mask_ratio)))
            target = max(0, min(target, total_tokens - 1))  # keep at least 1 visible

            bt = max(1, min(int(block_time), N))
            bc = max(1, min(int(block_chan), C))
            block_area = bt * bc

            # Number of blocks to paint (approx); we will correct to hit target exactly
            num_blocks = int((target + block_area - 1) // block_area) if block_area > 0 else 0
            num_blocks = max(1, num_blocks) if target > 0 else 0

            for _ in range(num_blocks):
                t0 = torch.randint(0, max(1, N - bt + 1), (1,), device=device).item()
                c0 = torch.randint(0, max(1, C - bc + 1), (1,), device=device).item()
                mask[t0:t0 + bt, c0:c0 + bc] = True

            # If we overshot target, randomly unmask extras.
            cur = int(mask.sum().item())
            if cur > target:
                idx = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
                drop = idx[torch.randperm(idx.numel(), device=device)[: (cur - target)]]
                mask.view(-1)[drop] = False

            # If we undershot target (can happen due to overlaps), randomly add more masks.
            cur = int(mask.sum().item())
            if cur < target:
                idx = torch.nonzero((~mask).flatten(), as_tuple=False).flatten()
                add = idx[torch.randperm(idx.numel(), device=device)[: (target - cur)]]
                mask.view(-1)[add] = True

        else:
            raise ValueError(f"Unknown mode={mode!r}; use 'per_channel', 'per_time', or 'square_block'.")

        mask = _enforce_not_fully_masked(mask)

        flat = torch.arange(N * C, device=device, dtype=torch.long).view(N, C)
        mask_y = flat[mask].reshape(-1)    # masked tokens (targets)
        mask_x = flat[~mask].reshape(-1)   # visible tokens (context)
        return mask_x, mask_y


    # ------------------------------------------------------------------
    # Teacher path (targets)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_target(self, x: torch.Tensor, mask_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C, N = self.encoder.num_patches
        mask_x_full = _full_keep_flat_indices(C=C, N=N, device=x.device)

        chan_ids = self.chans_id.to(x.device) if (self.use_channel_embed and self.chans_id.numel() > 0) else None
        h_all = self.target_encoder(x, chan_ids, mask_x=mask_x_full)

        if self.input_mode == "raw":
            B, Cx, Tx = x.shape
            P = self.reconstructor.patch_size
            xr = x.view(B, C, N, P).permute(0, 2, 1, 3).contiguous()  # [B,N,C,P]
        else:
            xr = x.permute(0, 2, 1, 3).contiguous()  # [B,N,C,D]

        y_masked = apply_mask(mask_y.to(x.device), xr)
        if self.USE_LN and y_masked.numel() > 0:
            y_masked = _ln_lastdim(y_masked, eps=self.norm_eps)

        return h_all, y_masked

    # ------------------------------------------------------------------
    # Student path (context + recon)
    # ------------------------------------------------------------------
    def forward_context(
        self, x: torch.Tensor, mask_x: torch.Tensor, mask_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chan_ids = self.chans_id.to(x.device) if (self.use_channel_embed and self.chans_id.numel() > 0) else None
        z_enc = self.encoder(x, chan_ids, mask_x=mask_x)

        z_pred, z_comb, masked_bool, _ = self.predictor(z_enc, mask_x=mask_x)
        if not self.USE_SKIP:
            z_comb = z_pred

        C, N = self.encoder.num_patches
        masked_y_bool = _flat_to_bool_mask(mask_y, C=C, N=N, device=x.device)
        r_masked, flat_order = self.reconstructor(z_comb, masked_y_bool)

        return z_enc, z_pred, z_comb, r_masked, masked_bool

    # ------------------------------------------------------------------
    # MTPE_GB weight update (val-based)
    # ------------------------------------------------------------------
    def _mtpe_gb_update(self, current_val: Dict[str, float]) -> None:
        enabled = self._enabled_mask()

        # if first baseline not set yet -> set and keep current weights (normalized)
        if not self.has_first_val:
            for k in self.mtl_losses:
                self.first_val[k] = float(current_val[k])
            self.has_first_val = True
            self.loss_weights = self._normalize_enabled_weights(self.loss_weights)
            return

        # compute urgency u_i = L0 / (L0 - Lt)
        u = {}
        for k in self.mtl_losses:
            if not enabled[k]:
                u[k] = 0.0
                continue
            L0 = float(self.first_val[k])
            Lt = float(current_val[k])
            denom = (L0 - Lt)

            if denom <= self.mtpe_gb_eps:
                ui = self.mtpe_gb_clamp
            else:
                ui = L0 / (denom + self.mtpe_gb_eps)

            if ui < 0:
                ui = self.mtpe_gb_clamp
            u[k] = float(min(ui, self.mtpe_gb_clamp))

        # normalize to q
        total = sum(u[k] for k in self.mtl_losses if enabled[k]) + self.mtpe_gb_eps
        q = {k: (u[k] / total if enabled[k] else 0.0) for k in self.mtl_losses}

        # EMA blend
        c = self.mtpe_gb_ema_c
        new_w = {}
        for k in self.mtl_losses:
            if not enabled[k]:
                new_w[k] = 0.0
            else:
                new_w[k] = (1 - c) * q[k] + c * float(self.loss_weights[k])
        
        # # NO EMA: w <- q
        # new_w = {k: float(q[k]) for k in self.mtl_losses}

        self.loss_weights = self._normalize_enabled_weights(new_w)

    # ------------------------------------------------------------------
    # Compute losses (ONLY recon + masked alignment) using dynamic weights
    # ------------------------------------------------------------------
    def _compute_losses(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        device = x.device
        # mask_x, mask_y = self._sample_masks(self.encoder.num_patches, mask_ratio=self.mask_ratio, device=device)

        mask_x, mask_y = self._sample_masks(
            self.encoder.num_patches,
            mask_ratio=self.mask_ratio,
            device=device,
            mode=self.mask_mode,
            block_time=self.mask_block_time,
            block_chan=self.mask_block_chan,
        )

        h_all, y_tgt = self.forward_target(x, mask_y)
        z_enc, z_pred, z_comb, r, masked_bool = self.forward_context(x, mask_x, mask_y)

        loss_align_mask = torch.zeros((), device=device)
        if self.USE_LOSS_A and self.USE_ALIGN_MASK:
            C, N = self.encoder.num_patches
            masked_y_bool = _flat_to_bool_mask(mask_y, C=C, N=N, device=device)
            h_m = _gather_tokens(h_all, masked_y_bool)
            z_m = _gather_tokens(z_pred, masked_y_bool)

            # h_m = _gather_tokens(h_all, masked_bool)
            # z_m = _gather_tokens(z_pred, masked_bool)
            if h_m.numel() > 0:
                loss_align_mask = self.loss_fn(h_m, z_m)

        loss_recon = self.loss_fn(y_tgt, r) if self.USE_LOSS_R else torch.zeros((), device=device)

        # dynamic weights (already normalized over enabled components)
        w_recon_factor = float(self.loss_weights["recon"])
        w_align_factor = float(self.loss_weights["align_mask"])

        loss = (w_recon_factor * loss_recon) + (w_align_factor * loss_align_mask)

        logs = dict(
            recon=loss_recon.detach(),
            align_mask=loss_align_mask.detach(),
            w_recon_factor=torch.tensor(w_recon_factor, device=device),
            w_align_mask_factor=torch.tensor(w_align_factor, device=device),
            recon_w=(w_recon_factor * loss_recon).detach(),
            align_mask_w=(w_align_factor * loss_align_mask).detach(),
        )
        return loss, loss_recon, loss_align_mask.detach(), logs

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_fit_start(self) -> None:
        if self.use_channel_embed and self.chans_id.device != self.device:
            self.chans_id = self.chans_id.to(self.device)

    def on_train_epoch_start(self) -> None:
        self.train_recon_mean.reset()
        self.train_align_mean.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_recon_mean.reset()
        self.val_align_mean.reset()

    def validation_step(self, batch, batch_idx):
        x = _get_x(batch)
        loss, loss_recon, loss_align, logs = self._compute_losses(x)

        # distributed-safe aggregation
        self.val_recon_mean.update(loss_recon.detach())
        self.val_align_mean.update(loss_align.detach())

        self.log("valid_loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)
        self.log("valid_loss_recon", loss_recon, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_loss_align", loss_align, on_epoch=True, on_step=False, sync_dist=True)

        self.log("w_recon", logs["w_recon_factor"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("w_align_mask", logs["w_align_mask_factor"], on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        # compute epoch means (synced)
        cur_recon = float(self.val_recon_mean.compute().item())
        cur_align = float(self.val_align_mean.compute().item())

        current_val = {"recon": cur_recon, "align_mask": cur_align}

        # update weights with MTPE_GB
        self._mtpe_gb_update(current_val)

        # log “first” baseline + current + weights (after update)
        self.log("mtpe_gb_first_recon", torch.tensor(self.first_val["recon"], device=self.device),
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log("mtpe_gb_first_align", torch.tensor(self.first_val["align_mask"], device=self.device),
                 on_epoch=True, on_step=False, sync_dist=True)

        self.log("mtpe_gb_cur_recon", torch.tensor(cur_recon, device=self.device),
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log("mtpe_gb_cur_align", torch.tensor(cur_align, device=self.device),
                 on_epoch=True, on_step=False, sync_dist=True)

        self.log("w_recon_after", torch.tensor(self.loss_weights["recon"], device=self.device),
                 on_epoch=True, on_step=False, sync_dist=True)
        self.log("w_align_after", torch.tensor(self.loss_weights["align_mask"], device=self.device),
                 on_epoch=True, on_step=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        x = _get_x(batch)
        loss, loss_recon, loss_align, logs = self._compute_losses(x)

        self.train_recon_mean.update(loss_recon.detach())
        self.train_align_mean.update(loss_align.detach())

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_loss_recon", loss_recon, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_align", loss_align, on_step=False, on_epoch=True, sync_dist=True)

        self.log("w_recon", logs["w_recon_factor"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("w_align_mask", logs["w_align_mask_factor"], on_step=False, on_epoch=True, sync_dist=True)

        return loss

    # checkpoint MTPE_GB state
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["mtpe_gb"] = {
            "loss_weights": {k: float(v) for k, v in self.loss_weights.items()},
            "first_val": {k: float(v) for k, v in self.first_val.items()},
            "has_first_val": bool(self.has_first_val),
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_load_checkpoint(checkpoint)
        st = checkpoint.get("mtpe_gb", None)
        if st is not None:
            self.loss_weights = {k: float(v) for k, v in st.get("loss_weights", self.loss_weights).items()}
            self.first_val = {k: float(v) for k, v in st.get("first_val", self.first_val).items()}
            self.has_first_val = bool(st.get("has_first_val", self.has_first_val))
            self.loss_weights = self._normalize_enabled_weights(self.loss_weights)

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.target_encoder.eval()
        if self.wd_scheduler is not None:
            self.wd_scheduler.step()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.momentum_scheduler is not None:
            with torch.no_grad():
                m = next(self.momentum_scheduler)
                for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    # ------------------------------------------------------------------
    # Optimizers / Schedulers (unchanged)
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        def _no_wd(name: str, p: torch.nn.Parameter) -> bool:
            if p.ndim == 1 or name.endswith(".bias"):
                return True
            no_wd_keys = (
                "chan_embed",
                "patch_embed.positional_encoding",
                "patch_embed.mask_encoding",
                "mask_query_time", "mask_query_chan",
                "time_embed.freqs", "rope_time.freqs", "rope_chan.freqs",
            )
            return any(k in name for k in no_wd_keys)

        def split_params(named_params):
            wd, no_wd = [], []
            for n, p in named_params:
                if not p.requires_grad:
                    continue
                (no_wd if _no_wd(n, p) else wd).append(p)
            return wd, no_wd

        enc_wd, enc_no_wd = split_params(self.encoder.named_parameters())
        pred_wd, pred_no_wd = split_params(self.predictor.named_parameters())
        recon_wd, recon_no_wd = split_params(self.reconstructor.named_parameters())

        param_groups = [
            {"params": enc_wd},
            {"params": pred_wd},
            {"params": recon_wd},
            {"params": enc_no_wd, "weight_decay": 0.0, "WD_exclude": True},
            {"params": pred_no_wd, "weight_decay": 0.0, "WD_exclude": True},
            {"params": recon_no_wd, "weight_decay": 0.0, "WD_exclude": True},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self.pretrain_max_lr)

        total_steps = int(self.max_epochs * self.steps_per_epoch)
        warmup_steps = max(500, int(0.1 * total_steps))

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

        lr_dict = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        self.wd_scheduler = CosineWDSchedule(
            optimizer, ref_wd=self.ref_wd, final_wd=self.final_wd, T_max=total_steps
        )

        ema0, ema1 = self.ema_start, self.ema_end
        self.momentum_scheduler = (ema0 + i * (ema1 - ema0) / total_steps for i in range(total_steps + 1))

        return {"optimizer": optimizer, "lr_scheduler": lr_dict}
