from __future__ import annotations

import copy
import math
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

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
# Helpers
# ---------------------------------------------------------------------
def _get_x(batch: Any) -> torch.Tensor:
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _ln_lastdim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # LN over last dim only (patch samples dim P), no learnable affine
    return F.layer_norm(x, (x.size(-1),), eps=eps)


def _masked_bool_from_flat(mask_flat: Optional[torch.Tensor], *, C: int, N: int, device: torch.device) -> torch.Tensor:
    """mask_flat contains indices in [0, N*C). Returns [N,C] bool mask (True=masked)."""
    masked = torch.zeros((N, C), dtype=torch.bool, device=device)
    if mask_flat is None or mask_flat.numel() == 0:
        return masked
    t = (mask_flat // C).long().to(device)
    c = (mask_flat % C).long().to(device)
    masked[t, c] = True
    return masked


def _gather_tokens_bNCD(x_bNCD: torch.Tensor, mask_NC: torch.Tensor) -> torch.Tensor:
    """Gather tokens from [B,N,C,D] using [N,C] bool -> [B,M,D]."""
    if mask_NC is None or mask_NC.numel() == 0 or int(mask_NC.sum().item()) == 0:
        B, _, _, D = x_bNCD.shape
        return x_bNCD.new_empty((B, 0, D))
    idx = torch.nonzero(mask_NC, as_tuple=False)  # [M,2] (t,c)
    t_idx, c_idx = idx[:, 0], idx[:, 1]
    return x_bNCD[:, t_idx, c_idx, :]  # [B,M,D]


def _full_keep_flat_indices(*, C: int, N: int, device: torch.device) -> torch.Tensor:
    """Returns [N,C] long indices with full keep (no masking) in time-major t*C+c."""
    return torch.arange(N * C, device=device, dtype=torch.long).view(N, C)


# ---------------------------------------------------------------------
# Lightning engine
# ---------------------------------------------------------------------
class LitSTELAR(pl.LightningModule):
    def __init__(
        self,
        *,
        models_configs: Dict,
        img_size: Tuple[int, int] = (19, 30),
        patch_size: int = 200,
        input_mode: str = "raw",
        montage: Optional[str] = None,     # kept for API compat (not used here)
        channels: Optional[List[str]] = None,  # kept for API compat (not used here)
        mask_ratio: float = 0.5,
        # model hyperparams
        norm_eps: float = 1e-6,
        qkv_bias: bool = True,
        mlp_ratio: float = 4.0,
        init_std: float = 0.02,
        # schedules
        steps_per_epoch: int = 1000,
        max_epochs: int = 50,
        pretrain_max_lr: float = 1e-4,
        ref_wd: float = 1e-4,
        final_wd: float = 1e-4,
        ema_start_end: Tuple[float, float] = (0.996, 1.0),
        # feature switches
        USE_LOSS_R: bool = True,
        USE_LOSS_A: bool = True,
        USE_ALIGN_VIS: bool = True,
        USE_ALIGN_MASK: bool = True,
        USE_SKIP: bool = True,
        USE_LN: bool = True,
        # fixed loss weights
        lambda_recon: float = 1.0,
        lambda_align_vis: float = 1.0,
        lambda_align_mask: float = 1.0,
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

        # feature flags
        self.USE_LOSS_A = bool(USE_LOSS_A)
        self.USE_LOSS_R = bool(USE_LOSS_R)
        self.USE_SKIP = bool(USE_SKIP)
        self.USE_ALIGN_MASK = bool(USE_ALIGN_MASK)
        self.USE_ALIGN_VIS = bool(USE_ALIGN_VIS)
        self.USE_LN = bool(USE_LN)

        self.norm_eps = float(norm_eps)
        self.mask_ratio = float(mask_ratio)

        # fixed weights (no scheduling)
        self.lambda_recon = float(lambda_recon)
        self.lambda_align_vis = float(lambda_align_vis)
        self.lambda_align_mask = float(lambda_align_mask)

        # schedules / scalars
        if steps_per_epoch <= 0 or max_epochs <= 0 or pretrain_max_lr <= 0:
            raise ValueError("steps_per_epoch, max_epochs, pretrain_max_lr must be positive.")
        self.steps_per_epoch = int(steps_per_epoch)
        self.max_epochs = int(max_epochs)
        self.pretrain_max_lr = float(pretrain_max_lr)
        self.ref_wd = float(ref_wd)
        self.final_wd = float(final_wd)
        self.ema_start, self.ema_end = float(ema_start_end[0]), float(ema_start_end[1])

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

        # EMA teacher
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
        recon_type = str(recon_cfg.pop("type", "mlp")).lower().strip()
        self.reconstructor_type = recon_type

        default_enc_dim = int(getattr(self.encoder, "embed_dim", models_configs["encoder"].get("embed_dim", 768)))
        recon_embed_dim = int(recon_cfg.pop("reconstructor_embed_dim", default_enc_dim))
        use_internal_embed = bool(recon_cfg.pop("use_internal_embed", recon_embed_dim != default_enc_dim))

        if recon_type == "mlp":
            # NOTE: your import says MLPReconstructor; original code used LightReconstructor (typo).
            self.reconstructor = MLPReconstructor(
                embed_dim=default_enc_dim,
                patch_size=patch_size,
                hidden_ratio=float(recon_cfg.get("hidden_ratio", 0.0)),
                act=str(recon_cfg.get("act", "gelu")),
                recon_embed_dim=recon_embed_dim,
                use_internal_embed=use_internal_embed,
            )
        elif recon_type == "transformer":
            self.reconstructor = TransformerReconstructor(
                embed_dim=default_enc_dim,
                patch_size=patch_size,
                recon_embed_dim=recon_embed_dim,
                use_internal_embed=use_internal_embed,
                depth=int(recon_cfg.get("depth", 1)),
                num_heads=int(recon_cfg.get("num_heads", 4)),
                mlp_ratio=float(recon_cfg.get("mlp_ratio", 4.0)),
                qkv_bias=bool(recon_cfg.get("qkv_bias", True)),
                drop=float(recon_cfg.get("drop", 0.0)),
                attn_drop=float(recon_cfg.get("attn_drop", 0.0)),
                drop_path=float(recon_cfg.get("drop_path", 0.0)),
                norm_layer=partial(nn.LayerNorm, eps=norm_eps),
                # if your adapted TransformerReconstructor exposes these knobs:
                use_channel_embed=bool(recon_cfg.get("use_channel_embed", True)),
                chan_embed_scale=float(recon_cfg.get("chan_embed_scale", 1.0)),
                rope_time=bool(recon_cfg.get("rope_time", True)),
                rope_dim_frac=float(recon_cfg.get("rope_dim_frac", 1.0)),
                rope_interp_time=float(recon_cfg.get("rope_interp_time", 1.0)),
            )
        else:
            raise ValueError(f"Unknown reconstructor type={recon_type!r}; use 'mlp' or 'transformer'.")

        self.loss_fn = nn.MSELoss()

        # schedulers state (created in configure_optimizers)
        self.wd_scheduler: Optional[CosineWDSchedule] = None
        self.momentum_scheduler = None

        self.encoder_cc = bool(getattr(self.encoder, "use_criss", False) or getattr(self.encoder, "use_criss_cross", False))
        self.predictor_cc = bool(getattr(self.predictor, "use_criss", False) or getattr(self.predictor, "use_criss_cross", False))

    # ------------------------------------------------------------------
    # Fixed effective weights
    # ------------------------------------------------------------------
    def _effective_loss_weights(self) -> Dict[str, float]:
        return {
            "lambda_recon": self.lambda_recon if self.USE_LOSS_R else 0.0,
            "lambda_align_vis": self.lambda_align_vis if (self.USE_LOSS_A and self.USE_ALIGN_VIS) else 0.0,
            "lambda_align_mask": self.lambda_align_mask if (self.USE_LOSS_A and self.USE_ALIGN_MASK) else 0.0,
        }

    # ------------------------------------------------------------------
    # Optional: fail-fast check for incoming batch shape
    # ------------------------------------------------------------------
    def _assert_input_shape(self, x: torch.Tensor) -> None:
        if self.input_mode == "raw":
            if x.ndim != 3:
                raise ValueError(f"raw mode expects x[B,C,T], got {tuple(x.shape)}")
            _, C, _ = x.shape
        else:
            if x.ndim != 4:
                raise ValueError(f"pretoken mode expects x[B,C,N,D], got {tuple(x.shape)}")
            _, C, _, _ = x.shape
        if C != 19:
            raise ValueError(f"Expected C=19 (TUH-19), got C={C}")

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_masks(
        num_patches: Tuple[int, int],
        mask_ratio: float,
        device: torch.device,
        mode: str = "per_channel",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          mask_x: flat indices of VISIBLE tokens (kept)
          mask_y: flat indices of MASKED tokens (to predict/reconstruct)
        """
        C, N = num_patches
        mask = torch.zeros(N, C, dtype=torch.bool, device=device)

        mode = str(mode).lower().strip()
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
        else:
            raise ValueError(f"Unknown mode={mode!r}; use 'per_channel' or 'per_time'.")

        # avoid fully masked rows/cols (stability)
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

        flat = torch.arange(N * C, device=device, dtype=torch.long).view(N, C)
        mask_y = flat[mask].reshape(-1)     # masked
        mask_x = flat[~mask].reshape(-1)    # visible
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

        # build raw patch targets xr: [B,N,C,P]
        if self.input_mode == "raw":
            B, Cx, Tx = x.shape
            if Cx != C:
                raise ValueError(f"Input C={Cx} != encoder C={C}")
            P = int(self.reconstructor.patch_size)
            if Tx != N * P:
                raise ValueError(f"T={Tx} must equal N*P={N*P} for raw-mode targets.")
            xr = x.view(B, C, N, P).permute(0, 2, 1, 3).contiguous()
        else:
            B, Cx, N_, P = x.shape
            if Cx != C or N_ != N:
                raise ValueError(f"Input (C,N)=({Cx},{N_}) != encoder ({C},{N})")
            xr = x.permute(0, 2, 1, 3).contiguous()

        # masked target patches
        y_masked = apply_mask(mask_y.to(x.device), xr)  # [B, M, P]
        if self.USE_LN and y_masked.numel() > 0:
            y_masked = _ln_lastdim(y_masked, eps=self.norm_eps)
        return h_all, y_masked

    # ------------------------------------------------------------------
    # Student path (context + recon)
    # ------------------------------------------------------------------
    def forward_context(
        self,
        x: torch.Tensor,
        mask_x: torch.Tensor,
        mask_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chan_ids = self.chans_id.to(x.device) if (self.use_channel_embed and self.chans_id.numel() > 0) else None

        # encoder context (masked tokens handled internally)
        z_enc = self.encoder(x, chan_ids, mask_x=mask_x)

        # predictor
        z_pred, z_comb, masked_bool, _ = self.predictor(z_enc, mask_x=mask_x)

        if not self.USE_SKIP:
            z_comb = z_pred

        # reconstructor expects mask_bool in [N,C]
        C, N = self.encoder.num_patches
        masked_y_bool = _masked_bool_from_flat(mask_y, C=C, N=N, device=x.device)

        # reconstructor returns [B,M,P] in the SAME order as flat_idx computed from mask_bool
        r_masked, _flat_order = self.reconstructor(z_comb, masked_y_bool)
        return z_enc, z_pred, z_comb, r_masked, masked_bool

    # ------------------------------------------------------------------
    # Loss compute
    # ------------------------------------------------------------------
    def _compute_losses(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        device = x.device
        mask_x, mask_y = self._sample_masks(self.encoder.num_patches, mask_ratio=self.mask_ratio, device=device)

        h_all, y_tgt = self.forward_target(x, mask_y)
        z_enc, z_pred, z_comb, r, masked_bool = self.forward_context(x, mask_x, mask_y)

        vis_bool = ~masked_bool

        loss_align_vis = torch.zeros((), device=device)
        loss_align_mask = torch.zeros((), device=device)

        if self.USE_LOSS_A and self.USE_ALIGN_VIS:
            h_vis = _gather_tokens_bNCD(h_all, vis_bool)
            z_vis = _gather_tokens_bNCD(z_enc, vis_bool)
            if h_vis.numel() > 0:
                loss_align_vis = self.loss_fn(h_vis, z_vis)

        if self.USE_LOSS_A and self.USE_ALIGN_MASK:
            h_m = _gather_tokens_bNCD(h_all, masked_bool)
            z_m = _gather_tokens_bNCD(z_pred, masked_bool)
            if h_m.numel() > 0:
                loss_align_mask = self.loss_fn(h_m, z_m)

        loss_recon = self.loss_fn(y_tgt, r) if self.USE_LOSS_R else torch.zeros((), device=device)

        eff = self._effective_loss_weights()
        w_recon = eff["lambda_recon"] * loss_recon
        w_align_vis = eff["lambda_align_vis"] * loss_align_vis
        w_align_mask = eff["lambda_align_mask"] * loss_align_mask
        w_align_total = w_align_vis + w_align_mask

        if self.USE_LOSS_A and self.USE_LOSS_R:
            loss = w_recon + w_align_total
        elif self.USE_LOSS_A and not self.USE_LOSS_R:
            loss = w_align_total
        elif (not self.USE_LOSS_A) and self.USE_LOSS_R:
            loss = w_recon
        else:
            loss = torch.zeros((), device=device)

        logs: Dict[str, torch.Tensor] = dict(
            align_vis=loss_align_vis.detach(),
            align_mask=loss_align_mask.detach(),
            recon=loss_recon.detach(),
            w_recon_factor=torch.tensor(eff["lambda_recon"], device=device),
            w_align_vis_factor=torch.tensor(eff["lambda_align_vis"], device=device),
            w_align_mask_factor=torch.tensor(eff["lambda_align_mask"], device=device),
            recon_w=w_recon.detach(),
            align_vis_w=w_align_vis.detach(),
            align_mask_w=w_align_mask.detach(),
            align_w=w_align_total.detach(),
        )
        return loss, loss_recon, (loss_align_vis + loss_align_mask), logs

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def on_fit_start(self) -> None:
        if self.use_channel_embed and self.chans_id.device != self.device:
            self.chans_id = self.chans_id.to(self.device)
        print(
            f"[Engine] input_mode={self.input_mode} | "
            f"encoder_cc={self.encoder_cc} | predictor_cc={self.predictor_cc} | "
            f"reconstructor={self.reconstructor_type} | LN_y={self.USE_LN}"
        )

    def validation_step(self, batch: Any, batch_idx: int):
        x = _get_x(batch)
        self._assert_input_shape(x)
        loss, loss_recon, loss_align, logs = self._compute_losses(x)

        self.log("valid_loss_recon", loss_recon, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_loss_align", loss_align, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_loss", loss, on_epoch=True, on_step=False, sync_dist=True, prog_bar=True)

        self.log("valid_align_vis", logs["align_vis"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_align_mask", logs["align_mask"], on_epoch=True, on_step=False, sync_dist=True)

        self.log("w_recon", logs["w_recon_factor"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("w_align_vis", logs["w_align_vis_factor"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("w_align_mask", logs["w_align_mask_factor"], on_epoch=True, on_step=False, sync_dist=True)

        self.log("valid_loss_recon_w", logs["recon_w"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_loss_align_w", logs["align_w"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_align_vis_w", logs["align_vis_w"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_align_mask_w", logs["align_mask_w"], on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        x = _get_x(batch)
        self._assert_input_shape(x)
        loss, loss_recon, loss_align, logs = self._compute_losses(x)

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_loss_recon", loss_recon, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_align", loss_align, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_align_vis", logs["align_vis"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_align_mask", logs["align_mask"], on_step=False, on_epoch=True, sync_dist=True)

        self.log("w_recon", logs["w_recon_factor"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("w_align_vis", logs["w_align_vis_factor"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("w_align_mask", logs["w_align_mask_factor"], on_step=False, on_epoch=True, sync_dist=True)

        self.log("train_loss_recon_w", logs["recon_w"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_loss_align_w", logs["align_w"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_align_vis_w", logs["align_vis_w"], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_align_mask_w", logs["align_mask_w"], on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.target_encoder.eval()
        if self.wd_scheduler is not None:
            self.wd_scheduler.step()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        grad_stats = grad_logger(self.encoder.named_parameters())
        self.log("grad_stats.first_layer", grad_stats.first_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log("grad_stats.last_layer", grad_stats.last_layer, on_epoch=True, on_step=False, sync_dist=True)
        self.log("grad_stats.min", grad_stats.min, on_epoch=True, on_step=False, sync_dist=True)
        self.log("grad_stats.max", grad_stats.max, on_epoch=True, on_step=False, sync_dist=True)

        if self.momentum_scheduler is not None:
            with torch.no_grad():
                m = next(self.momentum_scheduler)
                for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        res = super().on_load_checkpoint(checkpoint)
        self.configure_optimizers()
        return res

    # ------------------------------------------------------------------
    # Optimizers / Schedulers
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