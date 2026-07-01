# modules/eegpt_lp.py
from __future__ import annotations
from typing import Sequence, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .utils import Conv1dWithConstraint
from backbones.eegpt import EEGTransformer

from .base import LitBase
from .heads import make_head
from .optim import build_optim_and_sched

# (optional) richer metrics if available
try:
    from utils_eval import get_metrics  # (multiclass wrapper)
except Exception:
    try:
        from downstream.utils_eval import get_metrics  # type: ignore
    except Exception:
        get_metrics = None

# ---------------------- Default montages ----------------------
USE_CH_19: Sequence[str] = (
    "FP1","FP2",
    "F7","F3","FZ","F4","F8",
    "T7","C3","CZ","C4","T8",
    "P7","P3","PZ","P4","P8",
    "O1","O2",
)

USE_CH_58: Sequence[str] = (
    "FP1","FPZ","FP2","AF3","AF4",
    "F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO3","POZ","PO4","PO8","O1","OZ","O2",
)


class LitEEGPT(LitBase):
    """
    Linear-probe / full fine-tuning for EEGTransformer (token-grid output).

    x [B,C_in,T] --1x1 conv--> [B,C_model,T]
       -> target_encoder(x, chans_id) -> z [B,N,E,D]
       -> head(z) -> logits [B,K]
    """

    def __init__(
        self,
        models_cfg: Dict[str, Any],
        load_path: str,
        *,
        # dataset/model channels
        in_chans: int = 22,
        channel_names: Optional[Sequence[str]] = None,   # dataset channel names (order)
        use_channel_names: Optional[Sequence[str]] = None,  # encoder montage (order)
        target_chans: Optional[int] = 19,

        # sequence/patch
        seq_len: int = 1024,
        patch: int = 64,

        # task/head
        head_type: str = "avg",             # {"avg","per_time_attn","flatten"}
        num_classes: int = 4,
        hidden: int = 16,
        dropout: float = 0.5,
        # per_time_attn options
        time_pool: str = "mean",            # {"mean","max","meanmax"}
        project_kv: bool = True,
        temperature: Optional[float] = None,

        # -------- train policy --------
        freeze_encoder: bool = True,        # True = linear probe; False = FT (full or top-k)
        ft_top_k: int = 0,                  # NEW: if >0 and not frozen, unfreeze only last K transformer blocks

        # optim/schedule
        lr_head: float = 4e-4, lr_enc: float = 1e-4,
        wd_head: float = 0.01, wd_enc: float = 0.05,
        max_epochs: int = 50, steps_per_epoch: int = 100,
        scheduler: str = "onecycle", warmup_pct: float = 0.2,
    ) -> None:
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

        # ---------------- resolve channels ----------------
        if channel_names is not None:
            dataset_ch_names = list(channel_names)
            in_ch_for_conv = len(dataset_ch_names)
        else:
            dataset_ch_names = None
            in_ch_for_conv = int(in_chans)

        if use_channel_names is None:
            use_channel_names = list(USE_CH_19 if (target_chans == 19) else USE_CH_58)
        else:
            use_channel_names = list(use_channel_names)

        computed_target = len(use_channel_names)
        if target_chans is None:
            target_chans = computed_target
        elif target_chans != computed_target:
            raise ValueError(
                f"target_chans ({target_chans}) != len(use_channel_names) ({computed_target})"
            )

        self._dataset_channel_names = dataset_ch_names
        self.use_channel_names = use_channel_names
        self.in_chans = int(in_ch_for_conv)
        self.target_chans = int(target_chans)
        self.seq_len = int(seq_len)
        self.patch = int(patch)
        self.example_input_array = torch.zeros(2, self.in_chans, self.seq_len, dtype=torch.float32)

        # ---------------- conv adapter ----------------
        self.chan_conv = Conv1dWithConstraint(self.in_chans, self.target_chans, kernel_size=1, max_norm=1.0)
        # ---------------- backbone ----------------
        enc_cfg = dict(models_cfg.get("encoder", {}))
        self.target_encoder = EEGTransformer(
            img_size=[self.target_chans, self.seq_len],
            patch_size=self.patch,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            **enc_cfg,
        )
        # channel ids as encoder expects
        self.chans_id = self._prepare_chans_id(self.use_channel_names)
        # load ONLY target_encoder.* params (strict=False for resilience)
        self._load_target_encoder(load_path)

        # ---------------- head ----------------
        embed_dim: int = int(enc_cfg.get("embed_dim"))
        embed_num: int = int(enc_cfg.get("embed_num", 1))     # E
        n_time: int = int(self.seq_len // self.patch)         # N

        self.head = make_head(
            head_type=head_type,
            num_classes=num_classes,
            embed_dim=embed_dim,
            embed_slots=embed_num,
            hidden=hidden,
            dropout=dropout,
            n_time=n_time,
        )
        self.is_binary = False
        if num_classes == 2:
            self.is_binary = True
            
        # ---------------- train policy ----------------
        self._freeze_encoder = bool(freeze_encoder)
        self._ft_top_k = int(ft_top_k)
        self._apply_finetune_policy()

        # ---------------- optim cfg ----------------
        self._max_epochs = int(max_epochs)
        self._steps_per_epoch = int(steps_per_epoch)
        self._lr_head, self._lr_enc = float(lr_head), float(lr_enc)
        self._wd_head, self._wd_enc = float(wd_head), float(wd_enc)
        self._scheduler, self._warmup_pct = str(scheduler), float(warmup_pct)

        # epoch buffers for richer metrics
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=False

    # ---------- NEW: policy helpers ----------
    def _get_transformer_blocks(self) -> Optional[List[nn.Module]]:
        """
        Try common names to fetch the encoder's sequential blocks.
        Returns a Python list or None if not found.
        """
        if hasattr(self.target_encoder, "blocks"):
            blocks = self.target_encoder.blocks
        elif hasattr(self.target_encoder, "layers"):
            blocks = self.target_encoder.layers
        else:
            blocks = None
        if isinstance(blocks, (nn.ModuleList, list, tuple)):
            return list(blocks)
        return None

    def _apply_finetune_policy(self) -> None:
        """
        Freeze/unfreeze encoder according to:
        - freeze_encoder = True                -> LP (encoder frozen)
        - freeze_encoder = False & ft_top_k=0 -> full FT
        - freeze_encoder = False & ft_top_k>0 -> only last k blocks (+final norm) trainable
        """
        # default: freeze entire encoder
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # head + adapter always trainable
        for p in self.chan_conv.parameters():
            p.requires_grad_(True)
        for p in self.head.parameters():
            p.requires_grad_(True)

        if self._freeze_encoder:
            self.target_encoder.eval()
            return

        if self._ft_top_k <= 0:
            for p in self.target_encoder.parameters():
                p.requires_grad_(True)
            self.target_encoder.train()
            return

        # Unfreeze only the last k blocks
        blocks = self._get_transformer_blocks()
        if blocks:
            k = min(self._ft_top_k, len(blocks))
            for b in blocks[-k:]:
                for p in b.parameters():
                    p.requires_grad_(True)

        # Also unfreeze a common final norm if present
        for attr in ("norm", "final_norm", "ln_f"):
            if hasattr(self.target_encoder, attr):
                for p in getattr(self.target_encoder, attr).parameters():
                    p.requires_grad_(True)

        self.target_encoder.train()

    # ---------------- Lightning ----------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # keep encoder mode aligned with policy
        if any(p.requires_grad for p in self.target_encoder.parameters()):
            self.target_encoder.train()
        else:
            self.target_encoder.eval()
        xa = self.chan_conv(x)
        chan_ids = self.chans_id.to(xa.device)
        z = self.target_encoder(xa, chan_ids)    # [B,N,E,D]
        return x, self.head(z)                  # logits

    # keep mode consistent each epoch (no warm-unfreeze schedule)
    def on_train_epoch_start(self) -> None:
        if any(p.requires_grad for p in self.target_encoder.parameters()):
            self.target_encoder.train()
        else:
            self.target_encoder.eval()

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"]=[]
        return super().on_test_epoch_start()

    # ---------------- train/val/test loops ----------------

    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)

        pred = logit.argmax(dim=-1)
        accuracy = (pred == label).float().mean()

        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        return loss

    # --- add LR logging once per epoch ---
    def on_train_epoch_end(self) -> None:
        try:
            opt = self.optimizers()  # PL helper
            if not isinstance(opt, (list, tuple)):
                opt = [opt]
            if opt and hasattr(opt[0], "param_groups"):
                lr = opt[0].param_groups[0].get("lr", None)
                if lr is not None:
                    self.log("lr", lr, on_epoch=True, prog_bar=False)
        except Exception:
            pass  # keep robust on different PL versions

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()

        if get_metrics is None or len(self.running_scores["valid"]) == 0:
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)

        # Select metrics based on task type (like Conformer)
        if self.is_binary:
            metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "precision", "recall", "f1"]
        else:
            metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()

        x, logit = self.forward(x)
        loss = self.loss_fn(logit, label)
        pred = logit.argmax(dim=-1)
        accuracy = (pred == label).float().mean()
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)

        self.running_scores["valid"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, y.long())
        pred = logit.argmax(dim=-1)
        accuracy = (pred == label).float().mean()
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('test_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)

        self.running_scores["test"].append((label.clone().detach().cpu(), logit.clone().detach().cpu()))
        return loss

    def on_test_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_test_epoch_end()

        if get_metrics is None or len(self.running_scores["test"]) == 0:
            return super().on_test_epoch_end()

        label, y_score = [], []
        for x,y in self.running_scores["test"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)

        # Select metrics based on task type (like Conformer)
        if self.is_binary:
            metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "precision", "recall", "f1"]
        else:
            metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)
        
        for key, value in results.items():
            self.log('test_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_test_epoch_end()

    # ---------------- Utils ----------------

    def _prepare_chans_id(self, model_montage: Sequence[str]) -> torch.Tensor:
        if hasattr(self.target_encoder, "prepare_chan_ids"):
            ids = self.target_encoder.prepare_chan_ids(model_montage)
            return ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids, dtype=torch.long)
        return torch.arange(len(model_montage), dtype=torch.long)

    def _load_target_encoder(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        enc_state = {
            k[len("target_encoder."):] : v
            for k, v in state.items()
            if isinstance(k, str) and k.startswith("target_encoder.")
        }
        missing, unexpected = self.target_encoder.load_state_dict(enc_state, strict=False)
        matched = sum(
            1 for k, v in enc_state.items()
            if k in self.target_encoder.state_dict() and self.target_encoder.state_dict()[k].shape == v.shape
        )
        print(f"[ENC LOAD] matched {matched}/{len(enc_state)} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # ---------------- Optim ----------------
    def configure_optimizers(self):
        head_params = list(self.chan_conv.parameters()) + list(self.head.parameters())
        enc_params  = [p for p in self.target_encoder.parameters() if p.requires_grad]

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
        if enc_params:
            groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

        optimizer = torch.optim.AdamW(groups)

        # max_lr must match param group order
        max_lrs = [self._lr_head] + ([self._lr_enc] if enc_params else [])

        # --- IMPORTANT: use Lightning's true step count ---
        total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)

        # fallback if trainer not attached (rare)
        if total_steps <= 0:
            total_steps = int(self._steps_per_epoch * self._max_epochs)

        # --- Resume-safe clamp (fixes your off-by-one) ---
        total_steps = max(total_steps, int(self.global_step) + 1)

        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            total_steps=total_steps,
            pct_start=self._warmup_pct,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # def configure_optimizers(self):
    #     head_params = list(self.chan_conv.parameters()) + list(self.head.parameters())
    #     # only pass trainable encoder params (respects LP / top-k / full FT)
    #     enc_params  = [p for p in self.target_encoder.parameters() if p.requires_grad]

    #     groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
    #     if enc_params:
    #         groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

    #     optimizer = torch.optim.AdamW(groups)

    #     # max_lr must be per param group and in the same order as 'groups'
    #     max_lrs = [self._lr_head] + ([self._lr_enc] if enc_params else [])

    #     sched = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer,
    #         max_lr=max_lrs,
    #         steps_per_epoch=self._steps_per_epoch,
    #         epochs=self._max_epochs,
    #         pct_start=self._warmup_pct,
    #     )
    #     return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
