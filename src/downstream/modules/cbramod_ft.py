# modules/cbramod_ft.py
from __future__ import annotations
from typing import Sequence, Optional, Tuple, List, Dict, Any
import importlib
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import LitBase
from .heads import make_head

# -----------------------------------------------------------------------------
# Optional bits
# -----------------------------------------------------------------------------
# Optional: channel 1x1 with max-norm (fallback to nn.Conv1d)
try:
    from .utils import Conv1dWithConstraint  # 1x1 over channels (length = S*P)
    _HAS_CONV_CONSTRAINT = True
except Exception:
    Conv1dWithConstraint = None  # noqa: N816
    _HAS_CONV_CONSTRAINT = False

# Optional richer metrics
try:
    from utils_eval import get_metrics
    _HAS_METRICS = True
except Exception:
    try:
        from downstream.utils_eval import get_metrics  # type: ignore
        _HAS_METRICS = True
    except Exception:
        get_metrics = None
        _HAS_METRICS = False

# ---- CBraMod backbone (required) ----
try:
    from backbones.cbramod import CBraMod  # adjust to your tree
    _HAS_CBRAMOD = True
except Exception as _e:
    CBraMod = None  # type: ignore
    _HAS_CBRAMOD = False
    _CBRAMOD_IMPORT_ERR = _e

# -----------------------------------------------------------------------------
# Helpers: import report
# -----------------------------------------------------------------------------
def _mod_version(mod_name: str) -> str:
    try:
        m = importlib.import_module(mod_name)
    except Exception:
        return "n/a"
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(m, attr, None)
        if isinstance(v, (str, int, float)):
            return str(v)
    return "unknown"

def _where(obj: Any) -> str:
    try:
        p = importlib.import_module(obj.__module__).__file__
        return os.path.abspath(p) if p else "unknown"
    except Exception:
        return "unknown"

def _print_import_report(prefix: str = "[CBraMod-FT] ") -> None:
    lines = []
    lines.append(f"{prefix}torch: v{_mod_version('torch')}")
    lines.append(f"{prefix}einops: v{_mod_version('einops')}")
    if _HAS_CBRAMOD and CBraMod is not None:
        lines.append(f"{prefix}backbone CBraMod: OK ({_where(CBraMod)})")
    else:
        lines.append(f"{prefix}backbone CBraMod: MISSING")
    if _HAS_CONV_CONSTRAINT and Conv1dWithConstraint is not None:
        lines.append(f"{prefix}Conv1dWithConstraint: OK ({_where(Conv1dWithConstraint)})")
    else:
        lines.append(f"{prefix}Conv1dWithConstraint: not found → will use nn.Conv1d")
    if _HAS_METRICS and get_metrics is not None:
        lines.append(f"{prefix}get_metrics: OK ({_where(get_metrics)})")
    else:
        lines.append(f"{prefix}get_metrics: not found → will log basic acc/loss only")
    for ln in lines:
        print(ln, flush=True)

# -----------------------------------------------------------------------------
# Patch-size adapter (projects P_in → P_out = pretrained patch)
# -----------------------------------------------------------------------------
class PatchSizeAdapter(nn.Module):
    """
    x_in:  [B, C, S, P_in]
    x_out: [B, C, S, P_out]
    """
    def __init__(self, patch_in: int, patch_out: int) -> None:
        super().__init__()
        self.patch_in = int(patch_in)
        self.patch_out = int(patch_out)
        self.enabled = (self.patch_in != self.patch_out)
        if self.enabled:
            self.proj = nn.Linear(self.patch_in, self.patch_out)
        else:
            self.register_parameter("dummy", None)

    def forward(self, x_b_c_s_p: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x_b_c_s_p
        B, C, S, P = x_b_c_s_p.shape
        assert P == self.patch_in, f"Expected P={self.patch_in}, got {P}"
        x = x_b_c_s_p.reshape(B * C * S, P)      # [B*C*S, P_in]
        x = self.proj(x)                          # [B*C*S, P_out]
        x = x.reshape(B, C, S, self.patch_out).contiguous()
        return x

# -----------------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------------
class LitCBraMod(LitBase):
    """
    Linear-probe / full fine-tuning for CBraMod with patch-size adaptation.

    Inputs
    ------
    - [B, C, S, P_in] (already patched), or
    - [B, C, T]       (raw) -> patched into [B, C, S, P_in] via (patch_size, patch_stride)

    Patch adapter
    -------------
    Projects P_in → P_pretrained (= in_dim) so pretrained weights stay compatible.

    Backbone
    --------
    CBraMod(x) -> [B, C_model, S, D]
    Reordered to [B, N=S, E=C_model, D] for heads in .heads.
    """

    def __init__(
        self,
        *,
        # dataset/model channels
        in_chans: int = 19,
        target_chans: Optional[int] = 19,
        channel_names: Optional[Sequence[str]] = None,
        use_channel_names: Optional[Sequence[str]] = None,

        # patching if input is [B,C,T]
        patch_size: int = 200,                 # arbitrary at finetune time
        patch_stride: Optional[int] = None,    # default: = patch_size (no overlap)

        # CBraMod config (pretrained)
        in_dim: int = 200, out_dim: int = 200, d_model: int = 200,
        dim_feedforward: int = 800, seq_len: int = 30, n_layer: int = 12, nhead: int = 8,

        # pretrained / projection control
        # If load_path is None, the backbone is RANDOMLY INITIALIZED.
        load_path: Optional[str] = None,
        replace_proj_out_with_identity: bool = True,

        # head config
        head_type: str = "avg",     # {"avg","per_time_attn","flatten","all_patch_reps","eegpt_probe"}
        num_classes: int = 4,
        hidden: int = 128,
        dropout: float = 0.5,
        time_pool: str = "mean",
        project_kv: bool = True,
        temperature: Optional[float] = None,

        # train policy
        freeze_encoder: bool = True,

        # optim/schedule
        lr_head: float = 4e-4, lr_enc: float = 1e-4,
        wd_head: float = 0.01, wd_enc: float = 0.05,
        max_epochs: int = 50, steps_per_epoch: int = 100,
        scheduler: str = "onecycle",
        warmup_pct: float = 0.2,
    ) -> None:
        super().__init__()

        if not _HAS_CBRAMOD or CBraMod is None:
            hint = (
                "Could not import backbones.cbramod.CBraMod.\n"
                "• Ensure your PYTHONPATH includes the project root containing 'backbones/'.\n"
                "• If CBraMod lives elsewhere, update the import in modules/cbramod_ft.py.\n"
                f"• Original import error: {_CBRAMOD_IMPORT_ERR!r}"
            )
            raise ImportError(hint)
        _print_import_report()

        print(
            f"[CBraMod-FT] head={head_type}  freeze_encoder={bool(freeze_encoder)}  "
            f"scheduler={str(scheduler).lower()}  seq_len(tokens)={int(seq_len)}  "
            f"patch_size_in={int(patch_size)}  patch_stride={int(patch_stride) if patch_stride is not None else int(patch_size)}  "
            f"patch_pretrained(in_dim)={int(in_dim)}",
            flush=True,
        )

        self.loss_fn = nn.CrossEntropyLoss()

        self.in_chans = in_chans
        self.target_chans = in_chans

        print(f"[CSBrain-FT] n_classes: {num_classes}")
        self.is_binary = False
        if num_classes == 2:
            self.is_binary = True
            
        # ------------ patching params ------------
        self.patch_in = int(patch_size)                              # incoming patch size
        self.patch_stride = int(patch_stride) if patch_stride is not None else int(patch_size)
        self.seq_len_tokens = int(seq_len)                           # expected S (tokens)
        self.patch_pretrained = int(in_dim)                          # CBraMod pretraining patch (=in_dim)

        # example input for summary/ONNX (raw form)
        self.example_input_array = torch.zeros(
            2, self.in_chans, self.seq_len_tokens * self.patch_in, dtype=torch.float32
        )

        # ------------ channel adapter (1x1) ------------
        if Conv1dWithConstraint is not None:
            self.chan_conv = Conv1dWithConstraint(self.in_chans, self.target_chans, kernel_size=1, max_norm=1.0)
            self._using_conv_constraint = True
        else:
            self.chan_conv = nn.Conv1d(self.in_chans, self.target_chans, kernel_size=1, bias=True)
            self._using_conv_constraint = False

        # ------------ patch adapter (P_in → P_pretrained) ------------
        self.patch_adapter = PatchSizeAdapter(self.patch_in, self.patch_pretrained)

        # ------------ backbone (expects P = in_dim) ------------
        # By default this is Random Initialization
        self.encoder = CBraMod(
            in_dim=self.patch_pretrained, out_dim=out_dim, d_model=d_model,
            dim_feedforward=dim_feedforward, seq_len=self.seq_len_tokens,
            n_layer=n_layer, nhead=nhead,
        )
        if replace_proj_out_with_identity and hasattr(self.encoder, "proj_out"):
            self.encoder.proj_out = nn.Identity()

        # ------------ Weights Handling (Pretrained vs Random) ------------
        if load_path is not None and str(load_path).lower() != "none" and str(load_path).strip() != "":
            self._load_encoder_state(load_path)
            print(f"[CBraMod-FT] Loaded encoder weights from: {load_path}", flush=True)
            self._using_random_init = False
        else:
            print(f"[CBraMod-FT] No 'load_path' provided. Using RANDOM INITIALIZATION for backbone.", flush=True)
            self._using_random_init = True
            
            # Safety Check: If random init, usually we do NOT want to freeze encoder
            if freeze_encoder:
                print(
                    "\n[CBraMod-FT] !!! WARNING !!! \n"
                    "You selected Random Initialization (no load_path) but freeze_encoder=True. \n"
                    "The backbone will remain frozen random noise. Ensure this is intended.\n", 
                    flush=True
                )

        # ------------ head ------------
        embed_dim = int(out_dim if not replace_proj_out_with_identity else d_model)  # D
        embed_slots = int(self.target_chans)  # E (channels after adapter)
        n_time = int(self.seq_len_tokens)     # N (temporal tokens)

        self.head = make_head(
            head_type=head_type,
            num_classes=num_classes,
            embed_dim=embed_dim,
            embed_slots=embed_slots,
            hidden=hidden,
            dropout=dropout,
            n_time=n_time,
            time_pool=time_pool,
            project_kv=project_kv,
            temperature=temperature,
        )

        try:
            head_params = sum(p.numel() for p in self.head.parameters())
            print(f"[CBraMod-FT] head='{head_type}' params: {head_params:,}", flush=True)
        except Exception:
            pass

        # ------------ train policy ------------
        self._freeze_encoder = bool(freeze_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(not self._freeze_encoder)
        self.encoder.eval() if self._freeze_encoder else self.encoder.train()

        # ------------ optim cfg ------------
        self._max_epochs = int(max_epochs)
        self._steps_per_epoch = int(steps_per_epoch)
        self._lr_head, self._lr_enc = float(lr_head), float(lr_enc)
        self._wd_head, self._wd_enc = float(wd_head), float(wd_enc)
        self._scheduler, self._warmup_pct = str(scheduler).lower(), float(warmup_pct)

        # epoch buffers for optional rich metrics
        self.running_scores: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {
            "train": [], "valid": [], "test": []
        }

        print(
            f"[CBraMod-FT] metrics={'rich' if _HAS_METRICS and get_metrics is not None else 'basic-only'} | "
            f"chan_adapter={'Conv1dWithConstraint' if self._using_conv_constraint else 'nn.Conv1d'}",
            flush=True,
        )

    # -------------------------- helpers --------------------------
    @torch.no_grad()
    def _maybe_patch(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
          patches: [B, C, S, P_in]
          S      : number of tokens (possibly trimmed/padded to self.seq_len_tokens)
          P_in   : incoming patch size
        """
        # Input is already [B, C, S, P]?
        if x.dim() == 4:
            B, C, S, P = x.shape
            if C != self.in_chans:
                raise RuntimeError(f"Expected in_chans={self.in_chans}, got {C}")
            # If caller fed P != configured patch_in, we still adapt via PatchSizeAdapter later.
            patches = x
        else:
            if x.dim() != 3:
                raise RuntimeError(f"Expected input [B,C,T] or [B,C,S,P], got {tuple(x.shape)}")
            B, C, T = x.shape
            if C != self.in_chans:
                raise RuntimeError(f"Expected in_chans={self.in_chans}, got {C}")
            P = self.patch_in
            stride = self.patch_stride
            if T < P:
                raise RuntimeError(f"Sequence too short: T={T} < patch_size={P}")
            # valid windows: floor((T - P)/stride) + 1
            S = 1 + (T - P) // stride
            # unfold 1D time into patches
            x4 = x.reshape(B * C, 1, 1, T)
            patches = F.unfold(x4, kernel_size=(1, P), stride=(1, stride))  # [B*C, P, S]
            patches = patches.transpose(1, 2).reshape(B, C, S, P).contiguous()

        # match desired token count (trim/pad at the end)
        S_desired = self.seq_len_tokens
        if patches.size(2) < S_desired:
            pad_s = S_desired - patches.size(2)
            # zero-pad along S
            patches = F.pad(patches, (0, 0, 0, pad_s))  # pad S dimension at the end
            S = S_desired
        elif patches.size(2) > S_desired:
            S = S_desired
            patches = patches[:, :, :S, :]
        else:
            S = patches.size(2)

        return patches, S, patches.size(-1)  # P_in

    def _apply_channel_adapter(self, x_b_c_s_p: torch.Tensor) -> torch.Tensor:
        """
        Apply 1x1 channel projection over the flattened time axis (S*P).
        in : [B, C_in, S, P]
        out: [B, C_out, S, P]
        """
        B, C, S, P = x_b_c_s_p.shape
        x_flat = x_b_c_s_p.reshape(B, C, S * P)           # [B, C_in, L]
        x_proj = self.chan_conv(x_flat)                    # [B, C_out, L]
        return x_proj.reshape(B, self.target_chans, S, P)

    # -------------------------- forward --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, S, P_in] or [B, C, T]
        returns logits: [B, K]
        """
        if self._freeze_encoder:
            self.encoder.eval()


        # 1) Patch (or validate already-patched) → [B, C_in, S, P_in]
        x_patched, S, P_in = self._maybe_patch(x)

        # # 2) Channel adapter → [B, C_model, S, P_in]
        # xa = self._apply_channel_adapter(x_patched)

        # # 3) Patch-size adapter (to pretrained patch, i.e., in_dim) → [B, C_model, S, P_pretrained]
        # xa = self.patch_adapter(xa)

        xa = x_patched
        # 4) Encoder → [B, C_model, S, D]
        feats = self.encoder(xa)

        # 5) Reorder for heads and classify
        z_bned = rearrange(feats, "b c s d -> b s c d")   # [B, N=S, E=C_model, D]
        logits = self.head(z_bned)                        # [B, K]
        return logits

    # Keep mode consistent per epoch (no warm-unfreeze schedule)
    def on_train_epoch_start(self) -> None:
        self.encoder.eval() if self._freeze_encoder else self.encoder.train()
        return super().on_train_epoch_start()

    # ---------------- train/val/test loops ----------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        acc = (logit.argmax(dim=-1) == y).float().mean()

        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_accuracy',  acc,  on_epoch=True, on_step=False, sync_dist=True)

        if get_metrics is not None:
            self.running_scores["train"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_train_epoch_end(self) -> None:
        try:
            opt = self.optimizers()
            opts = opt if isinstance(opt, (list, tuple)) else [opt]
            if opts and hasattr(opts[0], "param_groups"):
                lr = opts[0].param_groups[0].get("lr", None)
                if lr is not None:
                    self.log("lr", lr, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass
        return super().on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        acc = (logit.argmax(dim=-1) == y).float().mean()

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_accuracy',  acc,  on_epoch=True, on_step=False, sync_dist=True)

        if get_metrics is not None:
            self.running_scores["valid"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_validation_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            return super().on_validation_epoch_end()
        if get_metrics is None:
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for l, s in self.running_scores["valid"]:
            label.append(l); y_score.append(s)
        if label:
            label = torch.cat(label, dim=0)
            y_score = torch.cat(y_score, dim=0)
            # Select metrics based on task type (like Conformer)
            if self.is_binary:
                metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "precision", "recall", "f1"]
            else:
                metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
            
            results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)
            for k, v in results.items():
                self.log('valid_' + k, v, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        acc = (logit.argmax(dim=-1) == y).float().mean()

        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('test_accuracy',  acc,  on_epoch=True, on_step=False)

        if get_metrics is not None:
            self.running_scores["test"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_test_epoch_end(self) -> None:
        if getattr(self.trainer, "sanity_checking", False):
            return super().on_test_epoch_end()
        if get_metrics is None:
            return super().on_test_epoch_end()

        label, y_score = [], []
        for l, s in self.running_scores["test"]:
            label.append(l); y_score.append(s)

        if label:
            import numpy as np
            label = torch.cat(label, 0)
            y_score = torch.cat(y_score, 0)
            # Select metrics based on task type (like Conformer)
            if self.is_binary:
                metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "precision", "recall", "f1"]
            else:
                metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
            
            results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)
            for k, v in results.items():
                try:
                    v = float(np.nan_to_num(v, nan=0.0))
                except Exception:
                    pass
                self.log("test_" + k, v, on_epoch=True, on_step=False, sync_dist=True)

        return super().on_test_epoch_end()

    # ---------------- loading ----------------
    def _load_encoder_state(self, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        prefixes = [
            "encoder.", "backbone.", "module.encoder.", "module.backbone.",
            "model.encoder.", "model.backbone.", ""
        ]
        for pref in prefixes:
            try_state = {k[len(pref):]: v for k, v in state.items() if k.startswith(pref)}
            try_state = try_state if try_state else state if pref == "" else {}
            if not try_state:
                continue
            try:
                self.encoder.load_state_dict(try_state, strict=False)
                return
            except Exception:
                continue
        raise KeyError(f"[CBraMod] Could not match weights from checkpoint '{path}'")

    # ---------------- Optim ----------------
    def configure_optimizers(self):
        # Head (including adapters) always trained
        head_params = (
            list(self.chan_conv.parameters()) +
            list(self.patch_adapter.parameters()) +
            list(self.head.parameters())
        )
        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]

        # Encoder params only if not frozen
        if not self._freeze_encoder:
            groups.append({"params": self.encoder.parameters(), "weight_decay": self._wd_enc, "lr": self._lr_enc})

        optimizer = torch.optim.AdamW(groups)

        sch_type = self._scheduler
        if sch_type == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[g["lr"] for g in optimizer.param_groups],
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                pct_start=self._warmup_pct,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        if sch_type == "cosine":
            total_steps = self._max_epochs * self._steps_per_epoch
            warmup_steps = int(self._warmup_pct * total_steps)
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps or 1)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps))
            sched = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps or 1])
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        return {"optimizer": optimizer}