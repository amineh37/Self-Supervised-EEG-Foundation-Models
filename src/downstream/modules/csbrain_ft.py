# modules/csbrain_ft.py
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
    from .utils import Conv1dWithConstraint
    _HAS_CONV_CONSTRAINT = True
except Exception:
    Conv1dWithConstraint = None
    _HAS_CONV_CONSTRAINT = False

# Optional richer metrics
try:
    from utils_eval import get_metrics
    _HAS_METRICS = True
except Exception:
    try:
        from downstream.utils_eval import get_metrics
        _HAS_METRICS = True
    except Exception:
        get_metrics = None
        _HAS_METRICS = False

# ---- CSBrain backbone (required) ----
try:
    from backbones.csbrain import CSBrain
    _HAS_CSBRAIN = True
except Exception as _e:
    CSBrain = None
    _HAS_CSBRAIN = False
    _CSBRAIN_IMPORT_ERR = _e

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

def _print_import_report(prefix: str = "[CSBrain-FT] ") -> None:
    lines = []
    lines.append(f"{prefix}torch: v{_mod_version('torch')}")
    lines.append(f"{prefix}einops: v{_mod_version('einops')}")
    if _HAS_CSBRAIN and CSBrain is not None:
        lines.append(f"{prefix}backbone CSBrain: OK ({_where(CSBrain)})")
    else:
        lines.append(f"{prefix}backbone CSBrain: MISSING")
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
        self.norm = nn.LayerNorm(patch_out)
    
    def forward(self, x_b_c_s_p: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x_b_c_s_p
        B, C, S, P = x_b_c_s_p.shape
        assert P == self.patch_in, f"Expected P={self.patch_in}, got {P}"
        x = x_b_c_s_p.reshape(B * C * S, P)
        x = self.proj(x)
        x = self.norm(x)
        x = x.reshape(B, C, S, self.patch_out).contiguous()
        return x

# -----------------------------------------------------------------------------
# Main module
# -----------------------------------------------------------------------------
class LitCSBrain(LitBase):
    """
    Linear-probe / full fine-tuning for CSBrain with patch-size adaptation.

    Inputs
    ------
    - [B, C, S, P_in] (already patched), or
    - [B, C, T]       (raw) -> patched into [B, C, S, P_in] via (patch_size, patch_stride)

    Patch adapter
    -------------
    Projects P_in → P_pretrained (= in_dim) so pretrained weights stay compatible.

    Backbone
    --------
    CSBrain(x) -> [B, C_model, S, D]
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
        patch_size: int = 200,
        patch_stride: Optional[int] = None,

        # CSBrain config (pretrained)
        in_dim: int = 200, out_dim: int = 200, d_model: int = 200,
        dim_feedforward: int = 800, seq_len: int = 30, n_layer: int = 12, nhead: int = 8,
        TemEmbed_kernel_sizes: List[Tuple[int, ...]] = [(1,), (3,), (5,)],
        brain_regions: Optional[List[int]] = None,
        sorted_indices: Optional[List[int]] = None,

        # pretrained / projection control
        load_path: Optional[str] = None,
        replace_proj_out_with_identity: bool = True,

        # head config
        head_type: str = "avg",
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

        if not _HAS_CSBRAIN or CSBrain is None:
            hint = (
                "Could not import backbones.csbrain.CSBrain.\n"
                "• Ensure your PYTHONPATH includes the project root containing 'backbones/'.\n"
                "• If CSBrain lives elsewhere, update the import in modules/csbrain_ft.py.\n"
                f"• Original import error: {_CSBRAIN_IMPORT_ERR!r}"
            )
            raise ImportError(hint)
        _print_import_report()

        print(
            f"[CSBrain-FT] head={head_type}  freeze_encoder={bool(freeze_encoder)}  "
            f"scheduler={str(scheduler).lower()}  seq_len(tokens)={int(seq_len)}  "
            f"patch_size_in={int(patch_size)}  patch_stride={int(patch_stride) if patch_stride is not None else int(patch_size)}  "
            f"patch_pretrained(in_dim)={int(in_dim)}",
            flush=True,
        )

        self.loss_fn = nn.CrossEntropyLoss()

        # ------------ resolve channels / montages ------------
        if channel_names is not None:
            dataset_ch_names = list(channel_names)
            in_ch_for_conv = len(dataset_ch_names)
        else:
            dataset_ch_names = None
            in_ch_for_conv = int(in_chans)

        if use_channel_names is None:
            use_channel_names = [f"CH{i}" for i in range(target_chans or in_ch_for_conv)]
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
        print(f"[CSBrain-FT] n_classes: {num_classes}")
        self.is_binary = False
        if num_classes == 2:
            self.is_binary = True
        self.num_classes = num_classes
        # ------------ patching params ------------
        self.patch_in = int(patch_size)
        self.patch_stride = int(patch_stride) if patch_stride is not None else int(patch_size)
        self.seq_len_tokens = int(seq_len)
        self.patch_pretrained = int(in_dim)

        # ------------ brain regions setup ------------
        if brain_regions is None:
            brain_regions = [0] * self.target_chans  # default: all channels in same region
        if sorted_indices is None:
            sorted_indices = list(range(self.target_chans))
        
        self.brain_regions = list(brain_regions)
        self.sorted_indices = list(sorted_indices)

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
        self.encoder = CSBrain(
            in_dim=self.patch_pretrained, out_dim=out_dim, d_model=d_model,
            dim_feedforward=dim_feedforward, seq_len=self.seq_len_tokens,
            n_layer=n_layer, nhead=nhead,
            TemEmbed_kernel_sizes=TemEmbed_kernel_sizes,
            brain_regions=self.brain_regions,
            sorted_indices=self.sorted_indices,
        )
        
        if replace_proj_out_with_identity and hasattr(self.encoder, "proj_out"):
            self.encoder.proj_out = nn.Identity()

        if load_path is not None:
            self._load_encoder_state(load_path)
            print(f"[CSBrain-FT] loaded encoder weights from: {load_path}", flush=True)

        # ------------ head ------------
        embed_dim = int(out_dim if not replace_proj_out_with_identity else d_model)
        embed_slots = int(self.target_chans)
        n_time = int(self.seq_len_tokens)

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
            print(f"[CSBrain-FT] head='{head_type}' params: {head_params:,}", flush=True)
        except Exception:
            pass

        # ------------ train policy ------------
        self._freeze_encoder = bool(freeze_encoder)
        self._apply_freeze_policy()

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
            f"[CSBrain-FT] metrics={'rich' if _HAS_METRICS and get_metrics is not None else 'basic-only'} | "
            f"chan_adapter={'Conv1dWithConstraint' if self._using_conv_constraint else 'nn.Conv1d'}",
            flush=True,
        )

    def _apply_freeze_policy(self):
        """
        Policy:
        - If freeze_encoder=True: Freeze all encoder layers EXCEPT patch_embedding.
        - If freeze_encoder=False: Unfreeze everything.
        """
        # 1. Start by setting gradients for the whole encoder
        for p in self.encoder.parameters():
            p.requires_grad_(not self._freeze_encoder)

        # # 2. Always unfreeze patch_embedding (if it exists)
        # # This overrides the freeze above if freeze_encoder=True
        # if hasattr(self.encoder, "patch_embedding"):
        #     for p in self.encoder.patch_embedding.parameters():
        #         p.requires_grad_(True)
        
        # 3. Handle train/eval mode logic
        if self._freeze_encoder:
            self.encoder.eval()
            # if hasattr(self.encoder, "patch_embedding"):
            #     self.encoder.patch_embedding.train()
        else:
            self.encoder.train()

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
            S = 1 + (T - P) // stride
            x4 = x.reshape(B * C, 1, 1, T)
            patches = F.unfold(x4, kernel_size=(1, P), stride=(1, stride))
            patches = patches.transpose(1, 2).reshape(B, C, S, P).contiguous()

        # match desired token count
        S_desired = self.seq_len_tokens
        if patches.size(2) < S_desired:
            pad_s = S_desired - patches.size(2)
            patches = F.pad(patches, (0, 0, 0, pad_s))
            S = S_desired
        elif patches.size(2) > S_desired:
            S = S_desired
            patches = patches[:, :, :S, :]
        else:
            S = patches.size(2)

        return patches, S, patches.size(-1)

    def _apply_channel_adapter(self, x_b_c_s_p: torch.Tensor) -> torch.Tensor:
        """
        Apply 1x1 channel projection over the flattened time axis (S*P).
        in : [B, C_in, S, P]
        out: [B, C_out, S, P]
        """
        B, C, S, P = x_b_c_s_p.shape
        x_flat = x_b_c_s_p.reshape(B, C, S * P)
        x_proj = self.chan_conv(x_flat)
        return x_proj.reshape(B, self.target_chans, S, P)

    # -------------------------- forward --------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, C, S, P_in] or [B, C, T]
        returns logits: [B, K]
        """
        # Maintain freeze/train logic during forward
        if self._freeze_encoder:
            self.encoder.eval()
            # if hasattr(self.encoder, "patch_embedding"):
            #     self.encoder.patch_embedding.train()
        else:
            self.encoder.train()

        # 1) Patch (or validate already-patched) → [B, C_in, S, P_in]
        x_patched, S, P_in = self._maybe_patch(x)

        # 2) Channel adapter → [B, C_model, S, P_in]
        # xa = self._apply_channel_adapter(x_patched)

        # 3) Patch-size adapter → [B, C_model, S, P_pretrained]
        # xa = self.patch_adapter(xa)

        # 4) Encoder → [B, C_model, S, D]
        feats = self.encoder(x_patched)

        # 5) Reorder for heads and classify
        z_bned = rearrange(feats, "b c s d -> b s c d")
        logits = self.head(z_bned)
        return logits

    def on_train_epoch_start(self) -> None:
        if self._freeze_encoder:
            self.encoder.eval()
            if hasattr(self.encoder, "patch_embedding"):
                self.encoder.patch_embedding.train()
        else:
            self.encoder.train()
        return super().on_train_epoch_start()

    # ---------------- train/val/test loops ----------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        label = y.long()

        logits = self(x)
        loss = self.loss_fn(logits, label)

        pred = logits.argmax(dim=-1)
        accuracy = (pred == label).float().mean()

        # Log like LitSTELAR
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('train_accuracy', accuracy, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
        # Data statistics (like LitSTELAR)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        
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
        if get_metrics is None or len(self.running_scores["valid"]) == 0:
            return super().on_validation_epoch_end()

        label, y_score = [], []
        for l, s in self.running_scores["valid"]:
            label.append(l)
            y_score.append(s)
        
        if label:
            label = torch.cat(label, dim=0)
            y_score = torch.cat(y_score, dim=0)
            
            # Select metrics based on task type (like Conformer)
            if self.is_binary:
                metrics = ["accuracy","balanced_accuracy","pr_auc","auroc"]
            else:
                metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted"]
            
            results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)
            
            for k, v in results.items():
                self.log('valid_' + k, v, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        
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
        if get_metrics is None or len(self.running_scores["test"]) == 0:
            return super().on_test_epoch_end()

        label, y_score = [], []
        for l, s in self.running_scores["test"]:
            label.append(l)
            y_score.append(s)

        if label:
            import numpy as np
            label = torch.cat(label, dim=0)
            y_score = torch.cat(y_score, dim=0)
            
            # Select metrics based on task type (like Conformer)
            if self.is_binary:
                metrics = ["accuracy","balanced_accuracy","pr_auc", "auroc"]
            else:
                metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted"]
            results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, self.is_binary)

            for k, v in results.items():
                try:
                    v = float(np.nan_to_num(v, nan=0.0))
                except Exception:
                    pass
                self.log("test_" + k, v, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)

        return super().on_test_epoch_end()

    # ---------------- loading ----------------
    def _load_encoder_state(self, path: str) -> None:
        print(f"[CSBrain-FT] Attempting to load encoder weights from: {path}", flush=True)
        
        state = torch.load(path, map_location="cpu")
        print(f"[CSBrain-FT] Checkpoint loaded. Keys in checkpoint: {len(state.keys())}", flush=True)
        
        # Print sample keys to understand structure
        print(f"[CSBrain-FT] Sample checkpoint keys (first 5):")
        for i, key in enumerate(list(state.keys())[:5]):
            print(f"  {i+1}. {key}")
        
        if isinstance(state, dict) and "state_dict" in state:
            print(f"[CSBrain-FT] Found 'state_dict' key, extracting...", flush=True)
            state = state["state_dict"]
            print(f"[CSBrain-FT] State dict keys: {len(state.keys())}", flush=True)
        
        # The checkpoint has keys like 'module.encoder.xxx'
        # We need to strip 'module.encoder.' or just 'module.' prefix
        prefixes_to_try = [
            "module.encoder.",
            "module.",
            "encoder.",
            "backbone.",
            ""
        ]
        
        for prefix in prefixes_to_try:
            # Strip the prefix from checkpoint keys
            if prefix:
                filtered_state = {
                    k[len(prefix):]: v 
                    for k, v in state.items() 
                    if k.startswith(prefix)
                }
            else:
                filtered_state = state
            
            if not filtered_state:
                print(f"[CSBrain-FT] Prefix '{prefix}': no matching keys", flush=True)
                continue
            
            print(f"[CSBrain-FT] Trying prefix '{prefix}': {len(filtered_state)} keys found", flush=True)
            
            # Get model's expected keys
            model_state = self.encoder.state_dict()
            
            # Match keys by name and shape
            matched_state = {}
            shape_mismatches = []
            
            for key, value in filtered_state.items():
                if key in model_state:
                    if value.shape == model_state[key].shape:
                        matched_state[key] = value
                    else:
                        shape_mismatches.append((key, value.shape, model_state[key].shape))
            
            print(f"[CSBrain-FT] Matched keys with correct shapes: {len(matched_state)}/{len(model_state)}", flush=True)
            
            if shape_mismatches:
                print(f"[CSBrain-FT] Shape mismatches found: {len(shape_mismatches)}", flush=True)
                for i, (key, ckpt_shape, model_shape) in enumerate(shape_mismatches[:3]):
                    print(f"  {i+1}. {key}: checkpoint{ckpt_shape} vs model{model_shape}")
            
            # Load if we have any matches
            if len(matched_state) > 0:
                try:
                    missing_keys, unexpected_keys = self.encoder.load_state_dict(matched_state, strict=False)
                    
                    loaded_count = len(matched_state)
                    total_count = len(model_state)
                    
                    print(f"[CSBrain-FT] ✓ Successfully loaded weights with prefix '{prefix}'", flush=True)
                    print(f"[CSBrain-FT]   - Loaded: {loaded_count}/{total_count} parameters", flush=True)
                    print(f"[CSBrain-FT]   - Missing: {len(missing_keys)} keys", flush=True)
                    
                    if missing_keys and len(missing_keys) <= 10:
                        print(f"[CSBrain-FT]   - Missing keys: {missing_keys}", flush=True)
                    elif missing_keys:
                        print(f"[CSBrain-FT]   - Missing keys (first 10): {missing_keys[:10]}", flush=True)
                    
                    # Calculate percentage loaded
                    percentage = (loaded_count / total_count) * 100
                    print(f"[CSBrain-FT]   - Coverage: {percentage:.1f}%", flush=True)
                    
                    return
                    
                except Exception as e:
                    print(f"[CSBrain-FT] ✗ Failed loading with prefix '{prefix}': {str(e)[:150]}", flush=True)
                    continue
        
        # If we get here, no prefix worked well
        print(f"[CSBrain-FT] WARNING: Could not find compatible weights in checkpoint", flush=True)
        print(f"[CSBrain-FT] Model will use random initialization", flush=True)
        
        # Don't raise an error - allow training with random init
        # raise KeyError(f"[CSBrain] Could not match weights from checkpoint '{path}'")

    # ---------------- Optim ----------------
    def configure_optimizers(self):
        # 1. ALWAYS Train: Head, Adapters, and PatchEmbedding
        head_params = (
            list(self.chan_conv.parameters()) +
            list(self.patch_adapter.parameters()) +
            list(self.head.parameters())
        )
        
        # Add patch_embedding params to this "always train" group
        if hasattr(self.encoder, "patch_embedding"):
            head_params += list(self.encoder.patch_embedding.parameters())

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]

        # 2. OPTIONALLY Train: Rest of encoder (if not frozen)
        if not self._freeze_encoder:
            # We need to exclude patch_embedding params since we added them to head_params
            patch_emb_ids = set()
            if hasattr(self.encoder, "patch_embedding"):
                patch_emb_ids = {id(p) for p in self.encoder.patch_embedding.parameters()}
            
            enc_params = [
                p for p in self.encoder.parameters() 
                if id(p) not in patch_emb_ids
            ]
            
            if enc_params:
                groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

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