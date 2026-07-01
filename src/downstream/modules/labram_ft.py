# modules/labram_finetune.py
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import LitBase
from .heads import make_head

try:
    from utils_eval import get_metrics
except Exception:
    try:
        from downstream.utils_eval import get_metrics  # type: ignore
    except Exception:
        get_metrics = None

from timm.models import create_model

try:
    import downstream.backbones.labram  # registers LaBraM models in timm
except Exception as e:
    print(f"[WARN] Could not import LaBraM model registry: {e}")


# ---------------- Patch-size adapter (projects P_in → P_out) ----------------
class PatchSizeAdapter(nn.Module):
    """
    Works on tokenized input:
      in : [B, C, N, P_in]
      out: [B, C, N, P_out]
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

    def forward(self, x_b_c_n_p: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x_b_c_n_p
        B, C, N, P = x_b_c_n_p.shape
        assert P == self.patch_in, f"PatchAdapter expected P_in={self.patch_in}, got {P}"
        if self.patch_in == self.patch_out:
            return x_b_c_n_p
        else:
            x = x_b_c_n_p.reshape(B * C * N, P)  # [B*C*N, P_in]
            x = self.proj(x)                      # [B*C*N, P_out]
            x = x.reshape(B, C, N, self.patch_out).contiguous()
            return x


def _infer_pretrained_patch_size(model, fallback: int) -> int:
    """Best-effort guess of the LaBraM pretrained patch size; fall back to `fallback`."""
    for attr in ("patch_size", "tubelet_size", "tok_patch", "p"):
        v = getattr(model, attr, None)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, (tuple, list)) and len(v) > 0 and isinstance(v[0], int):
            return int(v[0])

    candidates = [
        ("patch_embed", "proj"),
        ("token_embed", None),
        ("proj_in", None),
        ("proj", None),
    ]
    for root_name, leaf_name in candidates:
        root = getattr(model, root_name, None)
        if root is None:
            continue
        cand = getattr(root, leaf_name, None) if leaf_name else root
        if isinstance(cand, nn.Linear):
            return int(cand.in_features)
        if isinstance(cand, (nn.Conv1d, nn.Conv2d)):
            k = cand.kernel_size
            if isinstance(k, tuple) and len(k) >= 1:
                return int(k[-1])
            if isinstance(k, int):
                return int(k)
    return int(fallback)


def _init_module_(m: nn.Module, scheme: str) -> None:
    """Apply a simple weight init to Linear/Conv layers."""
    scheme = str(scheme).lower()
    for mod in m.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(mod.weight)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(mod.weight)
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(mod.weight, nonlinearity="linear")
            else:
                nn.init.xavier_uniform_(mod.weight)
            if getattr(mod, "bias", None) is not None:
                nn.init.zeros_(mod.bias)


class LitLaBraM(LitBase):
    """
    Linear-probe / finetune engine for LaBraM (timm) with patch-size adaptation.

    Input:
      x: [B, C, T]
    """

    def __init__(
        self,
        *,
        # data shape
        in_chans: int = 22,
        seq_len: int = 800,
        patch_size: int = 200,                 # finetune-time patch size
        handle_remainder: str = "crop",        # {"crop","pad"}
        pad_value: float = 0.0,

        # backbone
        timm_model_name: str = "labram_base_patch200_200",
        timm_kwargs: Optional[Dict[str, Any]] = None,
        load_path: Optional[str] = None,       # .pth with "model" dict (optionally "student."-prefixed)
        student_prefix: str = "student.",
        pretrained_patch_size: Optional[int] = None,  # override if needed

        # NEW: pretrain policy
        load_pretrained: bool = True,          # <----- added
        init_scheme: str = "xavier_uniform",   # <----- added

        # head/task
        head_type: str = "flatten",
        num_classes: int = 4,
        dropout: float = 0.5,

        # scheduling
        freeze_encoder: bool = True,

        # optim/sched
        lr_head: float = 4e-4,
        lr_enc: float = 1e-4,
        wd_head: float = 0.01,
        wd_enc: float = 0.05,
        max_epochs: int = 100,
        steps_per_epoch: int = 100,
        scheduler: str = "onecycle",
        warmup_pct: float = 0.2,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

        # shapes / patching
        self.in_chans = int(in_chans)
        self.seq_len = int(seq_len)
        self.patch_in = int(patch_size)
        assert self.patch_in > 0
        assert handle_remainder in {"crop", "pad"}
        self.handle_remainder = handle_remainder
        self.pad_value = float(pad_value)

        self.example_input_array = torch.zeros(8, self.in_chans, self.seq_len, dtype=torch.float32)

        # backbone
        kw = dict(timm_kwargs or {})
        kw.setdefault("qkv_bias", False)
        kw.setdefault("rel_pos_bias", True)
        kw.setdefault("num_classes", num_classes)  # encoder still gets a head inside; harmless
        kw.setdefault("drop_rate", 0.0)
        kw.setdefault("drop_path_rate", 0.1)
        kw.setdefault("attn_drop_rate", 0.0)
        kw.setdefault("drop_block_rate", None)
        kw.setdefault("use_mean_pooling", True)
        kw.setdefault("init_scale", 0.001)
        kw.setdefault("use_rel_pos_bias", True)
        kw.setdefault("use_abs_pos_emb", True)
        kw.setdefault("init_values", 0.1)
        kw.setdefault("pretrained", False)  # never auto-load timm weights

        self.encoder = create_model(timm_model_name, **kw)

        # (Optional) load pretrained weights from file
        if load_pretrained and load_path:
            try:
                ckpt = torch.load(load_path, map_location="cpu")
                state = ckpt.get("model", ckpt)
                new_state = {}
                for k, v in state.items():
                    if isinstance(k, str) and k.startswith(student_prefix):
                        new_state[k[len(student_prefix):]] = v
                    elif isinstance(k, str):
                        new_state[k] = v
                missing, unexpected = self.encoder.load_state_dict(new_state, strict=False)
                if self.global_rank == 0:
                    matched = sum(
                        1 for k, v in new_state.items()
                        if k in self.encoder.state_dict() and self.encoder.state_dict()[k].shape == v.shape
                    )
                    print(f"[LaBraM LOAD] matched {matched}/{len(new_state)} "
                          f"(missing={len(missing)}, unexpected={len(unexpected)})")
                    print(f"missing modules list: {missing}")
                    print(f"unexpected modules list: {unexpected}")
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[WARN] Failed to load LaBraM weights from {load_path}: {e}")
        else:
            # Fresh init path
            if self.global_rank == 0:
                reason = "no load_path provided" if not load_path else "--no-load_pretrained"
                print(f"[LaBraM INIT] initializing encoder from scratch ({reason}), scheme={init_scheme}")
            _init_module_(self.encoder, init_scheme)

        # determine pretrained patch size for adapter **after** encoder exists
        self.patch_pretrained = int(
            pretrained_patch_size if pretrained_patch_size is not None
            else _infer_pretrained_patch_size(self.encoder, fallback=self.patch_in)
        )
        if self.global_rank == 0:
            print(f"[LaBraM] patch_in={self.patch_in} → patch_pretrained={self.patch_pretrained}", flush=True)

        # patch-size adapter
        self.patch_adapter = PatchSizeAdapter(self.patch_in, self.patch_pretrained)

        # ---- infer token geometry ----
        with torch.no_grad():
            probe = torch.zeros(1, self.in_chans, self.seq_len, dtype=torch.float32)
            tokens, N_tokens, D_tok = self._tokens_and_shape(self._patchify(probe)[0])

        self.N = int(N_tokens)  # temporal tokens
        self.E = 1              # slots
        self.D_enc = int(D_tok)

        # head
        self.head = make_head(
            head_type=head_type,
            num_classes=num_classes,
            n_time=self.N,
            embed_slots=self.E,
            embed_dim=self.D_enc,
            dropout=dropout,
        )
        
        if num_classes == 2:
            self.is_binary = True
        else:
            self.is_binary = False
        # freeze policy
        self._freeze_encoder = bool(freeze_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(not self._freeze_encoder)
        self.encoder.eval() if self._freeze_encoder else self.encoder.train()

        # optim/sched cfg
        self._max_epochs = int(max_epochs)
        self._steps_per_epoch = int(steps_per_epoch)
        self._lr_head, self._lr_enc = float(lr_head), float(lr_enc)
        self._wd_head, self._wd_enc = float(wd_head), float(wd_enc)
        self._scheduler, self._warmup_pct = str(scheduler), float(warmup_pct)

        self.running_scores = {"train": [], "valid": [], "test": []}

    # ---------- helpers ----------

    def _patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """[B,C,T] -> [B,C,N,patch_in] with crop/pad handling."""
        B, C, T = x.shape
        if T % self.patch_in != 0:
            if self.handle_remainder == "crop":
                T2 = T - (T % self.patch_in)
                x = x[:, :, :T2]
            else:
                T2 = ((T + self.patch_in - 1) // self.patch_in) * self.patch_in
                pad = T2 - T
                x = F.pad(x, (0, pad), value=self.pad_value)
        x = x.unfold(2, self.patch_in, self.patch_in).contiguous()  # [B,C,N,patch_in]
        N = x.shape[2]
        return x, N, self.patch_in

    @torch.no_grad()
    def _tokens_and_shape(self, x_tokens_in: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        x_tokens_in: [B,C,N,patch_in]  → (adapter) → encoder → tokens [B,N,D]
        """
        B, C, N, P = x_tokens_in.shape
        x_tokens = self.patch_adapter(x_tokens_in)  # [B,C,N,patch_pretrained]
        input_chans = list(range(C + 1))
        tokens = self.encoder.forward_features(
            x_tokens,
            input_chans=input_chans,
            return_all_tokens=True
        )
        if isinstance(tokens, dict):
            for k in ("tokens", "x", "feat"):
                if k in tokens and torch.is_tensor(tokens[k]):
                    tokens = tokens[k]; break
        assert tokens.dim() == 3, f"Expected [B,N,D], got {tokens.shape}"
        return tokens, tokens.shape[1], tokens.shape[2]

    # -------------- main forward --------------
    def forward(self, x: torch.Tensor):
        if self._freeze_encoder:
            self.encoder.eval()

        xa_tok, _, _ = self._patchify(x)
        xa_tok_adapt = self.patch_adapter(xa_tok)
        # print("INPUT before patchify", x.shape)
        # print("INPUT after patchify", xa_tok.shape)
        # print("INPUT after adapt", xa_tok_adapt.shape)

        B, C, N, P = xa_tok_adapt.shape
        input_chans = list(range(C + 1))
        tokens = self.encoder.forward_features(
            xa_tok_adapt,
            input_chans=input_chans,
            return_all_tokens=True
        )

        
        if isinstance(tokens, dict):
            for k in ("tokens", "x", "feat"):
                if k in tokens and torch.is_tensor(tokens[k]):
                    tokens = tokens[k]; break
        # print("OUTPUT representation", tokens.shape)
        logits = self.head(tokens)
        return xa_tok, logits

    # ---------- Lightning ----------

    def on_train_epoch_start(self) -> None:
        self.encoder.eval() if self._freeze_encoder else self.encoder.train()

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    # ---------- steps ----------

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        x_vis, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        acc = (pred == y).float().mean()

        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_accuracy', acc, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x_vis.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x_vis.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x_vis.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x_vis.std(), on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        try:
            opt = self.optimizers()
            if not isinstance(opt, (list, tuple)):
                opt = [opt]
            if opt and hasattr(opt[0], "param_groups"):
                lr = opt[0].param_groups[0].get("lr", None)
                if lr is not None:
                    self.log("lr", lr, on_epoch=True, prog_bar=False, sync_dist=True)
        except Exception:
            pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        _, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        acc = (pred == y).float().mean()

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_accuracy', acc, on_epoch=True, on_step=False, sync_dist=True)

        if get_metrics is not None:
            self.running_scores["valid"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_validation_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["valid"]) == 0:
            return super().on_validation_epoch_end()

        labels, scores = zip(*self.running_scores["valid"])
        y = torch.cat(list(labels), dim=0)
        y_score = torch.cat(list(scores), dim=0)

        if self.is_binary and y_score.shape[1] == 2:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","pr_auc","f1_weighted", "auroc","f1_macro","f1_micro"]
        else:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted","f1_macro","f1_micro"]
        results = get_metrics(y_score, y, metrics, self.is_binary)
        for k, v in results.items():
            self.log('valid_'+k, v, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        _, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        acc = (pred == y).float().mean()

        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('test_accuracy', acc, on_epoch=True, on_step=False, sync_dist=True)

        if get_metrics is not None:
            self.running_scores["test"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_test_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["test"]) == 0:
            return super().on_test_epoch_end()

        labels, scores = zip(*self.running_scores["test"])
        y = torch.cat(list(labels), dim=0)
        y_score = torch.cat(list(scores), dim=0)

        if self.is_binary and y_score.shape[1] == 2:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","pr_auc","f1_weighted", "auroc","f1_macro","f1_micro"]
        else:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted","f1_macro","f1_micro"]
        results = get_metrics(y_score, y, metrics, self.is_binary)
        for k, v in results.items():
            self.log('test_'+k, v, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_test_epoch_end()

    # ---------- optim ----------

    def configure_optimizers(self):
        # Train head + patch adapter always; encoder only if unfrozen
        head_params = list(self.head.parameters()) + list(self.patch_adapter.parameters())
        enc_params  = [] if self._freeze_encoder else list(self.encoder.parameters())

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
        if enc_params:
            groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

        optimizer = torch.optim.AdamW(groups)
        max_lrs = [self._lr_head] + ([self._lr_enc] if enc_params else [])

        if self._scheduler.lower() == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                pct_start=self._warmup_pct,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return {"optimizer": optimizer}
