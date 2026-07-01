from __future__ import annotations
from typing import Sequence, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Conv1dWithConstraint
from .base import LitBase
from .heads import make_head

try:
    from .optim import build_optim_and_sched  # optional
except Exception:
    build_optim_and_sched = None  # type: ignore

# STELAR backbone (using ACPE / dynamic PE)
from downstream.backbones.stelar import TransformerEncoder

# Metrics (tolerant import)
try:
    from utils_eval import get_metrics
except Exception:
    try:
        from downstream.utils_eval import get_metrics  # type: ignore
    except Exception:
        get_metrics = None


# ---- Default montages ----
DEFAULT_CH_58: Sequence[str] = (
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO3","POZ","PO4","PO8","O1","OZ","O2",
)
DEFAULT_CH_19: Sequence[str] = (
    "FP1", "FP2",
    "F3", "F4",
    "C3", "C4",
    "P3", "P4",
    "O1", "O2",
    "F7", "F8",
    "T3", "T4",
    "T5", "T6",
    "FZ", "CZ", "PZ",
)

# DEFAULT_CH_19: Sequence[str] = (
#     "FP1","FP2","F7","F3","FZ","F4","F8",
#     "T7","C3","CZ","C4","T8",
#     "P7","P3","PZ","P4","P8",
#     "O1","O2",
# )

class LitSTELARPatchAdapt(LitBase):
    """
    STELAR fine-tuning engine.

    - Optional 1x1 channel-mixing Conv1d before the STELAR TransformerEncoder.
    - Handles both raw [B,C,T] and pre-patchified [B,C,N,patch].
    - Encoder is assumed to use ACPE (dynamic positional encoding) and
      may or may not use channel embedding.

    Encoder I/O:
      input:  [B, C, N, patch_size]
      output: [B, N, C_model, D_enc]
    """

    def __init__(
        self,
        models_cfg: Dict[str, Any],
        load_path: Optional[str],
        *,
        # dataset/model channels
        in_chans: int = 22,
        channel_names: Optional[Sequence[str]] = None,
        target_chans: int = 19,
        use_channel_names: Optional[Sequence[str]] = None,
        # sequence geometry
        seq_len: int = 1024,
        patch: int = 64,
        # head / task
        head_type: str = "avg",
        num_classes: int = 4,
        dropout: float = 0.5,
        # ---- front-end toggle ----
        use_chan_conv: bool = True,       # 1x1 channel-mixing Conv1d
        chan_max_norm: float = 1.0,       # max-norm for constrained conv
        # training policy
        freeze_encoder: bool = True,
        ft_top_k: int = 0,
        # optim/schedule
        lr_head: float = 4e-4,
        lr_enc: float = 1e-4,
        wd_head: float = 0.01,
        wd_enc: float = 0.05,
        max_epochs: int = 100,
        steps_per_epoch: int = 100,
        scheduler: str = "onecycle",
        warmup_pct: float = 0.2,
        # loss tweaks
        label_smoothing: float = 0.0,
        class_weights: Optional[Sequence[float]] = None,
        # ---- pretraining toggle ----
        load_pretrained: bool = True,
        init_scheme: str = "xavier_uniform",
    ):
        super().__init__()

        # --- loss config ---
        self._ls_eps = float(label_smoothing)
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w, persistent=False)
        else:
            self.class_weights = None

        # --- channels / montage ---
        if channel_names is not None:
            dataset_ch_names = list(channel_names)
            in_ch_for_adapter = len(dataset_ch_names)
        else:
            dataset_ch_names = None
            in_ch_for_adapter = int(in_chans)
        print(f"[INFO] dataset_ch_names = {dataset_ch_names}")
        print(f"[INFO] in_ch_for_adapter = {in_ch_for_adapter}")

        if use_channel_names is None:
            use_channel_names = list(DEFAULT_CH_19 if (target_chans == 19) else DEFAULT_CH_58)
        else:
            use_channel_names = list(use_channel_names)

        computed_target = len(use_channel_names)
        if target_chans is None:
            target_chans = computed_target
        elif target_chans != computed_target:
            raise ValueError(f"target_chans ({target_chans}) != len(use_channel_names) ({computed_target})")

        self._dataset_channel_names = dataset_ch_names
        self.use_channel_names = use_channel_names
        self.in_chans = int(in_ch_for_adapter)
        self.target_chans = int(target_chans)

        # --- geometry ---
        self.seq_len = int(seq_len)
        self.patch = int(patch)
        assert self.seq_len % self.patch == 0, "seq_len must be divisible by patch"
        self.N = self.seq_len // self.patch

        self.example_input_array = torch.zeros(8, self.in_chans, self.seq_len, dtype=torch.float32)

        # --- optional 1x1 channel-mixing conv over time axis ---
        self.use_chan_conv = bool(use_chan_conv)
        self.chan_conv = (
            Conv1dWithConstraint(self.in_chans, self.target_chans, kernel_size=1, max_norm=float(chan_max_norm))
            if self.use_chan_conv else None
        )

        # --- encoder & head ---
        enc_cfg = dict(models_cfg.get("encoder", {}))

        # # FORCE: no channel embedding + use ACPE/dynamic PE by default here
        # enc_cfg.setdefault("use_channel_embed", False)
        # enc_cfg.setdefault("use_dynamic_pe", True)

        # build STELAR encoder
        self.target_encoder = TransformerEncoder(**enc_cfg)
        self.D_enc = int(enc_cfg.get("embed_dim", 768))
        self.proj_in = nn.Linear(self.patch, 200)
        if num_classes == 2:
            self.is_binary = True
        else:
            self.is_binary = False
        ch_ids = self._prepare_chans_id(self.use_channel_names)
        self.register_buffer(
            "chans_id",
            ch_ids if isinstance(ch_ids, torch.Tensor) else torch.as_tensor(ch_ids, dtype=torch.long),
            persistent=False,
        )

        # ---- load (optional) pretrained weights ----
        self._loaded_pretrain = False
        if load_pretrained and load_path:
            try:
                pre = torch.load(load_path, map_location="cpu")
                state = pre.get("state_dict", pre)
                enc_state = {k[len("target_encoder."):]: v for k, v in state.items()
                             if isinstance(k, str) and k.startswith("target_encoder.")}
                if not enc_state:
                    enc_state = {k: v for k, v in state.items() if isinstance(k, str)}

                missing, unexpected = self.target_encoder.load_state_dict(enc_state, strict=False)
                matched = sum(
                    1
                    for k, v in enc_state.items()
                    if k in self.target_encoder.state_dict()
                    and self.target_encoder.state_dict()[k].shape == v.shape
                )
                if self.global_rank == 0:
                    print(
                        f"[ENC LOAD] matched {matched}/{len(enc_state)} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})"
                    )
                    if missing:
                        print(f"missing modules: {missing}")
                    if unexpected:
                        print(f"unexpected modules: {unexpected}")
                self._loaded_pretrain = True
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[WARN] Failed to load encoder weights from {load_path}: {e}")

        # ---- random-init if we didn't load ----
        if not self._loaded_pretrain:
            if self.global_rank == 0:
                print(f"[INIT] Randomly initializing encoder/front-end (scheme={init_scheme}).")
            self._reinit_all(init_scheme)

        # classification head on encoder tokens [B, N, C, D_enc]
        self.head = make_head(
            head_type=head_type,
            num_classes=num_classes,
            n_time=self.N,
            embed_slots=self.target_chans,
            embed_dim=self.D_enc,
            dropout=dropout,
        )

        # --- train policy / optim cfg ---
        self._freeze_encoder = bool(freeze_encoder)
        self._ft_top_k = int(ft_top_k)
        self._apply_finetune_policy()

        self._max_epochs = int(max_epochs)
        self._steps_per_epoch = int(steps_per_epoch)
        self._lr_head, self._lr_enc = float(lr_head), float(lr_enc)
        self._wd_head, self._wd_enc = float(wd_head), float(wd_enc)
        self._scheduler, self._warmup_pct = str(scheduler), float(warmup_pct)

        self.running_scores = {"train": [], "valid": [], "test": []}

    # ---- init helpers ----
    @staticmethod
    def _init_module(m: nn.Module, scheme: str = "xavier_uniform") -> None:
        with torch.no_grad():
            if isinstance(m, (nn.Linear,)):
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _reinit_all(self, scheme: str = "xavier_uniform") -> None:
        self.target_encoder.apply(lambda m: self._init_module(m, scheme))
        if self.chan_conv is not None:
            self.chan_conv.apply(lambda m: self._init_module(m, scheme))
        self.proj_in.apply(lambda m: self._init_module(m, scheme))

    # ---- loss ----
    def _loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._ls_eps <= 0.0:
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        C = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        y_true = F.one_hot(targets, num_classes=C).float()
        y_smooth = (1.0 - self._ls_eps) * y_true + self._ls_eps / C
        loss = -(y_smooth * logp).sum(dim=-1)
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()

    # ---- SAFE logging for DDP ----
    def on_train_start(self) -> None:
        if getattr(self, "logger", None) is not None and hasattr(self.logger, "log_hyperparams"):
            try:
                self.logger.log_hyperparams({
                    "label_smoothing": float(self._ls_eps),
                    "used_pretraining": int(getattr(self, "_loaded_pretrain", False)),
                })
            except Exception:
                pass
        return super().on_train_start()

    def on_train_epoch_start(self) -> None:
        self.log(
            "label_smoothing",
            torch.tensor(self._ls_eps, device=self.device),
            prog_bar=False,
            on_epoch=True,
            sync_dist=True,
        )
        return super().on_train_epoch_start()

    # ---- fine-tune policy ----
    def _get_blocks(self) -> Optional[List[nn.Module]]:
        # standard TransformerEncoder: blocks
        if hasattr(self.target_encoder, "blocks"):
            b = self.target_encoder.blocks
            return list(b) if isinstance(b, (nn.ModuleList, list, tuple)) else None
        # criss-cross variant: cc_stack.blocks
        if hasattr(self.target_encoder, "cc_stack") and hasattr(self.target_encoder.cc_stack, "blocks"):
            b = self.target_encoder.cc_stack.blocks
            return list(b) if isinstance(b, (nn.ModuleList, list, tuple)) else None
        return None

    def _apply_finetune_policy(self) -> None:
        # freeze all encoder params by default
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        # chan conv always trainable
        if self.chan_conv is not None:
            for p in self.chan_conv.parameters():
                p.requires_grad_(True)
        for p in self.proj_in.parameters():
            p.requires_grad_(True)

        if self._freeze_encoder:
            self.target_encoder.eval()
            return

        # unfreeze all encoder params
        if self._ft_top_k <= 0:
            for p in self.target_encoder.parameters():
                p.requires_grad_(True)
            self.target_encoder.train()
            return

        # unfreeze top-k blocks only
        blocks = self._get_blocks()
        if blocks:
            k = min(self._ft_top_k, len(blocks))
            for blk in blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)

        # always unfreeze final norm-ish layer if present
        for attr in ("norm", "final_norm", "ln_f"):
            if hasattr(self.target_encoder, attr):
                for p in getattr(self.target_encoder, attr).parameters():
                    p.requires_grad_(True)

        self.target_encoder.train()

    # ---- forward ----
    def forward(self, x: torch.Tensor):
        # set encoder mode consistent with finetune policy
        if any(p.requires_grad for p in self.target_encoder.parameters()):
            self.target_encoder.train()
        else:
            self.target_encoder.eval()

        xa = x  # [B,C,T] or [B,C,N,P]

        # (1) optional 1x1 channel-mixing conv over time
        if self.use_chan_conv and self.chan_conv is not None:
            if xa.dim() == 3:      # [B,C,T]
                xa = self.chan_conv(xa)                    # Conv1d over time axis
            elif xa.dim() == 4:    # [B,C,N,D] -> flatten time, apply, restore
                B, C, N, D = xa.shape
                xa_flat = xa.reshape(B, C, N * D)
                xa_flat = self.chan_conv(xa_flat)
                xa = xa_flat.reshape(B, self.target_chans, N, P)
            else:
                raise ValueError(f"Unexpected shape for chan_conv: {tuple(xa.shape)}")

        # (2) ensure token shape [B, C, N, patch]
        if xa.dim() == 3:
            B, C, T = xa.shape
            assert T % self.patch == 0, f"T={T} not multiple of patch={self.patch}"
            xa = xa.unfold(dimension=2, size=self.patch, step=self.patch).contiguous()
        elif xa.dim() == 4:
            B, C, N, P = xa.shape
            assert P == self.patch, f"Got P={P}, expected patch={self.patch}"
        else:
            raise ValueError(f"Expected [B,C,T] or [B,C,N,D], got {tuple(xa.shape)}")

        # (3) local projection → encoder → head
        xa = self.proj_in(xa)  # [B,C,N,200]
        B, C, N, D = xa.shape
        keep_all = torch.arange(N * C, device=xa.device, dtype=torch.long)

        z = self.target_encoder(
            xa,
            chan_ids=self.chans_id,
            mask_x=keep_all,
            mask_t=None,
        )  # [B,N,C,D_enc]

        logits = self.head(z)
        return x, logits  # x_vis (pre-encoder tokens) + logits

    # ---- loops ----
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        x_vis, logit = self.forward(x)
        loss = self._loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log('train_loss', loss,           on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_accuracy', accuracy,   on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x_vis.mean(),     on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x_vis.max(),      on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x_vis.min(),      on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x_vis.std(),      on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        try:
            opt = self.optimizers()
            if not isinstance(opt, (list, tuple)):
                opt = [opt]
            if opt and hasattr(opt[0], "param_groups"):
                lr = opt[0].param_groups[0].get("lr", None)
                if lr is not None:
                    self.log(
                        "lr",
                        torch.tensor(lr, device=self.device),
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
        except Exception:
            pass

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        _, logit = self.forward(x)
        loss = self._loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log('valid_loss',     loss,     on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.running_scores["valid"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_validation_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["valid"]) == 0:
            return super().on_validation_epoch_end()
        labels, scores = zip(*self.running_scores["valid"])
        y_true = torch.cat(list(labels), dim=0).cpu().numpy()
        y_score = torch.cat(list(scores), dim=0).cpu().numpy()
        if self.is_binary and y_score.shape[1] == 2:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","pr_auc","f1_weighted", "auroc","f1_macro","f1_micro"]
        else:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted","f1_macro","f1_micro"]
        results = get_metrics(y_score, y_true, metrics, self.is_binary)
        for k, v in results.items():
            self.log('valid_'+k, torch.tensor(v, device=self.device), on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        _, logit = self.forward(x)
        loss = self._loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log("test_loss",     loss,     on_epoch=True, on_step=False, prog_bar=True,  sync_dist=True)
        self.log('test_accuracy', accuracy, on_epoch=True, on_step=False,                sync_dist=True)
        self.running_scores["test"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_test_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["test"]) == 0:
            return super().on_test_epoch_end()
        labels, scores = zip(*self.running_scores["test"])
        y_true = torch.cat(list(labels), dim=0).cpu().numpy()
        y_score = torch.cat(list(scores), dim=0).cpu().numpy()
        if self.is_binary and y_score.shape[1] == 2:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","pr_auc","f1_weighted", "auroc","f1_macro","f1_micro"]
        else:
            metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted","f1_macro","f1_micro"]
        results = get_metrics(y_score, y_true, metrics, self.is_binary)
        for k, v in results.items():
            self.log('test_'+k, torch.tensor(v, device=self.device), on_epoch=True, on_step=False, sync_dist=True)
        return super().on_test_epoch_end()

    # ---- utils ----
    def _prepare_chans_id(self, model_montage: Sequence[str]) -> torch.Tensor:
        # Only used if encoder has channel embedding.
        if hasattr(self.target_encoder, "prepare_chan_ids"):
            ids = self.target_encoder.prepare_chan_ids(list(model_montage))
            return ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids, dtype=torch.long)
        return torch.arange(len(model_montage), dtype=torch.long)

    # ---- optim ----
    def configure_optimizers(self):
        head_params: List[nn.Parameter] = []
        head_params += list(self.proj_in.parameters())
        head_params += list(self.head.parameters())
        if self.chan_conv is not None:
            head_params += list(self.chan_conv.parameters())

        enc_params = [p for p in self.target_encoder.parameters() if p.requires_grad]

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
        if enc_params:
            groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

        optimizer = torch.optim.AdamW(groups)

        if self._scheduler.lower() == "onecycle":
            max_lrs = [self._lr_head] + ([self._lr_enc] if enc_params else [])

            # --- IMPORTANT: use Lightning’s computed stepping batches ---
            # available when configure_optimizers is called (trainer is attached)
            total_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0) or 0)

            # Fallback if ever called without trainer (rare)
            if total_steps <= 0:
                total_steps = int(self._steps_per_epoch * self._max_epochs)

            # --- Resume-safe clamp (fixes your exact off-by-one crash) ---
            total_steps = max(total_steps, int(self.global_step) + 1)

            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self._warmup_pct,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        if build_optim_and_sched is not None and self._scheduler.lower() != "none":
            return build_optim_and_sched(
                groups,
                scheduler=self._scheduler,
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                warmup_pct=self._warmup_pct,
            )

        return {"optimizer": optimizer}

    # def configure_optimizers(self):
    #     head_params: List[nn.Parameter] = []
    #     head_params += list(self.proj_in.parameters())
    #     head_params += list(self.head.parameters())
    #     if self.chan_conv is not None:
    #         head_params += list(self.chan_conv.parameters())

    #     enc_params = [p for p in self.target_encoder.parameters() if p.requires_grad]

    #     groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
    #     if enc_params:
    #         groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

    #     optimizer = torch.optim.AdamW(groups)

    #     if self._scheduler.lower() == "onecycle":
    #         max_lrs = [self._lr_head] + ([self._lr_enc] if enc_params else [])
    #         sched = torch.optim.lr_scheduler.OneCycleLR(
    #             optimizer,
    #             max_lr=max_lrs,
    #             steps_per_epoch=self._steps_per_epoch,
    #             epochs=self._max_epochs,
    #             pct_start=self._warmup_pct,
    #         )
    #         return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    #     if build_optim_and_sched is not None and self._scheduler.lower() != "none":
    #         return build_optim_and_sched(
    #             groups,
    #             scheduler=self._scheduler,
    #             steps_per_epoch=self._steps_per_epoch,
    #             epochs=self._max_epochs,
    #             warmup_pct=self._warmup_pct,
    #         )

    #     return {"optimizer": optimizer}
