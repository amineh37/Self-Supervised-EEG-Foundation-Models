from __future__ import annotations
from typing import Sequence, Dict, Any, Optional, List, Tuple

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

from downstream.backbones.stelar import TransformerEncoder

# Metrics (tolerant import)
try:
    from utils_eval import get_metrics
except Exception:
    try:
        from downstream.utils_eval import get_metrics  # type: ignore
    except Exception:
        get_metrics = None


# ---- Constrained 2D conv (EEGNet-style) ----
class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.doWeightNorm = doWeightNorm
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.doWeightNorm and self.weight is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


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


class LitSTELARPatchAdaptTUEV(LitBase):
    """
    STELAR fine-tune with an optional *single* front-end:
      - raw_channel_map: deterministic channel select/reorder by channel names
      - chan_conv:       standard 1×1 Conv1d channel mixer
      - time_chan_conv:  EEGNet-style temporal depthwise Conv2d

    Exactly one of them can be enabled (or neither).

    Inputs:  x ∈ [B, C_in, T] or [B, C_in, N, patch]
    Encoder: z ∈ [B, N, C_model, D_enc]
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
        # ---- front-end toggles (mutually exclusive) ----
        use_raw_channel_map: bool = False,  # direct select/reorder raw channels by name
        raw_map_missing: str = "error",     # {"error", "zero"}
        use_chan_conv: bool = False,        # 1×1 Conv1d channel mixer
        use_time_chan_conv: bool = False,   # EEGNet-style temporal depthwise Conv2d
        chan_max_norm: float = 1.0,         # max-norm for constrained convs
        conv_kernel_time: int = 55,
        conv_dropout: float = 0.8,
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
    ):
        super().__init__()

        # --- sanity: enforce exclusivity of front-ends ---
        n_frontends = int(bool(use_raw_channel_map)) + int(bool(use_chan_conv)) + int(bool(use_time_chan_conv))
        if n_frontends > 1:
            raise ValueError(
                "Enable at most one front-end among "
                "{use_raw_channel_map, use_chan_conv, use_time_chan_conv}."
            )

        if raw_map_missing not in {"error", "zero"}:
            raise ValueError(f"raw_map_missing must be one of {{'error','zero'}}, got {raw_map_missing}")

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
            in_ch_for_frontend = len(dataset_ch_names)
        else:
            dataset_ch_names = None
            in_ch_for_frontend = int(in_chans)

        if use_channel_names is None:
            use_channel_names = list(DEFAULT_CH_19 if (target_chans == 19) else DEFAULT_CH_58)
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
        self.in_chans = int(in_ch_for_frontend)
        self.target_chans = int(target_chans)

        # --- geometry ---
        self.seq_len = int(seq_len)
        self.patch = int(patch)
        assert self.seq_len % self.patch == 0, "seq_len must be divisible by patch"
        self.N = self.seq_len // self.patch

        self.example_input_array = torch.zeros(
            8, self.in_chans, self.seq_len, dtype=torch.float32
        )

        # --- raw deterministic channel mapping ---
        self.use_raw_channel_map = bool(use_raw_channel_map)
        self.raw_map_missing = str(raw_map_missing)

        if self.use_raw_channel_map and dataset_ch_names is None:
            raise ValueError(
                "use_raw_channel_map=True requires channel_names to be provided."
            )

        raw_map_index, raw_map_valid = self._build_raw_channel_map(
            dataset_ch_names=dataset_ch_names,
            target_channel_names=self.use_channel_names,
            missing_policy=self.raw_map_missing,
        )
        self.register_buffer("raw_map_index", raw_map_index, persistent=False)
        self.register_buffer("raw_map_valid", raw_map_valid, persistent=False)

        # --- OPTIONAL: 1×1 channel-mixing conv (over time axis) ---
        self.use_chan_conv = bool(use_chan_conv)
        self.chan_conv: Optional[nn.Module] = None
        if self.use_chan_conv:
            self.chan_conv = Conv1dWithConstraint(
                self.in_chans,
                self.target_chans,
                kernel_size=1,
                max_norm=float(chan_max_norm),
            )

        # --- OPTIONAL: EEGNet-style temporal depthwise conv ---
        self.use_time_chan_conv = bool(use_time_chan_conv)
        self.time_chan_conv: Optional[nn.Module] = None
        if self.use_time_chan_conv:
            k_time = int(conv_kernel_time)
            pad_t = (k_time - 1) // 2
            self.time_chan_conv = nn.Sequential(
                Conv2dWithConstraint(
                    in_channels=self.in_chans,
                    out_channels=self.target_chans,
                    kernel_size=1,
                    doWeightNorm=True,
                    max_norm=float(chan_max_norm),
                ),
                nn.BatchNorm2d(self.target_chans),
                nn.GELU(),
                nn.Conv2d(
                    in_channels=self.target_chans,
                    out_channels=self.target_chans,
                    kernel_size=(1, k_time),
                    groups=self.target_chans,
                    padding=(0, pad_t),
                ),
                nn.BatchNorm2d(self.target_chans),
                nn.Dropout(float(conv_dropout)),
            )

        # --- encoder & head ---
        enc_cfg = dict(models_cfg.get("encoder", {}))
        self.D_enc = int(enc_cfg.get("embed_dim", 768))
        self.pre_proj_ln = nn.LayerNorm(self.patch, elementwise_affine=True)
        self.proj_in = nn.Linear(self.patch, 200)

        self.target_encoder = TransformerEncoder(
            img_size=(self.target_chans, self.N),
            **enc_cfg,
        )

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

                enc_state = {
                    k[len("target_encoder."):]: v
                    for k, v in state.items()
                    if isinstance(k, str) and k.startswith("target_encoder.")
                }
                if not enc_state:
                    enc_state = {k: v for k, v in state.items() if isinstance(k, str)}

                missing, unexpected = self.target_encoder.load_state_dict(enc_state, strict=False)

                tgt_state = self.target_encoder.state_dict()
                matched = sum(
                    1 for k, v in enc_state.items()
                    if k in tgt_state and tgt_state[k].shape == v.shape
                )

                if self.global_rank == 0:
                    print(
                        f"[ENC LOAD] matched {matched}/{len(enc_state)} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})"
                    )
                    if unexpected:
                        print(f"unexpected modules: {unexpected}")

                self._loaded_pretrain = True

            except Exception as e:
                if self.global_rank == 0:
                    print(f"[WARN] Failed to load encoder weights from {load_path}: {e}")
                    print("[INIT] Keeping original module initialization.")

        elif self.global_rank == 0:
            print("[INIT] No pretrained checkpoint provided. Keeping original module initialization.")

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
        self._meta_logged = False

    # ---- channel-name helpers ----
    @staticmethod
    def _canon_ch_name(name: str) -> str:
        s = str(name).upper().strip().replace("EEG ", "").replace("-REF", "").replace("-LE", "").replace("-AVG", "")
        alias = {
            "T7": "T3",
            "T8": "T4",
            "P7": "T5",
            "P8": "T6",
        }
        return alias.get(s, s)

    @classmethod
    def _build_raw_channel_map(
        cls,
        dataset_ch_names: Optional[Sequence[str]],
        target_channel_names: Sequence[str],
        missing_policy: str = "error",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          raw_map_index: [C_target] source index if valid else 0
          raw_map_valid: [C_target] bool mask indicating whether that target channel exists
        """
        C_target = len(target_channel_names)

        if dataset_ch_names is None:
            return (
                torch.zeros(C_target, dtype=torch.long),
                torch.zeros(C_target, dtype=torch.bool),
            )

        src_lookup = {}
        for i, ch in enumerate(dataset_ch_names):
            src_lookup[cls._canon_ch_name(ch)] = i

        raw_map_index = torch.zeros(C_target, dtype=torch.long)
        raw_map_valid = torch.zeros(C_target, dtype=torch.bool)

        missing = []
        for j, ch in enumerate(target_channel_names):
            key = cls._canon_ch_name(ch)
            if key in src_lookup:
                raw_map_index[j] = int(src_lookup[key])
                raw_map_valid[j] = True
            else:
                missing.append(ch)

        if missing and missing_policy == "error":
            raise ValueError(
                f"Missing target channels for raw mapping: {missing}. "
                f"Provided dataset channels: {list(dataset_ch_names)}"
            )

        return raw_map_index, raw_map_valid

    def _apply_raw_channel_map(self, xa: torch.Tensor) -> torch.Tensor:
        """
        xa:
          [B, C_in, T] or [B, C_in, N, D]
        return:
          [B, C_target, T] or [B, C_target, N, D]
        """
        if xa.dim() not in (3, 4):
            raise ValueError(f"Expected [B,C,T] or [B,C,N,D], got {tuple(xa.shape)}")

        B = xa.shape[0]
        C_out = int(self.target_chans)
        valid_idx = torch.nonzero(self.raw_map_valid, as_tuple=False).flatten()

        if xa.dim() == 3:
            T = xa.shape[2]
            out = xa.new_zeros((B, C_out, T))
            if valid_idx.numel() > 0:
                src_idx = self.raw_map_index[valid_idx]
                out[:, valid_idx, :] = xa.index_select(dim=1, index=src_idx)
            return out

        N, D = xa.shape[2], xa.shape[3]
        out = xa.new_zeros((B, C_out, N, D))
        if valid_idx.numel() > 0:
            src_idx = self.raw_map_index[valid_idx]
            out[:, valid_idx, :, :] = xa.index_select(dim=1, index=src_idx)
        return out

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
        for attr in ("blocks", "layers"):
            if hasattr(self.target_encoder, attr):
                b = getattr(self.target_encoder, attr)
                return list(b) if isinstance(b, (nn.ModuleList, list, tuple)) else None
        return None

    def _apply_finetune_policy(self) -> None:
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        # front-end / projection trainability
        # raw_channel_map has no trainable params
        if self.chan_conv is not None:
            for p in self.chan_conv.parameters():
                p.requires_grad_(True)
        if self.time_chan_conv is not None:
            for p in self.time_chan_conv.parameters():
                p.requires_grad_(True)
        for p in self.proj_in.parameters():
            p.requires_grad_(True)

        if self._freeze_encoder:
            self.target_encoder.eval()
            return

        if self._ft_top_k <= 0:
            for p in self.target_encoder.parameters():
                p.requires_grad_(True)
            self.target_encoder.train()
            return

        blocks = self._get_blocks()
        if blocks:
            k = min(self._ft_top_k, len(blocks))
            for blk in blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)

        for attr in ("norm", "final_norm", "ln_f"):
            if hasattr(self.target_encoder, attr):
                for p in getattr(self.target_encoder, attr).parameters():
                    p.requires_grad_(True)

        self.target_encoder.train()

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(p.requires_grad for p in self.target_encoder.parameters()):
            self.target_encoder.train()
        else:
            self.target_encoder.eval()

        xa = x  # [B,C,T] or [B,C,N,D]

        # (0) optional deterministic raw channel mapping
        if self.use_raw_channel_map:
            xa = self._apply_raw_channel_map(xa)

        # (1) optional 1×1 channel-mixing conv over time
        if self.chan_conv is not None:
            if xa.dim() == 3:
                xa = self.chan_conv(xa)
            elif xa.dim() == 4:
                B, C, N, D = xa.shape
                xa_flat = xa.reshape(B, C, N * D)
                xa_flat = self.chan_conv(xa_flat)
                C2 = xa_flat.shape[1]
                xa = xa_flat.reshape(B, C2, N, D)
            else:
                raise ValueError(f"Unexpected shape for chan_conv: {tuple(xa.shape)}")

        # (2) optional EEGNet-style temporal depthwise conv
        if self.time_chan_conv is not None:
            if xa.dim() == 3:
                xa2d = xa[:, :, None, :]
                xa = self.time_chan_conv(xa2d)[:, :, 0]
            elif xa.dim() == 4:
                B, C, N, D = xa.shape
                xa_flat = xa.reshape(B, C, 1, N * D)
                xa_flat = self.time_chan_conv(xa_flat)
                C2 = xa_flat.shape[1]
                xa = xa_flat.reshape(B, C2, N, D)
            else:
                raise ValueError(f"Unexpected shape for time_chan_conv: {tuple(xa.shape)}")

        # (3) tokenize to [B,C,N,D]
        if xa.dim() == 3:
            B, C, T = xa.shape
            assert T % self.patch == 0, f"T={T} not multiple of patch={self.patch}"
            xa = xa.unfold(dimension=2, size=self.patch, step=self.patch).contiguous()
        elif xa.dim() == 4:
            B, C, N, D = xa.shape
            assert D == self.patch, f"Got D={D}, expected patch={self.patch}"
        else:
            raise ValueError(f"Expected [B,C,T] or [B,C,N,D], got {tuple(xa.shape)}")

        # optional pre-proj normalization if you want to use it
        # xa = self.pre_proj_ln(xa)

        xa = self.proj_in(xa)
        B, C, N, D = xa.shape
        keep_all = torch.arange(N * C, device=xa.device, dtype=torch.long)

        z = self.target_encoder(
            xa,
            chan_ids=self.chans_id,
            mask_x=keep_all,
            mask_t=None,
        )
        logits = self.head(z)
        return x, logits

    # ---- loops ----
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()

        x_vis, logit = self.forward(x)
        loss = self._loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_accuracy", accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log("data_avg", x_vis.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log("data_max", x_vis.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log("data_min", x_vis.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log("data_std", x_vis.std(), on_epoch=True, on_step=False, sync_dist=True)
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

        self.log("valid_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_accuracy", accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.running_scores["valid"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_validation_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["valid"]) == 0:
            return super().on_validation_epoch_end()

        labels, scores = zip(*self.running_scores["valid"])
        y_true = torch.cat(list(labels), dim=0).cpu().numpy()
        y_score = torch.cat(list(scores), dim=0).cpu().numpy()
        print("valid y_true classes:", sorted(set(y_true.tolist())))
        print("valid y_pred classes:", sorted(set(y_score.argmax(1).tolist())))
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score, y_true, metrics, False)
        for k, v in results.items():
            self.log("valid_" + k, v, on_epoch=True, on_step=False, sync_dist=True)

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

        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log("test_accuracy", accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.running_scores["test"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_test_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["test"]) == 0:
            return super().on_test_epoch_end()

        labels, scores = zip(*self.running_scores["test"])
        y_true = torch.cat(list(labels), dim=0).cpu().numpy()
        y_score = torch.cat(list(scores), dim=0).cpu().numpy()

        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
        results = get_metrics(y_score, y_true, metrics, False)
        for k, v in results.items():
            self.log("test_" + k, v, on_epoch=True, on_step=False, sync_dist=True)

        return super().on_test_epoch_end()

    # ---- utils ----
    def _prepare_chans_id(self, model_montage: Sequence[str]) -> torch.Tensor:
        if hasattr(self.target_encoder, "prepare_chan_ids"):
            ids = self.target_encoder.prepare_chan_ids(list(model_montage))
            return ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids, dtype=torch.long)
        return torch.arange(len(model_montage), dtype=torch.long)

    # ---- optim ----
    def configure_optimizers(self):
        head_params = list(self.proj_in.parameters()) + list(self.head.parameters())

        if self.chan_conv is not None:
            head_params += list(self.chan_conv.parameters())
        if self.time_chan_conv is not None:
            head_params += list(self.time_chan_conv.parameters())

        enc_params = [p for p in self.target_encoder.parameters() if p.requires_grad]

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
        if enc_params:
            groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

        optimizer = torch.optim.AdamW(groups)

        if self._scheduler.lower() == "onecycle":
            max_lrs = [self._lr_head] + ([self._lr_enc] if enc_params else [])
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                pct_start=self._warmup_pct,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": sched, "interval": "step"},
            }

        if build_optim_and_sched is not None and self._scheduler.lower() != "none":
            return build_optim_and_sched(
                groups,
                scheduler=self._scheduler,
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                warmup_pct=self._warmup_pct,
            )

        return {"optimizer": optimizer}