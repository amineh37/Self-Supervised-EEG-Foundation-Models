# modules/eegpt_lp.py
from __future__ import annotations
from typing import Sequence, Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .utils import Conv1dWithConstraint
from backbones.eegpt import EEGTransformer

from .base import LitBase
from .optim import build_optim_and_sched

try:
    from utils_eval import get_metrics
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


# ---------------------- Constrained layers (EEGNet-style) ----------------------
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.doWeightNorm = doWeightNorm
        self.max_norm = float(max_norm)

    def forward(self, x):
        if self.doWeightNorm and self.weight is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm: bool = True, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.doWeightNorm = doWeightNorm
        self.max_norm = float(max_norm)

    def forward(self, x):
        if self.doWeightNorm and self.weight is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class LitEEGPTTUEV(LitBase):
    """
    EEGPT-style finetune for TUEV with an optional *single* front-end:
      - chan_conv: 1×1 Conv1d channel mixer (in_chans -> target_chans)
      - time_chan_conv: EEGNet-style temporal depthwise Conv2d

    Inputs:  [B, C_in, T]
    Encoder: EEGTransformer(x, chan_ids) -> [B, N, E, D]
    Head:    flatten -> Dropout -> LinearWithConstraint -> logits
    """

    def __init__(
        self,
        models_cfg: Dict[str, Any],
        load_path: str,
        *,
        # dataset/model channels
        in_chans: int = 22,
        channel_names: Optional[Sequence[str]] = None,
        use_channel_names: Optional[Sequence[str]] = None,
        target_chans: Optional[int] = 19,

        # dummy args for api compatible
        head_type: str = "ignore",
        dropout: float = 0.5,

        # sequence/patch
        seq_len: int = 1000,   # 5 s @ 200 Hz
        patch: Optional[int] = None,  # kept for CLI compatibility (unused; read from enc cfg)

        # task
        num_classes: int = 6,

        # train policy
        freeze_encoder: bool = True,  # linear probe vs full FT
        ft_top_k: int = 0,            # if >0 and not frozen, unfreeze last K transformer blocks

        # optim/schedule (fallbacks if not set in cfg)
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
            in_ch_for_frontend = len(dataset_ch_names)
        else:
            dataset_ch_names = None
            in_ch_for_frontend = int(in_chans)

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
        self.in_chans = int(in_ch_for_frontend)
        self.target_chans = int(target_chans)
        self.seq_len = int(seq_len)
        self.example_input_array = torch.zeros(2, self.in_chans, self.seq_len, dtype=torch.float32)

        # ---------------- read cfg (encoder & classifier) ----------------
        enc_cfg = dict(models_cfg.get("encoder", {}))
        clf_cfg = dict(models_cfg.get("classifier", {}))

        # Encoder knobs (defaults mirror EEGPTClassifier)
        embed_dim   = int(enc_cfg.get("embed_dim",   512))
        embed_num   = int(enc_cfg.get("embed_num",   4))
        depth       = int(enc_cfg.get("depth",       8))
        num_heads   = int(enc_cfg.get("num_heads",   8))
        mlp_ratio   = float(enc_cfg.get("mlp_ratio", 4.0))
        drop_rate   = float(enc_cfg.get("drop_rate", 0.0))
        attn_drop   = float(enc_cfg.get("attn_drop_rate", 0.0))
        drop_path   = float(enc_cfg.get("drop_path_rate", 0.0))
        init_std    = float(enc_cfg.get("init_std", 0.02))
        qkv_bias    = bool(enc_cfg.get("qkv_bias", True))
        # patching
        patch_size   = int(patch)
        return_attn  = int(enc_cfg.get("return_attention_layer", -1))

        # Classifier/front-end knobs
        use_chan_conv_cfg      = bool(clf_cfg.get("use_chan_conv", False))
        use_time_chan_conv_cfg = bool(clf_cfg.get("use_time_chan_conv", True))
        conv_kernel_time       = int(clf_cfg.get("conv_kernel_time", 55))
        chan_max_norm          = float(clf_cfg.get("chan_max_norm", 1.0))
        conv_dropout           = float(clf_cfg.get("conv_dropout", 0.8))
        head_dropout           = float(clf_cfg.get("dropout", 0.5))
        head_max_norm          = float(clf_cfg.get("head_max_norm", 1.0))

        # ---- enforce mutual exclusivity ----
        if use_chan_conv_cfg and use_time_chan_conv_cfg:
            raise ValueError("Enable at most one front-end: set either use_chan_conv=True or use_time_chan_conv=True (not both).")

        self.use_chan_conv: bool = use_chan_conv_cfg
        self.chan_conv: Optional[nn.Module] = None
        self.time_chan_conv: Optional[nn.Module] = None

        # ---------------- front-end ----------------
        if use_time_chan_conv_cfg:
            k_time = int(conv_kernel_time)
            pad_t = (k_time - 1) // 2  # 'same' padding for time dimension
            self.time_chan_conv = nn.Sequential(
                Conv2dWithConstraint(
                    in_channels=self.in_chans,
                    out_channels=self.target_chans,
                    kernel_size=1,
                    doWeightNorm=True,
                    max_norm=chan_max_norm,
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
                nn.Dropout(conv_dropout),
            )
        elif use_chan_conv_cfg:
            self.chan_conv = Conv1dWithConstraint(
                in_channels=self.in_chans,
                out_channels=self.target_chans,
                kernel_size=1,
                max_norm=chan_max_norm,
            )

        # ---------------- backbone ----------------
        self.target_encoder = EEGTransformer(
            img_size=[self.target_chans, self.seq_len],
            patch_size=patch_size,
            embed_dim=embed_dim,
            embed_num=embed_num,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop,
            drop_path_rate=drop_path,
            init_std=init_std,
            qkv_bias=qkv_bias,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            return_attention_layer=return_attn,
        )
        self.chans_id = self._prepare_chans_id(self.use_channel_names)
        self._load_target_encoder(load_path)

        # time patches from encoder itself
        n_time = int(self.target_encoder.num_patches[1])

        # ---------------- EEGPT-style head ----------------
        head_in = n_time * embed_num * embed_dim
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            LinearWithConstraint(head_in, num_classes, doWeightNorm=True, max_norm=head_max_norm),
        )

        # ---------------- train policy ----------------
        self._freeze_encoder = bool(freeze_encoder)
        self._ft_top_k = int(max(0, ft_top_k))
        self._apply_finetune_policy(depth_hint=depth)

        # ---------------- optim cfg (fallbacks here; you can also read from models_cfg["optim"]) ----------------
        opt_cfg = dict(models_cfg.get("optim", {}))
        self._max_epochs      = int(opt_cfg.get("max_epochs",      max_epochs))
        self._steps_per_epoch = int(opt_cfg.get("steps_per_epoch", steps_per_epoch))
        self._lr_head         = float(opt_cfg.get("lr_head",       lr_head))
        self._lr_enc          = float(opt_cfg.get("lr_enc",        lr_enc))
        self._wd_head         = float(opt_cfg.get("wd_head",       wd_head))
        self._wd_enc          = float(opt_cfg.get("wd_enc",        wd_enc))
        self._scheduler       = str(opt_cfg.get("scheduler",       scheduler))
        self._warmup_pct      = float(opt_cfg.get("warmup_pct",    warmup_pct))

        # epoch buffers for richer metrics
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity=False

    # ---------------- Top-K FT helpers ----------------
    def _find_block_list(self, enc: nn.Module) -> Tuple[List[nn.Module], str]:
        for name in ["blocks", "layers", "encoder_blocks", "transformer_blocks"]:
            if hasattr(enc, name):
                bl = getattr(enc, name)
                if isinstance(bl, (nn.ModuleList, list)) and len(bl) > 0:
                    return list(bl), name
        longest: List[nn.Module] = []
        longest_name = ""
        for n, m in enc.named_children():
            if isinstance(m, nn.ModuleList) and len(m) > len(longest):
                longest = list(m)
                longest_name = n
        return longest, longest_name or "<unknown>"

    def _apply_finetune_policy(self, *, depth_hint: Optional[int] = None) -> None:
        if self._freeze_encoder:
            for p in self.target_encoder.parameters():
                p.requires_grad_(False)
            self.target_encoder.eval()
            if self._ft_top_k > 0:
                print(f"[FINETUNE] freeze_encoder=True -> ignoring ft_top_k={self._ft_top_k}")
            print("[FINETUNE] Encoder frozen (linear probe).")
            return

        if self._ft_top_k <= 0:
            for p in self.target_encoder.parameters():
                p.requires_grad_(True)
            self.target_encoder.train()
            print("[FINETUNE] Full encoder fine-tune (ft_top_k<=0).")
            return

        blocks, name = self._find_block_list(self.target_encoder)
        total = len(blocks) if blocks else int(depth_hint or 0)

        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

        if not blocks or total == 0:
            for p in self.target_encoder.parameters():
                p.requires_grad_(True)
            self.target_encoder.train()
            print(f"[FINETUNE] Could not locate transformer blocks (attr='{name}'). Falling back to FULL fine-tune.")
            return

        k = min(self._ft_top_k, total)
        for b in blocks[-k:]:
            for p in b.parameters():
                p.requires_grad_(True)

        for attr in ["norm", "ln_f", "final_norm"]:
            if hasattr(self.target_encoder, attr):
                for p in getattr(self.target_encoder, attr).parameters():
                    p.requires_grad_(True)

        self.target_encoder.train()
        trainable_ids = [i for i, b in enumerate(blocks) if any(p.requires_grad for p in b.parameters())]
        print(f"[FINETUNE] Top-K enabled: unfreezing last {k}/{total} blocks (attr='{name}'). Trainable block idx: {trainable_ids}")

    # ---------------- Lightning ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, T] -> optional front-end -> EEGTransformer -> flatten -> head
        """
        enc_has_trainable = any(p.requires_grad for p in self.target_encoder.parameters())
        self.target_encoder.train(enc_has_trainable)

        xa = x  # [B,C,T]

        # (1) optional 1×1 channel-mixing conv over time
        if self.chan_conv is not None:
            xa = self.chan_conv(xa)  # [B,target_chans,T]

        # (2) optional EEGNet-style temporal depthwise conv
        if self.time_chan_conv is not None:
            xa2d = xa[:, :, None, :]      # [B,C,1,T]
            xa = self.time_chan_conv(xa2d)[:, :, 0]  # [B,target_chans,T]

        z = self.target_encoder(xa, self.chans_id.to(xa.device))  # [B, N, E, D]
        z = z.flatten(1)                                          # [B, N*E*D]
        logits = self.head(z)                                     # [B, K]
        return x, logits

    def on_train_epoch_start(self) -> None:
        enc_has_trainable = any(p.requires_grad for p in self.target_encoder.parameters())
        self.target_encoder.train(enc_has_trainable)

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"]=[]
        return super().on_test_epoch_start()

    # ---------------- train/val/test loops ----------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        x, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        acc = (pred == y).float().mean()

        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_accuracy',  acc,  on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(),  on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(),  on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(),  on_epoch=True, on_step=False, sync_dist=True)
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
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        _, logit = self.forward(x)
        loss = self.loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        acc = (pred == y).float().mean()

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_accuracy',  acc,  on_epoch=True, on_step=False, sync_dist=True)
        if get_metrics is not None:
            self.running_scores["valid"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
        if get_metrics is None or len(self.running_scores["valid"]) == 0:
            return super().on_validation_epoch_end()

        label, y_score = zip(*self.running_scores["valid"])
        label  = torch.cat(label,  dim=0).numpy()
        y_score= torch.cat(y_score,dim=0).numpy()

        metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted","f1_macro","f1_micro"]
        results = get_metrics(y_score, label, metrics, False)
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
        self.log("test_accuracy",  acc,  on_epoch=True, on_step=False)
        if get_metrics is not None:
            self.running_scores["test"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_test_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_test_epoch_end()
        if get_metrics is None or len(self.running_scores["test"]) == 0:
            return super().on_test_epoch_end()

        label, y_score = zip(*self.running_scores["test"])
        label  = torch.cat(label,  dim=0).numpy()
        y_score= torch.cat(y_score,dim=0).numpy()

        metrics = ["accuracy","balanced_accuracy","cohen_kappa","f1_weighted","f1_macro","f1_micro"]
        results = get_metrics(y_score, label, metrics, False)
        for k, v in results.items():
            self.log('test_'+k, v, on_epoch=True, on_step=False, sync_dist=True)
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
        head_params: List[nn.Parameter] = []
        if self.chan_conv is not None:
            head_params += list(self.chan_conv.parameters())
        if self.time_chan_conv is not None:
            head_params += list(self.time_chan_conv.parameters())
        head_params += list(self.head.parameters())

        enc_params = [p for p in self.target_encoder.parameters() if p.requires_grad]

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
        if len(enc_params) > 0:
            groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

        if self._scheduler.lower() != "onecycle":
            return build_optim_and_sched(
                groups,
                scheduler=self._scheduler,
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                warmup_pct=self._warmup_pct,
            )

        optimizer = torch.optim.AdamW(groups)
        max_lrs = [self._lr_head] + ([self._lr_enc] if len(enc_params) > 0 else [])
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,
            steps_per_epoch=self._steps_per_epoch,
            epochs=self._max_epochs,
            pct_start=self._warmup_pct,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
