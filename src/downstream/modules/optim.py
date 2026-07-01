from typing import Iterable, Tuple, Any, Dict
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, LinearLR, CosineAnnealingLR, SequentialLR

def build_optim_and_sched(
    head_params: Iterable,
    enc_params: Iterable,
    *,
    steps_per_epoch: int,
    max_epochs: int,
    lr_head: float = 4e-4,
    lr_enc: float = 1e-4,
    wd_head: float = 0.01,
    wd_enc: float = 0.05,
    scheduler: str = "onecycle",   # {"onecycle","cosine","constant"}
    warmup_pct: float = 0.2,
    freeze_encoder: bool = True,   # <— caller decides if encoder is trainable
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build AdamW with param groups (head vs encoder) and an LR scheduler.
    If `freeze_encoder` is True or `enc_params` empty, the optimizer only sees the head group.
    We do NOT toggle requires_grad here — do that in the LightningModule.
    """
    # --- normalize counts ---
    steps_per_epoch = max(1, int(steps_per_epoch))
    max_epochs      = max(1, int(max_epochs))
    total_steps     = steps_per_epoch * max_epochs
    warmup_steps    = max(1, int(total_steps * float(warmup_pct)))

    # --- groups ---
    head_params = list(head_params)
    enc_params  = list(enc_params)
    use_encoder = (not freeze_encoder) and len(enc_params) > 0

    param_groups = [
        {"params": head_params, "lr": lr_head, "weight_decay": wd_head},
    ]
    if use_encoder:
        param_groups.append({"params": enc_params, "lr": lr_enc, "weight_decay": wd_enc})

    optimizer = AdamW(param_groups)

    # --- schedulers ---
    sch_name = str(scheduler).lower().strip()

    if sch_name == "onecycle":
        # OneCycle expects max_lr per param group
        max_lr = lr_head if not use_encoder else [lr_head, lr_enc]
        sched = OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
            pct_start=float(warmup_pct),
        )
        return optimizer, {"scheduler": sched, "interval": "step", "frequency": 1}

    elif sch_name == "cosine":
        # Warmup (linear) then cosine anneal, all stepped per step
        warm = LinearLR(
            optimizer=optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        eta_min = 0.1 * (lr_head if not use_encoder else min(lr_head, lr_enc))
        cos = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=eta_min,
        )
        sched = SequentialLR(optimizer, [warm, cos], milestones=[warmup_steps])
        return optimizer, {"scheduler": sched, "interval": "step", "frequency": 1}

    elif sch_name == "constant":
        # No-op schedule (keeps LR constant). Use a 1-iter LinearLR and step per epoch.
        const = LinearLR(optimizer=optimizer, start_factor=1.0, end_factor=1.0, total_iters=1)
        return optimizer, {"scheduler": const, "interval": "epoch", "frequency": 1}

    else:
        raise ValueError(f"Unknown scheduler '{scheduler}'. Choose from: onecycle, cosine, constant.")
