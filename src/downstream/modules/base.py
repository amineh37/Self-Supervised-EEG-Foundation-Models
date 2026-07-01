# modules/base.py
from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import pytorch_lightning as pl
from utils_eval import get_metrics  # external: expected in user's project


class LitBase(pl.LightningModule):
    """
    Minimal Lightning base with consistent metric logging.
    Expects subclasses to define:
      - self.loss_fn
      - forward(x) -> logits [B, K]
    """

    def __init__(self) -> None:
        super().__init__()
        self._buf: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {"valid": [], "test": []}
        self._did_sanity = False

    # ------------------------------- helpers -------------------------------
    def _log_split(self, split: str, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(-1) == y).float().mean()
        self.log(f"{split}_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{split}_acc", acc, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    # -------------------------------- train --------------------------------
    def training_step(self, batch, _):
        x, y = batch
        y = y.long()
        logits = self(x)
        return self._log_split("train", logits, y)

    # ------------------------------- validate ------------------------------
    def on_validation_epoch_start(self) -> None:
        self._buf["valid"] = []

    def validation_step(self, batch, _):
        x, y = batch
        y = y.long()
        with torch.inference_mode():
            logits = self(x)
            loss = self._log_split("valid", logits, y)
            self._buf["valid"].append((y.detach().cpu(), logits.detach().cpu()))
        return loss

    def on_validation_epoch_end(self) -> None:
        # Skip sanity checking epoch
        if not self._did_sanity:
            self._did_sanity = True
            return
        self._finalize("valid")

    # --------------------------------- test --------------------------------
    def on_test_epoch_start(self) -> None:
        self._buf["test"] = []

    def test_step(self, batch, _):
        x, y = batch
        y = y.long()
        with torch.inference_mode():
            logits = self(x)
            loss = self._log_split("test", logits, y)
            self._buf["test"].append((y.detach().cpu(), logits.detach().cpu()))
        return loss

    def on_test_epoch_end(self) -> None:
        self._finalize("test")

    # ------------------------------ utilities ------------------------------
    def _finalize(self, split: str) -> None:
        if not self._buf[split]:
            return
        yt, yp = zip(*self._buf[split])
        y_true = torch.cat(yt, 0).numpy()
        y_logit = torch.cat(yp, 0).numpy()
        is_binary = (y_logit.shape[1] == 2)
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa",
                   "f1_weighted", "f1_macro", "f1_micro"]
        for k, v in get_metrics(y_logit, y_true, metrics, is_binary=is_binary).items():
            self.log(f"{split}_{k}", v, on_epoch=True, on_step=False, sync_dist=True)
