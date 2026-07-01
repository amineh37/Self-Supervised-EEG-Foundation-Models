"""W&B-capable logger builder shared by the pretrain / downstream runners.

Default behaviour is unchanged from the original runners: TensorBoard + CSV.
Weights & Biases is *opt-in* (``--logger wandb`` or ``--wandb``) and ``wandb`` is
imported lazily, so the dependency is only required when actually used. Offline
mode is the safe default whenever W&B is requested without an API key, so a run
never blocks on network / login.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pytorch_lightning import loggers as pl_loggers

LOGGER_CHOICES = ("both", "tensorboard", "csv", "wandb", "none")


def add_logging_args(parser):
    """Add the (additive) logging CLI flags to an existing argparse parser."""
    g = parser.add_argument_group("logging / wandb")
    g.add_argument(
        "--logger",
        choices=LOGGER_CHOICES,
        default="both",
        help="File loggers to use. 'both' (default) = TensorBoard + CSV, "
        "as in the original runner. 'wandb' = Weights & Biases only.",
    )
    g.add_argument(
        "--wandb",
        action="store_true",
        help="Add a Weights & Biases logger on top of the selected file loggers.",
    )
    g.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Force W&B offline mode (no network; sync later with `wandb sync`).",
    )
    g.add_argument("--wandb_project", type=str, default="stelar")
    g.add_argument("--wandb_entity", type=str, default=None)
    g.add_argument("--wandb_name", type=str, default=None, help="Run name (default: derived from model_type).")
    g.add_argument("--wandb_tags", type=str, default="", help="Comma-separated tags.")
    g.add_argument("--wandb_notes", type=str, default=None)
    return parser


def _coerce_tags(tags: Union[str, Sequence[str], None]) -> List[str]:
    if not tags:
        return []
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return [str(t).strip() for t in tags if str(t).strip()]


def build_loggers(
    *,
    logger: str = "both",
    save_root: Union[str, Path],
    name: str,
    use_wandb: bool = False,
    wandb_offline: bool = False,
    wandb_project: str = "stelar",
    wandb_entity: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_tags: Union[str, Sequence[str], None] = None,
    wandb_notes: Optional[str] = None,
    wandb_config: Optional[dict] = None,
):
    """Return a list of PyTorch-Lightning loggers (possibly empty).

    Pass the returned list to ``Trainer(logger=...)`` (an empty list is treated
    as "no logger" by the caller).
    """
    save_root = Path(save_root)
    loggers: list = []

    want_tb = logger in ("both", "tensorboard")
    want_csv = logger in ("both", "csv")
    want_wandb = use_wandb or logger == "wandb"

    if want_tb:
        loggers.append(
            pl_loggers.TensorBoardLogger(save_dir=str(save_root / f"{name}_tb"), name="", version=None)
        )
    if want_csv:
        loggers.append(
            pl_loggers.CSVLogger(save_dir=str(save_root / f"{name}_csv"), name="", version=None)
        )

    if want_wandb:
        wandb_logger = make_wandb_logger(
            save_root=save_root,
            name=wandb_name or name,
            offline=wandb_offline,
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags,
            notes=wandb_notes,
            config=wandb_config,
        )
        if wandb_logger is not None:
            loggers.append(wandb_logger)

    return loggers


def make_wandb_logger(
    *,
    save_root: Union[str, Path],
    name: str,
    offline: bool = False,
    project: str = "stelar",
    entity: Optional[str] = None,
    tags: Union[str, Sequence[str], None] = None,
    notes: Optional[str] = None,
    config: Optional[dict] = None,
):
    """Build a single WandbLogger (or None on failure).

    Offline by default when no API key is set, so a run never blocks on login.
    """
    if offline or not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
    try:
        wandb_logger = pl_loggers.WandbLogger(
            name=name,
            save_dir=str(save_root),
            project=project,
            entity=entity,
            tags=_coerce_tags(tags),
            notes=notes,
        )
        if config:
            wandb_logger.experiment.config.update(config, allow_val_change=True)
        return wandb_logger
    except Exception as exc:  # pragma: no cover - network / import issues
        print(f"[wandb] disabled ({type(exc).__name__}: {exc}); continuing without W&B.")
        return None


def primary_log_dir(loggers, fallback: Union[str, Path]) -> Path:
    """Pick a concrete directory (for checkpoints) from the built loggers."""
    for lg in loggers:
        log_dir = getattr(lg, "log_dir", None)
        if log_dir:
            return Path(log_dir)
    return Path(fallback)
