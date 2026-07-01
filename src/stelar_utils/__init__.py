"""Shared, dependency-light helpers for the STELAR-improved entry points.

These utilities are intentionally additive: they wrap the existing argparse
runners with an optional YAML config layer and a W&B-capable logger builder,
**without** touching any model / training / loss logic. The runners keep working
exactly as before when no YAML / W&B flag is passed.
"""

from .loggers import add_logging_args, build_loggers
from .yaml_config import apply_yaml_defaults, load_yaml_flat

__all__ = [
    "add_logging_args",
    "build_loggers",
    "apply_yaml_defaults",
    "load_yaml_flat",
]
