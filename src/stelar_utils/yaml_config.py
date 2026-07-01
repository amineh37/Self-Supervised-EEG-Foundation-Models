"""Optional YAML config layer for the argparse-based runners.

A YAML file is read and *flattened* into a flat mapping of ``arg_name -> value``
that is applied to the argparse parser via ``set_defaults``. Because explicit
command-line flags always beat parser defaults, the precedence is:

    CLI flag  >  YAML value  >  built-in argparse default

This keeps every documented command working verbatim while letting a clean YAML
drive a run. Sections (``data:``, ``model:``, ``trainer:``, ``logging:``,
``improvements:``) are purely cosmetic: their leaf keys are merged into one flat
namespace, so each key must match an argparse destination name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_flat(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config and flatten one level of nested sections."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(raw).__name__}")

    flat: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def apply_yaml_defaults(parser, path: str | Path, *, warn_unknown: bool = True):
    """Apply a YAML config as parser defaults (CLI still overrides).

    Unknown keys (not matching any argparse dest) are dropped with a warning so a
    stray field never silently changes behaviour.
    """
    flat = load_yaml_flat(path)
    known = {a.dest for a in parser._actions}
    accepted, unknown = {}, []
    for key, value in flat.items():
        if key in known:
            accepted[key] = value
        else:
            unknown.append(key)
    if unknown and warn_unknown:
        print(f"[config] ignored unknown keys from {path}: {sorted(unknown)}")
    parser.set_defaults(**accepted)
    return accepted
