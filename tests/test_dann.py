from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (str(SRC), str(SRC / "downstream")):
    if p not in sys.path:
        sys.path.insert(0, p)

pytest.importorskip("torch")
from downstream.datasets.bcic2a_dataset import (  # noqa: E402
    LMDBDataset,
    LoadDataset,
    _key_to_subject_token,
)


# ----------------------- dataset subject-id plumbing -----------------------

def test_collate_two_tuple():
    batch = [(np.zeros((3, 4), np.float32), 1), (np.ones((3, 4), np.float32), 0)]
    out = LMDBDataset.collate(batch)
    assert len(out) == 2
    assert tuple(out[0].shape) == (2, 3, 4)
    assert out[1].tolist() == [1, 0]


def test_collate_three_tuple_dann():
    batch = [(np.zeros((3, 4), np.float32), 1, 5), (np.ones((3, 4), np.float32), 0, 2)]
    out = LMDBDataset.collate(batch)
    assert len(out) == 3
    assert out[2].tolist() == [5, 2]          # subject ids preserved


def test_subject_id_mapping_contiguous():
    class _P:
        datasets_dir = "/tmp/none"
        batch_size = 4
    ld = LoadDataset(_P())
    keys = ["A01T-0-1", "A03T-2-0", "A01T-1-3", "A02T-0-0"]
    ids = ld._subject_ids_for_keys(keys)
    # tokens sorted -> A01:0, A02:1, A03:2
    assert ids == [0, 2, 0, 1]
    assert ld.num_train_subjects == 3


def test_key_to_subject_token():
    assert _key_to_subject_token("A09T-12-7") == "A09"


# ------------------------- DANN module (needs timm) ------------------------

def test_grl_and_registry():
    pytest.importorskip("timm")
    import torch
    from downstream.modules import build_module
    from downstream.modules.stelar_dann_ft import GradientReversalLayer, LitSTELARDANN
    from downstream.modules.stelar_patch_adapt_ft import LitSTELARPatchAdapt

    # registry resolves the new scheme to the DANN module
    assert build_module("stelar_dann") is LitSTELARDANN
    assert issubclass(LitSTELARDANN, LitSTELARPatchAdapt)

    # GRL: identity forward, gradient negated and scaled by alpha on backward
    grl = GradientReversalLayer(alpha=2.0)
    x = torch.ones(3, requires_grad=True)
    y = grl(x)
    assert torch.allclose(y, x)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.full((3,), -2.0))
