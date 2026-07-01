from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = ROOT / "src" / "pretrain"

torch = pytest.importorskip("torch")
pytest.importorskip("pytorch_lightning")


def _make_synth(dirpath: Path, n: int = 16):
    dirpath.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    for i in range(n):
        torch.save(torch.randn(19, 30, 200) * 20.0, dirpath / f"s{i:03d}.pt")
    return dirpath


@pytest.mark.parametrize("preset", ["v0_baseline", "v2_aug"])
def test_pretrain_one_epoch_cpu(tmp_path, preset):
    data_dir = _make_synth(tmp_path / "data")
    log_dir = tmp_path / "logs"
    cmd = [
        sys.executable, "run_pretrain_improved.py",
        "--preset", preset,
        "--input_mode", "pretoken", "--data_backend", "files",
        "--train_root", str(data_dir), "--exts", "pt",
        "--val_fraction", "0.2", "--montage", "19",
        "--devices", "", "--num_workers", "0", "--batch_size", "8",
        "--tag", "tiny", "--transformer", "crisscross", "--mask_ratio", "0.5",
        "--max_epochs", "1", "--limit_train_batches", "2", "--limit_val_batches", "1",
        "--logger", "csv", "--log_root", str(log_dir),
    ]
    proc = subprocess.run(cmd, cwd=PRETRAIN, capture_output=True, text=True)
    assert proc.returncode == 0, f"runner failed:\nSTDOUT\n{proc.stdout}\nSTDERR\n{proc.stderr}"
    assert list(log_dir.rglob("*.ckpt")), "no checkpoint written"
