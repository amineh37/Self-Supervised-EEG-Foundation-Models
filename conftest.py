"""Make the src/ package dirs importable for the test-suite.

Project root is `architecture/`; the code lives under `architecture/src/`.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (SRC, SRC / "pretrain"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
