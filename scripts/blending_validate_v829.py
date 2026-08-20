"""Explicit deterministic Phase-A validation entry point for Blending V8.29."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    os.environ.setdefault("BLENDING_V8_STRONG_ANCHOR_ALPHA", "0.90")
    os.environ.setdefault("BLENDING_V229_OUTER_STRAND_RECOVERY", "0")
    from scripts import blending_train_v8

    print("[V2.29] explicit validator; Phase A deterministic diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
