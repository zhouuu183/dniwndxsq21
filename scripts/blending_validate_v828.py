"""Explicit deterministic validation entry point for Blending V8.28."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main() -> None:
    os.environ.setdefault("BLENDING_V8_STRONG_ANCHOR_ALPHA", "0.90")
    from scripts import blending_train_v8

    print("[V2.28] explicit validator; deterministic diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
