"""Explicit V2.31 high-resolution ViTMatte validation entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")

from scripts import blending_train_v8


def main() -> None:
    print("[V2.31] explicit validator; phased matting/recolor diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
