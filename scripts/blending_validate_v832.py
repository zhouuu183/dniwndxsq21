"""Explicit deterministic V2.32 foreground/recomposition validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V232_DIAGNOSTIC_ONLY", "1")

from scripts import blending_train_v8


def main() -> None:
    print("[V2.32] explicit validator; foreground-space phased diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
