"""Explicit V2.34 deterministic hair-carrier validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V234_DIAGNOSTIC_ONLY", "1")

from scripts import blending_train_v8


def main() -> None:
    print("[V2.34] explicit validator; hair-carrier chroma diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
