"""Explicit deterministic V2.33 confidence/ownership validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V233_DIAGNOSTIC_ONLY", "1")

from scripts import blending_train_v8


def main() -> None:
    print("[V2.33] explicit validator; deterministic validation only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
