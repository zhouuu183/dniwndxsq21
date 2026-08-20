"""Explicit V2.35 hair-only chroma disentanglement validator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("BLENDING_V8_DATASET_PROFILE", "small")
os.environ.setdefault("BLENDING_V235_DIAGNOSTIC_ONLY", "1")
# V2.35 uses a versioned color-FS namespace. Build it on the first validator
# run instead of silently falling back to legacy full-image color latents.
os.environ.setdefault("BLENDING_BUILD_CACHE_WITH_CURRENT_SATD", "1")

from scripts import blending_train_v8


def main() -> None:
    print("[V2.35] explicit validator; hair-only chroma diagnostic only; no training.")
    blending_train_v8.main()


if __name__ == "__main__":
    main()
