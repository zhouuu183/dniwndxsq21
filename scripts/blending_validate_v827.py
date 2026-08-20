"""Explicit command-line entry point for V2.27 diagnostic-only validation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate the frozen V2.26 projector with V2.27 gates")
    parser.add_argument(
        "--checkpoint",
        default="output/blending_train_v8_direct_anchor_v2_26_boundary_target_metric_aligned_small/checkpoints/v226_boundary_target_metric_aligned_pass.pth",
    )
    args = parser.parse_args()
    os.environ["BLENDING_V227_CHECKPOINT"] = args.checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.blending_train_v8 import main as run_diagnostic

    print("[V2.27] explicit validator entry; training=False diagnostic_only=True")
    run_diagnostic()


if __name__ == "__main__":
    main()

