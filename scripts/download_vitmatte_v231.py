"""Explicit, user-invoked ViTMatte-S checkpoint download."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pretrained_models/ViTMatte/vitmatte-small-composition-1k"),
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub manually before running this script") from exc
    snapshot_download(
        repo_id="hustvl/vitmatte-small-composition-1k",
        local_dir=str(args.output),
    )
    print(f"ViTMatte-S downloaded to {args.output}")


if __name__ == "__main__":
    main()
