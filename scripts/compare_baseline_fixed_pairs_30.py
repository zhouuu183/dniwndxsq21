"""Generate 30 fixed-pair results with the untouched author baseline.

This file intentionally imports only ``hair_swap.HairFast`` and the no-suffix
author modules it imports. It does not import V5/V6, SATD, or earring code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# The baseline must be imported from the separate official checkout.  Do not
# silently fall back to ppmodify: that could load a modified no-suffix helper.
OFFICIAL_BASELINE_ROOT = Path("/data/coding/HairFastGAN/HairFastGAN-main")
if not (OFFICIAL_BASELINE_ROOT / "hair_swap.py").is_file():
    raise FileNotFoundError(
        "Official baseline checkout is missing: "
        f"{OFFICIAL_BASELINE_ROOT / 'hair_swap.py'}. "
        "Set OFFICIAL_BASELINE_ROOT to the extracted author repository."
    )
sys.path.insert(0, str(OFFICIAL_BASELINE_ROOT))
sys.path.insert(1, str(ROOT))

from hair_swap import HairFast, get_parser  # noqa: E402
from fixed_pair_compare_common import (  # noqa: E402
    PAIR_COUNT,
    create_pairs,
    generate_results,
    make_quads,
    require_file,
    save_inputs,
)


# ============================== User config ===============================
SOURCE_DIR = OFFICIAL_BASELINE_ROOT / "images/ear"
SHAPE_REFERENCE_DIR = OFFICIAL_BASELINE_ROOT / "images/FFHQ_short"
COLOR_REFERENCE_DIR = OFFICIAL_BASELINE_ROOT / "images/FFHQ_short"

COMMON_DIR = Path("output/compare_fixed_pairs_30")
OUTPUT_DIR = COMMON_DIR / "baseline"
PAIR_MANIFEST = COMMON_DIR / "fixed_pairs.json"
REBUILD_FIXED_PAIRS = False

DEVICE = "cuda"
STYLEGAN_CHECKPOINT = "pretrained_models/StyleGAN/ffhq.pt"
ROTATE_CHECKPOINT = "pretrained_models/Rotate/rotate_best.pth"
BLENDING_CHECKPOINT = "pretrained_models/Blending/checkpoint.pth"
AUTHOR_PP_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"
# ============================================================================


def build_baseline() -> HairFast:
    official_files = (
        "hair_swap.py",
        "models/Alignment.py",
        "models/Blending.py",
        "models/Embedding.py",
        "models/Encoders.py",
        "models/Net.py",
        "models/stylegan2/op/fused_act.py",
        "models/stylegan2/op/upfirdn2d.py",
    )
    for relative_path in official_files:
        if not (OFFICIAL_BASELINE_ROOT / relative_path).is_file():
            raise FileNotFoundError(
                f"Official baseline file is missing: {OFFICIAL_BASELINE_ROOT / relative_path}"
            )
    for path, label in (
        (STYLEGAN_CHECKPOINT, "StyleGAN checkpoint"),
        (ROTATE_CHECKPOINT, "rotate checkpoint"),
        (BLENDING_CHECKPOINT, "author blending checkpoint"),
        (AUTHOR_PP_CHECKPOINT, "author PP checkpoint"),
    ):
        require_file(path, label)
    args = get_parser().parse_args([])
    args.device = DEVICE
    args.ckpt = STYLEGAN_CHECKPOINT
    args.rotate_checkpoint = ROTATE_CHECKPOINT
    args.blending_checkpoint = BLENDING_CHECKPOINT
    args.pp_checkpoint = AUTHOR_PP_CHECKPOINT
    args.save_all = False
    print(f"Baseline source root: {OFFICIAL_BASELINE_ROOT}")
    print("Baseline modules: official hair_swap.py + official models/* (no suffix)")
    print(f"Baseline weights: {STYLEGAN_CHECKPOINT}, {ROTATE_CHECKPOINT},")
    print(f"                 {BLENDING_CHECKPOINT}, {AUTHOR_PP_CHECKPOINT}")
    return HairFast(args)


def main() -> None:
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE is CUDA but PyTorch cannot see a CUDA device.")
    if PAIR_COUNT != 30:
        raise RuntimeError("This comparison script must generate exactly 30 pairs.")
    COMMON_DIR.mkdir(parents=True, exist_ok=True)
    pairs = create_pairs(
        SOURCE_DIR,
        SHAPE_REFERENCE_DIR,
        COLOR_REFERENCE_DIR,
        PAIR_MANIFEST,
        rebuild=REBUILD_FIXED_PAIRS,
    )
    input_dir = COMMON_DIR / "inputs"
    result_dir = OUTPUT_DIR / "results"
    quad_dir = OUTPUT_DIR / "quads"
    save_inputs(pairs, input_dir, SOURCE_DIR, SHAPE_REFERENCE_DIR, COLOR_REFERENCE_DIR)
    generate_results(
        pairs,
        build_baseline,
        result_dir,
        SOURCE_DIR,
        SHAPE_REFERENCE_DIR,
        COLOR_REFERENCE_DIR,
        "baseline",
    )
    make_quads(pairs, input_dir, result_dir, quad_dir, "baseline")
    print(f"Generated {len(pairs)} baseline result images: {result_dir}")
    print(f"Generated {len(pairs)} baseline four-column images: {quad_dir}")
    print(f"Fixed pairing manifest: {PAIR_MANIFEST}")


if __name__ == "__main__":
    main()
