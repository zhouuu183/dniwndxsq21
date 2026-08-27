"""Generate 30 fixed-pair results with the current V6 pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hair_swap_v6 import HairFastV6, get_parser  # noqa: E402
from fixed_pair_compare_common import (  # noqa: E402
    PAIR_COUNT,
    create_pairs,
    generate_results,
    make_quads,
    require_file,
    save_inputs,
)


# ============================== User config ===============================
SOURCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/ear")
SHAPE_REFERENCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
COLOR_REFERENCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")

COMMON_DIR = Path("output/compare_fixed_pairs_30")
OUTPUT_DIR = COMMON_DIR / "v6"
PAIR_MANIFEST = COMMON_DIR / "fixed_pairs.json"
REBUILD_FIXED_PAIRS = False

DEVICE = "cuda"
STYLEGAN_CHECKPOINT = "pretrained_models/StyleGAN/ffhq.pt"
ROTATE_CHECKPOINT = "pretrained_models/Rotate/rotate_best.pth"
AUTHOR_BLENDING_CHECKPOINT = "pretrained_models/Blending/checkpoint.pth"
AUTHOR_PP_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"
V6_PP_CHECKPOINT = "output/pp_v5_checkpoints_ear_short_long_hair_locked_v5/last.pth"
SATD_CHECKPOINT = "/data/coding/hairfast_ppmodify/checkpoints/satd_3000_best.pth"
USE_SATD = True
SATD_BLEND = 0.75
# ============================================================================


def build_v6() -> HairFastV6:
    for path, label in (
        (STYLEGAN_CHECKPOINT, "StyleGAN checkpoint"),
        (ROTATE_CHECKPOINT, "rotate checkpoint"),
        (AUTHOR_BLENDING_CHECKPOINT, "author blending checkpoint"),
        (AUTHOR_PP_CHECKPOINT, "author PP checkpoint"),
        (V6_PP_CHECKPOINT, "V6 PP checkpoint"),
    ):
        require_file(path, label)
    if USE_SATD:
        require_file(SATD_CHECKPOINT, "SATD checkpoint")

    args = get_parser().parse_args([])
    args.device = DEVICE
    args.ckpt = STYLEGAN_CHECKPOINT
    args.rotate_checkpoint = ROTATE_CHECKPOINT
    args.blending_checkpoint = AUTHOR_BLENDING_CHECKPOINT
    # V6 uses the author PP checkpoint to construct the author-compatible
    # transfer, then loads V6_PP_CHECKPOINT into the V6 PP decoder.
    args.pp_checkpoint = AUTHOR_PP_CHECKPOINT
    args.pp_v6_checkpoint = V6_PP_CHECKPOINT
    args.use_satd_v8 = USE_SATD
    args.satd_checkpoint_v8 = SATD_CHECKPOINT if USE_SATD else ""
    args.satd_blend_v8 = SATD_BLEND
    args.allow_legacy_blending_checkpoint_v8 = True
    args.save_all = False
    print("V6 modules: hair_swap_v6.py + models/*_v6.py")
    print(f"V6 author weights: {STYLEGAN_CHECKPOINT}, {ROTATE_CHECKPOINT},")
    print(f"                  {AUTHOR_BLENDING_CHECKPOINT}, {AUTHOR_PP_CHECKPOINT}")
    print(f"V6 PP weight: {V6_PP_CHECKPOINT}")
    print(f"SATD: {'enabled' if USE_SATD else 'disabled'}" + (f" ({SATD_CHECKPOINT})" if USE_SATD else ""))
    return HairFastV6(args)


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
        build_v6,
        result_dir,
        SOURCE_DIR,
        SHAPE_REFERENCE_DIR,
        COLOR_REFERENCE_DIR,
        "v6",
    )
    make_quads(pairs, input_dir, result_dir, quad_dir, "v6")
    print(f"Generated {len(pairs)} V6 result images: {result_dir}")
    print(f"Generated {len(pairs)} V6 four-column images: {quad_dir}")
    print(f"Fixed pairing manifest: {PAIR_MANIFEST}")


if __name__ == "__main__":
    main()
