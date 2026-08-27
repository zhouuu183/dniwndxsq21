"""Fixed-pair baseline/V6 PP comparison.

This script intentionally runs the real inference entry points:

* ``hair_swap.HairFast`` for the no-suffix author baseline;
* ``hair_swap_v6.HairFastV6`` for the current V6 pipeline.

The first run writes ``fixed_pairs.json``.  Later runs reuse that exact file,
so changing a PP checkpoint never changes the source, shape-reference, or
colour-reference pairing.  Do not delete that file when comparing weights.
"""

from __future__ import annotations

import gc
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as tvf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from hair_swap import HairFast, get_parser as get_baseline_parser
from hair_swap_v6 import HairFastV6, get_parser as get_v6_parser
from utils.image_utils import list_image_files


# ========================= User Config: edit here only =========================
# Source faces must come from ``images/ear``.  Both reference images come from
# ``images/mix_ear``; shape and colour are deliberately selected independently.
USER_SOURCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/ear")
USER_SHAPE_REFERENCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/mix_ear")
USER_COLOR_REFERENCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/mix_ear")

# Keep this directory across checkpoint comparisons.  ``fixed_pairs.json`` is
# the pairing contract.  Set REBUILD_FIXED_PAIRS=True only when deliberately
# starting a new comparison set.
USER_OUTPUT_DIR = Path("output/pp_compare_fixed_pairs_baseline_v6")
USER_PAIR_COUNT = 10
USER_PAIR_SEED = 3407
USER_MODEL_SEED = 91011
USER_REBUILD_FIXED_PAIRS = False
USER_DEVICE = "cuda"
USER_PREVIEW_SIDE = 512

USER_STYLEGAN_CHECKPOINT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CHECKPOINT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CHECKPOINT = "pretrained_models/Blending/checkpoint.pth"
USER_AUTHOR_PP_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"

# Add one entry for every PP checkpoint that should be compared.  ``kind`` is:
#
# * ``baseline``: exact no-suffix author pipeline. ``pp_checkpoint`` is the
#   author PP checkpoint to compare.
# * ``v6``: current V6 pipeline. ``v6_pp_checkpoint`` is the V6 training
#   checkpoint. ``author_pp_checkpoint`` remains the author checkpoint needed
#   internally to build V6's author pre-PP transfer path.
#
# A V6 entry can enable the current SATD post-decode background cleanup.  Keep
# the same setting for all V6 weights in a comparison.
USER_VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "baseline_author",
        "kind": "baseline",
        # The original author's PP checkpoint.
        "pp_checkpoint": "pretrained_models/PostProcess/pp_model.pth",
    },
    {
        "name": "v6_ear_short_long_hair_locked",
        "kind": "v6",
        # V6 still needs the author's PP checkpoint to construct the author
        # pre-PP hair-transfer image.  This is separate from the trained V6
        # checkpoint used by the V6 PP decoder below.
        "author_pp_checkpoint": USER_AUTHOR_PP_CHECKPOINT,
        "v6_pp_checkpoint": "output/pp_v5_checkpoints_ear_short_long_hair_locked_v5/last.pth",
        "use_satd_v8": True,
        "satd_checkpoint_v8": "/data/coding/hairfast_ppmodify/checkpoints/satd_3000_best.pth",
        "satd_blend_v8": 0.28,
        # The current blending checkpoint is legacy, as in pp_gen_v6.py.
        "allow_legacy_blending_checkpoint_v8": True,
    },
)
# ==============================================================================

PAIR_MANIFEST_NAME = "fixed_pairs.json"
RUN_CONFIG_NAME = "run_config.json"
VALID_KINDS = {"baseline", "v6"}


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def require_file(path: str | Path, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def repeated_permutation(files: list[str], count: int, rng: random.Random) -> list[str]:
    """Sample deterministically, without replacement inside each full pass."""
    values: list[str] = []
    while len(values) < count:
        shuffled = list(files)
        rng.shuffle(shuffled)
        values.extend(shuffled)
    return values[:count]


def pick_different_reference(
    candidates: list[str],
    forbidden_name: str,
    rng: random.Random,
) -> str:
    available = [name for name in candidates if name != forbidden_name]
    if not available:
        raise RuntimeError(
            "Need at least two distinct files in the reference directory to "
            "choose separate hairstyle and colour references."
        )
    return rng.choice(available)


def make_fixed_pairs() -> list[dict[str, str | int]]:
    require_directory(USER_SOURCE_DIR, "USER_SOURCE_DIR")
    require_directory(USER_SHAPE_REFERENCE_DIR, "USER_SHAPE_REFERENCE_DIR")
    require_directory(USER_COLOR_REFERENCE_DIR, "USER_COLOR_REFERENCE_DIR")

    source_files = list_image_files(USER_SOURCE_DIR)
    shape_files = list_image_files(USER_SHAPE_REFERENCE_DIR)
    color_files = list_image_files(USER_COLOR_REFERENCE_DIR)
    if not source_files:
        raise RuntimeError(f"No RGB image files found in source directory: {USER_SOURCE_DIR}")
    if not shape_files:
        raise RuntimeError(f"No RGB image files found in shape directory: {USER_SHAPE_REFERENCE_DIR}")
    if not color_files:
        raise RuntimeError(f"No RGB image files found in colour directory: {USER_COLOR_REFERENCE_DIR}")
    if USER_PAIR_COUNT <= 0:
        raise ValueError("USER_PAIR_COUNT must be positive")

    rng = random.Random(USER_PAIR_SEED)
    sources = repeated_permutation(source_files, USER_PAIR_COUNT, rng)
    shapes = repeated_permutation(shape_files, USER_PAIR_COUNT, rng)
    pairs: list[dict[str, str | int]] = []
    for index, (source, shape) in enumerate(zip(sources, shapes)):
        # When both reference directories are the same, this guarantees the
        # colour reference is a different photo from the shape reference.
        color = pick_different_reference(color_files, shape, rng)
        pairs.append(
            {
                "index": index,
                "source": source,
                "shape_reference": shape,
                "color_reference": color,
                "model_seed": USER_MODEL_SEED + index,
            }
        )
    return pairs


def load_or_create_pairs(manifest_path: Path) -> list[dict[str, str | int]]:
    if manifest_path.exists() and not USER_REBUILD_FIXED_PAIRS:
        with manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        pairs = payload.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise RuntimeError(f"Invalid fixed-pair manifest: {manifest_path}")
        if len(pairs) != USER_PAIR_COUNT:
            raise RuntimeError(
                f"{manifest_path} contains {len(pairs)} fixed pairs, but "
                f"USER_PAIR_COUNT is {USER_PAIR_COUNT}. Keep the existing "
                "count for a fair weight comparison, or explicitly set "
                "USER_REBUILD_FIXED_PAIRS=True to start a new set."
            )
        return pairs

    pairs = make_fixed_pairs()
    payload = {
        "pairing_seed": USER_PAIR_SEED,
        "source_dir": str(USER_SOURCE_DIR),
        "shape_reference_dir": str(USER_SHAPE_REFERENCE_DIR),
        "color_reference_dir": str(USER_COLOR_REFERENCE_DIR),
        "pairs": pairs,
    }
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return pairs


def validate_pairs(pairs: list[dict[str, str | int]]) -> None:
    for pair in pairs:
        index = pair.get("index")
        for directory, key in (
            (USER_SOURCE_DIR, "source"),
            (USER_SHAPE_REFERENCE_DIR, "shape_reference"),
            (USER_COLOR_REFERENCE_DIR, "color_reference"),
        ):
            filename = pair.get(key)
            if not isinstance(filename, str) or not (directory / filename).is_file():
                raise FileNotFoundError(
                    f"Fixed pair {index} references a missing {key} image: "
                    f"{directory / str(filename)}"
                )


def apply_common_args(args: Any) -> None:
    args.device = USER_DEVICE
    args.ckpt = USER_STYLEGAN_CHECKPOINT
    args.rotate_checkpoint = USER_ROTATE_CHECKPOINT
    args.blending_checkpoint = USER_BLENDING_CHECKPOINT
    args.save_all = False


def build_model(variant: dict[str, Any]) -> torch.nn.Module:
    name = variant.get("name")
    kind = variant.get("kind")
    if not isinstance(name, str) or not name:
        raise ValueError("Every USER_VARIANTS entry needs a non-empty 'name'.")
    if kind not in VALID_KINDS:
        raise ValueError(f"Variant {name!r} has unsupported kind={kind!r}.")

    if kind == "baseline":
        pp_checkpoint = variant.get("pp_checkpoint")
        if not isinstance(pp_checkpoint, str):
            raise ValueError(f"Baseline variant {name!r} needs 'pp_checkpoint'.")
        require_file(pp_checkpoint, f"Baseline PP checkpoint for {name!r}")
        args = get_baseline_parser().parse_args([])
        apply_common_args(args)
        args.pp_checkpoint = pp_checkpoint
        return HairFast(args)

    author_pp_checkpoint = variant.get("author_pp_checkpoint")
    v6_pp_checkpoint = variant.get("v6_pp_checkpoint")
    if not isinstance(author_pp_checkpoint, str):
        raise ValueError(f"V6 variant {name!r} needs 'author_pp_checkpoint'.")
    if not isinstance(v6_pp_checkpoint, str):
        raise ValueError(f"V6 variant {name!r} needs 'v6_pp_checkpoint'.")
    require_file(author_pp_checkpoint, f"Author PP checkpoint for V6 variant {name!r}")
    require_file(v6_pp_checkpoint, f"V6 PP checkpoint for {name!r}")

    args = get_v6_parser().parse_args([])
    apply_common_args(args)
    args.pp_checkpoint = author_pp_checkpoint
    args.pp_v6_checkpoint = v6_pp_checkpoint
    args.use_satd_v8 = bool(variant.get("use_satd_v8", False))
    args.satd_checkpoint_v8 = str(variant.get("satd_checkpoint_v8", ""))
    args.satd_blend_v8 = float(variant.get("satd_blend_v8", args.satd_blend_v8))
    args.allow_legacy_blending_checkpoint_v8 = bool(
        variant.get(
            "allow_legacy_blending_checkpoint_v8",
            args.allow_legacy_blending_checkpoint_v8,
        )
    )
    if args.use_satd_v8:
        require_file(args.satd_checkpoint_v8, f"SATD checkpoint for V6 variant {name!r}")
    return HairFastV6(args)


def tensor_to_rgb_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        if tensor.size(0) != 1:
            raise ValueError(f"Expected one output image, got batch {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.size(0) != 3:
        raise ValueError(f"Expected RGB output [3,H,W], got {tuple(tensor.shape)}")
    return tvf.to_pil_image(tensor.detach().cpu().float().clamp(0, 1)).convert("RGB")


def save_input_copies(pairs: list[dict[str, str | int]], input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        index = int(pair["index"])
        for prefix, directory, key in (
            ("source", USER_SOURCE_DIR, "source"),
            ("shape", USER_SHAPE_REFERENCE_DIR, "shape_reference"),
            ("color", USER_COLOR_REFERENCE_DIR, "color_reference"),
        ):
            destination = input_dir / f"pair_{index:03d}_{prefix}.png"
            if not destination.exists():
                with Image.open(directory / str(pair[key])) as image:
                    image.convert("RGB").save(destination)


def run_variant(variant: dict[str, Any], pairs: list[dict[str, str | int]], results_dir: Path) -> None:
    name = str(variant["name"])
    output_dir = results_dir / name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {name} ({variant['kind']})")
    model = build_model(variant)
    try:
        for pair in pairs:
            index = int(pair["index"])
            result_path = output_dir / f"pair_{index:03d}.png"
            source = USER_SOURCE_DIR / str(pair["source"])
            shape = USER_SHAPE_REFERENCE_DIR / str(pair["shape_reference"])
            color = USER_COLOR_REFERENCE_DIR / str(pair["color_reference"])
            print(f"[{name}] pair_{index:03d}: {source.name} | {shape.name} | {color.name}")
            result = model(source, shape, color, seed=int(pair["model_seed"]))
            tensor_to_rgb_image(result).save(result_path)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resized_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize(
            (USER_PREVIEW_SIDE, USER_PREVIEW_SIDE),
            Image.Resampling.LANCZOS,
        )


def make_comparisons(
    pairs: list[dict[str, str | int]],
    variants: tuple[dict[str, Any], ...],
    input_dir: Path,
    results_dir: Path,
    comparison_dir: Path,
) -> None:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    labels = ["source", "shape reference", "colour reference"] + [str(item["name"]) for item in variants]
    with (comparison_dir / "columns.txt").open("w", encoding="utf-8") as file:
        file.write(" | ".join(labels) + "\n")

    label_height = 26
    for pair in pairs:
        index = int(pair["index"])
        paths = [
            input_dir / f"pair_{index:03d}_source.png",
            input_dir / f"pair_{index:03d}_shape.png",
            input_dir / f"pair_{index:03d}_color.png",
        ] + [results_dir / str(item["name"]) / f"pair_{index:03d}.png" for item in variants]
        images = [resized_rgb(path) for path in paths]
        sheet = Image.new("RGB", (USER_PREVIEW_SIDE * len(images), USER_PREVIEW_SIDE + label_height), "white")
        draw = ImageDraw.Draw(sheet)
        for column, (label, image) in enumerate(zip(labels, images)):
            x = column * USER_PREVIEW_SIDE
            sheet.paste(image, (x, label_height))
            draw.text((x + 4, 5), label, fill="black")
        sheet.save(comparison_dir / f"pair_{index:03d}.png")


def write_run_config(run_path: Path, pairs: list[dict[str, str | int]]) -> None:
    payload = {
        "source_dir": str(USER_SOURCE_DIR),
        "shape_reference_dir": str(USER_SHAPE_REFERENCE_DIR),
        "color_reference_dir": str(USER_COLOR_REFERENCE_DIR),
        "pair_manifest": PAIR_MANIFEST_NAME,
        "pair_count": len(pairs),
        "stylegan_checkpoint": USER_STYLEGAN_CHECKPOINT,
        "rotate_checkpoint": USER_ROTATE_CHECKPOINT,
        "blending_checkpoint": USER_BLENDING_CHECKPOINT,
        "variants": list(USER_VARIANTS),
    }
    with run_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    if USER_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("USER_DEVICE is CUDA but PyTorch cannot see a CUDA device.")
    names = [item.get("name") for item in USER_VARIANTS]
    if len(USER_VARIANTS) < 2:
        raise ValueError("Configure at least two variants for a comparison.")
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Each USER_VARIANTS entry needs a unique non-empty name.")

    USER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = load_or_create_pairs(USER_OUTPUT_DIR / PAIR_MANIFEST_NAME)
    validate_pairs(pairs)

    input_dir = USER_OUTPUT_DIR / "inputs"
    results_dir = USER_OUTPUT_DIR / "results"
    comparison_dir = USER_OUTPUT_DIR / "comparisons"
    save_input_copies(pairs, input_dir)
    write_run_config(USER_OUTPUT_DIR / RUN_CONFIG_NAME, pairs)

    for variant in USER_VARIANTS:
        run_variant(variant, pairs, results_dir)
    make_comparisons(pairs, USER_VARIANTS, input_dir, results_dir, comparison_dir)
    print(f"Fixed-pair comparison complete: {USER_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
