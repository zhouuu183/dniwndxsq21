"""Shared fixed-pair generation helpers for the baseline and V6 runners."""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw
from torchvision.transforms import functional as tvf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PAIR_COUNT = 30
PAIR_SEED = 3407
MODEL_SEED = 91011
PREVIEW_SIDE = 512


def list_images(directory: Path) -> list[str]:
    return sorted(
        path.name
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def require_file(path: str | Path, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _cycled_sample(files: list[str], count: int, rng: random.Random) -> list[str]:
    result: list[str] = []
    while len(result) < count:
        block = list(files)
        rng.shuffle(block)
        result.extend(block)
    return result[:count]


def create_pairs(
    source_dir: Path,
    shape_dir: Path,
    color_dir: Path,
    manifest_path: Path,
    rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Create or load the one pairing manifest shared by both experiments."""
    for directory, label in (
        (source_dir, "source directory"),
        (shape_dir, "shape-reference directory"),
        (color_dir, "colour-reference directory"),
    ):
        require_directory(directory, label)

    if manifest_path.exists() and not rebuild:
        with manifest_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        pairs = payload.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != PAIR_COUNT:
            raise RuntimeError(
                f"{manifest_path} must contain exactly {PAIR_COUNT} pairs. "
                "Set REBUILD_FIXED_PAIRS=True only to deliberately start a new set."
            )
        validate_pairs(pairs, source_dir, shape_dir, color_dir)
        return pairs

    source_files = list_images(source_dir)
    shape_files = list_images(shape_dir)
    color_files = list_images(color_dir)
    if not source_files or not shape_files or not color_files:
        raise RuntimeError("Each configured image directory must contain at least one image.")
    if len(color_files) < 2:
        raise RuntimeError("The colour-reference directory needs at least two images.")

    rng = random.Random(PAIR_SEED)
    sources = _cycled_sample(source_files, PAIR_COUNT, rng)
    shapes = _cycled_sample(shape_files, PAIR_COUNT, rng)
    pairs: list[dict[str, Any]] = []
    for index, (source, shape) in enumerate(zip(sources, shapes)):
        choices = [name for name in color_files if name != shape]
        color = rng.choice(choices or color_files)
        pairs.append(
            {
                "index": index,
                "source": source,
                "shape_reference": shape,
                "color_reference": color,
                "model_seed": MODEL_SEED + index,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "pair_count": PAIR_COUNT,
                "pair_seed": PAIR_SEED,
                "source_dir": str(source_dir),
                "shape_reference_dir": str(shape_dir),
                "color_reference_dir": str(color_dir),
                "pairs": pairs,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")
    return pairs


def validate_pairs(
    pairs: list[dict[str, Any]],
    source_dir: Path,
    shape_dir: Path,
    color_dir: Path,
) -> None:
    for expected_index, pair in enumerate(pairs):
        if int(pair.get("index", -1)) != expected_index:
            raise RuntimeError(f"Fixed-pair manifest index {expected_index} is malformed.")
        for directory, key in (
            (source_dir, "source"),
            (shape_dir, "shape_reference"),
            (color_dir, "color_reference"),
        ):
            name = pair.get(key)
            if not isinstance(name, str) or not (directory / name).is_file():
                raise FileNotFoundError(
                    f"Fixed pair {expected_index} references missing {key}: {directory / str(name)}"
                )


def tensor_to_image(value: torch.Tensor) -> Image.Image:
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected one output image, got {tuple(value.shape)}")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(value.shape)}")
    return tvf.to_pil_image(value.detach().cpu().float().clamp(0, 1)).convert("RGB")


def save_inputs(
    pairs: list[dict[str, Any]],
    input_dir: Path,
    source_dir: Path,
    shape_dir: Path,
    color_dir: Path,
) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        index = int(pair["index"])
        for prefix, directory, key in (
            ("source", source_dir, "source"),
            ("shape", shape_dir, "shape_reference"),
            ("color", color_dir, "color_reference"),
        ):
            destination = input_dir / f"pair_{index:03d}_{prefix}.png"
            if not destination.exists():
                with Image.open(directory / str(pair[key])) as image:
                    image.convert("RGB").save(destination)


def generate_results(
    pairs: list[dict[str, Any]],
    model_factory: Callable[[], Any],
    output_dir: Path,
    source_dir: Path,
    shape_dir: Path,
    color_dir: Path,
    label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model_factory()
    try:
        for pair in pairs:
            index = int(pair["index"])
            source = source_dir / str(pair["source"])
            shape = shape_dir / str(pair["shape_reference"])
            color = color_dir / str(pair["color_reference"])
            print(f"[{label}] pair_{index:03d}: {source.name} | {shape.name} | {color.name}")
            with torch.inference_mode():
                result = model(source, shape, color, seed=int(pair["model_seed"]))
            tensor_to_image(result).save(output_dir / f"pair_{index:03d}.png")
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_quads(
    pairs: list[dict[str, Any]],
    input_dir: Path,
    result_dir: Path,
    quad_dir: Path,
    label: str,
) -> None:
    quad_dir.mkdir(parents=True, exist_ok=True)
    with (quad_dir / "columns.txt").open("w", encoding="utf-8") as file:
        file.write("source | hairstyle reference | hair-colour reference | " + label + "\n")

    for pair in pairs:
        index = int(pair["index"])
        result_path = result_dir / f"pair_{index:03d}.png"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing generated result: {result_path}")
        with Image.open(result_path) as image:
            result = image.convert("RGB")
        side = result.width
        paths = [
            input_dir / f"pair_{index:03d}_source.png",
            input_dir / f"pair_{index:03d}_shape.png",
            input_dir / f"pair_{index:03d}_color.png",
        ]
        images: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").resize((side, side), Image.Resampling.LANCZOS))
        images.append(result)
        sheet = Image.new("RGB", (side * 4, side), "white")
        for column, image in enumerate(images):
            sheet.paste(image, (column * side, 0))
        sheet.save(quad_dir / f"pair_{index:03d}.png")

