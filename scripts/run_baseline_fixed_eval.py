"""Run the official no-suffix HairFastGAN baseline on fixed image pairs.

The baseline is imported from ``--baseline_root``.  This script deliberately
does not import ``hair_swap_v5``, ``hair_swap_v6`` or ``hair_swap_v8``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/ear")
DEFAULT_REFERENCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
DEFAULT_BASELINE_ROOT = Path("/data/coding/HairFastGAN/HairFastGAN-main")
DEFAULT_MANIFEST = PROJECT_ROOT / "input" / "fixed_eval_pairs_30.json"
VERSIONED_STEM = re.compile(r".*_v(?:5|6|8)$", re.IGNORECASE)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def list_images(directory: Path) -> list[str]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    return sorted(
        item.name
        for item in directory.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def identity_key(name: str) -> str:
    return Path(name).stem.casefold()


def build_pairs(source_files: list[str], reference_files: list[str], count: int, seed: int):
    if count <= 0:
        raise ValueError("--count must be positive")
    if len({identity_key(name) for name in reference_files}) < 3:
        raise ValueError("The reference directory must contain at least three identities.")

    rng = random.Random(seed)

    def cycle_sample(files: list[str]) -> list[str]:
        selected: list[str] = []
        while len(selected) < count:
            block = list(files)
            rng.shuffle(block)
            selected.extend(block)
        return selected[:count]

    sources = cycle_sample(source_files)
    shapes = cycle_sample(reference_files)
    pairs = []
    for index, (source, shape) in enumerate(zip(sources, shapes)):
        excluded = {identity_key(source), identity_key(shape)}
        candidates = [name for name in reference_files if identity_key(name) not in excluded]
        if not candidates:
            raise ValueError(f"No distinct colour reference is available for pair {index}.")
        pairs.append(
            {
                "index": index,
                "source": source,
                "shape_reference": shape,
                "color_reference": rng.choice(candidates),
                "model_seed": int(seed) + index,
            }
        )
    return pairs


def manifest_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_or_create_manifest(
    path: Path,
    source_dir: Path,
    reference_dir: Path,
    count: int,
    seed: int,
) -> list[dict[str, object]]:
    source_files = list_images(source_dir)
    reference_files = list_images(reference_dir)
    if not source_files or not reference_files:
        raise ValueError("Both image directories must contain at least one image.")

    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        pairs = payload.get("pairs") if isinstance(payload, dict) else None
        if not isinstance(pairs, list) or len(pairs) != count:
            raise RuntimeError(
                f"{path} does not contain exactly {count} pairs. Use a new manifest path."
            )
        source_set = set(source_files)
        reference_set = set(reference_files)
        for index, pair in enumerate(pairs):
            if int(pair.get("index", -1)) != index:
                raise RuntimeError(f"Malformed pair index {index} in {path}.")
            for key, available, label in (
                ("source", source_set, "source"),
                ("shape_reference", reference_set, "shape reference"),
                ("color_reference", reference_set, "colour reference"),
            ):
                if pair.get(key) not in available:
                    raise FileNotFoundError(f"Missing {label} in pair {index}: {pair.get(key)}")
        print(f"Reusing fixed pair manifest: {path.resolve()}")
        return pairs

    pairs = build_pairs(source_files, reference_files, count, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": 1,
        "count": count,
        "seed": seed,
        "source_dir": str(source_dir),
        "reference_dir": str(reference_dir),
        "pairs": pairs,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    temporary.replace(path)
    print(f"Created fixed pair manifest: {path.resolve()}")
    return pairs


def save_tensor_image(value, path: Path) -> None:
    import numpy as np

    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected one output image, got {tuple(value.shape)}")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(value.shape)}")
    array = (
        value.detach().float().clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .add(0.5)
        .byte()
        .cpu()
        .numpy()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array), mode="RGB").save(path)


def save_quad(source: Path, shape: Path, color: Path, result: Path, path: Path) -> None:
    with Image.open(result) as image:
        side = image.width
        result_image = image.convert("RGB")
    images = []
    for image_path in (source, shape, color):
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").resize((side, side), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (side * 4, side), "white")
    for index, image in enumerate((*images, result_image)):
        sheet.paste(image, (index * side, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def import_official_baseline(baseline_root: Path):
    """Import only the official no-suffix baseline package."""
    require_file(baseline_root / "hair_swap.py", "Official baseline hair_swap.py")
    for module_name in list(sys.modules):
        if module_name == "hair_swap" or module_name == "models" or module_name.startswith("models."):
            del sys.modules[module_name]
    sys.path.insert(0, str(baseline_root))
    hair_swap = importlib.import_module("hair_swap")
    loaded = {
        name: importlib.import_module(name)
        for name in (
            "hair_swap",
            "models.Alignment",
            "models.Blending",
            "models.Embedding",
            "models.Net",
        )
    }
    for name, module in loaded.items():
        origin = Path(module.__file__).resolve()
        if baseline_root not in origin.parents:
            raise RuntimeError(f"Baseline import escaped --baseline_root: {name} -> {origin}")
        if VERSIONED_STEM.fullmatch(origin.stem):
            raise RuntimeError(f"Versioned module was imported for baseline: {origin}")
    print("Baseline modules:")
    for name, module in loaded.items():
        print(f"  {name}: {Path(module.__file__).resolve()}")
    return hair_swap.HairFast, hair_swap.get_parser


def validate_loaded_baseline_modules(baseline_root: Path) -> None:
    """Reject versioned or out-of-tree model modules used by the baseline."""
    offenders: list[str] = []
    for module in tuple(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if not origin:
            continue
        path = Path(origin).resolve()
        if module.__name__ == "models" or module.__name__.startswith("models."):
            if baseline_root not in path.parents:
                offenders.append(f"out-of-tree {module.__name__}: {path}")
                continue
        if baseline_root not in path.parents:
            continue
        if VERSIONED_STEM.fullmatch(path.stem):
            offenders.append(str(path))
    if offenders:
        raise RuntimeError(
            "The official baseline imported versioned modules; refusing to run:\n"
            + "\n".join(sorted(set(offenders)))
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "baseline_fixed_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stylegan_checkpoint", type=Path, default=None)
    parser.add_argument("--rotate_checkpoint", type=Path, default=None)
    parser.add_argument("--blending_checkpoint", type=Path, default=None)
    parser.add_argument("--pp_checkpoint", type=Path, default=None)
    parser.add_argument("--no_quads", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    for name in ("source_dir", "reference_dir", "baseline_root", "manifest", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    checkpoint_defaults = {
        "stylegan_checkpoint": args.baseline_root / "pretrained_models" / "StyleGAN" / "ffhq.pt",
        "rotate_checkpoint": args.baseline_root / "pretrained_models" / "Rotate" / "rotate_best.pth",
        "blending_checkpoint": args.baseline_root / "pretrained_models" / "Blending" / "checkpoint.pth",
        "pp_checkpoint": args.baseline_root / "pretrained_models" / "PostProcess" / "pp_model.pth",
    }
    for name, default in checkpoint_defaults.items():
        value = getattr(args, name) or default
        value = value.expanduser().resolve()
        if VERSIONED_STEM.fullmatch(value.stem):
            raise RuntimeError(f"Baseline checkpoint must be no-suffix: {value}")
        setattr(args, name, value)
        require_file(value, f"Baseline {name}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    pairs = load_or_create_manifest(
        args.manifest,
        args.source_dir,
        args.reference_dir,
        args.count,
        args.seed,
    )
    HairFast, get_parser = import_official_baseline(args.baseline_root)
    model_args = get_parser().parse_args([])
    model_args.device = args.device
    model_args.ckpt = str(args.stylegan_checkpoint)
    model_args.rotate_checkpoint = str(args.rotate_checkpoint)
    model_args.blending_checkpoint = str(args.blending_checkpoint)
    model_args.pp_checkpoint = str(args.pp_checkpoint)
    model_args.save_all = False

    args.output.mkdir(parents=True, exist_ok=True)
    result_dir = args.output / "results"
    quad_dir = args.output / "quads"
    result_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "method": "official_baseline_no_suffix",
                "count": len(pairs),
                "seed": args.seed,
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_sha256(args.manifest),
                "source_dir": str(args.source_dir),
                "reference_dir": str(args.reference_dir),
                "baseline_root": str(args.baseline_root),
                "checkpoints": {key: str(getattr(args, key)) for key in checkpoint_defaults},
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = HairFast(model_args)
    validate_loaded_baseline_modules(args.baseline_root)
    with torch.inference_mode():
        for pair in pairs:
            index = int(pair["index"])
            source = args.source_dir / str(pair["source"])
            shape = args.reference_dir / str(pair["shape_reference"])
            color = args.reference_dir / str(pair["color_reference"])
            output_path = result_dir / f"pair_{index:04d}.png"
            print(f"[baseline] pair_{index:04d}: {source.name} | {shape.name} | {color.name}")
            result = model(source, shape, color, seed=int(pair["model_seed"]))
            save_tensor_image(result, output_path)
            if not args.no_quads:
                save_quad(source, shape, color, output_path, quad_dir / f"pair_{index:04d}.png")
    print(f"Baseline results: {result_dir}")
    print(f"Pair manifest: {args.manifest} (sha256={manifest_sha256(args.manifest)})")


if __name__ == "__main__":
    main()
