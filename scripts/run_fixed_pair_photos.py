"""Run a fixed source/hairstyle/hair-colour pairing through one pipeline.

The manifest is created once and reused by every later invocation.  Existing
V5/V6 generators are deliberately untouched; this file is only a photo runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        result: list[str] = []
        while len(result) < count:
            block = list(files)
            rng.shuffle(block)
            result.extend(block)
        return result[:count]

    sources = cycle_sample(source_files)
    references = cycle_sample(reference_files)
    pairs = []
    for index, (source, shape) in enumerate(zip(sources, references)):
        source_id = identity_key(source)
        shape_id = identity_key(shape)
        color_candidates = [
            name
            for name in reference_files
            if identity_key(name) not in {source_id, shape_id}
        ]
        if not color_candidates:
            raise ValueError(f"Cannot choose a distinct colour reference for pair {index}.")
        color = rng.choice(color_candidates)
        pairs.append(
            {
                "index": index,
                "source": source,
                "shape_reference": shape,
                "color_reference": color,
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
                f"{path} does not contain exactly {count} pairs. "
                "Use a new manifest path for a new experiment."
            )
        source_set = set(source_files)
        reference_set = set(reference_files)
        for index, pair in enumerate(pairs):
            if int(pair.get("index", -1)) != index:
                raise RuntimeError(f"Malformed pair index {index} in {path}.")
            if pair.get("source") not in source_set:
                raise FileNotFoundError(f"Missing source image in pair {index}: {pair.get('source')}")
            if pair.get("shape_reference") not in reference_set:
                raise FileNotFoundError(f"Missing shape image in pair {index}: {pair.get('shape_reference')}")
            if pair.get("color_reference") not in reference_set:
                raise FileNotFoundError(f"Missing colour image in pair {index}: {pair.get('color_reference')}")
        print(f"Reusing fixed pair manifest: {path.resolve()}")
        return pairs

    pairs = build_pairs(source_files, reference_files, count, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": 1,
        "count": count,
        "seed": seed,
        "source_dir": str(source_dir.resolve()),
        "reference_dir": str(reference_dir.resolve()),
        "pairs": pairs,
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    temporary.replace(path)
    print(f"Created fixed pair manifest: {path.resolve()}")
    return pairs


def tensor_to_pil(value):
    import numpy as np

    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected one output image, got {tuple(value.shape)}")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(value.shape)}")
    array = (
        value.detach().cpu().float().clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .add(0.5)
        .byte()
        .numpy()
    )
    return Image.fromarray(np.asarray(array), mode="RGB")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def make_model(args):
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if args.method == "baseline":
        baseline_root = args.baseline_root.resolve()
        require_file(baseline_root / "hair_swap.py", "Official baseline hair_swap.py")
        sys.path.insert(0, str(baseline_root))
        sys.path.insert(1, str(PROJECT_ROOT))
        from hair_swap import HairFast, get_parser

        model_args = get_parser().parse_args([])
        model_args.device = args.device
        model_args.ckpt = str(args.stylegan_checkpoint)
        model_args.rotate_checkpoint = str(args.rotate_checkpoint)
        model_args.blending_checkpoint = str(args.blending_checkpoint)
        model_args.pp_checkpoint = str(args.author_pp_checkpoint)
        model_args.save_all = False
        print("Pipeline: official baseline hair_swap.py and author no-suffix modules")
        return HairFast(model_args)

    sys.path.insert(0, str(PROJECT_ROOT))
    from hair_swap_v6 import HairFastV6, get_parser

    model_args = get_parser().parse_args([])
    model_args.device = args.device
    model_args.ckpt = str(args.stylegan_checkpoint)
    model_args.rotate_checkpoint = str(args.rotate_checkpoint)
    model_args.blending_checkpoint = str(args.blending_checkpoint)
    model_args.pp_checkpoint = str(args.author_pp_checkpoint)
    model_args.pp_v6_checkpoint = str(args.v6_pp_checkpoint)
    model_args.use_satd_v8 = bool(args.use_satd)
    model_args.satd_checkpoint_v8 = str(args.satd_checkpoint) if args.use_satd else ""
    model_args.satd_blend_v8 = args.satd_blend
    model_args.allow_legacy_blending_checkpoint_v8 = True
    model_args.enable_earring_recall = bool(args.enable_earring_recall)
    model_args.direct_satd_pp_input = bool(args.use_satd)
    model_args.save_all = False
    print(
        "Pipeline: current V6; "
        f"SATD={'on' if args.use_satd else 'off'}, "
        f"earring={'on' if args.enable_earring_recall else 'off'}"
    )
    return HairFastV6(model_args)


def save_quad(source: Path, shape: Path, color: Path, result: Image.Image, path: Path) -> None:
    side = result.width
    images = []
    for image_path in (source, shape, color):
        with Image.open(image_path) as image:
            images.append(image.convert("RGB").resize((side, side), Image.Resampling.LANCZOS))
    sheet = Image.new("RGB", (side * 4, side), "white")
    for index, image in enumerate((*images, result.convert("RGB"))):
        sheet.paste(image, (index * side, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("baseline", "v6"), required=True)
    parser.add_argument("--source_dir", type=Path, default=Path("/root/shared-nvme/HairFastGAN/images/ear"))
    parser.add_argument("--reference_dir", type=Path, default=Path("/root/shared-nvme/HairFastGAN/images/FFHQ_short"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--manifest", type=Path, default=Path("input/fixed_hair_pairs.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline_root", type=Path, default=Path("/root/shared-nvme/HairFastGAN/HairFastGAN-main"))
    parser.add_argument("--stylegan_checkpoint", type=Path, default=Path("pretrained_models/StyleGAN/ffhq.pt"))
    parser.add_argument("--rotate_checkpoint", type=Path, default=Path("pretrained_models/Rotate/rotate_best.pth"))
    parser.add_argument("--blending_checkpoint", type=Path, default=Path("pretrained_models/Blending/checkpoint.pth"))
    parser.add_argument("--author_pp_checkpoint", type=Path, default=Path("pretrained_models/PostProcess/pp_model.pth"))
    parser.add_argument("--v6_pp_checkpoint", type=Path, default=Path("pretrained_models/PostProcess/pp_model.pth"))
    parser.add_argument("--satd_checkpoint", type=Path, default=Path("checkpoints/satd_3000_best.pth"))
    parser.add_argument("--use_satd", action="store_true")
    parser.add_argument("--satd_blend", type=float, default=0.75)
    parser.add_argument("--enable_earring_recall", action="store_true", default=False)
    parser.add_argument("--no_quads", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_dir = args.source_dir.expanduser().resolve()
    args.reference_dir = args.reference_dir.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    for name in ("stylegan_checkpoint", "rotate_checkpoint", "blending_checkpoint", "author_pp_checkpoint"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.v6_pp_checkpoint = args.v6_pp_checkpoint.expanduser().resolve()
    args.satd_checkpoint = args.satd_checkpoint.expanduser().resolve()
    if args.method == "baseline":
        require_file(args.baseline_root.resolve() / "hair_swap.py", "Official baseline hair_swap.py")
    for path, label in (
        (args.stylegan_checkpoint, "StyleGAN checkpoint"),
        (args.rotate_checkpoint, "rotate checkpoint"),
        (args.blending_checkpoint, "blending checkpoint"),
        (args.author_pp_checkpoint, "author PP checkpoint"),
    ):
        require_file(path, label)
    if args.method == "v6":
        require_file(args.v6_pp_checkpoint, "V6 PP checkpoint")
        if args.use_satd:
            require_file(args.satd_checkpoint, "SATD checkpoint")

    pairs = load_or_create_manifest(
        args.manifest,
        args.source_dir,
        args.reference_dir,
        args.count,
        args.seed,
    )
    result_dir = args.output / "results"
    quad_dir = args.output / "quads"
    result_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "method": args.method,
                "count": len(pairs),
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_sha256(args.manifest),
                "source_dir": str(args.source_dir),
                "reference_dir": str(args.reference_dir),
                "use_satd": bool(args.use_satd),
                "enable_earring_recall": bool(args.enable_earring_recall),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    import torch

    torch.manual_seed(args.seed)
    model = make_model(args)
    try:
        with torch.inference_mode():
            for pair in pairs:
                index = int(pair["index"])
                source = args.source_dir / str(pair["source"])
                shape = args.reference_dir / str(pair["shape_reference"])
                color = args.reference_dir / str(pair["color_reference"])
                output_path = result_dir / f"pair_{index:04d}.png"
                print(f"[{args.method}] pair_{index:04d}: {source.name} | {shape.name} | {color.name}")
                result = model(
                    source,
                    shape,
                    color,
                    seed=int(pair["model_seed"]),
                    exp_name=f"pair_{index:04d}",
                )
                if isinstance(result, tuple):
                    result = result[0]
                image = tensor_to_pil(result)
                image.save(output_path)
                if not args.no_quads:
                    save_quad(source, shape, color, image, quad_dir / f"pair_{index:04d}.png")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"Results: {result_dir}")
    if not args.no_quads:
        print(f"Four-column previews: {quad_dir}")
    print(f"Pair manifest: {args.manifest} (sha256={manifest_sha256(args.manifest)})")


if __name__ == "__main__":
    main()
