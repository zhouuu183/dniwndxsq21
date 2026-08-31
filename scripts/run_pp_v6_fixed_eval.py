"""Run the current V6 pipeline with a trained PP checkpoint on fixed pairs.

The pair manifest is intentionally the same default file as the official
baseline runner, so both experiments use identical source/shape/colour
images and per-pair seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/ear")
DEFAULT_REFERENCE_DIR = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
DEFAULT_MANIFEST = PROJECT_ROOT / "input" / "fixed_eval_pairs_30.json"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "pp_v6_fixed_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stylegan_checkpoint", type=Path, default=PROJECT_ROOT / "pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument("--rotate_checkpoint", type=Path, default=PROJECT_ROOT / "pretrained_models/Rotate/rotate_best.pth")
    parser.add_argument("--blending_checkpoint", type=Path, default=PROJECT_ROOT / "pretrained_models/Blending/checkpoint.pth")
    parser.add_argument("--author_pp_checkpoint", type=Path, default=PROJECT_ROOT / "pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument(
        "--pp_weight", "--pp_v6_checkpoint", dest="pp_weight", type=Path, required=True,
        help="Checkpoint produced by pp_train_v6.py.",
    )
    parser.add_argument("--satd_checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints/satd_3000_best.pth")
    parser.add_argument("--use_satd", type=int, choices=(0, 1), default=1)
    parser.add_argument("--satd_blend", type=float, default=0.34)
    parser.add_argument("--direct_satd_pp_input", type=int, choices=(0, 1), default=1)
    parser.add_argument("--enable_earring_recall", type=int, choices=(0, 1), default=1)
    parser.add_argument("--no_quads", action="store_true")
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    for name in (
        "source_dir", "reference_dir", "manifest", "output", "stylegan_checkpoint",
        "rotate_checkpoint", "blending_checkpoint", "author_pp_checkpoint", "pp_weight",
        "satd_checkpoint",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path, label in (
        (args.stylegan_checkpoint, "StyleGAN checkpoint"),
        (args.rotate_checkpoint, "Rotate checkpoint"),
        (args.blending_checkpoint, "Blending checkpoint"),
        (args.author_pp_checkpoint, "Author PP checkpoint"),
        (args.pp_weight, "Trained V6 PP checkpoint"),
    ):
        require_file(path, label)
    if args.use_satd:
        require_file(args.satd_checkpoint, "SATD checkpoint")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    pairs = load_or_create_manifest(
        args.manifest,
        args.source_dir,
        args.reference_dir,
        args.count,
        args.seed,
    )

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from hair_swap_v6 import HairFastV6, get_parser

    model_args = get_parser().parse_args([])
    model_args.device = args.device
    model_args.ckpt = str(args.stylegan_checkpoint)
    model_args.rotate_checkpoint = str(args.rotate_checkpoint)
    model_args.blending_checkpoint = str(args.blending_checkpoint)
    model_args.pp_checkpoint = str(args.author_pp_checkpoint)
    model_args.pp_v6_checkpoint = str(args.pp_weight)
    model_args.use_satd_v8 = bool(args.use_satd)
    model_args.satd_checkpoint_v8 = str(args.satd_checkpoint) if args.use_satd else ""
    model_args.satd_blend_v8 = float(args.satd_blend)
    model_args.direct_satd_pp_input = bool(args.direct_satd_pp_input and args.use_satd)
    model_args.enable_earring_recall = bool(args.enable_earring_recall)
    model_args.enable_earring_query_recall = bool(args.enable_earring_recall)
    model_args.allow_legacy_blending_checkpoint_v8 = True
    model_args.save_all = False
    model_args.v6_runtime_diagnostics = False

    args.output.mkdir(parents=True, exist_ok=True)
    result_dir = args.output / "results"
    quad_dir = args.output / "quads"
    result_dir.mkdir(parents=True, exist_ok=True)
    (args.output / "run_config.json").write_text(
        json.dumps(
            {
                "method": "current_v6_pp_weight",
                "count": len(pairs),
                "seed": args.seed,
                "manifest": str(args.manifest),
                "manifest_sha256": manifest_sha256(args.manifest),
                "source_dir": str(args.source_dir),
                "reference_dir": str(args.reference_dir),
                "pp_weight": str(args.pp_weight),
                "use_satd": bool(args.use_satd),
                "direct_satd_pp_input": bool(args.direct_satd_pp_input and args.use_satd),
                "enable_earring_recall": bool(args.enable_earring_recall),
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
    model = HairFastV6(model_args)
    with torch.inference_mode():
        for pair in pairs:
            index = int(pair["index"])
            source = args.source_dir / str(pair["source"])
            shape = args.reference_dir / str(pair["shape_reference"])
            color = args.reference_dir / str(pair["color_reference"])
            output_path = result_dir / f"pair_{index:04d}.png"
            print(f"[pp-v6] pair_{index:04d}: {source.name} | {shape.name} | {color.name}")
            result = model(source, shape, color, seed=int(pair["model_seed"]))
            save_tensor_image(result, output_path)
            if not args.no_quads:
                save_quad(source, shape, color, output_path, quad_dir / f"pair_{index:04d}.png")
    print(f"V6 PP results: {result_dir}")
    print(f"Pair manifest: {args.manifest} (sha256={manifest_sha256(args.manifest)})")


if __name__ == "__main__":
    main()
