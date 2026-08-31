"""Generate a fixed CelebA-1024 evaluation set for V6 ablations.

The manifest is the only source of pairing.  Both ablations therefore use the
same 3000 source/shape/color rows, sample seeds and author checkpoints.  The
only changed policy is the requested innovation:

  satd_only    SATD enabled, earring recall disabled
  earring_only SATD disabled, earring recall enabled

The result directory contains only generated PNGs; ``real_source`` contains
the corresponding source images for the existing FID/LPIPS metric scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from hair_swap_v6 import HairFastV6, get_parser
from utils.seed import set_seed


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_image(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_rows(manifest: Path, root: Path, expected_count: int) -> list[dict[str, object]]:
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Missing evaluation manifest: {manifest}. "
            "Run scripts/celeba_hq_make_eval_pairs_v5.py once first."
        )
    rows: list[dict[str, object]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Manifest line {line_number} is not an object.")
            rows.append(row)
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Expected exactly {expected_count} manifest rows, found {len(rows)}: {manifest}"
        )
    seen_outputs: set[str] = set()
    for expected_index, row in enumerate(rows, 1):
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest index is not contiguous at row {expected_index}.")
        if str(row.get("mode", "full")).lower() != "full":
            raise RuntimeError(
                f"Manifest row {expected_index} is not mode=full; use the fixed full triplet manifest."
            )
        output_file = Path(str(row.get("output_file", f"{expected_index:06d}.png")))
        if output_file.name != str(output_file) or output_file.suffix.lower() != ".png":
            raise RuntimeError(f"Unsafe output filename at row {expected_index}: {output_file}")
        if str(output_file) in seen_outputs:
            raise RuntimeError(f"Duplicate output filename: {output_file}")
        seen_outputs.add(str(output_file))
        row["output_file"] = str(output_file)
        role_values = [str(row.get(field, "")) for field in ("source_file", "shape_file", "color_file")]
        if len(set(role_values)) != 3:
            raise RuntimeError(
                f"Manifest row {expected_index} must contain three distinct source/shape/color images."
            )
        for field in ("source_file", "shape_file", "color_file"):
            value = str(row.get(field, ""))
            if not value:
                raise RuntimeError(f"Missing {field} at manifest row {expected_index}.")
            image_path = resolve_image(root, value)
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing {field} image at row {expected_index}: {image_path}")
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as error:  # noqa: BLE001
                raise RuntimeError(f"Unreadable {field} image at row {expected_index}: {image_path}") from error
    return rows


def make_model_args(args: argparse.Namespace) -> argparse.Namespace:
    model_args = get_parser().parse_args([])
    model_args.device = args.device
    model_args.ckpt = str(args.stylegan_checkpoint)
    model_args.rotate_checkpoint = str(args.rotate_checkpoint)
    model_args.blending_checkpoint = str(args.blending_checkpoint)
    model_args.pp_checkpoint = str(args.pp_checkpoint)
    model_args.pp_v6_checkpoint = str(args.pp_checkpoint)
    model_args.use_satd_v8 = args.ablation == "satd_only"
    model_args.satd_checkpoint_v8 = str(args.satd_checkpoint)
    model_args.direct_satd_pp_input = args.ablation == "satd_only"
    model_args.enable_earring_recall = args.ablation == "earring_only"
    model_args.enable_earring_query_recall = args.ablation == "earring_only"
    model_args.v6_runtime_diagnostics = False
    return model_args


def save_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.png")
    save_image(image.detach().cpu().clamp(0, 1), temporary)
    temporary.replace(path)


def copy_real_source(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main(forced_ablation: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fixed CelebA-1024 V6 ablation evaluator")
    parser.add_argument("--ablation", choices=("satd_only", "earring_only"), default=forced_ablation or "satd_only")
    parser.add_argument("--celeba_root", type=Path, default=Path("/root/shared-nvme/HairFastGAN/celeba-1024"))
    parser.add_argument("--manifest", type=Path, default=Path("input/eval_pairs_v5/celeba_hq_full_seed3407_3000.jsonl"))
    parser.add_argument("--output_root", type=Path, default=Path("output/celeba_hq_v6_ablation"))
    parser.add_argument("--pp_checkpoint", type=Path, default=Path("pretrained_models/PostProcess/pp_model.pth"))
    parser.add_argument("--blending_checkpoint", type=Path, default=Path("pretrained_models/Blending/checkpoint.pth"))
    parser.add_argument("--satd_checkpoint", type=Path, default=Path("checkpoints/satd_3000_best.pth"))
    parser.add_argument("--stylegan_checkpoint", type=Path, default=Path("pretrained_models/StyleGAN/ffhq.pt"))
    parser.add_argument("--rotate_checkpoint", type=Path, default=Path("pretrained_models/Rotate/rotate_best.pth"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if forced_ablation is not None and args.ablation != forced_ablation:
        raise RuntimeError(f"{forced_ablation} evaluator cannot run mode {args.ablation!r}.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    root = args.celeba_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    rows = load_rows(manifest, root, args.count)
    output_dir = (args.output_root / args.ablation).expanduser().resolve()
    result_dir = output_dir / "results"
    real_dir = output_dir / "real_source"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "schema": 1,
        "ablation": args.ablation,
        "count": len(rows),
        "seed": args.seed,
        "celeba_root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "pp_checkpoint": str(args.pp_checkpoint.resolve()),
        "blending_checkpoint": str(args.blending_checkpoint.resolve()),
        "satd_checkpoint": str(args.satd_checkpoint.resolve()),
        "checkpoint_sha256": {
            name: file_sha256(path.expanduser().resolve())
            for name, path in {
                "pp": args.pp_checkpoint,
                "blending": args.blending_checkpoint,
                "stylegan": args.stylegan_checkpoint,
                "rotate": args.rotate_checkpoint,
                "satd": args.satd_checkpoint,
            }.items()
            if path.expanduser().resolve().is_file()
        },
        "code_sha256": {
            name: file_sha256(Path(__file__).resolve().parents[1] / relative)
            for name, relative in {
                "hair_swap_v6": "hair_swap_v6.py",
                "alignment_v6": "models/Alignment_v6.py",
                "blending_v6": "models/Blending_v6.py",
                "postprocess_v6": "models/postprocess_v6.py",
            }.items()
        },
        "use_satd_v8": args.ablation == "satd_only",
        "enable_earring_recall": args.ablation == "earring_only",
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config and any(result_dir.glob("*.png")):
            raise RuntimeError(
                f"Existing run has a different policy: {output_dir}. Use a new --output_root."
            )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Seed before model construction as well as per-row HairFast calls.  This
    # keeps any newly initialized optional PP heads identical across the two
    # ablation runs.
    set_seed(args.seed)
    model = HairFastV6(make_model_args(args))
    generated = 0
    with torch.inference_mode():
        for row in tqdm(rows, desc=f"V6 {args.ablation} ({len(rows)} fixed pairs)"):
            output_file = str(row["output_file"])
            output_path = result_dir / output_file
            source_path = resolve_image(root, str(row["source_file"]))
            copy_real_source(source_path, real_dir / output_file)
            if output_path.is_file():
                continue
            sample_seed = int(row.get("sample_seed", args.seed + int(row["index"]) - 1))
            result = model(
                source_path,
                resolve_image(root, str(row["shape_file"])),
                resolve_image(root, str(row["color_file"])),
                align=False,
                seed=sample_seed,
                use_satd_v8=args.ablation == "satd_only",
                direct_satd_pp_input=args.ablation == "satd_only",
                enable_earring_recall=args.ablation == "earring_only",
                enable_earring_query_recall=args.ablation == "earring_only",
                exp_name=Path(output_file).stem,
            )
            if isinstance(result, tuple):
                result = result[0]
            if result.ndim == 4:
                result = result[0]
            save_png(result, output_path)
            generated += 1

    expected = {str(row["output_file"]) for row in rows}
    actual = {path.name for path in result_dir.glob("*.png")}
    if actual != expected:
        raise RuntimeError(f"Result set mismatch: missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}")
    print(f"Ablation: {args.ablation}")
    print(f"Manifest: {manifest} (sha256={config['manifest_sha256']})")
    print(f"Generated this run: {generated}")
    print(f"Results for FID: {result_dir}")
    print(f"Matched real source set: {real_dir}")


if __name__ == "__main__":
    main()
