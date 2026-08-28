"""Generate author-baseline reconstruction outputs on a frozen recon manifest.

This is deliberately separate from the FID generation script.  It calls the
same untouched author model, but passes one original image for all three model
inputs: source, hairstyle reference, and colour reference.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.transforms import functional as tvf
from tqdm.auto import tqdm


# ========================= User config: edit here only ========================
USER_MANIFEST = Path("input/eval_pairs_v5/celeba_hq_recon_seed3407_3000.jsonl")
USER_IMAGE_ROOT = Path("/root/shared-nvme/HairFastGAN/celeba-1024")
# Linux path to a clean author checkout.  The model imports the no-suffix
# baseline modules from this directory.  Extra *_v5.py/*_v6.py files are not
# selected by these imports.
USER_BASELINE_ROOT = Path("/root/shared-nvme/HairFastGAN")
# Fill these four paths with the exact weights for this reconstruction run.
# A relative path is resolved from USER_BASELINE_ROOT; absolute paths are used
# directly.
USER_STYLEGAN_CHECKPOINT = Path(
    "/root/shared-nvme/HairFastGAN/pretrained_models/StyleGAN/ffhq.pt"
)
USER_ROTATE_CHECKPOINT = Path(
    "/root/shared-nvme/HairFastGAN/pretrained_models/Rotate/rotate_best.pth"
)
USER_BLENDING_CHECKPOINT = Path(
    "/root/shared-nvme/HairFastGAN/pretrained_models/Blending/checkpoint.pth"
)
USER_PP_CHECKPOINT = Path(
    "/root/shared-nvme/HairFastGAN/pretrained_models/PostProcess/pp_model.pth"
)
USER_DEVICE = "cuda"
USER_CUDA_VISIBLE_DEVICES = "0"
USER_OUTPUT_DIR = Path("output/celeba_hq_baseline_recon/author_baseline/results")
USER_SKIP_EXISTING = True
USER_EXPECTED_COUNT = 3000
# ===============================================================================

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def image_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def baseline_file(root: Path, configured: Path) -> Path:
    path = configured.expanduser()
    return path if path.is_absolute() else root / path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reconstruction manifest: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"Manifest row {line_number} is not an object.")
            rows.append(row)
    if len(rows) != USER_EXPECTED_COUNT:
        raise RuntimeError(f"Expected {USER_EXPECTED_COUNT} rows, found {len(rows)}.")

    seen: set[str] = set()
    for expected_index, row in enumerate(rows, start=1):
        if int(row.get("index", -1)) != expected_index:
            raise RuntimeError(f"Manifest index is not contiguous at row {expected_index}.")
        source = str(row.get("source_file", ""))
        if not source or source != str(row.get("shape_file", "")) or source != str(row.get("color_file", "")):
            raise RuntimeError(f"Row {expected_index} must use one image for all three inputs.")
        if source in seen:
            raise RuntimeError(f"Duplicate source image in manifest: {source}")
        seen.add(source)
        output_file = Path(str(row.get("output_file", f"{expected_index:06d}.png")))
        if output_file.name != str(output_file) or output_file.suffix.lower() != ".png":
            raise RuntimeError(f"Unsafe output filename at row {expected_index}: {output_file}")
        row["source_file"] = source
        row["output_file"] = str(output_file)
    return rows


def verify_rgb(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise RuntimeError(f"Image must be RGB: {path} (mode={image.mode!r})")


def tensor_to_rgb_image(value: torch.Tensor) -> Image.Image:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"Expected one image, got {tuple(value.shape)}")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"Expected RGB [3,H,W], got {tuple(value.shape)}")
    return tvf.to_pil_image(value.detach().cpu().float().clamp(0, 1)).convert("RGB")


def build_author_baseline(baseline_root: Path, device: str):
    sys.path.insert(0, str(baseline_root))
    required_code = (
        "hair_swap.py",
        "models/Alignment.py",
        "models/Blending.py",
        "models/Embedding.py",
        "models/Net.py",
    )
    for relative in required_code:
        path = baseline_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing baseline code file: {path}")
    from hair_swap import HairFast, get_parser  # type: ignore[import-not-found]

    checkpoints = {
        "StyleGAN": baseline_file(baseline_root, USER_STYLEGAN_CHECKPOINT),
        "rotate": baseline_file(baseline_root, USER_ROTATE_CHECKPOINT),
        "blending": baseline_file(baseline_root, USER_BLENDING_CHECKPOINT),
        "postprocess": baseline_file(baseline_root, USER_PP_CHECKPOINT),
    }
    for label, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
    args = get_parser().parse_args([])
    args.device = device
    args.ckpt = str(checkpoints["StyleGAN"])
    args.rotate_checkpoint = str(checkpoints["rotate"])
    args.blending_checkpoint = str(checkpoints["blending"])
    args.pp_checkpoint = str(checkpoints["postprocess"])
    args.save_all = False
    return HairFast(args)


def main() -> None:
    if USER_CUDA_VISIBLE_DEVICES:
        os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES
    if USER_DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("USER_DEVICE is CUDA, but PyTorch cannot see a CUDA device.")

    manifest = repo_path(USER_MANIFEST).resolve()
    image_root = USER_IMAGE_ROOT.expanduser().resolve()
    baseline_root = USER_BASELINE_ROOT.expanduser().resolve()
    output_dir = repo_path(USER_OUTPUT_DIR).resolve()
    rows = load_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_cwd = Path.cwd()
    try:
        os.chdir(baseline_root)
        model = build_author_baseline(baseline_root, USER_DEVICE)
        print("Author baseline checkpoints loaded.", flush=True)
        with torch.inference_mode():
            for row in tqdm(rows, desc="Generate baseline recon"):
                source = image_path(image_root, str(row["source_file"]))
                if not source.is_file():
                    raise FileNotFoundError(f"Missing source image at row {row['index']}: {source}")
                verify_rgb(source)
                output = output_dir / str(row["output_file"])
                if USER_SKIP_EXISTING and output.is_file():
                    verify_rgb(output)
                    continue
                result = model(
                    source,
                    source,
                    source,
                    seed=int(row.get("sample_seed", 3407 + int(row["index"]) - 1)),
                    exp_name=Path(str(row["output_file"])).stem,
                )
                image = tensor_to_rgb_image(result)
                output.parent.mkdir(parents=True, exist_ok=True)
                image.save(output, format="PNG")
    finally:
        os.chdir(original_cwd)

    expected = {str(row["output_file"]) for row in rows}
    actual = {path.name for path in output_dir.glob("*.png")}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(f"Invalid recon result directory: missing={len(missing)}, extra={len(extra)}")
    print(f"Generated/reused {len(expected)} reconstruction images: {output_dir}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
