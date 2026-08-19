"""Export the source-side earring stages used by the HairFast v5 refinement.

The script creates two paper-ready images from one source face:

* ``*_01_earring_detection.png``: candidate earring pixels overlaid in red.
* ``*_02_mask_extraction.png``: the final binary source earring mask.

This is the source-side portion of ``PostProcessModelV5._enhance_query_recall``.
It intentionally does not run StyleGAN2 or require shape/color reference images.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

# Running ``python scripts/...`` puts only ``scripts`` on sys.path. Add the
# repository root so the v5 ``models`` package resolves in every shell.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.CtrlHair.external_code.face_parsing.model import BiSeNet
from models.ear_modules_v5 import (
    EarAnchoredQueryBuilder,
    build_weak_earring_masks,
    enhance_query_with_earring_recall,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the Earring Detection and Mask Extraction stages of HairFast v5."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input RGB face image with earrings.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/v5_earring_stages"))
    parser.add_argument(
        "--parser-checkpoint",
        type=Path,
        default=Path("pretrained_models/BiSeNet/face_parsing_79999_iter.pth"),
        help="The 19-class BiSeNet checkpoint used by v5.",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--work-size",
        type=int,
        default=256,
        help="v5 post-process resolution. Keep the default to match HairFast v5 inference.",
    )
    parser.add_argument("--parse-size", type=int, default=512, help="BiSeNet input size used by v5.")
    parser.add_argument("--ear-dilate", type=int, default=21)
    parser.add_argument("--hair-change-dilate", type=int, default=25)
    parser.add_argument("--earring-expand", type=int, default=15)
    parser.add_argument("--ear-downward-shift", type=int, default=10)
    parser.add_argument("--target-hair-dilate", type=int, default=11)
    parser.add_argument("--source-hair-block-dilate", type=int, default=5)
    parser.add_argument("--source-hair-block-strength", type=float, default=0.6)
    parser.add_argument("--target-visibility-expand", type=int, default=5)
    parser.add_argument("--max-target-hair-overlap", type=float, default=0.55)
    parser.add_argument("--recall-dilate", type=int, default=7)
    parser.add_argument("--recall-downward-shift", type=int, default=18)
    parser.add_argument("--recall-lower-lobe-weight", type=float, default=0.20)
    parser.add_argument("--recall-candidate-boost", type=float, default=0.90)
    parser.add_argument("--recall-block-protect", type=float, default=0.85)
    return parser.parse_args()


def load_rgb(path: Path) -> tuple[Image.Image, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find input image: {path}")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return rgb.copy(), pil_to_tensor(rgb).float().div(255)


def load_parser(checkpoint_path: Path, device: torch.device) -> BiSeNet:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Cannot find the v5 face-parsing checkpoint: "
            f"{checkpoint_path}. Set --parser-checkpoint to face_parsing_79999_iter.pth."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    parser = BiSeNet(n_classes=19).to(device)
    parser.load_state_dict(state_dict)
    return parser.eval()


@torch.inference_mode()
def parse_face(
    parser: BiSeNet,
    image_01: torch.Tensor,
    parse_size: int,
) -> torch.Tensor:
    """Match FaceParsingHelperV5: resize, ImageNet-normalize, then argmax."""
    parser_input = F.interpolate(
        image_01,
        size=(parse_size, parse_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = parser_input.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = parser_input.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    labels = parser((parser_input - mean) / std)[0].argmax(dim=1, keepdim=True).float()
    return F.interpolate(labels, size=image_01.shape[-2:], mode="nearest").long()


def to_full_size(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(mask.float(), size=size, mode="nearest")[0].squeeze(0).cpu()


def save_mask(path: Path, mask: torch.Tensor) -> None:
    Image.fromarray((mask.clamp(0, 1).numpy() * 255).round().astype("uint8"), mode="L").save(path)


def save_detection_overlay(path: Path, source: Image.Image, detection_mask: torch.Tensor) -> None:
    source_tensor = pil_to_tensor(source).float().div(255)
    mask = detection_mask.unsqueeze(0).clamp(0, 1)
    red = torch.zeros_like(source_tensor)
    red[0] = 1
    overlay = source_tensor * (1 - 0.60 * mask) + red * (0.60 * mask)
    image = (overlay.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(image, mode="RGB").save(path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.work_size <= 0 or args.parse_size <= 0:
        raise ValueError("--work-size and --parse-size must be positive integers.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or activate the CUDA environment.")

    source_pil, source_full = load_rgb(args.input)
    original_size = (source_pil.height, source_pil.width)
    source_01 = F.interpolate(
        source_full.unsqueeze(0).to(device),
        size=(args.work_size, args.work_size),
        mode="bilinear",
        align_corners=False,
    )

    face_parser = load_parser(args.parser_checkpoint, device)
    source_parsing = parse_face(face_parser, source_01, args.parse_size)

    query_builder = EarAnchoredQueryBuilder(
        ear_dilate=args.ear_dilate,
        hair_change_dilate=args.hair_change_dilate,
        earring_expand=args.earring_expand,
        downward_shift=args.ear_downward_shift,
        target_hair_dilate=args.target_hair_dilate,
        source_hair_block_dilate=args.source_hair_block_dilate,
        source_hair_block_strength=args.source_hair_block_strength,
        target_visibility_expand=args.target_visibility_expand,
        max_target_hair_overlap=args.max_target_hair_overlap,
    )

    # A single source image uses itself as the target only to construct the
    # v5 visible-ear ROI. The candidate detector remains source-image based.
    query_info = query_builder(source_parsing, source_parsing)
    weak_masks = build_weak_earring_masks(
        source_01,
        query_info["visible_ear_roi"],
        query_info["query_mask"],
        query_info["source_earring_mask"],
        query_info["source_hair_mask"],
        query_info["source_hair_block_mask"],
        source_parsing,
    )
    recall_info = enhance_query_with_earring_recall(
        query_info["query_mask"],
        query_info["source_earring_mask"],
        query_info["source_hair_block_mask"],
        weak_masks,
        recall_dilate=args.recall_dilate,
        downward_shift=args.recall_downward_shift,
        lower_lobe_weight=args.recall_lower_lobe_weight,
        candidate_boost=args.recall_candidate_boost,
        block_protect=args.recall_block_protect,
    )

    # These are the two source-side stages exposed by PostProcessModelV5.
    detection_mask = recall_info["online_earring_candidate_mask"]
    extraction_mask = torch.clamp(
        weak_masks["earring_confident_mask"] + recall_info["source_earring_mask"],
        0,
        1,
    )

    detection_full = to_full_size(detection_mask, original_size)
    extraction_full = to_full_size(extraction_mask, original_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    detection_path = args.output_dir / f"{stem}_01_earring_detection.png"
    extraction_path = args.output_dir / f"{stem}_02_mask_extraction.png"
    save_detection_overlay(detection_path, source_pil, detection_full)
    save_mask(extraction_path, extraction_full)

    detected_pixels = int((detection_full > 0).sum().item())
    mask_pixels = int((extraction_full > 0).sum().item())
    print(f"Earring Detection: {detection_path} ({detected_pixels} highlighted pixels)")
    print(f"Mask Extraction: {extraction_path} ({mask_pixels} white pixels)")


if __name__ == "__main__":
    main()
