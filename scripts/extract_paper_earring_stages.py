"""Create paper-ready earring Detection and Mask Extraction figures.

This utility uses the same 19-class BiSeNet face parser as HairFast v5 to
localize each ear. It then applies box-prompted GrabCut or Segment Anything
(SAM) to extract an entire earring instance. It is intended for visualisation:
the extracted object mask is a v5-guided extension, not the sparse
``fine_mask`` used by v5 injection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

# ``python scripts/...`` otherwise omits the repository root from sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.CtrlHair.external_code.face_parsing.model import BiSeNet


RAW_LEFT_EAR = 7
RAW_RIGHT_EAR = 8
RAW_EARRING = 9
RAW_FACE_SURFACE = (1, 10)


def parse_box(value: str) -> tuple[int, int, int, int]:
    try:
        x1, y1, x2, y2 = (int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("A box must be x1,y1,x2,y2.") from exc
    if x2 <= x1 or y2 <= y1:
        raise argparse.ArgumentTypeError("Box coordinates must satisfy x2>x1 and y2>y1.")
    return x1, y1, x2, y2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create v5-guided paper figures for complete earring instances."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input RGB face image.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/paper_earring_stages"))
    parser.add_argument(
        "--parser-checkpoint",
        type=Path,
        default=Path("pretrained_models/BiSeNet/face_parsing_79999_iter.pth"),
    )
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--parse-size", type=int, default=512)
    parser.add_argument(
        "--box",
        type=parse_box,
        action="append",
        help="Optional earring box x1,y1,x2,y2 in original-image pixels. Repeat for both ears.",
    )
    parser.add_argument(
        "--mask-method",
        choices=("grabcut", "sam"),
        default="grabcut",
        help="Use sam for publication-quality masks; grabcut needs no extra model but is less accurate.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="Path to a SAM checkpoint, required when --mask-method sam.",
    )
    parser.add_argument(
        "--sam-model-type",
        choices=("vit_b", "vit_l", "vit_h"),
        default="vit_b",
        help="SAM encoder matching --sam-checkpoint.",
    )
    parser.add_argument("--grabcut-iterations", type=int, default=5)
    parser.add_argument("--min-component-area", type=int, default=20)
    return parser.parse_args()


def load_rgb(path: Path) -> tuple[Image.Image, np.ndarray, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find input image: {path}")
    with Image.open(path) as image:
        rgb_image = image.convert("RGB").copy()
    rgb = np.asarray(rgb_image).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0)
    return rgb_image, rgb, tensor


def load_parser(checkpoint_path: Path, device: torch.device) -> BiSeNet:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Cannot find v5 BiSeNet checkpoint: {checkpoint_path}. "
            "Pass --parser-checkpoint with face_parsing_79999_iter.pth."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model = BiSeNet(n_classes=19).to(device)
    model.load_state_dict(state_dict)
    return model.eval()


@torch.inference_mode()
def parse_face(parser: BiSeNet, image_01: torch.Tensor, parse_size: int) -> torch.Tensor:
    parser_input = F.interpolate(image_01, size=(parse_size, parse_size), mode="bilinear", align_corners=False)
    mean = parser_input.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = parser_input.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    labels = parser((parser_input - mean) / std)[0].argmax(dim=1, keepdim=True).float()
    return F.interpolate(labels, size=image_01.shape[-2:], mode="nearest").long()[0, 0].cpu().numpy()


def mask_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clipped_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def auto_earring_boxes(labels: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Use v5 ear labels to create generous boxes for dangling earrings."""
    height, width = labels.shape
    face_box = mask_box(np.isin(labels, RAW_FACE_SURFACE))
    if face_box is None:
        face_box = (int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85))
    face_x1, face_y1, face_x2, face_y2 = face_box
    face_width = max(1, face_x2 - face_x1)
    face_height = max(1, face_y2 - face_y1)

    boxes: list[tuple[int, int, int, int]] = []
    for left_side, ear_label in ((True, RAW_LEFT_EAR), (False, RAW_RIGHT_EAR)):
        ear_mask = labels == ear_label
        earring_mask = labels == RAW_EARRING
        half = np.zeros_like(ear_mask, dtype=bool)
        if left_side:
            half[:, : width // 2 + 1] = True
        else:
            half[:, width // 2 + 1 :] = True
        anchor_box = mask_box((ear_mask | earring_mask) & half)

        if anchor_box is not None:
            _, anchor_y1, _, anchor_y2 = anchor_box
            anchor_y = int(round(0.60 * anchor_y1 + 0.40 * anchor_y2))
            anchor_x = anchor_box[0] if left_side else anchor_box[2] - 1
        else:
            anchor_y = int(round(face_y1 + 0.66 * face_height))
            anchor_x = face_x1 if left_side else face_x2 - 1

        half_width = max(40, int(round(0.16 * face_width)))
        top = anchor_y - max(20, int(round(0.13 * face_height)))
        bottom = anchor_y + max(100, int(round(0.55 * face_height)))
        boxes.append(clipped_box((anchor_x - half_width, top, anchor_x + half_width, bottom), width, height))
    return boxes


def jewelry_seed(image_bgr: np.ndarray, box: tuple[int, int, int, int], face_mask: np.ndarray) -> np.ndarray:
    """Create conservative silver/colour/edge seeds for GrabCut inside one box."""
    import cv2

    x1, y1, x2, y2 = box
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )

    local_gray = gray[y1:y2, x1:x2]
    local_gradient = gradient[y1:y2, x1:x2]
    bright_threshold = max(135, int(np.percentile(local_gray, 72)))
    edge_threshold = max(22.0, float(np.percentile(local_gradient, 66)))
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    bright_metal = (gray >= bright_threshold) & (saturation <= 185)
    coloured_metal = (saturation >= 55) & (value >= max(95, bright_threshold - 35))
    textured = gradient >= edge_threshold
    seed = (bright_metal | (coloured_metal & textured)) & textured
    seed &= ~face_mask

    output = np.zeros_like(gray, dtype=np.uint8)
    output[y1:y2, x1:x2] = seed[y1:y2, x1:x2].astype(np.uint8)
    return output


def retain_seeded_components(mask: np.ndarray, seed: np.ndarray, min_area: int) -> np.ndarray:
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    result = np.zeros_like(mask, dtype=np.uint8)
    seed_support = cv2.dilate(seed.astype(np.uint8), np.ones((7, 7), np.uint8))
    for index in range(1, count):
        component = labels == index
        if stats[index, cv2.CC_STAT_AREA] < min_area:
            continue
        if np.any(component & (seed_support > 0)):
            result[component] = 1
    return result


def extract_box_mask(
    image_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    face_mask: np.ndarray,
    iterations: int,
    min_component_area: int,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = box
    seed = jewelry_seed(image_bgr, box, face_mask)
    grabcut_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[y1:y2, x1:x2] = cv2.GC_PR_BGD
    grabcut_mask[seed > 0] = cv2.GC_FGD

    # Skin is not part of a hanging earring; use it only as probable background
    # so a metal loop touching the earlobe can still remain foreground.
    inside = np.zeros((height, width), dtype=bool)
    inside[y1:y2, x1:x2] = True
    grabcut_mask[inside & face_mask] = cv2.GC_PR_BGD
    if not np.any(seed):
        return np.zeros((height, width), dtype=np.uint8), seed

    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        image_bgr,
        grabcut_mask,
        None,
        background_model,
        foreground_model,
        max(1, iterations),
        cv2.GC_INIT_WITH_MASK,
    )
    foreground = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    foreground[~inside] = 0
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    foreground = retain_seeded_components(foreground, seed, min_component_area)
    return foreground, seed


@torch.inference_mode()
def extract_sam_masks(
    image_rgb: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    checkpoint_path: Path | None,
    model_type: str,
    device: torch.device,
) -> np.ndarray:
    if checkpoint_path is None:
        raise ValueError("--sam-checkpoint is required when --mask-method sam.")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot find SAM checkpoint: {checkpoint_path}")
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as exc:
        raise ImportError(
            "SAM is not installed. Install it with: pip install segment-anything"
        ) from exc

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path)).to(device=device).eval()
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    combined = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    for box in boxes:
        masks, scores, _ = predictor.predict(
            box=np.asarray(box, dtype=np.float32),
            multimask_output=True,
        )
        combined = np.maximum(combined, masks[int(np.argmax(scores))].astype(np.uint8))
    return combined


def save_binary_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_detection(path: Path, image: Image.Image, boxes: list[tuple[int, int, int, int]]) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    colors = ((238, 64, 64), (52, 187, 99))
    for index, box in enumerate(boxes):
        draw.rectangle(box, outline=colors[index % len(colors)], width=4)
    canvas.save(path)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.parse_size <= 0 or args.grabcut_iterations <= 0 or args.min_component_area <= 0:
        raise ValueError("--parse-size, --grabcut-iterations, and --min-component-area must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or activate the CUDA environment.")

    source_pil, source_rgb, source_tensor = load_rgb(args.input)
    parser = load_parser(args.parser_checkpoint, device)
    parsing = parse_face(parser, source_tensor.to(device), args.parse_size)
    boxes = [clipped_box(box, source_pil.width, source_pil.height) for box in args.box] if args.box else auto_earring_boxes(parsing)

    if args.mask_method == "sam":
        combined_mask = extract_sam_masks(
            source_rgb,
            boxes,
            args.sam_checkpoint,
            args.sam_model_type,
            device,
        )
    else:
        image_bgr = source_rgb[:, :, ::-1].copy()
        face_mask = np.isin(parsing, RAW_FACE_SURFACE)
        combined_mask = np.zeros(parsing.shape, dtype=np.uint8)
        for box in boxes:
            object_mask, _ = extract_box_mask(
                image_bgr,
                box,
                face_mask,
                args.grabcut_iterations,
                args.min_component_area,
            )
            combined_mask = np.maximum(combined_mask, object_mask)

    if not np.any(combined_mask):
        raise RuntimeError(
            "No foreground was extracted. Pass tighter manual --box x1,y1,x2,y2 values around each earring."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem
    detection_path = args.output_dir / f"{stem}_01_earring_detection.png"
    extraction_path = args.output_dir / f"{stem}_02_mask_extraction.png"
    save_detection(detection_path, source_pil, boxes)
    save_binary_mask(extraction_path, combined_mask)
    print(f"Earring Detection: {detection_path}")
    print(f"Mask Extraction: {extraction_path}")
    print(f"Detection boxes: {', '.join(','.join(map(str, box)) for box in boxes)}")


if __name__ == "__main__":
    main()
