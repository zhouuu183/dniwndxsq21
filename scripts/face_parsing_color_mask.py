import argparse
from contextlib import contextmanager
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.CtrlHair.external_code.face_parsing import resnet as face_parsing_resnet
from models.CtrlHair.external_code.face_parsing.model import BiSeNet


LABELS = {
    0: "background",
    1: "skin_other",
    2: "l_brow",
    3: "r_brow",
    4: "l_eye",
    5: "r_eye",
    6: "eye_g",
    7: "l_ear",
    8: "r_ear",
    9: "ear_r",
    10: "nose",
    11: "mouth",
    12: "u_lip",
    13: "l_lip",
    14: "neck",
    15: "neck_l",
    16: "cloth",
    17: "hair",
    18: "hat",
}

PALETTE = np.array(
    [
        [0, 128, 64],     # background
        [220, 0, 0],      # skin_other
        [0, 255, 255],    # l_brow
        [0, 220, 220],    # r_brow
        [255, 0, 220],    # l_eye
        [0, 64, 255],     # r_eye
        [255, 255, 255],  # eye_g
        [255, 160, 96],   # l_ear
        [255, 128, 64],   # r_ear
        [255, 255, 0],    # ear_r / earring
        [64, 160, 0],     # nose
        [255, 255, 0],    # mouth
        [120, 220, 40],   # u_lip
        [0, 0, 120],      # l_lip
        [255, 160, 48],   # neck
        [255, 192, 96],   # neck_l
        [0, 220, 0],      # cloth
        [0, 0, 220],      # hair
        [160, 0, 255],    # hat
    ],
    dtype=np.uint8,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a colored 19-class face parsing mask.")
    parser.add_argument("image", type=Path, help="Input image path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output colored mask path. Default: <image_stem>_parsing_color.png next to input.",
    )
    parser.add_argument(
        "--label_output",
        type=Path,
        default=None,
        help="Optional output path for raw label ids as an 8-bit grayscale PNG.",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="Optional output path for mask overlay on the source image.",
    )
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay alpha when --overlay is used.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained_models/BiSeNet/face_parsing_79999_iter.pth"),
        help="Face parsing checkpoint path.",
    )
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cuda', or 'cpu'.")
    return parser


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@contextmanager
def skip_resnet18_download():
    original_load_url = face_parsing_resnet.modelzoo.load_url
    face_parsing_resnet.modelzoo.load_url = lambda *args, **kwargs: {}
    try:
        yield
    finally:
        face_parsing_resnet.modelzoo.load_url = original_load_url


def load_model(checkpoint: Path, device: torch.device) -> BiSeNet:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    with skip_resnet18_download():
        model = BiSeNet(n_classes=19).to(device).eval()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    return model


def parse_image(model: BiSeNet, image: Image.Image, device: torch.device) -> np.ndarray:
    original_size = image.size
    resized = image.convert("RGB").resize((512, 512), Image.BILINEAR)
    image_np = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    image_np = (image_np - mean) / std
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0]
        parsing_512 = logits.squeeze(0).argmax(dim=0).byte().cpu().numpy()

    label_image = Image.fromarray(parsing_512, mode="L")
    label_image = label_image.resize(original_size, Image.NEAREST)
    return np.array(label_image, dtype=np.uint8)


def colorize(label_map: np.ndarray) -> Image.Image:
    label_map = np.clip(label_map, 0, len(PALETTE) - 1)
    return Image.fromarray(PALETTE[label_map], mode="RGB")


def save_label_legend(path: Path) -> None:
    legend_path = path.with_suffix(".labels.txt")
    with legend_path.open("w", encoding="utf-8") as file:
        for idx, name in LABELS.items():
            color = PALETTE[idx].tolist()
            file.write(f"{idx}: {name} rgb={tuple(color)}\n")


def main() -> None:
    args = build_parser().parse_args()
    image_path = args.image
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output = args.output or image_path.with_name(f"{image_path.stem}_parsing_color.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = load_model(args.checkpoint, device)
    image = Image.open(image_path).convert("RGB")
    label_map = parse_image(model, image, device)
    color_mask = colorize(label_map)
    color_mask.save(output)
    save_label_legend(output)

    if args.label_output is not None:
        args.label_output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(label_map, mode="L").save(args.label_output)

    if args.overlay is not None:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        alpha = max(0.0, min(1.0, float(args.alpha)))
        overlay = Image.blend(image, color_mask, alpha=alpha)
        overlay.save(args.overlay)

    print(f"Saved colored parsing mask: {output}")


if __name__ == "__main__":
    main()
