import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment import Alignment
from models.Embedding import Embedding
from models.Net import Net
from models.CtrlHair.global_value_utils import PARSING_COLOR_LIST
from utils.image_utils import equal_replacer
from utils.shape_predictor import align_face


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the baseline HairFast intermediate image after hairstyle shape transfer, "
            "before color transfer and before post-process."
        )
    )
    parser.add_argument("source", type=Path, help="Source face image.")
    parser.add_argument("shape", type=Path, help="Hairstyle reference image.")
    parser.add_argument("--output", type=Path, default=Path("output/baseline_shape_transfer_only.png"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--align", action="store_true", help="Run the author's face alignment before inference.")
    parser.add_argument(
        "--resize_inputs",
        action="store_true",
        help="Resize inputs to --size without FFHQ face alignment. This is only for already-centered crops.",
    )
    parser.add_argument("--save_masks", action="store_true", help="Save the target hair mask used by alignment.")
    parser.add_argument("--mask_output", type=Path, default=Path("output/baseline_shape_transfer_mask.png"))
    parser.add_argument("--source_hair_mask_output", type=Path, default=Path("output/baseline_source_hair_mask.png"))
    parser.add_argument("--remove_mask_output", type=Path, default=Path("output/baseline_remove_mask.png"))
    parser.add_argument("--debug_dir", type=Path, default=None, help="Optional directory to save the exact input tensors used.")
    parser.add_argument(
        "--allow_mask_like_shape",
        action="store_true",
        help="Allow shape input that looks like a colored parsing mask.",
    )

    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--ckpt", type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--save_all_dir", type=Path, default=Path("output"))
    parser.add_argument("--mixing", type=float, default=0.95)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--rotate_checkpoint", type=str, default="pretrained_models/Rotate/rotate_best.pth")
    return parser


def load_image_tensor(path: Path, size: int = 1024, resize: bool = False) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = Image.open(path).convert("RGB")
    if resize and image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def safe_equal_replacer(images: list[torch.Tensor]) -> list[torch.Tensor]:
    if all(tuple(image.shape) == tuple(images[0].shape) for image in images):
        return equal_replacer(images)

    normalized = []
    target_size = tuple(images[0].shape[-2:])
    for image in images:
        if image.dtype is torch.uint8:
            image = image.float() / 255
        if tuple(image.shape[-2:]) != target_size:
            image = F.interpolate(image.unsqueeze(0).float(), size=target_size, mode="bilinear", align_corners=False)[0]
        normalized.append(image)
    return equal_replacer(normalized)


def validate_input_shapes(images: list[torch.Tensor], args) -> None:
    shapes = [tuple(image.shape) for image in images]
    if args.align or args.resize_inputs:
        return
    if len(set(shapes)) != 1:
        raise ValueError(
            f"Input image tensor shapes differ: {shapes}. "
            "Use --align for raw photos, or --resize_inputs only if both images are already centered face crops."
        )


def save_image_01(tensor: torch.Tensor, path: Path) -> None:
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def looks_like_color_mask(tensor: torch.Tensor, max_mean_distance: float = 8.0) -> bool:
    image = tensor.detach().cpu().float().clamp(0, 1)
    if image.ndim == 4:
        image = image[0]
    rgb = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.int16)
    pixels = rgb.reshape(-1, 3)
    palette = np.asarray(PARSING_COLOR_LIST[:19], dtype=np.int16)
    distances = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    mean_distance = np.sqrt(distances.min(axis=1)).mean()
    unique_count = len(np.unique(pixels[:: max(1, len(pixels) // 20000)], axis=0))
    return mean_distance <= max_mean_distance and unique_count < 256


def save_debug_inputs(source: torch.Tensor, shape: torch.Tensor, debug_dir: Path) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_image_01(source, debug_dir / "source_used.png")
    save_image_01(shape, debug_dir / "shape_used.png")


def save_mask(mask: torch.Tensor, path: Path) -> None:
    if mask.ndim == 4:
        mask = mask[0, 0]
    elif mask.ndim == 3:
        mask = mask[0]
    mask = mask.detach().cpu().float().clamp(0, 1)
    array = (mask.numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    source = load_image_tensor(args.source, size=args.size, resize=args.resize_inputs)
    shape = load_image_tensor(args.shape, size=args.size, resize=args.resize_inputs)

    images = [source, shape]
    validate_input_shapes(images, args)
    if args.align:
        images = align_face(images)
    images = safe_equal_replacer(images)
    source, shape = images
    if args.debug_dir is not None:
        save_debug_inputs(source, shape, args.debug_dir)
    if looks_like_color_mask(shape) and not args.allow_mask_like_shape:
        raise ValueError(
            "The shape reference looks like a colored parsing mask, not a real hairstyle photo. "
            "Pass the original hairstyle reference photo as the second argument. "
            "Use --allow_mask_like_shape only if this is intentional."
        )

    net = Net(args)
    embed = Embedding(args, net=net)
    align = Alignment(args, embed.get_e4e_embed, net=net)

    images_to_name = defaultdict(list)
    images_to_name[source].append("face")
    images_to_name[shape].append("shape")

    name_to_embed = embed.embedding_images(images_to_name, exp_name="baseline_shape_transfer_only")
    align_shape = align.align_images("face", "shape", name_to_embed, exp_name="baseline_shape_transfer_only")

    generated, _ = net.generator(
        [name_to_embed["face"]["S"]],
        input_is_latent=True,
        return_latents=False,
        start_layer=4,
        end_layer=8,
        layer_in=align_shape["latent_F_align"],
    )
    result_01 = ((generated + 1) / 2).clamp(0, 1)
    save_image_01(result_01, args.output)

    if args.save_masks:
        source_hair_mask = torch.where(
            name_to_embed["face"]["mask"] == 13,
            torch.ones_like(name_to_embed["face"]["mask"]),
            torch.zeros_like(name_to_embed["face"]["mask"]),
        ).float()
        target_hair_mask = align_shape["HM_X"].float().clamp(0, 1)
        remove_mask = (source_hair_mask * (1.0 - target_hair_mask)).clamp(0, 1)

        save_mask(target_hair_mask, args.mask_output)
        save_mask(source_hair_mask, args.source_hair_mask_output)
        save_mask(remove_mask, args.remove_mask_output)

    print(f"Saved baseline shape-transfer-only image: {args.output}")
    if args.save_masks:
        print(f"Saved target hair mask: {args.mask_output}")
        print(f"Saved source hair mask: {args.source_hair_mask_output}")
        print(f"Saved remove mask: {args.remove_mask_output}")


if __name__ == "__main__":
    main()
