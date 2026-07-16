import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import dlib
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment import Alignment
from models.Blending import Blending
from models.Embedding import Embedding
from models.Net import Net, get_segmentation
from utils.drive import open_url
from utils.image_utils import equal_replacer
from utils.shape_predictor import get_landmark_from_tensors


OUTPUT_NAMES = {
    "aligned_source": "aligned_source.png",
    "aligned_shape": "aligned_shape.png",
    "aligned_color": "aligned_color.png",
    "rotated_shape": "shape_rotate_to_source.png",
    "rotated_color": "color_rotate_to_source.png",
    "shape_only": "shape_only_transfer.png",
    "shape_only_raw": "shape_only_alignment_raw.png",
    "source_mask": "source_hair_mask.png",
    "rotated_shape_mask": "rotated_shape_hair_mask.png",
    "target_shape_mask": "shape_transfer_target_hair_mask.png",
    "diff_mask": "source_vs_rotated_shape_hair_diff_mask.png",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export baseline HairFast alignment/rotation artifacts for one "
            "source, hairstyle reference, and hair-color reference triplet."
        )
    )
    parser.add_argument("source", type=Path, help="Source face photo.")
    parser.add_argument("shape", type=Path, help="Hairstyle reference photo.")
    parser.add_argument("color", type=Path, help="Hair-color reference photo.")
    parser.add_argument("--output_dir", type=Path, default=Path("output/baseline_rotate_artifacts"))
    parser.add_argument(
        "--no_align",
        action="store_false",
        dest="align",
        help="Skip FFHQ face alignment. Use only for already aligned 1024x1024 crops.",
    )
    parser.set_defaults(align=True)
    parser.add_argument(
        "--resize_inputs",
        action="store_true",
        help="When --no_align is used, resize inputs to --size before inference.",
    )
    parser.add_argument(
        "--align_padding_mode",
        choices=("edge", "reflect", "constant"),
        default="edge",
        help="Padding mode for FFHQ alignment. Baseline's blur padding is intentionally not used.",
    )
    parser.add_argument(
        "--mask_size",
        type=int,
        default=1024,
        choices=(256, 1024),
        help="Resolution for exported black/white masks.",
    )

    # Baseline HairFast arguments used by Net, Embedding, and Alignment.
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--ckpt", type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--save_all_dir", type=Path, default=Path("output"))
    parser.add_argument("--mixing", type=float, default=0.95)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--rotate_checkpoint", type=str, default="pretrained_models/Rotate/rotate_best.pth")
    parser.add_argument("--blending_checkpoint", type=str, default="pretrained_models/Blending/checkpoint.pth")
    parser.add_argument("--pp_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    return parser


def load_image_tensor(path: Path, size: int, resize: bool = False) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = Image.open(path).convert("RGB")
    if resize and image.size != (size, size):
        image = image.resize((size, size), Image.BILINEAR)
    return T.functional.to_tensor(image)


def save_image_01(tensor: torch.Tensor, path: Path) -> None:
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.detach().cpu().float().clamp(0, 1)
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def save_mask_bw(mask: torch.Tensor, path: Path, size: int) -> None:
    if mask.ndim == 2:
        mask = mask[None, None]
    elif mask.ndim == 3:
        mask = mask[None] if mask.size(0) == 1 else mask[:1][None]
    elif mask.ndim == 4:
        mask = mask[:, :1]
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    mask = mask.detach().float().cpu()
    if tuple(mask.shape[-2:]) != (size, size):
        mask = F.interpolate(mask, size=(size, size), mode="nearest")
    array = (mask[0, 0].clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="L").save(path)


def load_shape_predictor() -> dlib.shape_predictor:
    predictor_path = Path("pretrained_models/ShapeAdaptor/shape_predictor_68_face_landmarks.dat")
    if not predictor_path.is_file():
        print("Downloading Shape Predictor")
        data_io = open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx")
        predictor_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictor_path, "wb") as file:
            file.write(data_io.getbuffer())
    return dlib.shape_predictor(str(predictor_path))


def align_face_no_blur(
    tensors: list[torch.Tensor],
    padding_mode: str,
    output_size: int = 1024,
    transform_size: int = 4096,
) -> list[torch.Tensor]:
    predictor = load_shape_predictor()
    images, landmarks = get_landmark_from_tensors(tensors, predictor)
    aligned = []

    for image, lm in zip(images, landmarks):
        image = image.convert("RGB")
        lm_eye_left = lm[36:42]
        lm_eye_right = lm[42:48]
        lm_mouth_outer = lm[48:60]

        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        x_vec = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x_vec /= np.hypot(*x_vec)
        x_vec *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y_vec = np.flipud(x_vec) * [-1, 1]
        center = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([center - x_vec - y_vec, center - x_vec + y_vec, center + x_vec + y_vec, center + x_vec - y_vec])
        qsize = np.hypot(*x_vec) * 2

        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            resized = (int(np.rint(float(image.size[0]) / shrink)), int(np.rint(float(image.size[1]) / shrink)))
            image = image.resize(resized, Image.LANCZOS)
            quad /= shrink
            qsize /= shrink

        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        crop = (
            max(crop[0] - border, 0),
            max(crop[1] - border, 0),
            min(crop[2] + border, image.size[0]),
            min(crop[3] + border, image.size[1]),
        )
        if crop[2] - crop[0] < image.size[0] or crop[3] - crop[1] < image.size[1]:
            image = image.crop(crop)
            quad -= crop[0:2]

        pad = (
            int(np.floor(min(quad[:, 0]))),
            int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))),
        )
        pad = (
            max(-pad[0] + border, 0),
            max(-pad[1] + border, 0),
            max(pad[2] - image.size[0] + border, 0),
            max(pad[3] - image.size[1] + border, 0),
        )
        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            image_np = np.asarray(image)
            pad_width = ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0))
            if padding_mode == "constant":
                image_np = np.pad(image_np, pad_width, mode="constant", constant_values=0)
            else:
                image_np = np.pad(image_np, pad_width, mode=padding_mode)
            image = Image.fromarray(image_np.astype(np.uint8), mode="RGB")
            quad += pad[:2]

        image = image.transform(
            (transform_size, transform_size),
            Image.QUAD,
            (quad + 0.5).flatten(),
            Image.BILINEAR,
        )
        if output_size < transform_size:
            image = image.resize((output_size, output_size), Image.LANCZOS)
        aligned.append(T.functional.to_tensor(image).clamp(0, 1))

    return aligned


def safe_equal_replacer(images: list[torch.Tensor]) -> list[torch.Tensor]:
    if all(tuple(image.shape) == tuple(images[0].shape) for image in images):
        return equal_replacer(images)

    normalized = []
    target_size = tuple(images[0].shape[-2:])
    for image in images:
        if image.dtype is torch.uint8:
            image = image.float() / 255.0
        if tuple(image.shape[-2:]) != target_size:
            image = F.interpolate(
                image.unsqueeze(0).float(),
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )[0]
        normalized.append(image)
    return equal_replacer(normalized)


def prepare_images(args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = [
        load_image_tensor(args.source, args.size, resize=args.resize_inputs and not args.align),
        load_image_tensor(args.shape, args.size, resize=args.resize_inputs and not args.align),
        load_image_tensor(args.color, args.size, resize=args.resize_inputs and not args.align),
    ]

    if args.align:
        images = align_face_no_blur(images, padding_mode=args.align_padding_mode, output_size=args.size)
    elif not args.resize_inputs:
        bad_shapes = [tuple(image.shape) for image in images if tuple(image.shape[-2:]) != (args.size, args.size)]
        if bad_shapes:
            raise ValueError(
                f"Found non-{args.size}x{args.size} inputs: {bad_shapes}. "
                "Use default alignment, or pass --resize_inputs with --no_align."
            )

    return tuple(safe_equal_replacer(images))


def build_name_to_embed(
    embed: Embedding,
    source: torch.Tensor,
    shape: torch.Tensor,
    color: torch.Tensor,
):
    images_to_name = defaultdict(list)
    for image, name in ((source, "face"), (shape, "shape"), (color, "color")):
        images_to_name[image].append(name)
    return embed.embedding_images(images_to_name, exp_name="baseline_rotate_artifacts")


@torch.inference_mode()
def rotate_reference_to_source(
    align: Alignment,
    net: Net,
    name_to_embed,
    reference_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    source_w = name_to_embed["face"]["W"]
    reference_w = name_to_embed[reference_name]["W"]

    rotate_to = align.rotate_model(reference_w[:, :6], source_w[:, :6])
    rotate_to = torch.cat((rotate_to, reference_w[:, 6:]), dim=1)
    rotated_tanh, _ = net.generator([rotate_to], input_is_latent=True, return_latents=False)

    rotated_01 = ((rotated_tanh + 1.0) / 2.0).clamp(0, 1)
    rotated_for_seg = align.to_bisenet(rotated_01)
    rotated_mask = get_segmentation(rotated_for_seg)
    return rotated_01, rotated_mask


@torch.inference_mode()
def render_shape_only(
    align: Alignment,
    net: Net,
    name_to_embed,
) -> torch.Tensor:
    align_shape = align.align_images(
        "face",
        "shape",
        name_to_embed,
        exp_name="baseline_rotate_artifacts",
    )
    generated, _ = net.generator(
        [name_to_embed["face"]["S"]],
        input_is_latent=True,
        return_latents=False,
        start_layer=4,
        end_layer=8,
        layer_in=align_shape["latent_F_align"],
    )
    return ((generated + 1.0) / 2.0).clamp(0, 1)


@torch.inference_mode()
def render_shape_only_with_source_color(
    align: Alignment,
    blend: Blending,
    name_to_embed,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape_only_embed = dict(name_to_embed)
    shape_only_embed["color"] = shape_only_embed["face"]
    align_shape = align.align_images(
        "face",
        "shape",
        shape_only_embed,
        exp_name="baseline_shape_only_source_color",
    )
    align_color = align.shape_module(
        "face",
        "color",
        shape_only_embed,
        exp_name="baseline_shape_only_source_color",
    )
    final_image = blend.blend_images(
        align_shape,
        align_color,
        shape_only_embed,
        exp_name="baseline_shape_only_source_color",
    )
    return final_image.unsqueeze(0).clamp(0, 1), align_shape["HM_X"]


def hair_mask_from_label(label_mask: torch.Tensor) -> torch.Tensor:
    return (label_mask == 13).float()


@torch.inference_mode()
def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source, shape, color = prepare_images(args)
    save_image_01(source, args.output_dir / OUTPUT_NAMES["aligned_source"])
    save_image_01(shape, args.output_dir / OUTPUT_NAMES["aligned_shape"])
    save_image_01(color, args.output_dir / OUTPUT_NAMES["aligned_color"])

    net = Net(args)
    embed = Embedding(args, net=net)
    align = Alignment(args, embed.get_e4e_embed, net=net)
    blend = Blending(args, net=net)

    name_to_embed = build_name_to_embed(embed, source, shape, color)

    rotated_shape, rotated_shape_label = rotate_reference_to_source(align, net, name_to_embed, "shape")
    rotated_color, _ = rotate_reference_to_source(align, net, name_to_embed, "color")
    shape_only_raw = render_shape_only(align, net, name_to_embed)
    shape_only, target_shape_hair = render_shape_only_with_source_color(align, blend, name_to_embed)

    save_image_01(rotated_shape, args.output_dir / OUTPUT_NAMES["rotated_shape"])
    save_image_01(rotated_color, args.output_dir / OUTPUT_NAMES["rotated_color"])
    save_image_01(shape_only, args.output_dir / OUTPUT_NAMES["shape_only"])
    save_image_01(shape_only_raw, args.output_dir / OUTPUT_NAMES["shape_only_raw"])

    source_hair = hair_mask_from_label(name_to_embed["face"]["mask"])
    rotated_shape_hair = hair_mask_from_label(rotated_shape_label)
    diff_hair = (source_hair - rotated_shape_hair).abs().clamp(0, 1)

    save_mask_bw(source_hair, args.output_dir / OUTPUT_NAMES["source_mask"], args.mask_size)
    save_mask_bw(rotated_shape_hair, args.output_dir / OUTPUT_NAMES["rotated_shape_mask"], args.mask_size)
    save_mask_bw(target_shape_hair, args.output_dir / OUTPUT_NAMES["target_shape_mask"], args.mask_size)
    save_mask_bw(diff_hair, args.output_dir / OUTPUT_NAMES["diff_mask"], args.mask_size)

    print(f"Saved artifacts to: {args.output_dir}")
    for name in OUTPUT_NAMES.values():
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
