from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


# ========================= User Config: edit here only =========================
USER_CUDA_VISIBLE_DEVICES = "0"  # GPU id string passed to CUDA_VISIBLE_DEVICES. Use "" to keep the shell setting.
USER_DEVICE = "cuda"  # Runtime device for HairFast/SATD/Blending. Usually "cuda".

USER_SOURCE_DIR = Path("images/ear")  # Source/original images. These are the images whose face/details PP should preserve.
USER_DONOR_DIR = Path("images/FFHQ_short")  # Donor images used for hair shape/color transfer.
USER_OUTPUT_DIR = Path("images/pp_dataset_e1_small")  # Output directory for generated pp_part_*.dataset files.
USER_DATASET_SIZE = 0  # Number of triplets to generate. 0 means use every source image once.
USER_PAIRING_MODE = "random"  # "random", "by_index", or "self". self uses the source as its own donor.
USER_RANDOM_SEED = 3407  # Deterministic seed for triplet sampling.

USER_CHUNK_SIZE = 32  # Number of triplets rendered before writing one dataset part. Smaller is safer for resume.
USER_MASK_BATCH_SIZE = 8  # Batch size for parsing/mask extraction after rendering.
USER_RESUME = True  # If True, existing pp_part_*.dataset files are skipped.
USER_OVERWRITE = False  # If True, regenerate existing parts even when USER_RESUME is True.

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"  # StyleGAN2 checkpoint used by HairFast.
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"  # Original rotate checkpoint.
USER_PP_CHECKPOINT = "pretrained_models/PostProcess/pp_model.pth"  # Loaded by HairFast, but bypassed during dataset generation.
USER_BLENDING_CHECKPOINT = "output/blending_train_v8/checkpoints/best.pth"  # Blending v8 checkpoint.
USER_USE_SATD_V8 = True  # Enable SATD v8 in Alignment_v8 before PP.
USER_SATD_CHECKPOINT_V8 = "output/satd_train_v8_3000/checkpoints/satd_for_infer_v8.pth"  # SATD v8 weights.
USER_SATD_BLEND_V8 = 0.34  # SATD feature correction strength.
USER_SATD_BOUNDARY_V8 = 8  # Boundary width used by SATD cleanup masks.
USER_EQ8_REFERENCE_BLEND_V8 = 0.0  # Extra reference latent blend used by Alignment_v8.
USER_BLEND_COLOR_STRENGTH_V8 = 1.0  # Blending encoder color latent strength.
USER_DISABLE_EXACT_HAIR_COLOR_MATCH_V8 = False  # Keep False to use current Blending v8 exact color matching if available.

USER_SMOOTH = 5  # Dilation/erosion radius used by original HairFast masks.
USER_IMAGE_SIZE = 1024  # Training full-resolution image size expected by StyleGAN/PP.
# ============================================================================ 


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v8 import HairFast_v8, get_parser_v8
from models.Net import get_segmentation
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion, equal_replacer
from utils.train import seed_everything


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageException(Exception):
    def __init__(self, image: torch.Tensor):
        super().__init__("Captured image before PP")
        self.image = image


@torch.inference_mode()
def render_pre_pp_author_blend_v8(
    hair_fast: HairFast_v8,
    args,
    face_path: Path,
    shape_path: Path,
    color_path: Path,
) -> dict[str, torch.Tensor]:
    """
    Render the intermediate images used by PP training.

    This intentionally bypasses Blending_v8.blend_images() because some local v8
    variants add post-blend RCI modules with checkpoint-dependent channel
    assumptions. Dataset generation keeps both SATD/align output and the
    blending-encoder color transfer output before PostProcessModel.
    """
    face, shape, color = equal_replacer(
        [
            load_image_01(face_path, args.image_size),
            load_image_01(shape_path, args.image_size),
            load_image_01(color_path, args.image_size),
        ]
    )

    images_to_name = defaultdict(list)
    for image, name in zip((face, shape, color), ("face", "shape", "color")):
        images_to_name[image].append(name)

    name_to_embed = hair_fast.embed.embedding_images(images_to_name)
    align_kwargs = {
        "use_satd_v8": args.use_satd_v8,
        "satd_blend_v8": args.satd_blend_v8,
        "satd_boundary_v8": args.satd_boundary_v8,
        "eq8_reference_blend_v8": args.eq8_reference_blend_v8,
    }
    align_shape = hair_fast.align.align_images("face", "shape", name_to_embed, **align_kwargs)
    if shape is not color:
        align_color = hair_fast.align.shape_module("face", "color", name_to_embed, **align_kwargs)
    else:
        align_color = align_shape

    blend = hair_fast.blend
    I_1 = name_to_embed["face"]["image_norm_256"]
    I_2 = name_to_embed["shape"]["image_norm_256"]
    I_3 = name_to_embed["color"]["image_norm_256"]

    mask_de = blend.dilate_erosion.hair_from_mask(
        torch.cat([name_to_embed[x]["mask"] for x in ["face", "color"]], dim=0)
    )
    HM_1D, _ = mask_de[0][0].unsqueeze(0), mask_de[1][0].unsqueeze(0)
    HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

    latent_S_1 = name_to_embed["face"]["S"]
    latent_F_align = align_shape["latent_F_align"]
    HM_X = align_color["HM_X"]
    latent_S_3 = name_to_embed["color"]["S"]
    HM_XD, _ = blend.dilate_erosion.mask(HM_X)
    target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

    I_satd, _ = hair_fast.net.generator(
        [latent_S_1],
        input_is_latent=True,
        return_latents=False,
        start_layer=4,
        end_layer=8,
        layer_in=latent_F_align,
    )

    if I_1 is not I_3 or I_1 is not I_2:
        S_blend_6_18_raw = blend.blending_encoder(
            latent_S_1[:, 6:],
            latent_S_3[:, 6:],
            I_1 * target_mask,
            I_3 * HM_3E,
        )
        blend_strength = float(getattr(blend, "blend_color_strength", 1.0))
        S_blend_6_18 = latent_S_1[:, 6:] + blend_strength * (S_blend_6_18_raw - latent_S_1[:, 6:])
        S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
    else:
        S_blend = latent_S_1

    I_blend, _ = hair_fast.net.generator(
        [S_blend],
        input_is_latent=True,
        return_latents=False,
        start_layer=4,
        end_layer=8,
        layer_in=latent_F_align,
    )
    return {
        "satd": ((I_satd[0] + 1) / 2).clamp(0, 1),
        "blending": ((I_blend[0] + 1) / 2).clamp(0, 1),
    }


def str2path(value):
    return None if value in {None, "", "None"} else Path(value)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


def build_parser():
    parser = argparse.ArgumentParser(description="Generate PP e1 dataset with SATD v8 + Blending v8.")
    parser.add_argument("--source_dir", type=str2path, default=USER_SOURCE_DIR)
    parser.add_argument("--donor_dir", type=str2path, default=USER_DONOR_DIR)
    parser.add_argument("--output_dir", type=Path, default=USER_OUTPUT_DIR)
    parser.add_argument("--dataset_size", type=int, default=USER_DATASET_SIZE)
    parser.add_argument("--pairing_mode", choices=("random", "by_index", "self"), default=USER_PAIRING_MODE)
    parser.add_argument("--seed", type=int, default=USER_RANDOM_SEED)
    parser.add_argument("--chunk_size", type=int, default=USER_CHUNK_SIZE)
    parser.add_argument("--mask_batch_size", type=int, default=USER_MASK_BATCH_SIZE)
    parser.add_argument("--resume", type=str2bool, default=USER_RESUME)
    parser.add_argument("--overwrite", type=str2bool, default=USER_OVERWRITE)
    parser.add_argument("--device", type=str, default=USER_DEVICE)
    parser.add_argument("--image_size", type=int, default=USER_IMAGE_SIZE)
    parser.add_argument("--stylegan_ckpt", type=str, default=USER_STYLEGAN_CKPT)
    parser.add_argument("--rotate_checkpoint", type=str, default=USER_ROTATE_CKPT)
    parser.add_argument("--pp_checkpoint", type=str, default=USER_PP_CHECKPOINT)
    parser.add_argument("--blending_checkpoint", type=str, default=USER_BLENDING_CHECKPOINT)
    parser.add_argument("--use_satd_v8", type=str2bool, default=USER_USE_SATD_V8)
    parser.add_argument("--satd_checkpoint_v8", type=str, default=USER_SATD_CHECKPOINT_V8)
    parser.add_argument("--satd_blend_v8", type=float, default=USER_SATD_BLEND_V8)
    parser.add_argument("--satd_boundary_v8", type=int, default=USER_SATD_BOUNDARY_V8)
    parser.add_argument("--eq8_reference_blend_v8", type=float, default=USER_EQ8_REFERENCE_BLEND_V8)
    parser.add_argument("--blend_color_strength_v8", type=float, default=USER_BLEND_COLOR_STRENGTH_V8)
    parser.add_argument("--disable_exact_hair_color_match_v8", type=str2bool, default=USER_DISABLE_EXACT_HAIR_COLOR_MATCH_V8)
    parser.add_argument("--smooth", type=int, default=USER_SMOOTH)
    return parser


def list_images(root: Path) -> list[str]:
    if root is None or not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    files = sorted(path.name for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise RuntimeError(f"No images found in: {root}")
    return files


def build_experiment_plan(args, source_files: list[str], donor_files: list[str]) -> list[dict[str, str]]:
    rng = random.Random(args.seed)
    count = len(source_files) if args.dataset_size <= 0 else min(args.dataset_size, len(source_files))
    plan: list[dict[str, str]] = []

    for idx, source_name in enumerate(source_files[:count]):
        if args.pairing_mode == "self":
            shape_name = source_name
            color_name = source_name
            donor_tag = "self"
        elif args.pairing_mode == "by_index":
            shape_name = donor_files[idx % len(donor_files)]
            color_name = shape_name
            donor_tag = Path(shape_name).stem
        else:
            shape_name = rng.choice(donor_files)
            color_name = rng.choice(donor_files)
            donor_tag = f"{Path(shape_name).stem}_{Path(color_name).stem}"

        plan.append(
            {
                "source_name": source_name,
                "shape_name": shape_name,
                "color_name": color_name,
                "part_name": f"{Path(source_name).stem}__{donor_tag}.png",
            }
        )
    return plan


def resolve_triplet_paths(args, item: dict[str, str]) -> tuple[Path, Path, Path]:
    source_path = args.source_dir / item["source_name"]
    if args.pairing_mode == "self":
        shape_path = args.source_dir / item["shape_name"]
        color_path = args.source_dir / item["color_name"]
    else:
        shape_path = args.donor_dir / item["shape_name"]
        color_path = args.donor_dir / item["color_name"]
    return source_path, shape_path, color_path


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def dataset_part_has_e1_schema(path: Path) -> bool:
    required = {"satd", "blending", "shape_path", "color_path"}
    try:
        loaded = torch.load(path, map_location="cpu")
    except Exception:
        return False
    return isinstance(loaded, list) and bool(loaded) and required.issubset(loaded[0].keys())


def hairfast_capture_pre_pp(hair_fast):
    class CaptureDownsample(nn.Module):
        def forward(self, image):
            captured = ((image[0] + 1) / 2).clamp(0, 1)
            raise ImageException(captured)

    original_blend = hair_fast.blend.blend_images

    def wrapped_blend(*args, **kwargs):
        try:
            original_blend(*args, **kwargs)
        except ImageException as error:
            return error.image

    hair_fast.blend.downsample_256 = CaptureDownsample()
    hair_fast.blend.blend_images = wrapped_blend


def load_image_01(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.BILINEAR)
        return T.functional.to_tensor(image)


def label_mask(parsing: torch.Tensor, label: int) -> torch.Tensor:
    return (parsing.long() == int(label)).float()


class DatasetItemBuilder:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.downsample_512 = BicubicDownSample(factor=2).to(self.device)
        self.downsample_256 = BicubicDownSample(factor=4).to(self.device)
        self.dilate_erosion = DilateErosion(dilate_erosion=args.smooth, device=str(self.device))
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    @torch.no_grad()
    def parse(self, images_01: torch.Tensor) -> torch.Tensor:
        images_512 = self.downsample_512(images_01).clamp(0, 1)
        images_512 = self.to_bisenet(images_512)
        return torch.cat([get_segmentation(image.unsqueeze(0)) for image in images_512], dim=0).long()

    @torch.no_grad()
    def build(self, chunk_plan: list[dict[str, str]], source_dir: Path, rendered_dir: Path) -> list[dict[str, object]]:
        items: list[dict[str, object]] = []
        for start in range(0, len(chunk_plan), self.args.mask_batch_size):
            batch_plan = chunk_plan[start:start + self.args.mask_batch_size]
            source_full = torch.stack(
                [load_image_01(source_dir / item["source_name"], self.args.image_size) for item in batch_plan],
                dim=0,
            ).to(self.device)
            target_full = torch.stack(
                [load_image_01(rendered_dir / item["part_name"], self.args.image_size) for item in batch_plan],
                dim=0,
            ).to(self.device)
            satd_full = torch.stack(
                [load_image_01(rendered_dir / f"satd__{item['part_name']}", self.args.image_size) for item in batch_plan],
                dim=0,
            ).to(self.device)

            source_256 = self.downsample_256(source_full).clamp(0, 1)
            target_256 = self.downsample_256(target_full).clamp(0, 1)
            satd_256 = self.downsample_256(satd_full).clamp(0, 1)
            source_parsing = self.parse(source_full)
            target_parsing = self.parse(target_full)
            source_hair = label_mask(source_parsing, 13)
            target_hair = label_mask(target_parsing, 13)
            source_hair_d, _ = self.dilate_erosion.mask(source_hair)
            target_hair_d, target_hair_e = self.dilate_erosion.mask(target_hair)
            target_mask = ((1 - source_hair_d) * (1 - target_hair_d)).clamp(0, 1)

            for idx, item in enumerate(batch_plan):
                items.append(
                    {
                        "source_path": str(source_dir / item["source_name"]),
                        "shape_path": item.get("shape_path", ""),
                        "color_path": item.get("color_path", ""),
                        "source_name": item["source_name"],
                        "shape_name": item["shape_name"],
                        "color_name": item["color_name"],
                        "target": target_256[idx].cpu(),
                        "satd": satd_256[idx].cpu(),
                        "blending": target_256[idx].cpu(),
                        "target_mask": target_mask[idx].cpu(),
                        "HT_E": target_hair_e[idx].cpu(),
                        "source_parsing": source_parsing[idx].cpu(),
                        "target_parsing": target_parsing[idx].cpu(),
                        "target_hair_mask": target_hair[idx].cpu(),
                    }
                )

            del source_full, target_full, satd_full, source_256, target_256, satd_256
            del source_parsing, target_parsing, source_hair, target_hair
            del source_hair_d, target_hair_d, target_hair_e, target_mask
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        return items


def build_model_args(args):
    parser = get_parser_v8()
    model_args = parser.parse_args([])
    model_args.device = args.device
    model_args.size = args.image_size
    model_args.ckpt = args.stylegan_ckpt
    model_args.rotate_checkpoint = args.rotate_checkpoint
    model_args.pp_checkpoint = args.pp_checkpoint
    model_args.blending_checkpoint = args.blending_checkpoint
    model_args.smooth = args.smooth
    model_args.use_satd_v8 = bool(args.use_satd_v8)
    model_args.satd_checkpoint_v8 = args.satd_checkpoint_v8
    model_args.satd_blend_v8 = args.satd_blend_v8
    model_args.satd_boundary_v8 = args.satd_boundary_v8
    model_args.eq8_reference_blend_v8 = args.eq8_reference_blend_v8
    model_args.blend_color_strength_v8 = args.blend_color_strength_v8
    model_args.disable_exact_hair_color_match_v8 = bool(args.disable_exact_hair_color_match_v8)
    return model_args


def main(args):
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")
    if args.mask_batch_size <= 0:
        raise ValueError("--mask_batch_size must be positive")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_files = list_images(args.source_dir)
    donor_files = source_files if args.pairing_mode == "self" else list_images(args.donor_dir)
    plan = build_experiment_plan(args, source_files, donor_files)
    save_json(args.output_dir / "manifest.json", {"config": vars(args), "plan": plan})

    print(
        f"Generating PP e1 dataset: source={args.source_dir}, donor={args.donor_dir}, "
        f"items={len(plan)}, output={args.output_dir}"
    )

    model_args = build_model_args(args)
    hair_fast = HairFast_v8(model_args)
    item_builder = DatasetItemBuilder(args)

    total_parts = (len(plan) + args.chunk_size - 1) // args.chunk_size
    for part_idx, start in enumerate(range(0, len(plan), args.chunk_size), start=1):
        part_path = args.output_dir / f"pp_part_{part_idx:05d}.dataset"
        if args.resume and part_path.exists() and not args.overwrite:
            if dataset_part_has_e1_schema(part_path):
                print(f"Skip existing part {part_idx}/{total_parts}: {part_path.name}")
                continue
            print(f"Regenerate old-schema part {part_idx}/{total_parts}: {part_path.name}")

        chunk_plan = plan[start:start + args.chunk_size]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            for item in tqdm(chunk_plan, desc=f"Render part {part_idx}/{total_parts}"):
                source_path, shape_path, color_path = resolve_triplet_paths(args, item)
                item["source_path"] = str(source_path)
                item["shape_path"] = str(shape_path)
                item["color_path"] = str(color_path)
                rendered = render_pre_pp_author_blend_v8(hair_fast, args, source_path, shape_path, color_path)
                save_image(rendered["blending"], temp_path / item["part_name"])
                save_image(rendered["satd"], temp_path / f"satd__{item['part_name']}")

            dataset_items = item_builder.build(chunk_plan, args.source_dir, temp_path)
            tmp_part = part_path.with_suffix(".tmp")
            torch.save(dataset_items, tmp_part)
            os.replace(tmp_part, part_path)
            save_json(args.output_dir / "progress.json", {"last_written_part": part_idx, "total_parts": total_parts})

    print("Done.")


if __name__ == "__main__":
    main(build_parser().parse_args())
