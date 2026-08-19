import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v23 import HairFast_v23, get_parser_v23
from models.ColorTransfer_v23 import build_v23_protect_masks
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion, equal_replacer, list_image_files
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_latents


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("input/blending_dataset_v23")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("input/blending_dataset_v23_small")
USER_DATASET_SIZE_SMALL = 300

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True

USER_USE_SATD_V8 = True
USER_SATD_CHECKPOINT_V8 = "output/satd_train_v8_small/checkpoints/satd_for_infer_v8.pth"
USER_SATD_BLEND_V8 = 0.34
USER_SATD_BOUNDARY_V8 = 8
USER_EQ8_REFERENCE_BLEND_V8 = 0.0

USER_SAVE_PANELS = False
USER_INPUT_TRIPLET_SIZE = 1024
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "dataset_size": USER_DATASET_SIZE_FFHQ,
        },
        "small": {
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "output_dir": USER_OUTPUT_DIR_SMALL,
            "dataset_size": USER_DATASET_SIZE_SMALL,
        },
    }
    if USER_DATASET_PROFILE not in profiles:
        raise RuntimeError(
            f"Unsupported USER_DATASET_PROFILE={USER_DATASET_PROFILE!r}. "
            f"Choose one of: {', '.join(sorted(profiles))}."
        )
    return profiles[USER_DATASET_PROFILE]


PROFILE = resolve_dataset_profile()
ACTIVE_FACE_ROOT = PROFILE["face_root"]
ACTIVE_SHAPE_ROOT = PROFILE["shape_root"]
ACTIVE_COLOR_ROOT = PROFILE["color_root"]
ACTIVE_OUTPUT_DIR = PROFILE["output_dir"]
ACTIVE_DATASET_SIZE = PROFILE["dataset_size"]


def image_stem(file_name: str) -> str:
    return Path(file_name).stem


def role_key(role: str, stem: str) -> str:
    return f"{role}__{stem}"


def cache_name(face_name: str, shape_name: str, color_name: str) -> str:
    return f"{role_key('face', face_name)}_{role_key('shape', shape_name)}_{role_key('color', color_name)}.npz"


def find_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg in {root}")


def sample_excluding(rng: random.Random, pool: list[str], forbidden_stems: set[str]) -> str:
    candidates = [item for item in pool if image_stem(item) not in forbidden_stems]
    if not candidates:
        raise RuntimeError("No candidate left after excluding same-stem images.")
    return rng.choice(candidates)


def sample_triplets(
    face_files: list[str],
    shape_files: list[str],
    color_files: list[str],
    size: int,
    allow_reuse: bool,
    seed: int,
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    if not face_files or not shape_files or not color_files:
        raise RuntimeError("One of the input roots is empty.")
    if not allow_reuse and size > min(len(face_files), len(shape_files), len(color_files)):
        raise RuntimeError("Not enough unique images to sample without reuse.")

    face_pool = face_files.copy()
    shape_pool = shape_files.copy()
    color_pool = color_files.copy()
    triplets = []
    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_files)
            shape = sample_excluding(rng, shape_files, {image_stem(face)})
            color = sample_excluding(rng, color_files, {image_stem(face), image_stem(shape)})
        else:
            if not face_pool or not shape_pool or not color_pool:
                raise RuntimeError("The image pool has been exhausted.")
            face = rng.choice(face_pool)
            shape = sample_excluding(rng, shape_pool, {image_stem(face)})
            color = sample_excluding(rng, color_pool, {image_stem(face), image_stem(shape)})
            face_pool.remove(face)
            shape_pool.remove(shape)
            color_pool.remove(color)
        triplets.append((image_stem(face), image_stem(shape), image_stem(color)))
    return triplets


def build_model() -> HairFast_v23:
    args = get_parser_v23().parse_args([])
    args.device = USER_DEVICE
    args.save_all = False
    args.use_satd_v8 = bool(USER_USE_SATD_V8)
    args.satd_checkpoint_v8 = USER_SATD_CHECKPOINT_V8
    args.satd_blend_v8 = USER_SATD_BLEND_V8
    args.satd_boundary_v8 = USER_SATD_BOUNDARY_V8
    args.eq8_reference_blend_v8 = USER_EQ8_REFERENCE_BLEND_V8
    return HairFast_v23(args)


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return T.functional.to_tensor(image.convert("RGB"))


def resize_chw(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(image.shape[-2:]) == size:
        return image
    return F.interpolate(image.unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]


def save_panel(path: Path, face_path: Path, shape_path: Path, color_path: Path, satd_01: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (USER_INPUT_TRIPLET_SIZE, USER_INPUT_TRIPLET_SIZE)
    face = resize_chw(load_rgb(face_path), size)
    shape = resize_chw(load_rgb(shape_path), size)
    color = resize_chw(load_rgb(color_path), size)
    satd = resize_chw(satd_01.detach().cpu().clamp(0, 1), size)
    save_image(torch.cat([face, shape, color, satd], dim=2), path)


@torch.inference_mode()
def main() -> None:
    set_seed(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    face_files = list_image_files(ACTIVE_FACE_ROOT)
    shape_files = list_image_files(ACTIVE_SHAPE_ROOT)
    color_files = list_image_files(ACTIVE_COLOR_ROOT)
    triplets = sample_triplets(
        face_files,
        shape_files,
        color_files,
        ACTIVE_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
    )

    hair_fast = build_model()
    downsample_256 = BicubicDownSample(factor=4)
    dilate_erosion = DilateErosion(device=USER_DEVICE)
    cache_dir = ACTIVE_OUTPUT_DIR / "V23"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_name, shape_name, color_name in tqdm(triplets, desc="Build v23 SATD cache"):
            print(face_name, shape_name, color_name, file=f_exps, flush=True)

            face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
            shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
            color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)
            images = equal_replacer([load_rgb(face_path), load_rgb(shape_path), load_rgb(color_path)])

            images_to_name: dict[torch.Tensor, list[str]] = defaultdict(list)
            for image, name in zip(images, ("face", "shape", "color")):
                images_to_name[image].append(name)

            name_to_embed = hair_fast.embed.embedding_images(images_to_name)
            align_shape = hair_fast.align.align_images(
                "face",
                "shape",
                name_to_embed,
                use_satd_v8=USER_USE_SATD_V8,
                satd_blend_v8=USER_SATD_BLEND_V8,
                satd_boundary_v8=USER_SATD_BOUNDARY_V8,
                eq8_reference_blend_v8=USER_EQ8_REFERENCE_BLEND_V8,
            )

            color_primary_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
            color_hair = torch.where(
                color_primary_mask == 13,
                torch.ones_like(color_primary_mask),
                torch.zeros_like(color_primary_mask),
            ).float()
            _, color_hair_erode = dilate_erosion.mask(color_hair)

            i_satd, _ = hair_fast.net.generator(
                [name_to_embed["face"]["S"]],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=align_shape["latent_F_align"],
            )
            i_satd_256 = downsample_256(i_satd)
            protect_masks = build_v23_protect_masks(align_shape, size=i_satd_256.shape[-2:])

            save_latents(
                ACTIVE_OUTPUT_DIR,
                "V23",
                cache_name(face_name, shape_name, color_name),
                satd_256=((i_satd_256[0] + 1.0) * 0.5).clamp(0, 1),
                color_256=((name_to_embed["color"]["image_norm_256"][0] + 1.0) * 0.5).clamp(0, 1),
                target_hair_mask=align_shape["HM_X"][0].float().clamp(0, 1),
                reference_hair_mask=color_hair_erode[0].float().clamp(0, 1),
                hard_lock_mask=protect_masks["hard_lock"][0].float().clamp(0, 1),
                soft_lock_mask=protect_masks["soft_lock"][0].float().clamp(0, 1),
            )

            if USER_SAVE_PANELS:
                save_panel(
                    ACTIVE_OUTPUT_DIR / "panels" / f"{face_name}__{shape_name}__{color_name}.png",
                    face_path,
                    shape_path,
                    color_path,
                    ((i_satd_256[0] + 1.0) * 0.5).clamp(0, 1),
                )

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"saved {len(triplets)} triplets to {ACTIVE_OUTPUT_DIR / 'dataset.exps'}")
    print(f"v23 cache dir: {cache_dir}")


if __name__ == "__main__":
    main()
