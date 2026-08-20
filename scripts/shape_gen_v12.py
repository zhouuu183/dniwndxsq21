from __future__ import annotations

import os
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

# ========================= 用户配置区域：只改这里 =========================
USER_CUDA_VISIBLE_DEVICES = "0"
USER_DEVICE = "cuda"

USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ALPHA_ROOT_FFHQ = Path("")
USER_OUTPUT_DIR_FFHQ = Path("images/shape_dataset_v12_3000")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ALPHA_ROOT_SMALL = Path("images/FFHQ_fringe_alpha")
USER_OUTPUT_DIR_SMALL = Path("images/shape_dataset_v12_small")
USER_DATASET_SIZE_SMALL = 300

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True
USER_RESIZE_INPUT_TO_1024 = True

USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_ENCODER_BATCH_SIZE = 3
USER_MIXING = 0.95
USER_SMOOTH = 5

USER_TOPOLOGY_BOUNDARY_WIDTH_V12 = 9
USER_ALPHA_FALLBACK_BLUR_V12 = 9
USER_BANG_TOP_RATIO_V12 = 0.56
# ========================================================================


if USER_CUDA_VISIBLE_DEVICES:
    os.environ["CUDA_VISIBLE_DEVICES"] = USER_CUDA_VISIBLE_DEVICES

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment_v12 import Alignment_v12
from models.Embedding import Embedding
from models.Net import Net
from models.ShapeAdapter_v12 import V12_MASK_KEYS
from utils.image_utils import equal_replacer, list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "shape_alpha_root": USER_SHAPE_ALPHA_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "dataset_size": USER_DATASET_SIZE_FFHQ,
        },
        "small": {
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "shape_alpha_root": USER_SHAPE_ALPHA_ROOT_SMALL,
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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_SHAPE_ALPHA_ROOT = _DATASET_CFG["shape_alpha_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_DATASET_SIZE = _DATASET_CFG["dataset_size"]


def _optional_root(root: Path | None) -> Path | None:
    if root is None:
        return None
    root = Path(root)
    if str(root) in ("", "."):
        return None
    return root


def _assert_unique_stems(images: list[str], label: str):
    counts: dict[str, int] = {}
    for item in images:
        stem = Path(item).stem
        counts[stem] = counts.get(stem, 0) + 1
    duplicates = [stem for stem, count in counts.items() if count > 1]
    if duplicates:
        raise RuntimeError(f"{label} contains duplicate stems: {', '.join(sorted(duplicates)[:10])}")


def _sample_excluding_stems(rng: random.Random, candidates: list[str], forbidden_stems: set[str]) -> str:
    valid = [item for item in candidates if Path(item).stem not in forbidden_stems]
    if not valid:
        raise RuntimeError("No valid candidates left after excluding duplicate stems.")
    return rng.choice(valid)


def sample_triplets(
    face_images: list[str],
    shape_images: list[str],
    color_images: list[str],
    size: int,
    allow_reuse: bool,
    seed: int,
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    triplets = []
    if not face_images or not shape_images or not color_images:
        raise RuntimeError("Face/shape/color roots must all contain images.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = color_images.copy()
    if not allow_reuse and size > min(len(face_pool), len(shape_pool), len(color_pool)):
        raise RuntimeError("Not enough unique images to sample all triplets without reuse.")

    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_images)
            shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_images, {Path(face).stem, Path(shape).stem})
        else:
            face = rng.choice(face_pool)
            shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_pool, {Path(face).stem, Path(shape).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
            color_pool.remove(color)
        triplets.append((Path(face).stem, Path(shape).stem, Path(color).stem))
    return triplets


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


def resolve_alpha_path(root: Path | None, stem: str) -> Path | None:
    root = _optional_root(root)
    if root is None:
        return None
    for ext in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if USER_RESIZE_INPUT_TO_1024:
            image = image.resize((1024, 1024), Image.BICUBIC)
        return T.functional.to_tensor(image)


def load_alpha_tensor(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    with Image.open(path) as image:
        image = image.convert("L").resize((256, 256), Image.BICUBIC)
        return T.functional.to_tensor(image).unsqueeze(0)


def build_opts() -> Namespace:
    return Namespace(
        size=1024,
        ckpt=USER_STYLEGAN_CKPT,
        channel_multiplier=2,
        latent=512,
        n_mlp=8,
        device=USER_DEVICE if torch.cuda.is_available() else "cpu",
        batch_size=USER_ENCODER_BATCH_SIZE,
        save_all=False,
        save_all_dir=Path("output"),
        mixing=USER_MIXING,
        smooth=USER_SMOOTH,
        rotate_checkpoint=USER_ROTATE_CKPT,
        topology_boundary_width_v12=USER_TOPOLOGY_BOUNDARY_WIDTH_V12,
        alpha_fallback_blur_v12=USER_ALPHA_FALLBACK_BLUR_V12,
        bang_top_ratio_v12=USER_BANG_TOP_RATIO_V12,
        use_shape_adapter_v12=False,
        shape_adapter_v12_checkpoint="",
        shape_adapter_strength_v12=0.0,
    )


def save_fs_bundle(output_dir: Path, role: str, stem: str, embed: dict[str, torch.Tensor]):
    save_latents(
        output_dir,
        "FS",
        f"{role}_{stem}.npz",
        latent_S=embed["S"],
        latent_F32=embed["F"],
    )
    hair_mask = torch.where(embed["mask"] == 13, torch.ones_like(embed["mask"]), torch.zeros_like(embed["mask"])).float()
    save_latents(
        output_dir,
        "Mask",
        f"{role}_{stem}.npz",
        parsing_mask=embed["mask"],
        hair_mask_256=hair_mask,
        alpha_256=embed.get("alpha_256", hair_mask),
    )


def save_align_shape_bundle(output_dir: Path, face_stem: str, shape_stem: str, align_result: dict[str, torch.Tensor]):
    payload = {
        "latent_F_base": align_result["latent_F_base"],
        "latent_F": align_result["latent_F_align"],
        "latent_F_src": align_result["latent_F_src"],
        "latent_F_shape": align_result["latent_F_shape"],
        "latent_F_shape_raw": align_result.get("latent_F_shape_raw", align_result["latent_F_shape"]),
        "target_hair_mask": align_result["target_hair_mask"],
        "source_hair_mask": align_result["source_hair_mask"],
        "donor_hair_mask": align_result["donor_hair_mask"],
        "boundary_mask_256": align_result["boundary_mask_256"],
        "boundary_alpha_256": align_result["boundary_alpha_256"],
    }
    for key in V12_MASK_KEYS:
        payload[key] = align_result[key]
    save_latents(output_dir, "AlignShape", f"face_{face_stem}__alignshape_{shape_stem}.npz", **payload)


def save_align_color_bundle(output_dir: Path, face_stem: str, color_stem: str, align_color: dict[str, torch.Tensor]):
    payload = {
        "target_hair_mask": align_color["target_hair_mask"],
        "source_hair_mask": align_color["source_hair_mask"],
        "donor_hair_mask": align_color["donor_hair_mask"],
        "boundary_mask_256": align_color["boundary_mask_256"],
        "boundary_alpha_256": align_color["boundary_alpha_256"],
    }
    save_latents(output_dir, "AlignColor", f"face_{face_stem}__aligncolor_{color_stem}.npz", **payload)


def main():
    seed_everything(USER_RANDOM_SEED)
    face_images = list_image_files(ACTIVE_FACE_ROOT)
    shape_images = list_image_files(ACTIVE_SHAPE_ROOT)
    color_images = list_image_files(ACTIVE_COLOR_ROOT)
    _assert_unique_stems(face_images, "ACTIVE_FACE_ROOT")
    _assert_unique_stems(shape_images, "ACTIVE_SHAPE_ROOT")
    _assert_unique_stems(color_images, "ACTIVE_COLOR_ROOT")

    triplets = sample_triplets(
        face_images,
        shape_images,
        color_images,
        size=ACTIVE_DATASET_SIZE,
        allow_reuse=USER_ALLOW_REUSE_ACROSS_PAIRS,
        seed=USER_RANDOM_SEED,
    )

    opts = build_opts()
    net = Net(opts)
    embedder = Embedding(opts, net=net).eval()
    aligner = Alignment_v12(opts, latent_encoder=embedder.get_e4e_embed, net=net).eval()

    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_stem, shape_stem, color_stem in tqdm(triplets, desc="shape v12 cache"):
            f_exps.write(f"{face_stem} {shape_stem} {color_stem}\n")
            f_exps.flush()

            face_img, shape_img, color_img = equal_replacer(
                [
                    load_image_tensor(resolve_image_path(ACTIVE_FACE_ROOT, face_stem)),
                    load_image_tensor(resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)),
                    load_image_tensor(resolve_image_path(ACTIVE_COLOR_ROOT, color_stem)),
                ]
            )
            images_to_name = defaultdict(list)
            for image, name in zip((face_img, shape_img, color_img), ("face", "shape", "color")):
                images_to_name[image].append(name)

            name_to_embed = embedder.embedding_images(images_to_name)
            shape_alpha = load_alpha_tensor(resolve_alpha_path(ACTIVE_SHAPE_ALPHA_ROOT, shape_stem))
            if shape_alpha is not None:
                name_to_embed["shape"]["alpha_256"] = shape_alpha.to(opts.device)

            align_shape = aligner.align_images("face", "shape", name_to_embed, use_shape_adapter_v12=False)
            align_color = aligner.shape_module("face", "color", name_to_embed, only_target=True)

            save_fs_bundle(ACTIVE_OUTPUT_DIR, "face", face_stem, name_to_embed["face"])
            save_fs_bundle(ACTIVE_OUTPUT_DIR, "shape", shape_stem, name_to_embed["shape"])
            save_fs_bundle(ACTIVE_OUTPUT_DIR, "color", color_stem, name_to_embed["color"])
            save_align_shape_bundle(ACTIVE_OUTPUT_DIR, face_stem, shape_stem, align_shape)
            save_align_color_bundle(ACTIVE_OUTPUT_DIR, face_stem, color_stem, align_color)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"Saved {len(triplets)} v12 triplets to {ACTIVE_OUTPUT_DIR}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")
    print(f"shape alpha root: {ACTIVE_SHAPE_ALPHA_ROOT}")


if __name__ == "__main__":
    main()
