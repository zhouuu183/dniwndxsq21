from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as T
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.Alignment_v10 import Alignment_v10
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer, list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= 用户配置区域：只改这里 =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_FACE_ALPHA_ROOT_FFHQ = Path("")
USER_SHAPE_ALPHA_ROOT_FFHQ = Path("")
USER_COLOR_ALPHA_ROOT_FFHQ = Path("")
USER_OUTPUT_DIR_FFHQ = Path("images/blending_dataset_v10_3000")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_FACE_ALPHA_ROOT_SMALL = Path("")
USER_SHAPE_ALPHA_ROOT_SMALL = Path("images/FFHQ_fringe_alpha")
USER_COLOR_ALPHA_ROOT_SMALL = Path("")
USER_OUTPUT_DIR_SMALL = Path("images/blending_dataset_v11_noF64")
USER_DATASET_SIZE_SMALL = 300

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True
USER_RESIZE_INPUT_TO_1024 = True

USER_DEVICE = "cuda"
USER_ENCODER_BATCH_SIZE = 3
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_MIXING = 0.95
USER_SMOOTH = 5

USER_ALPHA_BOUNDARY_WIDTH_V10 = 9
USER_ALPHA_BOUNDARY_STRENGTH_V10 = 0.65
USER_ALPHA_FALLBACK_BLUR_V10 = 9

# 先关闭 F64 缓存/注入线，只验证 baseline F32 对齐 + MODNet alpha 边界融合。
USER_SAVE_F64_CACHE = False
# ========================================================================


def resolve_dataset_profile() -> dict[str, object]:
    profiles = {
        "ffhq": {
            "face_root": USER_FACE_ROOT_FFHQ,
            "shape_root": USER_SHAPE_ROOT_FFHQ,
            "color_root": USER_COLOR_ROOT_FFHQ,
            "face_alpha_root": USER_FACE_ALPHA_ROOT_FFHQ,
            "shape_alpha_root": USER_SHAPE_ALPHA_ROOT_FFHQ,
            "color_alpha_root": USER_COLOR_ALPHA_ROOT_FFHQ,
            "output_dir": USER_OUTPUT_DIR_FFHQ,
            "dataset_size": USER_DATASET_SIZE_FFHQ,
        },
        "small": {
            "face_root": USER_FACE_ROOT_SMALL,
            "shape_root": USER_SHAPE_ROOT_SMALL,
            "color_root": USER_COLOR_ROOT_SMALL,
            "face_alpha_root": USER_FACE_ALPHA_ROOT_SMALL,
            "shape_alpha_root": USER_SHAPE_ALPHA_ROOT_SMALL,
            "color_alpha_root": USER_COLOR_ALPHA_ROOT_SMALL,
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
ACTIVE_FACE_ALPHA_ROOT = _DATASET_CFG["face_alpha_root"]
ACTIVE_SHAPE_ALPHA_ROOT = _DATASET_CFG["shape_alpha_root"]
ACTIVE_COLOR_ALPHA_ROOT = _DATASET_CFG["color_alpha_root"]
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
        sample = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(
            f"{label} contains duplicate stems inside the same directory, which would make loading ambiguous: {sample}"
        )


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
    triplets: list[tuple[str, str, str]] = []

    if not face_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_FACE_ROOT}")
    if not shape_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_SHAPE_ROOT}")
    if not color_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_COLOR_ROOT}")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = color_images.copy()
    if not allow_reuse:
        max_size = min(len(face_pool), len(shape_pool), len(color_pool))
        if size > max_size:
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


@torch.no_grad()
def feature64_from_style(net: Net, latent_s: torch.Tensor, latent_f32: torch.Tensor) -> torch.Tensor:
    feature64, _ = net.generator(
        [latent_s],
        input_is_latent=True,
        return_latents=False,
        start_layer=4,
        end_layer=4,
        layer_in=latent_f32,
    )
    return feature64


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
        alpha_boundary_width_v10=USER_ALPHA_BOUNDARY_WIDTH_V10,
        alpha_boundary_strength_v10=USER_ALPHA_BOUNDARY_STRENGTH_V10,
        alpha_fallback_blur_v10=USER_ALPHA_FALLBACK_BLUR_V10,
    )


def save_fs_bundle(
    output_dir: Path,
    role: str,
    stem: str,
    embed: dict[str, torch.Tensor],
    feature64: torch.Tensor | None = None,
):
    payload = {
        "latent_S": embed["S"],
        "latent_F32": embed["F"],
    }
    if USER_SAVE_F64_CACHE and feature64 is not None:
        payload["latent_F64"] = feature64
    save_latents(output_dir, "FS", f"{role}_{stem}.npz", **payload)
    hair_mask = torch.where(embed["mask"] == 13, torch.ones_like(embed["mask"]), torch.zeros_like(embed["mask"])).float()
    save_latents(
        output_dir,
        "Mask",
        f"{role}_{stem}.npz",
        parsing_mask=embed["mask"],
        hair_mask_256=hair_mask,
        alpha_256=embed.get("alpha_256", hair_mask),
    )


def save_align_bundle(output_dir: Path, kind: str, face_stem: str, other_stem: str, align_result: dict[str, torch.Tensor]):
    payload = {
        "target_hair_mask": align_result["target_hair_mask"],
        "source_hair_mask": align_result["source_hair_mask"],
        "donor_hair_mask": align_result["donor_hair_mask"],
        "boundary_mask_256": align_result["boundary_mask_256"],
        "boundary_alpha_256": align_result["boundary_alpha_256"],
    }
    if "latent_F_align" in align_result:
        payload["latent_F"] = align_result["latent_F_align"]
    if "latent_F_hair" in align_result:
        payload["latent_F_hair"] = align_result["latent_F_hair"]

    save_latents(output_dir, kind, f"face_{face_stem}__{kind.lower()}_{other_stem}.npz", **payload)


def main():
    seed_everything(USER_RANDOM_SEED)

    face_images = list_image_files(ACTIVE_FACE_ROOT)
    shape_images = list_image_files(ACTIVE_SHAPE_ROOT)
    color_images = list_image_files(ACTIVE_COLOR_ROOT)
    _assert_unique_stems(face_images, "ACTIVE_FACE_ROOT")
    _assert_unique_stems(shape_images, "ACTIVE_SHAPE_ROOT")
    _assert_unique_stems(color_images, "ACTIVE_COLOR_ROOT")

    triplets = sample_triplets(
        face_images=face_images,
        shape_images=shape_images,
        color_images=color_images,
        size=ACTIVE_DATASET_SIZE,
        allow_reuse=USER_ALLOW_REUSE_ACROSS_PAIRS,
        seed=USER_RANDOM_SEED,
    )

    opts = build_opts()
    net = Net(opts)
    embedder = Embedding(opts, net=net).eval()
    aligner = Alignment_v10(opts, latent_encoder=embedder.get_e4e_embed, net=net).eval()

    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_stem, shape_stem, color_stem in tqdm(triplets, desc="blending v10 cache"):
            f_exps.write(f"{face_stem} {shape_stem} {color_stem}\n")
            f_exps.flush()

            face_path = resolve_image_path(ACTIVE_FACE_ROOT, face_stem)
            shape_path = resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)
            color_path = resolve_image_path(ACTIVE_COLOR_ROOT, color_stem)

            face_img, shape_img, color_img = equal_replacer(
                [load_image_tensor(face_path), load_image_tensor(shape_path), load_image_tensor(color_path)]
            )
            images_to_name = defaultdict(list)
            for image, name in zip((face_img, shape_img, color_img), ("face", "shape", "color")):
                images_to_name[image].append(name)

            name_to_embed = embedder.embedding_images(images_to_name)
            alpha_specs = {
                "face": (ACTIVE_FACE_ALPHA_ROOT, face_stem),
                "shape": (ACTIVE_SHAPE_ALPHA_ROOT, shape_stem),
                "color": (ACTIVE_COLOR_ALPHA_ROOT, color_stem),
            }
            for role, (alpha_root, stem) in alpha_specs.items():
                alpha = load_alpha_tensor(resolve_alpha_path(alpha_root, stem))
                if alpha is not None:
                    name_to_embed[role]["alpha_256"] = alpha.to(opts.device)

            feature64 = {}
            if USER_SAVE_F64_CACHE:
                feature64 = {
                    role: feature64_from_style(net, name_to_embed[role]["S"], name_to_embed[role]["F"])
                    for role in ("face", "shape", "color")
                }
                for role in ("face", "shape", "color"):
                    name_to_embed[role]["F64"] = feature64[role]

            align_shape = aligner.align_images("face", "shape", name_to_embed)
            align_color = aligner.shape_module("face", "color", name_to_embed, only_target=True)

            save_fs_bundle(ACTIVE_OUTPUT_DIR, "face", face_stem, name_to_embed["face"], feature64.get("face"))
            save_fs_bundle(ACTIVE_OUTPUT_DIR, "shape", shape_stem, name_to_embed["shape"], feature64.get("shape"))
            save_fs_bundle(ACTIVE_OUTPUT_DIR, "color", color_stem, name_to_embed["color"], feature64.get("color"))
            save_align_bundle(ACTIVE_OUTPUT_DIR, "AlignShape", face_stem, shape_stem, align_shape)
            save_align_bundle(ACTIVE_OUTPUT_DIR, "AlignColor", face_stem, color_stem, align_color)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"Saved {len(triplets)} blending v10 triplets to {ACTIVE_OUTPUT_DIR}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_COLOR_ROOT}")
    print(f"shape alpha root: {ACTIVE_SHAPE_ALPHA_ROOT}")
    print(f"save F64 cache: {USER_SAVE_F64_CACHE}")


if __name__ == "__main__":
    main()
