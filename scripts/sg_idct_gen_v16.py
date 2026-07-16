import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v16 import HairFast_v16, get_parser_v16
from utils.image_utils import list_image_files
from utils.train import seed_everything


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("input/sg_idct_dataset_v16")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("input/sg_idct_dataset_v16_small")
USER_DATASET_SIZE_SMALL = 300

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True
USER_FORCE_REFRESH = False
USER_SAVE_PREVIEWS = True

USER_USE_SATD_V16 = True
USER_SATD_CHECKPOINT_V16 = "output/satd_train_v8_3000/checkpoints/satd_for_infer_v8.pth"
USER_SATD_BLEND_V16 = 0.34
USER_SATD_BOUNDARY_V16 = 8
USER_EQ16_REFERENCE_BLEND_V16 = 0.0

USER_SG_IDCT_CHECKPOINT_V16 = ""
USER_SG_IDCT_BETA_V16 = 0.05
USER_SG_IDCT_CHROMA_STRENGTH_V16 = 0.90
USER_SG_IDCT_CHROMA_STD_STRENGTH_V16 = 0.0
USER_SG_IDCT_CHROMA_DELTA_LIMIT_V16 = 24.0
USER_SG_IDCT_TEXTURE_STRENGTH_V16 = 0.90
USER_SG_IDCT_LUMA_BINS_V16 = 9
USER_SG_IDCT_ALPHA_STRENGTH_V16 = 1.05
USER_SG_IDCT_ALPHA_DILATE_V16 = 6
USER_SG_IDCT_ALPHA_BLUR_RADIUS_V16 = 7
USER_SG_IDCT_USE_GAMUT_MAP_V16 = False
USER_SG_IDCT_SKIP_REFINEMENT_V16 = True
# ============================================================================


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


def find_image_path(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg"):
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg in {root}")


def _sample_excluding_stems(rng: random.Random, candidates: list[str], forbidden_stems: set[str]) -> str:
    valid = [item for item in candidates if Path(item).stem not in forbidden_stems]
    if not valid:
        raise RuntimeError("No valid candidates left after excluding duplicate stems.")
    return rng.choice(valid)


def _assert_unique_stems(images: list[str], label: str):
    counts: dict[str, int] = {}
    for item in images:
        stem = Path(item).stem
        counts[stem] = counts.get(stem, 0) + 1

    duplicates = [stem for stem, count in counts.items() if count > 1]
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(f"{label} contains duplicate stems, which would make loading ambiguous: {sample}")


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

    if not allow_reuse and (size > len(face_images) or size > len(shape_images) or size > len(color_images)):
        raise RuntimeError("Not enough unique images to sample all triplets without reuse.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = color_images.copy()

    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_images)
            shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_images, {Path(face).stem})
        else:
            if not face_pool or not shape_pool or not color_pool:
                raise RuntimeError("The image pool has been exhausted. Reduce dataset size or allow reuse.")
            face = rng.choice(face_pool)
            shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_pool, {Path(face).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
            color_pool.remove(color)

        triplets.append((Path(face).stem, Path(shape).stem, Path(color).stem))

    return triplets


def build_model() -> HairFast_v16:
    if USER_USE_SATD_V16:
        if not USER_SATD_CHECKPOINT_V16:
            raise RuntimeError("USER_SATD_CHECKPOINT_V16 is empty while USER_USE_SATD_V16=True.")
        if not Path(USER_SATD_CHECKPOINT_V16).exists():
            raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V16: {USER_SATD_CHECKPOINT_V16}")

    model_args = get_parser_v16().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v16 = bool(USER_USE_SATD_V16)
    model_args.satd_checkpoint_v16 = USER_SATD_CHECKPOINT_V16
    model_args.satd_blend_v16 = USER_SATD_BLEND_V16
    model_args.satd_boundary_v16 = USER_SATD_BOUNDARY_V16
    model_args.eq16_reference_blend_v16 = USER_EQ16_REFERENCE_BLEND_V16
    model_args.sg_idct_checkpoint_v16 = USER_SG_IDCT_CHECKPOINT_V16
    model_args.sg_idct_beta_v16 = USER_SG_IDCT_BETA_V16
    model_args.sg_idct_chroma_strength_v16 = USER_SG_IDCT_CHROMA_STRENGTH_V16
    model_args.sg_idct_chroma_std_strength_v16 = USER_SG_IDCT_CHROMA_STD_STRENGTH_V16
    model_args.sg_idct_chroma_delta_limit_v16 = USER_SG_IDCT_CHROMA_DELTA_LIMIT_V16
    model_args.sg_idct_texture_strength_v16 = USER_SG_IDCT_TEXTURE_STRENGTH_V16
    model_args.sg_idct_luma_bins_v16 = USER_SG_IDCT_LUMA_BINS_V16
    model_args.sg_idct_alpha_strength_v16 = USER_SG_IDCT_ALPHA_STRENGTH_V16
    model_args.sg_idct_alpha_dilate_v16 = USER_SG_IDCT_ALPHA_DILATE_V16
    model_args.sg_idct_alpha_blur_radius_v16 = USER_SG_IDCT_ALPHA_BLUR_RADIUS_V16
    model_args.sg_idct_use_gamut_map_v16 = USER_SG_IDCT_USE_GAMUT_MAP_V16
    model_args.sg_idct_skip_refinement_v16 = USER_SG_IDCT_SKIP_REFINEMENT_V16
    return HairFast_v16(model_args)


def save_preview(path: Path, tensors: list[torch.Tensor]):
    path.parent.mkdir(parents=True, exist_ok=True)
    target_size = tensors[0].shape[-2:]
    tiles = []
    for tensor in tensors:
        tensor = tensor.detach().cpu().clamp(0, 1)
        if tensor.shape[-2:] != target_size:
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0),
                size=target_size,
                mode="bicubic",
                align_corners=False,
            )[0].clamp(0, 1)
        tiles.append(tensor)
    save_image(torch.cat(tiles, dim=2), path)


def save_tensor_cache(path: Path, item: dict[str, torch.Tensor]):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        I_satd=item["satd"].detach().cpu().numpy(),
        I_color=item["color"].detach().cpu().numpy(),
        I_ct=item["ct"].detach().cpu().numpy(),
        H_align=item["H_align"].detach().cpu().numpy(),
        H_transfer=item.get("H_transfer", item["H_align"]).detach().cpu().numpy(),
        H_color=item["H_color"].detach().cpu().numpy(),
        M_remove=item["M_remove"].detach().cpu().numpy(),
        M_face=item["M_face"].detach().cpu().numpy(),
        M_neck=item["M_neck"].detach().cpu().numpy(),
        M_ear=item["M_ear"].detach().cpu().numpy(),
        M_safe=item["M_safe"].detach().cpu().numpy(),
        A_hair=item["A_hair"].detach().cpu().numpy(),
    )


def main():
    seed_everything(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tensor_dir = ACTIVE_OUTPUT_DIR / "Tensors"
    preview_dir = ACTIVE_OUTPUT_DIR / "Previews"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

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
        ACTIVE_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
    )
    hair_fast = build_model()

    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps, open(
        ACTIVE_OUTPUT_DIR / "manifest.tsv",
        "w",
        encoding="utf-8",
    ) as f_manifest:
        print("tensor_file\tface\tshape\tcolor", file=f_manifest)
        for idx, (face_name, shape_name, color_name) in enumerate(triplets):
            tensor_name = f"sample_{idx:06d}_{face_name}_{shape_name}_{color_name}.npz"
            tensor_path = tensor_dir / tensor_name
            print(face_name, shape_name, color_name, file=f_exps, flush=True)

            if tensor_path.exists() and not USER_FORCE_REFRESH:
                print(f"{tensor_name}\t{face_name}\t{shape_name}\t{color_name}", file=f_manifest, flush=True)
                continue

            item = hair_fast(
                find_image_path(ACTIVE_FACE_ROOT, face_name),
                find_image_path(ACTIVE_SHAPE_ROOT, shape_name),
                find_image_path(ACTIVE_COLOR_ROOT, color_name),
                seed=USER_RANDOM_SEED,
                exp_name=f"sample_{idx:06d}",
                return_intermediates_v16=True,
            )
            if not isinstance(item, dict):
                raise RuntimeError("HairFast_v16 did not return intermediates. Check return_intermediates_v16.")

            save_tensor_cache(tensor_path, item)
            if USER_SAVE_PREVIEWS:
                save_preview(preview_dir / f"sample_{idx:06d}.png", [item["color"], item["satd"], item["ct"], item["final"]])

            print(f"{tensor_name}\t{face_name}\t{shape_name}\t{color_name}", file=f_manifest, flush=True)

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"saved {len(triplets)} SG-IDCT v16 samples to {ACTIVE_OUTPUT_DIR}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")
    print(f"use satd v16: {USER_USE_SATD_V16}")
    print(f"satd checkpoint: {USER_SATD_CHECKPOINT_V16}")


if __name__ == "__main__":
    main()
