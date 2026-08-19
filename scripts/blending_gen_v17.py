import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v17 import HairFast_v17, get_parser_v17
from models.Blending_v17 import build_lock_mask_v17, build_safe_mask_v17
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("input/blending_dataset_v17")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("iamges/blending_dataset_v17_small")
USER_DATASET_SIZE_SMALL = 300

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True

USER_USE_SATD_V17 = True
USER_SATD_CHECKPOINT_V17 = "checkpoints/best.pth"
USER_SATD_BLEND_V17 = 0.34
USER_SATD_BOUNDARY_V17 = 8
USER_EQ_REFERENCE_BLEND_V17 = 0.0
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


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


def find_image_path(root: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find an image for stem={stem!r} in {root}")


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
        raise RuntimeError(
            f"{label} contains duplicate stems inside the same directory, which would make loading ambiguous: {sample}"
        )


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

    if not allow_reuse and (
        size > len(face_images) or size > len(shape_images) or size > len(color_images)
    ):
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
                raise RuntimeError("The image pool has been exhausted. Reduce ACTIVE_DATASET_SIZE or allow reuse.")
            face = rng.choice(face_pool)
            shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_pool, {Path(face).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
            color_pool.remove(color)

        triplets.append((Path(face).stem, Path(shape).stem, Path(color).stem))

    return triplets


def build_model() -> HairFast_v17:
    if USER_USE_SATD_V17:
        if not USER_SATD_CHECKPOINT_V17:
            raise RuntimeError("USER_SATD_CHECKPOINT_V17 is empty while USER_USE_SATD_V17=True.")
        if not Path(USER_SATD_CHECKPOINT_V17).exists():
            raise FileNotFoundError(f"Cannot find USER_SATD_CHECKPOINT_V17: {USER_SATD_CHECKPOINT_V17}")

    model_args = get_parser_v17().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v17 = bool(USER_USE_SATD_V17)
    model_args.satd_checkpoint_v17 = USER_SATD_CHECKPOINT_V17
    model_args.satd_blend_v17 = USER_SATD_BLEND_V17
    model_args.satd_boundary_v17 = USER_SATD_BOUNDARY_V17
    model_args.eq_reference_blend_v17 = USER_EQ_REFERENCE_BLEND_V17

    hair_fast = HairFast_v17(model_args)
    hair_fast.blend.blend_images = identity_func
    return hair_fast


def main():
    seed_everything(USER_RANDOM_SEED)
    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    face_images = list_image_files(ACTIVE_FACE_ROOT)
    shape_images = list_image_files(ACTIVE_SHAPE_ROOT)
    color_images = list_image_files(ACTIVE_COLOR_ROOT)
    _assert_unique_stems(face_images, "ACTIVE_FACE_ROOT")
    _assert_unique_stems(shape_images, "ACTIVE_SHAPE_ROOT")
    _assert_unique_stems(color_images, "ACTIVE_COLOR_ROOT")

    hair_fast = build_model()
    triplets = sample_triplets(
        face_images,
        shape_images,
        color_images,
        ACTIVE_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
    )

    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_name, shape_name, color_name in triplets:
            print(face_name, shape_name, color_name, file=f_exps, flush=True)

            face_path = find_image_path(ACTIVE_FACE_ROOT, face_name)
            shape_path = find_image_path(ACTIVE_SHAPE_ROOT, shape_name)
            color_path = find_image_path(ACTIVE_COLOR_ROOT, color_name)

            align_shape, _, name_to_embed = hair_fast(
                face_path,
                shape_path,
                color_path,
                align_flag=True,
            )

            hair_mask = align_shape["HM_X"].float()
            lock_mask = build_lock_mask_v17(align_shape)
            safe_mask = build_safe_mask_v17(hair_mask, lock_mask)

            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{face_name}.npz", latent_in=name_to_embed["face"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{shape_name}.npz", latent_in=name_to_embed["shape"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{color_name}.npz", latent_in=name_to_embed["color"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "Align", f"{face_name}_{shape_name}.npz", latent_F=align_shape["latent_F_align"])
            save_latents(
                ACTIVE_OUTPUT_DIR,
                "Masks",
                f"{face_name}_{shape_name}.npz",
                hair_mask=hair_mask,
                lock_mask=lock_mask,
                safe_mask=safe_mask,
            )

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"saved {len(triplets)} v17 blending triplets to {ACTIVE_OUTPUT_DIR / 'dataset.exps'}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")
    print(f"use satd v17: {USER_USE_SATD_V17}")
    print(f"satd checkpoint: {USER_SATD_CHECKPOINT_V17}")
    print(f"satd blend: {USER_SATD_BLEND_V17}")
    print(f"satd boundary: {USER_SATD_BOUNDARY_V17}")


if __name__ == "__main__":
    main()
