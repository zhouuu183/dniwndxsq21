import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v18 import HairFast_v18, get_parser_v18
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("input/blending_dataset_v18")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("input/blending_dataset_v18_small")
USER_DATASET_SIZE_SMALL = 300

USER_DEVICE = "cuda"
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True
USER_USE_SATD_V18 = False
USER_SATD_CKPT_V18 = ""
USER_SATD_BLEND_V18 = 0.28
# =============================================================================


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
    raise FileNotFoundError(f"Cannot find {stem}.png/.jpg/.jpeg/.webp in {root}")


def align_instead_shape(hair_fast):
    def shape_module(func):
        def wrapper(*args, **kwargs):
            if kwargs.get("align_flag", False):
                return hair_fast.align.align_images(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    def align_module(func):
        def wrapper(*args, **kwargs):
            if "align_flag" in kwargs:
                kwargs = kwargs.copy()
                kwargs.pop("align_flag")
            return func(*args, **kwargs)

        return wrapper

    hair_fast.align.shape_module = shape_module(hair_fast.align.shape_module)
    hair_fast.align.align_images = align_module(hair_fast.align.align_images)


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


def build_model() -> HairFast_v18:
    model_args = get_parser_v18().parse_args([])
    model_args.device = USER_DEVICE
    model_args.save_all = False
    model_args.use_satd_v18 = bool(USER_USE_SATD_V18 and USER_SATD_CKPT_V18)
    model_args.satd_checkpoint_v18 = USER_SATD_CKPT_V18
    model_args.satd_blend_v18 = USER_SATD_BLEND_V18
    model_args.use_post_satd_v18 = False
    model_args.use_postprocess_v18 = False

    hair_fast = HairFast_v18(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)
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

            align_shape, align_color, name_to_embed = hair_fast(
                face_path,
                shape_path,
                color_path,
                align_flag=True,
            )

            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{face_name}.npz", latent_in=name_to_embed["face"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{shape_name}.npz", latent_in=name_to_embed["shape"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{color_name}.npz", latent_in=name_to_embed["color"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "Align", f"{face_name}_{shape_name}.npz", latent_F=align_shape["latent_F_align"])
            save_latents(ACTIVE_OUTPUT_DIR, "Align", f"{face_name}_{color_name}.npz", latent_F=align_color["latent_F_align"])

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"saved {len(triplets)} blending-v18 triplets to {ACTIVE_OUTPUT_DIR / 'dataset.exps'}")
    print(f"face root: {ACTIVE_FACE_ROOT}")
    print(f"shape root: {ACTIVE_SHAPE_ROOT}")
    print(f"color root: {ACTIVE_COLOR_ROOT}")


if __name__ == "__main__":
    main()
