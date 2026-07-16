import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v7 import HairFast_v7, get_parser_v7
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= 用户配置区域：只改这里 =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("images/blending_dataset_v7")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("images/blending_dataset_v7_small")
USER_DATASET_SIZE_SMALL = 300

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True

USER_DEVICE = "cuda"
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_SAVE_ALL = False
# ========================================================================


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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_DATASET_SIZE = _DATASET_CFG["dataset_size"]


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


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


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


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


def save_align_bundle(output_dir: Path, file_name: str, align_result: dict):
    payload = {
        "latent_F": align_result["latent_F_align"],
        "target_hair_mask": align_result["HM_X"],
    }
    save_latents(output_dir, "Align", file_name, **payload)


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

    model_args = get_parser_v7().parse_args([])
    model_args.device = USER_DEVICE
    model_args.ckpt = USER_STYLEGAN_CKPT
    model_args.rotate_checkpoint = USER_ROTATE_CKPT
    model_args.blending_checkpoint = USER_BLENDING_CKPT
    model_args.pp_checkpoint = USER_PP_CKPT
    model_args.save_all = USER_SAVE_ALL
    hair_fast = HairFast_v7(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)

    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_stem, shape_stem, color_stem in triplets:
            f_exps.write(f"{face_stem} {shape_stem} {color_stem}\n")

            face_path = resolve_image_path(ACTIVE_FACE_ROOT, face_stem)
            shape_path = resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)
            color_path = resolve_image_path(ACTIVE_COLOR_ROOT, color_stem)

            align_shape, align_color, name_to_embed = hair_fast(face_path, shape_path, color_path, align_flag=True)
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{face_stem}.npz", latent_in=name_to_embed["face"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{shape_stem}.npz", latent_in=name_to_embed["shape"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{color_stem}.npz", latent_in=name_to_embed["color"]["S"])
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{face_stem}_{shape_stem}.npz", align_shape)
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{face_stem}_{color_stem}.npz", align_color)

    print(f"Saved {len(triplets)} blending triplets to {ACTIVE_OUTPUT_DIR}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_COLOR_ROOT}")


if __name__ == "__main__":
    main()
import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v7 import HairFast_v7, get_parser_v7
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= 用户配置区域：只改这里 =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("images/blending_dataset_v7")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("images/blending_dataset_v7_small")
USER_DATASET_SIZE_SMALL = 300

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True

USER_DEVICE = "cuda"
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_SAVE_ALL = False
# ========================================================================


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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_DATASET_SIZE = _DATASET_CFG["dataset_size"]


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


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


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


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


def save_align_bundle(output_dir: Path, file_name: str, align_result: dict):
    payload = {
        "latent_F": align_result["latent_F_align"],
        "target_hair_mask": align_result["HM_X"],
    }
    save_latents(output_dir, "Align", file_name, **payload)


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

    model_args = get_parser_v7().parse_args([])
    model_args.device = USER_DEVICE
    model_args.ckpt = USER_STYLEGAN_CKPT
    model_args.rotate_checkpoint = USER_ROTATE_CKPT
    model_args.blending_checkpoint = USER_BLENDING_CKPT
    model_args.pp_checkpoint = USER_PP_CKPT
    model_args.save_all = USER_SAVE_ALL
    hair_fast = HairFast_v7(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)

    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_stem, shape_stem, color_stem in triplets:
            f_exps.write(f"{face_stem} {shape_stem} {color_stem}\n")

            face_path = resolve_image_path(ACTIVE_FACE_ROOT, face_stem)
            shape_path = resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)
            color_path = resolve_image_path(ACTIVE_COLOR_ROOT, color_stem)

            align_shape, align_color, name_to_embed = hair_fast(face_path, shape_path, color_path, align_flag=True)
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{face_stem}.npz", latent_in=name_to_embed["face"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{shape_stem}.npz", latent_in=name_to_embed["shape"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{color_stem}.npz", latent_in=name_to_embed["color"]["S"])
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{face_stem}_{shape_stem}.npz", align_shape)
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{face_stem}_{color_stem}.npz", align_color)

    print(f"Saved {len(triplets)} blending triplets to {ACTIVE_OUTPUT_DIR}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_COLOR_ROOT}")


if __name__ == "__main__":
    main()
import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v7 import HairFast_v7, get_parser_v7
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= 用户配置区域：只改这里 =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("images/blending_dataset_v7")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_fringe")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ")
USER_OUTPUT_DIR_SMALL = Path("images/blending_dataset_v7_small")
USER_DATASET_SIZE_SMALL = 300

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True

USER_DEVICE = "cuda"
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
USER_BLENDING_CKPT = "pretrained_models/Blending/checkpoint.pth"
USER_PP_CKPT = "pretrained_models/PostProcess/pp_model.pth"
USER_SAVE_ALL = False
# ========================================================================


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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_DATASET_SIZE = _DATASET_CFG["dataset_size"]


def identity_func(align_shape, align_color, name_to_embed, **kwargs):
    return align_shape, align_color, name_to_embed


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


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = root / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find image for {stem} under {root}")


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


def save_align_bundle(output_dir: Path, file_name: str, align_result: dict):
    payload = {
        "latent_F": align_result["latent_F_align"],
        "target_hair_mask": align_result["HM_X"],
    }
    save_latents(output_dir, "Align", file_name, **payload)


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

    model_args = get_parser_v7().parse_args([])
    model_args.device = USER_DEVICE
    model_args.ckpt = USER_STYLEGAN_CKPT
    model_args.rotate_checkpoint = USER_ROTATE_CKPT
    model_args.blending_checkpoint = USER_BLENDING_CKPT
    model_args.pp_checkpoint = USER_PP_CKPT
    model_args.save_all = USER_SAVE_ALL
    hair_fast = HairFast_v7(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)

    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for face_stem, shape_stem, color_stem in triplets:
            f_exps.write(f"{face_stem} {shape_stem} {color_stem}\n")

            face_path = resolve_image_path(ACTIVE_FACE_ROOT, face_stem)
            shape_path = resolve_image_path(ACTIVE_SHAPE_ROOT, shape_stem)
            color_path = resolve_image_path(ACTIVE_COLOR_ROOT, color_stem)

            align_shape, align_color, name_to_embed = hair_fast(face_path, shape_path, color_path, align_flag=True)
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{face_stem}.npz", latent_in=name_to_embed["face"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{shape_stem}.npz", latent_in=name_to_embed["shape"]["S"])
            save_latents(ACTIVE_OUTPUT_DIR, "FS", f"{color_stem}.npz", latent_in=name_to_embed["color"]["S"])
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{face_stem}_{shape_stem}.npz", align_shape)
            save_align_bundle(ACTIVE_OUTPUT_DIR, f"{face_stem}_{color_stem}.npz", align_color)

    print(f"Saved {len(triplets)} blending triplets to {ACTIVE_OUTPUT_DIR}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_COLOR_ROOT}")


if __name__ == "__main__":
    main()
