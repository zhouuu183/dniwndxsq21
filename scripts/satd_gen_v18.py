import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.image_utils import list_image_files


# ========================= User Config: edit here only =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("images/satd_dataset_v18_3000")
USER_DATASET_SIZE_FFHQ = 3000

USER_FACE_ROOT_SMALL = Path("images/FFHQ_long")
USER_SHAPE_ROOT_SMALL = Path("images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("images/FFHQ_color")
USER_OUTPUT_DIR_SMALL = Path("images/satd_dataset_v18_small")
USER_DATASET_SIZE_SMALL = 300

USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_PAIRS = True
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


_DATASET_CFG = resolve_dataset_profile()
ACTIVE_FACE_ROOT = _DATASET_CFG["face_root"]
ACTIVE_SHAPE_ROOT = _DATASET_CFG["shape_root"]
ACTIVE_COLOR_ROOT = _DATASET_CFG["color_root"]
ACTIVE_OUTPUT_DIR = _DATASET_CFG["output_dir"]
ACTIVE_DATASET_SIZE = _DATASET_CFG["dataset_size"]


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

    if not allow_reuse and (size > len(face_images) or size > len(shape_images) or size > len(color_images)):
        raise RuntimeError("Not enough unique images to sample all triplets without reuse.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = color_images.copy()
    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_images)
            shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_images, {Path(face).stem, Path(shape).stem})
        else:
            if not face_pool or not shape_pool or not color_pool:
                raise RuntimeError("The image pool has been exhausted. Reduce ACTIVE_DATASET_SIZE or allow reuse.")
            face = rng.choice(face_pool)
            shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
            color = _sample_excluding_stems(rng, color_pool, {Path(face).stem, Path(shape).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
            color_pool.remove(color)
        triplets.append((Path(face).stem, Path(shape).stem, Path(color).stem))
    return triplets


def main():
    face_images = list_image_files(ACTIVE_FACE_ROOT)
    shape_images = list_image_files(ACTIVE_SHAPE_ROOT)
    color_images = list_image_files(ACTIVE_COLOR_ROOT)
    _assert_unique_stems(face_images, "ACTIVE_FACE_ROOT")
    _assert_unique_stems(shape_images, "ACTIVE_SHAPE_ROOT")
    _assert_unique_stems(color_images, "ACTIVE_COLOR_ROOT")

    ACTIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    triplets = sample_triplets(
        face_images,
        shape_images,
        color_images,
        ACTIVE_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_PAIRS,
        USER_RANDOM_SEED,
    )

    with open(ACTIVE_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f:
        for face, shape, color in triplets:
            f.write(f"{face} {shape} {color}\n")

    print(f"dataset profile: {USER_DATASET_PROFILE}")
    print(f"Saved {len(triplets)} SATD-v18 triplets to {ACTIVE_OUTPUT_DIR / 'dataset.exps'}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_COLOR_ROOT}")


if __name__ == "__main__":
    main()
