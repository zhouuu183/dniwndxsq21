"""Generate deterministic SATD-v5s triplets: face, shape, and color.

The three-column manifest is consumed by ``satd_train_v5s.py`` so training
uses the same S_blend conditioning as V6 inference.  The original
``satd_gen_v8.py`` remains unchanged.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.image_utils import list_image_files


# ========================= 用户配置区域：只改这里 =========================
USER_DATASET_PROFILE = "small"

USER_FACE_ROOT_FFHQ = Path("images/FFHQ")
USER_SHAPE_ROOT_FFHQ = Path("images/FFHQ")
USER_COLOR_ROOT_FFHQ = Path("images/FFHQ")
USER_OUTPUT_DIR_FFHQ = Path("images/satd_dataset_v5s_3000")
USER_DATASET_SIZE_FFHQ = 3000

# Keep the v5s experiment on the same galleries as the production V6 small
# profile.  This is required for row-for-row source/shape/colour pairing.
USER_FACE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_LONG")
USER_SHAPE_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
USER_COLOR_ROOT_SMALL = Path("/data/coding/HairFastGAN/HairFastGAN-main/images/FFHQ_short")
USER_OUTPUT_DIR_SMALL = Path("images/satd_dataset_v5s_small")
USER_DATASET_SIZE_SMALL = 100

USER_RANDOM_SEED = 3407
# Keep this flag for compatibility with the earlier v5s config.  The V6
# generator always samples the same source/donor identities as pp_gen_v6;
# when the requested size exceeds a gallery it permits source reuse.
USER_ALLOW_REUSE_ACROSS_PAIRS = True
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


def _identity_key(name: str) -> str:
    return Path(str(name)).stem.casefold()


def sample_triplets(
    face_images: list[str],
    shape_images: list[str],
    color_images: list[str],
    size: int,
    allow_reuse: bool,
    seed: int,
) -> list[tuple[str, str, str]]:
    """Use the exact identity-aware sampler used by ``pp_gen_v6.py``.

    Shape and colour are donor roles, not two unrelated random lists.  The
    source, shape, and colour identities are distinct for every row, which
    makes the generated manifest safe to feed into the production V6 path.
    ``allow_reuse`` only controls whether source images may repeat after the
    source gallery is exhausted; it does not change role assignment.
    """
    if not face_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_FACE_ROOT}")
    if not shape_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_SHAPE_ROOT}")
    if not color_images:
        raise RuntimeError(f"No png/jpg images found under {ACTIVE_COLOR_ROOT}")
    if size <= 0:
        raise ValueError("ACTIVE_DATASET_SIZE must be positive")
    if not allow_reuse and size > len(face_images):
        raise RuntimeError("Not enough unique source images; enable reuse or reduce dataset size.")

    # pp_gen_v6 seeds NumPy before calling this sampler.  Re-seeding locally
    # keeps v5s deterministic and reproduces the same order for the same
    # galleries, seed, and size.
    np.random.seed(seed)
    # Colour images can come from a separate directory, so keep their actual
    # candidates in a second map while using the union of identity keys.
    color_by_identity: dict[str, list[str]] = {}
    for name in color_images:
        color_by_identity.setdefault(_identity_key(name), []).append(name)
    shape_by_identity: dict[str, list[str]] = {}
    for name in shape_images:
        shape_by_identity.setdefault(_identity_key(name), []).append(name)

    donor_identities = np.asarray(sorted(set(shape_by_identity) & set(color_by_identity)), dtype=object)
    if donor_identities.size < 3:
        raise RuntimeError(
            "The shape/color galleries need at least three shared identity keys "
            "for source/shape/color separation."
        )

    if size <= len(face_images):
        sources = np.random.permutation(np.asarray(face_images, dtype=object))[:size]
    elif allow_reuse:
        initial = np.random.permutation(np.asarray(face_images, dtype=object))
        extra = np.random.choice(face_images, size=size - len(face_images), replace=True)
        sources = np.concatenate((initial, extra))
    else:
        raise RuntimeError("Not enough unique source images; enable reuse or reduce dataset size.")

    triplets: list[tuple[str, str, str]] = []
    for source_name in sources:
        source_identity = _identity_key(str(source_name))
        allowed_shape_ids = donor_identities[donor_identities != source_identity]
        if allowed_shape_ids.size == 0:
            raise RuntimeError(f"No donor shape identity available for source {source_name!r}")
        shape_identity = str(np.random.choice(allowed_shape_ids))
        allowed_color_ids = allowed_shape_ids[allowed_shape_ids != shape_identity]
        if allowed_color_ids.size == 0:
            raise RuntimeError(f"No donor colour identity available for source {source_name!r}")
        color_identity = str(np.random.choice(allowed_color_ids))
        shape_name = shape_by_identity[shape_identity][
            int(np.random.randint(len(shape_by_identity[shape_identity])))
        ]
        color_name = color_by_identity[color_identity][
            int(np.random.randint(len(color_by_identity[color_identity])))
        ]
        triplets.append((Path(str(source_name)).stem, Path(shape_name).stem, Path(color_name).stem))
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
    print(f"Saved {len(triplets)} SATD triplets to {ACTIVE_OUTPUT_DIR / 'dataset.exps'}")
    print(f"face/source root: {ACTIVE_FACE_ROOT}")
    print(f"shape/reference root: {ACTIVE_SHAPE_ROOT}")
    print(f"color/reference root: {ACTIVE_COLOR_ROOT}")


if __name__ == "__main__":
    main()
