import random
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.image_utils import list_image_files


# ========================= 用户配置区域：只改这里 =========================
# 原图目录。这里放长发图，作为 face/source。
USER_FACE_ROOT = Path("images/FFHQ_long")

# 参考发型目录。这里放短发图，作为 shape/reference。
USER_SHAPE_ROOT = Path("images/FFHQ_short")

# 输出目录。这里只生成用于 SATD_v4 训练的 face/shape 对列表。
USER_OUTPUT_DIR = Path("images/satd_dataset_v4_long2short")

# 生成多少对 (face, shape)。
USER_DATASET_SIZE = 100

# 随机种子。
USER_RANDOM_SEED = 3407

# 是否允许不同 pair 之间重复使用同一张图。
USER_ALLOW_REUSE_ACROSS_PAIRS = True
# ========================================================================


def _sample_excluding_stems(rng: random.Random, candidates: list[str], forbidden_stems: set[str]) -> str:
    valid = [item for item in candidates if Path(item).stem not in forbidden_stems]
    if not valid:
        raise RuntimeError("No valid candidates left after excluding duplicate stems.")
    return rng.choice(valid)


def _assert_disjoint_stems(images_a: list[str], images_b: list[str], label_a: str, label_b: str):
    overlap = {Path(item).stem for item in images_a} & {Path(item).stem for item in images_b}
    if overlap:
        sample = ", ".join(sorted(list(overlap))[:10])
        raise RuntimeError(
            f"{label_a} and {label_b} contain duplicate stems, which would make face/shape ambiguous: {sample}"
        )


def sample_pairs(face_images: list[str], shape_images: list[str], size: int, allow_reuse: bool, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []

    if not face_images:
        raise RuntimeError(f"No png/jpg images found under {USER_FACE_ROOT}")
    if not shape_images:
        raise RuntimeError(f"No png/jpg images found under {USER_SHAPE_ROOT}")

    if not allow_reuse and (size > len(face_images) or size > len(shape_images)):
        raise RuntimeError("Not enough unique images to sample all pairs without reuse.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    for _ in range(size):
        if allow_reuse:
            face = rng.choice(face_images)
            shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
        else:
            if not face_pool or not shape_pool:
                raise RuntimeError("The image pool has been exhausted. Reduce USER_DATASET_SIZE or allow reuse.")
            face = rng.choice(face_pool)
            shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
        pairs.append((Path(face).stem, Path(shape).stem))
    return pairs


def main():
    face_images = list_image_files(USER_FACE_ROOT)
    shape_images = list_image_files(USER_SHAPE_ROOT)
    _assert_disjoint_stems(face_images, shape_images, "USER_FACE_ROOT", "USER_SHAPE_ROOT")

    USER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs = sample_pairs(face_images, shape_images, USER_DATASET_SIZE, USER_ALLOW_REUSE_ACROSS_PAIRS, USER_RANDOM_SEED)

    with open(USER_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f:
        for face, shape in pairs:
            f.write(f"{face} {shape}\n")

    print(f"Saved {len(pairs)} SATD pairs to {USER_OUTPUT_DIR / 'dataset.exps'}")
    print(f"face/source root: {USER_FACE_ROOT}")
    print(f"shape/reference root: {USER_SHAPE_ROOT}")


if __name__ == "__main__":
    main()
