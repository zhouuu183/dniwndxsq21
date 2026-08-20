import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random
from pathlib import Path

from utils.image_utils import list_image_files
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ========================= 用户配置区域：只改这里 =========================
USER_FFHQ_ROOT = Path("images/FFHQ")
USER_OUTPUT_DIR = Path("images/blending_dataset_v3")
USER_DATASET_SIZE = 1000
USER_RANDOM_SEED = 3407
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True
# 是否强制让 color 参考和 shape 参考来自同一张图。
# 重要：
# 1. 按论文 Section 4 / Appendix 的通用 triples 训练设定，默认应保持 False；
# 2. 论文里的 both / full 更接近推理、测试、算指标时的两种使用模式，不是两套完全不同的训练范式；
# 3. 因此如果你在做通用训练，请保持 False，让 (face, shape, color) 三个角色独立采样；
# 4. 只有当你明确要做“单参考图同时提供发型和发色”的专项实验时，才设为 True。
# True : 生成 (face, shape, shape)，适合 same-reference / both-only 专项实验
# False: 生成 (face, shape, color)，shape 和 color 独立采样，推荐作为默认训练模式
USER_COLOR_EQUALS_SHAPE = False
# ========================================================================


def sample_triplets(
    images: list[str],
    size: int,
    allow_reuse: bool,
    seed: int,
    color_equals_shape: bool,
) -> list[tuple[str, str, str]]:
    rng = random.Random(seed)
    triplets: list[tuple[str, str, str]] = []

    min_required = 2 if color_equals_shape else 3
    if len(images) < min_required:
        raise RuntimeError(f"FFHQ image count is smaller than {min_required}, cannot build requested triplets.")

    per_triplet = 2 if color_equals_shape else 3
    if not allow_reuse and size * per_triplet > len(images):
        raise RuntimeError("Not enough unique images to sample all triplets without reuse.")

    pool = images.copy()
    for _ in range(size):
        if allow_reuse:
            if color_equals_shape:
                face, shape = rng.sample(images, k=2)
                color = shape
            else:
                face, shape, color = rng.sample(images, k=3)
        else:
            if len(pool) < per_triplet:
                raise RuntimeError("The image pool has been exhausted. Reduce USER_DATASET_SIZE or allow reuse.")
            if color_equals_shape:
                face, shape = rng.sample(pool, k=2)
                color = shape
                remove_items = (face, shape)
            else:
                face, shape, color = rng.sample(pool, k=3)
                remove_items = (face, shape, color)
            for item in remove_items:
                pool.remove(item)
        triplets.append((Path(face).stem, Path(shape).stem, Path(color).stem))
    return triplets


def main():
    images = list_image_files(USER_FFHQ_ROOT)
    if not images:
        raise RuntimeError(f"No png/jpg images found under {USER_FFHQ_ROOT}")

    USER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    triplets = sample_triplets(
        images,
        USER_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
        USER_COLOR_EQUALS_SHAPE,
    )

    with open(USER_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f:
        for face, shape, color in triplets:
            f.write(f"{face} {shape} {color}\n")

    mode = "same-reference triples" if USER_COLOR_EQUALS_SHAPE else "generic triples"
    print(f"Saved {len(triplets)} triplets to {USER_OUTPUT_DIR / 'dataset.exps'} ({mode})")


if __name__ == "__main__":
    main()
