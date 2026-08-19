import os
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v4 import HairFast_v4, get_parser_v4
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ========================= 用户配置区域：只改这里 =========================
# 原图目录。这里放长发图，作为 face/source。
USER_FACE_ROOT = Path("images/FFHQ_long")

# 参考发型目录。这里放短发图，作为 shape/reference。
USER_SHAPE_ROOT = Path("images/FFHQ_short")

# 参考发色目录。默认和短发目录相同。
USER_COLOR_ROOT = USER_SHAPE_ROOT

# 输出目录。会生成 dataset.exps / FS / Align。
USER_OUTPUT_DIR = Path("input/blending_dataset_v4")

# 生成多少组三元组。
USER_DATASET_SIZE = 3000

# 随机种子。
USER_RANDOM_SEED = 3407

# 是否允许不同 triplet 之间重复使用同一张图。
USER_ALLOW_REUSE_ACROSS_TRIPLETS = True

# 是否强制让 shape 和 color 来自同一张图。
# 训练默认建议 False，保持论文通用 triples 范式。
USER_COLOR_EQUALS_SHAPE = False

# 运行设备。
USER_DEVICE = "cuda"

# StyleGAN2 权重路径。
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"

# rotate 阶段最佳权重。
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"

# 是否在生成 blending 数据时启用 SATD_v4。
# 如果为 True，必须同时提供 USER_SATD_CHECKPOINT_V4。
USER_USE_SATD_V4 = True

# 已训练好的 SATD_v4 权重。
USER_SATD_CHECKPOINT_V4 = ""

# SATD_v4 与原始 Eq.(8) 融合比例。
USER_SATD_BLEND_V4 = 0.25

# 差值边界带宽度。
USER_SATD_BOUNDARY_V4 = 5
# ========================================================================


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
            f"{label_a} and {label_b} contain duplicate stems, which would overwrite FS/Align files: {sample}"
        )


def sample_triplets(
    face_images: list[str],
    shape_images: list[str],
    color_images: list[str],
    size: int,
    allow_reuse: bool,
    seed: int,
    color_equals_shape: bool,
):
    rng = random.Random(seed)
    triplets = []
    shared_shape_color_pool = shape_images is color_images

    if not face_images:
        raise RuntimeError(f"No png/jpg images found under {USER_FACE_ROOT}")
    if not shape_images:
        raise RuntimeError(f"No png/jpg images found under {USER_SHAPE_ROOT}")
    if not color_equals_shape and not color_images:
        raise RuntimeError(f"No png/jpg images found under {USER_COLOR_ROOT}")

    if not allow_reuse:
        if size > len(face_images):
            raise RuntimeError("Not enough unique face images to sample all triplets without reuse.")
        if color_equals_shape:
            if size > len(shape_images):
                raise RuntimeError("Not enough unique shape images to sample all triplets without reuse.")
        elif shared_shape_color_pool:
            if size * 2 > len(shape_images):
                raise RuntimeError("Not enough unique short-hair images to sample distinct shape/color without reuse.")
        else:
            if size > len(shape_images) or size > len(color_images):
                raise RuntimeError("Not enough unique shape/color images to sample all triplets without reuse.")

    face_pool = face_images.copy()
    shape_pool = shape_images.copy()
    color_pool = shape_pool if shared_shape_color_pool else color_images.copy()
    for _ in range(size):
        if allow_reuse:
            if color_equals_shape:
                face = rng.choice(face_images)
                shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
                color = shape
            else:
                face = rng.choice(face_images)
                shape = _sample_excluding_stems(rng, shape_images, {Path(face).stem})
                color = _sample_excluding_stems(rng, color_images, {Path(face).stem, Path(shape).stem})
        else:
            if not face_pool or not shape_pool or (not color_equals_shape and not color_pool):
                raise RuntimeError("The image pool has been exhausted. Reduce USER_DATASET_SIZE or allow reuse.")
            if color_equals_shape:
                face = rng.choice(face_pool)
                shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
                color = shape
            else:
                face = rng.choice(face_pool)
                shape = _sample_excluding_stems(rng, shape_pool, {Path(face).stem})
                color = _sample_excluding_stems(rng, color_pool, {Path(face).stem, Path(shape).stem})
            face_pool.remove(face)
            shape_pool.remove(shape)
            if not color_equals_shape:
                color_pool.remove(color)
        triplets.append((face, shape, color))
    return triplets


def save_align_bundle(output_dir: Path, file_name: str, align_result: dict):
    delta_masks = align_result.get("delta_masks", {})
    save_latents(
        output_dir,
        "Align",
        file_name,
        latent_F=align_result["latent_F_align"],
        source_image_256=align_result["source_image_256"],
        source_inpaint_256=align_result["source_inpaint_256"],
        target_hair_mask=align_result["HM_X"],
        remove_mask=delta_masks.get("M_remove", torch.zeros_like(align_result["HM_X"])),
        remove_halo=delta_masks.get("M_remove_halo", torch.zeros_like(align_result["HM_X"])),
        body_preserve=delta_masks.get("M_body_preserve", torch.zeros_like(align_result["HM_X"])),
        remove_tail=delta_masks.get("M_remove_tail", torch.zeros_like(align_result["HM_X"])),
    )


def main():
    seed_everything(USER_RANDOM_SEED)

    if USER_USE_SATD_V4 and not USER_SATD_CHECKPOINT_V4:
        raise RuntimeError("USER_USE_SATD_V4=True 时，必须填写 USER_SATD_CHECKPOINT_V4。")

    model_args = get_parser_v4().parse_args([])
    model_args.device = USER_DEVICE
    model_args.ckpt = USER_STYLEGAN_CKPT
    model_args.rotate_checkpoint = USER_ROTATE_CKPT
    model_args.use_satd_v4 = USER_USE_SATD_V4
    model_args.satd_checkpoint_v4 = USER_SATD_CHECKPOINT_V4
    model_args.satd_blend_v4 = USER_SATD_BLEND_V4
    model_args.satd_boundary_v4 = USER_SATD_BOUNDARY_V4

    hair_fast = HairFast_v4(model_args)
    hair_fast.blend.blend_images = identity_func
    align_instead_shape(hair_fast)

    face_images = list_image_files(USER_FACE_ROOT)
    shape_images = list_image_files(USER_SHAPE_ROOT)
    if USER_COLOR_EQUALS_SHAPE:
        color_images = shape_images
    else:
        color_images = list_image_files(USER_COLOR_ROOT)

    _assert_disjoint_stems(face_images, shape_images, "USER_FACE_ROOT", "USER_SHAPE_ROOT")
    if not USER_COLOR_EQUALS_SHAPE and USER_COLOR_ROOT.resolve() != USER_FACE_ROOT.resolve():
        _assert_disjoint_stems(face_images, color_images, "USER_FACE_ROOT", "USER_COLOR_ROOT")

    triplets = sample_triplets(
        face_images,
        shape_images,
        color_images,
        USER_DATASET_SIZE,
        USER_ALLOW_REUSE_ACROSS_TRIPLETS,
        USER_RANDOM_SEED,
        USER_COLOR_EQUALS_SHAPE,
    )

    os.makedirs(USER_OUTPUT_DIR, exist_ok=True)
    with open(USER_OUTPUT_DIR / "dataset.exps", "w", encoding="utf-8") as f_exps:
        for file_1, file_2, file_3 in tqdm(triplets):
            im1, im2, im3 = map(lambda im: Path(im).stem, (file_1, file_2, file_3))
            print(im1, im2, im3, file=f_exps, flush=True)

            pt1 = USER_FACE_ROOT / file_1
            pt2 = USER_SHAPE_ROOT / file_2
            pt3 = USER_SHAPE_ROOT / file_3 if USER_COLOR_EQUALS_SHAPE else USER_COLOR_ROOT / file_3
            align_shape, align_color, name_to_embed = hair_fast(
                pt1,
                pt2,
                pt3,
                align_flag=True,
                use_satd_v4=USER_USE_SATD_V4,
                satd_blend_v4=USER_SATD_BLEND_V4,
                satd_boundary_v4=USER_SATD_BOUNDARY_V4,
            )
            save_latents(USER_OUTPUT_DIR, "FS", f"{im1}.npz", latent_in=name_to_embed["face"]["S"])
            save_latents(USER_OUTPUT_DIR, "FS", f"{im2}.npz", latent_in=name_to_embed["shape"]["S"])
            save_latents(USER_OUTPUT_DIR, "FS", f"{im3}.npz", latent_in=name_to_embed["color"]["S"])
            save_align_bundle(USER_OUTPUT_DIR, f"{im1}_{im2}.npz", align_shape)
            save_align_bundle(USER_OUTPUT_DIR, f"{im1}_{im3}.npz", align_color)

    mode = "same-reference triples" if USER_COLOR_EQUALS_SHAPE else "generic triples"
    print(f"Saved {len(triplets)} blending triplets to {USER_OUTPUT_DIR / 'dataset.exps'} ({mode})")
    print(f"face/source root: {USER_FACE_ROOT}")
    print(f"shape/reference root: {USER_SHAPE_ROOT}")
    print(f"color/reference root: {USER_COLOR_ROOT if not USER_COLOR_EQUALS_SHAPE else USER_SHAPE_ROOT}")


if __name__ == "__main__":
    main()
