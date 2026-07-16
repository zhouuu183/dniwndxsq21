import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

# v1 bugfix:
# Running `python scripts/blending_gen_v1.py` sets sys.path to the scripts
# directory, so repo-root imports such as `hair_swap_v1` are not visible unless
# we append the project root explicitly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v1 import HairFast, get_parser
from utils.image_utils import list_image_files
from utils.save_utils import save_latents
from utils.train import seed_everything


# ============================================================
# 用户配置区域：只改这里
# ------------------------------------------------------------
# 这里填：FFHQ 数据集目录
USER_FFHQ_ROOT = Path("images/FFHQ")
#
# 这里填：第一阶段 rotate 训练好的 checkpoint
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
#
# 这里填：StyleGAN 权重
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
#
# 这里填：输出的第二阶段数据集目录
USER_OUTPUT_DIR = Path("images/blending_dataset_v1")
#
# 这里填：要生成多少组三元组样本
USER_DATASET_SIZE = 3000
#
# 这里填：运行设备
USER_DEVICE = "cuda"
#
# 这里填：Embedding/Alignment 阶段 batch size
USER_BATCH_SIZE = 3
#
# 这里填：mixing 系数
USER_MIXING = 0.95
#
# 这里填：mask 平滑参数
USER_SMOOTH = 5
#
# 这里填：delta 边界宽度
USER_DELTA_BOUNDARY = 5
# ============================================================


# v1 modification:
# This generator exports explicit delta masks for the new blending stage.
# The old generator only saved FS latents and aligned F codes.


def _hair_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask == 13, torch.ones_like(mask), torch.zeros_like(mask)).float()


def main(args):
    seed_everything(args.seed)

    model_parser = get_parser()
    model_args = model_parser.parse_args([])
    # v1 bugfix:
    # Expose the upstream model arguments that this dataset generator depends on
    # instead of silently using parser defaults.
    model_args.device = args.device
    model_args.batch_size = args.batch_size
    model_args.mixing = args.mixing
    model_args.smooth = args.smooth
    model_args.delta_boundary = args.delta_boundary
    model_args.rotate_checkpoint = args.rotate_checkpoint
    model_args.ckpt = args.ckpt
    hair_fast = HairFast(model_args, init_blending=False)

    images = list_image_files(args.FFHQ)
    face, shape, color = np.array_split(np.random.choice(images, size=3 * args.size), 3)

    os.makedirs(args.output, exist_ok=True)
    with open(args.output / "dataset.exps", "w") as f_exps:
        for imgs in tqdm(list(zip(face, shape, color))):
            im1, im2, im3 = map(lambda im: Path(im).stem, imgs)
            print(im1, im2, im3, file=f_exps, flush=True)

            pt1, pt2, pt3 = map(lambda im: args.FFHQ / im, imgs)
            name_to_embed, align_shape, align_color, _ = hair_fast.encode_and_align(pt1, pt2, pt3)

            # v1 modification:
            # FS now stores the segmentation-derived hair mask together with the latent.
            save_latents(
                args.output,
                "FS",
                f"{im1}.npz",
                latent_in=name_to_embed["face"]["S"],
                mask=_hair_mask(name_to_embed["face"]["mask"]),
            )
            save_latents(
                args.output,
                "FS",
                f"{im2}.npz",
                latent_in=name_to_embed["shape"]["S"],
                mask=_hair_mask(name_to_embed["shape"]["mask"]),
            )
            save_latents(
                args.output,
                "FS",
                f"{im3}.npz",
                latent_in=name_to_embed["color"]["S"],
                mask=_hair_mask(name_to_embed["color"]["mask"]),
            )

            save_latents(
                args.output,
                "Align",
                f"{im1}_{im2}.npz",
                latent_F_align=align_shape["latent_F_align"],
            )
            save_latents(
                args.output,
                "Align",
                f"{im1}_{im3}.npz",
                latent_F_align=align_color["latent_F_align"],
            )

            delta_masks = align_shape["delta_masks"]
            save_latents(
                args.output,
                "Delta",
                f"{im1}_{im2}_{im3}.npz",
                M_src=delta_masks["M_src"],
                M_src_hair=delta_masks["M_src_hair"],
                M_tgt=delta_masks["M_tgt"],
                M_add=delta_masks["M_add"],
                M_remove=delta_masks["M_remove"],
                M_keep=delta_masks["M_keep"],
                M_boundary=delta_masks["M_boundary"],
                M_accessory=delta_masks["M_accessory"],
                M_color=_hair_mask(name_to_embed["color"]["mask"]),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta-aware blending dataset generator")
    # v1 modification:
    # Local user-config block above keeps this script self-contained.
    parser.add_argument("--FFHQ", type=Path, default=USER_FFHQ_ROOT)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--size", type=int, default=USER_DATASET_SIZE)
    parser.add_argument("--output", type=Path, default=USER_OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=USER_DEVICE)
    parser.add_argument("--batch_size", type=int, default=USER_BATCH_SIZE)
    parser.add_argument("--mixing", type=float, default=USER_MIXING)
    parser.add_argument("--smooth", type=int, default=USER_SMOOTH)
    parser.add_argument("--delta_boundary", type=int, default=USER_DELTA_BOUNDARY)
    parser.add_argument("--rotate_checkpoint", type=str, default=USER_ROTATE_CKPT)
    parser.add_argument("--ckpt", type=str, default=USER_STYLEGAN_CKPT)
    args = parser.parse_args()

    main(args)
