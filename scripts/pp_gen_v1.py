import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm

# v1 bugfix:
# Allow repo-root imports when the script is launched as
# `python scripts/pp_gen_v1.py`.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hair_swap_v1 import HairFast, get_parser
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.image_utils import list_image_files
from utils.train import seed_everything


# ============================================================
# 用户配置区域：只改这里
# ------------------------------------------------------------
# 这里填：FFHQ 数据集目录
USER_FFHQ_ROOT = Path("input/FFHQ")
#
# 这里填：第一阶段 rotate 训练好的 checkpoint
USER_ROTATE_CKPT = "pretrained_models/Rotate/rotate_best.pth"
#
# 这里填：StyleGAN 权重
USER_STYLEGAN_CKPT = "pretrained_models/StyleGAN/ffhq.pt"
#
# 这里填：第二阶段 blending_v1 训练好的 checkpoint
USER_BLENDING_V1_CKPT = "pretrained_models/Blending_v1/best.pth"
#
# 这里填：如果你想加载已有 pp_v1 checkpoint，可以写路径；否则保持 None
USER_PP_V1_CKPT = None
#
# 这里填：输出的第三阶段数据集目录
USER_OUTPUT_DIR = Path("input/pp_dataset_v1")
#
# 这里填：要生成多少组三元组样本
USER_DATASET_SIZE = 10000
#
# 这里填：运行设备
USER_DEVICE = "cuda"
#
# 这里填：Embedding/Alignment/Blending 阶段 batch size
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
# pp dataset generation now reads explicit remove/boundary masks from the new
# blending stage instead of treating all artifacts as a single generic region.


def load_image(path: Path) -> torch.Tensor:
    return T.functional.to_tensor(Image.open(path).convert("RGB"))


class MaskExtractorV1:
    def __init__(self, device: str):
        self.device = device
        self.seg = BiSeNet(n_classes=16).to(device).eval()
        self.seg.load_state_dict(torch.load("pretrained_models/BiSeNet/seg.pth"))
        self.downsample_512 = BicubicDownSample(factor=2)

    @torch.no_grad()
    def generate_mask(self, image_0_1: torch.Tensor):
        IM = (self.downsample_512(image_0_1) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        hair_mask = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        hair_mask = F.interpolate(hair_mask.unsqueeze(1), size=(256, 256), mode="nearest")
        hair_d = F.max_pool2d(hair_mask, kernel_size=5, stride=1, padding=2)
        hair_e = 1.0 - F.max_pool2d(1.0 - hair_mask, kernel_size=5, stride=1, padding=2)
        return hair_d, hair_e


def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_parser = get_parser()
    model_args = model_parser.parse_args([])
    # v1 bugfix:
    # Do not silently fall back to default upstream checkpoints when generating
    # the pp dataset.
    model_args.device = args.device
    model_args.batch_size = args.batch_size
    model_args.mixing = args.mixing
    model_args.smooth = args.smooth
    model_args.delta_boundary = args.delta_boundary
    model_args.rotate_checkpoint = args.rotate_checkpoint
    model_args.ckpt = args.ckpt
    model_args.blending_checkpoint_v1 = args.blending_checkpoint_v1
    model_args.pp_checkpoint_v1 = args.pp_checkpoint_v1
    hair_fast = HairFast(model_args, init_blending=True)
    mask_extractor = MaskExtractorV1(device)

    os.makedirs(args.output, exist_ok=True)
    images = list_image_files(args.FFHQ)
    face, shape, color = np.array_split(np.random.choice(images, size=3 * args.size), 3)

    dataset = []
    for exp in tqdm(list(zip(face, shape, color))):
        face_name, shape_name, color_name = exp
        result, intermediates = hair_fast(
            args.FFHQ / face_name,
            args.FFHQ / shape_name,
            args.FFHQ / color_name,
            stop_before_pp=True,
            return_intermediates=True,
        )

        del result
        source = load_image(args.FFHQ / face_name).unsqueeze(0).to(device)
        target = intermediates["pre_pp_image_256"].unsqueeze(0).to(device)

        source_hair_d, _ = mask_extractor.generate_mask(source)
        target_hair_d, target_hair_e = mask_extractor.generate_mask(target)
        target_mask = (1 - source_hair_d) * (1 - target_hair_d)

        delta_masks = intermediates["delta_masks"]
        dataset.append(
            (
                str(args.FFHQ / face_name),
                intermediates["pre_pp_image_256"].cpu(),
                target_mask.cpu(),
                target_hair_e.cpu(),
                delta_masks["M_remove"].cpu(),
                delta_masks["M_boundary"].cpu(),
            )
        )

    torch.save(dataset, args.output / "pp_delta_v1.dataset")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate v1 pp dataset")
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
    parser.add_argument("--blending_checkpoint_v1", type=str, default=USER_BLENDING_V1_CKPT)
    parser.add_argument("--pp_checkpoint_v1", type=str, default=USER_PP_V1_CKPT)
    args = parser.parse_args()

    main(args)
