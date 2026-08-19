from __future__ import annotations

import argparse
import typing as tp
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, read_image

from models.Alignment_v6 import Alignment_v6
from models.Blending_v6 import Blending_v6
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session

TImage = tp.TypeVar("TImage", torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar("TPath", Path, str)
TReturn = tp.TypeVar("TReturn", torch.Tensor, tuple[torch.Tensor, ...], dict[str, tp.Any])


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


class HairFast_v6:
    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment_v6(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending_v6(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor, **kwargs):
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ("face", "shape", "color")):
            images_to_name[image].append(name)

        return_pipeline_info = kwargs.pop("return_pipeline_info", False)
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)
        align_shape = self.align.align_images("face", "shape", name_to_embed, **kwargs)
        align_color = self.align.shape_module("face", "color", name_to_embed, **kwargs) if shape is not color else align_shape
        return self.blend.blend_images(
            align_shape,
            align_color,
            name_to_embed,
            return_aux=return_pipeline_info,
            **kwargs,
        )

    def swap(
        self,
        face_img: TImage | TPath,
        shape_img: TImage | TPath,
        color_img: TImage | TPath,
        benchmark=False,
        align=False,
        seed=None,
        exp_name=None,
        **kwargs,
    ) -> TReturn:
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                if img not in path_to_images:
                    path_to_images[img] = read_image(str(img), mode=ImageReadMode.RGB)
                img = path_to_images[img]
            else:
                raise TypeError(f"Unsupported image format {type(img)}")
            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        output = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)
        if align and not isinstance(output, dict):
            return output, *images
        return output

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser_v6():
    parser = argparse.ArgumentParser(description="HairFast v6")
    parser.add_argument("--save_all_dir", type=Path, default=Path("output"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--ckpt", type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--mixing", type=float, default=0.95)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--rotate_checkpoint", type=str, default="pretrained_models/Rotate/rotate_best.pth")
    parser.add_argument("--blending_checkpoint", type=str, default="pretrained_models/Blending/checkpoint.pth")
    parser.add_argument("--pp_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--pp_v6_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--pp_v6_use_mod", type=str2bool, default=True)
    parser.add_argument("--pp_v6_finetune", action="store_true")
    parser.add_argument("--align_v6_mode", type=str, default="author_cleanup")
    parser.add_argument("--align_v6_cleanup_strength", type=float, default=0.70)
    parser.add_argument("--align_v6_fill_iterations", type=int, default=3)
    parser.add_argument("--align_v6_fill_dilation", type=int, default=2)
    parser.add_argument("--align_v6_face_structure_weight", type=float, default=0.90)
    parser.add_argument("--align_v6_ear_structure_weight", type=float, default=0.96)
    parser.add_argument("--align_v6_body_structure_weight", type=float, default=0.88)
    parser.add_argument("--align_v6_background_fill_weight", type=float, default=0.82)
    parser.add_argument("--align_v6_other_structure_weight", type=float, default=1.00)
    parser.add_argument("--diff_mask_dilate", type=int, default=5)
    parser.add_argument("--diff_mask_blur_kernel", type=int, default=11)
    parser.add_argument("--diff_mask_blur_sigma", type=float, default=0.0)
    return parser


if __name__ == "__main__":
    args = get_parser_v6().parse_args()
    HairFast_v6(args)
