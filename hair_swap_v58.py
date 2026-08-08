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

from models.Alignment_v8 import Alignment_v8
from models.Blending_v58 import BlendingV5
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session

TImage = tp.TypeVar("TImage", torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar("TPath", Path, str)
TReturn = tp.TypeVar("TReturn", torch.Tensor, tuple[torch.Tensor, ...])


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value}")


class HairFastV5:
    def __init__(self, args):
        self.args = args
        if getattr(self.args, "use_satd_v8", False) and not getattr(self.args, "satd_checkpoint_v8", ""):
            print("[HairFastV5] use_satd_v8=True but satd_checkpoint_v8 is empty; disabling SATD_v8 cleanup.")
            self.args.use_satd_v8 = False
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        # v5 now uses the v8 shadow-cleaned alignment path before the
        # ear-aware post-process stage.
        self.align = Alignment_v8(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = BlendingV5(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ("face", "shape", "color")):
            images_to_name[image].append(name)

        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)
        align_shape = self.align.align_images("face", "shape", name_to_embed, **kwargs)
        align_color = self.align.shape_module("face", "color", name_to_embed, **kwargs) if shape is not color else align_shape
        return self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
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

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)
        return (final_image, *images) if align else final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser(description="HairFast V5")
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
    parser.add_argument("--pp_v5_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--pp_v58_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--use_satd_v8", type=str2bool, default=False)
    parser.add_argument("--satd_checkpoint_v8", type=str, default="")
    parser.add_argument("--satd_blend_v8", type=float, default=0.28)
    parser.add_argument("--satd_boundary_v8", type=int, default=8)
    parser.add_argument("--eq8_reference_blend_v8", type=float, default=0.0)
    parser.add_argument("--pp_v5_use_mod", type=str2bool, default=True)
    parser.add_argument("--pp_v5_use_full", type=str2bool, default=True)
    parser.add_argument("--pp_v58_use_mod", type=str2bool, default=True)
    parser.add_argument("--pp_v58_use_full", type=str2bool, default=True)
    parser.add_argument("--ear_parse_size", type=int, default=512)
    parser.add_argument("--ear_feature_channels", type=int, default=128)
    parser.add_argument("--ear_low_alpha", type=float, default=0.1)
    parser.add_argument("--ear_dilate", type=int, default=21)
    parser.add_argument("--hair_change_dilate", type=int, default=25)
    parser.add_argument("--earring_expand", type=int, default=15)
    parser.add_argument("--ear_downward_shift", type=int, default=10)
    parser.add_argument("--target_hair_dilate", type=int, default=11)
    parser.add_argument("--earring_occlusion_dilate", type=int, default=3)
    parser.add_argument("--source_hair_block_dilate", type=int, default=5)
    parser.add_argument("--source_hair_block_strength", type=float, default=0.6)
    parser.add_argument("--target_visibility_expand", type=int, default=5)
    parser.add_argument("--max_target_hair_overlap", type=float, default=0.55)
    parser.add_argument("--min_target_visible_overlap", type=float, default=0.02)
    parser.add_argument("--min_target_ear_area", type=float, default=8.0)
    parser.add_argument("--earring_channel_down", type=int, default=32)
    parser.add_argument("--ear_blur_kernel", type=int, default=11)
    parser.add_argument("--ear_blur_sigma", type=float, default=3.0)
    parser.add_argument("--ear_mask_hidden", type=int, default=32)
    parser.add_argument("--earring_query_dilate", type=int, default=3)
    parser.add_argument("--earring_query_boost", type=float, default=1.0)
    parser.add_argument("--enable_earring_query_recall", type=str2bool, default=True)
    parser.add_argument("--earring_query_recall_dilate", type=int, default=7)
    parser.add_argument("--earring_query_downward_shift", type=int, default=18)
    parser.add_argument("--earring_query_lower_lobe_weight", type=float, default=0.20)
    parser.add_argument("--earring_query_candidate_boost", type=float, default=0.90)
    parser.add_argument("--earring_query_block_protect", type=float, default=0.85)
    parser.add_argument("--earring_align_max_shift", type=int, default=12)
    parser.add_argument("--ear_fine_support_dilate", type=int, default=3)
    parser.add_argument("--earring_fine_mask_floor", type=float, default=0.18)
    parser.add_argument("--earring_fine_mask_dilate", type=int, default=5)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    hair_fast = HairFastV5(args)
