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

from models.Alignment_v1 import Alignment
from models.Blending_v1 import Blending
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session

TImage = tp.TypeVar("TImage", torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar("TPath", Path, str)
TReturn = tp.TypeVar("TReturn", torch.Tensor, tuple[torch.Tensor, ...], tuple[torch.Tensor, dict[str, tp.Any]])


# v1 modification:
# HairFast_v1 exposes an explicit encode_and_align step so blending/pp data
# can be regenerated without loading the whole old pipeline.


class HairFast:
    def __init__(self, args, init_blending: bool = True):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending(args, net=self.net) if init_blending else None

    def _load_inputs(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath, align=False):
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
        return equal_replacer(images)

    @seed_setter
    @bench_session
    def _encode_and_align_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor, **kwargs):
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ("face", "shape", "color")):
            images_to_name[image].append(name)

        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)
        align_shape = self.align.align_images("face", "shape", name_to_embed, **kwargs)
        align_color = self.align.align_images("face", "color", name_to_embed, **kwargs)
        return name_to_embed, align_shape, align_color

    def encode_and_align(
        self,
        face_img: TImage | TPath,
        shape_img: TImage | TPath,
        color_img: TImage | TPath,
        align=False,
        seed=None,
        exp_name=None,
        benchmark=False,
        **kwargs,
    ):
        images = self._load_inputs(face_img, shape_img, color_img, align=align)
        name_to_embed, align_shape, align_color = self._encode_and_align_from_tensors(
            *images,
            seed=seed,
            exp_name=exp_name,
            benchmark=benchmark,
            **kwargs,
        )
        return name_to_embed, align_shape, align_color, images

    def swap(
        self,
        face_img: TImage | TPath,
        shape_img: TImage | TPath,
        color_img: TImage | TPath,
        benchmark=False,
        align=False,
        seed=None,
        exp_name=None,
        return_intermediates=False,
        **kwargs,
    ) -> TReturn:
        if self.blend is None:
            raise RuntimeError("HairFast_v1 was initialized without the blending stage.")

        name_to_embed, align_shape, align_color, images = self.encode_and_align(
            face_img,
            shape_img,
            color_img,
            benchmark=benchmark,
            align=align,
            seed=seed,
            exp_name=exp_name,
            **kwargs,
        )

        output = self.blend.blend_images(
            align_shape,
            align_color,
            name_to_embed,
            exp_name=exp_name,
            return_intermediates=return_intermediates,
            **kwargs,
        )

        if align:
            if return_intermediates:
                final_image, intermediates = output
                return (final_image, intermediates, *images)
            return output, *images
        return output

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser(description="HairFast_v1")
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
    parser.add_argument("--delta_boundary", type=int, default=5, help="v1 delta mask boundary width")
    parser.add_argument("--rotate_checkpoint", type=str, default="pretrained_models/Rotate/rotate_best.pth")

    # v1 modification:
    # New architecture checkpoints are separated from the legacy checkpoint names
    # so the baseline files remain untouched.
    parser.add_argument("--blending_checkpoint_v1", type=str, default=None)
    parser.add_argument("--pp_checkpoint_v1", type=str, default=None)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    HairFast(args)
