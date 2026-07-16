import argparse
import typing as tp
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.io import read_image, ImageReadMode

from models.Alignment import Alignment
from models.Blending_e1 import Blending
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session

TImage = tp.TypeVar('TImage', torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar('TPath', Path, str)
TReturn = tp.TypeVar('TReturn', torch.Tensor, tuple[torch.Tensor, ...])


class HairFast:
    """
    HairFast implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending(args, net=self.net)

    @staticmethod
    def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        blend_kwargs = dict(kwargs)
        blend_kwargs.setdefault('source_full_01', self._ensure_batch(face).to(self.args.device))
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **blend_kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFast on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return final_image, *images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser(description='HairFast')

    # I/O arguments
    parser.add_argument('--save_all_dir', type=Path, default=Path('output'),
                        help='the directory to save the latent codes and inversion images')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # Arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size for encoding images')
    parser.add_argument('--save_all', action='store_true', help='save and print mode information')

    # HairFast setting
    parser.add_argument('--mixing', type=float, default=0.95, help='hair blending in alignment')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')
    parser.add_argument('--rotate_checkpoint', type=str, default='pretrained_models/Rotate/rotate_best.pth')
    parser.add_argument('--blending_checkpoint', type=str, default='pretrained_models/Blending/checkpoint.pth')
    parser.add_argument('--pp_checkpoint', type=str, default='pretrained_models/PostProcess/pp_model.pth')

    # Optional direct earring restore after the original author PostProcessModel.
    parser.add_argument('--use_earring_direct_restore', action='store_true')
    parser.add_argument('--earring_restore_debug', action='store_true')
    parser.add_argument('--earring_restore_alpha_strength', type=float, default=0.92)
    parser.add_argument('--earring_restore_ear_roi_dilate', type=int, default=9)
    parser.add_argument('--earring_restore_ear_lobe_shift', type=int, default=10)
    parser.add_argument('--earring_restore_target_hair_exclude', type=int, default=5)
    parser.add_argument('--earring_restore_source_hair_exclude', type=int, default=2)
    parser.add_argument('--earring_restore_component_min_area', type=int, default=3)
    parser.add_argument('--earring_restore_component_keep_top', type=int, default=6)
    parser.add_argument('--earring_restore_align_max_shift', type=int, default=0)
    parser.add_argument('--earring_restore_object_grow_iters', type=int, default=1)
    parser.add_argument('--earring_restore_parser_weak_support_dilate', type=int, default=4)
    parser.add_argument('--disable_earring_restore_weak_fallback', action='store_true')
    parser.add_argument('--earring_restore_safe_region_dilate', type=int, default=2)
    parser.add_argument('--earring_restore_alpha_feather_dilate', type=int, default=1)
    parser.add_argument('--earring_restore_alpha_feather_strength', type=float, default=0.35)
    parser.add_argument('--earring_restore_alpha_blur_kernel', type=int, default=3)
    parser.add_argument('--earring_restore_alpha_blur_sigma', type=float, default=1.4)
    return parser


if __name__ == '__main__':
    model_args = get_parser()
    args = model_args.parse_args()
    hair_fast = HairFast(args)
