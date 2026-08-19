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

from models.Alignment_v7 import Alignment_v7
from models.Blending_v7 import Blending_v7
from models.Embedding import Embedding
from models.Net import Net
from models.ShadowCleanup import ShadowCleanup
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shadow_cleanup_masks import build_shadow_cleanup_masks
from utils.shape_predictor import align_face
from utils.time import bench_session

TImage = tp.TypeVar('TImage', torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar('TPath', Path, str)
TReturn = tp.TypeVar('TReturn', torch.Tensor, tuple[torch.Tensor, ...])


class HairFast_v7:
    """
    HairFast v7 with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment_v7(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending_v7(args, net=self.net)
        self.shadow_cleanup = ShadowCleanup(args)

    @staticmethod
    def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    def _apply_shadow_cleanup(
        self,
        face: torch.Tensor,
        final_image: torch.Tensor,
        name_to_embed,
        align_shape,
        **kwargs,
    ) -> torch.Tensor:
        use_cleanup = kwargs.get('use_shadow_cleanup', getattr(self.args, 'use_shadow_cleanup', False))
        if not use_cleanup:
            return final_image

        cleanup_masks = build_shadow_cleanup_masks(
            source_parsing=name_to_embed['face']['mask'],
            target_hair_mask=align_shape['HM_X'],
            ring_width=kwargs.get('shadow_cleanup_ring', getattr(self.args, 'shadow_cleanup_ring', 7)),
            halo_width=kwargs.get('shadow_cleanup_halo', getattr(self.args, 'shadow_cleanup_halo', 9)),
            protect_width=kwargs.get('shadow_cleanup_protect', getattr(self.args, 'shadow_cleanup_protect', 2)),
        )

        cleaned = self.shadow_cleanup(
            source_image=self._ensure_batch(face.to(final_image.device)),
            base_image=self._ensure_batch(final_image.to(self.args.device)),
            cleanup_masks={key: value.to(final_image.device) for key, value in cleanup_masks.items()},
            strength=kwargs.get('shadow_cleanup_strength', getattr(self.args, 'shadow_cleanup_strength', 0.75)),
            source_blend=kwargs.get('shadow_cleanup_source_blend', getattr(self.args, 'shadow_cleanup_source_blend', 0.55)),
            kernel_size=kwargs.get('shadow_cleanup_kernel', getattr(self.args, 'shadow_cleanup_kernel', 21)),
        )
        return cleaned[0]

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)
        align_shape = self.align.align_images('face', 'shape', name_to_embed, **kwargs)

        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        final_image = self._apply_shadow_cleanup(face, final_image, name_to_embed, align_shape, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
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


def get_parser_v7():
    parser = argparse.ArgumentParser(description='HairFast v7')

    parser.add_argument('--save_all_dir', type=Path, default=Path('output'))
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size for encoding images')
    parser.add_argument('--save_all', action='store_true', help='save and print mode information')
    parser.add_argument('--mixing', type=float, default=0.95, help='hair blending in alignment')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')
    parser.add_argument('--align_mask_size', type=int, default=32,
                        help='resolution of the alignment mask used for F-space mixing')
    parser.add_argument('--rotate_checkpoint', type=str, default='pretrained_models/Rotate/rotate_best.pth')
    parser.add_argument('--blending_checkpoint', type=str, default='pretrained_models/Blending/checkpoint.pth')
    parser.add_argument('--pp_checkpoint', type=str, default='pretrained_models/PostProcess/pp_model.pth')
    parser.add_argument('--disable_post_process', action='store_true',
                        help='skip the post-process module and return the blending output directly')
    parser.add_argument('--fringe_crop_size', type=int, default=128,
                        help='crop size used by the fringe detail branch')
    parser.add_argument('--fringe_crop_padding', type=int, default=16,
                        help='extra padding around the detected fringe ROI')
    parser.add_argument('--fringe_mask_channels', type=int, default=32,
                        help='hidden channels for the fringe mask head')
    parser.add_argument('--fringe_mask_prior_weight', type=float, default=0.8,
                        help='weight of the heuristic prior inside the fringe mask head')
    parser.add_argument('--fringe_boundary_kernel', type=int, default=9,
                        help='kernel size used to build the fringe boundary prior')
    parser.add_argument('--fringe_roi_kernel', type=int, default=25,
                        help='kernel size used to expand the fringe ROI prior')
    parser.add_argument('--fringe_encoder_channels', type=int, default=32,
                        help='base channels for the local fringe encoder')
    parser.add_argument('--fringe_attention_channels', type=int, default=128,
                        help='attention width for the fringe fusion block')
    parser.add_argument('--fringe_decoder_channels', type=int, default=64,
                        help='hidden channels for the fringe residual head')
    parser.add_argument('--use_shadow_cleanup', action='store_true')
    parser.add_argument('--shadow_cleanup_strength', type=float, default=0.75)
    parser.add_argument('--shadow_cleanup_source_blend', type=float, default=0.55)
    parser.add_argument('--shadow_cleanup_kernel', type=int, default=21)
    parser.add_argument('--shadow_cleanup_ring', type=int, default=7)
    parser.add_argument('--shadow_cleanup_halo', type=int, default=9)
    parser.add_argument('--shadow_cleanup_protect', type=int, default=2)
    return parser


if __name__ == '__main__':
    model_args = get_parser_v7()
    args = model_args.parse_args()
    hair_fast = HairFast_v7(args)
