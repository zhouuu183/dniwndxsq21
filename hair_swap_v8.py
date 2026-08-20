import argparse
import os
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
from models.Blending_v8 import Blending_v8
from models.Embedding import Embedding
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session

TImage = tp.TypeVar("TImage", torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar("TPath", Path, str)
TReturn = tp.TypeVar("TReturn", torch.Tensor, tuple[torch.Tensor, ...])


class HairFast_v8:
    """
    HairFast v8:
    author baseline alignment + SATD-v5 cleanup specialist.
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment_v8(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending_v8(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor, **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ("face", "shape", "color")):
            images_to_name[image].append(name)

        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)
        align_shape = self.align.align_images("face", "shape", name_to_embed, **kwargs)
        if shape is not color:
            align_color = self.align.shape_module("face", "color", name_to_embed, **kwargs)
        else:
            align_color = align_shape
        return self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)

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
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f"Unsupported image format {type(img)}")
            images.append(img)

        if align:
            images = align_face(images)
        if getattr(self.args, "v235_enabled", False):
            # V2.35 needs an independent color embedding even when the caller
            # supplies the same image for shape/color (or face/color). Keeping
            # the color tensor separate prevents role-key coalescing from
            # leaking hair-only preprocessing into the face/shape branches.
            images = [image / 255 if image.dtype == torch.uint8 else image for image in images]
            images[2] = images[2].clone()
        else:
            images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)
        if align:
            return final_image, *images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser_v8():
    parser = argparse.ArgumentParser(description="HairFast v8")
    parser.add_argument("--save_all_dir", type=Path, default=Path("output"))
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--ckpt", type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=3, help="batch size for encoding images")
    parser.add_argument("--save_all", action="store_true", help="save and print mode information")
    parser.add_argument("--mixing", type=float, default=0.95, help="hair blending in alignment")
    parser.add_argument("--smooth", type=int, default=5, help="dilation and erosion parameter")
    parser.add_argument("--rotate_checkpoint", type=str, default="pretrained_models/Rotate/rotate_best.pth")
    parser.add_argument(
        "--blending_checkpoint",
        type=str,
        default="output/blending_train_v8_direct_anchor_v2_26_boundary_target_metric_aligned_small/checkpoints/v226_boundary_target_metric_aligned_pass.pth",
    )
    parser.add_argument("--pp_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--disable-v228", dest="v228_enabled", action="store_false")
    parser.set_defaults(v228_enabled=True)
    parser.add_argument("--v228-matte-width", type=int, default=4)
    parser.add_argument("--v228-matte-max-distance", type=int, default=8)
    parser.add_argument("--v228-bg-residual-radius", type=int, default=7)
    parser.add_argument("--v228-bg-residual-strength", type=float, default=1.0)
    parser.add_argument("--v228-outer-ring-width", type=int, default=6)
    parser.add_argument("--disable-v229", dest="v229_enabled", action="store_false")
    parser.set_defaults(v229_enabled=True)
    parser.add_argument("--v229-carrier-low-radius", type=int, default=5)
    parser.add_argument("--v229-anchor-hf-gain", type=float, default=1.0)
    parser.add_argument("--v229-tone-radius", type=int, default=7)
    parser.add_argument("--v229-background-radius", type=int, default=7)
    parser.add_argument("--v229-outer-strand-recovery", action="store_true")
    parser.add_argument("--disable-v230", dest="v230_enabled", action="store_false")
    parser.set_defaults(v230_enabled=True)
    parser.add_argument("--v230-core-low-radius", type=int, default=5)
    parser.add_argument("--v230-anchor-detail-radius", type=int, default=5)
    parser.add_argument("--v230-detail-log-cap", type=float, default=0.25)
    parser.add_argument("--v230-anchor-detail-gain", type=float, default=1.0)
    parser.add_argument("--v230-pp-tone-radius", type=int, default=5)
    parser.add_argument("--v230-tone-radius", type=int, default=7)
    parser.add_argument("--v230-core-seam-width", type=int, default=5)
    parser.add_argument("--v230-face-contact-width", type=int, default=4)
    parser.add_argument("--v230-confidence-temperature", type=float, default=8.0)
    parser.add_argument("--v230-topology-hole-radius", type=int, default=2)
    parser.add_argument("--v230-topology-neighbor-threshold", type=float, default=0.75)
    parser.add_argument("--disable-v231", dest="v231_enabled", action="store_false")
    parser.set_defaults(v231_enabled=True)
    parser.add_argument(
        "--v231-vitmatte-path",
        type=str,
        default=os.environ.get(
            "BLENDING_V8_VITMATTE_PATH",
            "pretrained_models/ViTMatte/vitmatte-small-composition-1k",
        ),
    )
    parser.add_argument("--v231-trimap-inner-width", type=int, default=8)
    parser.add_argument("--v231-trimap-outer-width", type=int, default=8)
    parser.add_argument("--v231-face-contact-extra-inner", type=int, default=4)
    parser.add_argument("--v231-max-trimap-hole-area", type=int, default=16)
    parser.add_argument("--v231-tone-radius", type=int, default=9)
    parser.add_argument("--v231-tone-propagation-radius", type=int, default=15)
    parser.add_argument("--v231-context-radius", type=int, default=9)
    parser.add_argument("--v231-transition-expand", type=int, default=2)
    parser.add_argument("--v231-disable-phase-c", dest="v231_phase_c", action="store_false")
    parser.set_defaults(v231_phase_c=True)
    parser.add_argument("--disable-v232", dest="v232_enabled", action="store_false")
    parser.set_defaults(v232_enabled=True)
    parser.add_argument("--v232-foreground-roi-padding", type=int, default=64)
    parser.add_argument("--v232-foreground-cache", type=str, default="")
    parser.add_argument("--v232-tone-radius", type=int, default=9)
    parser.add_argument("--v232-residual-radius", type=int, default=21)
    parser.add_argument("--v232-face-prototype-radius", type=int, default=7)
    parser.add_argument("--v232-face-temperature", type=float, default=0.04)
    parser.add_argument("--v232-background-radius", type=int, default=9)
    parser.add_argument("--v232-transition-expand", type=int, default=2)
    parser.add_argument("--disable-v233", dest="v233_enabled", action="store_false")
    parser.set_defaults(v233_enabled=True)
    parser.add_argument("--v233-tone-radius", type=int, default=9)
    parser.add_argument("--v233-local-radius", type=int, default=21)
    parser.add_argument("--v233-propagation-radius", type=int, default=15)
    parser.add_argument("--v233-detail-gain", type=float, default=0.5)
    parser.add_argument("--v233-posterior-radius", type=int, default=5)
    parser.add_argument("--v233-face-temperature", type=float, default=0.04)
    parser.add_argument("--v233-background-radius", type=int, default=9)
    parser.add_argument("--v233-support-full", type=float, default=0.20)
    parser.add_argument("--disable-v234", dest="v234_enabled", action="store_false")
    parser.set_defaults(v234_enabled=True)
    parser.add_argument("--v234-reference-radius", type=int, default=9)
    parser.add_argument("--v234-edge-radius", type=int, default=3)
    parser.add_argument("--v234-edge-min-confidence", type=float, default=0.35)
    parser.add_argument("--disable-v235", dest="v235_enabled", action="store_false")
    parser.set_defaults(v235_enabled=True)
    parser.add_argument("--v235-chroma-radius", type=int, default=9)
    parser.add_argument("--v235-boundary-radius", type=int, default=3)
    parser.add_argument("--v235-boundary-min-confidence", type=float, default=0.20)
    parser.add_argument("--v235-chroma-scale", type=float, default=18.0)
    parser.add_argument("--use_satd_v8", action="store_true")
    parser.add_argument("--satd_checkpoint_v8", type=str, default="")
    parser.add_argument("--satd_blend_v8", type=float, default=0.28)
    parser.add_argument("--satd_boundary_v8", type=int, default=8)
    parser.add_argument("--eq8_reference_blend_v8", type=float, default=0.0)
    parser.add_argument("--ab-no-edit-threshold-v8", type=float, default=1.5)
    parser.add_argument("--ab-full-edit-threshold-v8", type=float, default=15.0)
    parser.add_argument("--hue-no-edit-deg-v8", type=float, default=4.0)
    parser.add_argument("--hue-full-edit-deg-v8", type=float, default=30.0)
    parser.add_argument("--chroma-mag-no-edit-v8", type=float, default=2.0)
    parser.add_argument("--chroma-mag-full-edit-v8", type=float, default=15.0)
    parser.add_argument("--color-dist-no-edit-v8", type=float, default=2.0)
    parser.add_argument("--color-dist-full-edit-v8", type=float, default=15.0)
    parser.add_argument("--lightness-no-edit-threshold-v8", type=float, default=3.0)
    parser.add_argument("--lightness-full-edit-threshold-v8", type=float, default=15.0)
    parser.add_argument("--max-global-l-shift-v8", type=float, default=40.0)
    parser.add_argument("--relative-luma-bins-v8", type=int, default=8)
    parser.add_argument("--relative-luma-min-scale-v8", type=float, default=3.0)
    parser.add_argument("--global-ab-fallback-min-reliability-v8", type=float, default=0.5)
    parser.add_argument("--min-safe-reference-fraction-v8", type=float, default=0.35)
    parser.add_argument("--alpha-init-v8", type=float, default=0.70)
    parser.add_argument("--layer-offset-max-v8", type=float, default=0.15)
    parser.add_argument("--correction-chroma-budget-ratio-v8", type=float, default=0.15)
    parser.add_argument("--correction-luma-budget-ratio-v8", type=float, default=0.10)
    parser.add_argument("--correction-orth-scale-v8", type=float, default=0.25)
    return parser


if __name__ == "__main__":
    args = get_parser_v8().parse_args()
    HairFast_v8(args)
