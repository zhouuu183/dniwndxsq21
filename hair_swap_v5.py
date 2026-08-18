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
from models.Blending_v5 import BlendingV5
from models.Embedding_v5 import EmbeddingV5
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
        self.embed = EmbeddingV5(args, net=self.net)
        # v5 now uses the v8 shadow-cleaned alignment path before the
        # ear-aware post-process stage.
        self.align = Alignment_v8(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = BlendingV5(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        for name, image in (("face", face), ("shape", shape), ("color", color)):
            if image.dim() == 4 and image.shape[0] != 1:
                raise ValueError(
                    f"HairFastV5 currently supports one sample per call; {name} has batch {image.shape[0]}."
                )
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ("face", "shape", "color")):
            images_to_name[image].append(name)

        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)
        align_shape = self.align.align_images("face", "shape", name_to_embed, **kwargs)
        # The deterministic colour stage reads the colour image and its parsing
        # directly.  A second face->colour shape alignment was unused and also
        # overwrote the real shape HM_X debug files.
        return self.blend.blend_images(align_shape, align_shape, name_to_embed, **kwargs)

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
    parser.add_argument(
        "--allow_legacy_blending_checkpoint_v8",
        type=str2bool,
        default=False,
        help=(
            "Allow a blending checkpoint without the v8 color_policy contract. "
            "Use only for cache construction or deliberate legacy inference."
        ),
    )
    parser.add_argument("--pp_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--pp_v5_checkpoint", type=str, default="pretrained_models/PostProcess/pp_model.pth")
    parser.add_argument("--use_satd_v8", type=str2bool, default=False)
    parser.add_argument("--satd_checkpoint_v8", type=str, default="")
    parser.add_argument("--satd_blend_v8", type=float, default=0.28)
    parser.add_argument("--satd_boundary_v8", type=int, default=8)
    parser.add_argument("--eq8_reference_blend_v8", type=float, default=0.0)
    parser.add_argument("--target_hair_close_kernel", type=int, default=9)
    parser.add_argument("--target_hair_hole_max_area", type=float, default=None)
    parser.add_argument("--target_hair_hole_max_area_ratio", type=float, default=0.003)
    parser.add_argument("--target_hair_hole_min_prior_coverage", type=float, default=0.10)
    parser.add_argument("--target_hair_hole_prior_evidence_radius", type=int, default=2)
    parser.add_argument("--target_hair_top_fill_only", type=str2bool, default=True)
    parser.add_argument("--target_hair_ear_bridge_radius", type=int, default=3)
    parser.add_argument("--target_hair_crown_repair_enabled", type=str2bool, default=True)
    parser.add_argument("--target_hair_crown_height_ratio", type=float, default=0.50)
    parser.add_argument("--target_hair_crown_bridge_radius", type=int, default=8)
    parser.add_argument("--target_hair_crown_prior_dilate", type=int, default=1)
    parser.add_argument("--target_hair_crown_component_distance", type=int, default=12)
    parser.add_argument(
        "--target_hair_crown_component_min_prior_overlap",
        type=float,
        default=0.15,
    )
    parser.add_argument("--target_hair_crown_max_added_area_ratio", type=float, default=0.008)
    parser.add_argument("--hair_color_preserve_strength", type=float, default=0.7,
                        help="Strength (0–1) of per-channel F statistics restoration "
                             "in the hair region after SATD. 0 = off, 1 = full. "
                             "Default 0.7 corrects A+C colour drift while keeping "
                             "most of SATD's structural benefit.")
    parser.add_argument("--satd_hair_exclude_strength", type=float, default=1.0,
                        help="Strength (0–1) of excluding the target hair region from "
                             "SATD's cleanup support. 1.0 = SATD never modifies hair F "
                             "(hair colour rendered at baseline quality), 0 = SATD may "
                             "modify hair F. Default 1.0 keeps SATD skin-only.")
    parser.add_argument("--blend_chroma_correct_strength", type=float, default=0.0,
                        help="Strength (0–1) of LAB A/B (hue/chroma) mean-shift correction "
                             "applied to I_blend in the hair region. This is a band-aid; "
                             "the root fix is satd_hair_exclude_strength (keep SATD off hair). "
                             "Default 0 = off. Turn on only if residual hue drift remains "
                             "after the SATD-hair-exclude fix.")
    parser.add_argument("--pp_v5_use_mod", type=str2bool, default=True)
    parser.add_argument("--pp_v5_use_full", type=str2bool, default=True)
    parser.add_argument("--ear_parse_size", type=int, default=512)
    parser.add_argument("--ear_feature_channels", type=int, default=128)
    parser.add_argument("--ear_low_alpha", type=float, default=0.1)
    parser.add_argument("--ear_dilate", type=int, default=21)
    parser.add_argument("--hair_change_dilate", type=int, default=25)
    parser.add_argument("--earring_expand", type=int, default=15)
    parser.add_argument("--ear_downward_shift", type=int, default=10)
    parser.add_argument("--target_hair_dilate", type=int, default=11)
    parser.add_argument("--earring_occlusion_dilate", type=int, default=3)
    parser.add_argument("--source_hair_block_dilate", type=int, default=8)
    parser.add_argument("--source_hair_block_strength", type=float, default=0.95)
    parser.add_argument("--target_visibility_expand", type=int, default=5)
    parser.add_argument("--max_target_hair_overlap", type=float, default=0.30)
    parser.add_argument("--min_target_visible_overlap", type=float, default=0.10)
    parser.add_argument("--min_target_ear_area", type=float, default=8.0)
    parser.add_argument("--earring_channel_down", type=int, default=32)
    parser.add_argument("--ear_blur_kernel", type=int, default=11)
    parser.add_argument("--ear_blur_sigma", type=float, default=3.0)
    parser.add_argument("--ear_mask_hidden", type=int, default=32)
    parser.add_argument("--earring_query_dilate", type=int, default=3)
    parser.add_argument("--earring_query_boost", type=float, default=1.0)
    parser.add_argument("--enable_earring_query_recall", type=str2bool, default=True)
    parser.add_argument("--earring_query_recall_dilate", type=int, default=7)
    parser.add_argument("--earring_query_downward_shift", type=int, default=10)
    parser.add_argument("--earring_query_lower_lobe_weight", type=float, default=0.20)
    parser.add_argument("--earring_query_candidate_boost", type=float, default=0.90)
    parser.add_argument("--earring_query_block_protect", type=float, default=0.95)
    parser.add_argument("--disable_earring_path_if_low_confidence", type=str2bool, default=True)
    parser.add_argument("--earring_source_presence_min_area", type=float, default=4.0)
    parser.add_argument("--earring_search_downward_shift", type=int, default=10)
    parser.add_argument("--earring_search_dilate", type=int, default=7)
    parser.add_argument("--earring_write_max_target_hair_overlap", type=float, default=0.30)
    parser.add_argument("--earring_write_source_block_dilate", type=int, default=3)
    parser.add_argument("--earring_write_dilate", type=int, default=3)
    parser.add_argument("--earring_write_connectivity_iters", type=int, default=32)
    parser.add_argument("--earring_write_connectivity_kernel", type=int, default=5)
    parser.add_argument("--earring_write_bridge_dilate", type=int, default=17)
    parser.add_argument("--earring_anchor_visible_dilate", type=int, default=3)
    # Face inputs are already aligned into one canonical frame before V5 runs.
    # Chasing noisy parser centroids moves a real source earring a second time
    # and can produce an offset duplicate, so V5 is source-coordinate by
    # default.  A caller can still opt in explicitly for a controlled study.
    parser.add_argument("--earring_align_max_shift", type=int, default=0)
    parser.add_argument("--ear_fine_support_dilate", type=int, default=3)
    parser.add_argument("--earring_fine_mask_floor", type=float, default=0.18)
    parser.add_argument("--earring_fine_mask_dilate", type=int, default=5)
    # Native source-instance compositing is the authority.  The learned branch
    # is opt-in and, when enabled, is restricted to a tiny completion band
    # around a measured native instance so it cannot invent a second earring.
    parser.add_argument("--earring_learned_fallback_alpha", type=float, default=0.0)
    parser.add_argument("--earring_target_hair_override_dilate", type=int, default=1)
    parser.add_argument("--enable_source_content_gate", type=str2bool, default=True)
    parser.add_argument("--source_content_gate_dilate", type=int, default=3)
    parser.add_argument("--source_hair_face_suppress_dilate", type=int, default=5)
    parser.add_argument("--source_hair_face_suppress_strength", type=float, default=0.65)
    parser.add_argument("--source_hair_face_suppress_max_y", type=float, default=0.45)
    parser.add_argument("--source_hair_face_suppress_ear_exclude_dilate", type=int, default=9)
    parser.add_argument("--enable_output_target_preserve", type=str2bool, default=True)
    parser.add_argument("--output_target_hair_preserve_dilate", type=int, default=5)
    parser.add_argument("--output_face_hair_seam_preserve_dilate", type=int, default=7)
    parser.add_argument("--output_earring_keep_dilate", type=int, default=0)
    parser.add_argument("--output_preserve_blur", type=int, default=1)
    parser.add_argument("--output_hairline_feather", type=int, default=0)
    parser.add_argument("--hair_color_reference_strength_v8", type=float, default=0.9)
    parser.add_argument("--hair_color_low_frequency_radius_v8", type=int, default=15)
    parser.add_argument("--hair_color_low_frequency_sigma_v8", type=float, default=None)
    parser.add_argument("--hair_color_feather_radius_v8", type=int, default=5)
    parser.add_argument("--hair_color_spatial_reference_weight_v8", type=float, default=0.75)
    parser.add_argument("--hair_color_detail_chroma_gain_v8", type=float, default=1.0)
    parser.add_argument("--hair_color_luma_reference_strength_v8", type=float, default=0.30)
    parser.add_argument("--hair_color_luma_mean_limit_v8", type=float, default=6.0)
    parser.add_argument("--hair_color_luma_std_ratio_limit_v8", type=float, default=1.25)
    parser.add_argument("--disable_reference_dominant_hair_color_v8", type=str2bool, default=False)
    parser.add_argument("--debug_save_intermediate_color", type=str2bool, default=True)
    parser.add_argument("--enable_revealed_skin_harmonize", type=str2bool, default=True)
    parser.add_argument("--revealed_skin_harmonize_strength", type=float, default=0.9)
    parser.add_argument("--revealed_skin_tone_kernel", type=int, default=15)
    parser.add_argument("--revealed_skin_tone_sigma", type=float, default=7.0)
    parser.add_argument("--revealed_skin_diffuse_iters", type=int, default=24)
    parser.add_argument("--revealed_skin_tone_limit", type=float, default=0.28)
    parser.add_argument("--revealed_skin_detail_gain", type=float, default=1.0)
    parser.add_argument("--revealed_skin_seam_band", type=int, default=7)
    parser.add_argument("--revealed_skin_min_reference_area", type=float, default=96.0)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    hair_fast = HairFastV5(args)
