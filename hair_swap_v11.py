from __future__ import annotations

import argparse
import typing as tp
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.io import ImageReadMode, read_image
from torchvision.utils import save_image

from models.Alignment import Alignment
from models.Blending import Blending
from models.DeocclusionRepair_v11 import DeocclusionRepair_v11, load_deocclusion_repair_v11
from models.Embedding import Embedding
from models.Net import Net
from utils.deocclusion_masks_v11 import build_deocclusion_masks_v11
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session

try:
    from models.ShadowCleanup import ShadowCleanup
    from utils.shadow_cleanup_masks import build_shadow_cleanup_masks
except ModuleNotFoundError:
    ShadowCleanup = None
    build_shadow_cleanup_masks = None

TImage = tp.TypeVar("TImage", torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar("TPath", Path, str)
TReturn = tp.TypeVar("TReturn", torch.Tensor, tuple[torch.Tensor, ...], dict[str, tp.Any])


class HairFast_v11:
    """
    Baseline HairFast plus an optional v11 image-space deocclusion repair stage.
    Existing baseline files are imported but not modified.
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, self.embed.get_e4e_embed, net=self.net)
        self.blend = Blending(args, net=self.net)
        self.shadow_cleanup = ShadowCleanup(args) if ShadowCleanup is not None else None

        self.deocclusion_repair = None
        if getattr(args, "use_deocclusion_v11", False) or getattr(args, "deocclusion_checkpoint_v11", ""):
            self.deocclusion_repair = DeocclusionRepair_v11(
                base_channels=getattr(args, "deocclusion_base_channels_v11", 48),
                delta_scale=getattr(args, "deocclusion_delta_scale_v11", 1.0),
            ).to(args.device).eval()
            ckpt_path = getattr(args, "deocclusion_checkpoint_v11", "")
            if ckpt_path:
                load_deocclusion_repair_v11(self.deocclusion_repair, ckpt_path, map_location=args.device)

    @staticmethod
    def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    @staticmethod
    def _parse_current_output_256(image: torch.Tensor) -> torch.Tensor | None:
        try:
            from models.CtrlHair.external_code.face_parsing.my_parsing_util import FaceParsing_tensor
        except ModuleNotFoundError:
            return None

        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image[:1].float().clamp(0, 1)
        image_512 = F_torch.interpolate(image, size=(512, 512), mode="bilinear", align_corners=False)
        mean = image_512.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = image_512.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        parsing, _ = FaceParsing_tensor.parsing_img((image_512 - mean) / std)
        parsing = FaceParsing_tensor.swap_parsing_label_to_celeba_mask(parsing)
        parsing = parsing.long()[None, None, ...].to(image.device)
        return F_torch.interpolate(parsing.float(), size=(256, 256), mode="nearest").long()

    def _apply_shadow_cleanup(
        self,
        face: torch.Tensor,
        final_image: torch.Tensor,
        name_to_embed,
        align_shape,
        **kwargs,
    ) -> torch.Tensor:
        use_cleanup = kwargs.get("use_shadow_cleanup", getattr(self.args, "use_shadow_cleanup", False))
        if not use_cleanup or self.shadow_cleanup is None or build_shadow_cleanup_masks is None:
            return final_image

        cleanup_masks = build_shadow_cleanup_masks(
            source_parsing=name_to_embed["face"]["mask"],
            target_hair_mask=align_shape["HM_X"],
            ring_width=kwargs.get("shadow_cleanup_ring", getattr(self.args, "shadow_cleanup_ring", 7)),
            halo_width=kwargs.get("shadow_cleanup_halo", getattr(self.args, "shadow_cleanup_halo", 9)),
            protect_width=kwargs.get("shadow_cleanup_protect", getattr(self.args, "shadow_cleanup_protect", 2)),
        )

        cleaned = self.shadow_cleanup(
            source_image=self._ensure_batch(face.to(final_image.device)),
            base_image=self._ensure_batch(final_image.to(self.args.device)),
            cleanup_masks={key: value.to(final_image.device) for key, value in cleanup_masks.items()},
            strength=kwargs.get("shadow_cleanup_strength", getattr(self.args, "shadow_cleanup_strength", 0.75)),
            source_blend=kwargs.get("shadow_cleanup_source_blend", getattr(self.args, "shadow_cleanup_source_blend", 0.55)),
            kernel_size=kwargs.get("shadow_cleanup_kernel", getattr(self.args, "shadow_cleanup_kernel", 21)),
        )
        return cleaned[0]

    @torch.inference_mode()
    def _apply_deocclusion_repair(
        self,
        face: torch.Tensor,
        base_image: torch.Tensor,
        name_to_embed,
        align_shape,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, tp.Any]]:
        use_repair = kwargs.get("use_deocclusion_v11", getattr(self.args, "use_deocclusion_v11", False))
        if not use_repair or self.deocclusion_repair is None:
            return base_image, {}

        repair_size = int(kwargs.get("deocclusion_input_size_v11", getattr(self.args, "deocclusion_input_size_v11", 256)))
        base_batch = self._ensure_batch(base_image.to(self.args.device))
        target_parsing = self._parse_current_output_256(base_batch)
        masks = build_deocclusion_masks_v11(
            source_parsing=name_to_embed["face"]["mask"],
            target_hair_mask=align_shape["HM_X"],
            target_parsing=target_parsing,
            target_protect_width=kwargs.get("deocclusion_target_protect_v11", getattr(self.args, "deocclusion_target_protect_v11", 2)),
            removed_dilate_width=kwargs.get("deocclusion_removed_dilate_v11", getattr(self.args, "deocclusion_removed_dilate_v11", 3)),
            halo_width=kwargs.get("deocclusion_halo_width_v11", getattr(self.args, "deocclusion_halo_width_v11", 8)),
            skin_expand_width=kwargs.get("deocclusion_skin_expand_v11", getattr(self.args, "deocclusion_skin_expand_v11", 8)),
            struct_expand_width=kwargs.get("deocclusion_struct_expand_v11", getattr(self.args, "deocclusion_struct_expand_v11", 22)),
            clean_boundary_width=kwargs.get("deocclusion_clean_boundary_v11", getattr(self.args, "deocclusion_clean_boundary_v11", 4)),
            fill_distance=kwargs.get("deocclusion_fill_distance_v11", getattr(self.args, "deocclusion_fill_distance_v11", 9)),
            context_width=kwargs.get("deocclusion_context_width_v11", getattr(self.args, "deocclusion_context_width_v11", 9)),
            safe_width=kwargs.get("deocclusion_safe_width_v11", getattr(self.args, "deocclusion_safe_width_v11", 4)),
            face_protect_width=kwargs.get("deocclusion_face_protect_v11", getattr(self.args, "deocclusion_face_protect_v11", 5)),
            tail_y_min=kwargs.get("deocclusion_tail_y_min_v11", getattr(self.args, "deocclusion_tail_y_min_v11", 0.56)),
            reveal_blur_kernel=kwargs.get("deocclusion_blur_kernel_v11", getattr(self.args, "deocclusion_blur_kernel_v11", 7)),
        )
        masks = {key: value.to(self.args.device) for key, value in masks.items()}
        min_reveal = float(kwargs.get("deocclusion_min_reveal_area_v11", getattr(self.args, "deocclusion_min_reveal_area_v11", 0.002)))
        reveal_area = float(masks["M_reveal"].mean().detach().cpu())
        if reveal_area < min_reveal:
            return base_image, {"deocclusion_masks": masks, "skip_reason": "small_reveal"}

        source_256 = name_to_embed["face"]["image_256"].to(self.args.device)
        base_repair = F_torch.interpolate(base_batch, size=(repair_size, repair_size), mode="bilinear", align_corners=False)
        source_repair = F_torch.interpolate(source_256, size=(repair_size, repair_size), mode="bilinear", align_corners=False)

        repaired_256, repair_aux = self.deocclusion_repair(
            source_image=source_repair,
            base_image=base_repair,
            masks=masks,
            source_inpaint=None,
            alpha_scale=kwargs.get("deocclusion_alpha_scale_v11", getattr(self.args, "deocclusion_alpha_scale_v11", 1.0)),
            enable_skin=not kwargs.get("disable_deocclusion_skin_v11", getattr(self.args, "disable_deocclusion_skin_v11", False)),
            enable_struct=not kwargs.get("disable_deocclusion_struct_v11", getattr(self.args, "disable_deocclusion_struct_v11", False)),
            enable_bg=not kwargs.get("disable_deocclusion_bg_v11", getattr(self.args, "disable_deocclusion_bg_v11", False)),
            enable_fill=not kwargs.get("disable_deocclusion_fill_v11", getattr(self.args, "disable_deocclusion_fill_v11", False)),
            enable_clean=not kwargs.get("disable_deocclusion_clean_v11", getattr(self.args, "disable_deocclusion_clean_v11", False)),
        )
        if repaired_256.dim() == 3:
            repaired_256 = repaired_256.unsqueeze(0)
        alpha_256 = repair_aux["alpha"]
        if alpha_256.dim() == 3:
            alpha_256 = alpha_256.unsqueeze(0)

        delta_high = F_torch.interpolate(repaired_256 - base_repair, size=base_batch.shape[-2:], mode="bilinear", align_corners=False)
        alpha_high = F_torch.interpolate(alpha_256, size=base_batch.shape[-2:], mode="bilinear", align_corners=False)
        blend_strength = float(kwargs.get("deocclusion_blend_v11", getattr(self.args, "deocclusion_blend_v11", 1.0)))
        final = (base_batch + delta_high * float(blend_strength)).clamp(0, 1)[0]

        if self.args.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.args.save_all_dir / exp_name / "Deocclusion_v11"
            output_dir.mkdir(parents=True, exist_ok=True)
            save_image(base_repair[0], output_dir / "base_256.png")
            save_image(repaired_256[0], output_dir / "repaired_256.png")
            save_image(alpha_256[0], output_dir / "alpha_256.png")
            for alpha_name in ("alpha_skin", "alpha_struct", "alpha_bg", "alpha_fill", "alpha_clean"):
                if alpha_name in repair_aux:
                    save_image(repair_aux[alpha_name][0].float().clamp(0, 1), output_dir / f"{alpha_name}.png")
            for key, value in masks.items():
                save_image(value[0].float().clamp(0, 1), output_dir / f"{key}.png")

        return final, {
            "deocclusion_masks": masks,
            "deocclusion_aux": repair_aux,
            "base_image": base_image,
            "repaired_256": repaired_256,
            "alpha_high": alpha_high,
        }

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor, **kwargs) -> TReturn:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ("face", "shape", "color")):
            images_to_name[image].append(name)

        return_info = kwargs.pop("return_deocclusion_info_v11", False)
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)
        align_shape = self.align.align_images("face", "shape", name_to_embed, **kwargs)
        if shape is not color:
            align_color = self.align.shape_module("face", "color", name_to_embed, **kwargs)
        else:
            align_color = align_shape

        base_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        base_image = self._apply_shadow_cleanup(face, base_image, name_to_embed, align_shape, **kwargs)
        final_image, repair_info = self._apply_deocclusion_repair(face, base_image, name_to_embed, align_shape, **kwargs)
        if not return_info:
            return final_image
        return {
            "final_image": final_image,
            "base_image": base_image,
            "align_shape": align_shape,
            "align_color": align_color,
            **repair_info,
        }

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
        images = equal_replacer(images)

        output = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)
        if align and not isinstance(output, dict):
            return output, *images
        return output

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser_v11():
    parser = argparse.ArgumentParser(description="HairFast v11")
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

    parser.add_argument("--use_shadow_cleanup", action="store_true")
    parser.add_argument("--shadow_cleanup_strength", type=float, default=0.75)
    parser.add_argument("--shadow_cleanup_source_blend", type=float, default=0.55)
    parser.add_argument("--shadow_cleanup_kernel", type=int, default=21)
    parser.add_argument("--shadow_cleanup_ring", type=int, default=7)
    parser.add_argument("--shadow_cleanup_halo", type=int, default=9)
    parser.add_argument("--shadow_cleanup_protect", type=int, default=2)

    parser.add_argument("--use_deocclusion_v11", action="store_true")
    parser.add_argument("--deocclusion_checkpoint_v11", type=str, default="")
    parser.add_argument("--deocclusion_input_size_v11", type=int, default=256)
    parser.add_argument("--deocclusion_blend_v11", type=float, default=1.0)
    parser.add_argument("--deocclusion_alpha_scale_v11", type=float, default=1.0)
    parser.add_argument("--deocclusion_delta_scale_v11", type=float, default=1.0)
    parser.add_argument("--deocclusion_base_channels_v11", type=int, default=48)
    parser.add_argument("--deocclusion_target_protect_v11", type=int, default=2)
    parser.add_argument("--deocclusion_removed_dilate_v11", type=int, default=3)
    parser.add_argument("--deocclusion_halo_width_v11", type=int, default=8)
    parser.add_argument("--deocclusion_skin_expand_v11", type=int, default=8)
    parser.add_argument("--deocclusion_struct_expand_v11", type=int, default=22)
    parser.add_argument("--deocclusion_clean_boundary_v11", type=int, default=4)
    parser.add_argument("--deocclusion_fill_distance_v11", type=int, default=9)
    parser.add_argument("--deocclusion_context_width_v11", type=int, default=9)
    parser.add_argument("--deocclusion_safe_width_v11", type=int, default=4)
    parser.add_argument("--deocclusion_face_protect_v11", type=int, default=5)
    parser.add_argument("--deocclusion_min_reveal_area_v11", type=float, default=0.002)
    parser.add_argument("--deocclusion_tail_y_min_v11", type=float, default=0.56)
    parser.add_argument("--deocclusion_blur_kernel_v11", type=int, default=7)
    parser.add_argument("--disable_deocclusion_skin_v11", action="store_true")
    parser.add_argument("--disable_deocclusion_struct_v11", action="store_true")
    parser.add_argument("--disable_deocclusion_bg_v11", action="store_true")
    parser.add_argument("--disable_deocclusion_fill_v11", action="store_true")
    parser.add_argument("--disable_deocclusion_clean_v11", action="store_true")
    return parser


if __name__ == "__main__":
    args = get_parser_v11().parse_args()
    HairFast_v11(args)
