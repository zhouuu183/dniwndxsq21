from __future__ import annotations

import os
from pathlib import Path

import torch
import torchvision.transforms as T
from torch import nn

from models.Net import Net
from models.PostProcess_v14 import PostProcessModelV14
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents

try:
    from utils.save_utils_v14 import save_tensor_image
except ImportError:
    _to_pil_v14 = T.ToPILImage()

    def _ensure_4d_v14(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        return tensor

    def _prepare_dir_v14(output_dir: Path | str, folder: str) -> Path:
        save_dir = Path(output_dir) / folder
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_tensor_image(output_dir: Path | str, folder: str, name: str, tensor: torch.Tensor, value_range: str = "tanh") -> None:
        tensor = _ensure_4d_v14(tensor)[0].detach().cpu().float()
        if tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
        if value_range == "tanh":
            tensor = ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)
        else:
            tensor = tensor.clamp(0.0, 1.0)
        _to_pil_v14(tensor).save(_prepare_dir_v14(output_dir, folder) / name)


class BlendingV14(nn.Module):
    """
    Original blending responsibility, but refinement must consume clean source.
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        self.net = Net(self.opts) if net is None else net

        from models.Encoders import ClipBlendingModel

        blending_checkpoint = torch.load(self.opts.blending_checkpoint)
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get("clip", "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint["model_state_dict"], strict=False)
        self.blending_encoder.to(self.opts.device).eval()

        self.post_process = PostProcessModelV14().to(self.opts.device).eval()
        pp_checkpoint_path = getattr(self.opts, "pp_v14_checkpoint", "") or self.opts.pp_checkpoint
        self.post_process.load_state_dict(torch.load(pp_checkpoint_path)["model_state_dict"], strict=False)

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)

    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        mask_de = self.dilate_erosion.hair_from_mask(torch.cat([name_to_embed[x]["mask"] for x in ["face", "color"]], dim=0))
        HM_1D = mask_de[0][0].unsqueeze(0)
        HM_3D, HM_3E = mask_de[0][1].unsqueeze(0), mask_de[1][1].unsqueeze(0)

        latent_S_1 = name_to_embed["face"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        latent_S_3 = name_to_embed["color"]["S"]
        HM_X = align_color["HM_X"]

        HM_XD, _ = self.dilate_erosion.mask(HM_X)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        same_face_color = torch.allclose(I_1, I_3)
        if not same_face_color:
            S_blend_6_18 = self.blending_encoder(latent_S_1[:, 6:], latent_S_3[:, 6:], I_1 * target_mask, I_3 * HM_3E)
            S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1

        I_blend, _ = self.net.generator(
            [S_blend],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_F_align,
        )
        I_blend_256 = self.downsample_256(I_blend)

        use_raw_source = kwargs.get(
            "nhr_use_raw_source_for_refine",
            getattr(self.opts, "nhr_use_raw_source_for_refine", False),
        )
        I_source_refine = I_1 if use_raw_source else align_shape.get("I_source_clean", I_1)

        S_final, F_final = self.post_process(I_source_refine, I_blend_256)
        I_final, _ = self.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final,
        )

        if self.opts.save_all:
            exp_name = exp_name if (exp_name := kwargs.get("exp_name")) is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v14", "blending.png", I_blend)
            save_latents(output_dir, "Blending_v14", "blending.npz", S_blend=S_blend)
            save_tensor_image(output_dir, "Blending_v14", "source_for_refine.png", I_source_refine)

            save_gen_image(output_dir, "Final_v14", "final.png", I_final)
            save_latents(output_dir, "Final_v14", "final.npz", S_final=S_final, F_final=F_final)

        final_image = ((I_final[0] + 1) / 2).clip(0, 1)
        if kwargs.get("return_pipeline_info", False):
            return {
                "final_image": final_image,
                "I_blend_1024": I_blend,
                "I_blend_256": I_blend_256,
                "I_source_refine": I_source_refine,
                "S_blend": S_blend,
                "S_final": S_final,
                "F_final": F_final,
            }
        return final_image
