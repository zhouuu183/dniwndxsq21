from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import ClipBlendingModel, PostProcessModel
from models.Net import Net
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.save_utils import save_gen_image, save_latents, save_vis_mask


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def _resize_mask(mask: torch.Tensor | None, size: tuple[int, int], like: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.zeros((like.shape[0], 1, size[0], size[1]), device=like.device, dtype=like.dtype)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(device=like.device, dtype=like.dtype)
    return F.interpolate(mask, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


class F64HairBypass(nn.Module):
    """
    Lightweight implementation of:
        F64_hat = Conv1x1([F64_current, F64_hair])

    The module starts as an identity residual branch, so loading it into the
    baseline path does not immediately disturb the pretrained model.
    """

    def __init__(self, channels: int = 512, hidden_channels: int = 512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            _zero_module(nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0)),
        )

    def forward(
        self,
        current_f64: torch.Tensor,
        hair_f64: torch.Tensor,
        hair_mask_256: torch.Tensor | None = None,
        boundary_mask_256: torch.Tensor | None = None,
        boundary_alpha_256: torch.Tensor | None = None,
        strength: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        size = current_f64.shape[-2:]
        hair_mask = _resize_mask(hair_mask_256, size, current_f64)
        boundary_mask = _resize_mask(boundary_mask_256, size, current_f64)
        if boundary_alpha_256 is None:
            boundary_alpha = boundary_mask
        else:
            boundary_alpha = _resize_mask(boundary_alpha_256, size, current_f64)

        boundary_weight = (boundary_mask * boundary_alpha).clamp(0.0, 1.0)
        boundary_feature = boundary_alpha * hair_f64 + (1.0 - boundary_alpha) * current_f64
        region_feature = current_f64 * (1.0 - boundary_weight) + boundary_feature * boundary_weight
        inject_mask = torch.maximum(hair_mask, boundary_mask).clamp(0.0, 1.0)
        delta = self.fusion(torch.cat([current_f64, region_feature], dim=1)) * inject_mask
        fused = current_f64 + float(strength) * delta

        return {
            "fused_f64": fused,
            "f64_delta": delta,
            "f64_hair_mask": hair_mask,
            "f64_boundary_mask": boundary_mask,
            "f64_boundary_alpha": boundary_alpha,
            "f64_inject_mask": inject_mask,
        }


class Blending_v10(nn.Module):
    """
    Baseline blending with v10 additions:
    - train/inference style-prefix consistency is preserved by using S[:, :6]
    - boundary alpha and donor hair F64 are consumed when available
    - final post-process feature can be refined by a trainable F64 bypass
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        blending_checkpoint = torch.load(self.opts.blending_checkpoint, map_location=self.opts.device)
        self.blending_encoder = ClipBlendingModel(blending_checkpoint.get("clip", "ViT-B/32"))
        self.blending_encoder.load_state_dict(blending_checkpoint["model_state_dict"], strict=False)
        self.blending_encoder.to(self.opts.device).eval()

        self.post_process = PostProcessModel().to(self.opts.device).eval()
        self.post_process.load_state_dict(torch.load(self.opts.pp_checkpoint, map_location=self.opts.device)["model_state_dict"])

        self.f64_bypass = F64HairBypass(
            channels=int(getattr(opts, "f64_channels_v10", 512)),
            hidden_channels=int(getattr(opts, "f64_hidden_channels_v10", 512)),
        ).to(self.opts.device)
        self._load_v10_checkpoint(getattr(opts, "blending_v10_checkpoint", ""))

        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        use_cuda = "cuda" in str(self.opts.device).lower()
        self.downsample_256 = BicubicDownSample(factor=4, cuda=use_cuda)

    def _load_v10_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location=self.opts.device)
        state_dict = checkpoint.get("f64_bypass_state_dict", checkpoint.get("model_state_dict", checkpoint))
        self.f64_bypass.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def feature64_from_style(self, latent_s: torch.Tensor, latent_f32: torch.Tensor) -> torch.Tensor:
        feature64, _ = self.net.generator(
            [latent_s],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=4,
            layer_in=latent_f32,
        )
        return feature64

    def _resolve_hair_f64(self, align_color: dict, name_to_embed) -> torch.Tensor:
        if "latent_F64_hair" in align_color:
            return align_color["latent_F64_hair"]
        color_embed = name_to_embed["color"]
        if "F64" in color_embed:
            return color_embed["F64"]
        return self.feature64_from_style(color_embed["S"], color_embed["F"])

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        mask_de = self.dilate_erosion.hair_from_mask(
            torch.cat([name_to_embed[x]["mask"] for x in ["face", "color"]], dim=0)
        )
        HM_1D = mask_de[0][0].unsqueeze(0)
        HM_3D = mask_de[0][1].unsqueeze(0)
        HM_3E = mask_de[1][1].unsqueeze(0)

        latent_S_1 = name_to_embed["face"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        latent_S_3 = name_to_embed["color"]["S"]
        if "target_hair_mask" in align_color:
            target_hair_mask = align_color["target_hair_mask"]
        else:
            target_hair_mask = align_color["HM_X"]

        HM_XD, _ = self.dilate_erosion.mask(target_hair_mask)
        target_mask = (1 - HM_1D) * (1 - HM_3D) * (1 - HM_XD)

        if I_1 is not I_3:
            S_blend_6_18 = self.blending_encoder(
                latent_S_1[:, 6:],
                latent_S_3[:, 6:],
                I_1 * target_mask,
                I_3 * HM_3E,
            )
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

        S_final, F_final = self.post_process(I_1, I_blend_256)
        use_f64 = kwargs.get("use_f64_bypass_v10", getattr(self.opts, "use_f64_bypass_v10", False))
        bypass_outputs = {
            "fused_f64": F_final,
            "f64_delta": torch.zeros_like(F_final),
            "f64_hair_mask": torch.zeros_like(F_final[:, :1]),
            "f64_boundary_mask": torch.zeros_like(F_final[:, :1]),
            "f64_boundary_alpha": torch.zeros_like(F_final[:, :1]),
            "f64_inject_mask": torch.zeros_like(F_final[:, :1]),
        }
        if use_f64:
            hair_f64 = self._resolve_hair_f64(align_color, name_to_embed)
            bypass_outputs = self.f64_bypass(
                current_f64=F_final,
                hair_f64=hair_f64,
                hair_mask_256=target_hair_mask,
                boundary_mask_256=align_color.get("boundary_mask_256"),
                boundary_alpha_256=align_color.get("boundary_alpha_256"),
                strength=kwargs.get("f64_bypass_strength_v10", getattr(self.opts, "f64_bypass_strength_v10", 1.0)),
            )
        F_final_v10 = bypass_outputs["fused_f64"]

        I_final, _ = self.net.generator(
            [S_final],
            input_is_latent=True,
            return_latents=False,
            start_layer=5,
            end_layer=8,
            layer_in=F_final_v10,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v10", "blending.png", I_blend)
            save_gen_image(output_dir, "Final_v10", "final.png", I_final)
            save_vis_mask(output_dir, "Blending_v10", "target_hair_mask.png", target_hair_mask)
            if "boundary_mask_256" in align_color:
                save_vis_mask(output_dir, "Blending_v10", "boundary_mask.png", align_color["boundary_mask_256"])
            save_latents(
                output_dir,
                "Blending_v10",
                "blending.npz",
                S_blend=S_blend,
                S_final=S_final,
                F_final=F_final,
                F_final_v10=F_final_v10,
                f64_delta=bypass_outputs["f64_delta"],
            )

        final_image = ((I_final[0] + 1) / 2).clip(0, 1)
        return final_image
