import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Encoders import ClipModel, FeatureiResnet, ModulationModule
from models.Net import FeatureEncoderMult
from models.delta_blending_blocks_v1 import (
    BoundaryFusionHeadV1,
    ContextRecoveryBranchV1,
    HairAdditionBranchV1,
)
from models.stylegan2.model import PixelNorm


# v1 modification:
# This file keeps the original encoder utilities intact through re-exports,
# and adds the new delta-aware blending model plus a lightly adapted
# post-process model that accepts remove/boundary masks.


class DeltaAwareBlendingModelV1(nn.Module):
    def __init__(self, n_styles: int = 12):
        super().__init__()
        self.delta_f_scale = 0.1
        self.hair_branch = HairAdditionBranchV1(ModulationModule, n_styles=n_styles)
        self.context_branch = ContextRecoveryBranchV1()
        self.boundary_head = BoundaryFusionHeadV1()

    def forward(
        self,
        latent_face_tail: torch.Tensor,
        latent_color_tail: torch.Tensor,
        source_image: torch.Tensor,
        color_image: torch.Tensor,
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        keep_mask = masks["M_keep"].float()
        add_mask = masks["M_add"].float()
        color_mask = masks["M_color"].float()
        boundary_mask = masks["M_boundary"].float()

        latent_tail = self.hair_branch(
            latent_face_tail,
            latent_color_tail,
            source_image,
            color_image,
            keep_mask=keep_mask,
            add_mask=add_mask,
            color_mask=color_mask,
            boundary_mask=boundary_mask,
        )

        delta_f = self.context_branch(
            source_image,
            src_mask=masks["M_src"].float(),
            tgt_mask=masks["M_tgt"].float(),
            remove_mask=masks["M_remove"].float(),
            boundary_mask=boundary_mask,
        )
        # v1 bugfix:
        # ContextRecoveryBranchV1 predicts a low-resolution feature map.
        # Upsample it explicitly to the 32x32 F-space resolution used by StyleGAN
        # blending so training/inference do not hit a shape mismatch.
        delta_f = F.interpolate(delta_f, size=(32, 32), mode="bilinear", align_corners=False)
        # v1 stabilization:
        # Keep the F-space residual small, similar in spirit to the 0.1 residual
        # scaling used by the style branch. This reduces early gray "fog/helmet"
        # artifacts and improves optimization stability.
        delta_f = self.delta_f_scale * delta_f

        boundary_gate = self.boundary_head(
            source_image,
            add_mask=add_mask,
            remove_mask=masks["M_remove"].float(),
            boundary_mask=boundary_mask,
            delta_f=delta_f,
        )

        return {
            "latent_tail": latent_tail,
            "delta_F": delta_f,
            "boundary_gate": boundary_gate,
        }


class DeltaAwarePostProcessModelV1(nn.Module):
    def __init__(self, pretrain: bool = False):
        super().__init__()
        self.pretrain = pretrain
        self.encoder_face = FeatureEncoderMult(
            fs_layers=[9],
            opts=argparse.Namespace(**{"arcface_model_path": "pretrained_models/ArcFace/backbone_ir50.pth"}),
        )
        # v1 bugfix:
        # Keep latent_avg as a registered buffer so `.to(device)` moves it together
        # with the module instead of pinning it to CUDA at load time.
        latent_avg = torch.load("pretrained_models/PostProcess/latent_avg.pt", map_location="cpu")
        self.register_buffer("latent_avg", latent_avg)
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
        self.pixelnorm = PixelNorm()

        # v1 modification:
        # A small mask-conditioned adapter lets pp focus on remove/boundary
        # regions without redesigning the whole refinement stage.
        self.mask_adapter = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        )
        self.mask_to_style = nn.Sequential(
            nn.Linear(2 * 16, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        remove_mask: torch.Tensor | None = None,
        boundary_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_face, [f_face] = self.encoder_face(source)
        if self.pretrain:
            # v1 bugfix:
            # Restore the original pretrain behaviour: learn source reconstruction
            # before enabling the full source/target fusion path.
            return self.latent_avg + s_face, f_face
        s_hair, [f_hair] = self.encoder_face(target)

        if remove_mask is None:
            remove_mask = torch.zeros(source.size(0), 1, source.size(2), source.size(3), device=source.device)
        if boundary_mask is None:
            boundary_mask = torch.zeros_like(remove_mask)

        dt_latent_face = self.pixelnorm(s_face)
        dt_latent_hair = self.pixelnorm(s_hair)

        pooled_masks = F.adaptive_avg_pool2d(torch.cat([remove_mask, boundary_mask], dim=1), (4, 4)).flatten(1)
        mask_style = self.mask_to_style(pooled_masks).unsqueeze(1).expand(-1, 18, -1)

        for mod_module in self.to_latent_1:
            dt_latent_face = mod_module(dt_latent_face, s_hair + mask_style)

        for mod_module in self.to_latent_2:
            dt_latent_hair = mod_module(dt_latent_hair, s_face + mask_style)

        final_s = self.latent_avg + 0.1 * (dt_latent_face + dt_latent_hair)

        mask_feat = F.interpolate(
            torch.cat([remove_mask, boundary_mask], dim=1),
            size=f_hair.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        mask_feat = self.mask_adapter(mask_feat)
        final_f = self.to_feature(torch.cat((f_face, f_hair + 0.1 * mask_feat), dim=1))
        return final_s, final_f


__all__ = [
    "ClipModel",
    "ModulationModule",
    "FeatureiResnet",
    "FeatureEncoderMult",
    "DeltaAwareBlendingModelV1",
    "DeltaAwarePostProcessModelV1",
]
