from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from models.Encoders import FeatureEncoderMult, FeatureiResnet, ModulationModule
from models.stylegan2.model import PixelNorm
from utils.mask_formula_v6 import build_soft_difference_mask, ensure_mask_4d, resize_mask


class PostProcessModelV6(nn.Module):
    """
    v6 trims source 64x64 features with M_diff_soft before feature fusion.
    """

    def __init__(self, args: argparse.Namespace | None = None):
        super().__init__()
        self.args = args or argparse.Namespace()

        self.encoder_face = FeatureEncoderMult(
            fs_layers=[9],
            opts=argparse.Namespace(arcface_model_path="pretrained_models/ArcFace/backbone_ir50.pth"),
        )
        if not getattr(self.args, "finetune", False):
            for param in self.encoder_face.parameters():
                param.requires_grad = False

        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.latent_avg = torch.load("pretrained_models/PostProcess/latent_avg.pt", map_location=map_location)
        self.to_feature = FeatureiResnet([[1024, 2], [768, 2], [512, 2]])

        self.use_mod = getattr(self.args, "use_mod", True)
        self.pretrain = getattr(self.args, "pretrain", False)
        self.diff_mask_dilate = int(getattr(self.args, "diff_mask_dilate", 5))
        self.diff_blur_kernel = int(getattr(self.args, "diff_mask_blur_kernel", 11))
        self.diff_blur_sigma = float(getattr(self.args, "diff_mask_blur_sigma", 0.0))

        if self.use_mod:
            self.to_latent_1 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.to_latent_2 = nn.ModuleList([ModulationModule(18, i == 4) for i in range(5)])
            self.pixelnorm = PixelNorm()
        else:
            self.to_latent = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm([1024]),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
            )

    def load_base_checkpoint(self, checkpoint_path: str | None):
        if not checkpoint_path:
            return None
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        return self.load_state_dict(state_dict, strict=False)

    def _compute_latent(self, s_face: torch.Tensor, s_hair: torch.Tensor) -> torch.Tensor:
        if self.use_mod:
            dt_latent_face = self.pixelnorm(s_face)
            dt_latent_hair = self.pixelnorm(s_hair)

            for mod_module in self.to_latent_1:
                dt_latent_face = mod_module(dt_latent_face, s_hair)
            for mod_module in self.to_latent_2:
                dt_latent_hair = mod_module(dt_latent_hair, s_face)

            return self.latent_avg.to(s_face.device) + 0.1 * (dt_latent_face + dt_latent_hair)

        cat_s = torch.cat((s_face, s_hair), dim=-1)
        return self.latent_avg.to(s_face.device) + self.to_latent(cat_s)

    def _build_feature(
        self,
        f_source: torch.Tensor,
        f_blend: torch.Tensor,
        diff_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        diff_soft = None
        if diff_mask is not None:
            diff_soft = build_soft_difference_mask(
                diff_mask,
                dilate_radius=self.diff_mask_dilate,
                blur_kernel_size=self.diff_blur_kernel,
                blur_sigma=self.diff_blur_sigma or None,
            )
            diff_soft = resize_mask(diff_soft, size=f_source.shape[-2:], mode="bilinear")
            f_source = f_source * (1.0 - diff_soft)

        cat_f = torch.cat((f_source, f_blend), dim=1)
        return self.to_feature(cat_f), diff_soft

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        HT_E: torch.Tensor | None = None,
        diff_mask: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        source_hair_mask: torch.Tensor | None = None,
        target_hair_mask: torch.Tensor | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        s_face, [f_face] = self.encoder_face(source)

        if diff_mask is None and source_hair_mask is not None and target_hair_mask is not None:
            diff_mask = (
                ensure_mask_4d(source_hair_mask).float()
                * (1.0 - ensure_mask_4d(target_hair_mask).float())
            ).clamp(0, 1)
        if valid_mask is None and source_hair_mask is not None and target_hair_mask is not None:
            valid_mask = (
                (1.0 - ensure_mask_4d(source_hair_mask).float())
                * (1.0 - ensure_mask_4d(target_hair_mask).float())
            ).clamp(0, 1)

        aux = {
            "diff_mask": ensure_mask_4d(diff_mask).float() if diff_mask is not None else None,
            "valid_mask": ensure_mask_4d(valid_mask).float() if valid_mask is not None else None,
            "source_hair_mask": ensure_mask_4d(source_hair_mask).float() if source_hair_mask is not None else None,
            "target_hair_mask": ensure_mask_4d(target_hair_mask).float() if target_hair_mask is not None else None,
            "target_mask": ensure_mask_4d(target_mask).float() if target_mask is not None else None,
            "HT_E": ensure_mask_4d(HT_E).float() if HT_E is not None else None,
        }

        if self.pretrain:
            aux["diff_soft_64"] = None
            return self.latent_avg.to(s_face.device) + s_face, f_face, aux

        s_hair, [f_hair] = self.encoder_face(target)
        finall_s = self._compute_latent(s_face, s_hair)
        finall_f, diff_soft = self._build_feature(f_face, f_hair, aux["diff_mask"])
        aux["diff_soft_64"] = diff_soft
        return finall_s, finall_f, aux
