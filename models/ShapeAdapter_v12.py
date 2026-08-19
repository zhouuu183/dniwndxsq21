from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


V12_MASK_KEYS = (
    "M_target_hair",
    "M_source_hair",
    "M_donor_hair",
    "M_add",
    "M_remove",
    "M_keep",
    "M_boundary",
    "M_bang",
    "M_face_protect",
    "M_alpha",
    "M_edit",
)


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def ensure_bchw(mask: torch.Tensor, like: torch.Tensor | None = None) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1) if mask.shape[0] != 1 else mask.unsqueeze(0)
    if like is not None:
        mask = mask.to(device=like.device, dtype=like.dtype)
    return mask


def resize_mask(mask: torch.Tensor, size: tuple[int, int], like: torch.Tensor) -> torch.Tensor:
    mask = ensure_bchw(mask, like)
    return F.interpolate(mask.float(), size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    denom = mask.sum().clamp(min=1.0)
    return ((pred - target).abs() * mask).sum() / denom


def build_mask_stack_v12(
    masks_256: dict[str, torch.Tensor],
    size: tuple[int, int],
    like: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    masks_32 = {}
    for key in V12_MASK_KEYS:
        if key in masks_256:
            masks_32[key] = resize_mask(masks_256[key], size, like)
        else:
            masks_32[key] = torch.zeros((like.shape[0], 1, size[0], size[1]), device=like.device, dtype=like.dtype)
    return torch.cat([masks_32[key] for key in V12_MASK_KEYS], dim=1), masks_32


class TopologyF32Adapter_v12(nn.Module):
    """
    Small residual adapter for correcting hairstyle topology in StyleGAN F32 space.

    It predicts only a gated delta on top of the baseline aligned F32 feature.
    The gate is built from topology masks, so the module cannot freely rewrite
    the face/background.
    """

    def __init__(
        self,
        feature_channels: int = 512,
        mask_channels: int = len(V12_MASK_KEYS),
        hidden_channels: int = 512,
    ):
        super().__init__()
        in_channels = feature_channels * 3 + mask_channels
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            _zero_module(nn.Conv2d(hidden_channels, feature_channels, kernel_size=1, stride=1, padding=0)),
        )

    def forward(
        self,
        F_base: torch.Tensor,
        F_src: torch.Tensor,
        F_shape: torch.Tensor,
        masks_256: dict[str, torch.Tensor],
        strength: float = 1.0,
        shape_prior_strength: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        size = F_base.shape[-2:]
        mask_stack, masks_32 = build_mask_stack_v12(masks_256, size, F_base)
        edit_gate = (
            masks_32["M_add"]
            + 0.75 * masks_32["M_remove"]
            + masks_32["M_boundary"]
            + masks_32["M_bang"]
            + 0.35 * masks_32["M_alpha"]
        ).clamp(0.0, 1.0)
        protect_gate = (1.0 - masks_32["M_face_protect"]).clamp(0.0, 1.0)
        gate = (edit_gate * protect_gate).clamp(0.0, 1.0)

        shape_pull_gate = (
            masks_32["M_add"]
            + masks_32["M_boundary"]
            + masks_32["M_bang"]
            + 0.25 * masks_32["M_keep"]
            + 0.15 * masks_32["M_target_hair"]
        ).clamp(0.0, 1.0)
        shape_pull_gate = (shape_pull_gate * protect_gate).clamp(0.0, 1.0)

        learned_delta = self.adapter(torch.cat([F_base, F_src, F_shape, mask_stack], dim=1)) * gate
        prior_delta = (F_shape - F_base) * shape_pull_gate
        delta = learned_delta + float(shape_prior_strength) * prior_delta
        F_refined = F_base + float(strength) * delta
        return {
            "latent_F_refined": F_refined,
            "delta_F32": delta,
            "learned_delta_F32": learned_delta,
            "prior_delta_F32": prior_delta,
            "gate32": gate,
            "shape_pull_gate32": shape_pull_gate,
            "masks32": masks_32,
            "mask_stack32": mask_stack,
        }


def load_shape_adapter_v12(model: TopologyF32Adapter_v12, checkpoint: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    state_dict = checkpoint.get("shape_adapter_v12_state_dict", checkpoint.get("model_state_dict", checkpoint))
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    skipped = sorted(key for key in state_dict if key not in compatible)
    model_state.update(compatible)
    model.load_state_dict(model_state, strict=False)
    return sorted(compatible), skipped
