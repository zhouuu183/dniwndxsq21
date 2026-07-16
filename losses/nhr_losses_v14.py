from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nhr_utils_v14 import HAIR_LABEL, get_celeba_parsing_logits


def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.expand_as(value)
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def image_gradients(image: torch.Tensor) -> torch.Tensor:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return torch.cat([dx, dy], dim=1)


def blur5(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)


def total_variation_loss(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dx = (image[..., :, 1:] - image[..., :, :-1]).abs()
    dy = (image[..., 1:, :] - image[..., :-1, :]).abs()
    mask_x = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
    mask_y = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
    return masked_mean(dx, mask_x) + masked_mean(dy, mask_y)


class MultiScaleL1(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        total = masked_mean((pred - target).abs(), mask)
        pred_s, target_s, mask_s = pred, target, mask
        for scale in range(2):
            pred_s = self.pool(pred_s)
            target_s = self.pool(target_s)
            mask_s = self.pool(mask_s).clamp(0.0, 1.0)
            total = total + (0.5 ** (scale + 1)) * masked_mean((pred_s - target_s).abs(), mask_s)
        return total


class NHRLossBuilderV14(nn.Module):
    def __init__(
        self,
        *,
        lambda_fill: float = 1.0,
        lambda_perc: float = 0.3,
        lambda_edge: float = 1.0,
        lambda_nonhair: float = 0.5,
        lambda_keep: float = 1.0,
        lambda_hair: float = 1.0,
        lambda_alpha: float = 0.05,
        lambda_erase_hair: float = 0.15,
        lambda_erase_alpha: float = 0.03,
        lambda_occ_anchor: float = 0.25,
        lambda_occ_tv: float = 0.05,
    ):
        super().__init__()
        self.lambda_fill = lambda_fill
        self.lambda_perc = lambda_perc
        self.lambda_edge = lambda_edge
        self.lambda_nonhair = lambda_nonhair
        self.lambda_keep = lambda_keep
        self.lambda_hair = lambda_hair
        self.lambda_alpha = lambda_alpha
        self.lambda_erase_hair = lambda_erase_hair
        self.lambda_erase_alpha = lambda_erase_alpha
        self.lambda_occ_anchor = lambda_occ_anchor
        self.lambda_occ_tv = lambda_occ_tv
        self.perceptual = MultiScaleL1()

    def _semantic_nonhair_loss(self, clean_bg: torch.Tensor, target_parsing: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = get_celeba_parsing_logits(clean_bg)
        target = target_parsing[:, 0].long()
        loss_map = F.cross_entropy(logits, target, reduction="none").unsqueeze(1)
        return masked_mean(loss_map, mask)

    def forward(
        self,
        *,
        I_clean_bg: torch.Tensor,
        I_gt: torch.Tensor | None,
        I_bg0: torch.Tensor | None = None,
        recon_mask: torch.Tensor | None = None,
        erase_mask: torch.Tensor | None = None,
        I_source: torch.Tensor,
        I_align_clean: torch.Tensor,
        I_hair_coarse: torch.Tensor,
        H_align: torch.Tensor,
        A: torch.Tensor,
        M_occ_d: torch.Tensor,
        M_ring: torch.Tensor,
        M_safe: torch.Tensor,
        semantic_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = {}
        recon_mask = M_occ_d if recon_mask is None else recon_mask.clamp(0.0, 1.0)
        erase_mask = None if erase_mask is None else erase_mask.clamp(0.0, 1.0)

        if I_gt is not None:
            losses["L_fill"] = masked_mean((I_clean_bg - I_gt).abs(), recon_mask)
            losses["L_perc"] = self.perceptual(I_clean_bg, I_gt, recon_mask)
            ring_mask = (recon_mask + M_ring).clamp(0.0, 1.0)
            losses["L_edge"] = masked_mean((image_gradients(I_clean_bg) - image_gradients(I_gt)).abs(), ring_mask)
        else:
            zero = I_clean_bg.new_tensor(0.0)
            losses["L_fill"] = zero
            losses["L_perc"] = zero
            losses["L_edge"] = zero

        losses["L_keep"] = masked_mean((I_clean_bg - I_source).abs(), M_safe)
        losses["L_hair"] = masked_mean((I_align_clean - I_hair_coarse).abs(), H_align)
        losses["L_alpha"] = A.abs().mean() + image_gradients(A).abs().mean()

        parsing_logits = None
        if semantic_target is not None or erase_mask is not None:
            parsing_logits = get_celeba_parsing_logits(I_clean_bg)

        if semantic_target is not None:
            target = semantic_target[:, 0].long()
            loss_map = F.cross_entropy(parsing_logits, target, reduction="none").unsqueeze(1)
            losses["L_nonhair"] = masked_mean(loss_map, M_occ_d)
        else:
            losses["L_nonhair"] = I_clean_bg.new_tensor(0.0)

        if erase_mask is not None:
            hair_prob = torch.softmax(parsing_logits, dim=1)[:, HAIR_LABEL : HAIR_LABEL + 1]
            losses["L_erase_hair"] = masked_mean(hair_prob, erase_mask)
            losses["L_erase_alpha"] = masked_mean(1.0 - A, erase_mask)
        else:
            zero = I_clean_bg.new_tensor(0.0)
            losses["L_erase_hair"] = zero
            losses["L_erase_alpha"] = zero

        if erase_mask is not None and I_bg0 is not None:
            losses["L_occ_anchor"] = masked_mean((blur5(I_clean_bg) - blur5(I_bg0)).abs(), erase_mask)
            losses["L_occ_tv"] = total_variation_loss(I_clean_bg, erase_mask)
        else:
            zero = I_clean_bg.new_tensor(0.0)
            losses["L_occ_anchor"] = zero
            losses["L_occ_tv"] = zero

        total = (
            self.lambda_fill * losses["L_fill"]
            + self.lambda_perc * losses["L_perc"]
            + self.lambda_edge * losses["L_edge"]
            + self.lambda_nonhair * losses["L_nonhair"]
            + self.lambda_keep * losses["L_keep"]
            + self.lambda_hair * losses["L_hair"]
            + self.lambda_alpha * losses["L_alpha"]
            + self.lambda_erase_hair * losses["L_erase_hair"]
            + self.lambda_erase_alpha * losses["L_erase_alpha"]
            + self.lambda_occ_anchor * losses["L_occ_anchor"]
            + self.lambda_occ_tv * losses["L_occ_tv"]
        )
        losses["L_total"] = total
        return total, losses
