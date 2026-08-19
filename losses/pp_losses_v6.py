from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.pp_losses import LossBuilderMulti, normalize
from utils.mask_formula_v6 import ensure_mask_4d


class LossBuilderV6(LossBuilderMulti):
    """
    v6 keeps hair / landmark supervision, but relaxes reconstruction in the
    removed-hair vacuum by switching face LPIPS / ID to M_valid.
    """

    def _resolve_valid_mask(self, aux: dict | None, target_mask: torch.Tensor) -> torch.Tensor:
        if aux is not None and aux.get("valid_mask") is not None:
            return ensure_mask_4d(aux["valid_mask"]).float()
        return ensure_mask_4d(target_mask).float()

    def _resolve_inpaint_mask(
        self,
        aux: dict | None,
        target_mask: torch.Tensor,
        HT_E: torch.Tensor,
    ) -> torch.Tensor:
        if aux is not None and aux.get("diff_mask") is not None:
            return ensure_mask_4d(aux["diff_mask"]).float()
        return ((1.0 - ensure_mask_4d(target_mask).float()) * (1.0 - ensure_mask_4d(HT_E).float())).clamp(0, 1)

    def __call__(self, source, target, target_mask, HT_E, gen_w, F_w, gen_F, F_gen, aux=None, **kwargs):
        losses = {}
        gen_w_256 = self.downsample_256(gen_w)
        gen_F_256 = self.downsample_256(gen_F)
        valid_mask = self._resolve_valid_mask(aux, target_mask)
        hair_mask = ensure_mask_4d(HT_E).float()

        with torch.no_grad():
            target_512 = F.interpolate(target, size=(512, 512), mode="bilinear").clip(0, 1)
            seg_target = self.DiceLoss.calc_landmark(target_512)
            seg_target = F.interpolate(seg_target, size=(256, 256), mode="nearest")
        seg_gen = F.interpolate(self.DiceLoss.calc_landmark((gen_F + 1) / 2), size=(256, 256), mode="nearest")
        losses["DiceLoss"] = self.losses_dict["landmark"] * self.DiceLoss(seg_gen, seg_target)

        losses["id_valid"] = self.losses_dict["id"] * (
            self.IDLoss(normalize(source) * valid_mask, gen_w_256 * valid_mask)
            + self.IDLoss(normalize(source) * valid_mask, gen_F_256 * valid_mask)
        )
        losses["feat_rec"] = self.losses_dict["feat_rec"] * self.FeatReconLoss(F_w.detach(), F_gen)
        losses["lpips_valid"] = 0.5 * self.losses_dict["lpips_scale"] * (
            self.LPIPS(normalize(source) * valid_mask, gen_w_256 * valid_mask)
            + self.LPIPS(normalize(source) * valid_mask, gen_F_256 * valid_mask)
        )
        losses["lpips_hair"] = 0.5 * self.losses_dict["lpips_scale"] * (
            self.LPIPS(normalize(target) * hair_mask, gen_w_256 * hair_mask)
            + self.LPIPS(normalize(target) * hair_mask, gen_F_256 * hair_mask)
        )

        if self.losses_dict["inpaint"] != 0.0:
            inpaint_mask = self._resolve_inpaint_mask(aux, target_mask, HT_E)
            smooth_mask = self.dilated(inpaint_mask)
            losses["inpaint"] = 0.5 * self.losses_dict["inpaint"] * self.LPIPS(
                normalize(target) * smooth_mask,
                gen_F_256 * smooth_mask,
            )
            losses["inpaint"] += 0.5 * self.losses_dict["inpaint"] * self.LPIPS(
                gen_w_256.detach() * smooth_mask * (1.0 - hair_mask),
                gen_F_256 * smooth_mask * (1.0 - hair_mask),
            )

        return losses
