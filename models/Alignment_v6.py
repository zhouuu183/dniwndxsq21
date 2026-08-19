from __future__ import annotations

import torch
import torch.nn.functional as F

from models.Alignment import Alignment
from models.sean_codes.models.pix2pix_model import decode_sean, encode_sean
from utils.mask_formula_v6 import build_difference_masks, resize_mask


class Alignment_v6(Alignment):
    """
    v6 supports three alignment modes:
    - author_eq8: keep the author's Eq.(8) alignment unchanged
    - author_cleanup: keep author alignment for hair shape, but clean only the
      removed-hair vacuum before blending
    - structure_first: full semantic structure-first replacement in removed-hair
      regions
    """

    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__(opts, latent_encoder=latent_encoder, net=net)
        self.align_mode = str(getattr(self.opts, "align_v6_mode", "author_cleanup")).lower()
        self.cleanup_strength = float(getattr(self.opts, "align_v6_cleanup_strength", 0.70))
        self.fill_iterations = int(getattr(self.opts, "align_v6_fill_iterations", 3))
        self.fill_dilation = int(getattr(self.opts, "align_v6_fill_dilation", 2))
        self.face_structure_weight = float(getattr(self.opts, "align_v6_face_structure_weight", 0.90))
        self.ear_structure_weight = float(getattr(self.opts, "align_v6_ear_structure_weight", 0.96))
        self.body_structure_weight = float(getattr(self.opts, "align_v6_body_structure_weight", 0.88))
        self.background_fill_weight = float(getattr(self.opts, "align_v6_background_fill_weight", 0.82))
        self.other_structure_weight = float(getattr(self.opts, "align_v6_other_structure_weight", 1.00))

    @staticmethod
    def _ensure_image_batch(image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return image

    @staticmethod
    def _resolve_semantic_masks(
        masks: dict[str, torch.Tensor],
        mask_ref: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zeros = torch.zeros_like(mask_ref)
        ones = torch.ones_like(mask_ref)

        sample_face = masks.get("M_sample_face", masks.get("M_sample_skin", zeros))
        sample_ear = masks.get("M_sample_ear", zeros)
        sample_body = masks.get("M_sample_body", zeros)
        sample_background = masks.get(
            "M_sample_background",
            masks.get("M_sample", masks.get("M_valid", ones)),
        )

        route_face = masks.get("M_route_face", masks.get("M_route_skin", zeros))
        route_ear = masks.get("M_route_ear", zeros)
        route_body = masks.get("M_route_body", zeros)
        route_background = masks.get("M_route_background", sample_background)
        route_other = masks.get("M_route_other")
        if route_other is None:
            route_known = (route_face + route_ear + route_body + route_background).clamp(0, 1)
            route_other = (1.0 - route_known).clamp(0, 1)

        return {
            "sample_face_mask": sample_face.float(),
            "sample_ear_mask": sample_ear.float(),
            "sample_body_mask": sample_body.float(),
            "sample_background_mask": sample_background.float(),
            "route_face_mask": route_face.float(),
            "route_ear_mask": route_ear.float(),
            "route_body_mask": route_body.float(),
            "route_background_mask": route_background.float(),
            "route_other_mask": route_other.float(),
        }

    def _dilated_partial_fill(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = resize_mask(valid_mask, size=features.shape[-2:], mode="nearest").float()
        filled = features * valid_mask
        known = valid_mask.clone()

        feat_kernel = torch.ones((features.shape[1], 1, 3, 3), device=features.device, dtype=features.dtype)
        count_kernel = torch.ones((1, 1, 3, 3), device=features.device, dtype=features.dtype)

        for _ in range(max(1, self.fill_iterations)):
            feat_sum = F.conv2d(
                filled,
                feat_kernel,
                padding=self.fill_dilation,
                dilation=self.fill_dilation,
                groups=features.shape[1],
            )
            feat_count = F.conv2d(
                known,
                count_kernel,
                padding=self.fill_dilation,
                dilation=self.fill_dilation,
            )
            propagated = feat_sum / feat_count.clamp(min=1.0e-6)
            can_fill = ((1.0 - known) * (feat_count > 0).float()).clamp(0, 1)
            filled = filled * known + propagated * can_fill
            known = torch.clamp(known + can_fill, 0, 1)

        return filled + (1.0 - known) * features

    def _author_eq8_fusion(
        self,
        latent_F_source: torch.Tensor,
        latent_F_shape: torch.Tensor,
        latent_F_source_inpaint: torch.Tensor,
        latent_F_shape_inpaint: torch.Tensor,
        hair_mask_source: torch.Tensor,
        hair_mask_shape: torch.Tensor,
        hair_mask_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        masks = torch.cat(
            [
                1.0 - (1.0 - hair_mask_source) * (1.0 - hair_mask_target),
                hair_mask_target,
                hair_mask_shape * hair_mask_target,
            ],
            dim=0,
        )
        dilate, erosion = self.dilate_erosion.mask(masks)
        free_mask = torch.stack([dilate[0], erosion[1], erosion[2]], dim=0)
        free_mask_down = F.interpolate(
            free_mask.float(),
            size=latent_F_source.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        interpolation_low = 1.0 - free_mask_down

        latent_F_align = latent_F_source_inpaint + interpolation_low[0:1] * (latent_F_source - latent_F_source_inpaint)
        latent_F_align = latent_F_shape_inpaint + interpolation_low[1:2] * (latent_F_align - latent_F_shape_inpaint)
        latent_F_align = latent_F_shape + interpolation_low[2:3] * (latent_F_align - latent_F_shape)

        return {
            "latent_F_align": latent_F_align,
            "author_free_mask_32": free_mask_down,
            "author_interpolation_low_32": interpolation_low,
        }

    def _author_cleanup_fusion(
        self,
        author_latent_F_align: torch.Tensor,
        latent_F_removed: torch.Tensor,
        hair_mask_target: torch.Tensor,
        diff_soft_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if diff_soft_mask is None:
            cleanup_mask_32 = resize_mask(hair_mask_target, author_latent_F_align.shape[-2:], mode="bicubic")
            cleanup_mask_32 = torch.zeros_like(cleanup_mask_32)
        else:
            cleanup_mask_32 = resize_mask(diff_soft_mask, author_latent_F_align.shape[-2:], mode="bilinear")

        h_align_32 = resize_mask(hair_mask_target, author_latent_F_align.shape[-2:], mode="bicubic")
        cleanup_mask_32 = (cleanup_mask_32 * (1.0 - h_align_32)).clamp(0, 1)
        latent_F_align = author_latent_F_align + self.cleanup_strength * cleanup_mask_32 * (
            latent_F_removed - author_latent_F_align
        )

        return {
            "latent_F_align": latent_F_align,
            "author_cleanup_mask_32": cleanup_mask_32,
        }

    def _v6_fusion(
        self,
        latent_F_source: torch.Tensor,
        latent_F_shape: torch.Tensor,
        latent_F_source_inpaint: torch.Tensor,
        latent_F_shape_inpaint: torch.Tensor,
        hair_mask_source: torch.Tensor,
        hair_mask_shape: torch.Tensor,
        hair_mask_target: torch.Tensor,
        sample_face_mask: torch.Tensor,
        sample_ear_mask: torch.Tensor,
        sample_body_mask: torch.Tensor,
        sample_background_mask: torch.Tensor,
        route_face_mask: torch.Tensor,
        route_ear_mask: torch.Tensor,
        route_body_mask: torch.Tensor,
        route_background_mask: torch.Tensor,
        route_other_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        h_source_32 = resize_mask(hair_mask_source, latent_F_source.shape[-2:], mode="bicubic")
        h_shape_32 = resize_mask(hair_mask_shape, latent_F_source.shape[-2:], mode="bicubic")
        h_align_32 = resize_mask(hair_mask_target, latent_F_source.shape[-2:], mode="bicubic")
        sample_face_mask_32 = resize_mask(sample_face_mask, latent_F_source.shape[-2:], mode="nearest")
        sample_ear_mask_32 = resize_mask(sample_ear_mask, latent_F_source.shape[-2:], mode="nearest")
        sample_body_mask_32 = resize_mask(sample_body_mask, latent_F_source.shape[-2:], mode="nearest")
        sample_background_mask_32 = resize_mask(sample_background_mask, latent_F_source.shape[-2:], mode="nearest")
        route_face_mask_32 = resize_mask(route_face_mask, latent_F_source.shape[-2:], mode="nearest")
        route_ear_mask_32 = resize_mask(route_ear_mask, latent_F_source.shape[-2:], mode="nearest")
        route_body_mask_32 = resize_mask(route_body_mask, latent_F_source.shape[-2:], mode="nearest")
        route_background_mask_32 = resize_mask(route_background_mask, latent_F_source.shape[-2:], mode="nearest")
        route_other_mask_32 = resize_mask(route_other_mask, latent_F_source.shape[-2:], mode="nearest")

        m_diff_32 = (h_source_32 - h_align_32).clamp(min=0.0, max=1.0)
        m_valid_32 = ((1.0 - h_source_32) * (1.0 - h_align_32)).clamp(0, 1)
        fill_valid_face_32 = (m_valid_32 * sample_face_mask_32).clamp(0, 1)
        fill_valid_ear_32 = (m_valid_32 * sample_ear_mask_32).clamp(0, 1)
        fill_valid_body_32 = (m_valid_32 * sample_body_mask_32).clamp(0, 1)
        fill_valid_background_32 = (m_valid_32 * sample_background_mask_32).clamp(0, 1)

        latent_F_fill_face = self._dilated_partial_fill(latent_F_source_inpaint, fill_valid_face_32)
        latent_F_fill_ear = self._dilated_partial_fill(latent_F_source_inpaint, fill_valid_ear_32)
        latent_F_fill_body = self._dilated_partial_fill(latent_F_source_inpaint, fill_valid_body_32)
        latent_F_fill_background = self._dilated_partial_fill(latent_F_source_inpaint, fill_valid_background_32)

        face_prior_32 = (
            self.face_structure_weight * latent_F_source_inpaint
            + (1.0 - self.face_structure_weight) * latent_F_fill_face
        )
        ear_prior_32 = (
            self.ear_structure_weight * latent_F_source_inpaint
            + (1.0 - self.ear_structure_weight) * latent_F_fill_ear
        )
        body_prior_32 = (
            self.body_structure_weight * latent_F_source_inpaint
            + (1.0 - self.body_structure_weight) * latent_F_fill_body
        )
        background_prior_32 = (
            self.background_fill_weight * latent_F_fill_background
            + (1.0 - self.background_fill_weight) * latent_F_source_inpaint
        )
        other_prior_32 = (
            self.other_structure_weight * latent_F_source_inpaint
            + (1.0 - self.other_structure_weight) * latent_F_fill_background
        )

        route_fallback_32 = (
            1.0
            - route_face_mask_32
            - route_ear_mask_32
            - route_body_mask_32
            - route_background_mask_32
            - route_other_mask_32
        ).clamp(0, 1)
        latent_F_removed = (
            route_face_mask_32 * face_prior_32
            + route_ear_mask_32 * ear_prior_32
            + route_body_mask_32 * body_prior_32
            + route_background_mask_32 * background_prior_32
            + (route_other_mask_32 + route_fallback_32) * other_prior_32
        )

        latent_F_align = (
            h_align_32 * h_shape_32 * latent_F_shape
            + h_align_32 * (1.0 - h_shape_32) * latent_F_shape_inpaint
            + (1.0 - h_align_32) * (1.0 - h_source_32) * latent_F_source
            + (1.0 - h_align_32) * h_source_32 * latent_F_removed
        )

        return {
            "latent_F_align": latent_F_align,
            "latent_F_fill": latent_F_removed,
            "latent_F_fill_face": latent_F_fill_face,
            "latent_F_fill_ear": latent_F_fill_ear,
            "latent_F_fill_body": latent_F_fill_body,
            "latent_F_fill_background": latent_F_fill_background,
            "latent_F_prior_face": face_prior_32,
            "latent_F_prior_ear": ear_prior_32,
            "latent_F_prior_body": body_prior_32,
            "latent_F_prior_background": background_prior_32,
            "H_source_32": h_source_32,
            "H_shape_32": h_shape_32,
            "H_align_32": h_align_32,
            "M_diff_32": m_diff_32,
            "M_valid_32": m_valid_32,
            "M_sample_face_32": sample_face_mask_32,
            "M_sample_ear_32": sample_ear_mask_32,
            "M_sample_body_32": sample_body_mask_32,
            "M_sample_background_32": sample_background_mask_32,
            "M_route_face_32": route_face_mask_32,
            "M_route_ear_32": route_ear_mask_32,
            "M_route_body_32": route_body_mask_32,
            "M_route_background_32": route_background_mask_32,
            "M_route_other_32": route_other_mask_32,
        }

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        img1_in = name_to_embed[im_name1]["image_256"]
        img2_in = name_to_embed[im_name2]["image_256"]
        latent_S_1, latent_F_1 = name_to_embed[im_name1]["S"], name_to_embed[im_name1]["F"]
        latent_F_2 = name_to_embed[im_name2]["F"]

        if img1_in is img2_in:
            source_parsing = name_to_embed[im_name1]["mask"]
            hair_mask_source = torch.where(
                source_parsing == 13,
                torch.ones_like(source_parsing),
                torch.zeros_like(source_parsing),
            ).float()
            masks = build_difference_masks(
                source_hair_mask=hair_mask_source,
                target_hair_mask=hair_mask_source,
                source_parsing=source_parsing,
                target_parsing=source_parsing,
                diff_dilate_radius=getattr(self.opts, "diff_mask_dilate", 5),
                diff_blur_kernel_size=getattr(self.opts, "diff_mask_blur_kernel", 11),
                diff_blur_sigma=getattr(self.opts, "diff_mask_blur_sigma", 0.0) or None,
            )
            return {
                "latent_F_align": latent_F_1,
                "HM_X": hair_mask_source,
                "source_hair_mask": hair_mask_source,
                "shape_hair_mask": hair_mask_source,
                "source_parsing": source_parsing,
                "target_parsing": source_parsing,
                "source_image_256": img1_in * 2 - 1,
                "shape_image_256": img1_in * 2 - 1,
                "source_inpaint_256": img1_in * 2 - 1,
                "shape_inpaint_256": img1_in * 2 - 1,
                **masks,
            }

        inp_mask1, hair_mask1, inp_mask2, hair_mask2, target_mask, hair_mask_target = (
            self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        )

        images = torch.cat([img1_in, img2_in], dim=0)
        labels = torch.cat([inp_mask1, inp_mask2], dim=0)
        img1_code, img2_code = encode_sean(self.sean_model, images, labels)
        gen1_sean = self._ensure_image_batch(decode_sean(self.sean_model, img1_code.unsqueeze(0), target_mask))
        gen2_sean = self._ensure_image_batch(decode_sean(self.sean_model, img2_code.unsqueeze(0), target_mask))

        enc_imgs = self.latent_encoder([gen1_sean[0], gen2_sean[0]])
        latent_F_source_inpaint = enc_imgs["F"][0].unsqueeze(0)
        latent_F_shape_inpaint = enc_imgs["F"][1].unsqueeze(0)
        author_fusion = self._author_eq8_fusion(
            latent_F_source=latent_F_1,
            latent_F_shape=latent_F_2,
            latent_F_source_inpaint=latent_F_source_inpaint,
            latent_F_shape_inpaint=latent_F_shape_inpaint,
            hair_mask_source=hair_mask1,
            hair_mask_shape=hair_mask2,
            hair_mask_target=hair_mask_target,
        )

        masks = build_difference_masks(
            source_hair_mask=hair_mask1,
            target_hair_mask=hair_mask_target,
            source_parsing=inp_mask1,
            target_parsing=target_mask,
            diff_dilate_radius=getattr(self.opts, "diff_mask_dilate", 5),
            diff_blur_kernel_size=getattr(self.opts, "diff_mask_blur_kernel", 11),
            diff_blur_sigma=getattr(self.opts, "diff_mask_blur_sigma", 0.0) or None,
        )
        semantic_masks = self._resolve_semantic_masks(masks, hair_mask1.float())
        v6_fusion = self._v6_fusion(
            latent_F_source=latent_F_1,
            latent_F_shape=latent_F_2,
            latent_F_source_inpaint=latent_F_source_inpaint,
            latent_F_shape_inpaint=latent_F_shape_inpaint,
            hair_mask_source=hair_mask1,
            hair_mask_shape=hair_mask2,
            hair_mask_target=hair_mask_target,
            sample_face_mask=semantic_masks["sample_face_mask"],
            sample_ear_mask=semantic_masks["sample_ear_mask"],
            sample_body_mask=semantic_masks["sample_body_mask"],
            sample_background_mask=semantic_masks["sample_background_mask"],
            route_face_mask=semantic_masks["route_face_mask"],
            route_ear_mask=semantic_masks["route_ear_mask"],
            route_body_mask=semantic_masks["route_body_mask"],
            route_background_mask=semantic_masks["route_background_mask"],
            route_other_mask=semantic_masks["route_other_mask"],
        )
        cleanup_fusion = self._author_cleanup_fusion(
            author_latent_F_align=author_fusion["latent_F_align"],
            latent_F_removed=v6_fusion["latent_F_fill"],
            hair_mask_target=hair_mask_target,
            diff_soft_mask=masks.get("M_diff_soft"),
        )
        if self.align_mode == "author_eq8":
            fusion = author_fusion
        elif self.align_mode == "author_cleanup":
            fusion = cleanup_fusion
        elif self.align_mode == "structure_first":
            fusion = v6_fusion
        else:
            raise ValueError(
                f"Unsupported align_v6_mode={self.align_mode!r}. "
                "Choose 'author_eq8', 'author_cleanup', or 'structure_first'."
            )

        gen1_sean_256 = F.interpolate(gen1_sean, size=img1_in.shape[-2:], mode="bicubic", align_corners=False)
        gen2_sean_256 = F.interpolate(gen2_sean, size=img2_in.shape[-2:], mode="bicubic", align_corners=False)

        return {
            "latent_F_align": fusion["latent_F_align"],
            "latent_F_fill": v6_fusion["latent_F_fill"],
            "HM_X": hair_mask_target,
            "source_hair_mask": hair_mask1.float(),
            "shape_hair_mask": hair_mask2.float(),
            "source_parsing": inp_mask1,
            "target_parsing": target_mask,
            "source_image_256": img1_in * 2 - 1,
            "shape_image_256": img2_in * 2 - 1,
            "source_inpaint_256": gen1_sean_256,
            "shape_inpaint_256": gen2_sean_256,
            "latent_F_align_author": author_fusion["latent_F_align"],
            "latent_F_align_cleanup": cleanup_fusion["latent_F_align"],
            "alignment_mode_author_eq8": torch.tensor(
                1.0 if self.align_mode == "author_eq8" else 0.0,
                device=latent_F_1.device,
            ),
            "alignment_mode_author_cleanup": torch.tensor(
                1.0 if self.align_mode == "author_cleanup" else 0.0,
                device=latent_F_1.device,
            ),
            "alignment_mode_structure_first": torch.tensor(
                1.0 if self.align_mode == "structure_first" else 0.0,
                device=latent_F_1.device,
            ),
            **masks,
            **v6_fusion,
            **author_fusion,
            **cleanup_fusion,
            "latent_F_align": fusion["latent_F_align"],
            "latent_S_src": latent_S_1,
        }
