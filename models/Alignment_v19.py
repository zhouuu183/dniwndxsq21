from __future__ import annotations

import torch
import torch.nn.functional as F

from models.Alignment import Alignment
from models.CtrlHair.global_value_utils import HAIR_IDX
from models.CtrlHair.shape_branch.solver import get_hair_face_code, get_new_shape
from models.Net import get_segmentation
from models.ShapeAdaptor_v19 import (
    BoundaryLayerShapeAdaptor_v19,
    build_pseudo_target_mask_v19,
    compose_boundary_detail_feature_v19,
    extract_boundary_priors_v19,
    face_protect_from_parsing_v19,
    hair_mask_from_parsing_v19,
    load_shape_adaptor_v19,
    merge_hair_mask_into_parsing_v19,
)
from models.sean_codes.models.pix2pix_model import decode_sean, encode_sean
from utils.mask_delta_v18 import drop_parsing_labels, filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_vis_mask


class Alignment_v19(Alignment):
    """
    v19 alignment in the source pose.

    The key rule is that the reference hairstyle is first projected into the
    source pose through the target parsing layout, and only then used to build
    priors and latent targets. Raw reference-pose hair masks are never mixed
    directly into the source-pose target.
    """

    def __init__(self, opts, latent_encoder=None, net=None):
        super().__init__(opts, latent_encoder=latent_encoder, net=net)
        self.boundary_radius = int(getattr(self.opts, "shape_v19_boundary_radius", 3))
        self.mask_residual_gain = float(getattr(self.opts, "shape_v19_mask_residual_gain", 1.0))
        self.shape_strength = float(getattr(self.opts, "shape_v19_strength", 1.0))
        self.shape_prior_strength = float(getattr(self.opts, "shape_v19_prior_strength", 0.35))
        self.shape_detail_strength = float(getattr(self.opts, "shape_v19_detail_strength", 1.0))
        self.use_rotation = bool(getattr(self.opts, "shape_v19_use_rotation", False))

        self.shape_adapter = None
        adapter_ckpt = str(getattr(self.opts, "shape_adapter_v19_checkpoint", "") or "").strip()
        if bool(getattr(self.opts, "use_shape_v19", False)) and adapter_ckpt:
            self.shape_adapter = BoundaryLayerShapeAdaptor_v19(
                hidden_channels=int(getattr(self.opts, "shape_adapter_v19_hidden", 256))
            ).to(self.opts.device).eval()
            checkpoint = torch.load(adapter_ckpt, map_location=self.opts.device)
            load_shape_adaptor_v19(self.shape_adapter, checkpoint)

    @staticmethod
    def _ensure_batch(image: torch.Tensor) -> torch.Tensor:
        return image.unsqueeze(0) if image.dim() == 3 else image

    def _segment_primary(
        self,
        image_norm_256: torch.Tensor,
        subject_support: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_01 = ((image_norm_256 + 1.0) / 2.0).clamp(0, 1)
        parsing = get_segmentation(self.to_bisenet(image_01))
        parsing, _ = filter_parsing_to_primary_subject(parsing, subject_support=subject_support)
        return parsing

    @staticmethod
    def _mix_source_identity_with_reference_hair(
        source_code: torch.Tensor,
        reference_code: torch.Tensor,
    ) -> torch.Tensor:
        mixed_code = source_code.clone()
        mixed_code[:, HAIR_IDX] = reference_code[:, HAIR_IDX]
        return mixed_code

    def _decode_source_pose_pair(
        self,
        source_image_256: torch.Tensor,
        reference_image_256: torch.Tensor,
        source_parsing: torch.Tensor,
        reference_parsing: torch.Tensor,
        target_parsing: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.cat([source_image_256, reference_image_256], dim=0)
        labels = torch.cat(
            [
                drop_parsing_labels(source_parsing),
                drop_parsing_labels(reference_parsing),
            ],
            dim=0,
        )
        target_clean = drop_parsing_labels(target_parsing)
        all_codes = encode_sean(self.sean_model, images, labels)
        source_code = all_codes[0:1]
        reference_code = all_codes[1:2]
        reference_hair_code = self._mix_source_identity_with_reference_hair(source_code, reference_code)
        source_render = self._ensure_batch(decode_sean(self.sean_model, source_code, target_clean))
        reference_render = self._ensure_batch(decode_sean(self.sean_model, reference_hair_code, target_clean))
        if source_render.shape[-2:] != source_image_256.shape[-2:]:
            source_render = F.interpolate(
                source_render,
                size=source_image_256.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
        if reference_render.shape[-2:] != source_image_256.shape[-2:]:
            reference_render = F.interpolate(
                reference_render,
                size=source_image_256.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )
        return source_render, reference_render

    def _derive_priors(
        self,
        aligned_reference_image_256: torch.Tensor,
        source_support: torch.Tensor,
        source_hair_mask: torch.Tensor,
        coarse_target_mask: torch.Tensor,
        source_face_protect: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        aligned_reference_parsing = self._segment_primary(aligned_reference_image_256, subject_support=source_support)
        aligned_reference_hair = hair_mask_from_parsing_v19(aligned_reference_parsing)
        priors = extract_boundary_priors_v19(
            image_256=aligned_reference_image_256,
            hair_mask_256=aligned_reference_hair,
            source_hair_mask_256=source_hair_mask,
            coarse_target_mask_256=coarse_target_mask,
            source_face_protect_256=source_face_protect,
            boundary_radius=self.boundary_radius,
        )
        pseudo_target_mask, pseudo_boundary_residual = build_pseudo_target_mask_v19(
            coarse_target_mask=coarse_target_mask,
            rotated_hair_mask=aligned_reference_hair,
            priors_256=priors,
            residual_gain=self.mask_residual_gain,
        )
        priors["pseudo_target_mask"] = pseudo_target_mask
        priors["pseudo_boundary_residual"] = pseudo_boundary_residual
        return aligned_reference_parsing, priors

    @staticmethod
    def _build_transfer_support(
        source_hair_mask: torch.Tensor,
        target_hair_mask: torch.Tensor,
        boundary_band: torch.Tensor,
        source_face_protect: torch.Tensor,
        out_hw: tuple[int, int],
        detail_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source_hair_mask = F.interpolate(source_hair_mask.float(), size=out_hw, mode="bilinear", align_corners=False)
        target_hair_mask = F.interpolate(target_hair_mask.float(), size=out_hw, mode="bilinear", align_corners=False)
        boundary_band = F.interpolate(boundary_band.float(), size=out_hw, mode="bilinear", align_corners=False)
        source_face_protect = F.interpolate(source_face_protect.float(), size=out_hw, mode="bilinear", align_corners=False)
        add_region = (target_hair_mask * (1.0 - source_hair_mask)).clamp(0, 1)
        remove_region = (source_hair_mask * (1.0 - target_hair_mask)).clamp(0, 1)
        support = torch.maximum(
            target_hair_mask,
            torch.maximum(
                (0.70 * remove_region).clamp(0, 1),
                torch.maximum((0.62 * add_region).clamp(0, 1), (0.58 * boundary_band).clamp(0, 1)),
            ),
        )
        if detail_map is not None:
            detail_map = F.interpolate(detail_map.float(), size=out_hw, mode="bilinear", align_corners=False)
            support = torch.maximum(support, (0.35 * target_hair_mask * detail_map).clamp(0, 1))
        source_face_protect = F.max_pool2d(source_face_protect, kernel_size=3, stride=1, padding=1)
        return (support * (1.0 - source_face_protect)).clamp(0, 1)

    def _build_base_feature(
        self,
        source_aligned: torch.Tensor,
        reference_aligned: torch.Tensor,
        source_hair_mask: torch.Tensor,
        target_hair_mask: torch.Tensor,
        boundary_band: torch.Tensor,
        source_face_protect: torch.Tensor,
        detail_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        support = self._build_transfer_support(
            source_hair_mask=source_hair_mask,
            target_hair_mask=target_hair_mask,
            boundary_band=boundary_band,
            source_face_protect=source_face_protect,
            out_hw=source_aligned.shape[-2:],
            detail_map=detail_map,
        )
        return source_aligned + support * (reference_aligned - source_aligned)

    def _run_shape_adapter(
        self,
        F_base: torch.Tensor,
        F_src: torch.Tensor,
        F_shape: torch.Tensor,
        F_base64: torch.Tensor,
        F_shape64: torch.Tensor,
        F_reference: torch.Tensor,
        F_reference64: torch.Tensor,
        priors: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.shape_adapter is None:
            latent_F64_detail = compose_boundary_detail_feature_v19(
                base_feature_64=F_base64,
                shape_feature_64=F_reference64,
                boundary_band_256=priors["boundary_band"],
                detail_map_256=priors["hair_detail_map"],
                strength=self.shape_detail_strength,
            )
            return F_base, latent_F64_detail

        adapter_outputs = self.shape_adapter(
            F_base=F_base,
            F_src=F_src,
            F_shape=F_shape,
            F_base64=F_base64,
            F_shape64=F_shape64,
            F_reference=F_reference,
            F_reference64=F_reference64,
            priors_256=priors,
            strength=self.shape_strength,
            shape_prior_strength=self.shape_prior_strength,
            detail_strength=self.shape_detail_strength,
        )
        return adapter_outputs["latent_F_refined"], adapter_outputs["latent_F64_detail"]

    @torch.inference_mode()
    def shape_module(self, im_name1: str, im_name2: str, name_to_embed, only_target: bool = False, **kwargs):
        device = self.opts.device
        source_image = name_to_embed[im_name1]["image_256"]
        reference_image = name_to_embed[im_name2]["image_256"]
        source_latent_w = name_to_embed[im_name1]["W"]
        reference_latent_w = name_to_embed[im_name2]["W"]

        source_parsing, source_support = filter_parsing_to_primary_subject(name_to_embed[im_name1]["mask"])
        reference_parsing, reference_support = filter_parsing_to_primary_subject(name_to_embed[im_name2]["mask"])

        style_reference_image = reference_image
        style_reference_parsing = reference_parsing
        shape_reference_parsing = reference_parsing
        if self.use_rotation and source_image is not reference_image:
            rotate_to = self.rotate_model(reference_latent_w[:, :6], source_latent_w[:, :6])
            rotate_to = torch.cat((rotate_to, reference_latent_w[:, 6:]), dim=1)
            rotated_reference_image, _ = self.net.generator([rotate_to], input_is_latent=True, return_latents=False)
            shape_reference_parsing = self._segment_primary(rotated_reference_image, subject_support=reference_support)

        if source_image is not reference_image:
            source_shape_mask = drop_parsing_labels(source_parsing)
            reference_shape_mask = drop_parsing_labels(shape_reference_parsing)
            face_code_source, _ = get_hair_face_code(self.mask_generator, source_shape_mask[0, 0, ...])
            _, hair_code_reference = get_hair_face_code(self.mask_generator, reference_shape_mask[0, 0, ...])
            target_mask = get_new_shape(self.mask_generator, face_code_source, hair_code_reference)[None, None]
            target_mask, _ = filter_parsing_to_primary_subject(target_mask, subject_support=source_support)
            coarse_target_mask = hair_mask_from_parsing_v19(target_mask.to(device))
        else:
            coarse_target_mask = hair_mask_from_parsing_v19(source_parsing.to(device))

        source_hair_mask = hair_mask_from_parsing_v19(source_parsing.to(device))
        reference_hair_mask = hair_mask_from_parsing_v19(reference_parsing.to(device))
        source_face_protect = face_protect_from_parsing_v19(source_parsing.to(device))
        target_parsing = merge_hair_mask_into_parsing_v19(
            base_parsing_mask=source_parsing.to(device),
            source_parsing_mask=source_parsing.to(device),
            hair_mask=coarse_target_mask,
        )

        if source_image is reference_image:
            priors = extract_boundary_priors_v19(
                image_256=name_to_embed[im_name1]["image_norm_256"],
                hair_mask_256=source_hair_mask,
                source_hair_mask_256=source_hair_mask,
                coarse_target_mask_256=coarse_target_mask,
                source_face_protect_256=source_face_protect,
                boundary_radius=self.boundary_radius,
            )
            pseudo_target_mask, pseudo_boundary_residual = build_pseudo_target_mask_v19(
                coarse_target_mask=coarse_target_mask,
                rotated_hair_mask=source_hair_mask,
                priors_256=priors,
                residual_gain=self.mask_residual_gain,
            )
            priors["pseudo_target_mask"] = pseudo_target_mask
            priors["pseudo_boundary_residual"] = pseudo_boundary_residual
            aligned_reference_parsing = source_parsing.to(device)
            aligned_reference_image_256 = name_to_embed[im_name1]["image_norm_256"]
            coarse_source_render_256 = aligned_reference_image_256
            coarse_reference_render_256 = aligned_reference_image_256
        else:
            coarse_source_render_256, coarse_reference_render_256 = self._decode_source_pose_pair(
                source_image_256=source_image.to(device),
                reference_image_256=style_reference_image.to(device),
                source_parsing=source_parsing.to(device),
                reference_parsing=style_reference_parsing.to(device),
                target_parsing=target_parsing,
            )
            aligned_reference_parsing, priors = self._derive_priors(
                aligned_reference_image_256=coarse_reference_render_256,
                source_support=source_support.to(device),
                source_hair_mask=source_hair_mask,
                coarse_target_mask=coarse_target_mask,
                source_face_protect=source_face_protect,
            )
            aligned_reference_image_256 = coarse_reference_render_256

        outputs = {
            "source_support": source_support.to(device),
            "source_parsing_mask": source_parsing.to(device),
            "reference_parsing_mask": reference_parsing.to(device),
            "aligned_reference_parsing_mask": style_reference_parsing.to(device),
            "alignment_reference_image_256": style_reference_image.to(device),
            "target_parsing_mask": target_parsing,
            "source_hair_mask": source_hair_mask,
            "reference_hair_mask": reference_hair_mask,
            "coarse_target_mask": coarse_target_mask,
            "target_hair_mask": priors["pseudo_target_mask"],
            "pseudo_target_mask": priors["pseudo_target_mask"],
            "source_face_protect": source_face_protect,
            "aligned_reference_image_256": aligned_reference_image_256,
            "coarse_source_render_256": coarse_source_render_256,
            "coarse_reference_render_256": coarse_reference_render_256,
            "priors": priors,
            "HM_X": priors["pseudo_target_mask"],
        }

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_vis_mask(output_dir, "Shape_v19", f"{im_name1}_source_mask.png", source_parsing)
            save_vis_mask(output_dir, "Shape_v19", f"{im_name2}_reference_mask.png", reference_parsing)
            if self.use_rotation and source_image is not reference_image:
                save_vis_mask(output_dir, "Shape_v19", f"{im_name2}_rotated_for_shape_mask.png", shape_reference_parsing)
            save_vis_mask(output_dir, "Shape_v19", f"{im_name1}_{im_name2}_target_mask.png", target_parsing)
            save_gen_image(output_dir, "Shape_v19", f"{im_name1}_{im_name2}_reference_source_pose.png", aligned_reference_image_256)

        if only_target:
            return {
                "HM_X": outputs["HM_X"],
                "pseudo_target_mask": outputs["pseudo_target_mask"],
                "priors": outputs["priors"],
            }
        return outputs

    @torch.inference_mode()
    def align_images(self, im_name1, im_name2, name_to_embed, **kwargs):
        if self.latent_encoder is None:
            raise RuntimeError("Alignment_v19 requires a latent_encoder that returns F32/F64 features.")

        source_image = name_to_embed[im_name1]["image_256"]
        reference_image = name_to_embed[im_name2]["image_256"]
        source_f32 = name_to_embed[im_name1]["F32"]
        source_f64 = name_to_embed[im_name1]["F64"]

        shape_state = self.shape_module(im_name1, im_name2, name_to_embed, only_target=False, **kwargs)
        coarse_target_mask = shape_state["coarse_target_mask"]
        source_hair_mask = shape_state["source_hair_mask"]
        source_face_protect = shape_state["source_face_protect"]

        if source_image is reference_image:
            priors = shape_state["priors"]
            latent_F_base = source_f32
            latent_F_base64 = source_f64
            latent_F_shape = source_f32
            latent_F_shape64 = source_f64
            latent_F_reference = source_f32
            latent_F_reference64 = source_f64
            latent_F_intermediate = source_f32
            latent_F_intermediate64 = source_f64
            latent_F_align, latent_F64_detail = self._run_shape_adapter(
                F_base=latent_F_base,
                F_src=source_f32,
                F_shape=latent_F_shape,
                F_base64=latent_F_base64,
                F_shape64=latent_F_shape64,
                F_reference=latent_F_reference,
                F_reference64=latent_F_reference64,
                priors=priors,
            )
            return {
                "latent_F_align": latent_F_align,
                "latent_F_base": latent_F_base,
                "latent_F_shape": latent_F_shape,
                "latent_F_reference": latent_F_reference,
                "latent_F_intermediate": latent_F_intermediate,
                "latent_F_base64": latent_F_base64,
                "latent_F_shape64": latent_F_shape64,
                "latent_F_reference64": latent_F_reference64,
                "latent_F_intermediate64": latent_F_intermediate64,
                "latent_F64_detail": latent_F64_detail,
                "aligned_reference_image_256": shape_state["aligned_reference_image_256"],
                "target_hair_mask": priors["pseudo_target_mask"],
                "boundary_band": priors["boundary_band"],
                "priors": priors,
                "HM_X": priors["pseudo_target_mask"],
            }

        refined_target_mask = shape_state["pseudo_target_mask"]
        refined_target_parsing = merge_hair_mask_into_parsing_v19(
            base_parsing_mask=shape_state["source_parsing_mask"],
            source_parsing_mask=shape_state["source_parsing_mask"],
            hair_mask=refined_target_mask,
        )
        refined_source_render_256, refined_reference_render_256 = self._decode_source_pose_pair(
            source_image_256=source_image.to(self.opts.device),
            reference_image_256=shape_state["alignment_reference_image_256"],
            source_parsing=shape_state["source_parsing_mask"],
            reference_parsing=shape_state["aligned_reference_parsing_mask"],
            target_parsing=refined_target_parsing,
        )

        encoded = self.latent_encoder(
            [
                shape_state["coarse_source_render_256"][0],
                shape_state["coarse_reference_render_256"][0],
                refined_source_render_256[0],
                refined_reference_render_256[0],
            ]
        )
        source_coarse_f32 = encoded["F32"][0].unsqueeze(0)
        reference_coarse_f32 = encoded["F32"][1].unsqueeze(0)
        source_refined_f32 = encoded["F32"][2].unsqueeze(0)
        reference_refined_f32 = encoded["F32"][3].unsqueeze(0)
        source_coarse_f64 = encoded["F64"][0].unsqueeze(0)
        reference_coarse_f64 = encoded["F64"][1].unsqueeze(0)
        source_refined_f64 = encoded["F64"][2].unsqueeze(0)
        reference_refined_f64 = encoded["F64"][3].unsqueeze(0)

        _, final_priors = self._derive_priors(
            aligned_reference_image_256=refined_reference_render_256,
            source_support=shape_state["source_support"],
            source_hair_mask=source_hair_mask,
            coarse_target_mask=coarse_target_mask,
            source_face_protect=source_face_protect,
        )
        final_priors["pseudo_target_mask"] = torch.maximum(
            final_priors["pseudo_target_mask"],
            refined_target_mask,
        ).clamp(0, 1)
        final_priors["pseudo_boundary_residual"] = (
            final_priors["pseudo_target_mask"] - coarse_target_mask
        ).clamp(-1, 1)

        latent_F_base = self._build_base_feature(
            source_aligned=source_refined_f32,
            reference_aligned=reference_refined_f32,
            source_hair_mask=source_hair_mask,
            target_hair_mask=final_priors["pseudo_target_mask"],
            boundary_band=final_priors["boundary_band"],
            source_face_protect=source_face_protect,
        )
        latent_F_base64 = self._build_base_feature(
            source_aligned=source_refined_f64,
            reference_aligned=reference_refined_f64,
            source_hair_mask=source_hair_mask,
            target_hair_mask=final_priors["pseudo_target_mask"],
            boundary_band=final_priors["boundary_band"],
            source_face_protect=source_face_protect,
            detail_map=final_priors["hair_detail_map"],
        )
        latent_F_align, latent_F64_detail = self._run_shape_adapter(
            F_base=latent_F_base,
            F_src=source_f32,
            F_shape=reference_coarse_f32,
            F_base64=latent_F_base64,
            F_shape64=reference_coarse_f64,
            F_reference=reference_refined_f32,
            F_reference64=reference_refined_f64,
            priors=final_priors,
        )

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name") or ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Align_v19", f"{im_name1}_{im_name2}_source_refined.png", refined_source_render_256)
            save_gen_image(output_dir, "Align_v19", f"{im_name1}_{im_name2}_reference_refined.png", refined_reference_render_256)
            save_vis_mask(output_dir, "Align_v19", f"{im_name1}_{im_name2}_pseudo_target.png", final_priors["pseudo_target_mask"])

        return {
            "latent_F_align": latent_F_align,
            "latent_F_base": latent_F_base,
            "latent_F_shape": reference_coarse_f32,
            "latent_F_reference": reference_refined_f32,
            "latent_F_intermediate": source_refined_f32,
            "latent_F_base64": latent_F_base64,
            "latent_F_shape64": reference_coarse_f64,
            "latent_F_reference64": reference_refined_f64,
            "latent_F_intermediate64": source_refined_f64,
            "latent_F64_detail": latent_F64_detail,
            "aligned_reference_image_256": refined_reference_render_256,
            "target_hair_mask": final_priors["pseudo_target_mask"],
            "boundary_band": final_priors["boundary_band"],
            "priors": final_priors,
            "HM_X": final_priors["pseudo_target_mask"],
        }
