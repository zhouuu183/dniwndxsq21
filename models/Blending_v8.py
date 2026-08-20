import torch
import torch.nn.functional as F

from models.Blending import Blending
from models.Encoders import (
    DIRECT_COLOR_ARCH_V8_4,
    DirectColorBlendAdapterV8,
    PostProcessModel,
    load_direct_color_adapter_state_v8,
)
from models.Net import Net
from models.color_condition_v8 import ColorConditionConfigV8, build_color_condition_bundle
from models.selective_color_projector_v822 import (
    DIRECT_COLOR_ARCH_V8_5,
    SelectiveHairColorProjectorV822,
    fixed_direct_anchor_tail,
)
from models.selective_color_projector_v823 import (
    FULL_COLOR_ARCH_V8_6,
    FullColorToneSelectiveProjectorV823,
    fixed_direct_anchor_tail as fixed_direct_anchor_tail_v823,
)
from models.boundary_masks_v824 import build_boundary_masks_v824
from models.selective_color_projector_v824 import (
    FULL_COLOR_ARCH_V8_7,
    BoundaryStableFullColorProjectorV824,
    fixed_direct_anchor_tail as fixed_direct_anchor_tail_v824,
)
from models.selective_color_projector_v825 import (
    FULL_COLOR_ARCH_V8_8,
    ReferenceConditionedBoundaryProjectorV825,
    fixed_direct_anchor_tail as fixed_direct_anchor_tail_v825,
)
from models.selective_color_projector_v826 import (
    FULL_COLOR_ARCH_V8_9,
    BoundaryTargetAlignedProjectorV826,
    fixed_direct_anchor_tail as fixed_direct_anchor_tail_v826,
    load_v226_projector_checkpoint,
)
from models.strong_anchor_compositor_v828 import (
    StrongAnchorAppearanceCompositorV828,
    apply_pp_hair_lock_v828,
)
from models.v828_runtime_inputs import build_v828_runtime_inputs
from models.boundary_recomposition_v829 import BoundaryRecompositionV829
from models.hair_ownership_v829 import apply_pp_hair_ownership_lock_v829
from models.hybrid_hair_carrier_v829 import HybridHairCarrierV829
from models.v829_runtime_inputs import build_v829_runtime_inputs
from models.achromatic_hair_carrier_v830 import AchromaticHairCarrierV830
from models.hair_topology_repair_v830 import HairTopologyRepairV830
from models.pp_guided_final_v830 import PPGuidedFinalV830
from models.v830_runtime_inputs import build_parser_regions_v830, build_v830_runtime_inputs
from models.hair_matting_v831 import HairMattingV831
from models.pp_unified_final_v831 import PPUnifiedFinalV831
from models.v831_runtime_inputs import build_v831_runtime_inputs
from models.v232_runtime_inputs import build_v832_runtime_inputs
from models.foreground_estimator_v832 import ForegroundEstimatorV832
from models.foreground_recolor_v832 import ForegroundRecolorV832
from models.face_side_alpha_calibrator_v832 import FaceSideAlphaCalibratorV832
from models.background_target_v832 import BackgroundTargetV832
from models.matting_recomposer_v832 import MattingRecomposerV832
from models.fb_confidence_v833 import FBConfidenceV833
from models.reliable_hair_foreground_target_v833 import ReliableHairForegroundTargetV833
from models.face_side_alpha_calibrator_v833 import FaceSideAlphaCalibratorV833
from models.background_target_v833 import BackgroundTargetV833
from models.v833_runtime_inputs import build_v833_runtime_inputs
from models.v833_pipeline import run_v833_pipeline
from models.hair_carrier_chroma_injection_v834 import HairCarrierChromaInjectionV834
from models.v834_runtime_inputs import build_v834_runtime_inputs
from models.hair_only_chroma_disentanglement_v835 import HairOnlyChromaDisentanglementV835
from models.v835_runtime_inputs import build_v835_runtime_inputs
from utils.bicubic import BicubicDownSample
from utils.image_utils import DilateErosion
from utils.mask_delta_v8 import filter_parsing_to_primary_subject
from utils.save_utils import save_gen_image, save_latents


class Blending_v8(Blending):
    """
    v8 keeps the stable blending branch, but makes the implementation
    self-contained so it does not depend on Blending_v4 from the server.
    """

    @staticmethod
    def load_v226_projector_for_diagnostic(checkpoint_path, device):
        """Restore the real inference projector without constructing HairFast."""
        return load_v226_projector_checkpoint(checkpoint_path, device)

    @staticmethod
    @torch.inference_mode()
    def run_v226_projector_debug(projector, **projector_inputs):
        """Use the exact projector call made by real V2.26 pre-PP inference."""
        return projector(return_aux=True, **projector_inputs)

    @staticmethod
    def build_v228_runtime_inputs(**kwargs):
        """Expose the shared V2.28 input contract to deterministic diagnostics."""
        return build_v828_runtime_inputs(**kwargs)

    @staticmethod
    @torch.inference_mode()
    def run_v228_compositor_debug(compositor, **runtime_inputs):
        return compositor(return_aux=True, **runtime_inputs)

    @staticmethod
    def build_v229_runtime_inputs(**kwargs):
        return build_v829_runtime_inputs(**kwargs)

    @staticmethod
    @torch.inference_mode()
    def run_v229_compositor_debug(carrier, recompositor, **runtime_inputs):
        hybrid, carrier_aux = carrier(
            anchor_rgb=runtime_inputs["anchor_rgb"],
            v226_rgb=runtime_inputs["v226_rgb"],
            base_rgb=runtime_inputs["base_rgb"],
            hair_core=runtime_inputs["target_hair_eroded"],
            return_aux=True,
        )
        prepp, aux = recompositor(
            hybrid_core_rgb=hybrid, return_aux=True, **runtime_inputs
        )
        aux["carrier_aux"] = carrier_aux
        return prepp, aux

    @staticmethod
    @torch.inference_mode()
    def run_v230_final_debug(finalizer, **runtime_inputs):
        return finalizer(return_aux=True, **runtime_inputs)

    @staticmethod
    def build_v230_runtime_inputs(**kwargs):
        return build_v830_runtime_inputs(**kwargs)

    @staticmethod
    def build_v231_runtime_inputs(**kwargs):
        return build_v831_runtime_inputs(**kwargs)

    @staticmethod
    def build_v232_runtime_inputs(**kwargs):
        return build_v832_runtime_inputs(**kwargs)

    @staticmethod
    def build_v233_runtime_inputs(**kwargs):
        return build_v833_runtime_inputs(**kwargs)

    @staticmethod
    def build_v235_runtime_inputs(**kwargs):
        return build_v835_runtime_inputs(**kwargs)

    @staticmethod
    @torch.inference_mode()
    def run_v235_disentanglement_debug(disentangler, **runtime_inputs):
        return disentangler(return_aux=True, **runtime_inputs)

    @staticmethod
    def run_v233_pipeline_debug(**kwargs):
        return run_v833_pipeline(**kwargs)

    @staticmethod
    @torch.inference_mode()
    def run_v231_final_debug(finalizer, **runtime_inputs):
        return finalizer(return_aux=True, **runtime_inputs)

    @staticmethod
    def build_v226_real_inference_projector_inputs(
        *, base_rgb, anchor_rgb, pseudo_lab, target_ref_ab, reference_delta_l,
        target_hair_mask, target_hair_eroded, target_hair_dilated,
    ):
        """Reproduce the mask/input algebra in `_blend_v225_selective`."""
        target_hair_mask = target_hair_mask.float().clamp(0, 1)
        target_hair_eroded = target_hair_eroded.float().clamp(0, 1)
        outside = (1.0 - target_hair_mask).clamp(0, 1)
        outer_guard = (
            target_hair_dilated.float().clamp(0, 1) - target_hair_mask
        ).clamp(0, 1) * outside
        return {
            "base_rgb": base_rgb, "anchor_rgb": anchor_rgb,
            "pseudo_lab": pseudo_lab, "target_ref_ab": target_ref_ab,
            "reference_delta_l": reference_delta_l,
            "target_hair_mask": target_hair_mask,
            "target_hair_eroded": target_hair_eroded,
            "outer_background_guard": outer_guard,
            "face_keep_mask": outside, "skin_protect_mask": outside,
            "satd_protect_mask": outside, "remove_mask": torch.zeros_like(outside),
        }

    def __init__(self, opts, net=None):
        torch.nn.Module.__init__(self)
        self.opts = opts
        self.net = Net(self.opts) if net is None else net

        checkpoint = torch.load(self.opts.blending_checkpoint, map_location=self.opts.device)
        checkpoint_arch = checkpoint.get("arch") if isinstance(checkpoint, dict) else None
        if checkpoint_arch not in (
            DIRECT_COLOR_ARCH_V8_4,
            DIRECT_COLOR_ARCH_V8_5,
            FULL_COLOR_ARCH_V8_6,
            FULL_COLOR_ARCH_V8_7,
            FULL_COLOR_ARCH_V8_8,
            FULL_COLOR_ARCH_V8_9,
        ):
            raise RuntimeError(
                f"Refusing incompatible BlendingV8 checkpoint {self.opts.blending_checkpoint}: "
                f"arch={checkpoint_arch!r}, required one of "
                f"{DIRECT_COLOR_ARCH_V8_4!r}, {DIRECT_COLOR_ARCH_V8_5!r}, "
                f"{FULL_COLOR_ARCH_V8_6!r}, {FULL_COLOR_ARCH_V8_7!r}, {FULL_COLOR_ARCH_V8_8!r}, {FULL_COLOR_ARCH_V8_9!r}"
            )
        self.v222 = checkpoint_arch == DIRECT_COLOR_ARCH_V8_5
        self.v223 = checkpoint_arch == FULL_COLOR_ARCH_V8_6
        self.v224 = checkpoint_arch == FULL_COLOR_ARCH_V8_7
        self.v225 = checkpoint_arch == FULL_COLOR_ARCH_V8_8
        self.v226 = checkpoint_arch == FULL_COLOR_ARCH_V8_9
        self.v228 = self.v226 and bool(getattr(self.opts, "v228_enabled", True))
        self.v229 = self.v228 and bool(getattr(self.opts, "v229_enabled", True))
        self.v230 = self.v229 and bool(getattr(self.opts, "v230_enabled", True))
        self.v235 = self.v226 and bool(getattr(self.opts, "v235_enabled", True))
        self.v234 = self.v226 and bool(getattr(self.opts, "v234_enabled", True)) and not self.v235
        # V2.34 is a direct carrier path. It must not initialize or call the
        # V2.31-V2.33 matting/recomposition stack when enabled.
        self.v231 = self.v230 and bool(getattr(self.opts, "v231_enabled", True)) and not self.v234 and not self.v235
        self.v232 = self.v231 and bool(getattr(self.opts, "v232_enabled", True))
        self.v233 = self.v232 and bool(getattr(self.opts, "v233_enabled", True))
        self.selective_runtime = self.v222 or self.v223 or self.v224 or self.v225 or self.v226
        self.base_alpha_v8 = checkpoint.get("alpha_star")
        self.strong_anchor_alpha_v222 = float(
            checkpoint.get("strong_anchor_alpha", 0.90)
        )
        self.blending_encoder = None
        self.selective_projector = None
        if self.v226:
            if checkpoint.get("version") != "v2.26" or checkpoint.get("reference_conditioned_boundary_target") is not True:
                raise RuntimeError("V2.26 checkpoint must enable reference-conditioned boundary target")
            config = checkpoint.get("projector_config", {})
            self.selective_projector = BoundaryTargetAlignedProjectorV826(
                target_dir_min_ab=config.get("target_dir_min_ab", 1.5),
                luma_low_radius=config.get("luma_low_radius", 9),
                max_low_l_shift=config.get("max_low_l_shift", 35.0),
                edge_chroma_strength=config.get("edge_chroma_strength", 0.70),
                edge_luma_strength=config.get("edge_luma_strength", 0.65),
                edge_luma_margin=config.get("edge_luma_margin", 2.5),
                edge_target_l_margin=config.get("edge_target_l_margin", 4.0),
                orth_keep=config.get("orth_keep", 0.10),
                hard_protect_threshold=config.get("hard_protect_threshold", 0.50),
                compensation_gamma=config.get("compensation_gamma", 0.0),
            )
            self.selective_projector.load_state_dict(checkpoint.get("projector_state_dict", {}), strict=True)
            self.selective_projector.to(self.opts.device).eval()
        elif self.v225:
            if checkpoint.get("version") != "v2.25" or checkpoint.get("reference_conditioned_boundary_target") is not True:
                raise RuntimeError("V2.25 checkpoint must enable reference-conditioned boundary target")
            config = checkpoint.get("projector_config", {})
            self.selective_projector = ReferenceConditionedBoundaryProjectorV825(
                target_dir_min_ab=config.get("target_dir_min_ab", 1.5),
                luma_low_radius=config.get("luma_low_radius", 9),
                max_low_l_shift=config.get("max_low_l_shift", 35.0),
                edge_chroma_strength=config.get("edge_chroma_strength", 0.70),
                edge_luma_strength=config.get("edge_luma_strength", 0.65),
                edge_luma_margin=config.get("edge_luma_margin", 2.5),
                edge_target_l_margin=config.get("edge_target_l_margin", 4.0),
                orth_keep=config.get("orth_keep", 0.10),
                hard_protect_threshold=config.get("hard_protect_threshold", 0.50),
            )
            self.selective_projector.load_state_dict(checkpoint.get("projector_state_dict", {}), strict=True)
            self.selective_projector.to(self.opts.device).eval()
        elif self.v224:
            if checkpoint.get("version") != "v2.24":
                raise RuntimeError(
                    "V2.24 architecture requires a checkpoint with version='v2.24'"
                )
            if checkpoint.get("boundary_single_alpha") is not True:
                raise RuntimeError("V2.24 checkpoint must enable boundary_single_alpha")
            if checkpoint.get("post_candidate_soft_blend") is not False:
                raise RuntimeError("V2.24 checkpoint must disable post-candidate soft blending")
            projector_config = checkpoint.get("projector_config", {})
            self.selective_projector = BoundaryStableFullColorProjectorV824(
                target_dir_min_ab=projector_config.get("target_dir_min_ab", 1.5),
                luma_low_radius=projector_config.get("luma_low_radius", 9),
                max_low_l_shift=projector_config.get("max_low_l_shift", 35.0),
                edge_chroma_strength=projector_config.get("edge_chroma_strength", 0.70),
                edge_luma_strength=projector_config.get("edge_luma_strength", 0.65),
                edge_luma_margin=projector_config.get(
                    "edge_luma_margin", getattr(self.opts, "edge_luma_margin_v8", 2.5)
                ),
                orth_keep=projector_config.get("orth_keep", 0.10),
                hard_protect_threshold=projector_config.get("hard_protect_threshold", 0.50),
            )
            self.selective_projector.load_state_dict(
                checkpoint.get("projector_state_dict", {}), strict=True
            )
            self.selective_projector.to(self.opts.device).eval()
            print(
                f"[Blending_v8] loaded arch={FULL_COLOR_ARCH_V8_7} "
                f"strong_anchor_alpha={self.strong_anchor_alpha_v222:.2f} "
                f"config={self.selective_projector.config_dict()}"
            )
        elif self.v223:
            if checkpoint.get("version") != "v2.23":
                raise RuntimeError(
                    "V2.23 architecture requires a checkpoint with version='v2.23'"
                )
            projector_config = checkpoint.get("projector_config", {})
            self.selective_projector = FullColorToneSelectiveProjectorV823(
                target_dir_min_ab=projector_config.get("target_dir_min_ab", 1.5),
                luma_low_radius=projector_config.get("luma_low_radius", 9),
                max_low_l_shift=projector_config.get("max_low_l_shift", 35.0),
                edge_chroma_strength=projector_config.get(
                    "edge_chroma_strength", 0.70
                ),
                edge_luma_strength=projector_config.get("edge_luma_strength", 0.65),
                edge_luma_margin=projector_config.get(
                    "edge_luma_margin", getattr(self.opts, "edge_luma_margin_v8", 2.5)
                ),
                orth_keep=projector_config.get("orth_keep", 0.10),
                hard_protect_threshold=projector_config.get(
                    "hard_protect_threshold", 0.50
                ),
            )
            self.selective_projector.load_state_dict(
                checkpoint.get("projector_state_dict", {}), strict=True
            )
            self.selective_projector.to(self.opts.device).eval()
            print(
                f"[Blending_v8] loaded arch={FULL_COLOR_ARCH_V8_6} "
                f"strong_anchor_alpha={self.strong_anchor_alpha_v222:.2f} "
                f"config={self.selective_projector.config_dict()}"
            )
        elif self.v222:
            projector_config = checkpoint.get("projector_config", {})
            self.selective_projector = SelectiveHairColorProjectorV822(
                target_dir_min_ab=projector_config.get("target_dir_min_ab", 1.5),
                edge_luma_margin=projector_config.get(
                    "edge_luma_margin", getattr(self.opts, "edge_luma_margin_v8", 2.5)
                ),
                halo_scale=projector_config.get("halo_scale", 4.0),
            )
            self.selective_projector.load_state_dict(
                checkpoint["projector_state_dict"], strict=True
            )
            self.selective_projector.to(self.opts.device).eval()
            print(
                f"[Blending_v8] loaded arch={DIRECT_COLOR_ARCH_V8_5} "
                f"strong_anchor_alpha={self.strong_anchor_alpha_v222:.2f} "
                f"projector_params={sum(p.numel() for p in self.selective_projector.parameters())}"
            )
        else:
            adapter_config = checkpoint.get("adapter_config", {})
            self.blending_encoder = DirectColorBlendAdapterV8(
                checkpoint.get("clip", "ViT-B/32"),
                alpha_init=adapter_config.get(
                    "alpha_init", getattr(self.opts, "alpha_init_v8", 0.70)
                ),
                layer_offset_max=adapter_config.get(
                    "layer_offset_max", getattr(self.opts, "layer_offset_max_v8", 0.15)
                ),
                correction_chroma_budget_ratio=adapter_config.get(
                    "correction_chroma_budget_ratio",
                    getattr(self.opts, "correction_chroma_budget_ratio_v8", 0.15),
                ),
                correction_luma_budget_ratio=adapter_config.get(
                    "correction_luma_budget_ratio",
                    getattr(self.opts, "correction_luma_budget_ratio_v8", 0.10),
                ),
                correction_orth_scale=adapter_config.get(
                    "correction_orth_scale", getattr(self.opts, "correction_orth_scale_v8", 0.25)
                ),
            )
            source_state = checkpoint.get("model_state_dict", checkpoint)
            report = load_direct_color_adapter_state_v8(self.blending_encoder, source_state)
            self.blending_encoder.set_anchor_trainable(False)
            self.blending_encoder.set_correction_trainable(False)
            print(
                f"[Blending_v8] loaded arch={DIRECT_COLOR_ARCH_V8_4} "
                f"strict adapter tensors={len(report['loaded'])}"
            )
            self.blending_encoder.to(self.opts.device).eval()

        self.post_process = PostProcessModel().to(self.opts.device).eval()
        postprocess_checkpoint = torch.load(self.opts.pp_checkpoint, map_location=self.opts.device)
        self.post_process.load_state_dict(postprocess_checkpoint["model_state_dict"])
        self.v228_compositor = StrongAnchorAppearanceCompositorV828(
            matte_max_distance=getattr(self.opts, "v228_matte_max_distance", 8),
            bg_residual_radius=getattr(self.opts, "v228_bg_residual_radius", 7),
            bg_residual_strength=getattr(self.opts, "v228_bg_residual_strength", 1.0),
            outer_ring_width=getattr(self.opts, "v228_outer_ring_width", 6),
        ).to(self.opts.device).eval()
        self.v229_carrier = HybridHairCarrierV829(
            low_radius=getattr(self.opts, "v229_carrier_low_radius", 5),
            anchor_hf_gain=getattr(self.opts, "v229_anchor_hf_gain", 1.0),
        ).to(self.opts.device).eval()
        self.v229_recompositor = BoundaryRecompositionV829(
            coverage_max_distance=getattr(self.opts, "v228_matte_max_distance", 8),
            carrier_low_radius=getattr(self.opts, "v229_carrier_low_radius", 5),
            tone_propagation_radius=getattr(self.opts, "v229_tone_radius", 7),
            background_radius=getattr(self.opts, "v229_background_radius", 7),
            outer_strand_recovery=getattr(self.opts, "v229_outer_strand_recovery", False),
        ).to(self.opts.device).eval()
        self.v230_carrier = AchromaticHairCarrierV830(
            low_radius=getattr(self.opts, "v230_core_low_radius", 5),
            detail_radius=getattr(self.opts, "v230_anchor_detail_radius", 5),
            detail_log_cap=getattr(self.opts, "v230_detail_log_cap", 0.25),
            detail_gain=getattr(self.opts, "v230_anchor_detail_gain", 1.0),
        ).to(self.opts.device).eval()
        self.v230_topology = HairTopologyRepairV830(
            hole_radius=getattr(self.opts, "v230_topology_hole_radius", 2),
            neighbor_threshold=getattr(self.opts, "v230_topology_neighbor_threshold", 0.75),
        ).to(self.opts.device).eval()
        self.v230_finalizer = PPGuidedFinalV830(
            core_seam_width=getattr(self.opts, "v230_core_seam_width", 5),
            pp_tone_radius=getattr(self.opts, "v230_pp_tone_radius", 5),
            tone_propagation_radius=getattr(self.opts, "v230_tone_radius", 7),
            face_contact_width=getattr(self.opts, "v230_face_contact_width", 4),
            confidence_temperature=getattr(self.opts, "v230_confidence_temperature", 8.0),
        ).to(self.opts.device).eval()
        self.v231_matting = None
        self.v231_finalizer = None
        self.v232_foreground = None
        self.v232_recolor = None
        self.v232_face_calibrator = None
        self.v232_background = None
        self.v232_recomposer = None
        self.v233_confidence = None
        self.v233_foreground_target = None
        self.v233_face_calibrator = None
        self.v233_background = None
        self.v234_carrier = None
        self.v235_disentangler = None
        if self.v235:
            self.v235_disentangler = HairOnlyChromaDisentanglementV835(
                chroma_radius=getattr(self.opts, "v235_chroma_radius", 9),
                boundary_radius=getattr(self.opts, "v235_boundary_radius", 3),
                boundary_min_confidence=getattr(self.opts, "v235_boundary_min_confidence", 0.20),
                chroma_scale=getattr(self.opts, "v235_chroma_scale", 18.0),
            ).to(self.opts.device).eval()
            print(
                "[Blending_v8] V2.35 hair-only chroma disentanglement enabled; "
                f"config={self.v235_disentangler.config_dict()}"
            )
        if self.v234:
            self.v234_carrier = HairCarrierChromaInjectionV834(
                reference_radius=getattr(self.opts, "v234_reference_radius", 9),
                edge_radius=getattr(self.opts, "v234_edge_radius", 3),
                edge_min_confidence=getattr(self.opts, "v234_edge_min_confidence", 0.35),
            ).to(self.opts.device).eval()
            print(
                "[Blending_v8] V2.34 Strong Anchor hair-carrier chroma injection enabled; "
                f"config={self.v234_carrier.config_dict()}"
            )
        if self.v231:
            self.v231_matting = HairMattingV831(
                getattr(
                    self.opts,
                    "v231_vitmatte_path",
                    "pretrained_models/ViTMatte/vitmatte-small-composition-1k",
                ),
                device=self.opts.device,
                inner_width=getattr(self.opts, "v231_trimap_inner_width", 8),
                outer_width=getattr(self.opts, "v231_trimap_outer_width", 8),
                face_contact_extra_inner=getattr(
                    self.opts, "v231_face_contact_extra_inner", 4
                ),
                max_trimap_hole_area=getattr(self.opts, "v231_max_trimap_hole_area", 16),
            )
            self.v231_finalizer = PPUnifiedFinalV831(
                tone_radius=getattr(self.opts, "v231_tone_radius", 9),
                tone_propagation_radius=getattr(
                    self.opts, "v231_tone_propagation_radius", 15
                ),
                context_radius=getattr(self.opts, "v231_context_radius", 9),
                transition_expand=getattr(self.opts, "v231_transition_expand", 2),
                enable_phase_c=getattr(self.opts, "v231_phase_c", True),
            ).to(self.opts.device).eval()
            if self.v232:
                self.v232_foreground = ForegroundEstimatorV832(
                    roi_padding=getattr(self.opts, "v232_foreground_roi_padding", 64),
                    cache_dir=(getattr(self.opts, "v232_foreground_cache", None) or None),
                )
                self.v232_recolor = ForegroundRecolorV832(
                    tone_radius=getattr(self.opts, "v232_tone_radius", 9),
                    residual_radius=getattr(self.opts, "v232_residual_radius", 21),
                ).to(self.opts.device).eval()
                self.v232_face_calibrator = FaceSideAlphaCalibratorV832(
                    prototype_radius=getattr(self.opts, "v232_face_prototype_radius", 7),
                    temperature=getattr(self.opts, "v232_face_temperature", 0.04),
                ).to(self.opts.device).eval()
                self.v232_background = BackgroundTargetV832(
                    radius=getattr(self.opts, "v232_background_radius", 9),
                    transition_expand=getattr(self.opts, "v232_transition_expand", 2),
                ).to(self.opts.device).eval()
                self.v232_recomposer = MattingRecomposerV832().to(self.opts.device).eval()
                if self.v233:
                    self.v233_confidence = FBConfidenceV833().to(self.opts.device).eval()
                    self.v233_foreground_target = ReliableHairForegroundTargetV833(
                        tone_radius=getattr(self.opts, "v233_tone_radius", 9),
                        local_radius=getattr(self.opts, "v233_local_radius", 21),
                        propagation_radius=getattr(self.opts, "v233_propagation_radius", 15),
                        detail_gain=getattr(self.opts, "v233_detail_gain", 0.5),
                    ).to(self.opts.device).eval()
                    self.v233_face_calibrator = FaceSideAlphaCalibratorV833(
                        posterior_radius=getattr(self.opts, "v233_posterior_radius", 5),
                        temperature=getattr(self.opts, "v233_face_temperature", 0.04),
                    ).to(self.opts.device).eval()
                    self.v233_background = BackgroundTargetV833(
                        radius=getattr(self.opts, "v233_background_radius", 9),
                        support_full=getattr(self.opts, "v233_support_full", 0.20),
                    ).to(self.opts.device).eval()
                    print(
                        "[Blending_v8] V2.33 confidence-limited foreground/alpha-consistent recomposition enabled; "
                        f"confidence={self.v233_confidence.config_dict()} "
                        f"foreground={self.v233_foreground_target.config_dict()} "
                        f"background={self.v233_background.config_dict()}"
                    )
                print(
                    "[Blending_v8] V2.32 foreground-space matting recomposition enabled; "
                    f"foreground={self.v232_foreground.config_dict()} "
                    f"recolor={self.v232_recolor.config_dict()}"
                )
            print(
                "[Blending_v8] V2.31 high-resolution ViTMatte PP unified recolor enabled; "
                f"matting={self.v231_matting.config_dict()} "
                f"finalizer={self.v231_finalizer.config_dict()}"
            )
        elif self.v230:
            print(
                "[Blending_v8] V2.30 PP-guided soft edge recolor enabled; "
                f"carrier={self.v230_carrier.config_dict()} "
                f"finalizer={self.v230_finalizer.config_dict()}"
            )
        elif self.v229:
            print(
                "[Blending_v8] V2.29 hybrid carrier/background recomposition enabled; "
                f"carrier={self.v229_carrier.config_dict()} "
                f"recomposition={self.v229_recompositor.config_dict()}"
            )
        elif self.v228:
            print(
                "[Blending_v8] V2.28 strong-anchor appearance carrier enabled; "
                f"config={self.v228_compositor.config_dict()}"
            )
        self.dilate_erosion = DilateErosion(dilate_erosion=self.opts.smooth, device=self.opts.device)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.color_config = ColorConditionConfigV8(
            ab_no_edit_threshold=getattr(self.opts, "ab_no_edit_threshold_v8", 1.5),
            ab_full_edit_threshold=getattr(self.opts, "ab_full_edit_threshold_v8", 15.0),
            hue_no_edit_deg=getattr(self.opts, "hue_no_edit_deg_v8", 4.0),
            hue_full_edit_deg=getattr(self.opts, "hue_full_edit_deg_v8", 30.0),
            chroma_mag_no_edit=getattr(self.opts, "chroma_mag_no_edit_v8", 2.0),
            chroma_mag_full_edit=getattr(self.opts, "chroma_mag_full_edit_v8", 15.0),
            color_dist_no_edit=getattr(self.opts, "color_dist_no_edit_v8", 2.0),
            color_dist_full_edit=getattr(self.opts, "color_dist_full_edit_v8", 15.0),
            lightness_no_edit_threshold=getattr(self.opts, "lightness_no_edit_threshold_v8", 3.0),
            lightness_full_edit_threshold=getattr(self.opts, "lightness_full_edit_threshold_v8", 15.0),
            max_global_l_shift=getattr(self.opts, "max_global_l_shift_v8", 40.0),
            relative_luma_bins=getattr(self.opts, "relative_luma_bins_v8", 8),
            relative_luma_min_scale=getattr(self.opts, "relative_luma_min_scale_v8", 3.0),
            global_ab_fallback_min_reliability=getattr(
                self.opts, "global_ab_fallback_min_reliability_v8", 0.5
            ),
            min_safe_fraction=getattr(self.opts, "min_safe_reference_fraction_v8", 0.35),
        )

    @torch.inference_mode()
    def _blend_v222_selective(
        self,
        *,
        base_rgb: torch.Tensor,
        latent_face: torch.Tensor,
        latent_color: torch.Tensor,
        latent_f_align: torch.Tensor,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        color_bundle: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        anchor_tail = fixed_direct_anchor_tail(
            latent_face[:, 6:], latent_color[:, 6:], self.strong_anchor_alpha_v222
        )
        anchor_latent = torch.cat((latent_face[:, :6], anchor_tail), dim=1)
        anchor_norm, _ = self.net.generator(
            [anchor_latent],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_align,
        )
        anchor_rgb = self.downsample_256(anchor_norm)
        anchor_rgb01 = ((anchor_rgb + 1.0) / 2.0).clamp(0, 1)
        target_hair_mask = target_hair_mask.float().clamp(0, 1)
        target_hair_eroded = target_hair_eroded.float().clamp(0, 1)
        transition_ring = (target_hair_mask - target_hair_eroded).clamp(0, 1)
        outside_hair = (1.0 - target_hair_mask).clamp(0, 1)
        target_hair_dilated, _ = self.dilate_erosion.mask(target_hair_mask)
        outer_background_guard = (
            (target_hair_dilated - target_hair_mask).clamp(0, 1) * outside_hair
        )
        face_keep = outside_hair
        skin_protect = outside_hair
        satd_protect = outside_hair
        remove_mask = torch.zeros_like(outside_hair)
        selective_rgb01, aux = self.selective_projector(
            base_rgb=((base_rgb + 1.0) / 2.0).clamp(0, 1),
            anchor_rgb=anchor_rgb01,
            pseudo_lab=color_bundle["pseudo_lab"],
            target_hair_mask=target_hair_mask,
            color_supervision_mask=target_hair_eroded,
            transition_ring=transition_ring,
            outer_background_guard=outer_background_guard,
            face_keep_mask=face_keep,
            skin_protect_mask=skin_protect,
            satd_protect_mask=satd_protect,
            remove_mask=remove_mask,
            return_aux=True,
        )
        return anchor_norm, selective_rgb01 * 2.0 - 1.0, aux

    @torch.inference_mode()
    def _blend_v223_selective(
        self,
        *,
        base_rgb: torch.Tensor,
        latent_face: torch.Tensor,
        latent_color: torch.Tensor,
        latent_f_align: torch.Tensor,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        color_bundle: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        anchor_tail = fixed_direct_anchor_tail_v823(
            latent_face[:, 6:], latent_color[:, 6:], self.strong_anchor_alpha_v222
        )
        anchor_latent = torch.cat((latent_face[:, :6], anchor_tail), dim=1)
        anchor_norm, _ = self.net.generator(
            [anchor_latent],
            input_is_latent=True,
            return_latents=False,
            start_layer=4,
            end_layer=8,
            layer_in=latent_f_align,
        )
        anchor_rgb = self.downsample_256(anchor_norm)
        anchor_rgb01 = ((anchor_rgb + 1.0) / 2.0).clamp(0, 1)
        target_hair_mask = target_hair_mask.float().clamp(0, 1)
        target_hair_eroded = target_hair_eroded.float().clamp(0, 1)
        transition_ring = (target_hair_mask - target_hair_eroded).clamp(0, 1)
        outside_hair = (1.0 - target_hair_mask).clamp(0, 1)
        target_hair_dilated, _ = self.dilate_erosion.mask(target_hair_mask)
        outer_background_guard = (
            (target_hair_dilated - target_hair_mask).clamp(0, 1) * outside_hair
        )
        zeros = torch.zeros_like(outside_hair)
        selective_rgb01, aux = self.selective_projector(
            base_rgb=((base_rgb + 1.0) / 2.0).clamp(0, 1),
            anchor_rgb=anchor_rgb01,
            pseudo_lab=color_bundle["pseudo_lab"],
            reference_delta_l=color_bundle["metrics"]["delta_l_global"],
            target_hair_mask=target_hair_mask,
            color_supervision_mask=target_hair_eroded,
            transition_ring=transition_ring,
            outer_background_guard=outer_background_guard,
            face_keep_mask=outside_hair,
            skin_protect_mask=outside_hair,
            satd_protect_mask=outside_hair,
            remove_mask=zeros,
            return_aux=True,
        )
        return anchor_norm, selective_rgb01 * 2.0 - 1.0, aux

    @staticmethod
    def build_v224_boundary_masks(
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        hard_protect: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return build_boundary_masks_v824(
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            hard_protect=hard_protect,
        )

    @torch.inference_mode()
    def _blend_v224_selective(
        self,
        *,
        base_rgb: torch.Tensor,
        latent_face: torch.Tensor,
        latent_color: torch.Tensor,
        latent_f_align: torch.Tensor,
        target_hair_mask: torch.Tensor,
        target_hair_eroded: torch.Tensor,
        color_bundle: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        anchor_tail = fixed_direct_anchor_tail_v824(
            latent_face[:, 6:], latent_color[:, 6:], self.strong_anchor_alpha_v222
        )
        anchor_latent = torch.cat((latent_face[:, :6], anchor_tail), dim=1)
        anchor_norm, _ = self.net.generator(
            [anchor_latent], input_is_latent=True, return_latents=False,
            start_layer=4, end_layer=8, layer_in=latent_f_align,
        )
        anchor_rgb01 = ((self.downsample_256(anchor_norm) + 1.0) / 2.0).clamp(0, 1)
        target_hair_mask = target_hair_mask.float().clamp(0, 1)
        target_hair_eroded = target_hair_eroded.float().clamp(0, 1)
        outside_hair = (1.0 - target_hair_mask).clamp(0, 1)
        target_hair_dilated, _ = self.dilate_erosion.mask(target_hair_mask)
        outer_background_guard = (
            (target_hair_dilated - target_hair_mask).clamp(0, 1) * outside_hair
        )
        zeros = torch.zeros_like(outside_hair)
        selective_rgb01, aux = self.selective_projector(
            base_rgb=((base_rgb + 1.0) / 2.0).clamp(0, 1),
            anchor_rgb=anchor_rgb01,
            pseudo_lab=color_bundle["pseudo_lab"],
            reference_delta_l=color_bundle["metrics"]["delta_l_global"],
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            outer_background_guard=outer_background_guard,
            face_keep_mask=outside_hair,
            skin_protect_mask=outside_hair,
            satd_protect_mask=outside_hair,
            remove_mask=zeros,
            return_aux=True,
        )
        aux["outer_background_guard"] = outer_background_guard
        return anchor_norm, selective_rgb01 * 2.0 - 1.0, aux

    @torch.inference_mode()
    def _blend_v225_selective(
        self, *, base_rgb, latent_face, latent_color, latent_f_align,
        target_hair_mask, target_hair_eroded, color_bundle,
    ):
        anchor_tail = fixed_direct_anchor_tail_v825(
            latent_face[:, 6:], latent_color[:, 6:], self.strong_anchor_alpha_v222
        )
        anchor_latent = torch.cat((latent_face[:, :6], anchor_tail), dim=1)
        anchor_norm, _ = self.net.generator(
            [anchor_latent], input_is_latent=True, return_latents=False,
            start_layer=4, end_layer=8, layer_in=latent_f_align,
        )
        anchor_rgb01 = ((self.downsample_256(anchor_norm) + 1.0) / 2.0).clamp(0, 1)
        target_hair_mask = target_hair_mask.float().clamp(0, 1)
        target_hair_eroded = target_hair_eroded.float().clamp(0, 1)
        outside = (1.0 - target_hair_mask).clamp(0, 1)
        dilated, _ = self.dilate_erosion.mask(target_hair_mask)
        outer_guard = (dilated - target_hair_mask).clamp(0, 1) * outside
        selective, aux = self.selective_projector(
            base_rgb=((base_rgb + 1.0) / 2.0).clamp(0, 1), anchor_rgb=anchor_rgb01,
            pseudo_lab=color_bundle["pseudo_lab"], target_ref_ab=color_bundle["target_ref_ab"],
            reference_delta_l=color_bundle["metrics"]["delta_l_global"],
            target_hair_mask=target_hair_mask, target_hair_eroded=target_hair_eroded,
            outer_background_guard=outer_guard, face_keep_mask=outside,
            skin_protect_mask=outside, satd_protect_mask=outside,
            remove_mask=torch.zeros_like(outside), return_aux=True,
        )
        aux["outer_background_guard"] = outer_guard
        return anchor_norm, selective * 2.0 - 1.0, aux

    _blend_v226_selective = _blend_v225_selective

    @torch.inference_mode()
    def _blend_v235_carrier(
        self, *, base_rgb, latent_face, latent_color, latent_f_align,
        target_hair_mask, target_hair_eroded, color_bundle,
    ):
        """Generate the Strong Anchor and frozen V2.26 reference baseline."""
        anchor_norm, v226_norm, v226_aux = self._blend_v226_selective(
            base_rgb=base_rgb, latent_face=latent_face, latent_color=latent_color,
            latent_f_align=latent_f_align, target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded, color_bundle=color_bundle,
        )
        return anchor_norm, v226_norm, {
            "v226_rgb": ((v226_norm + 1.0) / 2.0).clamp(0, 1),
            "v226_aux": v226_aux,
        }

    @torch.inference_mode()
    def _blend_v234_carrier(
        self, *, base_rgb, latent_face, latent_color, latent_f_align,
        target_hair_mask, target_hair_eroded, color_bundle,
    ):
        """Generate the frozen Strong Anchor and V2.26 diagnostic reference."""
        anchor_norm, v226_norm, v226_aux = self._blend_v226_selective(
            base_rgb=base_rgb, latent_face=latent_face, latent_color=latent_color,
            latent_f_align=latent_f_align, target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded, color_bundle=color_bundle,
        )
        return anchor_norm, v226_norm, {"v226_rgb": ((v226_norm + 1.0) / 2.0).clamp(0, 1), "v226_aux": v226_aux}

    @torch.inference_mode()
    def _blend_v228_appearance(
        self, *, base_rgb, latent_face, latent_color, latent_f_align,
        target_hair_mask, target_hair_eroded, target_hair_dilated,
        face_keep_mask, skin_protect_mask, hard_protect_mask, color_bundle,
        soft_hair_probability=None,
    ):
        # Keep the frozen V2.26 result for A/B diagnostics; it is not the V2.28
        # RGB carrier.
        anchor_norm, v226_norm, v226_aux = self._blend_v226_selective(
            base_rgb=base_rgb,
            latent_face=latent_face,
            latent_color=latent_color,
            latent_f_align=latent_f_align,
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            color_bundle=color_bundle,
        )
        base_rgb01 = ((base_rgb + 1.0) / 2.0).clamp(0, 1)
        anchor_rgb01 = ((self.downsample_256(anchor_norm) + 1.0) / 2.0).clamp(0, 1)
        runtime_inputs = build_v828_runtime_inputs(
            base_rgb=base_rgb01,
            anchor_rgb=anchor_rgb01,
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            target_hair_dilated=target_hair_dilated,
            face_keep_mask=face_keep_mask,
            skin_protect_mask=skin_protect_mask,
            hard_protect_mask=hard_protect_mask,
            soft_hair_probability=soft_hair_probability,
            matte_width=getattr(self.opts, "v228_matte_width", 4),
        )
        prepp_rgb, aux = self.v228_compositor(return_aux=True, **runtime_inputs)
        aux["hair_core"] = aux["sure_fg"]
        aux["hair_edge"] = aux["unknown_band"]
        aux["boundary_membership"] = aux["unknown_band"]
        aux["luma_transfer_weight"] = aux["hair_alpha"]
        aux["chroma_transfer_weight"] = aux["hair_alpha"]
        aux["outer_background_guard"] = aux["background_sample_ring"]
        aux["v226_rgb"] = ((v226_norm + 1.0) / 2.0).clamp(0, 1)
        aux["v226_aux"] = v226_aux
        aux["runtime_inputs"] = runtime_inputs
        return anchor_norm, prepp_rgb * 2.0 - 1.0, aux

    @torch.inference_mode()
    def _blend_v229_appearance(
        self, *, base_rgb, latent_face, latent_color, latent_f_align,
        target_hair_mask, target_hair_eroded, target_hair_dilated,
        source_subject_mask, source_skin_mask, hard_protect_mask, color_bundle,
    ):
        anchor_norm, v228_norm, v228_aux = self._blend_v228_appearance(
            base_rgb=base_rgb,
            latent_face=latent_face,
            latent_color=latent_color,
            latent_f_align=latent_f_align,
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            target_hair_dilated=target_hair_dilated,
            face_keep_mask=source_subject_mask * (1.0 - target_hair_mask),
            skin_protect_mask=source_skin_mask,
            hard_protect_mask=hard_protect_mask,
            color_bundle=color_bundle,
        )
        runtime_inputs = build_v829_runtime_inputs(
            base_rgb=((base_rgb + 1.0) / 2.0).clamp(0, 1),
            anchor_rgb=((self.downsample_256(anchor_norm) + 1.0) / 2.0).clamp(0, 1),
            v226_rgb=v228_aux["v226_rgb"],
            target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded,
            target_hair_dilated=target_hair_dilated,
            source_subject_mask=source_subject_mask,
            source_skin_mask=source_skin_mask,
            hard_protect_mask=hard_protect_mask,
            matte_width=getattr(self.opts, "v228_matte_width", 4),
        )
        prepp, aux = self.run_v229_compositor_debug(
            self.v229_carrier, self.v229_recompositor, **runtime_inputs
        )
        aux["v226_rgb"] = v228_aux["v226_rgb"]
        aux["v228_rgb"] = ((v228_norm + 1.0) / 2.0).clamp(0, 1)
        aux["v228_aux"] = v228_aux
        aux["runtime_inputs"] = runtime_inputs
        aux["hair_core"] = aux["core_owner"]
        aux["hair_edge"] = aux["inner_edge_owner"]
        aux["boundary_membership"] = aux["inner_edge_owner"]
        aux["luma_transfer_weight"] = aux["hair_ownership"]
        aux["chroma_transfer_weight"] = aux["hair_ownership"]
        aux["outer_background_guard"] = (
            aux["face_samples"] + aux["background_samples"]
        ).clamp(0, 1)
        return anchor_norm, prepp * 2.0 - 1.0, aux

    @torch.inference_mode()
    def _blend_v230_appearance(
        self, *, base_rgb, latent_face, latent_color, latent_f_align,
        target_hair_mask, target_hair_eroded, target_hair_dilated,
        source_subject_mask, source_face_mask, source_skin_mask,
        hard_protect_mask, parser_labels, color_bundle,
    ):
        anchor_norm, v228_norm, v228_aux = self._blend_v228_appearance(
            base_rgb=base_rgb, latent_face=latent_face, latent_color=latent_color,
            latent_f_align=latent_f_align, target_hair_mask=target_hair_mask,
            target_hair_eroded=target_hair_eroded, target_hair_dilated=target_hair_dilated,
            face_keep_mask=source_subject_mask * (1.0 - target_hair_mask),
            skin_protect_mask=source_skin_mask, hard_protect_mask=hard_protect_mask,
            color_bundle=color_bundle,
        )
        base01 = ((base_rgb + 1.0) / 2.0).clamp(0, 1)
        anchor01 = ((self.downsample_256(anchor_norm) + 1.0) / 2.0).clamp(0, 1)
        v226 = v228_aux["v226_rgb"]
        topology, topology_aux = self.v230_topology(
            target_hair_mask=target_hair_mask,
            source_face_mask=source_face_mask,
            return_aux=True,
        )
        repaired_dilated, repaired_eroded = self.dilate_erosion.mask(topology)
        runtime_inputs = build_v829_runtime_inputs(
            base_rgb=base01, anchor_rgb=anchor01, v226_rgb=v226,
            target_hair_mask=topology, target_hair_eroded=repaired_eroded,
            target_hair_dilated=repaired_dilated, source_subject_mask=source_subject_mask,
            source_skin_mask=source_skin_mask, hard_protect_mask=hard_protect_mask,
            matte_width=getattr(self.opts, "v228_matte_width", 4),
        )
        core, carrier_aux = self.v230_carrier(
            anchor_rgb=anchor01, v226_rgb=v226, repaired_hair_core=repaired_eroded,
            return_aux=True,
        )
        prepp, recomposition_aux = self.v229_recompositor(
            **runtime_inputs, hybrid_core_rgb=core, return_aux=True
        )
        aux = {
            **recomposition_aux, **topology_aux,
            "v226_rgb": v226, "v228_rgb": ((v228_norm + 1.0) / 2.0).clamp(0, 1),
            "achromatic_core_rgb": core, "carrier_aux": carrier_aux,
            "v230_target_low_core_rgb": carrier_aux["low_v226_core"],
            "runtime_inputs": runtime_inputs, "parser_labels": parser_labels,
            "repaired_hair_mask": topology,
            "target_hair_eroded": repaired_eroded,
            "target_hair_dilated": repaired_dilated,
            "hair_core": recomposition_aux["core_owner"],
            "hair_edge": recomposition_aux["inner_edge_owner"],
            "boundary_membership": recomposition_aux["inner_edge_owner"],
            "outer_background_guard": (
                recomposition_aux["face_samples"] + recomposition_aux["background_samples"]
            ).clamp(0, 1),
        }
        return anchor_norm, prepp * 2.0 - 1.0, aux

    @torch.inference_mode()
    def blend_images(self, align_shape, align_color, name_to_embed, **kwargs):
        del align_color
        I_1 = name_to_embed["face"]["image_norm_256"]
        I_2 = name_to_embed["shape"]["image_norm_256"]
        I_3 = name_to_embed["color"]["image_norm_256"]

        color_mask, _ = filter_parsing_to_primary_subject(name_to_embed["color"]["mask"])
        HM_3 = torch.where(color_mask == 13, torch.ones_like(color_mask), torch.zeros_like(color_mask)).float()
        _, HM_3E = self.dilate_erosion.mask(HM_3)
        hair_color_mask = HM_3E

        latent_S_1 = name_to_embed["face"]["S"]
        latent_S_3 = name_to_embed["color"]["S"]
        latent_F_align = align_shape["latent_F_align"]
        HM_X = align_shape["HM_X"]

        _, HM_XE = self.dilate_erosion.mask(HM_X)
        HM_XD, _ = self.dilate_erosion.mask(HM_X)

        source_parsing = name_to_embed["face"]["mask"].float()
        if source_parsing.dim() == 3:
            source_parsing = source_parsing.unsqueeze(1)
        source_subject = ((source_parsing > 0) & (source_parsing != 13)).float()
        source_subject = F.interpolate(source_subject, size=HM_X.shape[-2:], mode="nearest")
        parser_regions_v230 = build_parser_regions_v830(source_parsing, HM_X)
        # The parser available in the real runtime exposes hard labels only.
        # Use it as a boundary guard, not as an opacity source.
        face_keep_v228 = source_subject * (1.0 - HM_X)
        skin_protect_v228 = source_subject
        hard_protect_v228 = torch.zeros_like(HM_X)

        needs_blend = I_1 is not I_3 or I_1 is not I_2
        if needs_blend:
            I_base, _ = self.net.generator(
                [latent_S_1],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align,
            )
            I_base_256 = self.downsample_256(I_base)
            reference_for_condition = I_3
            if self.v235:
                # V2.35 never feeds face/background pixels into the color
                # condition branch; the neutral fill is not encoded as color.
                color_mask_for_condition = F.interpolate(
                    hair_color_mask.float(), size=I_3.shape[-2:], mode="nearest"
                )
                reference_for_condition = I_3 * color_mask_for_condition
            bundle = build_color_condition_bundle(
                reference_image=reference_for_condition,
                reference_hair_mask=hair_color_mask,
                base_image=I_base_256,
                target_hair_mask=HM_XE,
                config=self.color_config,
            )
            if self.v235:
                I_anchor, I_blend_256, blending_aux = self._blend_v235_carrier(
                    base_rgb=I_base_256, latent_face=latent_S_1, latent_color=latent_S_3,
                    latent_f_align=latent_F_align, target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE, color_bundle=bundle,
                )
                S_blend = None
            elif self.v234:
                I_anchor, I_blend_256, blending_aux = self._blend_v234_carrier(
                    base_rgb=I_base_256, latent_face=latent_S_1, latent_color=latent_S_3,
                    latent_f_align=latent_F_align, target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE, color_bundle=bundle,
                )
                S_blend = None
            elif self.v230:
                I_anchor, I_blend_256, blending_aux = self._blend_v230_appearance(
                    base_rgb=I_base_256, latent_face=latent_S_1, latent_color=latent_S_3,
                    latent_f_align=latent_F_align, target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE, target_hair_dilated=HM_XD,
                    source_subject_mask=source_subject,
                    source_face_mask=parser_regions_v230["source_face_mask"],
                    source_skin_mask=parser_regions_v230["source_skin_mask"],
                    hard_protect_mask=hard_protect_v228, parser_labels=source_parsing,
                    color_bundle=bundle,
                )
                S_blend = None
            elif self.v229:
                I_anchor, I_blend_256, blending_aux = self._blend_v229_appearance(
                    base_rgb=I_base_256, latent_face=latent_S_1, latent_color=latent_S_3,
                    latent_f_align=latent_F_align, target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE, target_hair_dilated=HM_XD,
                    source_subject_mask=source_subject, source_skin_mask=source_subject,
                    hard_protect_mask=hard_protect_v228, color_bundle=bundle,
                )
                S_blend = None
            elif self.v228:
                I_anchor, I_blend_256, blending_aux = self._blend_v228_appearance(
                    base_rgb=I_base_256, latent_face=latent_S_1, latent_color=latent_S_3,
                    latent_f_align=latent_F_align, target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE, target_hair_dilated=HM_XD,
                    face_keep_mask=face_keep_v228, skin_protect_mask=skin_protect_v228,
                    hard_protect_mask=hard_protect_v228, color_bundle=bundle,
                )
                S_blend = None
            elif self.v226:
                I_anchor, I_blend_256, blending_aux = self._blend_v226_selective(
                    base_rgb=I_base_256, latent_face=latent_S_1, latent_color=latent_S_3,
                    latent_f_align=latent_F_align, target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE, color_bundle=bundle,
                )
                S_blend = None
            elif self.v225:
                I_anchor, I_blend_256, blending_aux = self._blend_v225_selective(
                    base_rgb=I_base_256, latent_face=latent_S_1, latent_color=latent_S_3,
                    latent_f_align=latent_F_align, target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE, color_bundle=bundle,
                )
                S_blend = None
            elif self.v224:
                I_anchor, I_blend_256, blending_aux = self._blend_v224_selective(
                    base_rgb=I_base_256,
                    latent_face=latent_S_1,
                    latent_color=latent_S_3,
                    latent_f_align=latent_F_align,
                    target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE,
                    color_bundle=bundle,
                )
                S_blend = None
            elif self.v223:
                I_anchor, I_blend_256, blending_aux = self._blend_v223_selective(
                    base_rgb=I_base_256,
                    latent_face=latent_S_1,
                    latent_color=latent_S_3,
                    latent_f_align=latent_F_align,
                    target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE,
                    color_bundle=bundle,
                )
                S_blend = None
            elif self.v222:
                I_anchor, I_blend_256, blending_aux = self._blend_v222_selective(
                    base_rgb=I_base_256,
                    latent_face=latent_S_1,
                    latent_color=latent_S_3,
                    latent_f_align=latent_F_align,
                    target_hair_mask=HM_X,
                    target_hair_eroded=HM_XE,
                    color_bundle=bundle,
                )
                S_blend = None
            else:
                S_blend_6_18, blending_aux = self.blending_encoder(
                    latent_face=latent_S_1[:, 6:],
                    latent_color=latent_S_3[:, 6:],
                    color_descriptor=bundle["descriptor"],
                    chroma_need_gate=bundle["chroma_need_gate"],
                    lightness_need_gate=bundle["lightness_need_gate"],
                    edit_need_gate=bundle["edit_need_gate"],
                    correction_enabled=False,
                    base_alpha_override=self.base_alpha_v8,
                    return_aux=True,
                )
                S_blend = torch.cat((latent_S_1[:, :6], S_blend_6_18), dim=1)
        else:
            S_blend = latent_S_1
            bundle = None
            blending_aux = None

        if self.selective_runtime and needs_blend:
            I_blend = I_blend_256
        else:
            I_blend, _ = self.net.generator(
                [S_blend],
                input_is_latent=True,
                return_latents=False,
                start_layer=4,
                end_layer=8,
                layer_in=latent_F_align,
            )
            I_blend_256 = self.downsample_256(I_blend)

        S_final = F_final = None
        if (self.v235 or self.v234) and needs_blend:
            # V2.34/V2.35 deliberately bypass F/B recomposition: the generated
            # Strong Anchor remains the spatial/opacity carrier.
            I_final_pp = I_anchor
        else:
            S_final, F_final = self.post_process(I_1, I_blend_256)
            I_final_pp, _ = self.net.generator(
                [S_final],
                input_is_latent=True,
                return_latents=False,
                start_layer=5,
                end_layer=8,
                layer_in=F_final,
            )
        if self.v235 and needs_blend:
            anchor_rgb01 = ((I_anchor + 1.0) / 2.0).clamp(0, 1)
            reference_rgb01 = ((I_3 + 1.0) / 2.0).clamp(0, 1)
            reference_rgb01 = reference_rgb01 * F.interpolate(
                hair_color_mask.float(), size=reference_rgb01.shape[-2:], mode="nearest"
            ) + 0.5 * (1.0 - F.interpolate(
                hair_color_mask.float(), size=reference_rgb01.shape[-2:], mode="nearest"
            ))
            target_hair_hr = F.interpolate(HM_X.float(), size=anchor_rgb01.shape[-2:], mode="nearest")
            face_hr = F.interpolate(parser_regions_v230["source_face_mask"].float(), size=anchor_rgb01.shape[-2:], mode="nearest")
            v235_runtime = build_v835_runtime_inputs(
                strong_anchor_rgb=anchor_rgb01, color_reference_rgb=reference_rgb01,
                target_hair_mask=target_hair_hr, anchor_hair_mask=target_hair_hr,
                face_mask=face_hr,
                reference_hair_mask=F.interpolate(hair_color_mask.float(), size=anchor_rgb01.shape[-2:], mode="nearest"),
            )
            final_rgb01, v235_aux = self.v235_disentangler(return_aux=True, **v235_runtime)
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux.update({"v235_runtime": v235_runtime, "v235_final_rgb": final_rgb01, **v235_aux})
        elif self.v234 and needs_blend:
            anchor_rgb01 = ((I_anchor + 1.0) / 2.0).clamp(0, 1)
            reference_rgb01 = ((I_3 + 1.0) / 2.0).clamp(0, 1)
            target_hair_hr = F.interpolate(HM_X.float(), size=anchor_rgb01.shape[-2:], mode="nearest")
            face_hr = F.interpolate(parser_regions_v230["source_face_mask"].float(), size=anchor_rgb01.shape[-2:], mode="nearest")
            v234_runtime = build_v834_runtime_inputs(
                strong_anchor_rgb=anchor_rgb01, color_reference_rgb=reference_rgb01,
                hair_mask=target_hair_hr, anchor_hair_mask=target_hair_hr, face_mask=face_hr,
                reference_hair_mask=F.interpolate(hair_color_mask.float(), size=anchor_rgb01.shape[-2:], mode="nearest"),
            )
            final_rgb01, v234_aux = self.v234_carrier(return_aux=True, **v234_runtime)
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux.update({"v234_runtime": v234_runtime, "v234_final_rgb": final_rgb01, **v234_aux})
        elif self.v233 and needs_blend:
            pp_rgb01 = ((I_final_pp + 1.0) / 2.0).clamp(0, 1)
            v233_runtime = build_v833_runtime_inputs(
                pp_original_rgb=pp_rgb01,
                base_rgb=((I_base_256 + 1.0) / 2.0).clamp(0, 1),
                v226_rgb=blending_aux["v226_rgb"], target_hair_mask=HM_X,
                parser_labels=source_parsing,
            )
            alpha_hr, matte_aux = self.v231_matting(
                pp_rgb_1024=v233_runtime["pp_original_rgb"],
                target_hair_mask_256=v233_runtime["target_hair_mask"],
                source_face_mask_256=v233_runtime["source_face_mask"],
                source_skin_mask_256=v233_runtime["source_skin_mask"], return_aux=True,
            )
            final_rgb01, final_aux = run_v833_pipeline(
                foreground_estimator=self.v232_foreground,
                confidence_model=self.v233_confidence,
                foreground_targeter=self.v233_foreground_target,
                alpha_calibrator=self.v233_face_calibrator,
                background_targeter=self.v233_background,
                recomposer=self.v232_recomposer,
                runtime=v233_runtime, alpha=alpha_hr, matte_aux=matte_aux,
            )
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux.update({
                "pp_original_rgb": pp_rgb01, "v233_runtime": v233_runtime,
                "alpha_hr": alpha_hr, **matte_aux, **final_aux,
            })
        elif self.v232 and needs_blend:
            pp_rgb01 = ((I_final_pp + 1.0) / 2.0).clamp(0, 1)
            v232_runtime = build_v832_runtime_inputs(
                pp_original_rgb=pp_rgb01,
                base_rgb=((I_base_256 + 1.0) / 2.0).clamp(0, 1),
                v226_rgb=blending_aux["v226_rgb"],
                target_hair_mask=HM_X,
                parser_labels=source_parsing,
            )
            alpha_hr, matte_aux = self.v231_matting(
                pp_rgb_1024=v232_runtime["pp_original_rgb"],
                target_hair_mask_256=v232_runtime["target_hair_mask"],
                source_face_mask_256=v232_runtime["source_face_mask"],
                source_skin_mask_256=v232_runtime["source_skin_mask"],
                return_aux=True,
            )
            foreground_pp, background_pp, foreground_aux = self.v232_foreground(
                image_rgb_1024=v232_runtime["pp_original_rgb"], alpha_hr=alpha_hr,
                return_aux=True,
            )
            foreground_target, recolor_aux = self.v232_recolor(
                foreground_pp_rgb=foreground_pp,
                v226_rgb=v232_runtime["v226_rgb"], alpha_hr=alpha_hr,
                sure_fg=matte_aux["sure_fg"], return_aux=True,
            )
            alpha_eff, alpha_aux = self.v232_face_calibrator(
                alpha_hr=alpha_hr, foreground_pp_rgb=foreground_pp,
                background_pp_rgb=background_pp, base_rgb=v232_runtime["base_rgb"],
                observed_pp_rgb=v232_runtime["pp_original_rgb"],
                source_face_mask=v232_runtime["source_face_mask"],
                sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"],
                unknown=matte_aux["unknown"], return_aux=True,
            )
            background_target, background_aux = self.v232_background(
                background_pp_rgb=background_pp, base_rgb=v232_runtime["base_rgb"],
                alpha_hr=alpha_hr, unknown=matte_aux["unknown"],
                source_face_mask=v232_runtime["source_face_mask"],
                source_subject_mask=v232_runtime["source_subject_mask"],
                return_aux=True,
            )
            final_rgb01, recomposition_aux = self.v232_recomposer(
                alpha_eff=alpha_eff, foreground_target_rgb=foreground_target,
                background_target_rgb=background_target,
                pp_original_rgb=v232_runtime["pp_original_rgb"],
                transition_support=background_aux["transition_support"],
                sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"],
                return_aux=True,
            )
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux.update({
                "pp_original_rgb": pp_rgb01, "v232_runtime": v232_runtime,
                "alpha_hr": alpha_hr, "phase_b_rgb": alpha_hr * foreground_target + (1.0 - alpha_hr) * background_pp,
                "phase_c_rgb": alpha_eff * foreground_target + (1.0 - alpha_eff) * background_pp,
                "phase_d_final_rgb": final_rgb01, "foreground_pp_rgb": foreground_pp,
                "background_pp_rgb": background_pp, "foreground_target_rgb": foreground_target,
                "background_target_rgb": background_target, "alpha_eff": alpha_eff,
                **matte_aux, **foreground_aux, **recolor_aux, **alpha_aux,
                **background_aux, **recomposition_aux,
            })
        elif self.v231 and needs_blend:
            pp_rgb01 = ((I_final_pp + 1.0) / 2.0).clamp(0, 1)
            v231_runtime = build_v831_runtime_inputs(
                pp_original_rgb=pp_rgb01,
                base_rgb=((I_base_256 + 1.0) / 2.0).clamp(0, 1),
                v226_rgb=blending_aux["v226_rgb"],
                target_hair_mask=HM_X,
                parser_labels=source_parsing,
            )
            alpha_hr, matte_aux = self.v231_matting(
                pp_rgb_1024=v231_runtime["pp_original_rgb"],
                target_hair_mask_256=v231_runtime["target_hair_mask"],
                source_face_mask_256=v231_runtime["source_face_mask"],
                source_skin_mask_256=v231_runtime["source_skin_mask"],
                return_aux=True,
            )
            final_rgb01, final_aux = self.v231_finalizer(
                pp_original_rgb=v231_runtime["pp_original_rgb"],
                base_rgb=v231_runtime["base_rgb"],
                v226_rgb=v231_runtime["v226_rgb"],
                alpha_hr=alpha_hr,
                sure_fg=matte_aux["sure_fg"],
                unknown=matte_aux["unknown"],
                source_face_mask=v231_runtime["source_face_mask"],
                source_subject_mask=v231_runtime["source_subject_mask"],
                return_aux=True,
            )
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux.update({
                "pp_original_rgb": pp_rgb01,
                "v231_runtime": v231_runtime,
                **matte_aux,
                **final_aux,
            })
        elif self.v230 and needs_blend:
            pp_rgb01 = ((I_final_pp + 1.0) / 2.0).clamp(0, 1)
            v230_runtime = build_v830_runtime_inputs(
                base_rgb=((I_base_256 + 1.0) / 2.0).clamp(0, 1),
                anchor_rgb=((self.downsample_256(I_anchor) + 1.0) / 2.0).clamp(0, 1),
                v226_rgb=blending_aux["v226_rgb"],
                v229_prepp_rgb=((I_blend_256 + 1.0) / 2.0).clamp(0, 1),
                pp_original_rgb=pp_rgb01,
                target_hair_mask=blending_aux["repaired_hair_mask"],
                target_hair_eroded=blending_aux["target_hair_eroded"],
                target_hair_dilated=blending_aux["target_hair_dilated"],
                parser_labels=source_parsing,
                final_unlock_mask=kwargs.get("final_unlock_mask", kwargs.get("pp_unlock_mask")),
            )
            final_rgb01, final_aux = self.v230_finalizer(
                core_carrier_rgb=blending_aux["achromatic_core_rgb"],
                pp_original_rgb=v230_runtime["pp_original_rgb"],
                base_rgb=v230_runtime["base_rgb"], target_low_rgb=blending_aux["v230_target_low_core_rgb"],
                repaired_hair_mask=v230_runtime["target_hair_mask"],
                source_face_mask=v230_runtime["source_face_mask"],
                final_unlock_mask=v230_runtime["final_unlock_mask"], return_aux=True,
            )
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux.update({"pp_original_rgb": pp_rgb01, "v230_runtime": v230_runtime, **final_aux})
        elif self.v229 and needs_blend:
            prepp_1024 = F.interpolate(I_blend_256, size=I_final_pp.shape[-2:], mode="bicubic", align_corners=False)
            pp_rgb01 = ((I_final_pp + 1.0) / 2.0).clamp(0, 1)
            prepp_rgb01 = ((prepp_1024 + 1.0) / 2.0).clamp(0, 1)
            ownership = F.interpolate(
                blending_aux["hair_ownership"], size=I_final_pp.shape[-2:], mode="nearest"
            )
            final_rgb01, effective_lock = apply_pp_hair_ownership_lock_v829(
                prepp_rgb01, pp_rgb01, ownership, kwargs.get("pp_unlock_mask")
            )
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux["pp_original_rgb"] = pp_rgb01
            blending_aux["pp_effective_hair_lock"] = effective_lock
            blending_aux["final_locked_rgb"] = final_rgb01
        elif self.v228 and needs_blend:
            prepp_1024 = F.interpolate(I_blend_256, size=I_final_pp.shape[-2:], mode="bicubic", align_corners=False)
            pp_rgb01 = ((I_final_pp + 1.0) / 2.0).clamp(0, 1)
            prepp_rgb01 = ((prepp_1024 + 1.0) / 2.0).clamp(0, 1)
            alpha_lock = F.interpolate(
                blending_aux["hair_alpha"], size=I_final_pp.shape[-2:], mode="bilinear", align_corners=False
            ).clamp(0, 1)
            final_rgb01, effective_lock = apply_pp_hair_lock_v828(
                prepp_rgb01, pp_rgb01, alpha_lock, kwargs.get("pp_unlock_mask")
            )
            I_final = final_rgb01 * 2.0 - 1.0
            blending_aux["pp_original_rgb"] = pp_rgb01
            blending_aux["pp_effective_hair_lock"] = effective_lock
            blending_aux["final_locked_rgb"] = final_rgb01
        else:
            I_final = I_final_pp

        if self.opts.save_all:
            exp_name = kwargs.get("exp_name")
            exp_name = exp_name if exp_name is not None else ""
            output_dir = self.opts.save_all_dir / exp_name
            save_gen_image(output_dir, "Blending_v8", "blending.png", I_blend)
            if self.v224 or self.v225 or self.v226:
                save_gen_image(output_dir, "Blending_v8", "final.png", I_final)
            if S_blend is not None:
                save_latents(output_dir, "Blending_v8", "blending.npz", S_blend=S_blend)
            if bundle is not None:
                save_gen_image(output_dir, "Blending_v8", "base.png", I_base_256)
                if self.selective_runtime:
                    save_gen_image(output_dir, "Blending_v8", "strong_anchor.png", I_anchor)
                    selective_name = "selective_color_v226.png" if self.v226 else "selective_color_v225.png" if self.v225 else "selective_color_v224.png" if self.v224 else "selective_color.png"
                    save_gen_image(output_dir, "Blending_v8", selective_name, I_blend_256)
                    if (self.v223 or self.v224 or self.v225 or self.v226) and blending_aux is not None and not (self.v234 or self.v235):
                        for mask_name, aux_name in (
                            ("hair_core", "hair_core"),
                            ("boundary_membership" if (self.v224 or self.v225 or self.v226) else "transition_ring", "hair_edge"),
                            ("luma_transfer_weight", "luma_transfer_weight"),
                            ("chroma_transfer_weight", "chroma_transfer_weight"),
                            ("hard_protect", "hard_protect"),
                        ):
                            save_gen_image(
                                output_dir,
                                "Blending_v8",
                                f"{mask_name}.png",
                                blending_aux[aux_name] * 2.0 - 1.0,
                            )
                    if (self.v224 or self.v225 or self.v226) and not (self.v234 or self.v235):
                            save_gen_image(
                                output_dir, "Blending_v8", "outer_background_guard.png",
                                blending_aux["outer_background_guard"] * 2.0 - 1.0,
                            )
                    if self.v228 and not self.v229:
                        save_gen_image(output_dir, "Blending_v8", "v226_diagnostic.png", blending_aux["v226_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v228_hair_alpha.png", blending_aux["hair_alpha"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v228_prepp.png", I_blend_256)
                        save_gen_image(output_dir, "Blending_v8", "pp_original.png", blending_aux["pp_original_rgb"] * 2.0 - 1.0)
                    if self.v235:
                        save_gen_image(output_dir, "Blending_v8", "v235_color_hair_mask.png", blending_aux["color_hair_mask"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v235_color_hair_only.png", blending_aux["color_hair_only_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v235_chroma_feature_visual.png", blending_aux["chroma_feature_visual"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v235_leakage_map.png", blending_aux["leakage_map"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v235_final_hair_mask.png", blending_aux["final_hair_mask"] * 2.0 - 1.0)
                    elif self.v234:
                        save_gen_image(output_dir, "Blending_v8", "v234_chroma_only.png", blending_aux["chroma_only_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v234_edge_aware.png", blending_aux["edge_aware_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v234_confidence.png", blending_aux["confidence_map"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v234_hair_ownership.png", blending_aux["hair_ownership"] * 2.0 - 1.0)
                    elif self.v233:
                        save_gen_image(output_dir, "Blending_v8", "pp_original.png", blending_aux["pp_original_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v233_alpha_hr.png", blending_aux["alpha_hr"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v233_foreground_confidence.png", blending_aux["foreground_confidence"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v233_alpha_eff.png", blending_aux["alpha_eff"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v233_phase_b.png", blending_aux["phase_b_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v233_phase_c.png", blending_aux["phase_c_rgb"] * 2.0 - 1.0)
                    elif self.v231:
                        save_gen_image(output_dir, "Blending_v8", "pp_original.png", blending_aux["pp_original_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v231_alpha_hr.png", blending_aux["alpha_hr"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v231_phase_b.png", blending_aux["phase_b_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v231_phase_c.png", blending_aux["phase_c_rgb"] * 2.0 - 1.0)
                    elif self.v230:
                        save_gen_image(output_dir, "Blending_v8", "v229_diagnostic.png", blending_aux["v228_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v230_achromatic_core.png", blending_aux["achromatic_core_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v230_hair_tone_confidence.png", blending_aux["hair_tone_confidence"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v230_soft_core_weight.png", blending_aux["soft_core_weight"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v230_corrected_pp.png", blending_aux["corrected_pp_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v230_prepp.png", I_blend_256)
                        save_gen_image(output_dir, "Blending_v8", "pp_original.png", blending_aux["pp_original_rgb"] * 2.0 - 1.0)
                    elif self.v229:
                        save_gen_image(output_dir, "Blending_v8", "v228_diagnostic.png", blending_aux["v228_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v229_hybrid_core.png", blending_aux["hybrid_core_rgb"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v229_hair_ownership.png", blending_aux["hair_ownership"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v229_coverage_alpha.png", blending_aux["coverage_alpha"] * 2.0 - 1.0)
                        save_gen_image(output_dir, "Blending_v8", "v229_prepp.png", I_blend_256)
                        save_gen_image(output_dir, "Blending_v8", "pp_original.png", blending_aux["pp_original_rgb"] * 2.0 - 1.0)
                save_gen_image(output_dir, "Blending_v8", "color_proxy.png", bundle["color_proxy"])
                save_gen_image(output_dir, "Blending_v8", "pseudo_color.png", bundle["pseudo_rgb"])
                if not self.v222 and not self.v223 and not self.v224 and not self.v225 and not self.v226:
                    save_latents(
                        output_dir,
                        "Blending_v8",
                        "color_condition.npz",
                        descriptor=bundle["descriptor"],
                        chroma_need_gate=bundle["chroma_need_gate"],
                        lightness_need_gate=bundle["lightness_need_gate"],
                        edit_need_gate=bundle["edit_need_gate"],
                        safe_ref_mask=bundle["safe_ref_mask"],
                        rejected_highlight_mask=bundle["rejected_highlight_mask"],
                        composite_color_distance=bundle["composite_color_distance"],
                        hue_distance_deg=bundle["hue_distance_deg"],
                        chroma_distance=bundle["chroma_distance"],
                        distribution_distance=bundle["distribution_distance"],
                        relative_luma_reliability=bundle["relative_luma_reliability"],
                        pseudo_reference_fidelity=bundle["pseudo_reference_fidelity"],
                        direct_delta_norm=blending_aux["direct_delta_norm"],
                        direct_component_norm=blending_aux["direct_component_norm"],
                        predicted_alpha=blending_aux["predicted_alpha"],
                        layer_offset=blending_aux["layer_offset"],
                        effective_layer_mix=blending_aux["effective_layer_mix"],
                        layer_mix=blending_aux["layer_mix"],
                        correction_norm=blending_aux["correction_norm"],
                        correction_budget=blending_aux["correction_budget"],
                        correction_chroma_budget=blending_aux["correction_chroma_budget"],
                        correction_luma_budget=blending_aux["correction_luma_budget"],
                        direct_parallel_correction_coeff=blending_aux[
                            "direct_parallel_correction_coeff"
                        ],
                        negative_parallel_fraction=blending_aux[
                            "negative_parallel_fraction"
                        ],
                        anchor_frozen=blending_aux["anchor_frozen"],
                        total_delta_norm=blending_aux["total_delta_norm"],
                    )
            save_gen_image(output_dir, "Final_v8", "final.png", I_final)
            if S_final is not None and F_final is not None:
                save_latents(output_dir, "Final_v8", "final.npz", S_final=S_final, F_final=F_final)

        return ((I_final[0] + 1) / 2).clamp(0, 1)
