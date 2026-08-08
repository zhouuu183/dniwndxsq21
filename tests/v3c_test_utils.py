import torch

from models.ear_modules_v5 import EAR_POSITIVE, REQUIRED_DATASET_ITEM_KEYS


def square(reference, y0=20, y1=44, x0=20, x1=44, value=1.0):
    mask = torch.zeros_like(reference)
    mask[:, :, y0:y1, x0:x1] = value
    return mask


def renderer_data(size=64):
    reference = torch.zeros(1, 1, size, size)
    trusted = square(reference)
    zeros = torch.zeros_like(reference)
    return {
        "visible_trusted_core_1024": trusted,
        "visible_candidate_core_1024": zeros.clone(),
        "visible_thin_contour_1024": zeros.clone(),
        "visible_solid_interior_1024": trusted.clone(),
        "visible_candidate_support_1024": zeros.clone(),
        "visible_hollow_interior_1024": zeros.clone(),
        "visible_ambiguous_interior_1024": zeros.clone(),
        "effective_hard_negative_mask_1024": zeros.clone(),
        "raw_hard_negative_mask_1024": zeros.clone(),
        "accepted_object_guard_1024": trusted.clone(),
        "target_hair_alpha_1024": zeros.clone(),
        "target_hair_core_1024": zeros.clone(),
        "component_quality_map_1024": trusted.clone(),
        "component_restore_confidence_map_1024": trusted.clone(),
        "component_missingness_map_1024": trusted.clone(),
        "source_raw_difference_1024": trusted.clone(),
        "color_supervision_mask_1024": trusted.clone(),
        "adaptive_low_delta_scale": trusted * 0.25,
        "adaptive_detail_delta_scale": trusted * 0.30,
        "adaptive_chroma_delta_scale": trusted * 0.12,
        "trusted_support_1024": trusted.clone(),
        "candidate_support_1024": zeros.clone(),
        "shoulder_negative_1024": zeros.clone(),
        "neck_negative_1024": zeros.clone(),
        "skin_edge_negative_1024": zeros.clone(),
        "topology_skeleton_1024": trusted.clone(),
        "occluded_candidate_support_1024": zeros.clone(),
        "earring_status": torch.tensor([EAR_POSITIVE]),
        "earring_presence_score": torch.tensor([1.0]),
        "component_missingness": torch.tensor([1.0]),
        "component_restore_confidence": torch.tensor([1.0]),
        "component_source_chroma_confidence": torch.tensor([1.0]),
        "hard_negative_deletion_ratio": torch.tensor([0.0]),
        "left_bbox": torch.tensor([[0, 0, size // 2, size]]),
        "right_bbox": torch.tensor([[size // 2, 0, size, size]]),
    }


def proposal_data(size=64):
    reference = torch.zeros(1, 1, size, size)
    zeros = torch.zeros_like(reference)
    return {
        key: zeros.clone() for key in (
            "proposal_probability_1024", "appearance_evidence_1024", "curve_evidence_1024",
            "shape_evidence_1024", "material_evidence_1024", "parser_seed_1024",
            "parser_prior_1024", "earlobe_anchor_1024", "hair_highlight_negative_1024",
            "skin_edge_negative_1024", "shoulder_negative_1024", "neck_negative_1024",
            "material_strong_1024", "material_weak_1024", "shape_strong_1024",
            "shape_weak_1024", "small_object_candidate_1024", "local_scale_candidate_1024",
            "strong_candidate_1024", "weak_candidate_1024", "combined_weak_candidate_1024",
        )
    }


def roi_data(size=64):
    reference = torch.ones(1, 1, size, size)
    left = reference.clone()
    left[:, :, :, size // 2:] = 0
    right = reference - left
    return {
        "left_ear_roi_1024": left,
        "right_ear_roi_1024": right,
        "ear_local_roi_1024": reference,
    }


def dataset_item(size=16):
    mask = torch.zeros(1, size, size)
    item = {}
    path_keys = {"source_path", "shape_reference_path", "color_reference_path", "target_path", "raw_pp_path"}
    bool_keys = {
        "has_positive", "has_candidate", "has_hard_negative", "is_true_negative",
        "is_high_quality_thin", "is_solid_hollow",
    }
    for key in REQUIRED_DATASET_ITEM_KEYS:
        if key in ("source_chroma_1024", "raw_pp_chroma_1024"):
            item[key] = torch.zeros(2, size, size)
        elif key.endswith("_1024") or key.startswith("adaptive_"):
            item[key] = mask.clone()
        elif key in path_keys:
            item[key] = key + ".png"
        elif key in ("source_256", "target_256"):
            item[key] = torch.zeros(3, 16, 16)
        elif key == "hm_x":
            item[key] = mask.clone()
        elif key in ("left_bbox", "right_bbox"):
            item[key] = torch.tensor([0, 0, size, size])
        elif key == "fragments_per_group":
            item[key] = []
        elif key == "component_records_v3c":
            item[key] = []
        elif key in ("morphology_category", "occlusion_category", "sample_bucket"):
            item[key] = "true_negative" if key == "sample_bucket" else "negative"
        elif key in bool_keys:
            item[key] = key == "is_true_negative"
        else:
            item[key] = torch.tensor(0.0)
    item["earring_status"] = torch.tensor(0)
    return item
