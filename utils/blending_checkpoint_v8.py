"""Checkpoint contract for the v8 reference-dominant blending pipeline.

This module is intentionally dependency-free.  Training and inference can
therefore share the same policy schema without importing model modules (which
would create an import cycle through :mod:`models.Blending_v8`).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any


COLOR_POLICY_SCHEMA_VERSION_V8 = 2
COLOR_PIPELINE_V8 = "generator_raw_then_reference_dominant_final"
COLOR_PIPELINE_DISABLED_V8 = "generator_raw_without_reference_dominant_final"
COLOR_INPUT_RANGE_V8 = "normalized_minus1_1"
TARGET_HAIR_MASK_V8 = "HM_X_repaired"
REFERENCE_HAIR_MASK_V8 = "celeba19_parser_eroded_with_raw_fallback"


_MISSING = object()


def _config_value(config: object, overrides: Mapping[str, Any], name: str, default: Any) -> Any:
    if name in overrides:
        return overrides[name]
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def build_blending_inference_policy_v8(
    config: object,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the inference-critical portion of a v8 ``color_policy``.

    ``config`` may be an argparse namespace or a mapping.  Per-call keyword
    arguments belong in ``overrides`` because both Alignment_v8 and
    Blending_v8 allow those arguments to override parser defaults.
    """

    runtime = {} if overrides is None else overrides
    transfer_disabled = bool(
        _config_value(
            config,
            runtime,
            "disable_reference_dominant_hair_color_v8",
            False,
        )
    )
    sigma = _config_value(
        config,
        runtime,
        "hair_color_low_frequency_sigma_v8",
        None,
    )

    return {
        "schema_version": COLOR_POLICY_SCHEMA_VERSION_V8,
        "pipeline": (
            COLOR_PIPELINE_DISABLED_V8 if transfer_disabled else COLOR_PIPELINE_V8
        ),
        "input_range": COLOR_INPUT_RANGE_V8,
        "target_hair_mask": TARGET_HAIR_MASK_V8,
        "reference_hair_mask": REFERENCE_HAIR_MASK_V8,
        "target_hair_policy": {
            "close_kernel": int(
                _config_value(config, runtime, "target_hair_close_kernel", 9)
            ),
            "hole_max_area": (
                None
                if _config_value(
                    config,
                    runtime,
                    "target_hair_hole_max_area",
                    None,
                )
                is None
                else float(
                    _config_value(
                        config,
                        runtime,
                        "target_hair_hole_max_area",
                        None,
                    )
                )
            ),
            "hole_max_area_ratio": float(
                _config_value(
                    config,
                    runtime,
                    "target_hair_hole_max_area_ratio",
                    0.003,
                )
            ),
            "hole_min_prior_coverage": float(
                _config_value(
                    config,
                    runtime,
                    "target_hair_hole_min_prior_coverage",
                    0.10,
                )
            ),
            "hole_prior_evidence_radius": int(
                _config_value(
                    config,
                    runtime,
                    "target_hair_hole_prior_evidence_radius",
                    2,
                )
            ),
            "top_fill_only": bool(
                _config_value(config, runtime, "target_hair_top_fill_only", True)
            ),
            "ear_bridge_radius": int(
                _config_value(
                    config,
                    runtime,
                    "target_hair_ear_bridge_radius",
                    3,
                )
            ),
            "crown_repair_enabled": bool(
                _config_value(
                    config,
                    runtime,
                    "target_hair_crown_repair_enabled",
                    True,
                )
            ),
            "crown_height_ratio": float(
                _config_value(
                    config,
                    runtime,
                    "target_hair_crown_height_ratio",
                    0.50,
                )
            ),
            "crown_bridge_radius": int(
                _config_value(
                    config,
                    runtime,
                    "target_hair_crown_bridge_radius",
                    8,
                )
            ),
            "crown_prior_dilate": int(
                _config_value(
                    config,
                    runtime,
                    "target_hair_crown_prior_dilate",
                    1,
                )
            ),
            "crown_component_distance": int(
                _config_value(
                    config,
                    runtime,
                    "target_hair_crown_component_distance",
                    12,
                )
            ),
            "crown_component_min_prior_overlap": float(
                _config_value(
                    config,
                    runtime,
                    "target_hair_crown_component_min_prior_overlap",
                    0.15,
                )
            ),
            "crown_max_added_area_ratio": float(
                _config_value(
                    config,
                    runtime,
                    "target_hair_crown_max_added_area_ratio",
                    0.008,
                )
            ),
            "parser_guards_exclude_repaired_hair": True,
        },
        "transfer": {
            "reference_strength": float(
                _config_value(
                    config,
                    runtime,
                    "hair_color_reference_strength_v8",
                    0.9,
                )
            ),
            "low_frequency_radius": int(
                _config_value(
                    config,
                    runtime,
                    "hair_color_low_frequency_radius_v8",
                    15,
                )
            ),
            "low_frequency_sigma": None if sigma is None else float(sigma),
            "feather_radius": int(
                _config_value(
                    config,
                    runtime,
                    "hair_color_feather_radius_v8",
                    5,
                )
            ),
            "spatial_reference_weight": float(
                _config_value(
                    config,
                    runtime,
                    "hair_color_spatial_reference_weight_v8",
                    0.75,
                )
            ),
            "detail_chroma_gain": float(
                _config_value(
                    config,
                    runtime,
                    "hair_color_detail_chroma_gain_v8",
                    1.0,
                )
            ),
            "luma_reference_strength": float(
                _config_value(
                    config,
                    runtime,
                    "hair_color_luma_reference_strength_v8",
                    0.30,
                )
            ),
            "luma_mean_limit": float(
                _config_value(
                    config,
                    runtime,
                    "hair_color_luma_mean_limit_v8",
                    6.0,
                )
            ),
            "luma_std_ratio_limit": float(
                _config_value(
                    config,
                    runtime,
                    "hair_color_luma_std_ratio_limit_v8",
                    1.25,
                )
            ),
        },
    }


def _values_match(expected: Any, found: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(found, bool) and found is expected
    if isinstance(expected, float):
        return (
            isinstance(found, (int, float))
            and not isinstance(found, bool)
            and math.isclose(expected, float(found), rel_tol=1e-9, abs_tol=1e-12)
        )
    return type(found) is type(expected) and found == expected


def _policy_mismatches(
    expected: Mapping[str, Any],
    found: Mapping[str, Any],
    *,
    prefix: str = "",
) -> list[tuple[str, Any, Any]]:
    mismatches: list[tuple[str, Any, Any]] = []
    for key, expected_value in expected.items():
        path = f"{prefix}.{key}" if prefix else key
        found_value = found.get(key, _MISSING)
        if isinstance(expected_value, Mapping):
            if not isinstance(found_value, Mapping):
                mismatches.append((path, expected_value, found_value))
            else:
                mismatches.extend(
                    _policy_mismatches(expected_value, found_value, prefix=path)
                )
        elif found_value is _MISSING or not _values_match(expected_value, found_value):
            mismatches.append((path, expected_value, found_value))
    return mismatches


def validate_blending_checkpoint_policy_v8(
    checkpoint: object,
    runtime_config: object,
    *,
    overrides: Mapping[str, Any] | None = None,
    checkpoint_path: object | None = None,
) -> dict[str, Any] | None:
    """Validate a loaded blending checkpoint against the active v8 runtime.

    A legacy override permits only a checkpoint with no ``color_policy`` at
    all.  It never suppresses a mismatch on a policy-bearing checkpoint: that
    would make a newly trained but incompatible model look like an old model.
    """

    location = str(checkpoint_path) if checkpoint_path is not None else "<loaded checkpoint>"
    allow_legacy = bool(
        _config_value(
            runtime_config,
            {} if overrides is None else overrides,
            "allow_legacy_blending_checkpoint_v8",
            False,
        )
    )
    policy = checkpoint.get("color_policy") if isinstance(checkpoint, Mapping) else None
    if policy is None:
        if allow_legacy:
            warnings.warn(
                "Loading a legacy blending checkpoint without color_policy because "
                "allow_legacy_blending_checkpoint_v8=True. Its learned colour behavior "
                f"is not guaranteed to match the v8 pipeline: {location}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
        raise RuntimeError(
            "The v8/v5 blending checkpoint has no color_policy and cannot be used "
            "safely with the current reference-dominant colour pipeline: "
            f"{location}. Retrain with scripts/blending_train_v8.py. For dataset/cache "
            "construction or deliberate legacy inference only, set "
            "allow_legacy_blending_checkpoint_v8=True explicitly."
        )
    if not isinstance(policy, Mapping):
        raise RuntimeError(
            f"Invalid color_policy in blending checkpoint {location}: expected a mapping, "
            f"found {type(policy).__name__}."
        )

    expected = build_blending_inference_policy_v8(
        runtime_config,
        overrides=overrides,
    )
    mismatches = _policy_mismatches(expected, policy)
    if mismatches:
        details = []
        for path, expected_value, found_value in mismatches:
            rendered_found = "<missing>" if found_value is _MISSING else repr(found_value)
            details.append(
                f"  - {path}: runtime={expected_value!r}, checkpoint={rendered_found}"
            )
        raise RuntimeError(
            "The blending checkpoint color_policy does not match the active v8/v5 "
            f"runtime: {location}. Do not mix weights and colour/mask code from "
            "different runs.\n" + "\n".join(details)
        )
    return dict(policy)
