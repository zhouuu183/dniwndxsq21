"""Shared V2.33 inference pipeline used by runtime and diagnostics."""

from __future__ import annotations

import torch


@torch.inference_mode()
def run_v833_pipeline(*, foreground_estimator, confidence_model, foreground_targeter,
                      alpha_calibrator, background_targeter, recomposer,
                      runtime: dict[str, torch.Tensor], alpha: torch.Tensor,
                      matte_aux: dict[str, torch.Tensor], decomposition=None):
    if decomposition is None:
        foreground, background, fb_aux = foreground_estimator(
            image_rgb_1024=runtime["pp_original_rgb"], alpha_hr=alpha, return_aux=True,
        )
    else:
        foreground, background, fb_aux = decomposition
    foreground_conf, background_conf, transition_conf, confidence_aux = confidence_model(
        alpha=alpha, sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"],
        unknown=matte_aux["unknown"], reconstruction_error=fb_aux["reconstruction_error"],
        return_aux=True,
    )
    foreground_target, foreground_aux = foreground_targeter(
        foreground_pp_rgb=foreground, original_pp_rgb=runtime["pp_original_rgb"],
        v226_rgb=runtime["v226_rgb"], foreground_confidence=foreground_conf,
        return_aux=True,
    )
    phase_b = alpha * foreground_target + (1.0 - alpha) * background
    alpha_eff, alpha_aux = alpha_calibrator(
        alpha=alpha, observed_pp_rgb=runtime["pp_original_rgb"],
        hair_target_prior_rgb=foreground_aux["propagated_target_foreground_rgb"],
        base_rgb=runtime["base_rgb"], foreground_confidence=foreground_conf,
        source_face_mask=runtime["source_face_mask"],
        coarse_target_hair=runtime["target_hair_mask"], sure_fg=matte_aux["sure_fg"],
        sure_bg=matte_aux["sure_bg"], unknown=matte_aux["unknown"], return_aux=True,
    )
    phase_c = alpha_eff * foreground_target + (1.0 - alpha_eff) * background
    background_target, background_aux = background_targeter(
        background_pp_rgb=background, base_rgb=runtime["base_rgb"], alpha_eff=alpha_eff,
        background_confidence=background_conf, sure_fg=matte_aux["sure_fg"],
        sure_bg=matte_aux["sure_bg"], unknown=matte_aux["unknown"],
        source_face_mask=runtime["source_face_mask"],
        source_subject_mask=runtime["source_subject_mask"], return_aux=True,
    )
    final, recomposition_aux = recomposer(
        alpha_eff=alpha_eff, foreground_target_rgb=foreground_target,
        background_target_rgb=background_target, pp_original_rgb=runtime["pp_original_rgb"],
        transition_support=background_aux["recomposition_support"],
        sure_fg=matte_aux["sure_fg"], sure_bg=matte_aux["sure_bg"], return_aux=True,
    )
    aux = {
        "foreground_pp_rgb": foreground, "background_pp_rgb": background,
        "foreground_target_rgb": foreground_target, "background_target_rgb": background_target,
        "foreground_confidence": foreground_conf, "background_confidence": background_conf,
        "transition_confidence": transition_conf, "alpha_eff": alpha_eff,
        "phase_b_rgb": phase_b, "phase_c_rgb": phase_c, "phase_d_final_rgb": final,
        **fb_aux, **confidence_aux, **foreground_aux, **alpha_aux,
        **background_aux, **recomposition_aux,
    }
    return final, aux


__all__ = ["run_v833_pipeline"]
