"""Synthetic V2.33 tests A-K from the implementation specification."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.background_target_v833 import BackgroundTargetV833
from models.face_side_alpha_calibrator_v833 import FaceSideAlphaCalibratorV833
from models.fb_confidence_v833 import FBConfidenceV833
from models.reliable_hair_foreground_target_v833 import ReliableHairForegroundTargetV833
from models.matting_recomposer_v832 import MattingRecomposerV832
from models.v833_pipeline import run_v833_pipeline
from models.SG_IDCT_v16 import rgb_to_lab
from models.color_condition_v8 import compute_intrinsic_hair_color_stats
from utils.v233_metrics import classify_v233, split_v233_audits, v233_metric_tensors


def main() -> None:
    torch.manual_seed(33)
    h = w = 32
    alpha = torch.full((1, 1, h, w), 0.01)
    alpha[..., 10:22, 10:22] = 0.9
    sure_fg = torch.zeros_like(alpha); sure_fg[..., 12:20, 12:20] = 1
    sure_bg = torch.zeros_like(alpha); unknown = 1.0 - sure_fg
    recon_error = torch.zeros(1, 3, h, w)
    confidence = FBConfidenceV833()
    fg_conf, bg_conf, _, _ = confidence(
        alpha=alpha, sure_fg=sure_fg, sure_bg=sure_bg, unknown=unknown,
        reconstruction_error=recon_error, return_aux=True,
    )
    assert float(fg_conf[..., :8, :8].max()) < 0.01
    assert float(fg_conf[..., 12:20, 12:20].min()) == 1.0

    wrong = torch.rand(1, 3, h, w)
    wrong[..., 10:22, 10:22] = torch.tensor([.30, .16, .08])[:, None, None]
    target_color = torch.tensor([.85, .25, .55])[None, :, None, None]
    v226 = target_color.expand_as(wrong).clone()
    original = torch.full_like(wrong, .25)
    targeter = ReliableHairForegroundTargetV833(
        tone_radius=3, local_radius=5, propagation_radius=7, detail_radius=2
    )
    final_f, aux = targeter(
        foreground_pp_rgb=wrong, original_pp_rgb=original, v226_rgb=v226,
        foreground_confidence=fg_conf, return_aux=True,
    )
    # A: low-alpha random F is replaced by propagated target; B: high-alpha remains recolored F.
    low = final_f[..., 8:10, 10:22].mean((2, 3))
    unsupported_low = final_f[..., :2, :2].mean((2, 3))
    assert torch.linalg.vector_norm(low - target_color.flatten(2).mean(2)) < 0.35
    assert torch.linalg.vector_norm(unsupported_low - target_color.flatten(2).mean(2)) < 0.35
    assert torch.linalg.vector_norm(final_f[..., 14:18, 14:18].mean((2, 3)) - target_color.flatten(2).mean(2)) < 0.35
    # C: confidence blend is continuous and finite.
    assert torch.isfinite(final_f).all() and float(aux["foreground_confidence_blend"].min()) >= 0

    face = torch.ones_like(alpha); coarse = torch.ones_like(alpha)
    observed = torch.full((1, 3, h, w), .5); base = observed.clone()
    hair_prior = torch.zeros_like(observed); hair_prior[:, 0] = 1.0
    calibrator = FaceSideAlphaCalibratorV833(posterior_radius=3, temperature=.04)
    alpha_eff, cal_aux = calibrator(
        alpha=alpha, observed_pp_rgb=observed, hair_target_prior_rgb=hair_prior,
        base_rgb=base, foreground_confidence=torch.zeros_like(alpha),
        source_face_mask=face, coarse_target_hair=coarse, sure_fg=sure_fg,
        sure_bg=sure_bg, unknown=unknown, return_aux=True,
    )
    # D/F: invalid evidence never defaults to one and face-like observations reduce alpha.
    invalid = (cal_aux["posterior_valid"] < .5) & (cal_aux["face_contact_actual"] > .5)
    assert float(cal_aux["hair_posterior"][invalid].max()) < 1.0
    assert float(alpha_eff[..., 8:10, 10:22].mean()) < float(alpha[..., 8:10, 10:22].mean())
    # E: true bang/sure-FG is immutable.
    assert torch.all(alpha_eff[sure_fg > .5] == 1)

    backgrounder = BackgroundTargetV833(radius=3, support_full=.2)
    bg = torch.zeros_like(observed)
    subject = torch.zeros_like(alpha)
    out1, b1 = backgrounder(
        background_pp_rgb=bg, base_rgb=base, alpha_eff=torch.full_like(alpha, .1),
        background_confidence=torch.ones_like(alpha), sure_fg=torch.zeros_like(alpha),
        sure_bg=torch.zeros_like(alpha), unknown=torch.ones_like(alpha),
        source_face_mask=face, source_subject_mask=subject, return_aux=True,
    )
    _, b5 = backgrounder(
        background_pp_rgb=bg, base_rgb=base, alpha_eff=torch.full_like(alpha, .5),
        background_confidence=torch.ones_like(alpha), sure_fg=torch.zeros_like(alpha),
        sure_bg=torch.zeros_like(alpha), unknown=torch.ones_like(alpha),
        source_face_mask=face, source_subject_mask=subject, return_aux=True,
    )
    # G: transition is based on alpha_eff; H: support is continuous; I: same-pixel Base wins.
    assert torch.allclose(b1["transition_strength"], torch.full_like(alpha, .36), atol=1e-5)
    assert torch.allclose(b5["transition_strength"], torch.ones_like(alpha), atol=1e-5)
    assert torch.all((b1["support_confidence"] >= 0) & (b1["support_confidence"] <= 1))
    assert torch.allclose(b1["background_target_rgb"], out1)
    assert torch.allclose(b1["same_pixel_context_valid"], torch.ones_like(alpha))
    # J: propagated low-alpha foreground does not retain the random original patch.
    assert torch.linalg.vector_norm(low - target_color.flatten(2).mean(2)) < torch.linalg.vector_norm(wrong[..., 8:10, 10:22].mean((2, 3)) - target_color.flatten(2).mean(2))

    # All requested metric families execute and produce distinct audit payloads.
    ref_stats = compute_intrinsic_hair_color_stats(rgb_to_lab(v226), torch.ones_like(alpha))
    metric_values = v233_metric_tensors(
        base_rgb=base, v226_rgb=v226, pp_rgb=observed, output_rgb=out1,
        foreground_pp_rgb=wrong, foreground_target_rgb=final_f,
        alpha=alpha, alpha_eff=alpha_eff, foreground_confidence=fg_conf,
        background_confidence=bg_conf, sure_fg=sure_fg, sure_bg=sure_bg,
        face_unknown_all=cal_aux["face_unknown_all"],
        face_contact_actual=cal_aux["face_contact_actual"],
        false_positive_proxy=cal_aux["false_positive_proxy"],
        source_face_mask=face, source_subject_mask=subject,
        target_lab=rgb_to_lab(v226), ref_stats=ref_stats,
    )
    assert all(torch.isfinite(value).all() for value in metric_values.values())
    synthetic_summary = {"count": 1, **{f"median_{key}": float(value[0]) for key, value in metric_values.items()}}
    audits = split_v233_audits(synthetic_summary)
    assert len(audits) == 14
    assert len({frozenset(payload) for payload in audits.values()}) == len(audits)

    decomposition = (wrong, observed, {"reconstruction_error": torch.zeros_like(observed)})
    pipeline_final, pipeline_aux = run_v833_pipeline(
        foreground_estimator=None, confidence_model=confidence,
        foreground_targeter=targeter, alpha_calibrator=calibrator,
        background_targeter=backgrounder, recomposer=MattingRecomposerV832(),
        runtime={"pp_original_rgb": observed, "v226_rgb": v226, "base_rgb": base,
                 "source_face_mask": face, "source_subject_mask": subject,
                 "target_hair_mask": coarse},
        alpha=alpha, matte_aux={"sure_fg": sure_fg, "sure_bg": sure_bg, "unknown": unknown},
        decomposition=decomposition,
    )
    assert torch.isfinite(pipeline_final).all()
    assert pipeline_aux["foreground_target_rgb"].shape == observed.shape

    # K: failed gates are accumulated, rather than short-circuiting at the first phase.
    baseline = {
        "median_edge_foreground_progress": 1.0, "median_base_color_rim_fraction": 1.0,
        "median_original_color_patch_fraction": 1.0, "median_false_positive_alpha": 1.0,
        "median_face_contact_rgb": 1.0, "median_background_retention": 1.0,
        "median_core_ref_full": 1.0, "median_direct_ref_ab": 1.0, "median_contour_hf_ab": 1.0,
    }
    summary = dict(baseline)
    summary.update({"median_foreground_conf_high_alpha": 1.0, "median_edge_foreground_progress": 0.0,
                    "median_base_color_rim_fraction": 1.0, "median_face_contact_rgb": 1.0,
                    "median_background_retention": 1.0, "max_sure_fg_error": 0.0})
    decision = classify_v233(summary, baseline, parity={k: 0.0 for k in ("alpha", "alpha_eff", "foreground_target", "background_target", "final")})
    for expected in ("V233_FOREGROUND_TARGET_FAIL", "V233_BASE_RIM_FAIL", "V233_FACE_STRIP_FAIL", "V233_BACKGROUND_TARGET_FAIL"):
        assert expected in decision["failed_gates"]
    print("V2.33 synthetic confidence/ownership tests A-K: PASS")


if __name__ == "__main__":
    main()
