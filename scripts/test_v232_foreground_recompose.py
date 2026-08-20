"""Synthetic contract tests for the V2.32 foreground-space pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.background_target_v832 import BackgroundTargetV832
from models.face_side_alpha_calibrator_v832 import FaceSideAlphaCalibratorV832
from models.foreground_estimator_v832 import ForegroundEstimatorV832
from models.foreground_recolor_v832 import ForegroundRecolorV832
from models.matting_recomposer_v832 import MattingRecomposerV832


def fake_estimator(image, alpha, **kwargs):
    foreground = image.copy()
    background = image.copy()
    return foreground, background


def main() -> None:
    torch.manual_seed(3)
    h = w = 32
    alpha = torch.zeros(1, 1, h, w)
    alpha[..., 8:24, 8:24] = 0.2
    alpha[..., 12:20, 12:20] = 1.0
    image = torch.full((1, 3, h, w), 0.25)
    image[:, 0, 8:24, 8:24] = 0.45
    image[:, 1, 8:24, 8:24] = 0.30
    image[:, 2, 8:24, 8:24] = 0.20
    estimator = ForegroundEstimatorV832(roi_padding=4, estimator=fake_estimator)
    foreground, background, aux = estimator(image_rgb_1024=image, alpha_hr=alpha, return_aux=True)
    recon = alpha * foreground + (1 - alpha) * background
    assert torch.allclose(recon, image, atol=1e-6)
    assert float(aux["reconstruction_error"].max()) <= 1e-6

    sure_fg = (alpha >= 0.99).float()
    recolor = ForegroundRecolorV832(tone_radius=3, residual_radius=5)
    v226 = torch.full_like(image, 0.8)
    target, recolor_aux = recolor(
        foreground_pp_rgb=foreground, v226_rgb=v226, alpha_hr=alpha,
        sure_fg=sure_fg, return_aux=True,
    )
    # Low-alpha pixels receive the same foreground target; alpha is not a color gain.
    edge_delta = (target - foreground)[..., 8:12, 8:12].abs().mean()
    core_delta = (target - foreground)[..., 12:20, 12:20].abs().mean()
    assert float(edge_delta) > 0.01 and float(core_delta) > 0.01
    assert torch.isfinite(target).all()
    assert float(recolor_aux["local_support"].max()) >= 0.0

    face = torch.zeros_like(alpha)
    face[..., :8, :] = 1.0
    unknown = ((alpha > 0.01) & (alpha < 0.99)).float()
    sure_bg = (alpha <= 0.01).float()
    calibrator = FaceSideAlphaCalibratorV832(prototype_radius=3)
    base = image.clone()
    alpha_eff, alpha_aux = calibrator(
        alpha_hr=alpha, foreground_pp_rgb=foreground, background_pp_rgb=background,
        base_rgb=base, source_face_mask=face, sure_fg=sure_fg, sure_bg=sure_bg,
        unknown=unknown, return_aux=True,
    )
    assert torch.allclose(alpha_eff[sure_fg > 0.5], torch.ones_like(alpha_eff[sure_fg > 0.5]))
    assert torch.allclose(alpha_eff[sure_bg > 0.5], torch.zeros_like(alpha_eff[sure_bg > 0.5]))
    assert torch.isfinite(alpha_aux["alpha_eff"]).all()

    background_targeter = BackgroundTargetV832(radius=3, transition_expand=1)
    subject = face.clone()
    background_target, background_aux = background_targeter(
        background_pp_rgb=background, base_rgb=base, alpha_hr=alpha,
        unknown=unknown, source_face_mask=face, source_subject_mask=subject,
        return_aux=True,
    )
    recomposer = MattingRecomposerV832()
    final, final_aux = recomposer(
        alpha_eff=alpha_eff, foreground_target_rgb=target,
        background_target_rgb=background_target, pp_original_rgb=image,
        transition_support=background_aux["transition_support"],
        sure_fg=sure_fg, sure_bg=sure_bg, return_aux=True,
    )
    assert torch.isfinite(final).all()
    far = (1.0 - background_aux["transition_support"]) * (1.0 - sure_fg) * (1.0 - sure_bg)
    assert float(((final - image).abs() * far).max()) <= 1e-6
    assert float(final_aux["recomposition_equation_error"].max()) <= 1e-6
    print("V232 synthetic foreground/recomposition contract: PASS")


if __name__ == "__main__":
    main()
