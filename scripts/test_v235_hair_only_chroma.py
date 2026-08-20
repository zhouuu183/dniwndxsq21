"""Synthetic V2.35 hair-only chroma disentanglement contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.SG_IDCT_v16 import rgb_to_lab
from models.hair_only_chroma_disentanglement_v835 import HairOnlyChromaDisentanglementV835
from models.v835_runtime_inputs import build_v835_runtime_inputs
from utils.v235_metrics import (
    face_preservation_loss, hair_chroma_consistency_loss,
    leakage_penalty, texture_preservation_loss, v235_metric_tensors,
    v235_training_objective,
)


def main() -> None:
    torch.manual_seed(235)
    h = w = 48
    anchor = torch.full((1, 3, h, w), 0.28)
    anchor[:, 0, 10:38, 8:40] = 0.72
    anchor[:, 1, 10:38, 8:40] = 0.20
    reference = torch.full_like(anchor, 0.12)
    reference[:, 0, 10:38, 8:40] = 0.34
    reference[:, 1, 10:38, 8:40] = 0.58
    target_hair = torch.zeros(1, 1, h, w); target_hair[..., 10:38, 8:40] = 1
    reference_hair = target_hair.clone()
    face = torch.zeros_like(target_hair); face[..., 10:16, 8:40] = 1
    runtime = build_v835_runtime_inputs(
        strong_anchor_rgb=anchor, color_reference_rgb=reference,
        target_hair_mask=target_hair, anchor_hair_mask=target_hair,
        reference_hair_mask=reference_hair, face_mask=face,
    )
    model = HairOnlyChromaDisentanglementV835(chroma_radius=5, boundary_radius=2)
    final, aux = model(return_aux=True, **runtime)
    assert torch.isfinite(final).all()
    assert torch.allclose(aux["hair_ownership"] * face, torch.zeros_like(face), atol=1e-7)
    assert torch.allclose(aux["confidence_map"] * (1.0 - target_hair), torch.zeros_like(target_hair), atol=1e-7)
    assert float(aux["confidence_map"][..., 20:28, 20:28].min()) > 0.90
    assert float(aux["confidence_map"][..., 10, 35].max()) <= 0.25
    # L is owned by the Strong Anchor and the face/background are unchanged.
    assert (rgb_to_lab(final)[:, :1] - rgb_to_lab(anchor)[:, :1]).abs().max() < 2e-4
    assert float((final[..., 10:16, 8:40] - anchor[..., 10:16, 8:40]).abs().max()) < 1e-6
    assert float((final * (1.0 - target_hair) - anchor * (1.0 - target_hair)).abs().max()) < 1e-6
    assert aux["color_hair_only_rgb"][..., :8, :].mean() == 0.5
    metrics = v235_metric_tensors(
        strong_anchor_rgb=anchor, color_reference_rgb=reference, final_rgb=final,
        target_hair_mask=target_hair, face_mask=face, base_rgb=anchor,
    )
    assert all(torch.isfinite(value).all() for value in metrics.values())
    losses = (
        hair_chroma_consistency_loss(final, reference, target_hair),
        face_preservation_loss(final, anchor, face),
        leakage_penalty(final, anchor, target_hair),
        texture_preservation_loss(final, anchor, target_hair),
    )
    assert all(torch.isfinite(value) for value in losses)
    objective = v235_training_objective(
        final_rgb=final, reference_rgb=reference, target_rgb=anchor,
        hair_mask=target_hair, face_mask=face,
    )
    assert set(objective) == {"chroma", "face", "leakage", "texture", "total"}
    assert all(torch.isfinite(value) for value in objective.values())
    assert float(metrics["face_rgb_change_max"].max()) < 1e-6
    assert float(metrics["non_hair_leakage_max"].max()) < 1e-6
    print("V2.35 synthetic hair-only chroma tests: PASS")


if __name__ == "__main__":
    main()
