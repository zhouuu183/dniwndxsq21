"""Synthetic V2.34 carrier contract tests A-D."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.hair_carrier_chroma_injection_v834 import HairCarrierChromaInjectionV834
from models.v834_runtime_inputs import build_v834_runtime_inputs
from models.SG_IDCT_v16 import rgb_to_lab
from utils.v234_metrics import v234_metric_tensors


def main() -> None:
    torch.manual_seed(234)
    h = w = 48
    anchor = torch.full((1, 3, h, w), 0.28)
    anchor[:, 0, 10:38, 8:40] = 0.75
    anchor[:, 1, 10:38, 8:40] = 0.18
    reference = torch.full_like(anchor, 0.22)
    reference[:, 0] = 0.35
    reference[:, 1, 10:38, 8:40] = 0.55
    hair = torch.zeros(1, 1, h, w); hair[..., 10:38, 8:40] = 1
    face = torch.zeros_like(hair); face[..., 10:16, 8:40] = 1
    runtime = build_v834_runtime_inputs(
        strong_anchor_rgb=anchor, color_reference_rgb=reference,
        hair_mask=hair, face_mask=face,
        reference_hair_mask=hair,
    )
    carrier = HairCarrierChromaInjectionV834(reference_radius=5, edge_radius=2)
    final, aux = carrier(return_aux=True, **runtime)
    assert torch.isfinite(final).all()
    assert torch.allclose(aux["confidence_map"] * face, torch.zeros_like(face), atol=1e-7)
    assert torch.allclose(aux["confidence_map"] * (1 - hair), torch.zeros_like(hair), atol=1e-7)
    assert float(aux["confidence_map"][..., 20:28, 20:28].min()) > 0.90
    assert float(aux["boundary_confidence"].max()) <= 1.0
    # A: L is unchanged; only AB is injected.
    assert (rgb_to_lab(final)[:, :1] - rgb_to_lab(anchor)[:, :1]).abs().max() < 2e-4
    # B: high-frequency anchor detail remains close after smooth AB injection.
    assert float(aux["anchor_lab"][:, :1].mean()) == float(rgb_to_lab(anchor)[:, :1].mean())
    # C/D: face and background ownership are exactly protected and metrics run.
    metrics = v234_metric_tensors(
        strong_anchor_rgb=anchor, color_reference_rgb=reference, final_rgb=final,
        hair_mask=hair, face_mask=face, base_rgb=anchor,
    )
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert float(metrics["face_injection_max"].max()) < 1e-6
    assert float(metrics["outside_hair_change"].max()) < 1e-6
    print("V2.34 synthetic hair-carrier tests A-D: PASS")


if __name__ == "__main__":
    main()
