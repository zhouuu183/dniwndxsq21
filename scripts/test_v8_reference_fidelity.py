from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.color_condition_v8 import (
    ColorConditionConfigV8,
    compute_composite_color_distance,
    compute_composite_color_gate,
    compute_soft_l_shift,
    relative_luma_conditional_ab,
)


def stats(mean_ab, median_chroma, std_ab=(2.0, 2.0), q25=4.0, q75=8.0):
    mean_ab = torch.tensor([[*mean_ab]], dtype=torch.float32)
    median_chroma = torch.tensor([median_chroma], dtype=torch.float32)
    return {
        "mean_ab": mean_ab,
        "median_ab": mean_ab.clone(),
        "mean_chroma": median_chroma.clone(),
        "median_chroma": median_chroma,
        "std_ab": torch.tensor([[*std_ab]], dtype=torch.float32),
        "hue_unit": mean_ab / torch.linalg.vector_norm(mean_ab, dim=1, keepdim=True).clamp_min(1e-6),
        "hue_validity": torch.ones(1),
        "l_median": torch.tensor([50.0]),
        "l_q25": torch.tensor([40.0]),
        "l_q75": torch.tensor([60.0]),
        "c_q25": torch.tensor([q25]),
        "c_q50": median_chroma,
        "c_q75": torch.tensor([q75]),
    }


def main():
    config = ColorConditionConfigV8()
    base = stats((10.0, 0.0), 10.0)
    # Similar mean AB distance, but a different hue/chroma distribution.
    reference = stats((10.0, 2.0), 22.0, std_ab=(10.0, 1.0), q25=1.0, q75=20.0)
    distance = compute_composite_color_distance(reference, base)
    gate = compute_composite_color_gate(distance, config)["chroma_need_gate"]
    assert float(gate.item()) > 0.0

    same = stats((10.0, 0.0), 10.0)
    same_gate = compute_composite_color_gate(
        compute_composite_color_distance(same, base), config
    )["chroma_need_gate"]
    assert float(same_gate.item()) < 1e-5

    ref_luma = torch.linspace(35.0, 65.0, 32).reshape(1, 1, 1, 32).expand(1, 1, 32, 32)
    target_luma = ref_luma - 20.0
    ref_ab = torch.stack(
        [torch.full((32, 32), 30.0), torch.full((32, 32), -10.0)], dim=0
    ).unsqueeze(0)
    mask = torch.ones(1, 1, 32, 32)
    base_relative = dict(base)
    base_relative["l_median"] = torch.tensor([30.0])
    base_relative["l_q25"] = torch.tensor([20.0])
    base_relative["l_q75"] = torch.tensor([40.0])
    reference_relative = stats((30.0, -10.0), 31.62)
    target_ab, reliability, _ = relative_luma_conditional_ab(
        target_luma,
        ref_luma,
        ref_ab,
        mask,
        mask,
        reference_relative,
        base_relative,
        config,
    )
    assert float(reliability.item()) > 0.0
    assert torch.allclose(target_ab.mean(dim=(-2, -1)), ref_ab.mean(dim=(-2, -1)), atol=1e-2)

    shift = compute_soft_l_shift(torch.tensor([35.0]), 40.0)
    assert float(shift.item()) > 20.0
    assert float(shift.item()) < 40.0
    print("v8 reference fidelity tests passed")


if __name__ == "__main__":
    main()
