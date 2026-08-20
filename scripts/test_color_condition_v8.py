import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.color_condition_v8 import COLOR_DESCRIPTOR_DIM, build_color_condition_bundle


def solid_rgb(rgb: tuple[float, float, float], batch: int = 1, size: int = 64) -> torch.Tensor:
    value = torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1)
    return value.expand(batch, -1, size, size).clone() * 2.0 - 1.0


def main():
    batch = 2
    mask = torch.ones(batch, 1, 64, 64)
    base = solid_rgb((0.24, 0.16, 0.10), batch=batch)
    reference = torch.cat(
        [solid_rgb((0.25, 0.17, 0.11)), solid_rgb((0.05, 0.72, 0.18))],
        dim=0,
    )
    bundle = build_color_condition_bundle(reference, mask, base, mask)

    assert bundle["safe_ref_mask"].shape == (batch, 1, 64, 64)
    assert bundle["rejected_highlight_mask"].shape == (batch, 1, 64, 64)
    assert bundle["descriptor"].shape == (batch, COLOR_DESCRIPTOR_DIM)
    assert bundle["pseudo_lab"].shape == (batch, 3, 64, 64)
    assert bundle["pseudo_rgb"].shape == (batch, 3, 64, 64)
    assert bundle["color_proxy"].shape == (batch, 3, 64, 64)
    for key in ("descriptor", "pseudo_lab", "pseudo_rgb", "color_proxy"):
        assert torch.isfinite(bundle[key]).all(), key
    for gate_name in ("chroma_need_gate", "lightness_need_gate", "edit_need_gate"):
        assert ((bundle[gate_name] >= 0) & (bundle[gate_name] <= 1)).all(), gate_name
    assert bundle["chroma_need_gate"][0] < 0.1
    assert bundle["chroma_need_gate"][1] > 0.8
    assert bundle["edit_need_gate"][1] > 0.8
    assert (bundle["pseudo_rgb"][0] - base[0]).abs().mean() > 0.0

    normal_green = solid_rgb((0.18, 0.62, 0.22))
    normal_green_bundle = build_color_condition_bundle(normal_green, mask[:1], base[:1], mask[:1])
    highlighted = normal_green.clone()
    highlighted[:, :, 8:24, 20:44] = 0.99 * 2.0 - 1.0
    highlight_bundle = build_color_condition_bundle(highlighted, mask[:1], base[:1], mask[:1])
    assert highlight_bundle["rejected_highlight_mask"][:, :, 8:24, 20:44].mean() > 0.05
    assert highlight_bundle["metrics"]["safe_fraction"].item() >= 0.35 - 1e-4
    assert torch.linalg.vector_norm(
        normal_green_bundle["metrics"]["ref_mean_ab"]
        - highlight_bundle["metrics"]["ref_mean_ab"]
    ).item() < 3.0
    assert abs(
        normal_green_bundle["chroma_need_gate"].item()
        - highlight_bundle["chroma_need_gate"].item()
    ) < 0.1

    light_hair_safe_fractions = []
    for hair_rgb in ((0.62, 0.61, 0.59), (0.88, 0.87, 0.84), (0.86, 0.75, 0.48)):
        light_hair_bundle = build_color_condition_bundle(
            solid_rgb(hair_rgb), mask[:1], base[:1], mask[:1]
        )
        safe_fraction = light_hair_bundle["metrics"]["safe_fraction"].item()
        light_hair_safe_fractions.append(safe_fraction)
        assert safe_fraction > 0.9

    print(
        "color_condition_v8 smoke test passed: "
        f"chroma_gates={bundle['chroma_need_gate'].tolist()} "
        f"edit_gates={bundle['edit_need_gate'].tolist()} "
        f"highlight_safe={highlight_bundle['metrics']['safe_fraction'].item():.4f} "
        f"light_hair_safe={light_hair_safe_fractions}"
    )


if __name__ == "__main__":
    main()
