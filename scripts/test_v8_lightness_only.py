import torch

from v8_adapter_test_utils import build_adapter, run_adapter, synthetic_inputs


def main():
    torch.manual_seed(23)
    model = build_adapter().eval()
    latent_face, latent_color, descriptor = synthetic_inputs(batch=3)
    zeros = torch.zeros(3)
    ones = torch.ones(3)
    output, aux = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        zeros,
        ones,
        ones,
        correction_enabled=False,
    )
    direct_norm = float(aux["direct_component_norm"].max())
    luma_budget = float(aux["correction_luma_budget"].min())
    assert torch.equal(output, latent_face)
    assert direct_norm == 0.0
    assert luma_budget > 0.0
    print(
        "v8 lightness-only test passed: "
        f"direct_component_norm={direct_norm:.1e} minimum_luma_budget={luma_budget:.6f}"
    )


if __name__ == "__main__":
    main()
