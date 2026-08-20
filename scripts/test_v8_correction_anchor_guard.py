import torch

from v8_adapter_test_utils import build_adapter, run_adapter, synthetic_inputs


def main():
    torch.manual_seed(24)
    model = build_adapter().eval()
    latent_face, latent_color, descriptor = synthetic_inputs(batch=2)
    ones = torch.ones(2)
    direct_delta = latent_color - latent_face

    handle = model.correction_mlp.register_forward_hook(
        lambda _module, _inputs, _output: -direct_delta
    )
    try:
        output, aux = run_adapter(
            model,
            latent_face,
            latent_color,
            descriptor,
            ones,
            torch.zeros(2),
            ones,
            correction_enabled=True,
            layer_mix_override=1.0,
        )
    finally:
        handle.remove()

    expected = latent_face + direct_delta
    cancellation_error = float((output - expected).abs().max())
    assert torch.all(aux["direct_parallel_correction_coeff"] < 0)
    assert torch.all(aux["negative_parallel_fraction"] == 1)
    assert float(aux["correction_norm"].max()) == 0.0
    assert cancellation_error < 1e-6
    print(
        "v8 correction anchor guard test passed: "
        f"cancellation_error={cancellation_error:.1e} "
        f"parallel_coeff_max={float(aux['direct_parallel_correction_coeff'].max()):.3f}"
    )


if __name__ == "__main__":
    main()
