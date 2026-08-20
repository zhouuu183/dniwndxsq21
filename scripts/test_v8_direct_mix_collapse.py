import torch

from v8_adapter_test_utils import build_adapter, run_adapter, synthetic_inputs


def main():
    torch.manual_seed(22)
    model = build_adapter(alpha_init=0.70, layer_offset_max=0.15)
    model.set_anchor_trainable(True)
    model.set_correction_trainable(False)
    optimizer = torch.optim.Adam(list(model.anchor_parameters()), lr=5e-2)
    latent_face, latent_color, descriptor = synthetic_inputs(batch=4)
    ones = torch.ones(4)
    zeros = torch.zeros(4)

    initial_output, initial_aux = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        ones,
        zeros,
        ones,
        correction_enabled=False,
    )
    del initial_output
    assert torch.allclose(initial_aux["layer_mix"], torch.full_like(initial_aux["layer_mix"], 0.70))

    with torch.no_grad():
        # Move off the intentional constant initializer before teacher supervision.
        model.strength_head[-1].weight.normal_(mean=0.0, std=0.25)

    target_alpha = torch.tensor([0.05, 0.25, 0.85, 0.98])
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        _, aux = run_adapter(
            model,
            latent_face,
            latent_color,
            descriptor,
            ones,
            zeros,
            ones,
            correction_enabled=False,
        )
        torch.nn.functional.smooth_l1_loss(aux["predicted_alpha"], target_alpha).backward()
        optimizer.step()

    _, final_aux = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        ones,
        zeros,
        ones,
        correction_enabled=False,
    )
    predicted = final_aux["predicted_alpha"]
    assert float(predicted.std()) > 1e-3
    assert float(final_aux["layer_offset"].abs().max()) <= 0.15 + 1e-6
    print(f"v8 adaptive strength test passed: alpha_std={float(predicted.std()):.6f}")


if __name__ == "__main__":
    main()
