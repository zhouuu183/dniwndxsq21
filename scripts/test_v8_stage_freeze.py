import torch

from v8_adapter_test_utils import build_adapter, run_adapter, synthetic_inputs


def parameters(module):
    return list(module.parameters())


def main():
    torch.manual_seed(21)
    model = build_adapter()

    model.set_anchor_trainable(True)
    model.set_correction_trainable(False)
    assert all(parameter.requires_grad for parameter in parameters(model.descriptor_encoder))
    assert all(parameter.requires_grad for parameter in parameters(model.strength_head))
    assert all(parameter.requires_grad for parameter in parameters(model.layer_offset_head))
    assert not any(parameter.requires_grad for parameter in parameters(model.layer_embedding))
    assert not any(parameter.requires_grad for parameter in parameters(model.correction_mlp))

    model.set_anchor_trainable(False)
    model.set_correction_trainable(True)
    assert not any(parameter.requires_grad for parameter in parameters(model.descriptor_encoder))
    assert not any(parameter.requires_grad for parameter in parameters(model.strength_head))
    assert not any(parameter.requires_grad for parameter in parameters(model.layer_offset_head))
    assert all(parameter.requires_grad for parameter in parameters(model.layer_embedding))
    assert all(parameter.requires_grad for parameter in parameters(model.correction_mlp))

    anchor_before = {
        name: value.detach().clone()
        for name, value in {
            **{f"strength_head.{name}": value for name, value in model.strength_head.state_dict().items()},
            **{f"layer_offset_head.{name}": value for name, value in model.layer_offset_head.state_dict().items()},
        }.items()
    }
    correction_before = model.correction_mlp[-1].weight.detach().clone()
    optimizer = torch.optim.Adam(list(model.correction_parameters()), lr=1e-3)
    latent_face, latent_color, descriptor = synthetic_inputs(batch=2)
    ones = torch.ones(2)
    output, _ = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        ones,
        ones,
        ones,
        correction_enabled=True,
    )
    output.square().mean().backward()
    optimizer.step()

    anchor_change = max(
        float((model.state_dict()[name] - value).abs().max())
        for name, value in anchor_before.items()
    )
    correction_change = float(
        (model.correction_mlp[-1].weight.detach() - correction_before).abs().max()
    )
    assert anchor_change == 0.0
    assert correction_change > 0.0
    print(
        "v8 stage freeze test passed: "
        f"anchor_max_abs_change={anchor_change:.1e} "
        f"correction_max_abs_change={correction_change:.3e}"
    )


if __name__ == "__main__":
    main()
