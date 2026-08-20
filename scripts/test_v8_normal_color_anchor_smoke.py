import ast
import io
from pathlib import Path

import torch

from v8_adapter_test_utils import build_adapter, run_adapter, synthetic_inputs


def main():
    training_source = Path(__file__).with_name("blending_train_v8.py").read_text(
        encoding="utf-8-sig"
    )
    training_tree = ast.parse(training_source)
    schedule_node = next(
        node
        for node in training_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_alpha_teacher_weight"
    )
    schedule_namespace = {}
    exec(compile(ast.Module(body=[schedule_node], type_ignores=[]), "<schedule>", "exec"), schedule_namespace)
    schedule = [schedule_namespace["get_alpha_teacher_weight"](epoch) for epoch in range(6)]
    assert schedule == [0.0] * 6

    torch.manual_seed(220)
    model = build_adapter(alpha_init=0.75)
    model.set_anchor_trainable(True)
    model.set_correction_trainable(False)

    latent_face, latent_color, descriptor = synthetic_inputs(batch=3)
    ones = torch.ones(3)
    output, aux = run_adapter(
        model,
        latent_face,
        latent_color,
        descriptor,
        ones,
        ones,
        ones,
        correction_enabled=False,
    )
    loss = output.square().mean()
    assert torch.isfinite(loss)
    loss.backward()

    anchor_parameters = list(model.anchor_parameters())
    correction_parameters = list(model.correction_parameters())
    assert any(parameter.grad is not None for parameter in anchor_parameters)
    assert all(parameter.grad is None for parameter in correction_parameters)
    assert float(aux["correction"].abs().max()) < 1e-7

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    reloaded = build_adapter(alpha_init=0.75)
    reloaded.load_state_dict(torch.load(buffer, map_location="cpu"))
    _, reload_aux = run_adapter(
        reloaded,
        latent_face,
        latent_color,
        descriptor,
        ones,
        ones,
        ones,
        correction_enabled=True,
    )
    reload_correction_max = float(reload_aux["correction"].abs().max())
    assert reload_correction_max < 1e-7
    print(
        "v8 normal-color anchor smoke test passed: "
        f"loss={float(loss):.6f} correction_max={reload_correction_max:.1e} "
        f"teacher_schedule={schedule}"
    )


if __name__ == "__main__":
    main()
