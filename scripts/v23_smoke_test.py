import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ColorTransfer_v23 import V23ColorTransfer


def main() -> None:
    model = V23ColorTransfer(work_size=64, learnable=True)
    satd = torch.rand(2, 3, 128, 128)
    color = torch.rand(2, 3, 64, 64)

    target_hair = torch.zeros(2, 1, 64, 64)
    target_hair[:, :, 16:48, 16:48] = 1
    reference_hair = torch.zeros(2, 1, 64, 64)
    reference_hair[:, :, 10:54, 10:54] = 1
    hard_lock = torch.zeros(2, 1, 64, 64)
    soft_lock = torch.zeros(2, 1, 64, 64)

    output, aux = model(
        satd,
        color,
        target_hair,
        reference_hair,
        hard_lock,
        soft_lock,
        return_aux=True,
    )
    loss = output.mean()
    loss.backward()

    print(f"output_shape={tuple(output.shape)}")
    print(f"alpha_shape={tuple(aux['alpha_full'].shape)}")
    print(f"learnable_params={len(list(model.parameters()))}")
    print(f"ab_strength_grad={float(model.ab_strength.grad):.8f}")


if __name__ == "__main__":
    main()
