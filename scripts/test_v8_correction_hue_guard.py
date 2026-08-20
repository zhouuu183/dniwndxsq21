from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.color_condition_v8 import correction_hue_regression_loss


def main():
    anchor = torch.tensor([5.0])
    final = torch.tensor([10.0])
    loss = correction_hue_regression_loss(anchor, final, tolerance_deg=1.5)
    assert float(loss.item()) > 0.0
    assert float(correction_hue_regression_loss(anchor, torch.tensor([6.0]), 1.5).item()) == 0.0
    print("v8 correction hue guard tests passed")


if __name__ == "__main__":
    main()
