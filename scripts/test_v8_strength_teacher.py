from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.direct_strength_teacher_v8 import (
    DEFAULT_TEACHER_ALPHA_CANDIDATES,
    select_teacher_strength,
    summarize_teacher_records,
)


def main():
    scores = torch.tensor([[5.0, 4.0, 2.0, 1.5, 0.8, 0.1]])
    selected = select_teacher_strength(DEFAULT_TEACHER_ALPHA_CANDIDATES, scores)
    assert float(selected["teacher_alpha"].item()) == 1.0
    assert float(selected["teacher_confidence"].item()) > 0.0
    records = {
        "a": {"teacher_alpha": 0.0, "high_color": False, "white_or_light_hair": False},
        "b": {"teacher_alpha": 1.0, "high_color": True, "white_or_light_hair": True},
    }
    summary = summarize_teacher_records(records)
    assert summary["std_alpha"] > 0.1
    print("v8 strength teacher tests passed")


if __name__ == "__main__":
    main()
