"""Shared V2.33 diagnostic and real-inference tensor contract."""

from __future__ import annotations

import torch

from models.v232_runtime_inputs import build_v832_runtime_inputs


def build_v833_runtime_inputs(**kwargs) -> dict[str, torch.Tensor]:
    return build_v832_runtime_inputs(**kwargs)


__all__ = ["build_v833_runtime_inputs"]
