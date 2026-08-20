import os
import sys
import types

import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

clip_stub = types.ModuleType("clip")
clip_stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(
    RuntimeError("clip.load is not used in adapter tests")
)
sys.modules.setdefault("clip", clip_stub)

net_stub = types.ModuleType("models.Net")
net_stub.FeatureEncoderMult = nn.Module
net_stub.IBasicBlock = nn.Module
net_stub.conv1x1 = lambda *args, **kwargs: nn.Identity()
sys.modules.setdefault("models.Net", net_stub)


class PixelNormStub(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-8)


stylegan_stub = types.ModuleType("models.stylegan2.model")
stylegan_stub.PixelNorm = PixelNormStub
sys.modules.setdefault("models.stylegan2.model", stylegan_stub)

from models.Encoders import DirectColorBlendAdapterV8
from models.color_condition_v8 import COLOR_DESCRIPTOR_DIM


class DummyClip(nn.Module):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=(-2, -1)).mean(dim=1, keepdim=True).expand(-1, 512)


def build_adapter(**kwargs) -> DirectColorBlendAdapterV8:
    defaults = {
        "clip_model_module": DummyClip(),
        "descriptor_hidden": 32,
        "correction_hidden": 32,
        "layer_embed_dim": 8,
    }
    defaults.update(kwargs)
    return DirectColorBlendAdapterV8(**defaults)


def synthetic_inputs(batch: int = 3):
    latent_face = torch.randn(batch, 12, 512)
    latent_color = latent_face + 0.5 * torch.randn_like(latent_face)
    descriptor = torch.randn(batch, COLOR_DESCRIPTOR_DIM)
    return latent_face, latent_color, descriptor


def run_adapter(
    model: DirectColorBlendAdapterV8,
    latent_face: torch.Tensor,
    latent_color: torch.Tensor,
    descriptor: torch.Tensor,
    chroma_gate: torch.Tensor,
    lightness_gate: torch.Tensor,
    edit_gate: torch.Tensor,
    **kwargs,
):
    return model(
        latent_face=latent_face,
        latent_color=latent_color,
        color_descriptor=descriptor,
        chroma_need_gate=chroma_gate,
        lightness_need_gate=lightness_gate,
        edit_need_gate=edit_gate,
        return_aux=True,
        **kwargs,
    )
