import os
import sys
import types

import torch
from torch import nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

clip_stub = types.ModuleType("clip")
clip_stub.load = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("clip.load is not used in this test"))
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

from models.Encoders import DirectColorBlendAdapterV8, load_direct_color_adapter_state_v8
from models.color_condition_v8 import COLOR_DESCRIPTOR_DIM


class DummyClip(nn.Module):
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=(-2, -1)).mean(dim=1, keepdim=True).expand(-1, 512)


def run_model(model, latent_face, latent_color, descriptor, edit_gate, **kwargs):
    return model(
        latent_face=latent_face,
        latent_color=latent_color,
        color_descriptor=descriptor,
        chroma_need_gate=edit_gate,
        lightness_need_gate=torch.zeros_like(edit_gate),
        edit_need_gate=edit_gate,
        return_aux=True,
        **kwargs,
    )


def main():
    torch.manual_seed(11)
    batch = 3
    model = DirectColorBlendAdapterV8(
        clip_model_module=DummyClip(),
        descriptor_hidden=32,
        correction_hidden=32,
        layer_embed_dim=8,
    ).eval()
    latent_face = torch.randn(batch, 12, 512)
    latent_color = latent_face + torch.randn_like(latent_face) * 0.5
    descriptor = torch.randn(batch, COLOR_DESCRIPTOR_DIM)
    zeros = torch.zeros(batch)
    ones = torch.ones(batch)

    output_noop, _ = run_model(
        model, latent_face, latent_color, descriptor, zeros,
        correction_enabled=False,
        layer_mix_override=0.0,
    )
    assert torch.equal(output_noop, latent_face), "Test A: gate=0 must be an exact no-op"

    output_full, full_aux = run_model(
        model, latent_face, latent_color, descriptor, ones,
        correction_enabled=False,
        layer_mix_override=1.0,
    )
    assert (output_full - latent_color).abs().max() < 1e-6, "Test B: full direct path must equal Color-S"
    assert full_aux["correction_raw_norm"].max() == 0, "Correction must start at zero"

    output_init, _ = run_model(
        model, latent_face, latent_color, descriptor, ones,
        correction_enabled=False,
    )
    expected_init = latent_face + 0.70 * (latent_color - latent_face)
    assert (output_init - expected_init).abs().max() < 1e-6, "Test C: initial direct mix formula mismatch"

    with torch.no_grad():
        model.correction_mlp[-1].weight.fill_(0.05)
        model.correction_mlp[-1].bias.fill_(0.05)
    output_corrected, correction_aux = run_model(
        model, latent_face, latent_color, descriptor, ones,
        correction_enabled=True,
    )
    assert output_corrected.shape == (batch, 12, 512), "Test E: batch output contract failed"
    correction_excess = correction_aux["correction_norm"] - correction_aux["correction_budget"]
    assert torch.all(correction_excess <= 1e-5), (
        "Test D: correction exceeded its independent budget: "
        f"max_excess={correction_excess.max().item():.8f} "
        f"norm={correction_aux['correction_norm'].tolist()} "
        f"budget={correction_aux['correction_budget'].tolist()}"
    )
    assert torch.isfinite(output_corrected).all()

    reloaded = DirectColorBlendAdapterV8(
        clip_model_module=DummyClip(),
        descriptor_hidden=32,
        correction_hidden=32,
        layer_embed_dim=8,
    ).eval()
    saved_state = {
        key: value.clone()
        for key, value in model.state_dict().items()
        if not key.startswith("clip_model.")
    }
    report = load_direct_color_adapter_state_v8(reloaded, saved_state)
    assert len(report["loaded"]) == len(saved_state)
    incomplete_state = dict(saved_state)
    incomplete_state.pop("strength_head.2.bias")
    try:
        load_direct_color_adapter_state_v8(reloaded, incomplete_state)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Strict V8.4 state loading accepted a missing non-CLIP tensor")

    print(
        "direct_color_adapter_v8 tests passed: "
        f"batch={batch} noop_error={(output_noop-latent_face).abs().max().item():.1e} "
        f"full_error={(output_full-latent_color).abs().max().item():.1e} "
        f"mix_error={(output_init-expected_init).abs().max().item():.1e} "
        f"correction_ratio_max={correction_aux['correction_to_direct_ratio'].max().item():.4f} "
        f"strict_state_tensors={len(report['loaded'])}"
    )


if __name__ == "__main__":
    main()
