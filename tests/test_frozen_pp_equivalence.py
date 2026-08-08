import torch
import torch.nn as nn

from models.postprocess_v5 import PostProcessModelV5
from scripts.validate_frozen_pp import deterministic_render, official_render


class RecordingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, styles, **kwargs):
        self.calls.append(kwargs)
        layer_in = kwargs["layer_in"]
        latent_term = styles[0].mean(dim=(1, 2)).view(-1, 1, 1, 1)
        return layer_in[:, :3] * self.scale + latent_term, None


def test_frozen_pp_official_one_shot_render_equivalence():
    torch.manual_seed(11)
    latent_s = torch.randn(2, 18, 8, requires_grad=True)
    latent_f = torch.randn(2, 4, 16, 16, requires_grad=True)
    generator = RecordingGenerator()
    expected = official_render(generator, latent_s, latent_f)
    generator.calls.clear()
    actual = PostProcessModelV5.render_frozen_pp(object(), generator, latent_s, latent_f)
    error = float((actual - expected).abs().max().item())
    assert error < 1e-5
    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert call["start_layer"] == 5
    assert call["end_layer"] == 8
    assert call["layer_in"] is latent_f
    assert "skip" not in call
    assert not actual.requires_grad
    assert generator.scale.grad is None


def test_validation_render_uses_fixed_registered_noise():
    generator = RecordingGenerator()
    latent_s = torch.zeros(1, 18, 8)
    latent_f = torch.zeros(1, 4, 8, 8)
    first = deterministic_render(generator, latent_s, latent_f)
    second = deterministic_render(generator, latent_s, latent_f)
    assert torch.equal(first, second)
    assert len(generator.calls) == 2
    assert all(call["randomize_noise"] is False for call in generator.calls)
