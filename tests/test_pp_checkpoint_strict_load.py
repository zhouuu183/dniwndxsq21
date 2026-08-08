import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from models.postprocess_v5 import FrozenPPBackbone


class TinyFrozenPP(FrozenPPBackbone):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder_face = nn.Linear(3, 4)
        self.to_feature = nn.Conv2d(2, 2, 1)
        self.to_latent_1 = nn.ModuleList([nn.Linear(4, 4)])
        self.to_latent_2 = nn.ModuleList([nn.Linear(4, 4)])
        self.register_buffer("latent_avg", torch.zeros(1), persistent=False)
        self.checkpoint_path = None
        self.checkpoint_sha256 = None
        self.load_statistics = None
        self._freeze()


def _save(path, state):
    torch.save({"model_state_dict": state}, str(path))


def _expect_runtime_error(function, text):
    try:
        function()
    except RuntimeError as error:
        assert text in str(error)
        return
    raise AssertionError("Expected RuntimeError containing %r" % text)


def test_checkpoint_prefix_normalization_and_strict_load():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prefixed.pth"
        model = TinyFrozenPP()
        state = {"module.model.post_process.pp." + key: value.clone() for key, value in model.state_dict().items()}
        _save(path, state)
        statistics = model.load_checkpoint_strict(str(path))
        assert statistics["encoder_face"]["loaded"] == statistics["encoder_face"]["expected"]
        model.train(True)
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())


def test_missing_or_shape_mismatched_required_parameter_fails():
    with tempfile.TemporaryDirectory() as directory:
        base = TinyFrozenPP().state_dict()
        missing_path = Path(directory) / "missing.pth"
        missing = dict(base)
        del missing["to_feature.weight"]
        _save(missing_path, missing)
        _expect_runtime_error(
            lambda: TinyFrozenPP().load_checkpoint_strict(str(missing_path)),
            "missing_required",
        )

        mismatch_path = Path(directory) / "mismatch.pth"
        mismatch = dict(base)
        mismatch["to_latent_1.0.weight"] = torch.zeros(1, 1)
        _save(mismatch_path, mismatch)
        _expect_runtime_error(
            lambda: TinyFrozenPP().load_checkpoint_strict(str(mismatch_path)),
            "shape_mismatch_required",
        )

