import os
import shutil
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
fused = None
_fused_load_attempted = False


def _get_fused_extension():
    global fused, _fused_load_attempted
    if _fused_load_attempted:
        return fused
    _fused_load_attempted = True
    if os.name == "nt" and shutil.which("cl") is None:
        warnings.warn("MSVC cl.exe is unavailable; using the PyTorch fused-leaky-ReLU fallback.")
        return None
    try:
        fused = load(
            "fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
        )
    except (OSError, RuntimeError) as error:
        warnings.warn("Unable to load fused_bias_act CUDA extension; using PyTorch fallback: %s" % error)
        fused = None
    return fused


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        extension = _get_fused_extension()
        if extension is None:
            raise RuntimeError("Fused CUDA backward was selected without an available extension.")
        grad_input = extension.fused_bias_act(
            grad_output, empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        (out,) = ctx.saved_tensors
        extension = _get_fused_extension()
        if extension is None:
            raise RuntimeError("Fused CUDA double-backward was selected without an available extension.")
        gradgrad_out = extension.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)
        extension = _get_fused_extension()
        if extension is None:
            raise RuntimeError("Fused CUDA forward was selected without an available extension.")
        out = extension.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.negative_slope, ctx.scale
        )

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    extension = None if input.device.type == "cpu" else _get_fused_extension()
    if input.device.type == "cpu" or extension is None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )

    else:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
