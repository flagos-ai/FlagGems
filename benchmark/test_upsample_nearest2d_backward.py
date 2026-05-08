from typing import Generator

import pytest
import torch

from . import base, consts


class UpsampleNearest2dBackwardBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        # Override default shapes — explicit 4D tensors small enough that
        # `output = input * 2` does not push past 1.5GB at bf16.
        shapes_4d = [
            (4, 3, 224, 224),
            (16, 64, 56, 56),
            (32, 128, 28, 28),
            (64, 256, 14, 14),
            (128, 512, 7, 7),
        ]
        for shape in shapes_4d:
            yield from self.input_fn(shape, dtype, self.device)


def _input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    scale_factors = (2, 2)
    output_size = (height * scale_factors[0], width * scale_factors[1])
    grad_output = torch.randn(
        (batch, channel, output_size[0], output_size[1]),
        device=device,
        dtype=dtype,
    )
    yield grad_output, {
        "output_size": output_size,
        "input_size": (batch, channel, height, width),
        "scales_h": None,
        "scales_w": None,
    }


def _torch_ref(grad_output, output_size, input_size, scales_h, scales_w):
    """Build the forward graph and use autograd to derive the gradient — that
    way the benchmark compares Triton against what the eager PyTorch user
    actually sees, not a hand-rolled scatter loop."""
    x = torch.zeros(
        input_size,
        device=grad_output.device,
        dtype=grad_output.dtype,
        requires_grad=True,
    )
    y = torch._C._nn.upsample_nearest2d(x, output_size, scales_h, scales_w)
    return torch.autograd.grad(y, x, grad_output)[0]


@pytest.mark.upsample_nearest2d_backward
def test_upsample_nearest2d_backward():
    bench = UpsampleNearest2dBackwardBenchmark(
        op_name="upsample_nearest2d_backward",
        input_fn=_input_fn,
        torch_op=_torch_ref,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
