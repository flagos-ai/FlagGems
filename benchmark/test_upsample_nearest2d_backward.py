import pytest
import torch

from . import base, consts


class UpsampleNearest2dBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # (N, C, H, W) for the input shape; output shape is scaled up.
        return []


def _input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    scale_factors = (2, 2)
    output_size = (height * scale_factors[0], width * scale_factors[1])
    grad_output = torch.randn(
        (batch, channel, output_size[0], output_size[1]),
        device=device,
        dtype=dtype,
    )
    yield {
        "grad_output": grad_output,
        "output_size": output_size,
        "input_size": (batch, channel, height, width),
        "scales_h": None,
        "scales_w": None,
    },


def _torch_ref(grad_output, output_size, input_size, scales_h, scales_w):
    """Build the forward graph once and use autograd to get the backward;
    that way the comparison reflects what PyTorch users actually see."""
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
