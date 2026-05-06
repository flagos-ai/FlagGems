import pytest
import torch

from . import base, consts


class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


def _input_fn(shape, dtype, device):
    batch, channel, height, weight = shape
    input = torch.randn(size=shape, device=device, dtype=dtype)
    scale_factors = (2, 2)
    output_size = (
        int(height * scale_factors[0]),
        int(weight * scale_factors[1]),
    )
    yield {
        "input": input,
        "output_size": output_size,
        "scales_h": None,
        "scales_w": None,
    },


def _backward_input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    scale_factors = (2, 2)
    OH = int(height * scale_factors[0])
    OW = int(width * scale_factors[1])
    grad_output = torch.randn(batch, channel, OH, OW, device=device, dtype=dtype)
    output_size = (OH, OW)
    input_size = (batch, channel, height, width)
    yield {
        "grad_output": grad_output,
        "output_size": output_size,
        "input_size": input_size,
        "scales_h": None,
        "scales_w": None,
    },


@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d():
    bench = UpsampleBenchmark(
        op_name="upsample_nearest2d",
        input_fn=_input_fn,
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()


@pytest.mark.upsample_nearest2d_backward
def test_upsample_nearest2d_backward():
    bench = UpsampleBenchmark(
        op_name="upsample_nearest2d_backward",
        input_fn=_backward_input_fn,
        torch_op=torch.ops.aten.upsample_nearest2d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
