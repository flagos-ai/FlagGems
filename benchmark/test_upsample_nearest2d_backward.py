import pytest
import torch

from . import base, consts


class UpsampleBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    scale_factors = (2, 2)
    output_size = (
        int(height * scale_factors[0]),
        int(width * scale_factors[1]),
    )
    grad_output = torch.randn(
        (batch, channel, output_size[0], output_size[1]),
        device=device,
        dtype=dtype,
    )
    yield {
        "grad_output": grad_output,
        "output_size": output_size,
        "input_size": list(shape),
        "scales_h": None,
        "scales_w": None,
    },


def torch_backward_op(grad_output, output_size, input_size, scales_h=None, scales_w=None):
    return torch.ops.aten.upsample_nearest2d_backward(
        grad_output, output_size, input_size, scales_h, scales_w
    )


@pytest.mark.upsample_nearest2d_backward
def test_upsample_nearest2d_backward():
    bench = UpsampleBackwardBenchmark(
        op_name="upsample_nearest2d",
        torch_op=torch_backward_op,
        dtypes=consts.FLOAT_DTYPES,
        input_fn=_input_fn,
    )

    bench.run()
