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


@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d():
    bench = UpsampleBenchmark(
        op_name="upsample_nearest2d",
        input_fn=_input_fn,
        torch_op=torch._C._nn.upsample_nearest2d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
