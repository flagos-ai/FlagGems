import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class UpsampleBenchmark(utils.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


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
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
