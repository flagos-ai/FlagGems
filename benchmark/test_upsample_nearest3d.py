import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class UpsampleBenchmark(utils.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


def upsample_nearest3d_input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    depth = 4
    width = width // 4
    new_height = height // depth
    real_shape = (batch, channel, depth, new_height, width)

    input = torch.randn(size=real_shape, device=device, dtype=dtype)
    scale_factors = (2.0, 2.0, 2.0)
    output_size = (
        int(depth * scale_factors[0]),
        int(new_height * scale_factors[1]),
        int(width * scale_factors[2]),
    )

    yield {
        "input": input,
        "output_size": output_size,
        "scales_d": None,
        "scales_h": None,
        "scales_w": None,
    },


@pytest.mark.upsample_nearest3d
def test_upsample_nearest3d():
    bench = UpsampleBenchmark(
        input_fn=upsample_nearest3d_input_fn,
        op_name="upsample_nearest3d",
        torch_op=torch._C._nn.upsample_nearest3d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
