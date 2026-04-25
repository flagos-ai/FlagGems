import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class UpsampleBenchmark(utils.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return None


@pytest.mark.upsample_linear1d
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_linear1d(align_corners):
    def upsample_linear1d_input_fn(shape, dtype, device):
        batch, channel, height, width = shape
        length = height * width
        input = torch.randn((batch, channel, length), device=device, dtype=dtype)
        scale_factors = 2
        output_size = int(length * scale_factors)
        yield {
            "input": input,
            "output_size": (output_size,),
            "align_corners": align_corners,
        },

    bench = UpsampleBenchmark(
        input_fn=upsample_linear1d_input_fn,
        op_name=f"upsample_linear1d_align_{align_corners}",
        torch_op=torch._C._nn.upsample_linear1d,
        dtypes=attr_utils.FLOAT_DTYPES,
    )
    bench.run()
