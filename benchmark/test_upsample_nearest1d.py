import pytest
import torch

from . import base, consts


# TODO(Qiming): Kill this class
class UpsampleBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # self.shapes is a list of tuples, each containing three elements:
        # (N, C, H, W).
        return []


def _input_fn(shape, dtype, device):
    batch, channel, height, width = shape
    length = height * width  # flatten spatial dims to 1D length
    input = torch.randn((batch, channel, length), device=device, dtype=dtype)
    scale_factors = 2
    output_size = int(length * scale_factors)
    yield {
        "input": input,
        "output_size": (output_size,),
        "scales": None,
    },


@pytest.mark.upsample_nearest1d
def test_upsample_nearest1d():
    bench = UpsampleBenchmark(
        input_fn=_input_fn,
        op_name="upsample_nearest1d",
        torch_op=torch._C._nn.upsample_nearest1d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
