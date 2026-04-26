from typing import Generator

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark


class ConvTranspose2DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return [
            (16, 32, 24, 24, 24, 3, 3, 1, 1, 0, 2, 1),
            (16, 32, 24, 24, 24, 3, 3, 2, 1, 1, 2, 1),
            (32, 64, 64, 64, 32, 3, 3, 2, 1, 1, 1, 1),
            (32, 64, 128, 128, 32, 5, 5, 2, 2, 1, 1, 1),
            (4, 32, 128, 128, 32, 5, 5, 3, 2, 0, 1, 1),
            (16, 32, 24, 24, 24, 3, 3, 3, 1, 0, 2, 1),
            (16, 32, 24, 24, 24, 3, 3, 4, 1, 0, 2, 1),
            (8, 32, 32, 32, 32, 3, 3, 4, 1, 0, 1, 1),
            (4, 32, 32, 32, 32, 3, 3, 8, 1, 0, 1, 1),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.set_more_shapes():
            yield from self.input_fn(shape, cur_dtype, self.device)


def _input_fn(shape, dtype, device):
    (
        batch,
        input_c,
        input_h,
        input_w,
        out_c,
        kernel_h,
        kernel_w,
        stride,
        padding,
        output_padding,
        groups,
        dilation,
    ) = shape
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
    input = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    yield {
        "input": input,
        "weight": weight,
        "bias": None,
        "stride": stride,
        "padding": padding,
        "output_padding": output_padding,
        "groups": groups,
        "dilation": dilation,
    },


@pytest.mark.conv_transpose2d
def test_conv_transpose2d():
    def gems_conv_transpose2d(**kwargs):
        with torch.no_grad():
            return flag_gems.conv_transpose2d(**kwargs)

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        input_fn=_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(gems_conv_transpose2d)
    bench.run()
