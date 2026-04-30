import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark


class ConvTranspose2DBenchmark(GenericBenchmark):
    def set_more_shapes(self):
        return [
            # (batch, in_c, in_h, in_w, out_c, kh, kw, stride, padding, groups)
            (32, 64, 16, 16, 32, 3, 3, 1, 1, 1),
            (32, 64, 16, 16, 32, 3, 3, 2, 1, 1),
            (16, 32, 32, 32, 64, 3, 3, 2, 1, 1),
            (16, 32, 8, 8, 24, 5, 5, 2, 2, 1),
            (8, 128, 4, 4, 64, 4, 4, 2, 1, 1),
            (4, 256, 8, 8, 128, 3, 3, 2, 1, 1),
        ]


@pytest.mark.conv_transpose2d
def test_perf_conv_transpose2d():
    def conv_transpose2d_input_fn(shape, dtype, device):
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
            groups,
        ) = shape
        input_shape = (batch, input_c, input_h, input_w)
        weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
        inp = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": inp,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        input_fn=conv_transpose2d_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_transpose2d)
    bench.run()
