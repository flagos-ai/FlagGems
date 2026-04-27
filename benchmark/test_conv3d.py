import pytest
import torch

import flag_gems

from . import base, consts


class Conv3DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


@pytest.mark.conv3d
def test_conv3d():
    def conv3d_input_fn(shape, dtype, device):
        (
            batch,
            input_c,
            input_d,
            input_h,
            input_w,
            out_c,
            kernel_d,
            kernel_h,
            kernel_w,
            stride,
            padding,
            groups,
        ) = shape
        input_shape = (batch, input_c, input_d, input_h, input_w)
        weight_shape = (out_c, input_c // groups, kernel_d, kernel_h, kernel_w)
        input = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield {
            "input": input,
            "weight": weight,
            "bias": None,
            "groups": groups,
            "stride": stride,
            "padding": padding,
        },

    torch.backends.cudnn.allow_tf32 = False
    bench = Conv3DBenchmark(
        op_name="conv3d",
        input_fn=conv3d_input_fn,
        torch_op=torch.nn.functional.conv3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv3d)
    bench.run()
