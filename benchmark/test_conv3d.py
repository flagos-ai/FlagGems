import pytest
import torch

import flag_gems

from . import base, consts


class Conv3DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


class Conv3DPaddingBenchmark(base.GenericBenchmark):
    CONV3D_PADDING_SHAPES = [
        (2, 3, 9, 9, 9, 3, 3, 3, 3, 1, "valid", 1),
        (9, 16, 4, 4, 4, 128, 2, 2, 2, 1, "valid", 4),
        (32, 8, 8, 8, 8, 32, 3, 3, 3, 1, "valid", 1),
        (2, 3, 9, 9, 9, 3, 3, 3, 3, 1, "same", 1),
        (9, 16, 4, 4, 4, 128, 2, 2, 2, 1, "same", 4),
        (32, 8, 8, 8, 8, 32, 3, 3, 3, 1, "same", 1),
    ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.CONV3D_PADDING_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


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


@pytest.mark.conv3d_padding
def test_conv3d_padding():
    def conv3d_padding_input_fn(shape, dtype, device):
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
    bench = Conv3DPaddingBenchmark(
        op_name="conv3d_padding",
        input_fn=conv3d_padding_input_fn,
        torch_op=torch.nn.functional.conv3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv3d)
    bench.run()
