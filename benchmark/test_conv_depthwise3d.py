import pytest
import torch

import flag_gems

from . import base, consts


class ConvDepthwise3DBenchmark(base.GenericBenchmark):
    # (batch, channels, input_d, input_h, input_w,
    #  kernel_d, kernel_h, kernel_w, stride, padding, dilation)
    CONV_DEPTHWISE3D_SHAPES = [
        (1, 32, 16, 32, 32, 3, 3, 3, 1, 1, 1),
        (1, 64, 8, 16, 16, 3, 3, 3, 2, 1, 1),
        (2, 16, 8, 8, 8, 3, 3, 3, 1, 1, 1),
        (4, 32, 4, 16, 16, 3, 3, 3, 1, 0, 1),
    ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.CONV_DEPTHWISE3D_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


def _input_fn(shape, dtype, device):
    (
        batch,
        channels,
        input_d,
        input_h,
        input_w,
        kernel_d,
        kernel_h,
        kernel_w,
        stride,
        padding,
        dilation,
    ) = shape
    input_shape = (batch, channels, input_d, input_h, input_w)
    weight_shape = (channels, 1, kernel_d, kernel_h, kernel_w)
    input_tensor = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    # Pass as positional args since the first arg is named 'self' in aten op
    yield (
        input_tensor,
        weight,
        [kernel_d, kernel_h, kernel_w],
        None,  # bias
        [stride, stride, stride],
        [padding, padding, padding],
        [dilation, dilation, dilation],
    )


@pytest.mark.conv_depthwise3d
def test_conv_depthwise3d():
    torch.backends.cudnn.allow_tf32 = False

    bench = ConvDepthwise3DBenchmark(
        op_name="conv_depthwise3d",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.conv_depthwise3d,
        gems_op=flag_gems.conv_depthwise3d,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
