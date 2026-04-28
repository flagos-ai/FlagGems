import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class UpsampleBackwardBenchmark(utils.GenericBenchmark):
    DEFAULT_SHAPES = [
        (1, 64, 56, 56, 1.0, 1.0, 0, 0),
        (1, 3, 224, 224, 2.0, 2.0, 0, 0),
        (8, 3, 224, 224, 2.0, 2.0, 1, 0),
        (1, 64, 56, 56, 2.0, 2.0, 1, 0),
        (1, 64, 56, 56, 2.0, 2.0, 1, 1),
        (16, 64, 56, 56, 2.0, 2.0, 1, 0),
        (8, 128, 28, 28, 2.1, 3.7, 1, 0),
        (4, 256, 14, 14, 1.3, 5.1, 1, 0),
        (1, 64, 128, 256, 0.5, 0.5, 1, 0),
        (1, 64, 128, 256, 0.3, 0.5, 1, 0),
    ]
    DEFAULT_SHAPE_DESC = "N, C, H, W, scale_h, scale_w, use_scales, channels_last"

    def set_shapes(self, shape_file_path=None):
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return []

    def record_shapes(self, *args, **kwargs):
        grad_output = kwargs["grad_output"]
        output_size = kwargs["output_size"]
        input_size = kwargs["input_size"]
        scales_h = kwargs["scales_h"]
        scales_w = kwargs["scales_w"]
        layout = (
            "channels_last"
            if grad_output.is_contiguous(memory_format=torch.channels_last)
            and grad_output.stride(1) == 1
            else "contiguous"
        )
        return {
            "case": _case_name(input_size[-2:], output_size, scales_h, scales_w),
            "layout": layout,
            "grad_output": grad_output.size(),
            "output_size": output_size,
            "input_size": input_size,
            "scales_h": scales_h,
            "scales_w": scales_w,
        }


def _input_fn(shape, dtype, device):
    batch, channel, height, weight, scale_h, scale_w, use_scales, channels_last = shape
    output_size = (
        int(height * scale_h),
        int(weight * scale_w),
    )
    grad_output = torch.randn(
        size=(batch, channel, *output_size), device=device, dtype=dtype
    )
    if channels_last:
        grad_output = grad_output.contiguous(memory_format=torch.channels_last)
    yield {
        "grad_output": grad_output,
        "output_size": output_size,
        "input_size": (batch, channel, height, weight),
        "scales_h": scale_h if use_scales else None,
        "scales_w": scale_w if use_scales else None,
    },


def _case_name(input_hw, output_size, scales_h, scales_w):
    in_h, in_w = input_hw
    out_h, out_w = output_size
    if out_h == in_h and out_w == in_w and scales_h is None and scales_w is None:
        return "identity_fresh_copy"
    if out_h < in_h or out_w < in_w:
        return "downsample"
    if scales_h == 2.0 and scales_w == 2.0:
        return "x2_scales"
    if out_h == in_h * 2 and out_w == in_w * 2:
        return "x2_output_size"
    return "non_integer_upsample"


@pytest.mark.upsample_nearest2d
def test_upsample_nearest2d_backward():
    bench = UpsampleBackwardBenchmark(
        op_name="upsample_nearest2d_backward",
        input_fn=_input_fn,
        torch_op=torch.ops.aten.upsample_nearest2d_backward,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
