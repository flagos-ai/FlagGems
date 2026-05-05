import pytest
import torch

import flag_gems

from . import base, consts


class ConvTranspose2DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        # (batch, input_c, input_h, input_w, out_c, kernel_h, kernel_w, stride, padding, groups)
        return [
            (32, 64, 32, 32, 128, 3, 3, 2, 1, 1),
            (16, 32, 16, 16, 64, 3, 3, 2, 1, 1),
            (4, 3, 32, 32, 6, 3, 3, 2, 1, 1),
            (4, 3, 32, 32, 6, 3, 3, 1, 1, 1),
            (8, 16, 16, 16, 32, 3, 3, 2, 1, 2),
            (4, 3, 32, 32, 6, 3, 3, 2, 1, 1),
            (2, 16, 32, 32, 32, 3, 3, 2, 1, 2),
            (4, 3, 32, 32, 6, 5, 5, 2, 2, 1),
        ]

    def set_shapes(self, shape_file_path=None):
        # Only use conv-specific shapes, not generic DEFAULT_SHAPES
        self.shapes = self.set_more_shapes() or []


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
        groups,
    ) = shape
    # conv_transpose2d weight layout: (C_in, C_out/groups, kH, kW)
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (input_c, out_c // groups, kernel_h, kernel_w)
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


@pytest.mark.conv_transpose2d
def test_conv_transpose2d(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvTranspose2DBenchmark(
        input_fn=_input_fn,
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv_transpose2d)

    bench.run()
