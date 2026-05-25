import pytest
import torch

import flag_gems

from . import base, consts


class Conv2DBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
            (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
            (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
            (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
            (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
            (16, 32, 12, 12, 24, 3, 3, 2, "valid", 1),
            (32, 64, 128, 128, 32, 3, 3, 1, "valid", 1),
            (16, 32, 24, 24, 24, 3, 3, 1, "same", 2),
            (32, 64, 210, 210, 16, 5, 5, 1, "same", 1),
        ]


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
    input_shape = (batch, input_c, input_h, input_w)
    weight_shape = (out_c, input_c // groups, kernel_h, kernel_w)
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


@pytest.mark.conv2d
def test_conv2d(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = Conv2DBenchmark(
        input_fn=_input_fn,
        op_name="conv2d",
        torch_op=torch.nn.functional.conv2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv2d)

    bench.run()


class Conv2DPaddingBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (16, 32, 12, 12, 24, 3, 3, 1, "valid", 1),
        (32, 64, 128, 128, 32, 3, 3, 1, "valid", 1),
        (32, 64, 210, 210, 16, 5, 5, 1, "valid", 1),
        (16, 32, 24, 24, 24, 3, 3, 1, "same", 1),
        (32, 64, 128, 128, 32, 3, 3, 1, "same", 1),
        (32, 64, 210, 210, 16, 5, 5, 1, "same", 1),
        (16, 32, 24, 24, 24, 3, 3, 1, "same", 2),
    ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.DEFAULT_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.conv2d_padding
def test_conv2d_padding(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = Conv2DPaddingBenchmark(
        input_fn=_input_fn,
        op_name="conv2d_padding",
        torch_op=torch.nn.functional.conv2d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.conv2d)

    bench.run()
