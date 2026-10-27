import pytest
import torch

import flag_gems

from . import base, consts

# 2D convolution cases (valid / same / same+dilation). Each entry is
# (N, C_in, H, W, C_out, Kh, Kw, stride, padding, dilation, groups).
DEFAULT_SHAPES = [
    (16, 32, 24, 24, 24, 3, 3, 1, "valid", 1, 1),
    (32, 64, 128, 128, 32, 3, 3, 1, "valid", 1, 1),
    (32, 64, 210, 210, 16, 5, 5, 1, "valid", 1, 1),
    (16, 32, 24, 24, 24, 3, 3, 1, "same", 1, 1),
    (32, 64, 128, 128, 32, 3, 3, 1, "same", 1, 1),
    (32, 64, 210, 210, 16, 5, 5, 1, "same", 1, 1),
    (16, 32, 24, 24, 24, 3, 3, 1, "same", 2, 1),
]


class ConvolutionModeBenchmark(base.GenericBenchmark):
    """Benchmark ``torch._convolution_mode`` via the FlagGems dispatcher.

    Each shape encodes a full 2D convolution case as
    ``(batch, input_c, input_h, input_w, out_c, kernel_h, kernel_w, stride,
    padding, dilation, groups)`` where ``padding`` is the string mode
    (``"valid"`` or ``"same"``); ``same`` requires unit strides.
    """

    # 2D convolution cases (valid / same / same+dilation). Each entry is
    # (N, C_in, H, W, C_out, Kh, Kw, stride, padding, dilation, groups).
    DEFAULT_SHAPES = [
        (16, 32, 24, 24, 24, 3, 3, 1, "valid", 1, 1),
        (32, 64, 128, 128, 32, 3, 3, 1, "valid", 1, 1),
        (32, 64, 210, 210, 16, 5, 5, 1, "valid", 1, 1),
        (16, 32, 24, 24, 24, 3, 3, 1, "same", 1, 1),
        (32, 64, 128, 128, 32, 3, 3, 1, "same", 1, 1),
        (32, 64, 210, 210, 16, 5, 5, 1, "same", 1, 1),
        (16, 32, 24, 24, 24, 3, 3, 1, "same", 2, 1),
    ]

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.DEFAULT_SHAPES:
            yield from self.input_fn(shape, dtype, self.device)


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
        dilation,
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
        "stride": (stride, stride),
        "padding": padding,
        "dilation": (dilation, dilation),
        "groups": groups,
    },


@pytest.mark.convolution_mode
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issue #4131: conv kernels not working",
)
def test_convolution_mode(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvolutionModeBenchmark(
        input_fn=_input_fn,
        op_name="convolution_mode",
        torch_op=torch.ops.aten._convolution_mode,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems._convolution_mode)

    bench.run()
