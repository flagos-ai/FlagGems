import pytest
import torch

import flag_gems

from . import base, consts


class ConvolutionBenchmark(base.GenericBenchmark):
    # (batch, in_c, in_h, in_w, out_c, kernel_h, kernel_w, stride, padding, groups)
    # covering common CNN patterns: large feature maps, larger kernels/strides,
    # and small feature maps with strided convolutions.
    DEFAULT_SHAPES = [
        (32, 64, 128, 128, 32, 3, 3, 1, 1, 1),
        (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
        (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
        (16, 32, 24, 24, 24, 3, 3, 2, 2, 1),
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
        "stride": [stride, stride],
        "padding": [padding, padding],
        "dilation": [1, 1],
        "transposed": False,
        "output_padding": [0, 0],
        "groups": groups,
    },


@pytest.mark.convolution
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_convolution(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvolutionBenchmark(
        input_fn=_input_fn,
        op_name="convolution",
        torch_op=torch.convolution,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.convolution)

    bench.run()
