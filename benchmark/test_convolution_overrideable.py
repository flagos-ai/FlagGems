import pytest
import torch

import flag_gems

from . import base, consts

# 2D convolution cases. Each entry is
# (N, C_in, H, W, C_out, Kh, Kw, stride, padding, dilation, groups).
# Channels are >= 16 to satisfy the tl.dot K >= 16 constraint of the kernel.
DEFAULT_SHAPES = [
    (16, 32, 24, 24, 24, 3, 3, 1, 1, 1, 1),
    (32, 64, 64, 64, 32, 3, 3, 1, 1, 1, 1),
    (32, 64, 128, 128, 32, 3, 3, 1, 1, 1, 1),
    (16, 32, 24, 24, 24, 3, 3, 2, 1, 1, 1),
    (32, 64, 64, 64, 32, 5, 5, 1, 2, 1, 1),
]


class ConvolutionOverrideableBenchmark(base.GenericBenchmark):
    """Benchmark ``aten::convolution_overrideable`` via the FlagGems dispatcher.

    ``convolution_overrideable`` is a dispatch-only stub with no native CUDA
    kernel, so the torch baseline uses the fully-general ``aten::convolution``
    while the gems path exercises the FlagGems Triton implementation.
    """

    DEFAULT_SHAPES = DEFAULT_SHAPES

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
        "stride": [stride, stride],
        "padding": [padding, padding],
        "dilation": [dilation, dilation],
        "transposed": False,
        "output_padding": [0, 0],
        "groups": groups,
    },


@pytest.mark.convolution_overrideable
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issue #4131: conv kernels not working",
)
def test_convolution_overrideable(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False
    bench = ConvolutionOverrideableBenchmark(
        input_fn=_input_fn,
        op_name="convolution_overrideable",
        torch_op=torch.ops.aten.convolution,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.convolution_overrideable)

    bench.run()


@pytest.mark.convolution_overrideable_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issue #4131: conv kernels not working",
)
def test_convolution_overrideable_out(monkeypatch):
    if flag_gems.vendor_name == "hygon":
        monkeypatch.setenv("TRITON_HIP_USE_NEW_STREAM_PIPELINE", "0")

    torch.backends.cudnn.allow_tf32 = False

    def _out_gems(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    ):
        out = torch.empty(
            (
                input.shape[0],
                weight.shape[0],
                (
                    input.shape[2]
                    + 2 * padding[0]
                    - dilation[0] * (weight.shape[2] - 1)
                    - 1
                )
                // stride[0]
                + 1,
                (
                    input.shape[3]
                    + 2 * padding[1]
                    - dilation[1] * (weight.shape[3] - 1)
                    - 1
                )
                // stride[1]
                + 1,
            ),
            device=input.device,
            dtype=input.dtype,
        )
        return flag_gems.convolution_overrideable_out(
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            out=out,
        )

    bench = ConvolutionOverrideableBenchmark(
        input_fn=_input_fn,
        op_name="convolution_overrideable_out",
        torch_op=torch.ops.aten.convolution,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(_out_gems)

    bench.run()
