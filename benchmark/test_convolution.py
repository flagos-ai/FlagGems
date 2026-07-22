from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


def convolution_input_fn(shape, dtype, device):
    """Generate convolution test inputs from shape tuple.

    Shape format: (batch, input_c, input_h, input_w, out_c, kernel_h, kernel_w, stride, padding, groups)
    """
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

    inp = torch.randn(size=input_shape, device=device, dtype=dtype)
    weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

    yield {
        "input": inp,
        "weight": weight,
        "bias": None,
        "stride": [stride, stride] if isinstance(stride, int) else stride,
        "padding": [padding, padding] if isinstance(padding, int) else padding,
        "dilation": [1, 1],
        "transposed": False,
        "output_padding": [0, 0],
        "groups": groups,
    },


class ConvolutionBenchmark(base.GenericBenchmark):
    """Custom benchmark for convolution operator."""

    def get_input_iter(self, cur_dtype) -> Generator:
        # Custom shapes: (batch, input_c, input_h, input_w, out_c, kernel_h, kernel_w, stride, padding, groups)
        shapes = [
            (1, 3, 32, 32, 8, 3, 3, 1, 1, 1),
            (1, 8, 16, 16, 16, 3, 3, 1, 1, 1),
            (1, 16, 8, 8, 32, 3, 3, 1, 1, 1),
            (2, 8, 16, 16, 16, 3, 3, 1, 1, 1),
            (2, 16, 8, 8, 32, 3, 3, 2, 1, 1),
        ]

        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            shapes.extend(
                [
                    (4, 64, 56, 56, 64, 3, 3, 1, 1, 1),
                    (4, 128, 28, 28, 128, 3, 3, 1, 1, 1),
                    (8, 32, 16, 16, 32, 3, 3, 1, 1, 1),
                ]
            )

        for shape in shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.convolution
def test_convolution():
    torch.backends.cudnn.allow_tf32 = False
    bench = ConvolutionBenchmark(
        input_fn=convolution_input_fn,
        op_name="convolution",
        torch_op=torch.convolution,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.convolution)
    bench.run()
