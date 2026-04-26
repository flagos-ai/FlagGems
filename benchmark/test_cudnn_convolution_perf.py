import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import GenericBenchmark


class CudnnConv2DBenchmark(GenericBenchmark):
    # (batch, input_c, input_h, input_w, out_c, kernel_h, kernel_w, stride, padding, groups)
    DEFAULT_SHAPES = [
        (32, 64, 128, 128, 32, 3, 3, 1, 2, 1),
        (32, 64, 210, 210, 16, 5, 5, 2, 1, 1),
        (16, 32, 12, 12, 24, 3, 3, 2, 1, 1),
    ]

    def set_more_shapes(self):
        return [
            (16, 32, 24, 24, 24, 3, 3, 2, 2, 2),
            (16, 32, 24, 24, 24, 3, 3, 1, 2, 2),
        ]


@pytest.mark.cudnn_convolution
def test_cudnn_convolution():
    def cudnn_convolution_input_fn(shape, dtype, device):
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
        input_tensor = torch.randn(size=input_shape, device=device, dtype=dtype)
        weight = torch.randn(size=weight_shape, device=device, dtype=dtype)

        yield (
            input_tensor,
            weight,
            [padding, padding],
            [stride, stride],
            [1, 1],
            groups,
            False,
            False,
            False,
        )

    torch.backends.cudnn.allow_tf32 = False
    bench = CudnnConv2DBenchmark(
        input_fn=cudnn_convolution_input_fn,
        op_name="cudnn_convolution",
        torch_op=torch.ops.aten.cudnn_convolution.default,
        dtypes=FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_convolution)
    bench.run()
