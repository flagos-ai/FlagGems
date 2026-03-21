from typing import Generator

import pytest
import torch

from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input
from flag_gems.ops.upsample_linear1d_backward import upsample_linear1d_backward


class UpsampleLinear1dBackwardBenchmark(Benchmark):
    """
    Benchmark for upsample_linear1d_backward operator.
    """

    def set_more_shapes(self):
        shapes_3d = [(4, 16, 2**i) for i in range(4, 14, 2)]
        shapes_2d = [(16, 2**i) for i in range(6, 16, 2)]
        return shapes_3d + shapes_2d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            for scale_factor in [0.5, 2.0]:
                for align_corners in [False, True]:
                    in_w = shape[-1]
                    out_w = max(1, int(in_w * scale_factor))

                    grad_shape = list(shape)
                    grad_shape[-1] = out_w

                    grad = torch.randn(
                        grad_shape,
                        device=self.device,
                        dtype=cur_dtype,
                    )

                    yield grad, None, list(shape), align_corners, [scale_factor]

    def get_tflops(self, op, *args, **kwargs):
        grad, output_size, input_size, align_corners, scale_factors = args
        return grad.numel() * 2


@pytest.mark.upsample_linear1d_backward
@pytest.mark.parametrize(
    "dtype",
    FLOAT_DTYPES,
)
def test_upsample_linear1d_backward_perf(dtype):
    bench = UpsampleLinear1dBackwardBenchmark(
        op_name="upsample_linear1d_backward",
        torch_op=upsample_linear1d_backward,
        dtypes=[dtype],
    )

    bench.run()
