import pytest
import torch
from typing import Generator

from benchmark.performance_utils import Benchmark, generate_tensor_input
from benchmark.attri_util import FLOAT_DTYPES

from flag_gems.ops.select_backward import select_backward


class SelectBackwardBenchmark(Benchmark):
    """
    Benchmark for select_backward operator.
    """

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:

            x = generate_tensor_input(shape, cur_dtype, self.device)
            ndim = len(shape)

            dim = 1 if ndim > 1 else 0
            actual_dim = dim if dim >= 0 else dim + ndim

            index = shape[actual_dim] // 2

            y = torch.select(x, actual_dim, index)
            grad = torch.randn_like(y)

            yield grad, shape, actual_dim, index

    def get_tflops(self, op, *args, **kwargs):
        grad, shape, _, _ = args
        return grad.numel()


@pytest.mark.select_backward
@pytest.mark.parametrize(
    "dtype",
    FLOAT_DTYPES,
)
def test_select_backward_perf(dtype):

    bench = SelectBackwardBenchmark(
        op_name="select_backward",
        torch_op=select_backward,
        dtypes=[dtype],
    )

    bench.run()
