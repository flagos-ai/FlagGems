import pytest
import torch

from . import base, consts, utils


class SelectBackwardBenchmark(base.Benchmark):
    """
    Benchmark for select_backward operator.
    """

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**exponent) for exponent in range(0, 20, 4)]
        special_shapes_3d = [(64, 64, 2**exponent) for exponent in range(0, 15, 4)]
        return special_shapes_2d + special_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, cur_dtype, self.device)
            ndim = len(shape)

            dim = 1 if ndim > 1 else 0
            actual_dim = dim if dim >= 0 else dim + ndim
            index = shape[actual_dim] // 2

            y = torch.select(x, actual_dim, index)
            grad = torch.randn_like(y)

            yield grad, shape, actual_dim, index

    def get_tflops(self, op, *args, **kwargs):
        grad = args[0]
        return grad.numel()


@pytest.mark.select_backward
def test_select_backward():
    bench = SelectBackwardBenchmark(
        op_name="select_backward",
        torch_op=torch.ops.aten.select_backward,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
