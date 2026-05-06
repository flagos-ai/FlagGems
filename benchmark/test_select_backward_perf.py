import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils

_BASE_2D_SIZE = 1024
_BASE_3D_SIZE = 64
_MIN_EXPONENT = 0
_MAX_2D_EXPONENT = 20
_MAX_3D_EXPONENT = 15
_EXPONENT_STEP = 4
_FIRST_DIM = 0
_DEFAULT_SELECT_DIM = 1
_INDEX_DIVISOR = 2


class SelectBackwardBenchmark(utils.Benchmark):
    """
    Benchmark for select_backward operator.
    """

    def set_more_shapes(self):
        special_shapes_2d = [
            (_BASE_2D_SIZE, 2**exponent)
            for exponent in range(_MIN_EXPONENT, _MAX_2D_EXPONENT, _EXPONENT_STEP)
        ]
        special_shapes_3d = [
            (_BASE_3D_SIZE, _BASE_3D_SIZE, 2**exponent)
            for exponent in range(_MIN_EXPONENT, _MAX_3D_EXPONENT, _EXPONENT_STEP)
        ]
        return special_shapes_2d + special_shapes_3d

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = utils.generate_tensor_input(shape, cur_dtype, self.device)
            ndim = len(shape)

            dim = _DEFAULT_SELECT_DIM if ndim > _DEFAULT_SELECT_DIM else _FIRST_DIM
            actual_dim = dim if dim >= 0 else dim + ndim
            index = shape[actual_dim] // _INDEX_DIVISOR

            y = torch.select(x, actual_dim, index)
            grad = torch.randn_like(y)

            yield grad, shape, actual_dim, index

    def get_tflops(self, op, *args, **kwargs):
        grad = args[0]
        return grad.numel()


@pytest.mark.select_backward
@pytest.mark.parametrize(
    "dtype",
    attr_utils.FLOAT_DTYPES,
)
def test_select_backward_perf(dtype):
    bench = SelectBackwardBenchmark(
        op_name="select_backward",
        torch_op=torch.ops.aten.select_backward,
        dtypes=[dtype],
    )

    bench.run()
