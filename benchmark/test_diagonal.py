import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp,

    if utils.Config.bench_level == utils.BenchLevel.COMPREHENSIVE:
        yield inp, {"offset": 1, "dim1": 0, "dim2": -1},


@pytest.mark.diagonal_backward
def test_diagonal_backward():
    bench = utils.GenericBenchmarkExcluse1D(
        op_name="diagonal_backward",
        input_fn=_input_fn,
        torch_op=torch.diagonal,
        dtypes=attr_utils.FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()
