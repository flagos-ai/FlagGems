import pytest
import torch

from . import base, consts, utils


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp,

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield inp, {"offset": 1, "dim1": 0, "dim2": -1},


@pytest.mark.diagonal_backward
def test_diagonal_backward():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="diagonal_backward",
        input_fn=_input_fn,
        torch_op=torch.diagonal,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )

    bench.run()
