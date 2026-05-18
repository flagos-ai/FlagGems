import pytest
import torch
from . import base, consts, utils


@pytest.mark.median
def test_median():
    bench = base.UnaryReductionBenchmark(
        op_name="median", torch_op=torch.median, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def _median_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": 1}


@pytest.mark.median_dim
def test_median_dim():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_median_dim_input_fn,
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
