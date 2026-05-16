import pytest
import torch

from . import base, consts, utils


@pytest.mark.mean
def test_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="mean", torch_op=torch.mean, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def _mean_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, [1]


@pytest.mark.mean_dim
def test_mean_dim():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=_mean_dim_input_fn,
        op_name="mean_dim",
        torch_op=torch.mean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
