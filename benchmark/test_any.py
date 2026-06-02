import pytest
import torch

from . import base, consts


@pytest.mark.any
def test_any():
    bench = base.UnaryReductionBenchmark(
        op_name="any",
        torch_op=torch.any,
        dtypes=consts.FLOAT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.any_dim
def test_any_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="any_dim",
        torch_op=torch.any,
        dtypes=consts.FLOAT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


def any_dims_input_fn(shape, dtype, device):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, dtype=dtype, device=device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
    yield inp, {"dim": [0, 1]}


@pytest.mark.any_dims
def test_any_dims():
    bench = base.GenericBenchmarkExcluse1D(
        op_name="any_dims",
        torch_op=torch.any,
        input_fn=any_dims_input_fn,
        dtypes=consts.FLOAT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
