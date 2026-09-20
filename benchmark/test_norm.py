import pytest
import torch

from . import base, consts, utils


def norm_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    p = 2
    yield inp, p


def norm_scalaropt_dim_input_fn(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    p = 2
    dim = -1
    keepdim = False
    yield inp, p, dim, keepdim


@pytest.mark.norm
def test_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=norm_input_fn,
        op_name="norm",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.norm_scalar
def test_norm_scalar():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
        op_name="norm_scalar",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.norm_scalaropt_dim
def test_norm_scalaropt_dim():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=norm_scalaropt_dim_input_fn,
        op_name="norm_scalaropt_dim",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
