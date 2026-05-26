import pytest
import torch

from . import base, consts, utils


@pytest.mark.norm
def test_norm():
    bench = base.GenericBenchmarkExcluse1D(
        input_fn=utils.unary_input_fn,
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
        input_fn=utils.unary_input_fn,
        op_name="norm_scalaropt_dim",
        torch_op=torch.norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
