import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.var_mean
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_var_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="var_mean", torch_op=torch.var_mean, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def var_mean_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = 1 if len(shape) > 1 else 0
    yield inp, dim


@pytest.mark.var_mean
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_var_mean_dim():
    bench = base.GenericBenchmark(
        op_name="var_mean_dim",
        torch_op=torch.var_mean,
        input_fn=var_mean_dim_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
