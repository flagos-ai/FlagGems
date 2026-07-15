import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.std
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_std():
    bench = base.UnaryReductionBenchmark(
        op_name="std", torch_op=torch.std, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def std_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = 1 if len(shape) > 1 else 0
    yield inp, dim


@pytest.mark.std
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_std_dim():
    bench = base.GenericBenchmark(
        op_name="std_dim",
        torch_op=torch.std,
        input_fn=std_dim_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
