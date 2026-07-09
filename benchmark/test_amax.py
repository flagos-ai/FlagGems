import pytest
import torch

import flag_gems

from . import base, consts, utils


@pytest.mark.amax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_amax():
    bench = base.UnaryReductionBenchmark(
        op_name="amax", torch_op=torch.amax, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


def amax_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = 1 if len(shape) > 1 else 0
    yield inp, dim


@pytest.mark.amax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_amax_dim():
    bench = base.GenericBenchmark(
        op_name="amax_dim",
        torch_op=torch.amax,
        input_fn=amax_dim_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
