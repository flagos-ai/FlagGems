import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.all
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_all():
    bench = base.UnaryReductionBenchmark(
        op_name="all", torch_op=torch.all, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.all_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_all_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="all_dim", torch_op=torch.all, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.all_dims
def test_all_dims():
    def all_dims_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        yield inp, {"dim": [0, 1]}

    bench = base.GenericBenchmarkExcluse1D(
        input_fn=all_dims_input_fn,
        op_name="all_dims",
        torch_op=torch.all,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
