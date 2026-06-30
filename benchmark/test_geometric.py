import pytest
import torch

import flag_gems

from . import base, consts


def input_fn(shape, cur_dtype, device):
    self = torch.randn(shape, dtype=cur_dtype, device=device)
    p = 0.5
    yield self, p


@pytest.mark.geometric_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_geometric_inplace():
    bench = base.GenericBenchmark(
        op_name="geometric_",
        input_fn=input_fn,
        torch_op=torch.Tensor.geometric_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.geometric
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_geometric():
    bench = base.GenericBenchmark(
        op_name="geometric",
        input_fn=input_fn,
        torch_op=torch.ops.aten.geometric,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
