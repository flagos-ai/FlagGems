import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.nonzero_numpy
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_nonzero_numpy():
    bench = base.GenericBenchmark2DOnly(
        input_fn=base.unary_input_fn,
        op_name="nonzero_numpy",
        torch_op=torch.ops.aten.nonzero_numpy,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
