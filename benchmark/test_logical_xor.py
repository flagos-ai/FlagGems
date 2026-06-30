import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.logical_xor
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logical_xor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_xor",
        torch_op=torch.logical_xor,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
