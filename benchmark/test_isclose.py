import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.isclose
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_isclose():
    bench = base.BinaryPointwiseBenchmark(
        op_name="isclose",
        torch_op=torch.isclose,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
