import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.allclose
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_allclose():
    bench = base.BinaryPointwiseBenchmark(
        op_name="allclose",
        torch_op=torch.allclose,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
