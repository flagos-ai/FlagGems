import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.equal
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_equal():
    bench = base.BinaryPointwiseBenchmark(
        op_name="equal",
        torch_op=torch.sub,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
