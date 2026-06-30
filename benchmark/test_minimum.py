import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.minimum
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_minimum():
    bench = base.BinaryPointwiseBenchmark(
        op_name="minimum",
        torch_op=torch.minimum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
