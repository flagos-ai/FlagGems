import pytest
import torch

import flag_gems

from . import base


@pytest.mark.polar
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_polar():
    bench = base.BinaryPointwiseBenchmark(
        op_name="polar",
        torch_op=torch.polar,
        dtypes=[torch.float32],
    )
    bench.run()
