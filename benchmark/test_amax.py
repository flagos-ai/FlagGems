import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.amax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_amax():
    bench = base.UnaryReductionBenchmark(
        op_name="amax", torch_op=torch.amax, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
