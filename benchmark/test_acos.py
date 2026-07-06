import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.acos
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_acos():
    bench = base.UnaryPointwiseBenchmark(
        op_name="acos", torch_op=torch.acos, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
