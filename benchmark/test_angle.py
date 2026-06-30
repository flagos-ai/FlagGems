import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.angle
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_angle():
    bench = base.UnaryPointwiseBenchmark(
        op_name="angle",
        torch_op=torch.angle,
        dtypes=consts.COMPLEX_DTYPES
        + [torch.float32]
        + consts.INT_DTYPES
        + consts.BOOL_DTYPES,
    )
    bench.run()
