import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.atanh
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_atanh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atanh",
        torch_op=torch.atanh,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
