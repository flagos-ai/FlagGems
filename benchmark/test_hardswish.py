import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.hardswish_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_hardswish_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="hardswish_",
        torch_op=torch.ops.aten.hardswish_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
