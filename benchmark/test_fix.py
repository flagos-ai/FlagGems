import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.fix
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fix():
    bench = base.UnaryPointwiseBenchmark(
        op_name="fix", torch_op=torch.fix, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
