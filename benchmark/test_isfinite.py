import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.isfinite
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_isfinite():
    bench = base.UnaryPointwiseBenchmark(
        op_name="isfinite", torch_op=torch.isfinite, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
