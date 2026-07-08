import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.argmax
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_argmax():
    bench = base.UnaryReductionBenchmark(
        op_name="argmax", torch_op=torch.argmax, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
