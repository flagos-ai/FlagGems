import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.std
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_std():
    bench = base.UnaryReductionBenchmark(
        op_name="std", torch_op=torch.std, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
