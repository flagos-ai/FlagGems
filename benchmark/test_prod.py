import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.prod
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_prod():
    bench = base.UnaryReductionBenchmark(
        op_name="prod", torch_op=torch.prod, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
