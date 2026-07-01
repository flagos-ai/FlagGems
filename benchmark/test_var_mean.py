import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.var_mean
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_var_mean():
    bench = base.UnaryReductionBenchmark(
        op_name="var_mean", torch_op=torch.var_mean, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
