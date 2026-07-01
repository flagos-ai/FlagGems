import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.min
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_min():
    bench = base.UnaryReductionBenchmark(
        op_name="min", torch_op=torch.min, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.min_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_min_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="min_dim", torch_op=torch.min, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
