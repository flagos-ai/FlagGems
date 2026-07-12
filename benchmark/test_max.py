import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.max
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max():
    bench = base.UnaryReductionBenchmark(
        op_name="max", torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.max_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_max_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="max_dim", torch_op=torch.max, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()
