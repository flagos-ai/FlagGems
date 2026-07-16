import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.reciprocal
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_reciprocal():
    bench = base.UnaryPointwiseBenchmark(
        op_name="reciprocal", torch_op=torch.reciprocal, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.reciprocal_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_reciprocal_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="reciprocal_",
        torch_op=torch.reciprocal_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
