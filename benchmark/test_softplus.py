import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.softplus
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softplus():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus",
        torch_op=torch.nn.functional.softplus,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softplus_backward
def test_softplus_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus_backward",
        torch_op=torch.nn.functional.softplus,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
