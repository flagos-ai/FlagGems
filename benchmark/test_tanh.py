import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.tanh
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tanh():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh", torch_op=torch.tanh, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.tanh_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tanh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh_",
        torch_op=torch.tanh_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()


@pytest.mark.tanh_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_tanh_backward():
    bench = base.UnaryPointwiseBenchmark(
        op_name="tanh_backward",
        torch_op=torch.tanh,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
