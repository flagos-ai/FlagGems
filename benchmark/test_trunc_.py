import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.trunc_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_trunc():
    bench = base.UnaryPointwiseBenchmark(
        op_name="trunc",
        torch_op=torch.trunc,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.trunc_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_trunc_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="trunc_",
        torch_op=torch.trunc_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
