import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.special_i1
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_special_i1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_i1", torch_op=torch.special.i1, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.special_i1_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_special_i1_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_i1_out",
        torch_op=torch.special.i1,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
