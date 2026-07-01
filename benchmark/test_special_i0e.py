import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.special_i0e
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_special_i0e():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_i0e",
        torch_op=torch.ops.aten.special_i0e,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.special_i0e_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_special_i0e_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="special_i0e_out",
        torch_op=torch.ops.aten.special_i0e,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
