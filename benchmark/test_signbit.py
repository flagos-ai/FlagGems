import pytest
import torch

import flag_gems

from . import base, consts


@pytest.mark.signbit
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_signbit():
    bench = base.UnaryPointwiseBenchmark(
        op_name="signbit", torch_op=torch.signbit, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.skip(reason="No support to non-boolean outputs: issue #2689.")
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.signbit_out
def test_signbit_out():
    bench = base.UnaryPointwiseOutBenchmark(
        op_name="signbit_out",
        torch_op=torch.signbit,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
