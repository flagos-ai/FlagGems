import pytest

import flag_gems

from . import base, consts


@pytest.mark.logical_xor_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_logical_xor_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="logical_xor_",
        torch_op=lambda a, b: a.logical_xor_(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
