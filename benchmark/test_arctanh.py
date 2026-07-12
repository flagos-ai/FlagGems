import pytest

import flag_gems

from . import base, consts


@pytest.mark.arctanh_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_arctanh_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="arctanh_",
        torch_op=lambda a: a.arctanh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
