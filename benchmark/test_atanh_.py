import pytest

import flag_gems

from . import base, consts


@pytest.mark.atanh_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_atanh_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="atanh_",
        torch_op=lambda a: a.atanh_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
