import pytest

import flag_gems

from . import base, consts


@pytest.mark.digamma_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_digamma_inplace():
    bench = base.UnaryPointwiseBenchmark(
        op_name="digamma_",
        torch_op=lambda a: a.digamma_(),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
