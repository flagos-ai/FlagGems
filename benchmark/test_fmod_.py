import pytest

import flag_gems

from . import base, consts


@pytest.mark.fmod_
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_fmod_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="fmod_",
        torch_op=lambda a, b: a.fmod_(b),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
