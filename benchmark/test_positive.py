import pytest

import flag_gems

from . import base, consts


@pytest.mark.positive
def test_positive():
    bench = base.UnaryPointwiseBenchmark(
        op_name="positive",
        torch_op=flag_gems.positive,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
