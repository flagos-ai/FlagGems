import pytest

import flag_gems

from . import base, consts


@pytest.mark.cast
def test_cast():
    bench = base.UnaryPointwiseBenchmark(
        op_name="cast",
        torch_op=flag_gems.cast,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.cast_
def test_cast_():
    bench = base.UnaryPointwiseBenchmark(
        op_name="cast_",
        torch_op=flag_gems.cast_,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
