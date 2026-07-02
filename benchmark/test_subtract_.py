import pytest

from . import base, consts


@pytest.mark.subtract_
def test_subtract_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="subtract_",
        torch_op=lambda a, b: a.subtract_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
