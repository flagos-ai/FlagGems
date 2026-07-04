import pytest

from . import base, consts


@pytest.mark.multiply_
def test_multiply_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="multiply_",
        torch_op=lambda a, b: a.multiply_(b),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
