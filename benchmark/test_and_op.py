import pytest

from . import base, consts


@pytest.mark.and_op
def test_and_op():
    bench = base.BinaryPointwiseBenchmark(
        op_name="and_op",
        torch_op=lambda a, b: a.__and__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
