import pytest

from . import base, consts


@pytest.mark.and_
def test_and_():
    bench = base.BinaryPointwiseBenchmark(
        op_name="and_",
        torch_op=lambda a, b: a.__and__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
